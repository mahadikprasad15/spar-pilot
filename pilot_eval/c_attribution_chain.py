"""C-attribution runner, launch record and teardown (ticket 09).

preflight -> qualification -> rows -> attribution -> aggregate_freeze -> generation -> report, then on every
exit path: writers joined -> sealed, hash-verified volume copy -> pre-export summary (on the volume and inside
the backup) -> HF export with readback -> final summary -> Pod stop requested with both command forms.

Reused from production-1 (``attribution_chain``): the resume decision on the writer marker, timeouts around
copy/export/stop, the sealed-copy and empty-disk restore helpers, the morning-summary layout and the
automatic sign-off rule. What differs, by the user's decisions:

- **No time caps** (2026-11-18). Elapsed time per stage is logged and persisted across resumes for planning
  only; nothing stops for it. The watchdog runs with its deadline disabled.
- **Typed launch record.** Every leaf is checked against ``SCHEMA`` at seal and at load: unknown or missing
  keys, a bool anywhere, text where a number belongs (PyYAML reads ``1e-4`` as text) and a non-integer
  count are refused. Nothing is chosen at run time.
- **Per-run state.** Volume state lives under ``$VOL/c-attribution/<run-id>/``; a run ID whose run has
  finished is refused, so a new launch needs a new run ID. Resume reuses the same run ID only.
- **What stops the chain.** Input integrity (preflight, bundle, rows) and failed qualification stop it
  (``StopBeforeProduction``); an adapter-restore mismatch raises and stops it. Analysis outcomes (stability,
  unavailable controls, shortfalls) never stop it.
- **Resume.** Local disk may be empty: sealed outputs are restored from the volume, and a completed stage is
  skipped only after every output it declared is re-verified by hash.
"""
import argparse
import fnmatch
import functools
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from pilot_eval import attribution_chain as pc
from pilot_eval import attribution_units as au
from pilot_eval.attribution_chain import (LaunchRefused, ResumeRefused, StopBeforeProduction, call_with_timeout,
                                          resume_decision)

LAUNCH_SCHEMA = 'c-attribution/launch-record-v1'
SUBTREE = 'c-attribution'
STAGES = ['preflight', 'qualification', 'rows', 'attribution', 'aggregate_freeze', 'generation', 'report']
ARMS = ['online-grpo', 'offline-grpo']
CHAIN_DIR = f'{SUBTREE}/chain'
STATE_FILE = f'{CHAIN_DIR}/state.json'
MARKER_FILE = f'{CHAIN_DIR}/writer_marker.json'
PROGRESS_FILE = f'{CHAIN_DIR}/progress.json'
DONE_FILE = f'{CHAIN_DIR}/done'            # the study finished (state complete): the run ID is closed
ATTEMPT_END_FILE = f'{CHAIN_DIR}/last_attempt_end.json'  # an attempt ended (any state); resumable unless done
STOP_WITHHELD_FILE = f'{CHAIN_DIR}/stop_withheld.json'  # no complete verified backup: the Pod was left running
STOP_RECORD_FILE = f'{CHAIN_DIR}/stop_record.json'  # written before the stop request, updated if the runner survives it
LOG_FILE = f'{SUBTREE}/logs/chain.log'
LAUNCH_DIR = f'{SUBTREE}/launch'
SIGNOFF_DIR = f'{SUBTREE}/signoff'
SUMMARY_DIR = f'{SUBTREE}/summaries'
AUTO_REVIEWER = pc.AUTO_REVIEWER
MEASURE_ONLY = 'measure-only'

# --- launch record schema ---------------------------------------------------------------------------
# Leaf types: 'text', 'int' (counts, seeds, sizes), 'number' (int or float, never bool), 'tolerance' (a number,
# or 'measure-only' in a rehearsal), 'ids' ('all' or a list of text). A dict is a fixed set of keys; {'*': spec}
# is a map with free keys; [spec] is a nonempty list.

NUM, INT, TEXT, TOL, IDS, COST = 'number', 'int', 'text', 'tolerance', 'ids', 'cost'
NOT_TRACKED = 'not-tracked'  # user 2026-11-20: the Pod price is not tracked; the summary then has no cost line
SCHEMA = {
    'schema': TEXT, 'run_id': TEXT, 'mode': TEXT, 'label': TEXT, 'code_commit': TEXT,
    'pod_id': TEXT, 'pod_stop_commands': [TEXT], 'volume_identity': TEXT, 'export_repo': TEXT,
    'hourly_cost_usd': COST, 'arms': [TEXT],
    'timeouts_seconds': {'copy': NUM, 'export_attempt': NUM, 'stop_attempt': NUM, 'writer_join': NUM},
    'marker_stale_seconds': NUM,
    'disk': {'min_free_gb': NUM, 'container_total_gb': NUM},
    'tolerances': {
        'finite_difference': {'epsilon': NUM, 'median_rel_err_max': NUM, 'max_rel_err_max': NUM, 'precision': TEXT},
        'neuron_sum_identity': NUM, 'batch_invariance': NUM, 'memory_fraction': NUM,
        'cached_vs_forced': {'per_token_abs_max': TOL, 'per_row_abs_M_max': TOL}},
    'qualification': {'rows_per_arm': INT, 'fd_rows': INT, 'fd_adapters': INT, 'fd_neurons': INT,
                      'head_chunk': INT, 'noop_greedy_tokens': INT, 'batch_invariance_budgets': [INT]},
    'scope': {'attribution_questions': {'*': IDS}, 'generation_questions': {'*': IDS},
              'generation_conditions': IDS},
    'attribution': {'token_budget': INT, 'questions_per_unit': INT},
    'generation': {'batch_size': INT, 'cap': INT, 'eos_token_id': INT},
    'selection': {'k_adapters': INT, 'k_reads': INT, 'k_harmful': INT, 'tolerance': NUM, 'seeds': {'*': INT}},
    'statistics': {'split_seed': INT, 'bootstrap_seed': INT, 'resamples': INT},
    'hot_layers': {'layers': [INT], 'source': {'role': TEXT, 'path': TEXT, 'sha256': TEXT}, 'context': TEXT,
                   'step': INT, 'k': INT, 'module': TEXT, 'arms': [TEXT], 'combine': TEXT},
    'decisions': {'*': TEXT},
    'inputs': {'artifact_root': TEXT, 'bundle': TEXT, 'model_dir': TEXT, 'work_dir': TEXT,
               'sources': {'*': {'responses_role': TEXT, 'config_role': TEXT, 'adapter_role': TEXT,
                                 'catalog_id': TEXT}}},
}
USER_FILLED = ('code_commit', 'pod_id', 'inputs.artifact_root', 'inputs.bundle', 'inputs.model_dir', 'inputs.work_dir')


def _leaf_problem(value, kind, path, mode):
    if isinstance(value, bool):
        return f'{path} ({value!r} is bool; expected {kind})'
    if kind == TEXT:
        ok = isinstance(value, str) and value != ''
    elif kind == INT:
        ok = isinstance(value, int)
    elif kind == NUM:
        ok = isinstance(value, (int, float))
    elif kind == TOL:
        ok = isinstance(value, (int, float)) or (value == MEASURE_ONLY and mode == 'rehearsal')
        if value == MEASURE_ONLY and mode != 'rehearsal':
            return f"{path} ('{MEASURE_ONLY}' is allowed only in a rehearsal; production needs a number)"
    elif kind == COST:
        ok = isinstance(value, (int, float)) or value == NOT_TRACKED
    elif kind == IDS:
        ok = value == 'all' or (isinstance(value, list) and value and all(isinstance(v, str) and v for v in value))
    else:
        raise ValueError(kind)
    return None if ok else f'{path} ({value!r} is {type(value).__name__}; expected {kind})'


def typecheck(value, spec=SCHEMA, path='', *, mode=None):
    """Every problem with ``value`` against ``spec``: missing, unknown, null, '[proposed]' and wrong-typed leaves."""
    if mode is None and isinstance(value, dict):
        mode = value.get('mode')
    if value is None:
        return [f'{path or "<root>"} (unresolved: null)']
    if isinstance(value, str) and 'proposed' in value.lower():
        return [f'{path} (unresolved: {value!r})']
    if isinstance(spec, str):
        problem = _leaf_problem(value, spec, path, mode)
        return [problem] if problem else []
    if isinstance(spec, list):
        if not isinstance(value, list) or not value:
            return [f'{path} (expected a nonempty list)']
        return [p for i, v in enumerate(value) for p in typecheck(v, spec[0], f'{path}[{i}]', mode=mode)]
    if not isinstance(value, dict):
        return [f'{path} (expected a mapping, got {type(value).__name__})']
    problems = []
    if set(spec) == {'*'}:
        if not value:
            return [f'{path} (expected a nonempty mapping)']
        for key, v in value.items():
            problems += typecheck(v, spec['*'], f'{path}.{key}' if path else str(key), mode=mode)
        return problems
    for key in spec:
        sub = f'{path}.{key}' if path else key
        if key not in value:
            problems.append(f'{sub} (missing)')
        else:
            problems += typecheck(value[key], spec[key], sub, mode=mode)
    problems += [f'{path + "." if path else ""}{k} (unknown key)' for k in sorted(set(value) - set(spec))]
    return problems


def record_sha256(record):
    return hashlib.sha256(pc._canonical(record).encode()).hexdigest()


# Where a run executes, not what it computes: a replacement Pod reseals only these (``rebind``).
BINDING_FIELDS = ('pod_id', 'inputs.artifact_root', 'inputs.bundle', 'inputs.model_dir', 'inputs.work_dir')


def run_sha256(record):
    """The run's identity: the launch record without its Pod binding. Chain state and every sealed artifact bind
    to this, so a rebound record on a replacement Pod continues the same run; any other change does not."""
    body = json.loads(json.dumps(record))
    for dotted in BINDING_FIELDS:
        parent, keys = body, dotted.split('.')
        for k in keys[:-1]:
            parent = parent.get(k, {})
        parent.pop(keys[-1], None)
    return hashlib.sha256(pc._canonical(body).encode()).hexdigest()


def launch_dir(record):
    """Each sealed record of the run (the original and any rebinds) under its own hash."""
    return f'{LAUNCH_DIR}/{record_sha256(record)[:16]}'


def stop_commands(record):
    commands = pc.stop_commands(record)
    texts = [' '.join(c) for c in commands]
    pod = record['pod_id']
    if texts != [f'runpodctl stop pod {pod}', f'runpodctl pod stop {pod}']:
        raise LaunchRefused("pod_stop_commands must be ['runpodctl stop pod {id}', 'runpodctl pod stop {id}'] "
                            '(both syntaxes, in that order; lesson pod-v1)')
    return commands


def validate(record, *, code_commit):
    problems = typecheck(record)
    if problems:
        raise LaunchRefused(f'launch record problems: {problems[:10]}')
    if record['schema'] != LAUNCH_SCHEMA:
        raise LaunchRefused('not a C-attribution launch record')
    if record['code_commit'] != code_commit:
        raise LaunchRefused(f"code commit {code_commit} differs from the record's {record['code_commit']}")
    if record['mode'] not in ('rehearsal', 'production'):
        raise LaunchRefused('mode must be production or rehearsal')
    if record['mode'] == 'rehearsal' and not record['label'].startswith('rehearsal'):
        raise LaunchRefused('a rehearsal needs a label starting with "rehearsal"')
    if record['mode'] == 'production':
        if (record['generation']['cap'], record['generation']['eos_token_id']) != (1024, 151645):
            raise LaunchRefused('production uses the runs\' cap 1024 and EOS 151645 (spec: Settings)')
        if record['label'] != 'production':
            raise LaunchRefused('a production record is labelled "production"')
        scope = record['scope']
        if any(v != 'all' for part in ('attribution_questions', 'generation_questions')
               for v in scope[part].values()) or scope['generation_conditions'] != 'all':
            raise LaunchRefused('production covers every question and condition (scope: all)')
    if sorted(record['arms']) != sorted(ARMS):
        raise LaunchRefused(f'arms must be {ARMS}')
    for part in ('attribution_questions', 'generation_questions'):
        if sorted(record['scope'][part]) != sorted(record['arms']):
            raise LaunchRefused(f'scope.{part} needs one entry per arm')
    if sorted(record['inputs']['sources']) != sorted(record['arms']):
        raise LaunchRefused('inputs.sources needs one entry per arm')
    if record['tolerances']['finite_difference']['precision'] != 'fp64':
        raise LaunchRefused('finite differences use the FP64 reference (spec: mandatory for seqlogp)')
    from pilot_eval.c_attribution_selection import GROUPS
    if sorted(record['selection']['seeds']) != sorted(GROUPS):
        raise LaunchRefused(f'selection.seeds needs one seed per group {list(GROUPS)}')
    if not str(record['run_id']).replace('-', '').isalnum():
        raise LaunchRefused(f"run_id {record['run_id']!r} must be a simple name")
    stop_commands(record)
    return record


def rebind(record_path, out_path, *, code_commit, pod_id, fill=None):
    """On a replacement Pod: the old record resealed for this Pod (and, optionally, new local paths). Nothing
    else may change, so the run continues under the same run ID once preflight matches the qualified setup."""
    if not pod_id:
        raise LaunchRefused('RUNPOD_POD_ID is not set: run rebind on the replacement Pod')
    record = load_launch_record(record_path, code_commit=code_commit)
    record = json.loads(json.dumps(record))
    record['pod_id'] = pod_id
    for dotted, value in (fill or {}).items():
        if dotted not in BINDING_FIELDS or dotted == 'pod_id':
            raise LaunchRefused(f'{dotted} is not a binding field; only the Pod and local paths change on rebind')
        if value is not None:
            record['inputs'][dotted.split('.')[1]] = str(value)
    validate(record, code_commit=code_commit)
    path = seal_launch_record(record, out_path)
    load_launch_record(path, code_commit=code_commit)
    return path


def seal_launch_record(record, path):
    return pc.seal_launch_record(record, path)


def load_launch_record(path, *, code_commit, validator=None):
    """A sealed record, refused unless it matches its seal and passes ``validator`` (default: this module's
    ``validate``; other runs that reuse the sealed, typed record pass their own, e.g. ``sparse_rewiring_plan``)."""
    path = Path(path)
    record = json.loads(path.read_text())
    seal = path.with_suffix('.seal.json')
    if not seal.is_file() or json.loads(seal.read_text())['sha256'] != record_sha256(record):
        raise LaunchRefused('launch record does not match its seal')
    return (validator or validate)(record, code_commit=code_commit)


def finalize_draft(draft_path, out_path, *, code_commit, pod_id=None, fill=None, validator=None,
                   fillable=USER_FILLED):
    """Approved YAML draft -> sealed, typed JSON record. Fills only empty (null) fields: the code commit, the
    Pod ID (from the Pod's environment) and the input paths in ``fill`` ({'inputs.work_dir': ...}).
    ``validator`` and ``fillable`` default to this module's; another run's record passes its own."""
    validator = validator or validate
    import yaml
    record = yaml.safe_load(Path(draft_path).read_text())
    record['code_commit'] = code_commit
    if record.get('pod_id') is None and pod_id:
        record['pod_id'] = pod_id
    for dotted, value in (fill or {}).items():
        if dotted not in fillable:
            raise LaunchRefused(f'{dotted} is not a fillable field')
        parent, key = record, dotted.split('.')
        for part in key[:-1]:
            parent = parent.setdefault(part, {})
        if parent.get(key[-1]) is None and value is not None:
            parent[key[-1]] = str(value)
    validator(record, code_commit=code_commit)
    path = seal_launch_record(record, out_path)
    load_launch_record(path, code_commit=code_commit, validator=validator)
    return path


# --- per-run volume state ---------------------------------------------------------------------------

def run_root(volume, run_id):
    """``$VOL/c-attribution/<run-id>``: all of one launch's volume state."""
    return Path(volume) / SUBTREE / run_id


def check_pod(record, current_pod):
    """The stop target is the record's Pod: refuse a launch on any other Pod (review 06-10, P1)."""
    if not current_pod:
        raise ResumeRefused('RUNPOD_POD_ID is not set: cannot confirm this is the Pod the record will stop '
                            f"({record['pod_id']})")
    if current_pod != record['pod_id']:
        raise ResumeRefused(f"this Pod is {current_pod} but the launch record stops Pod {record['pod_id']}; "
                            'seal a record on this Pod')


def precheck(record, volume, *, now=time.time, host=None, current_pod=None):
    """Before the launcher takes watchdog ownership: refuse a record for another Pod, a finished run ID, and a
    live, foreign or ambiguous writer on this run's marker (a refused second launch stops nothing). A run whose
    last attempt ended (failed or stopped) resumes under the same run ID."""
    check_pod(record, os.environ.get('RUNPOD_POD_ID', '') if current_pod is None else current_pod)
    root = run_root(volume, record['run_id'])
    if (root / DONE_FILE).is_file():
        raise ResumeRefused(f"run {record['run_id']} already finished ({(root / DONE_FILE).read_text().strip()}); "
                            'a new launch needs a new run ID')
    return resume_decision(root / MARKER_FILE, record['run_id'], stale_seconds=record['marker_stale_seconds'],
                           host=host, now=now)


# --- the chain --------------------------------------------------------------------------------------

class Context:
    def __init__(self, chain, stage):
        self.chain, self.stage, self.record = chain, stage, chain.record
        self.local_root, self.volume_root = chain.local_root, chain.volume_root
        self.units_done = 0

    @property
    def attempt(self):
        return self.chain.attempt

    def boundary(self, **progress):
        """Unit boundary: mirror newly sealed work to the volume, refresh the writer heartbeat and the visible
        progress record (with the mirror's result). Never stops for time."""
        self.units_done += 1
        mirror = self.chain.mirror()
        self.chain._beat('running', stage=self.stage, units_done=self.units_done)
        self.chain.progress(self.stage, units_done=self.units_done, mirror=mirror, **progress)

    def expect(self, *runs):
        self.chain.expected_runs |= {str(r) for r in runs}

    def log(self, message):
        self.chain.log(f'[{self.stage}] {message}')


class Chain:
    def __init__(self, record, *, local_root, volume, stages, exporter=None, runner=subprocess.run,
                 volume_identity=None, host=None, copier=None, current_pod=None):
        self.record, self.stages = record, stages
        self.record_sha256 = record_sha256(record)
        self.run_sha256 = run_sha256(record)
        self.local_root = Path(local_root)
        self.volume_root = run_root(volume, record['run_id'])
        self.timeouts = record['timeouts_seconds']
        # the real exporter's Hub calls run in child processes killed at export_attempt (OnlineHubProcess)
        self.exporter = exporter or functools.partial(_default_exporter, hub_timeout=self.timeouts['export_attempt'])
        self.runner = runner
        # The teardown copy runs in a child process that is ended on timeout (``in_child``), so it must be a
        # module-level function.
        self.copier = copier or copy_sealed
        self.current_pod = os.environ.get('RUNPOD_POD_ID', '') if current_pod is None else current_pod
        self.pod_confirmed = False
        self.volume_identity, self.host = volume_identity, host or socket.gethostname()
        self.timeouts = record['timeouts_seconds']
        self.expected_runs = {launch_dir(record)}
        self.attempt = None
        self.identity = None
        self._threads = []

    def log(self, message):
        path = self.local_root / LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}\n')

    def progress(self, stage, **detail):
        pc._write_json(self.volume_root / PROGRESS_FILE, dict(run_id=self.record['run_id'], attempt=self.attempt,
                                                              stage=stage, last_update_unix=time.time(), **detail))

    def _beat(self, state, **extra):
        pc._write_json(self.volume_root / MARKER_FILE,
                       dict(self._marker, state=state, heartbeat_unix=time.time(), **extra))

    def _save(self, state):
        pc._write_json(self.volume_root / STATE_FILE, state)

    def mirror(self):
        """Copy newly sealed work to the volume now (incremental; already-copied runs are skipped)."""
        try:
            result = self.copier(self.local_root, self.volume_root, sealed_only=True, skip_present=True)
        except Exception as exc:  # recorded; the teardown copy and the export remain
            result = dict(verified=False, copied=[], problems=[f'{type(exc).__name__}: {exc}'])
        if result['problems']:
            self.log(f"mirror problems: {result['problems']}")
        return dict(verified=result['verified'], copied=len(result['copied']), problems=result['problems'][:5],
                    unix=time.time())

    def _start(self):
        check_pod(self.record, self.current_pod)  # first: it decides whether teardown may stop the record's Pod
        self.pod_confirmed = True
        if self.volume_identity != self.record['volume_identity']:
            raise ResumeRefused(f'volume identity {self.volume_identity!r} differs from the launch record')
        decision = precheck(self.record, self.volume_root.parent.parent, host=self.host, current_pod=self.current_pod)
        state_path = self.volume_root / STATE_FILE
        state = json.loads(state_path.read_text()) if state_path.is_file() else dict(
            run_id=self.record['run_id'], run_sha256=self.run_sha256, completed={}, elapsed={}, bindings=[])
        if state['run_id'] != self.record['run_id']:
            raise ResumeRefused('chain state belongs to another run ID')
        if state.get('run_sha256') != self.run_sha256:
            raise ResumeRefused('chain state was written under a different launch record (beyond its Pod binding); '
                                'a new launch needs a new run ID')
        binding = dict(record_sha256=self.record_sha256, pod_id=self.record['pod_id'])
        if binding not in [{k: b[k] for k in binding} for b in state.setdefault('bindings', [])]:
            state['bindings'].append(dict(binding, first_attempt=state.get('attempts', 0) + 1))
        self.restored = pc.restore_from_volume(self.volume_root, self.local_root, subtree=SUBTREE)
        state['attempts'] = state.get('attempts', 0) + 1
        self.attempt = f"attempt-{state['attempts']:03d}"
        self._marker = dict(run_id=self.record['run_id'], host=self.host, pid=os.getpid(), started_unix=time.time())
        self._beat('running')
        self._save(state)
        au.seal_run(self.local_root / launch_dir(self.record), {'launch_record.json': self.record})
        return state, decision

    # --- resume: a completed stage is skipped only if its outputs re-verify -------------------------

    def _outputs_verified(self, result):
        problems = []
        for relative in (result or {}).get('outputs', []):
            try:
                au.verify_run(self.local_root / relative)
            except ValueError as exc:
                problems.append(str(exc))
        return problems

    def _identity_problem(self, state):
        """Qualification and every later unit are valid only for the runtime and inputs they were qualified
        under; a resume on anything else is refused (a new run ID requalifies)."""
        if 'qualification' not in state['completed']:
            return None
        qualified = state.get('qualified_identity')
        if qualified != self.identity:
            fields = sorted(_differing(qualified, self.identity))
            details = '; '.join(f'{f}: qualified {_lookup(qualified, f)!r}, now {_lookup(self.identity, f)!r} -> '
                                f'{_remedy(f, _lookup(qualified, f))}' for f in fields)
            return (f'this attempt runs under a runtime/inputs other than the run was qualified under '
                    f'(fields {fields}). {details}. Fix that and relaunch, or use a new run ID to requalify')
        return None

    # --- sign-off (production-1's automatic rule) ---------------------------------------------------

    def _signoff(self, result):
        reasons = [f'check failed: {k}' for k, v in sorted(result.get('checks', {}).items()) if v is not True]
        if not result.get('checks'):
            reasons.append('no qualification checks recorded')
        reasons += [f'review flag: {f}' for f in result.get('review_flags', [])]
        missing = sorted(set(self.record['arms']) - set(result.get('arms', [])))
        if missing:
            reasons.append(f'missing arms: {missing}')
        if reasons:
            raise StopBeforeProduction('; '.join(reasons))
        record = dict(by=AUTO_REVIEWER, decision='accept', checks=result['checks'],
                      rule="every qualification check passes against the launch record's fixed tolerances and no "
                           'review flag is raised', qualification_sha256=record_sha256(result), signed_unix=time.time())
        au.seal_run(self.local_root / SIGNOFF_DIR, {'signoff.json': record})
        self.expected_runs.add(SIGNOFF_DIR)
        return record

    def _check_signoff(self, state):
        later = [s for s in STAGES[STAGES.index('qualification'):] if s in state['completed']]
        if not later:
            return
        signoff, qual = state.get('signoff') or {}, state['completed'].get('qualification')
        if qual is None or signoff.get('decision') != 'accept' or signoff.get('by') != AUTO_REVIEWER or \
                signoff.get('qualification_sha256') != record_sha256(qual):
            raise RuntimeError('completed stages without a matching automatic sign-off of the completed '
                               'qualification; investigate')

    # --- run --------------------------------------------------------------------------------------

    def run(self):
        started = time.time()
        summary = dict(run_id=self.record['run_id'], mode=self.record['mode'], label=self.record['label'],
                       stages={}, failures=[], state='running', launch_record_sha256=self.record_sha256,
                       run_sha256=self.run_sha256, pod_id=self.record['pod_id'])
        state = None
        try:
            state, decision = self._start()
            summary.update(attempt=self.attempt, resume=decision, restored_from_volume=self.restored,
                           signoff=state.get('signoff'))
            self._check_signoff(state)
            for name in STAGES:
                previous = state['completed'].get(name)
                if name == 'preflight':
                    previous = None  # today's inputs and runtime are always checked again (review 06-10, P1)
                if previous is not None:
                    problems = self._outputs_verified(previous)
                    if not problems:
                        summary['stages'][name] = dict(status='complete', resumed=True)
                        self.expected_runs |= set(previous.get('outputs', []))
                        continue
                    self.log(f'stage {name} was complete but its outputs do not verify; re-running: {problems}')
                    del state['completed'][name]
                self._beat('running', stage=name)
                self.progress(name, units_done=0)
                self.log(f'stage {name} start')
                t0 = time.monotonic()
                try:
                    result = self.stages[name](Context(self, name)) or {}
                except StopBeforeProduction as exc:
                    summary['stages'][name] = dict(status='stopped', reason=str(exc))
                    summary['state'], summary['stop_reason'] = 'stopped', f'{name}: {exc}'
                    break
                finally:
                    elapsed = time.monotonic() - t0
                    state['elapsed'][name] = state['elapsed'].get(name, 0.0) + elapsed
                    self.log(f'stage {name} elapsed {elapsed:.1f}s (total {state["elapsed"][name]:.1f}s; '
                             'logged for planning only, no caps)')
                summary['stages'][name] = dict(status='complete', **{k: v for k, v in result.items()
                                                                     if k in ('analysis', 'counts', 'notes', 'host')})
                if name == 'preflight':
                    self.identity = result.get('identity')
                    problem = self._identity_problem(state)
                    if problem:
                        summary['stages'][name] = dict(status='stopped', reason=problem)
                        summary['state'], summary['stop_reason'] = 'stopped', f'preflight: {problem}'
                        break
                if name == 'qualification':
                    try:
                        signoff = self._signoff(result)
                    except StopBeforeProduction as exc:
                        summary['stages'][name] = dict(status='stopped', reason=str(exc))
                        summary['state'], summary['stop_reason'] = 'stopped', f'qualification: {exc}'
                        break
                    state['signoff'] = summary['signoff'] = signoff
                    state['qualified_identity'] = self.identity
                self.expected_runs |= set(result.get('outputs', []))
                state['completed'][name] = result
                summary['stages'][name]['mirror'] = self.mirror()
                if name == 'report':
                    summary['report'] = result.get('headline')
                self._save(state)
            else:
                summary['state'] = 'complete'
        except BaseException as exc:  # every exit path tears down
            self.log('failure:\n' + traceback.format_exc())
            summary['state'] = 'failed'
            summary['failures'].append(dict(stage=next((s for s in STAGES if s not in summary['stages']), None),
                                            error=f'{type(exc).__name__}: {exc}', log=LOG_FILE))
        finally:
            if state is not None:
                self._save(state)
                summary['elapsed_seconds'] = dict(state['elapsed'])
            summary['wall_seconds'] = time.time() - started
            cost = self.record['hourly_cost_usd']
            summary['cost_estimate_usd'] = None if cost == NOT_TRACKED else summary['wall_seconds'] / 3600 * cost
            if getattr(self, '_marker', None) is None:
                self._marker = dict(run_id=self.record['run_id'], host=self.host, pid=os.getpid(),
                                    started_unix=started)
            self.attempt = self.attempt or 'attempt-000'
            self.teardown(summary)
        return summary

    # --- teardown ---------------------------------------------------------------------------------

    def _export(self, *, include, expected_runs, attempts):
        tries, outcome = [], dict(verified=False)
        for _ in range(attempts):
            kind, value = self._bounded_thread(self.exporter, self.timeouts['export_attempt'], self.local_root,
                                               label=self.record['run_id'], include=include, expected_units=[],
                                               expected_runs=expected_runs, repo=self.record['export_repo'])
            if kind == 'ok':
                tries.append(dict(outcome='ok'))
                outcome = dict(verified=bool(value.get('readback_verified', True)), receipt=value,
                               completeness=value.get('completeness'))
                break
            tries.append(dict(outcome=kind, error=None if value is None else f'{type(value).__name__}: {value}'))
        outcome['attempts'] = tries
        return outcome

    def _write_summary(self, summary, phase, missing):
        relative = f'{SUMMARY_DIR}/{self.attempt}/{phase}'
        au.seal_run(self.local_root / relative, {'run_summary.json': summary, 'run_summary.md': _markdown(summary),
                                                 'missing_work.json': list(missing)})
        kind, value, ended = in_child(copy_run, self.timeouts['copy'], str(self.local_root / relative),
                                      str(self.volume_root / relative))
        return relative, kind == 'ok' and bool(value) and ended

    def _volume_copy(self):
        kind, value, ended = in_child(self.copier, self.timeouts['copy'], str(self.local_root),
                                      str(self.volume_root), sealed_only=True)
        if kind == 'timeout':
            copy = dict(verified=False, outcome='timeout', timeout_seconds=self.timeouts['copy'])
        elif kind == 'error':
            copy = dict(verified=False, outcome='error', error=value)
        else:
            copy = dict(value, outcome='ok', verified=bool(value.get('verified')) and not value.get('problems'))
        copy['ended'] = ended  # the copy process has exited before anything later runs
        copied = {c.split(':', 1)[1] for c in copy.get('copied', []) if c.startswith('run:')}
        # Everything a Pod stop would lose: the expected outputs and every run sealed on local disk (a sealed unit
        # of an unfinished stage is not an expected output yet). What is on the volume counts, whether this copy
        # or an earlier mirror put it there.
        self.local_work = self.expected_runs | _local_sealed(self.local_root)
        present = {r for r in self.local_work if r in copied or _verifies(self.volume_root / r)}
        copy['missing'] = sorted(self.local_work - present)
        copy['coverage'] = 'complete' if copy['verified'] and not copy['missing'] else 'partial'
        return copy

    def _statement(self, summary, backups):
        """Verified bytes and complete coverage are reported separately (review 06-10, P1). ``safe``: verified
        copies of all local work (the Pod may be stopped); ``complete``: safe and the study finished."""
        not_completed = [s for s in STAGES if summary['stages'].get(s, {}).get('status') != 'complete']
        run_complete = summary['state'] == 'complete'
        safe = [name for name, b in backups if b['verified'] is True and b.get('coverage') == 'complete']
        complete = safe if run_complete else []
        parts = []
        for name, b in backups:
            if b['verified'] is not True:
                parts.append(f'{name}: NOT verified')
            elif name in complete:
                parts.append(f'{name}: verified, complete')
            else:
                why = [] if run_complete else ['the run did not complete']
                if b.get('missing'):
                    why.append(f"missing {b['missing'][:5]}")
                parts.append(f"{name}: verified bytes, partial ({'; '.join(why) or 'incomplete coverage'})")
        if complete:
            head = 'complete verified backups: ' + ', '.join(complete)
        elif safe:
            head = 'NO COMPLETE VERIFIED BACKUP of a finished study; verified backups of all work so far: ' + ', '.join(safe)
        else:
            head = ('NO COMPLETE VERIFIED BACKUP. STOP WITHHELD: the Pod is left running so its local work can still be '
                    f'backed up ({STOP_WITHHELD_FILE})')
        if not run_complete:
            head += f'; run did not complete (stages never finished: {not_completed})'
        return dict(statement=head + '. ' + '; '.join(parts), complete=complete, safe=safe,
                    stages_not_completed=not_completed)

    def teardown(self, summary):
        steps = summary['teardown'] = []
        copy = self._volume_copy()
        steps.append('volume_copy')
        missing = copy['missing']
        summary['backups'] = dict(volume_copy=copy, hf_export=dict(verified='pending'), complete=[],
                                  statement='HF export pending; this pre-export summary is the one inside the backup')
        pre, pre_on_volume = self._write_summary(summary, 'pre-export', missing)
        steps.append('summary_sealed')
        t0 = time.monotonic()
        export = self._export(include=[SUBTREE], expected_runs=sorted(self.local_work | {pre}), attempts=3)
        export['seconds'] = time.monotonic() - t0
        export['missing'] = list((export.get('receipt') or {}).get('missing_expected_runs') or [])
        export['coverage'] = 'complete' if export['verified'] is True and export.get('completeness') == 'complete' \
            and not export['missing'] else 'partial'
        steps.append('hf_export')
        summary['backups'] = dict(volume_copy=copy, hf_export=export,
                                  pre_export_summary=dict(path=pre, on_volume=pre_on_volume),
                                  **self._statement(summary, [('volume copy', copy), ('HF export', export)]))
        final, summary['summary_on_volume'] = self._write_summary(summary, 'final', missing)
        steps.append('summary_on_volume')
        upload = self._export(include=[final], expected_runs=[final], attempts=1)
        summary['final_summary_upload'] = dict(outcome='ok' if upload['verified'] else 'failed',
                                               attempts=upload['attempts'], path=final)
        summary['teardown_io_running'] = [t.name for t in self._threads if t.is_alive()]
        safe = bool(summary['backups']['safe'])
        if safe:
            # A real stop ends this container inside the stop call (rehearsal, 2026-11-22): every closing record
            # (stop record, attempt end, released marker, done) is written first, then the stop is requested.
            (self.volume_root / STOP_WITHHELD_FILE).unlink(missing_ok=True)
            commands = [' '.join(c) for c in stop_commands(self.record)] if self.pod_confirmed else []
            self._close(summary, steps + ['stop_requesting'], safe,
                        stop=dict(requested='in progress', commands=commands, externally_verified=False),
                        outcome='requesting')
            summary['stop'] = self._stop()
        else:  # stopping would lose the only copy of local work (review 06-10 follow-up, P1)
            summary['stop'] = self._withhold(summary)
        steps.append('stop_requested' if summary['stop']['requested'] else
                     'stop_withheld' if summary['stop'].get('withheld') else 'stop_not_requested')
        self._close(summary, steps, safe, stop=summary['stop'],
                    outcome='requested' if summary['stop']['requested'] else
                    'withheld' if summary['stop'].get('withheld') else 'not-requested')
        return summary

    def _close(self, summary, steps, safe, *, stop, outcome):
        pc._write_json(self.volume_root / STOP_RECORD_FILE, dict(stop=stop, steps=steps, attempt=self.attempt))
        # The watchdog reads this: an attempt the runner tore down itself is not a crash (it never stops it again).
        pc._write_json(self.volume_root / ATTEMPT_END_FILE, dict(
            attempt=self.attempt, state=summary['state'], writer_pid=os.getpid(), unix=time.time(), stop=outcome))
        self._beat('released', final_state=summary['state'])
        if summary['state'] == 'complete' and safe:  # a finished, backed-up study closes the run ID
            (self.volume_root / DONE_FILE).write_text(summary['state'] + '\n')

    def _withhold(self, summary):
        record = dict(attempt=self.attempt, unix=time.time(), state=summary['state'],
                      reason='no complete verified backup: neither the volume copy nor the HF export holds all local '
                             'work, and a Pod stop would delete the local disk',
                      missing_on_volume=summary['backups']['volume_copy'].get('missing', [])[:50],
                      hf_export=summary['backups']['hf_export'].get('attempts'),
                      next='fix the volume or Hub problem, then relaunch on this Pod with the same record '
                           '(bash scripts/c_attribution_launch.sh RECORD LOCAL VOLUME): completed stages are '
                           're-verified and skipped, the backups are retried, and the Pod is stopped once one is '
                           f"complete. Or stop the Pod by hand: {' '.join(stop_commands(self.record)[0])}")
        try:
            pc._write_json(self.volume_root / STOP_WITHHELD_FILE, record)
        except OSError as exc:
            record['volume_write_failed'] = str(exc)
        pc._write_json(self.local_root / STOP_WITHHELD_FILE, record)
        print(f"STOP WITHHELD: {record['reason']}. Next: {record['next']}", file=sys.stderr)
        return dict(requested=False, withheld=True, reason=record['reason'], attempts=[])

    def _bounded_thread(self, fn, timeout, *args, **kwargs):
        """The exporter is injectable (an in-memory fake Hub in tests), so it runs in a thread. The real exporter
        runs every Hub call in a child process killed at ``export_attempt``; after a timeout the thread is joined
        once more for that long and, if still alive, reported in ``teardown_io_running``."""
        import threading
        box = {}

        def target():
            try:
                box['value'] = fn(*args, **kwargs)
            except BaseException as exc:
                box['error'] = exc
        thread = threading.Thread(target=target, daemon=True, name=f'export-{len(self._threads)}')
        self._threads.append(thread)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            thread.join(timeout)
            return 'timeout', None
        if 'error' in box:
            return 'error', box['error']
        return 'ok', box.get('value')

    def _stop(self):
        if not self.pod_confirmed:  # the record's Pod may not be this one: never stop it (review 06-10, P1)
            return dict(requested=False, reason="the launch record's Pod was not confirmed as this Pod; nothing "
                                                'was stopped. Stop this Pod by hand.', attempts=[])
        attempts, requested = [], False
        for command in stop_commands(self.record):
            try:
                result = self.runner(command, capture_output=True, text=True, timeout=self.timeouts['stop_attempt'])
                attempts.append(dict(command=command, outcome='returned', returncode=result.returncode,
                                     output=((result.stdout or '') + (result.stderr or ''))[-2000:]))
                if result.returncode == 0:
                    requested = True
                    break
            except subprocess.TimeoutExpired:
                attempts.append(dict(command=command, outcome='timeout'))
            except OSError as exc:
                attempts.append(dict(command=command, outcome='error', error=str(exc)))
        return dict(requested=requested, externally_verified=False, attempts=attempts)


def copy_sealed(local_root, volume_root, sealed_only=True, *, skip_present=False):
    """Sealed, hash-verified copy of this run's subtree to its volume root (module level: it runs in a child)."""
    return pc.copy_sealed(local_root, volume_root, sealed_only=sealed_only, subtree=SUBTREE,
                          skip_present=skip_present)


def copy_run(source, target):
    source, target = Path(source), Path(target)
    files = au.verify_run(source)
    for name in sorted(files) + ['complete.json']:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes((source / name).read_bytes())
    au.verify_run(target)
    return True


def _child(conn, fn, args, kwargs):
    try:
        conn.send(('ok', fn(*args, **kwargs)))
    except BaseException as exc:
        conn.send(('error', f'{type(exc).__name__}: {exc}'))


def in_child(fn, timeout, *args, **kwargs):
    """Run ``fn`` in a spawned child; on timeout the child is terminated, then killed, and always reaped, so the
    work has ended when this returns. ('ok'|'error'|'timeout', value, ended)."""
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')  # never fork a CUDA process
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(child, fn, args, kwargs), daemon=True)
    process.start()
    child.close()
    result = None
    if parent.poll(timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = ('error', 'child exited without a result')
    process.join(5 if result else 0)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join()
    if result is None:
        result = ('timeout', None)
    return result[0], result[1], not process.is_alive()


def _local_sealed(local_root):
    base = Path(local_root) / SUBTREE
    return {str(p.parent.relative_to(local_root)) for p in base.rglob('complete.json')} if base.is_dir() else set()


def _verifies(directory):
    try:
        au.verify_run(directory)
        return True
    except (ValueError, OSError):
        return False


PIP_FIELDS = ('torch', 'transformers', 'peft')


def _lookup(value, dotted):
    for key in dotted.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _remedy(field, qualified):
    """What makes a resume possible again: reinstall the qualified package, or rent the same GPU type."""
    parts = field.split('.')
    if parts[:2] == ['runtime', 'cuda_libraries'] and len(parts) == 3:
        return f'pip install {parts[2]}=={qualified}' if qualified else f'pip uninstall {parts[2]}'
    if parts[0] == 'runtime' and len(parts) == 2 and parts[1] in PIP_FIELDS:
        return f'pip install {parts[1]}=={qualified}' + (' (from the cu128 index)' if parts[1] == 'torch' else '')
    if field == 'runtime.device_name':
        return f'rent the same GPU type ({qualified})'
    if parts[0] == 'runtime' and parts[1] in ('cuda_runtime', 'cudnn'):
        return 'reinstall the qualified torch build, which brings this version'
    return 'not fixable on this Pod; a new run ID requalifies'


def _differing(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            out |= _differing(a.get(k), b.get(k), f'{prefix}{k}.')
        return out
    return set() if a == b else {prefix.rstrip('.') or '<all>'}


def _markdown(summary):
    lines = [f"# Run summary — {summary['run_id']} ({summary['mode']}, {summary.get('label')})", '',
             f"State: **{summary['state']}**",
             f"Wall time: {summary.get('wall_seconds', 0):.0f} s; " + (
                 'cost not tracked' if summary.get('cost_estimate_usd') is None
                 else f"cost estimate: ${summary['cost_estimate_usd']:.2f}"),
             f"Elapsed per stage (planning only, no caps): {summary.get('elapsed_seconds', {})}", '', '## Stages', '']
    lines += [f'- {name}: {info}' for name, info in summary['stages'].items()]
    if summary.get('stop_reason'):
        lines += ['', f"Stopped: {summary['stop_reason']}"]
    for failure in summary['failures']:
        lines += ['', f"Failure in {failure['stage']}: {failure['error']} (log: {failure['log']})"]
    if summary.get('report'):
        lines += ['', '## Report headline', '', '```', json.dumps(summary['report'], indent=2, default=str), '```']
    backups = summary['backups']
    lines += ['', '## Backups', '', backups['statement'],
              f"- volume copy: {backups['volume_copy'].get('outcome')} verified={backups['volume_copy']['verified']}",
              f"- HF export: verified={backups['hf_export']['verified']}", '',
              'The Pod stop is requested after this summary; its outcome is in stop_record.json on the volume '
              '(requested, not externally verified).', '']
    return '\n'.join(lines)


def _default_exporter(local_root, *, label, include, expected_units, expected_runs, repo, hub_timeout=None):
    from pilot_eval.neuron_audit_export import OnlineHubProcess
    receipt = au.export(local_root, label=label, include=include, expected_units=expected_units,
                        expected_runs=expected_runs, client=OnlineHubProcess(timeout=hub_timeout), repo=repo, attempts=1,
                        temp_dir=Path(local_root) / 'tmp')
    return dict(receipt, readback_verified=True)  # upload returns only after a fresh readback hash matched


# --- CLI ----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    seal = sub.add_parser('seal', help='turn the approved YAML draft into the sealed, typed JSON launch record')
    seal.add_argument('--draft', required=True)
    seal.add_argument('--out', required=True)
    for key in ('artifact-root', 'bundle', 'model-dir', 'work-dir'):
        seal.add_argument(f'--{key}')
    rb = sub.add_parser('rebind', help='on a replacement Pod: reseal the old record for this Pod (RUNPOD_POD_ID)')
    rb.add_argument('--record', required=True)
    rb.add_argument('--out', required=True)
    for key in ('artifact-root', 'bundle', 'model-dir', 'work-dir'):
        rb.add_argument(f'--{key}')
    for name in ['run', 'precheck', 'watchdog-copy', 'paths']:
        p = sub.add_parser(name)
        p.add_argument('--local', required=True)
        p.add_argument('--volume', required=True)
        p.add_argument('--record', required=True)
    args = parser.parse_args(argv)
    if args.command == 'seal':
        fill = {'inputs.artifact_root': args.artifact_root, 'inputs.bundle': args.bundle,
                'inputs.model_dir': args.model_dir, 'inputs.work_dir': args.work_dir}
        print(finalize_draft(args.draft, args.out, code_commit=pc._commit(),
                             pod_id=os.environ.get('RUNPOD_POD_ID'), fill=fill))
        return 0
    if args.command == 'rebind':
        fill = {'inputs.artifact_root': args.artifact_root, 'inputs.bundle': args.bundle,
                'inputs.model_dir': args.model_dir, 'inputs.work_dir': args.work_dir}
        print(rebind(args.record, args.out, code_commit=pc._commit(), pod_id=os.environ.get('RUNPOD_POD_ID', ''),
                     fill=fill))
        return 0
    record = load_launch_record(args.record, code_commit=pc._commit())
    root = run_root(args.volume, record['run_id'])
    if args.command == 'paths':  # for the launcher: the run's volume subtree and the stop commands
        print(json.dumps(dict(run_root=str(root), chain_dir=str(root / CHAIN_DIR),
                              stop=[' '.join(c) for c in stop_commands(record)],
                              copy_timeout=record['timeouts_seconds']['copy'],
                              stop_timeout=record['timeouts_seconds']['stop_attempt'])))
        return 0
    if args.command == 'watchdog-copy':
        result = copy_sealed(args.local, root)
        print(json.dumps(result, indent=2))
        return 0 if result['verified'] and not result['problems'] else 1  # the watchdog withholds its stop on 1
    if args.command == 'precheck':
        try:
            decision = precheck(record, args.volume)
        except ResumeRefused as exc:
            print(f'precheck refused: {exc}', file=sys.stderr)
            return 3
        print(json.dumps(dict(precheck='ok', run_id=record['run_id'], decision=decision['decision'])))
        return 0
    from pilot_eval import c_attribution_stages
    identity = pc.observed_volume_identity(args.volume, args.local, expected=record['volume_identity'])
    chain = Chain(record, local_root=args.local, volume=args.volume, volume_identity=identity,
                  stages=c_attribution_stages.build(record, args.local, args.volume))
    summary = chain.run()
    print(json.dumps(dict(state=summary['state'], backups=summary['backups']['statement'], stop=summary['stop']),
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
