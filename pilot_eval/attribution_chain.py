"""Ticket 11: unattended overnight chain for adapter attribution (spec §Overnight unattended chain).

preflight -> qualification -> base_traces -> attribution_and_freeze -> stage_c -> stage_d -> report, then on
*every* exit path (success, pre-production stop, cap, exception):

1. stop and join the scientific writer and background writers;
2. copy sealed, hash-verified units and sealed run directories to the volume (timeout);
3. export to the private HF dataset with fresh readback (3 attempts, each with a timeout);
4. finalize the sealed morning summary with both backup outcomes (unverified is never labelled verified);
5. write the summary to the volume;
6. request the Pod stop with each command form in turn (timeout per attempt), recording *requested* vs
   *externally verified* (an in-Pod process can only request). The user's policy stops even if the HF
   export failed, so ``neuron_audit_export.may_stop_pod`` is not used for this decision.

- **Launch record**: sealed (canonical-JSON SHA-256 in a sibling seal file) and bound to the code commit; any
  null or ``[proposed]`` value refuses the launch. Production refuses a missing arm; rehearsal validates
  against its own labelled expected set.
- **Sign-off** after qualification is the record's automatic rule, recorded as the rule, never a person. A
  failed check, review flag, missing arm, unmet minimum or no feasible budget stops before production.
  Analysis outcomes never stop the run.
- **Caps**: ``attribution_units.Budget`` persisted on the volume; stages call ``ctx.boundary()`` between
  units; reaching a cap ends that stage there and the chain goes to report-and-stop. Projections advisory.
- **Resume**: only if this run's previous worker is verifiably gone (same host, PID not running, heartbeat
  older than ``marker_stale_seconds``) and the volume identity matches; then the marker is taken over. A live
  writer, a foreign run ID or anything ambiguous stops for investigation. The marker is advisory.
- **Watchdog**: ``scripts/attribution_watchdog.sh`` runs outside this Python, timed from the persisted
  remaining total cap + margin (``watchdog_deadline``); the chain cancels it with the done file.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import traceback
from pathlib import Path

from pilot_eval import attribution_units as au
from pilot_eval.neuron_audit_blocks import BlockStore, CorruptUnit, mirror_store

LAUNCH_SCHEMA = f'{au.STUDY}/launch-record-v1'
STAGES = ['preflight', 'qualification', 'base_traces', 'attribution_and_freeze', 'stage_c', 'stage_d', 'report']
PRE_PRODUCTION = {'preflight', 'qualification', 'base_traces'}
PRODUCTION_ARMS = ['gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5']
REQUIRED = ['run_id', 'mode', 'code_commit', 'pod_id', 'pod_stop_commands', 'expected_arms', 'volume_identity',
            'export_repo', 'caps_seconds', 'tolerances', 'minimums', 'selection', 'statistics',
            'generation_batch_rule', 'token_budget_sweep', 'watchdog_margin_seconds', 'timeouts_seconds',
            'marker_stale_seconds', 'hourly_cost_usd', 'compute', 'stage_d']
STAGE_D_ARMS = ['gold-sft', 'online-grpo']
AUTO_REVIEWER = 'automatic-rule (launch record sign-off rule; recorded as the rule, not as a person)'
STATE_FILE = 'attribution/chain/state.json'
BUDGET_FILE = 'attribution/chain/budget.json'
MARKER_FILE = 'attribution/chain/writer_marker.json'
DONE_FILE = 'attribution/chain/done'
LOG_FILE = 'attribution/logs/chain.log'
LAUNCH_DIR = 'attribution/launch'


class LaunchRefused(RuntimeError):
    pass


class ResumeRefused(RuntimeError):
    pass


class StopBeforeProduction(RuntimeError):
    pass


# --- launch record ------------------------------------------------------------------------------------

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def seal_launch_record(record, path):
    """Write the record and its seal (used when the user's approved record is finalized)."""
    path = Path(path)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    path.with_suffix('.seal.json').write_text(json.dumps(dict(
        sha256=hashlib.sha256(_canonical(record).encode()).hexdigest(), sealed_unix=time.time())) + '\n')
    return path


def unresolved(value, prefix=''):
    if value is None:
        return [prefix or '<root>']
    if isinstance(value, str) and 'proposed' in value.lower():
        return [prefix]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unresolved(v, f'{prefix}.{k}' if prefix else k)]
    if isinstance(value, list):
        if not value:
            return [prefix]
        return [p for i, v in enumerate(value) for p in unresolved(v, f'{prefix}[{i}]')]
    return []


def load_launch_record(path, *, code_commit):
    path = Path(path)
    record = json.loads(path.read_text())
    seal = path.with_suffix('.seal.json')
    if not seal.is_file() or json.loads(seal.read_text())['sha256'] != hashlib.sha256(
            _canonical(record).encode()).hexdigest():
        raise LaunchRefused('launch record does not match its seal')
    if record.get('schema') != LAUNCH_SCHEMA:
        raise LaunchRefused('not a launch record')
    missing = [k for k in REQUIRED if k not in record]
    problems = unresolved(record)
    if missing or problems:
        raise LaunchRefused(f'unresolved launch record values: {(missing + problems)[:8]}')
    if record['code_commit'] != code_commit:
        raise LaunchRefused(f"code commit {code_commit} differs from the record's {record['code_commit']}")
    if record['mode'] == 'production':
        if sorted(record['expected_arms']) != sorted(PRODUCTION_ARMS):
            raise LaunchRefused('production must expect all six arms; missing arms are refused')
        if sorted(record['stage_d'].get('pilot_results', {})) != STAGE_D_ARMS:
            raise LaunchRefused('production needs saved pilot results for gold-sft and online-grpo '
                                '(stage_d.pilot_results) for the item-level comparison')
    elif record['mode'] == 'rehearsal':
        if not str(record.get('rehearsal_label', '')).startswith('rehearsal'):
            raise LaunchRefused('a rehearsal needs an explicit rehearsal label for its expected arm set')
    else:
        raise LaunchRefused('mode must be production or rehearsal')
    from pilot_eval.attribution_report import early_stop_threshold
    try:
        early_stop_threshold([dict(tokens=1, stop_reason='eos')], rule=record['stage_d'].get('early_stop_rule'))
    except ValueError as exc:
        raise LaunchRefused(f'unresolved launch record values: stage_d.early_stop_rule ({exc})')
    for name in STAGES:
        if name not in record['caps_seconds']:
            raise LaunchRefused(f'unresolved launch record values: caps_seconds.{name}')
    _ = stop_commands(record)
    return record


def stop_commands(record):
    pod = str(record['pod_id'])
    if not pod.replace('-', '').isalnum():
        raise LaunchRefused(f'unexpected Pod id {pod!r}')
    return [template.format(id=pod).split() for template in record['pod_stop_commands']]


# --- helpers ----------------------------------------------------------------------------------------

def call_with_timeout(fn, timeout, *args, **kwargs):
    """Run ``fn`` in a daemon thread; ('ok', value) | ('error', exc) | ('timeout', None)."""
    box = {}

    def target():
        try:
            box['value'] = fn(*args, **kwargs)
        except BaseException as exc:  # reported, never swallowed silently
            box['error'] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return 'timeout', None
    if 'error' in box:
        return 'error', box['error']
    return 'ok', box.get('value')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:  # direct write; no rename semantics assumed on the volume
        stream.write(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _pid_running(pid):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def resume_decision(marker_path, run_id, *, stale_seconds, host=None, now=time.time):
    host = host or socket.gethostname()
    path = Path(marker_path)
    if not path.is_file():
        return dict(decision='fresh')
    try:
        other = json.loads(path.read_text())
    except json.JSONDecodeError:
        raise ResumeRefused('ambiguous writer marker (unreadable); investigate')
    if other.get('state') == 'released':
        return dict(decision='fresh', previous=other)
    if other.get('run_id') != run_id:
        raise ResumeRefused(f"foreign run ID {other.get('run_id')!r} holds the writer marker; investigate")
    if other.get('host') == host and _pid_running(other.get('pid')):
        raise ResumeRefused(f"live writer {other.get('host')}:{other.get('pid')} holds the marker; investigate")
    age = now() - float(other.get('heartbeat_unix', now()))
    if other.get('host') != host or age < stale_seconds:
        raise ResumeRefused(f"ambiguous writer marker (host {other.get('host')}, heartbeat {age:.0f}s ago); "
                            'investigate, never take over')
    return dict(decision='take-over', previous=other, heartbeat_age_seconds=age)


def watchdog_deadline(budget, *, margin_seconds, wall_now=None):
    """Wall-clock deadline for the watchdog: now + persisted remaining total cap + margin."""
    wall_now = time.time() if wall_now is None else wall_now
    return wall_now + max(0.0, budget.remaining_total()) + margin_seconds


def copy_sealed(local_root, volume_root, *, sealed_only=True):
    """Copy only sealed, hash-verified units and run directories to the volume and re-verify the copies."""
    if not sealed_only:
        raise ValueError('only sealed outputs are copied')
    local, volume = Path(local_root), Path(volume_root)
    problems, copied = [], []
    store_root = local / au.UNITS_DIR
    if store_root.is_dir():
        sealed = []
        for key, state in BlockStore(store_root).status().items():
            if state == 'sealed':
                try:
                    BlockStore(store_root).verify_unit(key, force=True)
                    sealed.append(key)
                except CorruptUnit as exc:
                    problems.append(f'{key}: {exc}')
        try:
            copied += [f'unit:{k}' for k in mirror_store(store_root, volume / au.UNITS_DIR, keys=sealed)]
        except CorruptUnit as exc:
            problems.append(str(exc))
    for seal in sorted((local / 'attribution').rglob('complete.json')) if (local / 'attribution').is_dir() else []:
        directory = seal.parent
        if directory.is_relative_to(store_root):
            continue
        try:
            files = au.verify_run(directory)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        target = volume / directory.relative_to(local)
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, target / name)
        shutil.copyfile(seal, target / 'complete.json')
        try:
            au.verify_run(target)
            copied.append(f'run:{directory.relative_to(local)}')
        except ValueError as exc:
            problems.append(f'volume copy {target}: {exc}')
    return dict(verified=not problems, copied=copied, problems=problems)


def run_units(units, arm, batches, compute, *, boundary, input_sha256, kind='ab', condition=None):
    """Seal one unit per batch in order; sealed units are skipped (resume); ``boundary`` between units."""
    done = []
    for index, batch in enumerate(batches):
        key = au.unit_key(kind, arm, batch=index, condition=condition)
        if units.state(key) == 'sealed':
            done.append((key, 'skipped-sealed'))
            continue
        result = compute(batch)
        identity = units.identity(kind, arm, batch, input_sha256=input_sha256, batch=index, condition=condition)
        done.append((key, units.write(key, identity=identity, example_ids=batch, blocks=result['blocks'],
                                      counts=result['counts'], results=result.get('results'),
                                      config=result.get('config'), log=result.get('log'))))
        boundary()
    return done


def run_conditions(units, arm, conditions, evaluate, *, example_ids, boundary, input_sha256):
    """Stage C units per (arm, condition); a skipped condition (e.g. unavailable match) is recorded, not run."""
    skipped, done = [], []
    for condition in conditions:
        if condition['status'] == 'skipped':
            skipped.append(dict(name=condition['name'], reason=condition['reason']))
            continue
        key = au.unit_key('c', arm, condition=condition['name'])
        if units.state(key) == 'sealed':
            done.append((key, 'skipped-sealed'))
            continue
        result = evaluate(condition)
        identity = units.identity('c', arm, example_ids, input_sha256=input_sha256, condition=condition['name'])
        done.append((key, units.write(key, identity=identity, example_ids=example_ids, blocks=result['blocks'],
                                      counts=result['counts'], results=result.get('results'),
                                      config=dict(condition=condition))))
        boundary()
    return dict(done=done, skipped=skipped)


# --- the chain ------------------------------------------------------------------------------------------

class Context:
    def __init__(self, chain, stage):
        self.chain, self.stage, self.record = chain, stage, chain.record
        self.local_root, self.volume_root = chain.local_root, chain.volume_root
        self.writers = chain.writers

    def boundary(self):
        return self.chain.budget.boundary(self.stage)

    def projection(self, *, done, total):
        record = self.chain.budget.projection(self.stage, done=done, total=total)
        self.chain.log(f'projection (advisory): {json.dumps(record)}')
        return record

    def expect(self, *, units=(), runs=()):
        self.chain.expected_units |= set(units)
        self.chain.expected_runs |= set(runs)


class Chain:
    def __init__(self, record, *, local_root, volume_root, stages, copier=copy_sealed, exporter=None,
                 runner=subprocess.run, now=time.monotonic, writers=(), volume_identity=None, host=None):
        self.record, self.stages = record, stages
        self.local_root, self.volume_root = Path(local_root), Path(volume_root)
        self.copier, self.exporter, self.runner, self.now = copier, exporter or _default_exporter, runner, now
        self.writers, self.volume_identity, self.host = list(writers), volume_identity, host or socket.gethostname()
        self.expected_units, self.expected_runs = set(), {au.SUMMARY_DIR, LAUNCH_DIR}
        self.timeouts = record['timeouts_seconds']
        self.budget = None

    def log(self, message):
        path = self.local_root / LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}\n')

    # --- start -------------------------------------------------------------------------------------

    def _start(self):
        if self.volume_identity != self.record['volume_identity']:
            raise ResumeRefused(f'volume identity {self.volume_identity!r} differs from the launch record')
        marker = self.volume_root / MARKER_FILE
        decision = resume_decision(marker, self.record['run_id'], stale_seconds=self.record['marker_stale_seconds'],
                                   host=self.host)
        state_path = self.volume_root / STATE_FILE
        state = json.loads(state_path.read_text()) if state_path.is_file() else dict(
            run_id=self.record['run_id'], completed={})
        if state['run_id'] != self.record['run_id']:
            raise ResumeRefused('chain state belongs to another run ID')
        self._marker = dict(run_id=self.record['run_id'], host=self.host, pid=os.getpid(), started_unix=time.time())
        self._beat('running')
        caps = dict(self.record['caps_seconds'])
        total = caps.pop('total')
        self.budget = au.Budget(self.volume_root / BUDGET_FILE, stage_caps=caps, total_cap=total, now=self.now)
        au.seal_run(self.local_root / LAUNCH_DIR, {'launch_record.json': self.record})
        return state, decision

    def _beat(self, state, **extra):
        _write_json(self.volume_root / MARKER_FILE, dict(self._marker, state=state, heartbeat_unix=time.time(), **extra))

    def _save(self, state):
        _write_json(self.volume_root / STATE_FILE, state)

    # --- sign-off ------------------------------------------------------------------------------------

    def _signoff(self, result):
        reasons = [f'check failed: {k}' for k, v in sorted(result.get('checks', {}).items()) if v is not True]
        if not result.get('checks'):
            reasons.append('no qualification checks recorded')
        reasons += [f'review flag: {f}' for f in result.get('review_flags', [])]
        missing = sorted(set(self.record['expected_arms']) - set(result.get('arms', [])))
        if missing:
            reasons.append(f'missing arms: {missing}')
        if reasons:
            raise StopBeforeProduction('; '.join(reasons))
        record = dict(by=AUTO_REVIEWER, decision='accept', rule='every qualification check passes against the '
                      "launch record's fixed tolerances and no review flag is raised", checks=result['checks'],
                      signed_unix=time.time())
        au.seal_run(self.local_root / 'attribution/signoff', {'signoff.json': record})
        return record

    # --- run -------------------------------------------------------------------------------------------

    def run(self):
        from pilot_eval.attribution_traces import MinimumUnmet
        state, decision = self._start()
        started = time.time()
        summary = dict(run_id=self.record['run_id'], mode=self.record['mode'], resume=decision, stages={},
                       failures=[], state='running', signoff=state.get('signoff'))
        skip_to_report = False
        try:
            for name in STAGES:
                if name in state['completed']:
                    summary['stages'][name] = dict(status='complete', resumed=True)
                    continue
                if skip_to_report and name != 'report':
                    summary['stages'][name] = dict(status='not-run', reason='cap reached earlier')
                    continue
                self.budget.start(name)
                self._beat('running', stage=name)
                self.log(f'stage {name} start')
                try:
                    result = self.stages[name](Context(self, name)) or {}
                except au.CapReached as exc:
                    self.log(f'stage {name} cap reached: {exc}')
                    summary['stages'][name] = dict(status='cap-reached', scope=exc.scope)
                    if name in PRE_PRODUCTION:
                        summary['state'], summary['stop_reason'] = 'stopped-before-production', str(exc)
                        break
                    skip_to_report, summary['state'] = True, 'cap-reached'
                    continue
                except (StopBeforeProduction, MinimumUnmet) as exc:
                    summary['stages'][name] = dict(status='stopped', reason=str(exc))
                    summary['state'], summary['stop_reason'] = 'stopped-before-production', str(exc)
                    break
                summary['stages'][name] = dict(status='complete')
                state['completed'][name] = result
                if name == 'qualification':
                    try:
                        state['signoff'] = summary['signoff'] = self._signoff(result)
                    except StopBeforeProduction as exc:
                        summary['stages'][name] = dict(status='stopped', reason=str(exc))
                        summary['state'], summary['stop_reason'] = 'stopped-before-production', str(exc)
                        self._save(state)
                        break
                if name == 'report':
                    summary['report'] = result
                self._save(state)
            else:
                if summary['state'] == 'running':
                    summary['state'] = 'complete'
        except BaseException as exc:  # every exit path tears down
            self.log('failure:\n' + traceback.format_exc())
            summary['state'] = 'failed'
            summary['failures'].append(dict(stage=next((s for s in STAGES if s not in summary['stages']), None),
                                            error=f'{type(exc).__name__}: {exc}', log=LOG_FILE))
        finally:
            self._save(state)
            summary['wall_seconds'] = time.time() - started
            summary['cost_estimate_usd'] = summary['wall_seconds'] / 3600 * self.record['hourly_cost_usd']
            summary['budget'] = dict(elapsed=self.budget.state['elapsed'], remaining_total=self.budget.remaining_total())
            self.teardown(summary)
        return summary

    # --- teardown ------------------------------------------------------------------------------------

    def teardown(self, summary):
        steps = summary['teardown'] = []
        for writer in self.writers:
            writer.stop()
        joined = [bool(writer.join(timeout=self.timeouts['writer_join'])) for writer in self.writers]
        summary['writers'] = dict(count=len(self.writers), joined=joined)
        steps.append('writers_joined')
        kind, value = call_with_timeout(self.copier, self.timeouts['copy'], self.local_root, self.volume_root,
                                        sealed_only=True)
        if kind == 'timeout':
            copy = dict(verified=False, outcome='timeout', timeout_seconds=self.timeouts['copy'])
        elif kind == 'error':
            copy = dict(verified=False, outcome='error', error=f'{type(value).__name__}: {value}')
        else:
            copy = dict(value, outcome='ok', verified=bool(value.get('verified')))
        steps.append('volume_copy')
        attempts, export = [], dict(verified=False)
        for attempt in range(3):
            kind, value = call_with_timeout(self.exporter, self.timeouts['export_attempt'], self.local_root,
                                            label=self.record['run_id'], expected_units=sorted(self.expected_units),
                                            expected_runs=sorted(self.expected_runs), repo=self.record['export_repo'])
            if kind == 'ok':
                attempts.append(dict(outcome='ok'))
                export = dict(verified=bool(value.get('readback_verified', True)), receipt=value,
                              completeness=value.get('completeness'))
                break
            attempts.append(dict(outcome=kind, error=None if value is None else f'{type(value).__name__}: {value}'))
        export['attempts'] = attempts
        steps.append('hf_export')
        verified = [n for n, b in [('volume copy', copy), ('HF export', export)] if b['verified']]
        statement = ('verified backups: ' + ', '.join(verified)) if verified else \
            'NO VERIFIED BACKUP; possibly missing outputs: every unit and run directory not on the volume or HF'
        summary['backups'] = dict(volume_copy=copy, hf_export=export, statement=statement)
        missing = sorted(self.expected_units - {c.split(':', 1)[1] for c in copy.get('copied', [])
                                                if c.startswith('unit:')}) if self.expected_units else []
        markdown = _markdown(summary)
        try:
            au.write_summary(self.local_root, summary=summary, markdown=markdown, missing_work=missing)
        except ValueError:  # a resumed run that already sealed a summary: keep a dated copy beside it
            au.write_summary(self.local_root / f'attribution/summaries/{int(time.time())}', summary=summary,
                             markdown=markdown, missing_work=missing)
        steps.append('summary_sealed')
        kind, value = call_with_timeout(self._summary_to_volume, self.timeouts['copy'])
        summary['summary_on_volume'] = kind == 'ok' and bool(value)
        steps.append('summary_on_volume')
        summary['stop'] = self._stop()
        steps.append('stop_requested')
        _write_json(self.volume_root / 'attribution/chain/stop_record.json', dict(stop=summary['stop'], steps=steps))
        self._beat('released', final_state=summary['state'])
        (self.volume_root / DONE_FILE).write_text(summary['state'] + '\n')
        return summary

    def _summary_to_volume(self):
        source, target = self.local_root / au.SUMMARY_DIR, self.volume_root / au.SUMMARY_DIR
        files = au.verify_run(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(files) + ['complete.json']:
            shutil.copyfile(source / name, target / name)
        au.verify_run(target)
        return True

    def _stop(self):
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


def _markdown(summary):
    lines = [f"# Morning summary — {summary['run_id']} ({summary['mode']})", '', f"State: **{summary['state']}**",
             f"Wall time: {summary.get('wall_seconds', 0):.0f} s; cost estimate: ${summary.get('cost_estimate_usd', 0):.2f}",
             '', '## Stages', '']
    lines += [f'- {name}: {info}' for name, info in summary['stages'].items()]
    if summary.get('stop_reason'):
        lines += ['', f"Stopped: {summary['stop_reason']}"]
    for failure in summary['failures']:
        lines += ['', f"Failure in {failure['stage']}: {failure['error']} (log: {failure['log']})"]
    if summary.get('report'):
        lines += ['', '## Gates and primaries', '', '```', json.dumps(summary['report'], indent=2, default=str), '```']
    backups = summary['backups']
    lines += ['', '## Backups', '', backups['statement'],
              f"- volume copy: {backups['volume_copy'].get('outcome')} verified={backups['volume_copy']['verified']}",
              f"- HF export: verified={backups['hf_export']['verified']} attempts={len(backups['hf_export']['attempts'])}",
              '', 'The Pod stop is requested after this summary is written; its outcome is in stop_record.json on '
              'the volume (requested, not externally verified).', '']
    return '\n'.join(lines)


def _default_exporter(local_root, *, label, expected_units, expected_runs, repo):
    from pilot_eval.neuron_audit_export import OnlineHubProcess
    receipt = au.export(local_root, label=label, include=['attribution'], expected_units=expected_units,
                        expected_runs=expected_runs, client=OnlineHubProcess(), repo=repo, attempts=1,
                        temp_dir=Path(local_root) / 'tmp')
    return dict(receipt, readback_verified=True)  # upload returns only after a fresh readback hash matched


# --- CLI ----------------------------------------------------------------------------------------------

def _commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True,
                          cwd=Path(__file__).resolve().parents[1]).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'deadline', 'watchdog-copy']:
        p = sub.add_parser(name)
        p.add_argument('--local', required=True)
        p.add_argument('--volume', required=True)
        if name != 'watchdog-copy':
            p.add_argument('--record', required=True)
    args = parser.parse_args(argv)
    if args.command == 'watchdog-copy':
        print(json.dumps(copy_sealed(args.local, args.volume), indent=2))
        return 0
    record = load_launch_record(args.record, code_commit=_commit())
    if args.command == 'deadline':
        caps = dict(record['caps_seconds'])
        total = caps.pop('total')
        budget = au.Budget(Path(args.volume) / BUDGET_FILE, stage_caps=caps, total_cap=total)
        print(f"{watchdog_deadline(budget, margin_seconds=record['watchdog_margin_seconds']):.0f}")
        return 0
    from pilot_eval import attribution_stages
    identity_path = Path(args.volume) / 'volume_identity.json'
    identity = json.loads(identity_path.read_text()).get('identity') if identity_path.is_file() else None
    chain = Chain(record, local_root=args.local, volume_root=args.volume,
                  stages=attribution_stages.build(record, args.local, args.volume), volume_identity=identity)
    summary = chain.run()
    print(json.dumps(dict(state=summary['state'], backups=summary['backups']['statement'], stop=summary['stop']),
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
