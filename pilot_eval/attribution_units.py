"""Ticket 05: units, sealed run directories, hard-cap budget, export and restore for adapter attribution.

Built on the joint audit's primitives (``neuron_audit_blocks.BlockStore``, ``neuron_audit_export``):

- **Units** (arrays): A+B per (arm, batch) ``ab-<arm>-b<NNNNN>``; C per (arm, condition)
  ``c-<arm>-<condition>``; D per (arm, condition, batch) ``d-<arm>-<condition>-b<NNNNN>``. Each is one
  sealed ``BlockStore`` unit whose identity carries the kind, arm, condition/batch, example IDs and the
  input, plan, code and runtime hashes; per-unit config, log and per-example results live in the sealed
  index. A sealed unit with the same identity is reused after hash verification (resume); a different
  identity or an example already sealed in another unit of the same (kind, arm, condition) is refused.
- **Sealed run directories** (everything else: launch record, freezes, trace/response JSONL, summaries,
  receipts): files plus ``complete.json`` listing their SHA-256, the layout ``neuron_audit_export.snapshot``
  archives. A standalone file outside such a directory is not exported, by design.
- **Budget**: per-stage and total hard caps checked at unit boundaries, export time included, elapsed time
  persisted so a restart continues the remaining cap. Projections are advisory and never stop a stage.
- **Export/restore**: snapshot with a declared expected set of units and run directories (``partial``
  unless all are sealed), online upload with fresh readback, and clean-root restore that brings back the
  morning summary and the missing-work list.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from pilot_eval import neuron_audit_export as ex
from pilot_eval.neuron_audit_blocks import BlockStore, CorruptUnit

STUDY = 'adapter-neuron-attribution'
UNITS_DIR = 'attribution/store'
SUMMARY_DIR = 'attribution/summaries'   # <attempt>/<phase>/ ; phases in order: pre-export, final
SUMMARY_PHASES = ['pre-export', 'final']
RUN_SCHEMA = f'{STUDY}/sealed-run-v1'
UNIT_IDENTITY_SCHEMA = f'{STUDY}/unit-v1'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


class CapReached(RuntimeError):
    def __init__(self, scope, elapsed, cap):
        super().__init__(f'{scope} cap reached: {elapsed:.1f}s of {cap:.1f}s')
        self.scope, self.elapsed, self.cap = scope, elapsed, cap


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _name(value, what):
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError(f'{what} must be a simple name: {value!r}')
    return value


def unit_key(kind, arm, *, batch=None, condition=None):
    _name(arm, 'arm')
    if kind == 'ab' and batch is not None and condition is None:
        return f'ab-{arm}-b{int(batch):05d}'
    if kind == 'c' and condition is not None and batch is None:
        return f'c-{arm}-{_name(condition, "condition")}'
    if kind == 'd' and condition is not None and batch is not None:
        return f'd-{arm}-{_name(condition, "condition")}-b{int(batch):05d}'
    raise ValueError(f'unit kind {kind!r} needs: ab(batch), c(condition), d(condition, batch)')


class Units:
    """This study's units over one ``BlockStore`` at ``<root>/attribution/store``."""

    def __init__(self, root, *, plan_sha256, code_sha256, runtime_sha256):
        self.root = Path(root)
        self.store = BlockStore(self.root / UNITS_DIR)
        self.hashes = dict(plan_sha256=plan_sha256, code_sha256=code_sha256, runtime_sha256=runtime_sha256)
        if any(not isinstance(v, str) or not v for v in self.hashes.values()):
            raise ValueError('plan, code and runtime hashes are required')

    def identity(self, kind, arm, example_ids, *, input_sha256, batch=None, condition=None):
        unit_key(kind, arm, batch=batch, condition=condition)
        return dict(schema=UNIT_IDENTITY_SCHEMA, kind=kind, arm=arm, batch=batch, condition=condition,
                    example_ids=list(example_ids), input_sha256=input_sha256, **self.hashes)

    def state(self, key):
        state = self.store.state(key)
        if state != 'sealed':
            return state
        try:
            self.store.verify_unit(key, force=True)
        except CorruptUnit:
            return 'corrupt'
        return 'sealed'

    def _sealed_examples(self, identity, key):
        """Example IDs already sealed in other units of the same kind, arm and condition."""
        seen = set()
        for other, state in self.store.status().items():
            if other == key or state != 'sealed':
                continue
            other_identity = json.loads((self.store.unit_dir(other) / 'seal.json').read_text())['identity']
            if all(other_identity.get(k) == identity[k] for k in ('kind', 'arm', 'condition')):
                seen |= set(other_identity.get('example_ids', []))
        return seen

    def write(self, key, *, identity, example_ids, blocks, counts, results=None, config=None, log=None):
        """Seal one unit; 'written' or 'reused' (same identity, hash-verified)."""
        if identity['example_ids'] != list(example_ids) or unit_key(
                identity['kind'], identity['arm'], batch=identity['batch'], condition=identity['condition']) != key:
            raise ValueError('unit identity does not match its key or example IDs')
        if any(identity.get(k) != v for k, v in self.hashes.items()):
            raise ValueError('unit identity carries different plan/code/runtime hashes')
        if self.store.state(key) != 'sealed':
            overlap = self._sealed_examples(identity, key) & set(example_ids)
            if overlap:
                raise ValueError(f'duplicate examples already sealed in another unit: {sorted(overlap)[:5]}')
        meta = dict(config=config or {}, results=results or [], log=log or '')
        if results is not None and len(results) != len(example_ids):
            raise ValueError('one per-example result record per example')
        return self.store.write_unit(key, identity=identity, example_ids=list(example_ids), counts=counts,
                                     blocks=blocks, meta=meta)

    def read(self, key):
        index = self.store.index(key)
        arrays = {(b['group'], b['layer'], b['stat']): self.store.read_block(key, b['group'], b['layer'], b['stat'])
                  for b in index['blocks']}
        return arrays, index['meta']

    def manifest(self, expected_keys):
        """Partial-state manifest of the expected units (never raises on corruption)."""
        states = {k: self.state(k) for k in sorted(set(expected_keys) | set(self.store.status()))}
        by = {s: sorted(k for k, v in states.items() if v == s) for s in ('sealed', 'interrupted', 'corrupt', 'missing')}
        missing_work = sorted(k for k in expected_keys if states[k] != 'sealed')
        return dict(by, expected=sorted(expected_keys), missing_work=missing_work,
                    completeness='complete' if not missing_work and not by['corrupt'] else 'partial')


# --- sealed run directories ----------------------------------------------------------------------

def _encode(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + '\n').encode()


def _fsync(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def seal_run(directory, files):
    """Write ``files`` ({relative name: bytes | str | JSON value}) and seal them with ``complete.json`` last.

    Returns 'written', or 'reused' when an identical sealed directory exists; any difference is refused.
    """
    directory = Path(directory)
    data = {}
    for name, value in files.items():
        parts = Path(name).parts
        if not parts or Path(name).is_absolute() or '..' in parts or name in ('complete.json',) or \
                name.startswith('meta/'):
            raise ValueError(f'unsafe or reserved run file name: {name}')
        data[name] = _encode(value)
    digests = {n: _sha_bytes(d) for n, d in sorted(data.items())}
    seal_path = directory / 'complete.json'
    if seal_path.is_file():
        existing = json.loads(seal_path.read_text())
        if existing.get('files') != digests:
            raise ValueError(f'sealed run {directory} exists with different content; refusing to rewrite')
        verify_run(directory)
        return 'reused'
    if directory.exists() and any(directory.iterdir()):
        evidence = directory.with_name(f'{directory.name}.interrupted-{time.time_ns()}')
        os.replace(directory, evidence)
    for name, payload in data.items():
        _fsync(directory / name, payload)
    tmp = directory / 'complete.json.tmp'
    _fsync(tmp, _encode(dict(schema=RUN_SCHEMA, files=digests, sealed_unix=time.time())))
    os.replace(tmp, seal_path)
    return 'written'


def verify_run(directory):
    directory = Path(directory)
    seal = directory / 'complete.json'
    if not seal.is_file():
        raise ValueError(f'run {directory} is not sealed')
    files = json.loads(seal.read_text())['files']
    for name, digest in files.items():
        path = directory / name
        if not path.is_file() or _sha_bytes(path.read_bytes()) != digest:
            raise ValueError(f'sealed run {directory}: {name} failed its hash')
    return set(files)


def summary_dir(attempt, phase):
    """Relative directory of one attempt's summary in one phase (each attempt and phase is its own sealed run)."""
    if phase not in SUMMARY_PHASES:
        raise ValueError(f'summary phase must be one of {SUMMARY_PHASES}')
    return f'{SUMMARY_DIR}/{_name(attempt, "attempt")}/{phase}'


def write_summary(root, *, summary, markdown, missing_work, attempt='attempt-001', phase='final'):
    """The morning summary (``run_summary.md`` + ``run_summary.json``) and missing-work list, sealed.

    A retry writes its own attempt directory, so it never collides with (or copies) an older summary."""
    relative = summary_dir(attempt, phase)
    seal_run(Path(root) / relative, {'run_summary.json': summary, 'run_summary.md': markdown,
                                     'missing_work.json': list(missing_work)})
    return relative


def latest_summary(root):
    """The newest sealed summary under ``root``: latest attempt, then the latest phase within it."""
    base = Path(root) / SUMMARY_DIR
    found = []
    for seal in base.glob('*/*/complete.json') if base.is_dir() else []:
        attempt, phase = seal.parent.parent.name, seal.parent.name
        if phase in SUMMARY_PHASES:
            found.append((attempt, SUMMARY_PHASES.index(phase), seal.parent))
    if not found:
        raise ValueError(f'no sealed summary under {base}')
    attempt, phase_index, directory = max(found)
    verify_run(directory)
    return dict(attempt=attempt, phase=SUMMARY_PHASES[phase_index], directory=str(directory),
                summary=json.loads((directory / 'run_summary.json').read_text()),
                markdown=(directory / 'run_summary.md').read_text(),
                missing_work=json.loads((directory / 'missing_work.json').read_text()))


# --- budget ------------------------------------------------------------------------------------

class Budget:
    """Hard caps (seconds) per stage and in total, enforced at unit boundaries; elapsed time persisted."""

    def __init__(self, path, *, stage_caps, total_cap, now=time.monotonic):
        caps = dict(stage_caps)
        if not caps or any(not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0
                           for v in list(caps.values()) + [total_cap]):
            raise ValueError('every stage cap and the total cap must be a fixed positive number')
        self.path, self.now = Path(path), now
        self.state = dict(stage_caps=caps, total_cap=total_cap, elapsed={}, charges=[])
        if self.path.is_file():
            saved = json.loads(self.path.read_text())
            if saved['stage_caps'] != caps or saved['total_cap'] != total_cap:
                raise ValueError('persisted budget has different caps; refusing to reuse it')
            self.state = saved
        self.mark = None
        self.stage = None
        self._save()

    def _save(self):
        _fsync(self.path, _encode(self.state))

    def elapsed(self, stage=None):
        if stage is None:
            return sum(self.state['elapsed'].values())
        return self.state['elapsed'].get(stage, 0.0)

    def remaining_total(self):
        return self.state['total_cap'] - self.elapsed()

    def start(self, stage):
        self.stage, self.mark = stage, self.now()

    def charge(self, stage, seconds):
        self.state['elapsed'][stage] = self.elapsed(stage) + float(seconds)
        self.state['charges'].append(dict(stage=stage, seconds=float(seconds)))
        self._save()

    def boundary(self, stage):
        """Charge time since the last mark; raise ``CapReached`` if a cap is met. Call between units."""
        if stage != self.stage or self.mark is None:
            raise ValueError(f'stage {stage} was not started')
        now = self.now()
        self.state['elapsed'][stage] = self.elapsed(stage) + (now - self.mark)
        self.mark = now
        self._save()
        cap = self.state['stage_caps'].get(stage)
        if cap is not None and self.elapsed(stage) >= cap:
            raise CapReached(stage, self.elapsed(stage), cap)
        if self.elapsed() >= self.state['total_cap']:
            raise CapReached('total', self.elapsed(), self.state['total_cap'])
        return dict(stage=stage, elapsed_stage=self.elapsed(stage), remaining_total=self.remaining_total())

    def close(self, stage):
        """Charge the time since the last mark when a stage ends (success, cap, stop or exception)."""
        if self.stage == stage and self.mark is not None:
            now = self.now()
            self.state['elapsed'][stage] = self.elapsed(stage) + (now - self.mark)
            self.mark = None
            self._save()
        return self.elapsed(stage)

    def exceeded(self, stage):
        """Which cap, if any, is met now: the stage's own or the total."""
        cap = self.state['stage_caps'].get(stage)
        if cap is not None and self.elapsed(stage) >= cap:
            return stage
        if self.elapsed() >= self.state['total_cap']:
            return 'total'
        return None

    def projection(self, stage, *, done, total):
        """Linear forecast, logged only: longest-first packing makes early forecasts pessimistic."""
        projected = self.elapsed(stage) / done * total if done else None
        return dict(stage=stage, done=done, total=total, projected_seconds=projected, advisory=True,
                    note='advisory only; never stops a stage')


# --- export / restore --------------------------------------------------------------------------

def export(root, *, label, include, expected_units, expected_runs, client=None, repo=ex.DEFAULT_REPO,
           attempts=3, deadline_seconds=None, temp_dir=None, on_event=print):
    # ``include`` may name a single sealed directory (e.g. the final summary's second, small upload)
    """Snapshot (declared expected set) -> private upload with fresh readback. Incomplete scope is ``partial``."""
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    snap = ex.snapshot(root, include, label, expected_units=expected_units, expected_runs=expected_runs,
                       stage='attribution')
    receipt = ex.upload(Path(root) / snap['snapshot_path'], root, repo=repo, client=client, attempts=attempts,
                        deadline_seconds=deadline_seconds, on_event=on_event, temp_dir=temp_dir)
    return dict(receipt, snapshot_path=snap['snapshot_path'], completeness=snap['completeness'],
                missing_expected_units=snap['missing_expected_units'],
                missing_expected_runs=snap['missing_expected_runs'])


def restore_summary(receipt, clean_root, *, client=None, attempts=3, on_event=print):
    """Clean-root restore; returns the restore proof, morning summary and missing-work list."""
    proof = ex.restore(receipt, clean_root, client=client, attempts=attempts, on_event=on_event)
    return dict(latest_summary(clean_root), proof=proof)
