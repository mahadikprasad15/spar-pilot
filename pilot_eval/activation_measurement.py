"""Frozen production execution, verified numeric shards and CPU completion checks."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import load_prepared, STEPS, VIEWS, PROJECTIONS
from pilot_eval.activation_profile import BATCHES, AGREEMENT, _relative, _verified, _check_gate
from pilot_eval.activation_workflow import HFActivationDependencies, _save_arrays, _verify_complete
from pilot_eval.run import _write_json, _write_state
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen

BASE_KEYS = {'block_count', 'block_base_sum', 'block_base_norm_sum'}
CHECKPOINT_KEYS = {'block_delta_sum', 'block_delta_norm_sum', 'module_count',
                   'module_base_norm_sum', 'module_delta_norm_sum', 'module_ratio_sum',
                   'module_defined_count'}
THRESHOLDS = {'atol': 1e-6, 'rtol': 1e-5, 'rounding_factor': 4}
AXES = {'block_vectors': ['example', 'view', 'layer', 'hidden'],
        'block_scalars': ['example', 'view', 'layer'],
        'module_scalars': ['example', 'view', 'layer', 'projection']}


def load_execution(config_path, output_root):
    """Read-only verification of preparation, profile, review and ordered batches."""
    root = Path(output_root).resolve()
    path = _relative(config_path, root)
    execution = json.loads(path.read_text())
    prepared_path = _relative(execution['prepared_path'], root)
    prepared, rows = load_prepared(prepared_path, root)
    name = execution['run_id']
    batch_size = execution['batch_size']
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
            or batch_size not in BATCHES or execution['schema_version'] != 1
            or execution['state'] != 'frozen' or execution['checkpoint_steps'] != STEPS
            or execution['views'] != VIEWS or execution['projections'] != PROJECTIONS
            or execution['precision'] != 'float32' or execution['reduction_precision'] != 'float64'
            or execution['rank1_thresholds'] != THRESHOLDS or execution['agreement'] != AGREEMENT
            or execution['prepared_sha256'] != _hash(prepared)
            or execution['inputs_sha256'] != prepared['items_sha256']
            or not execution['review']['notes'].strip()
            or execution['review']['batch_size'] != batch_size):
        raise ValueError('unsupported or incompatible frozen execution identity')
    if path != root / 'plans' / name / 'activation.execution.json':
        raise ValueError('frozen execution manifest/name path mismatch')
    if name == prepared['run_id']:
        raise ValueError('production execution needs a distinct name from input preparation')
    if execution['run_path'] != str(Path(prepared['run_path']).parent / name):
        raise ValueError('frozen execution run path mismatch')
    if len(rows) != 300 or Counter(r['corpus'] for r in rows) != {'gsm8k': 150, 'fineweb': 150}:
        raise ValueError('production requires both complete frozen cohorts')
    expected = [{'index': i // batch_size, 'example_ids': [r['id'] for r in rows[i:i + batch_size]],
                 'inputs_sha256': _hash(rows[i:i + batch_size])} for i in range(0, len(rows), batch_size)]
    if execution['batches'] != expected or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('frozen batch membership/ordering mismatch')
    profile = _relative(execution['profile_path'], root)
    if file_hash(profile / 'complete.json') != execution['profile_complete_sha256']:
        raise ValueError('frozen profile marker hash mismatch')
    identity = json.loads((profile / 'config.json').read_text())
    result = _verified(profile, identity, ['config.json', 'results.json'] + [
        f'batch-{b}/{p}' for b in BATCHES for p in ['complete.json', 'results.json']])
    if (identity['prepared_sha256'] != _hash(prepared) or identity['runtime'] != execution['runtime']
            or identity['agreement'] != AGREEMENT
            or not any(r['batch_size'] == batch_size and r['status'] == 'passed' for r in result['measurements'])):
        raise ValueError('production profile/runtime or passing-candidate mismatch')
    diagnostic_dir = root / prepared['run_path'] / 'validation' / (
        'diagnostic-' + _hash(identity['validation_identity']['example_ids'])[:12])
    if file_hash(diagnostic_dir / 'complete.json') != execution['validation_complete_sha256']:
        raise ValueError('frozen diagnostic marker hash mismatch')
    if not _verify_complete(diagnostic_dir, identity['validation_identity'])['all_gates_passed']:
        raise ValueError('production instrument diagnostic failed')
    return execution, prepared, rows


def _identity(execution, batch, *, step=None, baseline=None):
    identity = {'execution_sha256': _hash(execution), 'prepared_sha256': execution['prepared_sha256'],
                'batch': batch, 'kind': 'baseline' if step is None else 'checkpoint'}
    if step is not None:
        identity.update(step=step, baseline_complete_sha256=baseline)
    return identity


def _validate_arrays(arrays, rows, *, baseline=None, step=None):
    keys = BASE_KEYS if baseline is None else CHECKPOINT_KEYS
    if set(arrays) != keys or any(v.dtype != np.float64 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('incomplete, nonfinite or non-FP64 shard arrays')
    count = np.repeat(np.array([[sum(row['masks'][v]) for v in VIEWS] for row in rows],
                               dtype=np.float64)[..., None], 28, axis=2)
    hidden = arrays['block_base_sum'].shape[-1] if baseline is None else baseline['block_base_sum'].shape[-1]
    if hidden < 1:
        raise ValueError('invalid hidden axis')
    for key, value in arrays.items():
        shape = (len(rows), 3, 28, 7) if key.startswith('module_') else (
            (len(rows), 3, 28, hidden) if key.endswith('_sum') and key in ['block_base_sum', 'block_delta_sum']
            else (len(rows), 3, 28))
        if value.shape != shape:
            raise ValueError('shard summary shape mismatch')
        mask = np.repeat((count > 0)[..., None], shape[-1], axis=-1) if value.ndim == 4 else count > 0
        if np.count_nonzero(value[~mask]):
            raise ValueError('uncounted view contains nonzero summary')
        if key not in ['block_base_sum', 'block_delta_sum'] and (value < 0).any():
            raise ValueError('negative summary count or magnitude')
    if baseline is None:
        if not np.array_equal(arrays['block_count'], count):
            raise ValueError('baseline counts do not match frozen masks')
    else:
        module_count = np.repeat(count[..., None], 7, axis=-1)
        defined = arrays['module_defined_count']
        if (not np.array_equal(arrays['module_count'], module_count)
                or (defined > module_count).any() or not np.array_equal(defined, np.floor(defined))
                or np.count_nonzero(arrays['module_ratio_sum'][defined == 0])
                or np.count_nonzero(defined[arrays['module_base_norm_sum'] == 0])):
            raise ValueError('checkpoint count or undefined coverage mismatch')
        if step == 0 and any(np.count_nonzero(arrays[key]) for key in [
                'block_delta_sum', 'block_delta_norm_sum', 'module_delta_norm_sum', 'module_ratio_sum']):
            raise ValueError('step 0 saved summaries contain nonzero write')


def _validate_evidence(evidence, prepared, step, rows):
    _check_gate(evidence, prepared, step)
    block_hooks = [f'model.layers.{i}' for i in range(28)]
    module_hooks = [f"model.layers.{i}.{'self_attn' if p in PROJECTIONS[:4] else 'mlp'}.{p}"
                    for i in range(28) for p in PROJECTIONS]
    if (evidence['block_hooks'] != block_hooks or evidence['module_hooks'] != module_hooks
            or evidence['sample_limit'] != 16 or evidence['thresholds'] != THRESHOLDS
            or evidence['block_position'] != 'decoder-block-output-before-final-model-norm'):
        raise ValueError('checkpoint hook identity or validation policy mismatch')
    expected = {}
    for view in VIEWS:
        by_example = [[i for i, chosen in enumerate(row['masks'][view]) if chosen] for row in rows]
        positions = [[rows[b]['id'], indices[offset]]
                     for offset in range(max(map(len, by_example), default=0))
                     for b, indices in enumerate(by_example) if offset < len(indices)][:16]
        if positions:
            expected.update({(layer, projection, view): positions
                             for layer in range(28) for projection in PROJECTIONS})
    seen = set()
    for record in evidence['positions']:
        key = (record['layer'], record['projection'], record['view'])
        if key in seen or key not in expected or record['positions'] != expected[key]:
            raise ValueError('rank-1 sampled token/module coverage mismatch')
        seen.add(key)
        for name in ['branch_max_error', 'subtraction_max_error', 'branch_max_fraction', 'subtraction_max_fraction']:
            value = record[name]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError('rank-1 nonfinite or negative validation evidence')
        if record['branch_max_fraction'] > 1 or record['subtraction_max_fraction'] > 1:
            raise ValueError('rank-1 saved threshold exceeded')
        count, below = record['coordinate_count'], record['below_resolution_coordinates']
        if (not isinstance(count, int) or count < len(expected[key])
                or not isinstance(below, int) or not 0 <= below <= count):
            raise ValueError('rank-1 resolution coverage mismatch')
    if seen != set(expected):
        raise ValueError('rank-1 incomplete sampled module/view coverage')



def _commit_shard(directory, identity, arrays, *, evidence=None):
    # Only these owned, unmarked payloads are replaced. No marked data is discarded.
    if (directory / 'complete.json').exists():
        raise ValueError('cannot overwrite a marked shard')
    _save_arrays(directory / 'summaries.npz', arrays)
    metadata = {'schema_version': 1, 'identity': identity, 'axes': AXES,
                'shapes': {k: list(v.shape) for k, v in arrays.items()},
                'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                'views': VIEWS, 'projections': PROJECTIONS}
    _write_json(directory / 'metadata.json', metadata)
    files = ['summaries.npz', 'metadata.json']
    if evidence is not None:
        _write_json(directory / 'validation.json', evidence)
        files.append('validation.json')
    _write_json(directory / 'complete.json', {'identity': identity,
                 'files': {name: file_hash(directory / name) for name in files}})


def _read_shard(directory, identity, rows, prepared, *, baseline=None, step=None):
    marker = json.loads((directory / 'complete.json').read_text())
    required = {'summaries.npz', 'metadata.json'} | ({'validation.json'} if step is not None else set())
    if marker['identity'] != identity or set(marker['files']) != required:
        raise ValueError('shard identity or completion payload inventory mismatch')
    for name, digest in marker['files'].items():
        if file_hash(directory / name) != digest:
            raise ValueError('marked shard payload hash mismatch')
    with np.load(directory / 'summaries.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    _validate_arrays(arrays, rows, baseline=baseline, step=step)
    metadata = json.loads((directory / 'metadata.json').read_text())
    expected = {'schema_version': 1, 'identity': identity, 'axes': AXES,
                'shapes': {k: list(v.shape) for k, v in arrays.items()},
                'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                'views': VIEWS, 'projections': PROJECTIONS}
    if metadata != expected:
        raise ValueError('shard saved shape/axis metadata mismatch')
    if step is not None:
        _validate_evidence(json.loads((directory / 'validation.json').read_text()), prepared, step, rows)
    return arrays


def _batch_rows(batch, by_id):
    return [by_id[i] for i in batch['example_ids']]


def _batch_directory(directory, batch):
    return directory / 'batches' / f"batch-{batch['index']:06d}"


def _scan(directory, execution, prepared, rows, *, require_complete=False):
    """Scan expected memberships, never glob/count arbitrary or unmarked files."""
    by_id = {r['id']: r for r in rows}
    completed, totals, files = set(), {}, []
    hidden = None
    for batch in execution['batches']:
        batch_rows = _batch_rows(batch, by_id)
        folder = _batch_directory(directory, batch)
        base = None
        if (folder / 'baseline/complete.json').exists():
            base = _read_shard(folder / 'baseline', _identity(execution, batch), batch_rows, prepared)
            width = base['block_base_sum'].shape[-1]
            if hidden is not None and width != hidden:
                raise ValueError('baseline hidden axis changed across batches')
            hidden = width
            for k, v in base.items():
                name = 'baseline_' + k
                totals[name] = totals.get(name, np.zeros(v.shape[1:], dtype=np.float64)) + v.sum(axis=0)
            files.extend(folder / 'baseline' / p for p in ['complete.json', 'metadata.json', 'summaries.npz'])
        for step in STEPS:
            shard = folder / 'checkpoints' / f'step-{step}'
            if not (shard / 'complete.json').exists():
                if require_complete:
                    raise ValueError('measurement missing a required completed combination')
                continue
            if base is None:
                raise ValueError('marked checkpoint has no verified baseline')
            identity = _identity(execution, batch, step=step, baseline=file_hash(folder / 'baseline/complete.json'))
            payload = _read_shard(shard, identity, batch_rows, prepared, baseline=base, step=step)
            completed.add((batch['index'], step))
            for k, v in payload.items():
                name = f'step{step}_' + k
                totals[name] = totals.get(name, np.zeros(v.shape[1:], dtype=np.float64)) + v.sum(axis=0)
            files.extend(shard / p for p in ['complete.json', 'metadata.json', 'summaries.npz', 'validation.json'])
    return completed, totals, files


def verify_measurement(config_path, output_root):
    """CPU-only strict completion verification, for reuse and subsequent reporting."""
    root = Path(output_root).resolve()
    execution, prepared, rows = load_execution(config_path, root)
    directory = root / execution['run_path']
    marker = json.loads((directory / 'complete.json').read_text())
    if marker['execution_sha256'] != _hash(execution):
        raise ValueError('measurement completion identity mismatch')
    completed, totals, files = _scan(directory, execution, prepared, rows, require_complete=True)
    required = {str(p.relative_to(directory)) for p in files} | {
        'config.json', 'meta/run_manifest.json', 'results/results.json', 'results/aggregate-sums.npz'}
    if set(marker['files']) != required:
        raise ValueError('measurement completion omits required evidence')
    for name, digest in marker['files'].items():
        if file_hash(_relative(directory / name, directory.resolve())) != digest:
            raise ValueError('measurement completion payload hash mismatch')
    with np.load(directory / 'results/aggregate-sums.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(totals) or any(not np.array_equal(saved[k], v) for k, v in totals.items()):
            raise ValueError('aggregate sums disagree with completed shards')
    result = json.loads((directory / 'results/results.json').read_text())
    if (result['completed_combinations'] != len(completed) or not result['measurement_complete']
            or not result['all_gates_passed'] or result['base_sha256'] != prepared['source_evidence']['base_sha256']):
        raise ValueError('measurement results/completion mismatch')
    return result


def _progress(directory, state, completed, total, error=None):
    _write_state(directory, state, len(completed), total, error)
    _write_json(directory / 'checkpoints/progress.json', {
        'completed': len(completed), 'total': total,
        'completed_units': [list(unit) for unit in sorted(completed)],
        'unit': 'batch-index/checkpoint-step',
        'source_of_truth': 'verified-completion-markers'})


def measure_activation(config_path, output_root, *, dependencies=None):
    root = Path(output_root).resolve()
    execution, prepared, rows = load_execution(config_path, root)
    directory = root / execution['run_path']
    deps = dependencies or HFActivationDependencies()
    total = len(execution['batches']) * len(STEPS)
    with run_lock(directory):
        engine, reference, completed = None, None, set()
        stage, current = 'verify-runtime', None
        try:
            if deps.runtime() != execution['runtime']:
                raise ValueError('production runtime mismatch; use a new reviewed execution plan')
            _save_frozen(directory / 'config.json', {'execution': execution, 'prepared': prepared})
            manifest = {'schema_version': 1, 'execution_sha256': _hash(execution),
                        'config': 'config.json', 'inputs': execution['prepared_path'],
                        'axes': AXES, 'views': VIEWS, 'projections': PROJECTIONS,
                        'decoding': None, 'scorer': None, 'execution': 'forward-only'}
            _save_frozen(directory / 'meta/run_manifest.json', manifest)
            if (directory / 'complete.json').exists():
                result = verify_measurement(config_path, root)
                completed = {(b['index'], s) for b in execution['batches'] for s in STEPS}
                _progress(directory, 'completed', completed, total)
                return result
            stage = 'verify-existing-shards'
            completed, _, _ = _scan(directory, execution, prepared, rows)
            _progress(directory, 'running', completed, total)
            by_id = {r['id']: r for r in rows}
            for batch in execution['batches']:
                pending = [step for step in STEPS if (batch['index'], step) not in completed]
                if not pending:
                    continue
                current = {'batch_index': batch['index'], 'step': None}
                batch_rows = _batch_rows(batch, by_id)
                folder = _batch_directory(directory, batch)
                if engine is None:
                    stage = 'load-model'
                    engine = deps.activation_engine(prepared, root)
                    if engine.base_hash() != prepared['source_evidence']['base_sha256']:
                        raise ValueError('production loaded base identity mismatch')
                stage = 'capture-reference'
                reference = engine.capture_reference(batch_rows)
                base = engine.summarize_reference(reference)
                _validate_arrays(base, batch_rows)
                baseline_dir = folder / 'baseline'
                if (baseline_dir / 'complete.json').exists():
                    saved = _read_shard(baseline_dir, _identity(execution, batch), batch_rows, prepared)
                    if any(not np.array_equal(saved[k], base[k]) for k in BASE_KEYS):
                        raise ValueError('recreated reference disagrees with saved baseline summaries')
                else:
                    _commit_shard(baseline_dir, _identity(execution, batch), base)
                    _read_shard(baseline_dir, _identity(execution, batch), batch_rows, prepared)
                for step in pending:
                    current['step'] = step
                    stage = 'measure-checkpoint'
                    source = prepared['source_evidence']['checkpoints'][str(step)]
                    checkpoint = root / source['path']
                    if (file_hash(checkpoint / 'complete.json') != source['complete_sha256']
                            or file_hash(checkpoint / 'adapter_model.safetensors') != source['adapter_sha256']):
                        raise ValueError('source checkpoint changed during production')
                    measured = engine.measure(checkpoint, reference, step=step)
                    _validate_evidence(measured['validation'], prepared, step, batch_rows)
                    if set(measured['arrays']) != BASE_KEYS | CHECKPOINT_KEYS or any(
                            not np.array_equal(base[k], measured['arrays'][k]) for k in BASE_KEYS):
                        raise ValueError('checkpoint reference/count summaries disagree')
                    payload = {k: measured['arrays'][k] for k in CHECKPOINT_KEYS}
                    _validate_arrays(payload, batch_rows, baseline=base, step=step)
                    shard = folder / 'checkpoints' / f'step-{step}'
                    identity = _identity(execution, batch, step=step, baseline=file_hash(baseline_dir / 'complete.json'))
                    stage = 'commit-checkpoint'
                    _commit_shard(shard, identity, payload, evidence=measured['validation'])
                    _read_shard(shard, identity, batch_rows, prepared, baseline=base, step=step)
                    completed.add((batch['index'], step))
                    _progress(directory, 'running', completed, total)
                    print(f"activation measurement: {len(completed)}/{total}; batch {batch['index']}; step {step}; examples {len(batch_rows)}", flush=True)
                    del measured, payload
                stage = 'release-reference'
                try:
                    engine.release_reference(reference)
                finally:
                    reference = None
            stage = 'final-frozen-base-check'
            if engine is not None and engine.base_hash() != prepared['source_evidence']['base_sha256']:
                raise ValueError('production frozen base changed')
            stage = 'aggregate-verified-shards'
            completed, totals, files = _scan(directory, execution, prepared, rows, require_complete=True)
            _save_arrays(directory / 'results/aggregate-sums.npz', totals)
            result = {'schema_version': 1, 'mode': 'activation-measurement',
                      'measurement_complete': True, 'reported': False, 'all_gates_passed': True,
                      'execution_sha256': _hash(execution), 'example_count': len(rows),
                      'example_ids': [r['id'] for r in rows], 'corpus_counts': dict(Counter(r['corpus'] for r in rows)),
                      'completed_combinations': len(completed), 'batch_count': len(execution['batches']),
                      'checkpoint_steps': STEPS, 'views': VIEWS,
                      'base_sha256': prepared['source_evidence']['base_sha256'],
                      'aggregate_arrays': 'results/aggregate-sums.npz',
                      'summary_scope': 'sufficient sums; interpretation, weighting and intervals belong to CPU report'}
            _write_json(directory / 'results/results.json', result)
            files.extend(directory / p for p in ['config.json', 'meta/run_manifest.json',
                                                 'results/results.json', 'results/aggregate-sums.npz'])
            _write_json(directory / 'complete.json', {'execution_sha256': _hash(execution),
                'files': {str(p.relative_to(directory)): file_hash(p) for p in files}})
            result = verify_measurement(config_path, root)
            _progress(directory, 'completed', completed, total)
            return result
        except BaseException as exc:
            _progress(directory, 'failed', completed, total, str(exc))
            log = directory / 'logs/errors.jsonl'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                stream.write(json.dumps({'stage': stage, 'unit': current, 'error': str(exc),
                    'type': type(exc).__name__, 'diagnostics': getattr(exc, 'diagnostics', {})}) + '\n')
            raise
        finally:
            try:
                if engine is not None and reference is not None:
                    engine.release_reference(reference)
            finally:
                reference = None
                if engine is not None:
                    engine.close()
                engine = None


def monitor_activation_process(process, run_directory, *, last_snapshot=None):
    """Observe a Popen child; mutable progress never proves scientific completion."""
    returncode = process.poll()
    state = 'running' if returncode is None else ('exited-successfully' if returncode == 0 else 'exited-with-error')
    def valid(snapshot):
        return (isinstance(snapshot, dict) and snapshot.get('state') in ['running', 'failed', 'completed', 'paused']
                and type(snapshot.get('completed')) is int and type(snapshot.get('total')) is int
                and 0 <= snapshot['completed'] <= snapshot['total'])
    snapshot, error, current = None, None, False
    try:
        snapshot = json.loads((Path(run_directory) / 'meta/status.json').read_text())
        if not valid(snapshot):
            raise ValueError('invalid progress snapshot')
        current = True
    except (OSError, ValueError) as exc:
        error = str(exc)
        snapshot = dict(last_snapshot) if valid(last_snapshot) else None
    return {'process_state': state, 'returncode': returncode, 'snapshot': snapshot,
            'snapshot_is_current': current, 'status_error': error, 'completion_verified': False}
