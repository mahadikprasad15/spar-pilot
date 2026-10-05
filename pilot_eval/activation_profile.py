"""Profile the validated activation workload and freeze reviewed execution plans."""
import gc
import json
import re
import time
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import load_prepared, STEPS, VIEWS, PROJECTIONS
from pilot_eval.activation_workflow import (
    HFActivationDependencies, validate_activation, _save_arrays, _verify_complete,
)
from pilot_eval.run import _write_json, _write_state
from pilot_eval.training import run_lock, file_hash
from pilot_eval.workflow import _hash, _save_frozen

BATCHES = [1, 2, 4, 8, 16]
AGREEMENT = {'atol': 1e-5, 'rtol': 1e-5}
ARRAYS = {'block_count', 'block_base_norm_sum', 'block_delta_norm_sum',
          'block_base_sum', 'block_delta_sum', 'module_count', 'module_base_norm_sum',
          'module_delta_norm_sum', 'module_ratio_sum', 'module_defined_count'}


class HFProfileDependencies(HFActivationDependencies):
    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def clock(self):
        return time.perf_counter()

    def reset_peak_memory(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        import torch
        return {'allocated_bytes': torch.cuda.max_memory_allocated(),
                'reserved_bytes': torch.cuda.max_memory_reserved()}

    def cleanup(self):
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def is_oom(self, exc):
        import torch
        return isinstance(exc, torch.cuda.OutOfMemoryError)


def _relative(path, root):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path outside output root')
    return path


def _seal(directory, identity, paths):
    _write_json(directory / 'complete.json', {'identity': identity,
                'files': {str(p.relative_to(directory)): file_hash(p) for p in paths}})


def _verified(directory, identity, required):
    marker = json.loads((directory / 'complete.json').read_text())
    if marker['identity'] != identity or not set(required).issubset(marker['files']):
        raise ValueError('profile identity or required evidence mismatch')
    for name, digest in marker['files'].items():
        path = _relative(directory / name, directory.resolve())
        if file_hash(path) != digest:
            raise ValueError('profile evidence hash mismatch')
    return json.loads((directory / 'results.json').read_text())


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if set(arrays) != ARRAYS or any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('incomplete or nonfinite profile summary evidence')
    return arrays


def _agreement(reference, candidate):
    failures = []
    for step in STEPS:
        for key in sorted(ARRAYS):
            a, b = reference[step][key], candidate[step][key]
            if a.shape != b.shape:
                raise ValueError('profile summary shape changed')
            exact = key.endswith('count')
            matching = np.array_equal(a, b) if exact else np.allclose(a, b, **AGREEMENT)
            # An almost-zero norm and zero norm have different undefined coverage.
            if key.endswith('base_norm_sum'):
                matching = matching and np.array_equal(a == 0, b == 0)
            if key == 'block_base_sum':
                matching = matching and np.array_equal(np.linalg.norm(a, axis=-1) == 0,
                                                       np.linalg.norm(b, axis=-1) == 0)
            if not matching:
                failures.append({'step': step, 'array': key,
                                 'max_absolute_error': float(np.max(np.abs(a - b)))})
    return {'passed': not failures, 'thresholds': AGREEMENT, 'differences': failures}


def _check_gate(evidence, config, step):
    if (not evidence['rank1_passed'] or not evidence['reference_invariant']
            or evidence['module_count'] != 196
            or evidence['base_sha256'] != config['source_evidence']['base_sha256']
            or (step == 0 and not evidence['exact_zero'])):
        raise ValueError('profile instrument acceptance gate failed')


def _workload(config, root, rows, engine, deps, *, timed):
    collected = {s: {key: [] for key in ARRAYS} for s in STEPS}
    evidence, reference_seconds, checkpoint_seconds = [], 0., 0.
    for batch in rows:
        reference = None
        try:
            deps.synchronize()
            started = deps.clock()
            reference = engine.capture_reference(batch)
            deps.synchronize()
            reference_seconds += deps.clock() - started
            for step in STEPS:
                source = config['source_evidence']['checkpoints'][str(step)]
                checkpoint = root / source['path']
                if (file_hash(checkpoint / 'adapter_model.safetensors') != source['adapter_sha256']
                        or file_hash(checkpoint / 'complete.json') != source['complete_sha256']):
                    raise ValueError('source checkpoint changed during profile')
                deps.synchronize()
                started = deps.clock()
                measured = engine.measure(checkpoint, reference, step=step)
                deps.synchronize()
                checkpoint_seconds += deps.clock() - started
                _check_gate(measured['validation'], config, step)
                if set(measured['arrays']) != ARRAYS:
                    raise ValueError('profile summary schema mismatch')
                for key, value in measured['arrays'].items():
                    if value.shape[0] != len(batch) or not np.isfinite(value).all():
                        raise ValueError('invalid profile example axis or nonfinite summary')
                    if timed:
                        collected[step][key].append(value)
                if timed:
                    evidence.append({'step': step, 'example_ids': [r['id'] for r in batch],
                                     'validation': measured['validation'],
                                     'timing': measured.get('timing')})
                del measured
        finally:
            del reference
    arrays = {s: {k: np.concatenate(v) for k, v in values.items()} for s, values in collected.items()} if timed else {}
    return arrays, evidence, reference_seconds, checkpoint_seconds


def profile_activation(config_path, output_root, *, dependencies=None):
    root = Path(output_root).resolve()
    config, rows = load_prepared(config_path, root)
    deps = dependencies or HFProfileDependencies()
    runtime = deps.runtime()
    diagnostic = validate_activation(config_path, root, dependencies=deps)
    if not diagnostic['all_gates_passed'] or diagnostic['identity']['runtime'] != runtime:
        raise ValueError('profile requires validated matching runtime')
    selected = []
    for corpus in ['gsm8k', 'fineweb']:
        selected.extend(sorted((r for r in rows if r['corpus'] == corpus),
                               key=lambda r: (-len(r['input_ids']), r['id']))[:8])
    if len(selected) != 16:
        raise ValueError('profile requires eight prepared examples per corpus')
    identity = {'prepared_sha256': _hash(config), 'runtime': runtime,
                'example_ids': [r['id'] for r in selected], 'inputs_sha256': _hash(selected),
                'selection': 'longest-eight-per-corpus-length-descending-id-tiebreak',
                'batch_sizes': BATCHES, 'agreement': AGREEMENT,
                'validation_identity': diagnostic['identity'], 'schema_version': 1,
                'views': VIEWS, 'projections': PROJECTIONS, 'checkpoint_steps': STEPS}
    directory = root / config['run_path'] / 'profile'
    required = ['config.json', 'results.json']
    for batch in BATCHES:
        required.extend([f'batch-{batch}/complete.json', f'batch-{batch}/results.json'])
    with run_lock(directory):
        if (directory / 'complete.json').exists():
            return _verified(directory, identity, required)
        _save_frozen(directory / 'config.json', identity)
        measurements, baseline, files = [], None, [directory / 'config.json']
        _write_state(directory, 'running', 0, len(BATCHES))
        try:
            for batch in BATCHES:
                target = directory / f'batch-{batch}'
                candidate_identity = {'profile_identity': _hash(identity), 'batch_size': batch}
                engine = None
                if (target / 'complete.json').exists():
                    result = _verified(target, candidate_identity, ['results.json'])
                    candidate = {s: _load_arrays(target / f'step-{s}.npz') for s in STEPS} if result['status'] != 'oom' else None
                else:
                    try:
                        deps.cleanup()
                        engine = deps.activation_engine(config, root)
                        if engine.base_hash() != config['source_evidence']['base_sha256']:
                            raise ValueError('profile loaded base identity mismatch')
                        # Same longest input batch, complete five-checkpoint workload, outside timing.
                        _workload(config, root, [selected[:batch]], engine, deps, timed=False)
                        deps.synchronize()
                        deps.reset_peak_memory()
                        chunks = [selected[i:i + batch] for i in range(0, len(selected), batch)]
                        candidate, evidence, reference_time, checkpoint_time = _workload(
                            config, root, chunks, engine, deps, timed=True)
                        elapsed = reference_time + checkpoint_time
                        if elapsed <= 0 or not np.isfinite(elapsed):
                            raise ValueError('profile timing must be finite and positive')
                        memory = deps.peak_memory()
                        agreement = _agreement(candidate if batch == 1 else baseline, candidate) if baseline is not None or batch == 1 else {'passed': False, 'reason': 'batch-1-reference-unavailable'}
                        result = {'batch_size': batch, 'status': 'passed' if agreement['passed'] else 'numerical-mismatch',
                                  'agreement': agreement, 'example_ids': identity['example_ids'],
                                  'seconds': elapsed, 'reference_seconds': reference_time,
                                  'validated_checkpoint_seconds': checkpoint_time,
                                  'input_tokens_per_second': sum(len(r['input_ids']) for r in selected) / elapsed,
                                  'examples_per_second': len(selected) / elapsed,
                                  'memory': memory,
                                  'timing_scope': 'one-reference-plus-five-validated-checkpoints-per-batch; excludes load, warmup and file writes',
                                  'validation': evidence}
                        target.mkdir(parents=True, exist_ok=True)
                        for step in STEPS:
                            _save_arrays(target / f'step-{step}.npz', candidate[step])
                    except Exception as exc:
                        if not deps.is_oom(exc):
                            raise
                        candidate = None
                        result = {'batch_size': batch, 'status': 'oom', 'error': str(exc),
                                  'example_ids': identity['example_ids']}
                    finally:
                        if engine is not None:
                            engine.close()
                        del engine
                        deps.cleanup()
                    _write_json(target / 'results.json', result)
                    payloads = [target / 'results.json'] + ([target / f'step-{s}.npz' for s in STEPS] if candidate is not None else [])
                    _seal(target, candidate_identity, payloads)
                if result['status'] == 'numerical-mismatch':
                    raise ValueError('profile summary agreement failed; inspect saved candidate evidence')
                if batch == 1 and candidate is not None:
                    baseline = candidate
                measurements.append(result)
                print(f"activation profile: batch {batch}: {result['status']}", flush=True)
                files.extend(p for p in target.iterdir() if p.suffix in ['.json', '.npz'])
                _write_state(directory, 'running', len(measurements), len(BATCHES))
            passed = [r for r in measurements if r['status'] == 'passed']
            result = {'mode': 'profile-only', 'scientific_run_complete': False, 'identity': identity,
                      'example_ids': identity['example_ids'], 'measurements': measurements,
                      'provisional_fastest_batch': min(passed, key=lambda r: r['seconds'])['batch_size'] if passed else None,
                      'profile_path': str(directory.relative_to(root))}
            _write_json(directory / 'results.json', result)
            _seal(directory, identity, files + [directory / 'results.json'])
            _write_state(directory, 'completed', len(BATCHES), len(BATCHES))
            return result
        except BaseException as exc:
            _write_state(directory, 'failed', len(measurements), len(BATCHES), str(exc))
            errors = directory / 'logs/errors.jsonl'
            errors.parent.mkdir(parents=True, exist_ok=True)
            with errors.open('a') as stream:
                stream.write(json.dumps({'error': str(exc), 'diagnostics': getattr(exc, 'diagnostics', {})}) + '\n')
            raise


def freeze_execution(config_path, output_root, *, profile, batch_size, name, review_notes):
    """CPU-only explicit review; production will verify the saved runtime again."""
    root = Path(output_root).resolve()
    config, rows = load_prepared(config_path, root)
    if not review_notes.strip() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('freeze requires review notes and a safe new execution name')
    directory = _relative(profile, root)
    identity = json.loads((directory / 'config.json').read_text())
    result = _verified(directory, identity, ['config.json', 'results.json'] + [
        f'batch-{b}/{p}' for b in BATCHES for p in ['complete.json', 'results.json']])
    if identity['prepared_sha256'] != _hash(config):
        raise ValueError('profile belongs to different prepared inputs')
    candidate = next((r for r in result['measurements'] if r['batch_size'] == batch_size), None)
    if candidate is None or candidate['status'] != 'passed':
        raise ValueError('only a reviewed passing profile candidate can be frozen')
    # Re-read numeric evidence instead of trusting the displayed recommendation.
    reference = {s: _load_arrays(directory / 'batch-1' / f'step-{s}.npz') for s in STEPS}
    arrays = {s: _load_arrays(directory / f'batch-{batch_size}' / f'step-{s}.npz') for s in STEPS}
    if not _agreement(reference, arrays)['passed']:
        raise ValueError('selected batch disagrees with batch 1')
    diagnostic_dir = root / config['run_path'] / 'validation' / ('diagnostic-' + _hash(identity['validation_identity']['example_ids'])[:12])
    diagnostic = _verify_complete(diagnostic_dir, identity['validation_identity'])
    if not diagnostic['all_gates_passed']:
        raise ValueError('instrument diagnostic not passed')
    execution = {'schema_version': 1, 'state': 'frozen', 'run_id': name,
                 'prepared_path': str(_relative(config_path, root).relative_to(root)),
                 'prepared_sha256': _hash(config), 'inputs_sha256': config['items_sha256'],
                 'profile_path': str(directory.relative_to(root)),
                 'profile_complete_sha256': file_hash(directory / 'complete.json'),
                 'validation_complete_sha256': file_hash(diagnostic_dir / 'complete.json'),
                 'runtime': identity['runtime'], 'batch_size': batch_size, 'agreement': AGREEMENT,
                 'review': {'notes': review_notes, 'batch_size': batch_size},
                 'checkpoint_steps': STEPS, 'views': VIEWS, 'projections': PROJECTIONS,
                 'run_path': str(Path(config['run_path']).parent / name),
                 'batches': [{'index': i // batch_size, 'example_ids': [r['id'] for r in rows[i:i + batch_size]],
                              'inputs_sha256': _hash(rows[i:i + batch_size])}
                             for i in range(0, len(rows), batch_size)]}
    manifest = root / 'plans' / name / 'activation.execution.json'
    with run_lock(manifest.parent):
        _save_frozen(manifest, execution)
    return manifest
