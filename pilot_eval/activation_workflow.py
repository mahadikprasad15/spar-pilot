"""Public diagnostic workflow: verified fixed inputs through durable summaries."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import load_prepared, STEPS, VIEWS, PROJECTIONS
from pilot_eval.run import _write_json, _write_state
from pilot_eval.training import run_lock, file_hash
from pilot_eval.workflow import _hash, _save_frozen


class HFActivationDependencies:
    def runtime(self):
        import torch
        from pilot_eval.sft_backend import PINS
        versions = {name: importlib.metadata.version(name) for name in PINS if name != 'trl'}
        if any(versions[name].split('+')[0] != PINS[name] for name in versions):
            raise ValueError(f'activation dependencies differ from validated pins: {versions}')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('real activation validation requires one visible CUDA GPU')
        return {'versions': versions, 'python': sys.version, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0),
                'compute_capability': list(torch.cuda.get_device_capability(0)),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    cwd=Path(__file__).resolve().parent, text=True).strip(),
                'precision': 'float32', 'autocast': False, 'tf32': False,
                'attention': 'eager', 'padding': 'right', 'position_ids': 'attention-cumsum-minus-one',
                'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8'}

    def activation_engine(self, config, root):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        from transformers import AutoModelForCausalLM, enable_full_determinism
        from peft import PeftModel
        from pilot_eval.activation_engine import ActivationEngine
        enable_full_determinism(config['seed'])
        model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['model_revision'],
            dtype=torch.float32, attn_implementation='eager', device_map={'': 0})
        checkpoint = Path(root) / config['source_evidence']['checkpoints']['0']['path']
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        model.config.use_cache = False
        return ActivationEngine(model=model, expected_base_hash=config['source_evidence']['base_sha256'])


def _save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite summary arrays')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
            raise ValueError('summary array read-back mismatch')
    os.replace(temporary, path)


def _verify_complete(directory, identity):
    marker = json.loads((directory / 'complete.json').read_text())
    if marker['identity'] != identity:
        raise ValueError('diagnostic identity/runtime mismatch; use a separately named plan')
    required = {'config.json', 'results/results.json'} | {
        f'checkpoints/step-{step}/{name}' for step in STEPS for name in ['summaries.npz', 'validation.json']}
    if not required.issubset(marker['files']):
        raise ValueError('diagnostic completion marker omits required evidence')
    for relative, digest in marker['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != digest:
            raise ValueError('diagnostic payload hash mismatch')
    return json.loads((directory / 'results/results.json').read_text())


def validate_activation(config_path, output_root, *, batch_size=2, dependencies=None):
    """Validate one fixed diagnostic batch, never claim full-cohort completion."""
    root = Path(output_root).resolve()
    config, rows = load_prepared(config_path, root)
    if batch_size < 1 or batch_size > len(rows):
        raise ValueError('invalid diagnostic batch size')
    selected = []
    for corpus in ['gsm8k', 'fineweb']:
        selected.append(max((row for row in rows if row['corpus'] == corpus), key=lambda row: len(row['input_ids'])))
    selected = selected[:batch_size]
    selected.extend(row for row in rows if row['id'] not in {r['id'] for r in selected})
    selected = selected[:batch_size]
    dependencies = dependencies or HFActivationDependencies()
    identity = {'prepared_sha256': _hash(config), 'example_ids': [row['id'] for row in selected],
                'inputs_sha256': _hash(selected), 'runtime': dependencies.runtime(),
                'mode': 'diagnostic-only', 'schema_version': 1,
                'axes': {'block_vectors': ['example', 'view', 'layer', 'hidden'],
                         'block_scalars': ['example', 'view', 'layer'],
                         'module_scalars': ['example', 'view', 'layer', 'projection']},
                'views': VIEWS, 'projections': PROJECTIONS, 'checkpoint_steps': STEPS}
    directory = root / config['run_path'] / 'validation' / ('diagnostic-' + _hash(identity['example_ids'])[:12])
    with run_lock(directory):
        if (directory / 'complete.json').exists():
            return _verify_complete(directory, identity)
        _save_frozen(directory / 'config.json', identity)
        engine = None
        completed, stage = 0, 'load-model'
        try:
            _write_state(directory, 'running', completed, 5)
            engine = dependencies.activation_engine(config, root)
            if engine.base_hash() != config['source_evidence']['base_sha256']:
                raise ValueError('loaded base identity mismatch')
            stage = 'capture-reference'
            reference = engine.capture_reference(selected)
            files = [directory / 'config.json']
            for step in STEPS:
                stage = f'checkpoint-{step}'
                source = config['source_evidence']['checkpoints'][str(step)]
                checkpoint = root / source['path']
                if (file_hash(checkpoint / 'adapter_model.safetensors') != source['adapter_sha256']
                        or file_hash(checkpoint / 'complete.json') != source['complete_sha256']):
                    raise ValueError('source checkpoint changed during diagnostic')
                print(f'activation validation: step {step}; examples {len(selected)}', flush=True)
                measured = engine.measure(checkpoint, reference, step=step)
                evidence = measured['validation']
                if (not evidence['rank1_passed'] or not evidence['reference_invariant']
                        or evidence['module_count'] != 196
                        or evidence['base_sha256'] != config['source_evidence']['base_sha256']
                        or (step == 0 and not evidence['exact_zero'])):
                    raise ValueError('instrument acceptance gate failed')
                payload = directory / f'checkpoints/step-{step}/summaries.npz'
                _save_arrays(payload, measured['arrays'])
                evidence_path = payload.parent / 'validation.json'
                _write_json(evidence_path, evidence)
                files.extend([payload, evidence_path])
                completed += 1
                _write_state(directory, 'running', completed, 5)
            if engine.base_hash() != config['source_evidence']['base_sha256']:
                raise ValueError('frozen base changed during diagnostic')
            result = {'mode': 'diagnostic-only', 'scientific_run_complete': False,
                      'checkpoint_steps': STEPS, 'example_ids': identity['example_ids'],
                      'base_sha256': engine.base_hash(), 'all_gates_passed': True,
                      'identity': identity}
            result_path = directory / 'results/results.json'
            _write_json(result_path, result)
            files.append(result_path)
            _write_json(directory / 'complete.json', {'identity': identity,
                'files': {str(path.relative_to(directory)): file_hash(path) for path in files}})
            _write_state(directory, 'completed', 5, 5)
            return result
        except BaseException as exc:
            _write_state(directory, 'failed', completed, 5, str(exc))
            errors = directory / 'logs/errors.jsonl'
            errors.parent.mkdir(parents=True, exist_ok=True)
            with errors.open('a') as stream:
                stream.write(json.dumps({'stage': stage, 'error': str(exc),
                    'diagnostics': getattr(exc, 'diagnostics', {})}) + '\n')
            raise
        finally:
            if engine is not None:
                engine.close()
