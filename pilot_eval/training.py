"""Durable training orchestration and verified checkpoint resume."""

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import load_sft
from pilot_eval.workflow import HFDependencies, _save_frozen

CHECKPOINT_FILES = {'adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_directory(root, config):
    return Path(root).resolve() / 'runs' / 'pilot-2' / config['model'].replace('/', '--') / \
        'gsm8k' / 'train-512-seed-42' / 'rank1-float32' / config['run_id']


def seal_checkpoint(directory, step):
    """Publish completeness only after all required state files exist."""
    directory = Path(directory)
    if any(not (directory / name).is_file() for name in CHECKPOINT_FILES):
        raise ValueError('incomplete checkpoint state')
    files = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'complete.json' and not p.name.endswith('.tmp')}
    _write_json(directory / 'complete.json', dict(step=step, files=files))


def verified_checkpoints(directory):
    found = {}
    for path in (Path(directory) / 'checkpoints').glob('checkpoint-*'):
        if not (path / 'complete.json').exists():
            continue
        manifest = json.loads((path / 'complete.json').read_text())
        step = int(path.name.split('-')[-1])
        if manifest['step'] != step:
            raise ValueError('checkpoint step mismatch')
        if not CHECKPOINT_FILES.issubset(manifest['files']):
            raise ValueError('checkpoint manifest omits required state')
        for relative, digest in manifest['files'].items():
            file = (path / relative).resolve()
            if not file.is_relative_to(path.resolve()) or not file.is_file() or file_hash(file) != digest:
                raise ValueError('checkpoint content hash mismatch')
        state = json.loads((path / 'trainer_state.json').read_text())
        if state.get('global_step') != step:
            raise ValueError('checkpoint trainer step mismatch')
        found[step] = path
    return found


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process is already using this run') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class TrainingDependencies(HFDependencies):
    def runtime(self, config):
        from pilot_eval.sft_backend import HFTrainingEngine
        return HFTrainingEngine.runtime(config)

    def training_engine(self, config, rows):
        from pilot_eval.sft_backend import HFTrainingEngine
        return HFTrainingEngine(config, rows)


def run_sft(config_path, output_root, *, preflight_only=False, dependencies=None):
    config, rows, _ = load_sft(config_path, output_root)
    directory = training_directory(output_root, config)
    deps = dependencies or TrainingDependencies()
    with run_lock(directory):
        engine = None
        checkpoints = {}
        try:
            _save_frozen(directory / 'config.json', config)
            runtime = deps.runtime(config)
            _save_frozen(directory / 'meta/runtime.json', runtime)
            _save_frozen(directory / 'inputs/training.json', rows)
            _save_frozen(directory / 'meta/run_manifest.json', dict(
                config=config, runtime=runtime, results='results/results.json',
                preflight='results/preflight.json', checkpoints='checkpoints'))
            checkpoints = verified_checkpoints(directory)
            result_path = directory / 'results/results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if sorted(checkpoints) != config['checkpoint_steps']:
                    raise ValueError('completed run lacks verified checkpoints')
                return result
            preflight_path = directory / 'results/preflight.json'
            if preflight_only and preflight_path.exists():
                return json.loads(preflight_path.read_text())
            engine = deps.training_engine(config, rows)
            if not preflight_path.exists():
                _write_state(directory, 'preflight', 0, 64)
                evidence = engine.preflight(directory)
                if (not all(evidence.get(k) is True for k in ['finite_loss', 'finite_gradients', 'zero_write'])
                        or evidence.get('target_count') != 196):
                    raise ValueError('preflight integrity check failed')
                _write_json(preflight_path, evidence)
            if preflight_only:
                _write_state(directory, 'preflight-completed', 0, 64)
                return json.loads(preflight_path.read_text())
            resume = checkpoints[max(checkpoints)] if checkpoints else None
            _write_state(directory, 'running', max(checkpoints, default=0), 64)
            result = engine.train(directory, resume)
            if (result['successful_steps'] != 64 or result['example_exposures'] != 512
                    or result['base_sha256_before'] != result['base_sha256_after']
                    or result['base_sha256_before'] != json.loads(preflight_path.read_text())['base_sha256']):
                raise ValueError('training completion integrity check failed')
            if sorted(verified_checkpoints(directory)) != config['checkpoint_steps']:
                raise ValueError('training lacks complete checkpoint set')
            _write_json(result_path, result)
            _write_state(directory, 'completed', 64, 64)
            return result
        except Exception as exc:
            _write_state(directory, 'failed', max(checkpoints, default=0), 64, str(exc))
            error_path = directory / 'logs/errors.jsonl'
            with error_path.open('a') as stream:
                stream.write(json.dumps(dict(type=type(exc).__name__, error=str(exc))) + '\n')
            raise
        finally:
            if engine is not None and hasattr(engine, 'close'):
                engine.close()
