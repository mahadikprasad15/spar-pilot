import json
from pathlib import Path

import pytest

from test_sft_prepare import TrainingData, source_plan


class Engine:
    calls = []
    interrupt = False

    def preflight(self, directory):
        return {'finite_loss': True, 'finite_gradients': True, 'zero_write': True,
                'target_count': 196, 'base_sha256': 'b' * 64, 'peak_memory_bytes': 100}

    def train(self, directory, resume):
        from pilot_eval.training import seal_checkpoint
        self.calls.append(resume)
        start = int(resume.name.split('-')[-1]) if resume else 0
        for step in [0, 8, 16, 32, 64]:
            if step < start:
                continue
            cp = directory / 'checkpoints' / f'checkpoint-{step}'
            cp.mkdir(parents=True, exist_ok=True)
            for file in ['adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                         'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
                (cp / file).write_text(json.dumps({'global_step': step}))
            seal_checkpoint(cp, step)
            if self.interrupt and step == 8:
                raise RuntimeError('simulated interruption')
        return {'successful_steps': 64, 'base_sha256_before': 'b' * 64,
                'base_sha256_after': 'b' * 64, 'example_exposures': 512}


class TrainingBoundary(TrainingData):
    engine = Engine()

    def training_engine(self, config, rows):
        return self.engine


def test_public_training_recovers_from_sealed_checkpoint_and_skips_completed(tmp_path):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'sft', dependencies=TrainingData())
    deps = TrainingBoundary()
    deps.engine = Engine()
    deps.engine.calls = []
    preflight = run_sft(path, tmp_path, preflight_only=True, dependencies=deps)
    assert preflight['zero_write']
    deps.engine.interrupt = True
    with pytest.raises(RuntimeError, match='interruption'):
        run_sft(path, tmp_path, dependencies=deps)
    directory = next(tmp_path.glob('runs/pilot-2/**/sft/config.json')).parent
    assert json.loads((directory / 'meta/status.json').read_text())['state'] == 'failed'
    deps.engine.interrupt = False
    result = run_sft(path, tmp_path, dependencies=deps)
    assert deps.engine.calls[-1].name == 'checkpoint-8'
    assert result['successful_steps'] == 64
    count = len(deps.engine.calls)
    assert run_sft(path, tmp_path, dependencies=deps) == result
    assert len(deps.engine.calls) == count
    # Completed runs must still verify their checkpoint contents.
    (directory / 'checkpoints/checkpoint-64/optimizer.pt').write_text('corrupt')
    with pytest.raises(ValueError, match='checkpoint'):
        run_sft(path, tmp_path, dependencies=deps)


def test_model_loading_failure_is_saved_without_masking_original_error(tmp_path):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft, training_directory
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'failed', dependencies=TrainingData())

    class FailingModel(TrainingBoundary):
        def training_engine(self, config, rows):
            raise RuntimeError('CUDA out of memory during model load')

    with pytest.raises(RuntimeError, match='model load'):
        run_sft(path, tmp_path, dependencies=FailingModel())
    directory = training_directory(tmp_path, json.loads(path.read_text()))
    status = json.loads((directory / 'meta/status.json').read_text())
    assert status['state'] == 'failed'
    assert 'model load' in status['error']
    assert 'CUDA out of memory' in (directory / 'logs/errors.jsonl').read_text()


def test_checkpoint_manifest_cannot_omit_required_resume_state(tmp_path):
    from pilot_eval.training import verified_checkpoints
    Engine().train(tmp_path, None)
    checkpoint = tmp_path / 'checkpoints/checkpoint-8'
    manifest_path = checkpoint / 'complete.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['files']['optimizer.pt']
    manifest_path.write_text(json.dumps(manifest))
    (checkpoint / 'optimizer.pt').unlink()
    with pytest.raises(ValueError, match='required state'):
        verified_checkpoints(tmp_path)


def test_incompatible_invocation_preserves_completed_run(tmp_path):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft, training_directory
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'sft', dependencies=TrainingData())
    deps = TrainingBoundary()
    deps.engine = Engine()
    run_sft(path, tmp_path, dependencies=deps)
    config = json.loads(path.read_text())
    directory = training_directory(tmp_path, config)
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    config['source_config']['batch_size'] = 2
    changed = path.parent / 'changed.config.json'
    changed.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen artifact mismatch'):
        run_sft(changed, tmp_path, dependencies=deps)
    assert all(p.read_bytes() == content for p, content in before.items())
