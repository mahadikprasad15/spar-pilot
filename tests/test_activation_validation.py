import json
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.activation_prepare import prepare_activation
from test_activation_prepare import ActivationData, completed_source


@pytest.mark.parametrize('model_pad', [None, 151643])
def test_hf_loader_propagates_pinned_tokenizer_padding_before_adapter_load(tmp_path, monkeypatch, model_pad):
    from types import SimpleNamespace
    pytest.importorskip('torch', exc_type=ImportError)
    import transformers
    import peft
    import pilot_eval.activation_engine as instrument
    from pilot_eval.activation_workflow import HFActivationDependencies
    config = {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'model_revision': 'm' * 40,
              'tokenizer_revision': 't' * 40, 'seed': 42,
              'source_evidence': {'base_sha256': 'b' * 64,
                                  'checkpoints': {'0': {'path': 'adapter-zero'}}}}
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=model_pad,
                                                   vocab_size=151936, use_cache=True))
    tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
    calls = []
    def load_tokenizer(name, **settings):
        calls.append((name, settings))
        return tokenizer
    def load_adapter(base, path, **settings):
        assert base.config.pad_token_id == 151643
        assert path == tmp_path / 'adapter-zero'
        return base
    monkeypatch.setattr(transformers, 'enable_full_determinism', lambda seed: None)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: model)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained', load_adapter)
    monkeypatch.setattr(instrument, 'ActivationEngine', lambda **settings: settings)
    result = HFActivationDependencies().activation_engine(config, tmp_path)
    assert result['model'].config.pad_token_id == 151643
    assert result['model'].config.use_cache is False
    assert result['expected_base_hash'] == 'b' * 64
    assert calls == [(config['model'], {'revision': config['tokenizer_revision']})]


@pytest.mark.parametrize('tokenizer_pad,model_pad,message', [
    (None, None, 'explicit valid padding'),
    (151643, 151645, 'padding token IDs disagree'),
])
def test_hf_loader_refuses_missing_or_conflicting_padding(tmp_path, monkeypatch, tokenizer_pad, model_pad, message):
    from types import SimpleNamespace
    pytest.importorskip('torch', exc_type=ImportError)
    import transformers
    import peft
    from pilot_eval.activation_workflow import HFActivationDependencies
    monkeypatch.setattr(transformers, 'enable_full_determinism', lambda seed: None)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k:
                        SimpleNamespace(pad_token_id=tokenizer_pad, eos_token_id=151645))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k:
                        SimpleNamespace(config=SimpleNamespace(pad_token_id=model_pad, vocab_size=151936)))
    def unexpected_adapter_load(*args, **kwargs):
        raise AssertionError('invalid padding must stop before adapter loading')
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained', unexpected_adapter_load)
    with pytest.raises(ValueError, match=message):
        HFActivationDependencies().activation_engine({'model': 'fixture', 'model_revision': 'pinned',
            'tokenizer_revision': 'pinned-tokenizer', 'seed': 42}, tmp_path)


class DiagnosticEngine:
    closed = False
    calls = []
    failure = None

    def capture_reference(self, rows):
        return rows

    def measure(self, checkpoint, reference, *, step):
        from pilot_eval.activation_engine import InstrumentFailure
        self.calls.append(step)
        if self.failure == step:
            raise InstrumentFailure('planted rank-1 failure', {'layer': 5, 'projection': 'v_proj'})
        arrays = {'block_delta_sum': np.full((len(reference), 3, 28, 4), step, dtype=np.float64),
                  'module_count': np.ones((len(reference), 3, 28, 7), dtype=np.float64)}
        return {'arrays': arrays, 'validation': {'exact_zero': step == 0, 'rank1_passed': True,
                'reference_invariant': True, 'module_count': 196, 'base_sha256': 'b' * 64}}

    def base_hash(self):
        return 'b' * 64

    def close(self):
        self.closed = True


class DiagnosticDependencies:
    def __init__(self, failure=None):
        self.engine = DiagnosticEngine()
        self.engine.calls = []
        self.engine.failure = failure

    def runtime(self):
        return {'device': 'controlled-CPU-fixture', 'precision': 'float32', 'tf32': False}

    def activation_engine(self, config, root):
        return self.engine


def prepared(root):
    return prepare_activation(completed_source(root), root, 'instrument', dependencies=ActivationData())


def test_public_diagnostic_saves_five_verified_summaries_and_reuses_completion(tmp_path):
    config = prepared(tmp_path)
    args = ['activation-validate', '--config', str(config), '--output-root', str(tmp_path), '--batch-size', '2']
    dependencies = DiagnosticDependencies()
    assert main(args, dependencies=dependencies) == 0
    assert dependencies.engine.calls == [0, 8, 16, 32, 64]
    assert dependencies.engine.closed
    run = tmp_path / json.loads(config.read_text())['run_path']
    diagnostic = next((run / 'validation').glob('diagnostic-*'))
    result = json.loads((diagnostic / 'results/results.json').read_text())
    assert result['mode'] == 'diagnostic-only'
    assert result['scientific_run_complete'] is False
    assert result['checkpoint_steps'] == [0, 8, 16, 32, 64]
    assert len(result['example_ids']) == 2
    for step in result['checkpoint_steps']:
        with np.load(diagnostic / f'checkpoints/step-{step}/summaries.npz', allow_pickle=False) as saved:
            assert saved['block_delta_sum'][0, 0, 0, 0] == step
    assert main(args, dependencies=dependencies) == 0
    assert dependencies.engine.calls == [0, 8, 16, 32, 64]
    payload = diagnostic / 'checkpoints/step-8/summaries.npz'
    payload.write_bytes(b'corrupt')
    assert main(args, dependencies=dependencies) == 1


def test_failed_instrument_persists_diagnostics_and_does_not_publish_success(tmp_path):
    config = prepared(tmp_path)
    dependencies = DiagnosticDependencies(failure=16)
    assert main(['activation-validate', '--config', str(config), '--output-root', str(tmp_path)],
                dependencies=dependencies) == 1
    run = tmp_path / json.loads(config.read_text())['run_path']
    diagnostic = next((run / 'validation').glob('diagnostic-*'))
    assert not (diagnostic / 'complete.json').exists()
    assert json.loads((diagnostic / 'meta/status.json').read_text())['state'] == 'failed'
    errors = [json.loads(line) for line in (diagnostic / 'logs/errors.jsonl').read_text().splitlines()]
    assert errors[-1]['diagnostics'] == {'layer': 5, 'projection': 'v_proj'}
    assert dependencies.engine.closed


def test_padding_failure_recovery_allows_new_runtime_to_validate_same_prepared_inputs(tmp_path, monkeypatch):
    from scripts.recover_pilot3_padding import main as recover
    config = prepared(tmp_path)
    config_before = config.read_bytes()
    inputs = tmp_path / json.loads(config_before)['items_path']
    inputs_before = inputs.read_bytes()
    dependencies = DiagnosticDependencies()
    dependencies.runtime = lambda: {'git_commit': 'old-code-pin'}
    def missing_padding(rows):
        raise ValueError('explicit padding token required')
    dependencies.engine.capture_reference = missing_padding
    args = ['activation-validate', '--config', str(config), '--output-root', str(tmp_path)]
    assert main(args, dependencies=dependencies) == 1
    run = tmp_path / json.loads(config_before)['run_path']
    directory = next((run / 'validation').glob('diagnostic-*'))
    failed_log = (directory / 'logs/errors.jsonl').read_bytes()
    monkeypatch.setattr('sys.argv', ['recover', '--config', str(config), '--output-root', str(tmp_path)])
    recover()
    archive = next((run / 'validation/failed-attempts').glob('diagnostic-*'))
    assert (archive / 'logs/errors.jsonl').read_bytes() == failed_log
    assert main(args, dependencies=DiagnosticDependencies()) == 0
    assert (directory / 'complete.json').is_file()
    assert config.read_bytes() == config_before
    assert inputs.read_bytes() == inputs_before
