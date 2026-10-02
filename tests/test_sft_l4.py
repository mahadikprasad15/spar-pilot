import json
import pytest
from test_sft_prepare import TrainingData, source_plan


def test_l4_plan_freezes_hardware_batch_and_preserves_t4_inputs(tmp_path):
    from pilot_eval.sft import prepare_sft, load_sft
    from pilot_eval.sft_evaluation import evaluation_config
    from pilot_eval.training import training_directory
    source = source_plan(tmp_path)
    t4 = prepare_sft(source, tmp_path, 't4', dependencies=TrainingData())
    original = t4.read_bytes()
    l4 = prepare_sft(source, tmp_path, 'l4', hardware='L4', evaluation_batch_size=2,
                     dependencies=TrainingData())
    config, rows, items = load_sft(l4, tmp_path)
    assert config['hardware'] == 'L4'
    assert config['evaluation_batch_size'] == 2
    old = json.loads(original)
    assert config['training_indices'] == old['training_indices']
    assert config['evaluation_items_sha256'] == old['evaluation_items_sha256']
    assert t4.read_bytes() == original
    directory = training_directory(tmp_path, config)
    (directory / 'results').mkdir(parents=True)
    (directory / 'results/preflight.json').write_text('{}')
    _, evaluation = evaluation_config(config, tmp_path, 'baseline')
    assert evaluation['hardware'] == 'L4'
    assert evaluation['batch_size'] == 2
    with pytest.raises(ValueError, match='variant|mismatch'):
        prepare_sft(source, tmp_path, 'l4', hardware='T4', dependencies=TrainingData())


def test_runtime_requires_recorded_l4_and_does_not_accept_old_t4_plan(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from pilot_eval.sft import prepare_sft, load_sft
    from pilot_eval.sft_backend import HFTrainingEngine, PINS
    import importlib.metadata
    source = source_plan(tmp_path)
    path = prepare_sft(source, tmp_path, 'l4', hardware='L4', evaluation_batch_size=2,
                       dependencies=TrainingData())
    config, _, _ = load_sft(path, tmp_path)
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: PINS.get(name) or original(name))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'NVIDIA L4')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda i: (8, 9))
    assert HFTrainingEngine.runtime(config)['compute_capability'] == [8, 9]
    t4 = prepare_sft(source, tmp_path, 't4', dependencies=TrainingData())
    with pytest.raises(ValueError, match='T4'):
        HFTrainingEngine.runtime(load_sft(t4, tmp_path)[0])


def test_profiled_larger_batch_can_be_frozen_for_new_l4_plan(tmp_path):
    from pilot_eval.sft import prepare_sft, load_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'l4-batch8', hardware='L4',
                       evaluation_batch_size=8, dependencies=TrainingData())
    assert load_sft(path, tmp_path)[0]['evaluation_batch_size'] == 8
