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
