import json

import pytest

from test_sft_prepare import TrainingData, source_plan
from test_sft_training import TrainingBoundary, Engine


def completed_training(root):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft
    path = prepare_sft(source_plan(root), root, 'sft', dependencies=TrainingData())
    deps = TrainingBoundary()
    deps.engine = Engine()
    run_sft(path, root, dependencies=deps)
    return path, deps


def test_matched_fp32_evaluation_preserves_inputs_and_reuses_completed_work(tmp_path):
    from pilot_eval.sft_evaluation import evaluate_sft
    path, deps = completed_training(tmp_path)
    baseline = evaluate_sft(path, tmp_path, 'baseline', dependencies=deps)
    zero = evaluate_sft(path, tmp_path, '0', dependencies=deps)
    assert zero['zero_checkpoint_matches_baseline'] is True
    assert baseline['strict_accuracy'] == 1.0
    assert baseline['flexible_v2_accuracy'] == 1.0
    assert evaluate_sft(path, tmp_path, '0', dependencies=deps) == zero
    source = json.loads(path.read_text())['source_config']
    assert source['dtype'] == 'bfloat16'
    configs = list(tmp_path.glob('plans/sft/evaluations/*.config.json'))
    assert len(configs) == 2
    assert all(json.loads(p.read_text())['dtype'] == 'float32' for p in configs)
    assert all(json.loads(p.read_text())['items_sha256'] == source['items_sha256'] for p in configs)


def test_checkpoint_zero_mismatch_blocks_trajectory(tmp_path):
    from pilot_eval.sft_evaluation import evaluate_sft
    path, deps = completed_training(tmp_path)
    evaluate_sft(path, tmp_path, 'baseline', dependencies=deps)

    class Changed(TrainingBoundary):
        def load_backend(self, config):
            class Backend:
                def generate_batch(self, prompts, decoding):
                    return [dict(text='#### 3', token_count=2, stop_reason='eos') for p in prompts]
            return Backend()

    with pytest.raises(ValueError, match='checkpoint zero'):
        evaluate_sft(path, tmp_path, '0', dependencies=Changed())
