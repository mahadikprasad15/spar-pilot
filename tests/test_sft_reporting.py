import json

import pytest

from test_sft_evaluation import completed_training
from test_sft_training import TrainingBoundary


def test_paired_report_distinguishes_formatting_from_flexible_accuracy(tmp_path):
    from pilot_eval.sft_evaluation import evaluate_sft
    from pilot_eval.sft_reporting import compare_sft
    path, deps = completed_training(tmp_path)
    for checkpoint in ['baseline', '0']:
        evaluate_sft(path, tmp_path, checkpoint, dependencies=deps)

    class Prose(TrainingBoundary):
        def load_backend(self, config):
            class Backend:
                def generate_batch(self, prompts, decoding):
                    return [dict(text='Final answer: It takes 2 bolts.', token_count=1,
                                 stop_reason='eos') for p in prompts]
            return Backend()

    for checkpoint in ['8', '16', '32', '64']:
        evaluate_sft(path, tmp_path, checkpoint, dependencies=Prose())
    result = compare_sft(path, tmp_path)
    final = result['trajectory'][-1]
    assert final['step'] == 64
    assert final['strict_accuracy'] == 0
    assert final['flexible_v2_accuracy'] == 1
    assert final['strict_accuracy_change'] == -1
    assert final['flexible_v2_accuracy_change'] == 0
    assert final['mean_length_ratio'] == 0.5
    assert final['strict_change_95'] == [-1, -1]
    assert result['binary_collapse_threshold'] is None
    report = tmp_path / 'reports/sft-trajectory'
    assert len((report / 'results/paired-items.jsonl').read_text().splitlines()) == 750
    assert (report / 'results/trajectory.svg').exists()
    assert compare_sft(path, tmp_path) == result
    # Source corruption must invalidate report reuse, rather than trusting summaries.
    response = next(tmp_path.glob('runs/pilot-2-eval/**/sft-step-64/results/responses.jsonl'))
    response.write_text(response.read_text().replace('It takes 2', 'It takes 3', 1))
    with pytest.raises(ValueError):
        compare_sft(path, tmp_path)


def test_comparison_rejects_partial_checkpoint_evaluation(tmp_path):
    from pilot_eval.sft_evaluation import evaluate_sft
    from pilot_eval.sft_reporting import compare_sft
    path, deps = completed_training(tmp_path)
    evaluate_sft(path, tmp_path, 'baseline', dependencies=deps)
    evaluate_sft(path, tmp_path, '0', dependencies=deps)
    with pytest.raises(ValueError, match='incomplete'):
        compare_sft(path, tmp_path)
