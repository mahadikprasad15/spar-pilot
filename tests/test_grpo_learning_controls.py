"""Frozen real-model acceptance gates; no model downloads."""
from test_grpo_algorithm import torch_stack

def test_two_seed_real_model_positive_and_negative_controls(tmp_path):
    torch = torch_stack()
    from pilot_eval.grpo_controls import run_controls
    torch.set_num_threads(1)
    original = torch.random.get_rng_state().clone()
    result = run_controls(tmp_path, 'controls')
    assert result['passed'] is True
    assert torch.equal(original, torch.random.get_rng_state())
    assert {(r['seed'],r['sign']) for r in result['runs']} == {(42,1),(42,-1),(43,1),(43,-1)}
    assert all(r['signed_change'] >= .2 for r in result['runs'])
    assert all(r['comparison_baseline'] == 'pretraining_exact_expected_reward' for r in result['runs'])
    assert all(abs(r['baseline_reward'] - .5) < 1e-6 for r in result['runs'])
    for run in result['runs']:
        assert run['frozen_base_unchanged'] is True
    assert run_controls(tmp_path, 'controls') == result


def test_control_fixture_has_balanced_start_and_readout_headroom():
    torch = torch_stack()
    import json
    from pilot_eval.grpo_controls import TASK_PATH, tiny_model
    task = json.loads(TASK_PATH.read_text())
    for seed in task['seeds']:
        model, _ = tiny_model(task, seed)
        model.eval()
        with torch.no_grad():
            probability = model(input_ids=torch.tensor([task['prompt_tokens']])).logits[0, -1].softmax(-1)[1]
        assert abs(float(probability) - .5) < 1e-6
        base = model.get_base_model()
        direction = base.lm_head.weight[1] - base.lm_head.weight[0]
        assert abs(float(direction.norm()) - 1) < 1e-6
        assert base.lm_head.weight.requires_grad is False


def test_learning_gate_uses_untouched_baseline():
    import json
    from pilot_eval.grpo_controls import TASK_PATH
    criterion = json.loads(TASK_PATH.read_text())['criterion']
    assert criterion['baseline'] == 'pretraining_exact_expected_reward'
    assert criterion['late_steps'] == [16, 20]
    assert criterion['minimum_change'] == .2
