"""Behavioral tests at the agreed summary/engine/workflow interfaces."""

import math

import numpy as np
import pytest


def test_summary_contract_preserves_cancellation_weighting_and_undefined_ratios():
    from pilot_eval.activation_math import block_summary, module_summary, derive_measurements
    # Example one: two opposite writes; example two: one write orthogonal to them.
    base = np.array([[[3., 0.], [0., 4.]], [[0., 2.], [999., 999.]]], dtype=np.float32)
    delta = np.array([[[1., 0.], [-1., 0.]], [[0., 2.], [999., 999.]]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])
    summary = block_summary(base, delta, mask)
    token = derive_measurements(summary, 'token')
    equal = derive_measurements(summary, 'example')
    assert token['mean_delta'] == pytest.approx([0, 2 / 3])
    assert token['mean_base_norm'] == pytest.approx(3)
    assert token['relative_write'] == pytest.approx(2 / 9)
    assert token['mean_delta_norm'] == pytest.approx(4 / 3)
    assert equal['mean_delta'] == pytest.approx([0, 1])
    assert equal['relative_write'] == pytest.approx(1 / 2.75)
    # Individual zero denominator stays in the ratio of sums, not per-token ratios.
    ordinary = np.array([[[0.], [4.]], [[2.], [999.]]], dtype=np.float32)
    direct = np.array([[[2.], [2.]], [[2.], [999.]]], dtype=np.float32)
    local = module_summary(ordinary, direct, mask)
    assert derive_measurements(local, 'token')['relative_write'] == pytest.approx(1)
    assert derive_measurements(local, 'token')['mean_token_ratio'] == pytest.approx(.75)
    assert derive_measurements(local, 'token')['defined_token_count'] == 2
    assert derive_measurements(local, 'example')['mean_token_ratio'] == pytest.approx(.75)
    zero = module_summary(np.zeros_like(ordinary), direct, mask)
    assert derive_measurements(zero, 'token')['relative_write'] is None
    assert derive_measurements(zero, 'token')['mean_token_ratio'] is None
    assert derive_measurements(zero, 'token')['undefined_token_count'] == 3


def test_real_qwen_engine_validates_zero_switching_and_all_projection_hooks(tmp_path):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine
    from pilot_eval.sft_backend import frozen_weight_hash
    from pilot_eval.activation_prepare import PROJECTIONS
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=4,
        intermediate_size=8, num_hidden_layers=2, num_attention_heads=1,
        num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0,
        attn_implementation='eager')), LoraConfig(r=1, lora_alpha=2,
            lora_dropout=0, target_modules=PROJECTIONS, task_type='CAUSAL_LM'))
    zero = tmp_path / 'zero'
    model.save_pretrained(zero)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if 'lora_B' in name:
                value.fill_(.1)
    trained = tmp_path / 'trained'
    model.save_pretrained(trained)
    rows = [dict(id='one', input_ids=[1, 2, 3], attention_mask=[1, 1, 1],
                 masks={'question': [True, False, False], 'solution': [False, True, True], 'user': [False] * 3}),
            dict(id='two', input_ids=[4, 5], attention_mask=[1, 1],
                 masks={'question': [True, False], 'solution': [False, True], 'user': [False] * 2})]
    initial_hash = frozen_weight_hash(model)
    engine = ActivationEngine(model=model, expected_layers=2, expected_alpha=2,
                              expected_base_hash=initial_hash, validation_limit=None)
    reference = engine.capture_reference(rows)
    zero_result = engine.measure(zero, reference, step=0)
    assert zero_result['validation']['exact_zero']
    assert zero_result['validation']['module_count'] == 14
    assert all(np.count_nonzero(v) == 0 for k, v in zero_result['arrays'].items() if k.endswith('delta_sum') or k.endswith('delta_norm_sum'))
    result = engine.measure(trained, reference, step=8)
    assert result['validation']['reference_invariant']
    assert result['validation']['rank1_passed']
    assert np.count_nonzero(result['arrays']['block_delta_sum']) > 0
    assert result['arrays']['block_delta_sum'].shape == (2, 3, 2, 4)
    assert result['arrays']['module_count'].shape == (2, 3, 2, 7)
    assert result['arrays']['block_count'][:, 1, 0].tolist() == [2, 1]
    assert engine.base_hash() == initial_hash
    engine.close()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
