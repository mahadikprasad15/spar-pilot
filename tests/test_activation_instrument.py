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


@pytest.mark.parametrize('layer_count', [2, 28])
def test_real_qwen_engine_validates_zero_switching_and_all_projection_hooks(tmp_path, layer_count, monkeypatch):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine
    from pilot_eval.sft_backend import frozen_weight_hash
    from pilot_eval.activation_prepare import PROJECTIONS
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=4,
        intermediate_size=8, num_hidden_layers=layer_count, num_attention_heads=1,
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
    # Once the model is supplied, switching local adapters must not perform Hub
    # config lookups to guess whether frozen embeddings should be exported.
    import peft.utils.save_and_load as peft_io
    model.peft_config['default'].base_model_name_or_path = 'offline-fixture-model'
    def forbid_hub(*args, **kwargs):
        raise AssertionError('local activation measurement attempted a Hub query')
    monkeypatch.setattr(peft_io, 'check_file_exists_on_hf_hub', forbid_hub)
    rows = [dict(id='one', input_ids=[1, 2, 3], attention_mask=[1, 1, 1],
                 masks={'question': [True, False, False], 'solution': [False, True, True], 'user': [False] * 3}),
            dict(id='two', input_ids=[4, 5], attention_mask=[1, 1],
                 masks={'question': [True, False], 'solution': [False, True], 'user': [False] * 2})]
    initial_hash = frozen_weight_hash(model)
    engine = ActivationEngine(model=model, expected_layers=layer_count, expected_alpha=2,
                              expected_base_hash=initial_hash, validation_limit=None)
    reference = engine.capture_reference(rows)
    # Prove the memory-saving transformer-body path hooks the same block outputs
    # as a complete causal-LM forward, without comparing final normalized states.
    causal_blocks = {}
    handles = [layer.register_forward_hook(lambda m, a, output, i=i:
               causal_blocks.__setitem__(i, (output[0] if isinstance(output, tuple) else output).detach().clone()))
               for i, layer in enumerate(model.get_base_model().model.layers)]
    with torch.no_grad(), model.disable_adapter():
        model(**reference['inputs'], use_cache=False)
    for handle in handles:
        handle.remove()
    assert all(torch.equal(value, causal_blocks[i]) for i, value in reference['blocks'].items())
    zero_result = engine.measure(zero, reference, step=0)
    assert zero_result['validation']['exact_zero']
    assert zero_result['validation']['module_count'] == 7 * layer_count
    assert all(np.count_nonzero(v) == 0 for k, v in zero_result['arrays'].items() if k.endswith('delta_sum') or k.endswith('delta_norm_sum'))
    result = engine.measure(trained, reference, step=8)
    assert result['validation']['reference_invariant']
    assert result['validation']['rank1_passed']
    assert np.count_nonzero(result['arrays']['block_delta_sum']) > 0
    assert result['arrays']['block_delta_sum'].shape == (2, 3, layer_count, 4)
    assert result['arrays']['module_count'].shape == (2, 3, layer_count, 7)
    assert result['arrays']['block_count'][:, 1, 0].tolist() == [2, 1]
    assert engine.base_hash() == initial_hash
    engine.close()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize('mistake', ['sign', 'scale', 'token', 'module'])
def test_negative_controls_fail_rank1_and_cleanup_restores_adapter_state(tmp_path, mistake):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine, InstrumentFailure
    from pilot_eval.activation_prepare import PROJECTIONS
    from pilot_eval.sft_backend import frozen_weight_hash
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=4,
        intermediate_size=8, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0,
        attn_implementation='eager')), LoraConfig(r=1, lora_alpha=2,
            lora_dropout=0, target_modules=PROJECTIONS, task_type='CAUSAL_LM'))
    original = {name: value.detach().clone() for name, value in model.named_parameters() if 'lora_' in name}
    with torch.no_grad():
        for name, value in model.named_parameters():
            if 'lora_B' in name:
                value.fill_(.1)
    checkpoint = tmp_path / 'checkpoint'
    model.save_pretrained(checkpoint)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in original:
                value.copy_(original[name])
    engine = ActivationEngine(model=model, expected_layers=1, expected_alpha=2,
        expected_base_hash=frozen_weight_hash(model), validation_limit=None)
    reference = engine.capture_reference([{'id': 'test', 'input_ids': [1, 2, 3],
        'attention_mask': [1, 1, 1], 'masks': {'question': [True] * 3,
                                           'solution': [False] * 3, 'user': [False] * 3}}])
    query = model.get_base_model().model.layers[0].self_attn.q_proj
    key = model.get_base_model().model.layers[0].self_attn.k_proj
    captured = {}
    def remember(module, args):
        captured['x'] = args[0]
    def corrupt(module, args, output):
        if mistake == 'sign':
            return -output
        if mistake == 'scale':
            return output * .5
        if mistake == 'token':
            return output.roll(1, dims=1)
        return torch.nn.functional.linear(torch.nn.functional.linear(captured['x'],
            key.lora_A['default'].weight), key.lora_B['default'].weight)
    handles = [query.register_forward_pre_hook(remember),
               query.lora_B['default'].register_forward_hook(corrupt)]
    try:
        with pytest.raises(InstrumentFailure, match='rank-1'):
            engine.measure(checkpoint, reference, step=8)
    finally:
        for handle in handles:
            handle.remove()
        engine.close()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert all(torch.equal(value, original[name]) for name, value in model.named_parameters() if name in original)


def test_step_zero_rejects_a_write_far_below_approximate_tolerance(tmp_path):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine, InstrumentFailure
    from pilot_eval.activation_prepare import PROJECTIONS
    from pilot_eval.sft_backend import frozen_weight_hash
    torch.set_num_threads(1)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=4,
        intermediate_size=8, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, pad_token_id=0, attn_implementation='eager')),
        LoraConfig(r=1, lora_alpha=1, lora_dropout=0, target_modules=PROJECTIONS, task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if 'lora_B' in name:
                value.fill_(1e-12)
    checkpoint = tmp_path / 'small-nonzero'
    model.save_pretrained(checkpoint)
    engine = ActivationEngine(model=model, expected_layers=1, expected_base_hash=frozen_weight_hash(model))
    reference = engine.capture_reference([{'id': 'tiny', 'input_ids': [1, 2], 'attention_mask': [1, 1],
        'masks': {'question': [True, True], 'solution': [False, False], 'user': [False, False]}}])
    try:
        with pytest.raises(InstrumentFailure, match='step 0'):
            engine.measure(checkpoint, reference, step=0)
    finally:
        engine.close()


def test_summary_rejects_nonfinite_values_and_reports_empty_views():
    from pilot_eval.activation_math import block_summary, derive_measurements
    zeros = np.zeros((2, 3, 4), dtype=np.float32)
    empty = block_summary(zeros, zeros, np.zeros((2, 3), dtype=bool))
    result = derive_measurements(empty, 'example')
    assert result['relative_write'] is None
    assert result['mean_delta'] is None
    assert result['empty_example_count'] == 2
    zeros[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        block_summary(zeros, zeros, np.ones((2, 3), dtype=bool))
