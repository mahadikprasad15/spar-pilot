"""Ticket 03a collector on a tiny random Qwen2/PEFT model (CPU, no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import neuron_audit_collector as nc  # noqa: E402
from pilot_eval.neuron_audit_blocks import BlockStore, CorruptUnit  # noqa: E402
from pilot_eval.sft_backend import frozen_weight_hash  # noqa: E402

PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
TOL = dict(z_reconstruction_relative_to_layer_max=1e-6, identity_atol=1e-6, identity_rtol=1e-5)


def tiny(targets=PROJ, alpha=2, seed=0, randomize_b=True):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        pad_token_id=0, attn_implementation='eager'))
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=alpha, lora_dropout=0, target_modules=targets))
    if randomize_b:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(0, 0.5)
    return model.eval()


def rows():
    def row(i, n):
        ids = [(7 * i + k) % 31 + 1 for k in range(n)]
        content = list(range(3, n - 1))
        return dict(id=f'e{i}', input_ids=ids, masks=dict(response_content=content, reply_start=[2],
                                                          termination=[n - 2, n - 1]))
    out = [row(0, 9), row(1, 6), row(2, 11), row(3, 7), row(4, 8)]
    out[3]['masks']['termination'] = []  # truncated example: undefined termination means
    return out


@pytest.fixture
def adapted(tmp_path):
    model = tiny()
    model.save_pretrained(tmp_path / 'adapter')
    return model, tmp_path / 'adapter', nc.load_saved_factors(tmp_path / 'adapter', [0, 1], scale=2.0)


def test_base_collection_has_no_adapter_fields_and_disabled_routing(adapted):
    model, _, _ = adapted
    collector = nc.NeuronCollector(model, tolerances=TOL, expected_base_hash=frozen_weight_hash(model))
    out = collector.collect(rows()[:3], factors=None)
    assert {k[2] for k in out['arrays']} == {'mean_z', 'mean_abs_z'}
    assert out['tokens'] is None and out['routing'] == dict(lora_a_calls=0, lora_b_calls=0)
    assert out['arrays'][('response_content', 0, 'mean_z')].shape == (3, 12)
    assert {r['check'] for r in out['validation']} == {'z_reconstruction', 'reduction'}
    runtime = collector.runtime()
    assert set(runtime['fp32_precision'].values()) == {'ieee'} and len(runtime['fp32_precision']) == 6
    assert runtime['matmul_precision'] == 'highest' and runtime['torch_compile'] is False
    collector.close()
    assert torch.backends.cudnn.conv.fp32_precision == 'tf32'  # previous values restored


def test_adapted_collection_passes_independent_checks_and_stores_reads(adapted):
    model, _, factors = adapted
    out = nc.NeuronCollector(model, tolerances=TOL).collect(rows(), factors=factors)
    checks = {r['check'] for r in out['validation']}
    assert {'z_reconstruction', 'reduction', 'lora_a_read', 'adapter_branch'} <= checks
    assert all(r['status'] == 'pass' for r in out['validation'] if r.get('gate', True))
    assert out['routing'] == dict(lora_a_calls=2, lora_b_calls=2)
    tokens = out['tokens']
    union = sum(len(set(r['masks']['response_content']) | set(r['masks']['reply_start']) |
                    set(r['masks']['termination'])) for r in rows())
    assert tokens['ct'].shape == (2, union) and tokens['ct'].dtype == np.float32
    # mean |a_j z_j| == |a_j| * mean |z_j| for a fixed checkpoint
    a = factors['layers'][1]['a']
    assert np.allclose(out['arrays'][('response_content', 1, 'mean_abs_term')],
                       np.abs(a) * out['arrays'][('response_content', 1, 'mean_abs_z')], rtol=1e-5)
    # truncated example: termination undefined (NaN), never zero
    assert np.isnan(out['arrays'][('termination', 0, 'mean_z')][3]).all()
    assert out['counts']['termination'][3] == 0


def test_wrong_sign_and_wrong_layer_factors_fail_independent_read_check(adapted):
    model, _, factors = adapted
    flipped = dict(scale=2.0, layers={l: dict(a=-f['a'], b=f['b']) for l, f in factors['layers'].items()})
    with pytest.raises(nc.CollectorFailure, match='independent'):
        nc.NeuronCollector(model, tolerances=TOL).collect(rows(), factors=flipped)
    swapped = dict(scale=2.0, layers={0: factors['layers'][1], 1: factors['layers'][0]})
    with pytest.raises(nc.CollectorFailure, match='independent'):
        nc.NeuronCollector(model, tolerances=TOL).collect(rows(), factors=swapped)
    wrong_scale = dict(factors, scale=1.0)
    with pytest.raises(nc.CollectorFailure, match='independent'):
        nc.NeuronCollector(model, tolerances=TOL).collect(rows(), factors=wrong_scale)


def test_paired_inputs_down_proj_only_adapter_leaves_first_layer_input_unchanged():
    model = tiny(targets=['down_proj'])
    collector = nc.NeuronCollector(model, tolerances=TOL)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        factors = nc.load_saved_factors(d, [0, 1], scale=2.0)
    base = collector.collect(rows(), factors=None)
    arm = collector.collect(rows(), factors=factors)
    key0, key1 = ('response_content', 0, 'mean_abs_z'), ('response_content', 1, 'mean_abs_z')
    assert np.array_equal(base['arrays'][key0], arm['arrays'][key0])  # same z: d = 0 exactly
    assert not np.array_equal(base['arrays'][key1], arm['arrays'][key1])  # upstream write propagates


def test_padding_and_batch_composition_change_values_only_within_tolerance(adapted):
    model, _, factors = adapted
    collector = nc.NeuronCollector(model, tolerances=TOL)
    batch = collector.collect(rows(), factors=factors)
    single = collector.collect([rows()[2]], factors=factors, padded_width=20)
    key = ('response_content', 1, 'mean_z')
    assert np.allclose(batch['arrays'][key][2], single['arrays'][key][0], rtol=1e-5, atol=1e-6)


def test_empty_primary_mask_and_missing_tolerance_rejected(adapted):
    model, _, _ = adapted
    bad = rows()[:1]
    bad[0]['masks']['response_content'] = []
    with pytest.raises(nc.CollectorFailure, match='empty primary'):
        nc.NeuronCollector(model, tolerances=TOL).collect(bad)
    with pytest.raises(nc.CollectorFailure, match='tolerance'):
        nc.NeuronCollector(model, tolerances={})


def test_base_hash_mismatch_rejected(adapted):
    model, _, _ = adapted
    with pytest.raises(nc.CollectorFailure, match='hash'):
        nc.NeuronCollector(model, tolerances=TOL, expected_base_hash='0' * 64)


def test_live_adapter_matches_saved_and_detects_changes(adapted):
    model, path, _ = adapted
    ok = nc.check_live_adapter(model, path, expected_scale=2.0)
    assert ok['status'] == 'pass' and ok['key_count'] == 28
    assert nc.check_live_adapter(model, path, expected_scale=1.0)['status'] == 'fail'
    with torch.no_grad():
        model.base_model.model.model.layers[1].mlp.down_proj.lora_A['default'].weight[0, 0] += 1e-7
    changed = nc.check_live_adapter(model, path, expected_scale=2.0)
    assert changed['status'] == 'fail' and changed['mismatched_keys'] == [
        'base_model.model.model.layers.1.mlp.down_proj.lora_A.weight']


def test_live_adapter_detects_wrong_keys_and_disabled_routing(adapted, tmp_path):
    model, path, _ = adapted
    keys = nc.check_live_adapter(model, path, expected_scale=2.0, expected_keys=['x'])
    assert keys['status'] == 'fail' and 'inventory' in keys['reasons'][0]
    for module in model.modules():
        if hasattr(module, 'lora_A'):
            module._disable_adapters = True
            break
    assert nc.check_live_adapter(model, path, expected_scale=2.0)['status'] == 'fail'


def test_legitimate_zero_b_is_classified_not_failed(tmp_path):
    model = tiny(randomize_b=False)
    model.save_pretrained(tmp_path / 'zero')
    report = nc.check_live_adapter(model, tmp_path / 'zero', expected_scale=2.0)
    assert report['status'] == 'pass' and len(report['zero_b_modules']) == 14
    inputs, _ = nc.NeuronCollector(model, tolerances=TOL).batch(rows()[:2])
    effect = nc.adapter_effect(model, inputs)
    assert effect['classification'] == 'no-logit-change-unresolved'


def test_active_adapter_changes_logits_and_disabled_matches_independent_base(adapted):
    model, _, _ = adapted
    reference = tiny(randomize_b=False)  # same seed => same base weights, separately constructed
    inputs, _ = nc.NeuronCollector(model, tolerances=TOL).batch(rows()[:2])
    with torch.no_grad():
        ref_logits = reference(**inputs, use_cache=False).logits
    effect = nc.adapter_effect(model, inputs, reference_logits=ref_logits)
    assert effect['classification'] == 'changes-logits'
    assert effect['disabled_minus_reference_max_abs'] == 0.0
    assert effect['disabled_reference_tolerance'] is None  # unresolved in the candidate plan


def test_cohort_persists_paired_units_resumes_and_detects_corruption(adapted, tmp_path):
    model, _, factors = adapted
    collector = nc.NeuronCollector(model, tolerances=TOL)
    store = BlockStore(tmp_path / 'blocks')
    ident = dict(plan='fixture', sequences='s')
    base = nc.collect_cohort(collector, store, rows(), arm='untuned', batch_size=2, identity=ident)
    arm = nc.collect_cohort(collector, store, rows(), arm='gold', batch_size=2, factors=factors, identity=ident)
    assert [s for _, s in base] == ['written'] * 3 and [s for _, s in arm] == ['written'] * 3
    for (bk, _), (ak, _) in zip(base, arm):
        assert store.index(bk)['example_ids'] == store.index(ak)['example_ids']
    direct = collector.collect(rows()[2:4], factors=factors)
    stored = store.read_block('gold-rows-000002-000003', 'response_content', 1, 'mean_abs_term')
    assert np.array_equal(stored, direct['arrays'][('response_content', 1, 'mean_abs_term')])
    assert store.read_tokens('untuned-rows-000000-000001') is None
    again = nc.collect_cohort(collector, BlockStore(tmp_path / 'blocks'), rows(), arm='gold', batch_size=2,
                              factors=factors, identity=ident)
    assert [s for _, s in again] == ['reused'] * 3
    target = tmp_path / 'blocks/units/gold-rows-000004-000004/ct.bin'
    target.write_bytes(b'\0' + target.read_bytes()[1:])
    with pytest.raises(CorruptUnit):
        nc.collect_cohort(collector, BlockStore(tmp_path / 'blocks'), rows(), arm='gold', batch_size=2,
                          factors=factors, identity=ident)


def test_phase_1b_head_statistics_and_independent_head_read_check(adapted):
    model, path, factors = adapted
    o_factors = nc.load_saved_factors(path, [0, 1], scale=2.0, projection='o_proj')
    collector = nc.NeuronCollector(model, tolerances=TOL)
    out = collector.collect(rows(), factors=factors, heads=[1], head_factors=o_factors)
    assert out['arrays'][('response_content', 1, 'head_mean_norm')].shape == (5, 2)
    assert ('response_content', 0, 'head_mean_norm') not in out['arrays']
    checks = {r['check'] for r in out['validation']}
    assert {'head_read_sum', 'o_proj_lora_a_read'} <= checks
    flipped = dict(scale=2.0, layers={1: dict(a=-o_factors['layers'][1]['a'], b=o_factors['layers'][1]['b'])})
    with pytest.raises(nc.CollectorFailure, match='head read'):
        collector.collect(rows(), factors=factors, heads=[1], head_factors=flipped)
    base = collector.collect(rows(), heads=[0, 1])
    assert ('response_content', 0, 'head_mean_term') not in base['arrays']


def test_float64_reference_collector_keeps_float64_means(adapted):
    model, _, factors = adapted
    model64 = model.double()
    out = nc.NeuronCollector(model64, tolerances=TOL, dtype='float64').collect(rows(), factors=factors)
    assert out['arrays'][('response_content', 1, 'mean_abs_z')].dtype == np.float64
    with pytest.raises(nc.CollectorFailure, match='float32 parameters'):
        nc.NeuronCollector(model64, tolerances=TOL)


def test_explicit_pad_token_is_used_when_the_model_config_has_none():
    model = tiny()
    model.config.pad_token_id = None  # like the pinned Qwen2.5 config.json
    with pytest.raises(nc.CollectorFailure, match='explicit padding token required'):
        nc.NeuronCollector(model, tolerances=TOL).collect(rows())
    collector = nc.NeuronCollector(model, tolerances=TOL, pad_token_id=0)
    out = collector.collect(rows())
    assert collector.runtime()['pad_token_id'] == 0 and collector.runtime()['pad_source'] == 'explicit (tokenizer)'
    assert out['arrays']
