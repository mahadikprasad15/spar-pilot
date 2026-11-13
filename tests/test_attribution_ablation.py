"""Ticket 04: removal, edited checkpoints, matched controls, 64-term and constant reads, freeze, Stage C (CPU)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import attribution_ablation as ab  # noqa: E402
from pilot_eval import attribution_instrument as ai  # noqa: E402
from pilot_eval import attribution_metrics as am  # noqa: E402
from pilot_eval import attribution_traces as at  # noqa: E402
from pilot_eval import attribution_units as au  # noqa: E402
from tests.test_attribution_instrument import PROJ, STOP, rows, tiny  # noqa: E402


def logits(model, rows_):
    inst = ai.Instrument(model, stop_ids=STOP, pad_token_id=0, identity_rtol=1e-4)
    ids, attention, positions = inst.batch(rows_)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits


@pytest.fixture(scope='module')
def model():
    return tiny()


def test_empty_removal_is_the_full_model(model):
    full = logits(model, rows())
    with ab.removal(model, []):
        assert torch.equal(logits(model, rows()), full)


def test_removal_gate_equals_edited_checkpoint(tmp_path):
    from peft import PeftModel
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = tiny(seed=3)
    model.save_pretrained(tmp_path / 'adapter')
    chosen = [6, 9, 12]  # (0, down_proj), (1, o_proj), (1, up_proj)
    record = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited', [ai.adapter_modules(model)[i][:2]
                                                                            for i in chosen])
    assert len(record['zeroed_keys']) == 3 and record['source_sha256'] != record['edited_sha256']
    torch.manual_seed(3)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        attn_implementation='eager'))
    edited = PeftModel.from_pretrained(base, tmp_path / 'edited').eval()
    with ab.removal(model, chosen):
        gated = logits(model, rows())
    assert torch.allclose(logits(edited, rows()), gated, atol=1e-6, rtol=0)
    assert ab.verify_edit(tmp_path / 'edited', record)
    with pytest.raises(ValueError):
        ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited2', [(5, 'down_proj')])


def test_natural_read_all_neurons_is_full_and_none_is_removal(model):
    full = logits(model, rows())
    with ab.natural_read(model, 1, list(range(12))):
        assert torch.allclose(logits(model, rows()), full, atol=1e-6)
    with ab.removal(model, [13]):
        removed = logits(model, rows())
    with ab.natural_read(model, 1, []):
        assert torch.allclose(logits(model, rows()), removed, atol=1e-7)
    with ab.natural_read(model, 1, [0, 3, 5]):
        partial = logits(model, rows())
    assert not torch.allclose(partial, full, atol=1e-6)


def test_natural_read_is_not_rescaled(model):
    module = model.base_model.model.model.layers[1].mlp.down_proj
    seen, z = {}, {}
    h2 = module.register_forward_pre_hook(lambda m, a: z.__setitem__('z', a[0].detach()))
    with ab.natural_read(model, 1, [2, 7]):
        # registered after the read hook, so it observes the replaced read
        h = module.lora_A['default'].register_forward_hook(lambda m, a, o: seen.__setitem__('c', o.detach()))
        logits(model, rows()[:1])
    h.remove()
    h2.remove()
    a = module.lora_A['default'].weight[0]
    assert torch.allclose(seen['c'][..., 0], z['z'][..., 2] * a[2] + z['z'][..., 7] * a[7], atol=1e-7)


def test_constant_read_writes_s_b_cbar(model):
    module = model.base_model.model.model.layers[0].mlp.down_proj
    out = {}
    h = module.register_forward_hook(lambda m, a, o: out.__setitem__('y', o.detach()))
    h2 = module.base_layer.register_forward_hook(lambda m, a, o: out.__setitem__('base', o.detach()))
    with ab.constant_read(model, 6, 0.37):
        logits(model, rows()[:2])
    h.remove()
    h2.remove()
    b, s = module.lora_B['default'].weight[:, 0], float(module.scaling['default'])
    assert torch.allclose(out['y'] - out['base'], (s * b * 0.37).expand_as(out['y']), atol=1e-6)


def test_top_k_by_absolute_mean_with_layer_then_module_ties():
    mean = np.array([0.1, -0.5, 0.5, 0.0, 0.3, -0.3])
    assert ab.top_k(mean, 4) == [1, 2, 4, 5]


def write_fixture():
    # 14 adapters: projections in order per layer; writes chosen so matches exist for some targets only
    projections = [p for _ in range(2) for p in PROJ]
    write = np.array([1.0, 1.1, 5.0, 1.0, 2.0, 2.1, 3.0,
                      1.05, 0.95, 5.0, 1.15, 2.05, 9.0, 3.1])
    return projections, write


def test_matched_controls_exclusions_determinism_and_unavailable():
    projections, write = write_fixture()
    exclude = {0, 4, 6}  # union of both metrics' top-10 sets
    targets = [0, 4, 6]
    first = ab.matched_controls(targets, write, projections, exclude=exclude, tolerance=0.20, seed=20000)
    again = ab.matched_controls(targets, write, projections, exclude=exclude, tolerance=0.20, seed=20000)
    assert first == again
    per = first['per_target']
    assert per['0'] in (7,) and per['4'] in (11,) and per['6'] in (13,)  # same type, within +-20%, not excluded
    assert set(per.values()).isdisjoint(exclude) and first['group'] == [7, 11, 13]
    # without replacement: two targets competing for one eligible control
    two = ab.matched_controls([0, 7], write, projections, exclude={0, 7}, tolerance=0.20, seed=20000)
    assert two['per_target'] == {'0': None, '7': None} or len(set(v for v in two['per_target'].values() if v)) == \
        len([v for v in two['per_target'].values() if v])
    # unavailable is recorded, never loosened
    none = ab.matched_controls([12], write, projections, exclude={12}, tolerance=0.20, seed=20000)
    assert none['per_target'] == {'12': None} and none['group'] is None and none['unavailable'] == [12]


def test_matched_controls_without_replacement_across_targets():
    projections = ['q_proj'] * 4
    write = np.array([1.0, 1.0, 1.0, 1.0])
    out = ab.matched_controls([0, 1], write, projections, exclude={0, 1}, tolerance=0.2, seed=1)
    assert sorted(out['per_target'].values()) == [2, 3]
    out = ab.matched_controls([0, 1, 2], write, projections, exclude={0, 1, 2}, tolerance=0.2, seed=1)
    assert list(out['per_target'].values()).count(None) == 2 and out['group'] is None


def test_neuron_orders():
    attr_j = np.array([0.5, -0.2, 0.1, 0.9, -0.4])
    orders = ab.neuron_orders(attr_j, attr_i=float(attr_j.sum()), term_size=np.array([1, 5, 2, 2, 0.5]),
                              random_seeds=[11, 12, 13])
    assert orders['attribution'] == [3, 0, 2, 1, 4]  # largest contribution to the adapter's own effect first
    assert ab.neuron_orders(-attr_j, attr_i=float(-attr_j.sum()), term_size=np.ones(5),
                            random_seeds=[1, 2, 3])['attribution'] == [3, 0, 2, 1, 4]
    assert orders['term_size'] == [1, 2, 3, 0, 4] and len(orders['random']) == 3
    assert all(sorted(o) == list(range(5)) for o in orders['random'])
    assert orders['random'] == ab.neuron_orders(attr_j, attr_i=1.0, term_size=np.ones(5),
                                                random_seeds=[11, 12, 13])['random']
    with pytest.raises(ValueError):
        ab.neuron_orders(attr_j, attr_i=1.0, term_size=np.ones(5), random_seeds=None)


def discovery_inputs(seed=0):
    rng = np.random.default_rng(seed)
    projections, write = write_fixture()
    return dict(adapters=[(l, p) for l in range(2) for p in PROJ],
                mean_attr=dict(trace=rng.normal(size=14), stop=rng.normal(size=14)),
                write=write, signed_sum_c=np.array([3.0, -2.0]), token_total=np.array([10.0, 10.0]),
                neuron=dict(trace={0: rng.normal(size=12), 1: rng.normal(size=12)},
                            stop={0: rng.normal(size=12), 1: rng.normal(size=12)}),
                term_size={0: rng.random(12), 1: rng.random(12)})


SETTINGS = dict(tolerance=0.20, seed=20000, random_seeds=[101, 102, 103], top_group=10, top_singles=5, read_size=4)


def test_freeze_and_conditions_from_a_sealed_freeze(tmp_path):
    freeze = ab.build_freeze({'gold-sft': discovery_inputs()}, sources=[dict(split='discovery', sha256='x')],
                             analysis=dict(bootstrap=dict(resamples=10000)), **SETTINGS)
    arm = freeze['arms']['gold-sft']['trace']
    assert len(arm['top10']) == 10 and arm['top5'] == arm['top10'][:5] and len(arm['predicted_all']) == 14
    assert arm['down_proj']['index'] in (6, 13) and arm['down_proj']['cbar'] in (0.3, -0.2)
    path = ab.seal_freeze(tmp_path, freeze)
    conds = ab.conditions(path, 'gold-sft', 'trace')
    names = [c['name'] for c in conds]
    assert len(conds) <= 19 and 'top10-trace' in names and 'cbar-trace' in names
    assert sum(n.startswith('read64-') for n in names) == 5
    skipped = [c for c in conds if c['status'] == 'skipped']
    assert all(c['reason'] == 'matched-control-unavailable' for c in skipped)
    for c in conds:
        au.unit_key('c', 'gold-sft', condition=c['name'])  # valid unit names
    with pytest.raises(ValueError):
        ab.conditions(tmp_path / 'not-sealed', 'gold-sft', 'trace')


def test_unavailable_matched_group_is_a_recorded_skip_not_an_error(tmp_path):
    inputs = discovery_inputs()
    inputs['write'] = np.arange(1, 15, dtype=float) * 10  # nothing within +-20% of anything of the same type
    freeze = ab.build_freeze({'gold-sft': inputs}, sources=[dict(split='discovery', sha256='x')], analysis={},
                             **SETTINGS)
    conds = ab.conditions(ab.seal_freeze(tmp_path, freeze), 'gold-sft', 'stop')
    group = next(c for c in conds if c['name'] == 'match10-stop')
    assert group['status'] == 'skipped' and group['reason'] == 'matched-control-unavailable'


def test_freeze_refuses_confirmation_or_cohort_derived_inputs():
    for split in ['heldout', 'generation-cohort', 'calibration']:
        with pytest.raises(ValueError, match='discovery'):
            ab.build_freeze({'gold-sft': discovery_inputs()}, sources=[dict(split=split, sha256='x')], analysis={},
                            **SETTINGS)
    with pytest.raises(ValueError):  # random-order seeds must be supplied (launch record), never defaulted
        ab.build_freeze({'gold-sft': discovery_inputs()}, sources=[dict(split='discovery', sha256='x')], analysis={},
                        **dict(SETTINGS, random_seeds=None))


def test_stage_c_reference_is_the_same_arm_unablated(model):
    stage = ab.StageC(model, stop_ids=STOP, pad_token_id=0, head_chunk=3)
    none = stage.evaluate(rows(), dict(kind='removal', adapters=[]))
    assert np.all(none['delta_trace'] == 0) and np.all(none['kl'] == 0)
    cond = dict(kind='removal', adapters=[5, 13])
    out = stage.evaluate(rows(), cond)
    full = logits(model, rows())
    with ab.removal(model, [5, 13]):
        cut = logits(model, rows())
    inst = ai.Instrument(model, stop_ids=STOP, pad_token_id=0, identity_rtol=1e-4)
    ids, _, _ = inst.batch(rows())
    masks = [r['masks'] for r in rows()]
    expected_dl = (am.trace_loglik(cut, ids, masks) - am.trace_loglik(full, ids, masks)).numpy()
    assert np.allclose(out['delta_trace'], expected_dl, rtol=1e-5, atol=1e-7)
    assert np.allclose(out['kl'], am.kl_full_vs_condition(full, cut, masks).numpy(), rtol=1e-5, atol=1e-9)
    assert np.isnan(out['delta_stop'][3]) and np.isfinite(out['delta_stop'][:3]).all()


def test_generation_from_edited_checkpoint_matches_gated_generation_in_manifest_order(tmp_path):
    from peft import PeftModel
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = tiny(seed=5)
    model.save_pretrained(tmp_path / 'adapter')
    chosen = [2, 13]
    record = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited',
                                [ai.adapter_modules(model)[i][:2] for i in chosen])
    torch.manual_seed(5)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        attn_implementation='eager'))
    edited = PeftModel.from_pretrained(base, tmp_path / 'edited').eval()
    prompts = [dict(id=f'p{i}', prompt_ids=[(3 * i + k) % 30 + 1 for k in range(3 + i)]) for i in range(4)]
    manifest = [['p3', 'p1'], ['p2', 'p0']]
    gen = lambda m: at.HFGenerator(m, pad_token_id=0, eos_token_id=31, max_new_tokens=6, decode=str)  # noqa: E731
    from_checkpoint = at.generate(gen(edited), prompts, manifest)
    with ab.removal(model, chosen):
        gated = at.generate(gen(model), prompts, manifest)
    assert [o['id'] for o in from_checkpoint] == ['p3', 'p1', 'p2', 'p0']
    assert [o['generated_ids'] for o in from_checkpoint] == [o['generated_ids'] for o in gated]
    assert record['edited_sha256'] == ab.verify_edit(tmp_path / 'edited', record) and record['edited_sha256']


def test_discovery_write_is_scaled_b_norm_times_mean_abs_read():
    w = ab.discovery_write(sum_abs_c=np.array([4.0, 9.0]), token_total=np.array([2.0, 3.0]),
                           b_norms=np.array([0.5, 2.0]), scale=2.0)
    assert np.allclose(w, [2.0 * 0.5 * 2.0, 2.0 * 2.0 * 3.0])
