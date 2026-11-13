"""Ticket 03: scalar-read attribution instrument on a tiny random Qwen2 + PEFT 0.18.0 (CPU, no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import attribution_instrument as ai  # noqa: E402
from pilot_eval import attribution_metrics as am  # noqa: E402

PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
STOP = [31]
EPS = 1e-2


def tiny(alpha=2, seed=0, zero_b=False):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        attn_implementation='eager'))
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=alpha, lora_dropout=0, target_modules=PROJ))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(0, 0.0 if zero_b else 0.8)
            if 'lora_A' in name:
                p.normal_(0, 0.8)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def rows(n=3):
    out = []
    for i in range(n):
        size = 9 + 2 * i
        ids = [(5 * i + 3 * k) % 29 + 1 for k in range(size)]
        out.append(dict(id=f'e{i}', input_ids=ids,
                        masks=dict(trace_targets=list(range(4, size - 1)), stop_targets=list(range(4, size - 3)),
                                   length=size)))
    out.append(dict(id='single', input_ids=[2, 4, 6, 8, 10, 12, 14],
                    masks=dict(trace_targets=[4, 5], stop_targets=[], length=7)))  # empty stop mask
    return out


def instrument(model, **kw):
    return ai.Instrument(model, stop_ids=STOP, pad_token_id=0, head_chunk=3, identity_rtol=1e-4, **kw)


def metric(model, rows_, name, gates=None, neuron=None):
    """Independent metric path: plain full logits from the model + attribution_metrics; optional gates."""
    inst = ai.Instrument(model, stop_ids=STOP, pad_token_id=0, identity_rtol=1e-4)
    ids, attention, positions = inst.batch(rows_)
    with torch.no_grad(), ai.branch_gates(model, gates or {}), ai.neuron_gate(model, neuron):
        logits = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
    masks = [r['masks'] for r in rows_]
    if name == 'trace':
        return am.trace_loglik(logits, ids, masks).numpy()
    return am.stop_pull(logits, masks, stop_ids=STOP)[0].numpy()


def finite_difference(model, rows_, name, index):
    n = len(rows_)
    up = metric(model, rows_, name, {index: torch.full((n, 1, 1), 1 + EPS)})
    down = metric(model, rows_, name, {index: torch.full((n, 1, 1), 1 - EPS)})
    return -(up - down) / (2 * EPS)


@pytest.fixture(scope='module')
def model():
    return tiny()


@pytest.fixture(scope='module')
def result(model):
    inst = instrument(model)
    return inst, inst.attribute(rows())


def test_adapter_order_layers_then_projection_order(model):
    inst = instrument(model)
    assert inst.adapters[:7] == [(0, p) for p in PROJ] and len(inst.adapters) == 14
    assert inst.down_index == {0: 6, 1: 13}


def test_frozen_model_without_embedding_graph_has_no_grad_fn(model):
    """Documents the review round-2 failure: frozen weights + constant gate give logits without a graph."""
    inst = instrument(model)
    ids, attention, positions = inst.batch(rows()[:1])
    with torch.enable_grad():
        logits = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
    assert logits.grad_fn is None and not logits.requires_grad


def test_g1_hooks_leave_logits_identical(model, result):
    inst, _ = result
    ids, attention, positions = inst.batch(rows())
    with torch.no_grad():
        plain = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
    assert torch.equal(inst.hooked_logits(rows()), plain)


def test_metric_values_match_full_logit_path(model, result):
    _, out = result
    assert np.allclose(out['metrics']['trace'], metric(model, rows(), 'trace'), rtol=1e-6, atol=1e-7)
    stop = metric(model, rows(), 'stop')
    assert np.allclose(out['metrics']['stop'][:3], stop[:3], rtol=1e-6) and np.isnan(out['metrics']['stop'][3])
    assert out['stop_reasons'] == [None, None, None, 'empty-stop-pull-mask']


@pytest.mark.parametrize('name', ['trace', 'stop'])
def test_scalar_read_attribution_matches_branch_output_oracle_every_module_scale_two(model, result, name):
    _, out = result
    oracle = ai.branch_oracle(model, rows(), name, stop_ids=STOP, pad_token_id=0)
    got = out['attr'][name]
    defined = ~np.isnan(got[:, 0])
    assert defined.sum() == (4 if name == 'trace' else 3)
    assert np.allclose(got[defined], oracle[defined], rtol=1e-5, atol=1e-7)
    assert float(model.base_model.model.model.layers[0].mlp.down_proj.scaling['default']) == 2.0


@pytest.mark.parametrize('name', ['trace', 'stop'])
def test_attributions_match_independent_finite_differences(model, result, name):
    inst, out = result
    for index in [0, 3, 5, 6, 13]:  # 5 adapters incl. both down_proj
        fd = finite_difference(model, rows(), name, index)
        got = out['attr'][name][:, index]
        defined = ~np.isnan(fd)
        assert np.allclose(got[defined], fd[defined], rtol=2e-3, atol=1e-5), (index, got, fd)


def test_neuron_attributions_match_finite_differences_and_sum_identity(model, result):
    inst, out = result
    n = len(rows())
    for layer, j in [(0, 1), (1, 4), (1, 11)]:
        for name in ['trace', 'stop']:
            up = metric(model, rows(), name, neuron=(layer, j, torch.full((n, 1, 1), 1 + EPS)))
            down = metric(model, rows(), name, neuron=(layer, j, torch.full((n, 1, 1), 1 - EPS)))
            fd = -(up - down) / (2 * EPS)
            got = out['neuron'][name][layer][:, j]
            defined = ~np.isnan(fd)
            assert np.allclose(got[defined], fd[defined], rtol=2e-3, atol=1e-5)
    for name in ['trace', 'stop']:
        for layer in (0, 1):
            total = out['neuron'][name][layer].sum(1)
            adapter = out['attr'][name][:, inst.down_index[layer]]
            defined = ~np.isnan(adapter)
            assert np.allclose(total[defined], adapter[defined], rtol=1e-6, atol=1e-9)


def test_metric_order_does_not_change_results(model, result):
    _, out = result
    reverse = instrument(model, metric_order=('stop', 'trace')).attribute(rows())
    for name in ['trace', 'stop']:
        assert np.array_equal(out['attr'][name], reverse['attr'][name], equal_nan=True)


def test_untuned_and_zero_b_give_exact_zero(model):
    out = instrument(model).attribute(rows(), untuned=True)
    assert out['routing']['lora_a_calls'] == 0
    assert np.all(out['attr']['trace'] == 0) and np.all(np.nan_to_num(out['attr']['stop']) == 0)
    zero = instrument(tiny(zero_b=True)).attribute(rows())
    assert np.all(zero['attr']['trace'] == 0)
    assert all(np.all(v == 0) for v in zero['neuron']['trace'].values())


def test_write_and_read_summaries_for_matching_and_cbar(model, result):
    inst, out = result
    ids, attention, positions = inst.batch(rows())
    raw = inst.attribute(rows(), raw=True)['raw']
    unpadded = attention.bool()
    c = raw['c'][6][..., 0].double()
    assert np.allclose(out['sum_abs_c'][:, 6], (c.abs() * unpadded).sum(1).numpy())
    assert np.array_equal(out['token_count'], unpadded.sum(1).numpy())
    assert np.allclose(out['signed_sum_c'][:, 0], (c * unpadded).sum(1).numpy())  # down_proj layer 0


def test_pass2_recompute_equals_stored_z_reference(model, result):
    inst, out = result
    ids, attention, positions = inst.batch(rows())
    raw = inst.attribute(rows(), raw=True)['raw']
    z = {}
    hooks = [model.base_model.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, a, l=l: z.__setitem__(l, a[0].detach().double())) for l in (0, 1)]
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
    for h in hooks:
        h.remove()
    for layer in (0, 1):
        a = model.base_model.model.model.layers[layer].mlp.down_proj.lora_A['default'].weight[0].double()
        g = raw['g']['trace'][inst.down_index[layer]][..., 0].double() * attention
        reference = -(a * (z[layer] * g[..., None]).sum(1))
        assert np.allclose(out['neuron']['trace'][layer], reference.numpy(), rtol=1e-10, atol=1e-12)


def test_batch_one_versus_batch_of_all(model, result):
    _, out = result
    inst = instrument(model)
    for k, row in enumerate(rows()):
        single = inst.attribute([row])
        for name in ['trace', 'stop']:
            assert np.allclose(single['attr'][name][0], out['attr'][name][k], rtol=1e-5, atol=1e-7, equal_nan=True)


@pytest.mark.parametrize('k', [-2.0, 0.5])
def test_gauge_invariance(model, result, k):
    _, out = result
    gauged = tiny()
    with torch.no_grad():
        for name, p in gauged.named_parameters():
            if 'lora_A' in name:
                p.mul_(k)
            if 'lora_B' in name:
                p.div_(k)
    other = instrument(gauged).attribute(rows())
    for name in ['trace', 'stop']:
        assert np.allclose(other['attr'][name], out['attr'][name], rtol=1e-4, atol=1e-6, equal_nan=True)
        for layer in (0, 1):
            assert np.allclose(other['neuron'][name][layer], out['neuron'][name][layer], rtol=1e-4, atol=1e-6,
                               equal_nan=True)


def test_prompt_only_adapter_effect_is_attributed(model):
    """The adapter at (layer 0, v_proj) acts only at prompt position 1: attribution must still match its FD."""
    index = 2
    module = model.base_model.model.model.layers[0].self_attn.v_proj

    def only_prompt(m, args, out):
        mask = torch.zeros_like(out)
        mask[:, 1] = 40.0  # large enough for FP32 central differences to resolve
        return out * mask

    handle = module.lora_A['default'].register_forward_hook(only_prompt)
    try:
        inst = instrument(model)
        out = inst.attribute(rows(), raw=True)
        fd = finite_difference(model, rows(), 'trace', index)
        assert np.abs(fd).max() > 1e-4
        # FP32 central differences resolve ~ulp(|M|)/eps ~ 1.2e-5 here; the largest effect must match tightly.
        assert np.allclose(out['attr']['trace'][:, index], fd, rtol=2e-3, atol=1.5e-5)
        big = np.argmax(np.abs(fd))
        assert out['attr']['trace'][big, index] == pytest.approx(fd[big], rel=5e-3)
        # mutation: summing only response positions misses the effect entirely
        response_only = ai.contract(out['raw']['c'][index], out['raw']['g']['trace'][index],
                                    ai.response_mask(inst, rows()))
        assert np.allclose(response_only, 0) and not np.allclose(response_only, fd, atol=1e-6)
    finally:
        handle.remove()


def test_mutations_each_fail_against_finite_differences(model):
    inst = instrument(model)
    out = inst.attribute(rows(), raw=True)
    raw = out['raw']
    ids, attention, positions = inst.batch(rows())
    index = 5  # layer 0 up_proj
    fd = finite_difference(model, rows(), 'trace', index)
    correct = ai.contract(raw['c'][index], raw['g']['trace'][index], attention)
    assert np.allclose(correct, fd, rtol=2e-3, atol=1e-5)
    mutants = dict(
        wrong_layer=ai.contract(raw['c'][index], raw['g']['trace'][index + 7], attention),
        wrong_branch=ai.contract(raw['c'][index], raw['g']['trace'][index - 1], attention),
        wrong_sign=-correct,
        mean_not_sum=correct / attention.sum(1).numpy(),
        response_only=ai.contract(raw['c'][index], raw['g']['trace'][index], ai.response_mask(inst, rows())))
    for name, value in mutants.items():
        assert not np.allclose(value, fd, rtol=2e-3, atol=1e-5), name


def test_routing_and_frozen_parameters_checked(model):
    inst = instrument(model)
    assert inst.check_routing()['status'] == 'pass'
    p = next(p for n, p in model.named_parameters() if 'lora_A' in n)
    p.requires_grad_(True)
    try:
        with pytest.raises(ai.InstrumentFailure, match='frozen'):
            inst.attribute(rows())
    finally:
        p.requires_grad_(False)
    model.base_model.model.model.layers[0].mlp.up_proj._disable_adapters = True
    try:
        with pytest.raises(ai.InstrumentFailure, match='routing'):
            inst.attribute(rows())
    finally:
        model.base_model.model.model.layers[0].mlp.up_proj._disable_adapters = False


def test_set_adapter_rechecked_between_arms():
    m = tiny()
    from peft import LoraConfig
    m.add_adapter('other', LoraConfig(r=1, lora_alpha=2, lora_dropout=0, target_modules=PROJ))
    for p in m.parameters():
        p.requires_grad_(False)
    inst = ai.Instrument(m, adapter='other', stop_ids=STOP, pad_token_id=0, identity_rtol=1e-4)
    with pytest.raises(ai.InstrumentFailure, match='routing'):
        inst.attribute(rows())  # 'default' is still active
    m.set_adapter('other')
    for p in m.parameters():
        p.requires_grad_(False)  # set_adapter marks the new adapter trainable; the instrument refuses that
    assert inst.attribute(rows())['routing']['lora_a_calls'] == 14


def test_neuron_identity_violation_raises(model):
    inst = instrument(model)
    with pytest.raises(ai.InstrumentFailure, match='identity'):
        inst.attribute(rows(), _corrupt_pass2=True)


def test_unit_blocks_round_trip_through_a_sealed_unit(model, result, tmp_path):
    from pilot_eval import attribution_units as au
    inst, out = result
    blocks, counts, results = ai.unit_blocks(out)
    assert counts['examples'] == [1, 1, 1, 1] and counts['stop'] == [1, 1, 1, 0]
    units = au.Units(tmp_path, plan_sha256='p', code_sha256='c', runtime_sha256='r')
    ids = out['example_ids']
    key = au.unit_key('ab', 'gold-sft', batch=0)
    units.write(key, identity=units.identity('ab', 'gold-sft', ids, input_sha256='i', batch=0), example_ids=ids,
                blocks=blocks, counts=counts, results=results)
    arrays, meta = units.read(key)
    assert np.allclose(arrays[('examples', 0, 'attr_trace')], out['attr']['trace'], rtol=1e-6)
    assert np.isnan(arrays[('stop', 0, 'attr_stop')][3]).all()
    assert np.allclose(arrays[('examples', 1, 'neuron_trace')], out['neuron']['trace'][1], rtol=1e-6, atol=1e-12)
    assert meta['results'][3]['stop_reason'] == 'empty-stop-pull-mask'


def test_pass2_saves_term_size_for_neuron_ordering(model, result):
    inst, out = result
    ids, attention, positions = inst.batch(rows())
    z = {}
    hook = model.base_model.model.model.layers[1].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: z.__setitem__(1, a[0].detach().double()))
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
    hook.remove()
    a = model.base_model.model.model.layers[1].mlp.down_proj.lora_A['default'].weight[0].double()
    expected = (a.abs() * (z[1].abs() * attention[..., None]).sum(1)).numpy()
    assert np.allclose(out['term_size'][1], expected, rtol=1e-10)
    blocks, _, _ = ai.unit_blocks(out)
    assert ('examples', 1, 'term_size') in blocks
