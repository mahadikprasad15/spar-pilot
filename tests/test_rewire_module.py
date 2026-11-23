"""Ticket 02: masked read-change (rewire) module. Spec tests 1-4, 10b, 23 and 25, on CPU with no download."""
import copy
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from pilot_eval import c_attribution_checks as cc  # noqa: E402
from pilot_eval import rewire_module as rw  # noqa: E402
from pilot_eval.sft_backend import HFTrainingEngine, completion_loss, frozen_weight_hash  # noqa: E402

FIXTURE = Path(__file__).parent / 'fixtures' / 'rewire' / 'config.json'
PINNED = Path(os.environ.get('PILOT_ARTIFACTS', '/Users/prasadmahadik/spar/pilot1/artifacts')) / \
    'inputs/models/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
SHAPE_FIELDS = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'hidden_act', 'model_type')
LAMBDA = 6e-4


def base_model(seed=0, layers=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=layers,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        attn_implementation='eager', use_cache=False)).eval()


def rewired(rank=1, seed=0, **kw):
    base = base_model(seed)
    model = copy.deepcopy(base)
    rw.install(model, rank=rank, **kw)
    return base, model


def ids(n=9, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 32, (1, n), generator=g)


def logits(model, inputs):
    with torch.no_grad():
        return model(input_ids=inputs, use_cache=False).logits


def fill_coefficients(model, value=0.3, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in rw.modules(model):
            for c in m.coefficients().values():
                c.copy_(value * torch.randn(c.shape, generator=g))


def expected_active(model):
    """Independent L0 term from the documented equation: Σ σ(logα − β ln(−γ/ζ))."""
    shift = (2 / 3) * math.log(0.1 / 1.1)
    return sum(torch.sigmoid(m.rewire_log_alpha - shift).sum() for m in rw.modules(model))


def objective(model, inputs):
    labels = inputs.clone()
    labels[:, :3] = -100
    return completion_loss(model(input_ids=inputs, use_cache=False), labels) + LAMBDA * expected_active(model)


# --- shapes from the saved config fixture ----------------------------------------------------------

def test_shape_fixture_matches_the_pinned_config_when_present():
    fixture = json.loads(FIXTURE.read_text())
    assert 'pad_token_id' not in fixture  # like the real file
    assert (fixture['hidden_size'], fixture['intermediate_size'], fixture['num_hidden_layers']) == (1536, 8960, 28)
    if (PINNED / 'config.json').exists():
        real = json.loads((PINNED / 'config.json').read_text())
        assert {k: real[k] for k in SHAPE_FIELDS} == {k: fixture[k] for k in SHAPE_FIELDS}


@pytest.mark.parametrize('rank, read', [(1, 587_776), (0, 770_703_360)])
def test_installed_parameter_counts_on_the_pinned_shapes(rank, read):
    """The full pinned architecture on the meta device (no memory, no download), shapes from the fixture."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(**{k: v for k, v in json.loads(FIXTURE.read_text()).items()
                            if k not in ('architectures', 'torch_dtype', 'transformers_version')})
    with torch.device('meta'):
        model = Qwen2ForCausalLM(config)
    rw.install(model, rank=rank)
    named = dict(rw.trainable_parameters(model))
    counted = {'read': sum(p.numel() for n, p in named.items() if 'log_alpha' not in n),
               'mask': sum(p.numel() for n, p in named.items() if 'log_alpha' in n)}
    assert counted == {'read': read, 'mask': 250_880}
    assert rw.parameter_counts(1536, 8960, 28, rank) == counted
    assert all(p.device.type == 'meta' for p in named.values())


# --- 1. off switch and zero coefficients --------------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 0, 3])
def test_zero_coefficients_and_off_switch_equal_the_base_model_bit_for_bit(rank):
    base, model = rewired(rank)
    inputs = ids()
    reference = logits(base, inputs)
    assert torch.equal(logits(model, inputs), reference)  # zero coefficients
    fill_coefficients(model)
    assert not torch.equal(logits(model, inputs), reference)  # the change is live when on
    with rw.off(model):
        assert torch.equal(logits(model, inputs), reference)
    assert all(m.enabled for m in rw.modules(model))  # restored after the context
    # per MLP, the off module reproduces the original Qwen2MLP forward exactly
    x = torch.randn(1, 5, 8)
    with rw.off(model):
        for layer, original in zip(model.model.layers, base.model.layers):
            assert torch.equal(layer.mlp(x), original.mlp(x))


def test_install_keeps_base_parameter_names_and_hash():
    base, model = rewired(1)
    base_names = {n for n, _ in base.named_parameters()}
    names = {n for n, _ in model.named_parameters()}
    assert base_names <= names and all(rw.is_rewire_parameter(n) for n in names - base_names)
    assert frozen_weight_hash(model, rw.is_rewire_parameter) == frozen_weight_hash(base)
    with pytest.raises(ValueError, match='already rewired'):
        rw.install(model, rank=1)


def test_init_follows_the_documented_seeding():
    _, model = rewired(1, seed=0)
    for m in rw.modules(model):
        g = torch.Generator(device='cpu').manual_seed(42 + m.layer)
        for part in rw.PARTS:  # gate, then up
            expected = torch.randn(1, 8, generator=g) / math.sqrt(8)
            assert torch.equal(getattr(m, f'rewire_A_{part}'), expected)
            assert torch.count_nonzero(getattr(m, f'rewire_B_{part}')) == 0
        assert torch.equal(m.rewire_log_alpha, torch.ones(12))
    a0, a1 = (rw.modules(model)[i].rewire_A_gate for i in (0, 1))
    assert not torch.equal(a0, a1)


@pytest.mark.parametrize('rank', [1, 0, 2])
def test_pre_activations_match_the_documented_equations_for_both_parts(rank):
    """gate' = gate(x) + z ⊙ Δgate(x) and up' = up(x) + z ⊙ Δup(x), one z per neuron shared by both parts."""
    _, model = rewired(rank)
    fill_coefficients(model)
    m = rw.modules(model)[1]
    z = torch.linspace(0.0, 1.0, 12)
    m.gate = lambda log_alpha: z
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        gate, up = m.pre_activations(x)
        for part, got in (('gate', gate), ('up', up)):
            if rank == 0:
                change = torch.einsum('btd,nd->btn', x, getattr(m, f'rewire_W_{part}'))
            else:
                A, B = getattr(m, f'rewire_A_{part}'), getattr(m, f'rewire_B_{part}')
                change = sum(torch.einsum('btd,d->bt', x, A[k])[..., None] * B[:, k] for k in range(rank))
            base = getattr(m, f'{part}_proj')(x)
            assert torch.allclose(got, base + z * change, atol=1e-6, rtol=0), part
            assert torch.equal(got[..., 0], base[..., 0])  # z = 0 removes the change entirely


# --- 2. shapes and zero-based indexing ---------------------------------------------------------------

def _layer_inputs(model, inputs):
    captured = {}
    handles = [layer.mlp.register_forward_pre_hook(lambda mod, args, i=i: captured.__setitem__(i, args[0].detach()))
               for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        model(input_ids=inputs, use_cache=False)
    for h in handles:
        h.remove()
    return captured


@pytest.mark.parametrize('rank', [1, 0])
@pytest.mark.parametrize('layer, neuron', [(0, 0), (1, 7), (1, 11)])
def test_perturbing_one_coefficient_changes_only_that_neurons_gate_preactivation(rank, layer, neuron):
    _, model = rewired(rank)
    inputs = ids()
    before = _layer_inputs(model, inputs)
    m = rw.modules(model)[layer]
    x = before[layer]
    with torch.no_grad():
        gate0, up0 = m.pre_activations(x)
        coefficient = m.coefficients()['gate']
        if rank == 0:
            coefficient[neuron] = 0.5
        else:
            coefficient[neuron, 0] = 0.5
        gate1, up1 = m.pre_activations(x)
    changed = (gate1 != gate0).any(dim=(0, 1))
    assert changed.nonzero().flatten().tolist() == [neuron]
    assert torch.equal(up1, up0)
    after = _layer_inputs(model, inputs)
    assert all(torch.equal(after[i], before[i]) for i in range(layer + 1))  # nothing upstream moves


# --- 3. trainable set and frozen base -------------------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 0])
def test_only_rewire_parameters_train_and_the_base_hash_is_unchanged(rank):
    _, model = rewired(rank)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {n for n, _ in rw.trainable_parameters(model)}
    expected = {'rewire_log_alpha'} | ({'rewire_W_gate', 'rewire_W_up'} if rank == 0 else
                                       {'rewire_A_gate', 'rewire_A_up', 'rewire_B_gate', 'rewire_B_up'})
    assert {n.split('.')[-1] for n in trainable} == expected
    before = frozen_weight_hash(model, rw.is_rewire_parameter)
    optimizer = torch.optim.AdamW([p for _, p in rw.trainable_parameters(model)], lr=1e-2, foreach=False)
    objective(model, ids()).backward()
    optimizer.step()
    assert frozen_weight_hash(model, rw.is_rewire_parameter) == before


def test_mutation_unfreezing_a_base_weight_changes_the_base_hash():
    _, model = rewired(1)
    model.model.layers[0].mlp.down_proj.weight.requires_grad_(True)
    before = frozen_weight_hash(model, rw.is_rewire_parameter)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, foreach=False)
    objective(model, ids()).backward()
    optimizer.step()
    assert frozen_weight_hash(model, rw.is_rewire_parameter) != before


# --- 4. gradients -------------------------------------------------------------------------------------

def _grads(model, inputs):
    model.zero_grad(set_to_none=True)
    objective(model, inputs).backward()
    return {n: p.grad.clone() for n, p in rw.trainable_parameters(model)}


def _l0_only_grads(model):
    model.zero_grad(set_to_none=True)
    (LAMBDA * expected_active(model)).backward()
    grads = {n: p.grad.clone() for n, p in rw.trainable_parameters(model) if 'log_alpha' in n}
    model.zero_grad(set_to_none=True)
    return grads


@pytest.mark.parametrize('rank', [1, 0])
def test_gradients_at_init_and_after_one_step(rank):
    _, model = rewired(rank)
    inputs = ids()
    grads = _grads(model, inputs)
    l0 = _l0_only_grads(model)
    for name, g in grads.items():
        tail = name.split('.')[-1]
        if tail.startswith('rewire_A_'):
            assert torch.count_nonzero(g) == 0, name  # b = 0, so the direction has no gradient yet
        elif tail.startswith(('rewire_B_', 'rewire_W_')):
            assert torch.count_nonzero(g) > 0, name
        else:
            assert torch.equal(g, l0[name]), name  # only the L0 term reaches logα while the change is zero
    optimizer = torch.optim.AdamW([p for _, p in rw.trainable_parameters(model)], lr=1e-2, foreach=False)
    model.zero_grad(set_to_none=True)
    objective(model, inputs).backward()
    optimizer.step()
    grads = _grads(model, inputs)
    l0 = _l0_only_grads(model)
    for name, g in grads.items():
        assert torch.count_nonzero(g) > 0, name
        if 'log_alpha' in name:
            assert not torch.equal(g, l0[name]), name  # the CE gradient now reaches the mask


def test_mutation_detaching_the_gate_starves_the_mask_of_the_ce_gradient():
    _, model = rewired(1)
    rw.set_gate(model, lambda log_alpha: rw.deterministic_gate(log_alpha).detach())
    inputs = ids()
    optimizer = torch.optim.AdamW([p for _, p in rw.trainable_parameters(model)], lr=1e-2, foreach=False)
    objective(model, inputs).backward()
    optimizer.step()
    grads, l0 = _grads(model, inputs), _l0_only_grads(model)
    assert all(torch.equal(g, l0[n]) for n, g in grads.items() if 'log_alpha' in n)  # test 4 would fail


def test_deterministic_gate_equation():
    log_alpha = torch.tensor([-30.0, -math.log(11.0) - 1e-3, 0.0, 1.0, 30.0])
    z = rw.deterministic_gate(log_alpha)
    expected = [min(1.0, max(0.0, 1.2 / (1 + math.exp(-v)) - 0.1)) for v in log_alpha.tolist()]
    assert z.tolist() == pytest.approx(expected, abs=1e-7)
    assert z[0] == 0 and z[1] == 0 and z[-1] == 1


# --- 10b. rank 0 and rank 1 share one code path -------------------------------------------------------

def _dyadic(shape, gen, scale):
    return torch.randint(-4, 5, shape, generator=gen).float() * scale


def test_rank0_rows_set_to_b_i_a_reproduce_rank1_forward_and_mask_gradients_exactly():
    """Small dyadic values make every product and sum exact in FP32, so the two parameterisations of the
    same change must agree bit for bit (any extra term or different gating would break equality)."""
    base = base_model(3, layers=1)
    r1, r0 = copy.deepcopy(base), copy.deepcopy(base)
    m1, = rw.install(r1, rank=1)
    m0, = rw.install(r0, rank=0)
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for part in rw.PARTS:
            a = _dyadic((1, 8), gen, 0.25)
            b = _dyadic((12, 1), gen, 0.125)
            getattr(m1, f'rewire_A_{part}').copy_(a)
            getattr(m1, f'rewire_B_{part}').copy_(b)
            getattr(m0, f'rewire_W_{part}').copy_(b @ a)  # row i = b_i a
        alpha = _dyadic((12,), gen, 0.5)
        m1.rewire_log_alpha.copy_(alpha)
        m0.rewire_log_alpha.copy_(alpha)
    x = _dyadic((2, 5, 8), gen, 0.5)
    out = []
    for m in (m1, m0):
        m.zero_grad(set_to_none=True)
        y = m(x)
        (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
        out.append((y.detach(), m.rewire_log_alpha.grad.clone(), m.pre_activations(x)))
    (y1, g1, (gate1, up1)), (y0, g0, (gate0, up0)) = out
    assert torch.equal(gate1, gate0) and torch.equal(up1, up0)
    assert torch.equal(y1, y0)
    assert torch.equal(g1, g0) and torch.count_nonzero(g1) > 0


def test_rank_must_be_a_non_negative_integer():
    for rank in (-1, 1.0, None):
        with pytest.raises(ValueError, match='rank'):
            rw.install(base_model(), rank=rank)


# --- 23. device placement -----------------------------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 0])
def test_rewire_parameters_live_on_the_model_device_and_meta_forward_runs(rank):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    with torch.device('meta'):
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1,
                                             attn_implementation='eager'))
    mods = rw.install(model, rank=rank)
    assert all(p.device.type == 'meta' for _, p in rw.trainable_parameters(model))
    x = torch.empty(1, 4, 8, device='meta')
    for m in mods:
        assert m(x).device.type == 'meta'


def test_mutation_a_cpu_tensor_in_the_meta_forward_fails():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    with torch.device('meta'):
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
                                             num_attention_heads=2, num_key_value_heads=1,
                                             attn_implementation='eager'))
    m, = rw.install(model, rank=1)
    m.rewire_log_alpha = torch.nn.Parameter(torch.ones(12))  # left on CPU
    with pytest.raises((RuntimeError, NotImplementedError)):
        m(torch.empty(1, 4, 8, device='meta'))


# --- 25. extended noop_check and HFTrainingEngine.preflight ------------------------------------------

def _rows(n=2):
    return [{'id': f'r{i}', 'input_ids': ids(7 + i, seed=i).flatten().tolist()} for i in range(n)]


def test_noop_check_on_a_rewired_model_with_the_rewire_filter():
    _, model = rewired(1)
    rows = _rows()
    assert cc.noop_check(model, rows, cc.empty_context, parameter_filter=rw.is_rewire_parameter)['passed']
    # zero-change identity at step 0: the off switch is a no-op on the initial model
    assert cc.noop_check(model, rows, lambda: rw.off(model), parameter_filter=rw.is_rewire_parameter)['passed']

    @contextmanager
    def leaky():
        with torch.no_grad():
            rw.modules(model)[0].rewire_A_gate[0, 0] += 1.0  # never restored
        yield

    result = cc.noop_check(model, rows, leaky, parameter_filter=rw.is_rewire_parameter)
    assert not result['passed'] and not result['restored']


def test_noop_check_refuses_a_filter_that_selects_nothing():
    """The old ``lora_`` filter on a rewired model would hash nothing and pass vacuously."""
    _, model = rewired(1)
    with pytest.raises(ValueError, match='selects no parameter'):
        cc.noop_check(model, _rows(), cc.empty_context)


def _engine(model):
    rows = [{'id': 'a', 'input_ids': ids(9).flatten().tolist(), 'prompt_tokens': 4,
             'labels': [-100] * 4 + ids(9).flatten().tolist()[4:]}]
    engine = object.__new__(HFTrainingEngine)  # the PEFT constructor is not the code under test
    engine.model, engine.rows, engine.tokenizer, engine.config = model, rows, None, {'seed': 0}
    engine.targets = [f'model.layers.{m.layer}.mlp' for m in rw.modules(model)]
    engine.base_hash = frozen_weight_hash(model)
    return engine


@pytest.mark.parametrize('rank', [1, 0])
def test_preflight_runs_on_a_rewired_model_through_the_hooks(tmp_path, rank):
    _, model = rewired(rank)
    before = {n: p.detach().clone() for n, p in rw.trainable_parameters(model)}
    evidence = _engine(model).preflight(tmp_path, zero_change=rw.zero_change, trainable=rw.trainable_parameters)
    assert evidence['zero_write'] and evidence['finite_gradients'] and evidence['finite_loss']
    assert all(torch.equal(p, before[n]) for n, p in rw.trainable_parameters(model))  # restored exactly


def test_preflight_refuses_a_nonzero_initial_change_and_a_wrong_trainable_set(tmp_path):
    _, model = rewired(1)
    fill_coefficients(model)
    with pytest.raises(ValueError, match='zero write'):
        _engine(model).preflight(tmp_path, zero_change=rw.zero_change, trainable=rw.trainable_parameters)
    _, model = rewired(1)
    with pytest.raises(ValueError, match='trainable set'):
        _engine(model).preflight(tmp_path, zero_change=rw.zero_change,
                                 trainable=lambda m: [(n, p) for n, p in rw.trainable_parameters(m)
                                                      if 'log_alpha' not in n])
    model.model.layers[0].mlp.down_proj.weight.requires_grad_(True)  # a base weight left trainable
    with pytest.raises(ValueError, match='trainable set'):
        _engine(model).preflight(tmp_path, zero_change=rw.zero_change, trainable=rw.trainable_parameters)


def test_default_preflight_hooks_are_the_lora_behaviour():
    """Gold's call passes no hooks; the defaults are the PEFT checks it always made."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base_model(), LoraConfig(r=1, lora_alpha=1, target_modules=['down_proj'],
                                                    init_lora_weights=True))
    inputs = ids()
    assert HFTrainingEngine.lora_zero_change(model, inputs)
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if 'lora_B' in n).fill_(0.5)
    assert not HFTrainingEngine.lora_zero_change(model, inputs)
    assert [n for n, _ in HFTrainingEngine.requires_grad_parameters(model)] == \
        [n for n, p in model.named_parameters() if p.requires_grad]


def test_adapter_hash_default_is_unchanged_for_peft_models():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base_model(), LoraConfig(r=1, lora_alpha=1, target_modules=['down_proj']))
    assert cc.adapter_hash(model) == cc.adapter_hash(model, lambda n: 'lora_' in n)
