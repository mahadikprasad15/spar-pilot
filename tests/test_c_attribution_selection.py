"""Ticket 06: interventions, selection, matched controls, hot layers and the freeze."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pilot_eval import attribution_ablation as ab
from pilot_eval import c_attribution_selection as sel

ARTIFACTS = Path(os.environ.get('PILOT_ARTIFACTS', '/Users/prasadmahadik/spar/pilot1/artifacts'))
WRITES = ARTIFACTS / 'analysis/results-hub-v1/data/writes_modules.json'
WRITES_SHA = '35747fda0615032d996e3671833ed656a4eebf7a9b047a5a27656f6002ffd30d'


# --- matching ------------------------------------------------------------------------------------

def _f3():
    # Review round 1, F3: targets 0, 1 (1.00, 1.40); candidates 2, 3 (1.16, 0.85); one module type; tol 0.20.
    return dict(write=np.array([1.00, 1.40, 1.16, 0.85]), types=['q_proj'] * 4)


def test_f3_counterexample_fails_with_the_sequential_matcher():
    f = _f3()
    failures = [s for s in range(50)
                if ab.matched_controls([0, 1], f['write'], f['types'], exclude={0, 1}, tolerance=0.20, seed=s)['group'] is None]
    assert failures, 'the sequential matcher should report a false unavailable for some seed'


@pytest.mark.parametrize('seed', range(20))
def test_f3_counterexample_is_matched_by_the_group_matcher(seed):
    f = _f3()
    out = sel.match_group([0, 1], f['write'], f['types'], exclude=[], tolerance=0.20, seed=seed)
    assert out['available'] and out['group'] == [3, 2]
    assert out['per_target'] == {'0': 3, '1': 2} and out['deficient_targets'] == []


def test_infeasible_group_names_the_deficient_target_and_never_loosens():
    write = np.array([1.0, 3.0, 1.1, 1.15])
    out = sel.match_group([0, 1], write, ['x'] * 4, exclude=[], tolerance=0.20, seed=0)
    assert not out['available'] and out['group'] is None
    assert out['deficient_targets'] == [1]
    assert out['eligible_candidates'] == {'0': 2, '1': 0}
    assert out['largest_feasible_matching'] == 1 and out['deficient_strata'] == ['x']


def test_matching_respects_strata_exclusion_and_is_deterministic():
    rng = np.random.default_rng(0)
    size = rng.uniform(1, 2, 200)
    strata = [i % 4 for i in range(200)]
    targets = [0, 1, 2, 3, 4, 5, 8, 9]
    exclude = targets + [12, 13]
    a = sel.match_group(targets, size, strata, exclude=exclude, tolerance=0.2, seed=7)
    assert a == sel.match_group(targets, size, strata, exclude=exclude, tolerance=0.2, seed=7)
    assert a['available']
    for t, c in zip(targets, a['group']):
        assert strata[c] == strata[t] and c not in exclude
        assert abs(size[c] - size[t]) <= 0.2 * size[t]
    assert len(set(a['group'])) == len(targets)
    assert sel.match_group(targets, size, strata, exclude=exclude, tolerance=0.2, seed=8)['group'] != a['group']
    assert 0.8 <= a['match_ratio']['min'] <= a['match_ratio']['max'] <= 1.2


def test_unavailable_only_when_infeasible_per_stratum():
    size = np.array([1.0, 1.0, 1.05, 5.0, 1.0])
    strata = [0, 1, 0, 1, 1]
    out = sel.match_group([0, 1], size, strata, exclude=[4], tolerance=0.2, seed=0)
    assert out['per_stratum']['0']['available'] and not out['per_stratum']['1']['available']
    assert out['deficient_strata'] == ['1'] and out['deficient_targets'] == [1]


# --- selection ------------------------------------------------------------------------------------

def _summary(L=3, W=100, A=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=A)
    r = rng.normal(size=(L, W))
    return {'adapters': {'dJdlambda': a, 'removal_J': -a},
            'reads': {'layers': list(range(L)), 'dJdlambda': r, 'removal_J': -r}}


def _adapters(A=12):
    return [(i // 2, ['q_proj', 'down_proj'][i % 2]) for i in range(A)]


SEEDS = {g: i for i, g in enumerate(sel.GROUPS)}


def test_selection_is_positive_only_with_ties_by_index_and_shortfall_recorded():
    s = _summary(W=100)
    s['reads']['dJdlambda'][1] = -np.abs(s['reads']['dJdlambda'][1])
    s['reads']['dJdlambda'][1, [5, 9, 70]] = [0.3, 0.3, 0.1]   # only 3 positive reads in layer 1; 5 and 9 tie
    s['reads']['removal_J'] = -s['reads']['dJdlambda']
    rng = np.random.default_rng(1)
    out = sel.select_arm(s, adapters=_adapters(), layers=[0, 1, 2], adapter_size=rng.uniform(1, 2, 12),
                         read_size=rng.uniform(1, 2, (3, 100)), hot_layers=[1], seeds=SEEDS, k_reads=10, k_adapters=4)
    hot = out['groups']['hot64-reads']
    assert hot['reads'] == {'1': [5, 9, 70]}
    assert hot['targets'][:2] == [105, 109]  # tie broken by index
    assert hot['shortfall']['1'] == {'requested': 10, 'selected': 3, 'shortfall': 7, 'positive_available': 3}
    assert out['groups']['layer64-reads']['shortfall']['1']['selected'] == 3
    d = s['reads']['dJdlambda'].ravel()
    for name in ('global64-reads', 'hot64-reads', 'layer64-reads'):
        assert all(d[t] > 0 for t in out['groups'][name]['targets'])
    top = out['groups']['top10-adapters']
    assert all(s['adapters']['dJdlambda'][t] > 0 for t in top['targets']) and len(top['targets']) <= 4


def test_controls_exclude_every_same_kind_target_and_carry_predictions():
    s = _summary(W=300)
    rng = np.random.default_rng(2)
    out = sel.select_arm(s, adapters=_adapters(), layers=[0, 1, 2], adapter_size=np.ones(12) + rng.uniform(0, 0.1, 12),
                         read_size=rng.uniform(1, 1.2, (3, 300)), hot_layers=[0, 2], seeds=SEEDS,
                         k_reads=20, k_adapters=3)
    read_targets = {t for g in out['groups'].values() if g['kind'] == 'reads' for t in g['targets']}
    for name, g in out['groups'].items():
        c = g['control']
        assert c['available'], name
        same_kind = read_targets if g['kind'] == 'reads' else set(out['groups']['top10-adapters']['targets'])
        assert not set(c['group']) & same_kind
        assert g['prediction']['label'] == c['prediction']['label']
        assert g['prediction']['J'] == pytest.approx(-s['reads']['dJdlambda'].ravel()[g['targets']].sum()
                                                     if g['kind'] == 'reads' else
                                                     -s['adapters']['dJdlambda'][g['targets']].sum())
    assert out['groups']['layer64-reads']['layer_distribution'] == {'0': 20, '1': 20, '2': 20}
    harmful = out['harmful_adapters_reported_only']
    assert all(h['dJdlambda'] < 0 for h in harmful)


def test_infeasible_layer64_control_leaves_the_target_and_names_the_layer():
    s = _summary(W=80)
    s['reads']['dJdlambda'] = np.abs(s['reads']['dJdlambda'])
    s['reads']['removal_J'] = -s['reads']['dJdlambda']
    size = np.ones((3, 80))
    size[2] = 1.3 ** np.arange(80)  # layer 2: neighbours differ by 30%, so no read has a ±20% partner
    out = sel.select_arm(s, adapters=_adapters(), layers=[0, 1, 2], adapter_size=np.ones(12), read_size=size,
                         hot_layers=[0], seeds=SEEDS, k_reads=10, k_adapters=3)
    layer = out['groups']['layer64-reads']
    assert len(layer['targets']) == 30
    assert not layer['control']['available'] and layer['control']['deficient_strata'] == ['2']
    assert out['groups']['hot64-reads']['control']['available']


# --- hot layers -----------------------------------------------------------------------------------

def test_hot_layers_from_the_real_effective_write_table():
    if not WRITES.exists():
        pytest.skip('effective-write table not local')
    hot = sel.hot_layers(WRITES, WRITES_SHA)
    assert hot['layers'] == [11, 15, 16, 24]
    assert hot['per_arm_ranks'] == {'online': {11: 9, 15: 3, 16: 4}, 'offline': {15: 5, 16: 4, 24: 8}}
    table = sel.hot_layer_sensitivity(WRITES, WRITES_SHA)
    assert [row['intersection'] for row in table] == [[15, 16], [15, 16], [15, 16], [15, 16, 24]]
    assert table[1]['union'] == [11, 15, 16, 24]
    with pytest.raises(sel.SelectionError):
        sel.hot_layers(WRITES, '0' * 64)


def _writes_file(tmp_path):
    rows = []
    for arm, hot in (('online', {3: 1.0}), ('offline', {5: 1.0})):
        for layer in range(8):
            for module in ('down', 'up'):
                rows.append({'arm': arm, 'context': 'gsm8k_solution', 'step': 64, 'layer': layer, 'module': module,
                             'write': hot.get(layer, 0.01 * layer) if module == 'down' else 0.001})
    path = tmp_path / 'writes.json'
    path.write_text(json.dumps({'rows': rows}))
    return path


def test_hot_layer_binding_refuses_missing_late_and_non_derived_lists(tmp_path):
    path = _writes_file(tmp_path)
    sha = sel.file_sha256(path)
    hot = sel.hot_layers(path, sha, k=1)
    assert hot['layers'] == [3, 5]
    record = {'hot_layers': hot}
    good = sel.check_hot_layer_binding(record, {'launch_record_sha256': 'L'}, launch_record_sha256='L')
    assert good['layers'] == [3, 5]
    with pytest.raises(sel.SelectionError, match='no hot-layer'):
        sel.check_hot_layer_binding({}, {'launch_record_sha256': 'L'}, launch_record_sha256='L')
    with pytest.raises(sel.SelectionError, match='late'):
        sel.check_hot_layer_binding(record, {'launch_record_sha256': 'other'}, launch_record_sha256='L')
    with pytest.raises(sel.SelectionError, match='differ'):
        sel.check_hot_layer_binding({'hot_layers': dict(hot, layers=[3, 5, 7])}, {'launch_record_sha256': 'L'},
                                    launch_record_sha256='L')


# --- freeze ---------------------------------------------------------------------------------------

def test_freeze_is_sealed_and_refuses_existing_generation_outputs(tmp_path):
    freeze = sel.build_freeze('online-grpo', {'groups': {}}, launch_record_sha256='L', hot={'layers': [1]},
                              split={'A': [], 'B': []}, inputs={})
    sel.seal_freeze(tmp_path, 'online-grpo', freeze)
    assert sel.read_freeze(tmp_path, 'online-grpo')['launch_record_sha256'] == 'L'
    assert sel.seal_freeze(tmp_path, 'online-grpo', freeze)  # identical reseal is a reuse
    (tmp_path / sel.GENERATION_DIR).mkdir(parents=True)
    (tmp_path / sel.GENERATION_DIR / 'unit').write_text('x')
    with pytest.raises(sel.SelectionError):
        sel.seal_freeze(tmp_path, 'offline-grpo', freeze)
    with pytest.raises(ValueError):
        sel.read_freeze(tmp_path, 'offline-grpo')


# --- interventions --------------------------------------------------------------------------------

torch = pytest.importorskip('torch')


def _model():
    from test_attribution_instrument import tiny
    return tiny()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits


IDS = [3, 7, 11, 2, 9, 4, 13, 5]


def test_single_read_removal_equals_neuron_gate_zero():
    from pilot_eval import attribution_instrument as ai
    model = _model()
    with ab.neuron_read_removal(model, {1: [4]}):
        edited = _logits(model, IDS)
    with ai.neuron_gate(model, (1, 4, torch.zeros(1, 1, 1))):
        gated = _logits(model, IDS)
    torch.testing.assert_close(edited, gated, rtol=1e-6, atol=1e-6)
    assert not torch.equal(edited, _logits(model, IDS))


def test_multi_read_removal_across_layers_equals_the_hook_at_prefill_and_cached_decode():
    model = _model()
    reads = {0: [1, 5, 7], 1: [0, 4, 11]}
    with ab.neuron_read_removal(model, reads):
        edited = _logits(model, IDS)
        with torch.no_grad():
            gen_edit = model.generate(input_ids=torch.tensor([IDS[:4]]), max_new_tokens=6, min_new_tokens=6,
                                      do_sample=False, pad_token_id=0, output_scores=True, return_dict_in_generate=True)
    with ab.neuron_read_removal_hook(model, reads):
        hooked = _logits(model, IDS)
        with torch.no_grad():
            gen_hook = model.generate(input_ids=torch.tensor([IDS[:4]]), max_new_tokens=6, min_new_tokens=6,
                                      do_sample=False, pad_token_id=0, output_scores=True, return_dict_in_generate=True)
    torch.testing.assert_close(edited, hooked, rtol=1e-6, atol=1e-6)
    assert gen_edit.sequences.tolist() == gen_hook.sequences.tolist()
    for a, b in zip(gen_edit.scores, gen_hook.scores):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_read_removal_restores_bit_exact_and_detects_a_leak():
    model = _model()
    before = ab.lora_hash(model)
    with ab.neuron_read_removal(model, {0: [2]}):
        assert ab.lora_hash(model) != before
    assert ab.lora_hash(model) == before
    with pytest.raises(RuntimeError):
        with ab.neuron_read_removal(model, {0: [2]}):
            with torch.no_grad():
                ab._down_proj(model, 1).lora_A['default'].weight[0, 0] += 1.0


def test_adapter_zeroing_equals_removal_hook_and_edited_checkpoint(tmp_path):
    from test_attribution_instrument import tiny
    model = _model()
    indices = [3, 10]
    with ab.adapter_zeroing(model, indices):
        zeroed = _logits(model, IDS)
    with ab.removal(model, indices):
        removed = _logits(model, IDS)
    torch.testing.assert_close(zeroed, removed, rtol=0, atol=0)
    source = tmp_path / 'adapter'
    model.save_pretrained(str(source))
    adapters = [(l, p) for l, p, _ in ab.adapter_modules(model)]
    ab.edit_checkpoint(source, tmp_path / 'edited', [adapters[i] for i in indices])
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    fresh = tiny()  # same seed: identical base, LoRA then overwritten from the edited checkpoint
    set_peft_model_state_dict(fresh, load_file(str(tmp_path / 'edited' / 'adapter_model.safetensors')))
    torch.testing.assert_close(_logits(fresh, IDS), zeroed, rtol=1e-6, atol=1e-6)
    assert ab.lora_hash(model) == ab.lora_hash(_model())
