"""Ticket 07: greedy generation-ablation stage."""
import json

import numpy as np
import pytest

from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_selection as sel

torch = pytest.importorskip('torch')


def _freeze(control_available=True):
    def group(kind, targets, control, reads=None, creads=None):
        g = {'kind': kind, 'targets': targets, 'prediction': {'label': 'p', 'J': -0.01, 'pp': -1.0},
             'control': {'available': control is not None, 'group': control,
                         'prediction': {'label': 'p', 'J': -0.001, 'pp': -0.1} if control else None,
                         'deficient_strata': [] if control else ['1'], 'deficient_targets': [] if control else [9]}}
        if reads:
            g['reads'] = reads
            if control:
                g['control']['reads'] = creads
        return g
    return {'groups': {
        'top10-adapters': group('adapters', [0, 13], [7, 6]),
        'global64-reads': group('reads', [3, 20], [5, 22], {'0': [3], '1': [8]}, {'0': [5], '1': [10]}),
        'hot64-reads': group('reads', [12], [14], {'1': [0]}, {'1': [2]}),
        'layer64-reads': group('reads', [1, 13], [2, 15] if control_available else None,
                               {'0': [1], '1': [1]}, {'0': [2], '1': [3]}),
    }}


def test_conditions_follow_the_spec_order_and_record_skips():
    c = cg.conditions(_freeze(control_available=False))
    assert [x['name'] for x in c] == list(cg.ORDER)
    skipped = [x for x in c if x['status'] == 'skipped']
    assert [x['name'] for x in skipped] == ['match-layer64']
    assert skipped[0]['reason'] == 'matched-control-unavailable' and skipped[0]['deficient_strata'] == ['1']
    assert next(x for x in c if x['name'] == 'top10-adapters')['adapters'] == [0, 13]
    assert next(x for x in c if x['name'] == 'match-hot64')['reads'] == {'1': [2]}


def test_batches_are_fixed_chunks():
    assert cg.batches(['a', 'b', 'c', 'd', 'e'], 2) == [['a', 'b'], ['c', 'd'], ['e']]
    with pytest.raises(cg.GenerationError):
        cg.batches(['a', 'a'], 2)


def test_mcnemar_exact_values():
    assert cg.mcnemar_exact(0, 0) == 1.0
    assert cg.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert cg.mcnemar_exact(3, 3) == 1.0
    assert cg.mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)


# --- tiny-model runs ------------------------------------------------------------------------------

def _model():
    from test_attribution_instrument import tiny
    return tiny()


QUESTIONS = {f'q{i}': {'prompt_ids': [(3 * i + k) % 29 + 1 for k in range(3 + i % 3)], 'prompt': f'p{i}',
                       'gold': '#### 5'} for i in range(5)}
DECODE = lambda ids: 'The answer is ' + str(len(ids) % 7)  # noqa: E731
KW = dict(decode=DECODE, pad_token_id=0, eos_token_id=31, cap=6)


def _sealed_freeze(root, arm='online-grpo'):
    sel.seal_freeze(root, arm, sel.build_freeze(arm, _freeze(), launch_record_sha256='L', hot={'layers': [1]},
                                                split={}, inputs={}))


def test_greedy_runs_reproduce_and_batch_membership_is_fixed(tmp_path):
    model = _model()
    plan = cg.batches(list(QUESTIONS), 2)
    a = cg.greedy(model, [QUESTIONS[q]['prompt_ids'] for q in plan[0]], pad_token_id=0, eos_token_id=31, cap=6)
    b = cg.greedy(model, [QUESTIONS[q]['prompt_ids'] for q in plan[0]], pad_token_id=0, eos_token_id=31, cap=6)
    assert a == b
    _sealed_freeze(tmp_path)
    out = cg.run_arm(tmp_path, 'online-grpo', model, QUESTIONS, plan, **KW)
    assert out['match-layer64']['status'] == 'run'
    for name in ('intact', 'global64-reads', 'intact-repeat'):
        assert out[name]['units'] == ['written'] * 3
        for b_index, members in enumerate(plan):
            config = json.loads((cg.unit_dir(tmp_path, 'online-grpo', name, b_index) / 'config.json').read_text())
            assert config['question_ids'] == members
    again = cg.run_arm(tmp_path, 'online-grpo', model, QUESTIONS, plan, **KW)
    assert again['intact']['units'] == ['reused'] * 3
    rows = cg.load_condition(tmp_path, 'online-grpo', 'intact', plan)
    assert set(rows) == set(QUESTIONS)
    assert all({'generated_token_ids', 'text', 'stop_reason', 'flexible', 'reward'} <= set(r) for r in rows.values())
    repeat = cg.load_condition(tmp_path, 'online-grpo', 'intact-repeat', plan)
    assert all(rows[q]['generated_token_ids'] == repeat[q]['generated_token_ids'] for q in QUESTIONS)


def test_generation_refuses_to_start_without_a_freeze(tmp_path):
    with pytest.raises(ValueError):
        cg.run_arm(tmp_path, 'online-grpo', _model(), QUESTIONS, cg.batches(list(QUESTIONS), 2), **KW)


def test_a_changed_batch_membership_is_refused_on_resume(tmp_path):
    model = _model()
    _sealed_freeze(tmp_path)
    intact = cg.conditions(_freeze())[0]
    cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, cg.batches(list(QUESTIONS), 2), **KW)
    with pytest.raises(cg.GenerationError):
        cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, cg.batches(list(QUESTIONS)[::-1], 2), **KW)


def test_interventions_change_logits_only_where_expected_and_restore():
    from pilot_eval import attribution_ablation as ab
    model = _model()
    ids = torch.tensor([[3, 7, 11, 2, 9]])

    def logits():
        with torch.no_grad():
            return model(input_ids=ids).logits

    base = logits()
    before = ab.lora_hash(model)
    c = {c['name']: c for c in cg.conditions(_freeze())}
    for name in ('top10-adapters', 'global64-reads'):
        with cg.intervention(model, c[name]):
            assert not torch.equal(logits(), base)
        assert ab.lora_hash(model) == before
    with cg.intervention(model, c['intact']):
        assert torch.equal(logits(), base)
    with cg.intervention(model, cg.untuned_condition()):
        assert not torch.equal(logits(), base)
    assert torch.equal(logits(), base)
    # A read removal leaves layers it does not name untouched: layer 0 input is unaffected by a layer-1 edit.
    captured = {}
    handle = model.get_base_model().model.layers[0].register_forward_hook(lambda m, a, o: captured.__setitem__('h', o[0].clone()))
    logits()
    intact_h = captured['h']
    with ab.neuron_read_removal(model, {1: [0, 1]}):
        logits()
    handle.remove()
    assert torch.equal(captured['h'], intact_h)


# --- summaries ------------------------------------------------------------------------------------

def _row(reward, tokens=10, invalid=False, capped=False, ids=None):
    return {'reward': reward, 'n_tokens': tokens, 'invalid': invalid, 'capped': capped,
            'generated_token_ids': ids or [reward]}


def test_summary_hand_checked_with_skip_and_floor():
    ids = ['a', 'b', 'c', 'd']
    conds = cg.conditions(_freeze(control_available=False))
    results = {c['name']: {q: _row(1) for q in ids} for c in conds if c['status'] == 'run'}
    results['intact'] = {'a': _row(1), 'b': _row(1), 'c': _row(0), 'd': _row(1)}
    results['global64-reads'] = {'a': _row(0), 'b': _row(0, invalid=True), 'c': _row(1), 'd': _row(1, capped=True, tokens=30)}
    results['match-global64'] = {q: _row(1) for q in ids}
    results['intact-repeat'] = {'a': _row(1), 'b': _row(1), 'c': _row(0, ids=[9]), 'd': _row(1)}
    counts = np.ones((3, 4))  # every resample = the original sample, so the interval collapses to the estimate
    out = cg.summarize_arm(results, ids, conds, counts=counts)
    g = out['conditions']['global64-reads']
    assert g['accuracy'] == 0.5
    assert g['delta_pp_vs_intact'] == {'estimate': -25.0, 'p2.5': -25.0, 'p97.5': -25.0}
    assert g['percent_of_intact'] == pytest.approx(100 * 0.5 / 0.75)
    assert (g['gains'], g['losses']) == (1, 2) and g['mcnemar_exact_p'] == pytest.approx(1.0)
    assert (g['invalid'], g['capped'], g['tokens_mean'], g['tokens_median']) == (1, 1, 15.0, 10.0)
    assert g['target_minus_control_pp']['estimate'] == pytest.approx(-50.0)
    assert g['predicted_removal_effect']['pp'] == -1.0 and g['control_predicted_removal_effect']['pp'] == -0.1
    layer = out['conditions']['layer64-reads']
    assert layer['target_minus_control_pp'] is None and layer['target_minus_control_reason'] == 'matched-control-unavailable'
    assert out['conditions']['match-layer64'] == {'status': 'skipped', 'reason': 'matched-control-unavailable',
                                                  'deficient_strata': ['1'], 'deficient_targets': [9]}
    assert out['nondeterminism_floor']['questions_differing'] == 1
    assert out['nondeterminism_floor']['correctness_flips'] == 0
    assert out['labels'] == list(cg.LABELS)


def test_zero_intact_accuracy_leaves_the_ratio_undefined():
    ids = ['a', 'b']
    conds = [c for c in cg.conditions(_freeze()) if c['name'] in ('intact', 'top10-adapters')]
    results = {'intact': {q: _row(0) for q in ids}, 'top10-adapters': {q: _row(1) for q in ids}}
    out = cg.summarize_arm(results, ids, conds, counts=np.ones((2, 2)))
    assert out['conditions']['top10-adapters']['percent_of_intact'] is None


def test_a_plan_other_than_the_frozen_one_is_refused(tmp_path):
    model = _model()
    plan = cg.batches(list(QUESTIONS), 2)
    sel.seal_freeze(tmp_path, 'online-grpo', dict(sel.build_freeze('online-grpo', _freeze(), launch_record_sha256='L',
                                                                   hot={'layers': [1]}, split={}, inputs={}),
                                                  generation_plan=plan))
    with pytest.raises(cg.GenerationError, match='frozen generation plan'):
        cg.run_arm(tmp_path, 'online-grpo', model, QUESTIONS, cg.batches(list(QUESTIONS), 3), **KW)


def test_an_unrestored_adapter_stops_before_the_unit_is_sealed(tmp_path, monkeypatch):
    from contextlib import contextmanager
    model = _model()
    plan = cg.batches(list(QUESTIONS), 2)

    @contextmanager
    def leaky(model_, condition):  # zeroes a lora_B and never puts it back
        module = model_.get_base_model().model.layers[0].mlp.down_proj
        module.lora_B['default'].weight.data.zero_()
        yield

    monkeypatch.setattr(cg, 'intervention', leaky)
    condition = cg.conditions(_freeze())[1]
    with pytest.raises(cg.GenerationError, match='adapter not restored'):
        cg.run_condition(tmp_path, 'online-grpo', condition, model, QUESTIONS, plan, **KW)
    assert not (cg.unit_dir(tmp_path, 'online-grpo', condition['name'], 0) / 'complete.json').exists()


def test_a_sealed_unit_is_not_reused_after_its_cap_or_adapter_changes(tmp_path):
    """Review P1: reuse is bound to the unit's identity, not only to its seal and question IDs."""
    model = _model()
    plan = cg.batches(list(QUESTIONS), 2)
    intact = cg.conditions(_freeze())[0]
    cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, plan, **KW)
    assert cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, plan, **KW)['units'] == ['reused'] * 3
    with pytest.raises(cg.GenerationError, match='identity'):
        cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, plan, **dict(KW, cap=1))
    other = {q: dict(v, gold='#### 9') for q, v in QUESTIONS.items()}
    with pytest.raises(cg.GenerationError, match='identity'):
        cg.run_condition(tmp_path, 'online-grpo', intact, model, other, plan, **KW)
    with torch.no_grad():
        model.get_base_model().model.layers[0].mlp.down_proj.lora_B['default'].weight.add_(0.01)
    with pytest.raises(cg.GenerationError, match='identity'):
        cg.run_condition(tmp_path, 'online-grpo', intact, model, QUESTIONS, plan, **KW)
