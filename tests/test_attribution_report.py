"""Ticket 10: report rules on planted fixtures (gate, primaries 1-3, Holm, concordance, rendering). CPU only."""
import json

import numpy as np
import pytest

from pilot_eval import attribution_report as ar

SETTINGS = dict(resamples=2000, seed=1234, alpha=0.05, margin=0.25, sparse_lower=0.70, dense_upper=0.30,
                gate_singles_min=4, resolution_floor=0.0)


def idx(n, resamples=2000, seed=1234):
    return ar.bootstrap_indices(n, resamples=resamples, seed=seed)


def test_bootstrap_indices_are_seeded_and_shared():
    a, b = idx(50), idx(50)
    assert a.shape == (2000, 50) and np.array_equal(a, b)
    with pytest.raises(ValueError):
        ar.bootstrap_indices(5, resamples=None, seed=1)


def test_interval_and_two_sided_p_with_nan_rows():
    values = np.r_[np.full(40, 1.0) + np.linspace(-0.1, 0.1, 40), [np.nan] * 10]
    out = ar.mean_interval(values, idx(50))
    assert out['estimate'] == pytest.approx(1.0) and out['interval_95'][0] > 0.9 and out['p_two_sided'] < 1e-3
    assert out['n_defined'] == 40 and out['n_excluded'] == 10
    null = ar.mean_interval(np.linspace(-1, 1, 50), idx(50))
    assert null['p_two_sided'] > 0.5
    empty = ar.mean_interval(np.full(5, np.nan), idx(5))
    assert empty['estimate'] is None and empty['reason'] == 'no-defined-values'


def test_holm():
    out = ar.holm({'a': 0.01, 'b': 0.04, 'c': 0.03}, alpha=0.05)
    assert out['a']['adjusted'] == pytest.approx(0.03) and out['a']['reject']
    assert out['c']['adjusted'] == pytest.approx(0.06) and not out['c']['reject']
    assert out['b']['adjusted'] == pytest.approx(0.06) and not out['b']['reject']


def test_spearman_average_ranks_for_ties():
    assert ar.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert ar.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ar.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def gate_case(pred_sign=1.0, n=60, seed=0):
    rng = np.random.default_rng(seed)
    effects = [0.5, 0.4, 0.3, 0.2, 0.1]
    stage = {f'top{r + 1}-trace': dict(delta_trace=e + rng.normal(0, 0.05, n)) for r, e in enumerate(effects)}
    stage.update({f'match{r + 1}-trace': dict(delta_trace=rng.normal(0, 0.05, n)) for r in range(5)})
    stage['top10-trace'] = dict(delta_trace=1.2 + rng.normal(0, 0.05, n))
    stage['match10-trace'] = dict(delta_trace=rng.normal(0, 0.05, n))
    freeze = dict(top5=[10, 11, 12, 13, 14], top10=list(range(10, 20)),
                  predicted={str(i): pred_sign * (0.5 - 0.1 * k) for k, i in enumerate(range(10, 20))},
                  controls=dict(per_target={str(i): 30 + i for i in range(10, 20)}, group=list(range(40, 50))),
                  control_predicted={str(30 + i): 0.0 for i in range(10, 20)})
    return stage, freeze


def test_gate_passes_with_correct_signs_and_trips_on_a_bad_estimator():
    stage, freeze = gate_case()
    good = ar.validity_gate(stage, freeze, 'trace', idx(60), singles_min=4)
    assert good['pass'] is True and good['top_singles_sign_agree'] == 5 and good['group_sign_agree'] is True
    assert 'limited check on selected removals' in good['label']
    assert {'spearman', 'sign_agreement', 'median_abs_error', 'slope'} <= set(good['diagnostics'])
    stage, freeze = gate_case(pred_sign=-1.0)
    bad = ar.validity_gate(stage, freeze, 'trace', idx(60), singles_min=4)
    assert bad['pass'] is False and bad['top_singles_sign_agree'] == 0


def p1_case(d_s=-0.5, d_l=0.0, d_a=1.0, noise_a=0.05, n=80, seed=1):
    rng = np.random.default_rng(seed)
    return {'top10-stop': dict(delta_stop=d_s + rng.normal(0, 0.05, n), delta_trace=d_l + rng.normal(0, 0.05, n)),
            'match10-stop': dict(delta_stop=rng.normal(0, 0.05, n), delta_trace=rng.normal(0, 0.05, n)),
            'top10-trace': dict(delta_trace=d_a + rng.normal(0, noise_a, n)),
            'match10-trace': dict(delta_trace=rng.normal(0, 0.05, n))}


def test_primary1_supported_contradicted_and_inconclusive():
    supported = ar.primary1(p1_case(), idx(80), margin=0.25)
    assert supported['outcome'] == 'supported' and supported['r_upper_95'] < 0.25
    contradicted = ar.primary1(p1_case(d_l=0.8), idx(80), margin=0.25)
    assert contradicted['outcome'] == 'contradicted' and contradicted['r_lower_95'] >= 0.25
    no_reference = ar.primary1(p1_case(d_a=0.0), idx(80), margin=0.25)
    assert no_reference['outcome'] == 'inconclusive' and no_reference['reasons'][0].startswith('reference')
    no_stop = ar.primary1(p1_case(d_s=0.0), idx(80), margin=0.25)
    assert no_stop['outcome'] == 'inconclusive'


def test_primary1_noisy_reference_point_rule_would_claim_equivalence_joint_rule_does_not():
    case = p1_case(d_l=0.04, d_a=0.35, noise_a=1.2, n=80, seed=7)
    out = ar.primary1(case, idx(80), margin=0.25)
    assert out['r_point'] < 0.25  # a point-estimate rule would say "equivalent"
    assert out['outcome'] == 'inconclusive'


def test_primary1_unavailable_matched_group_is_inconclusive_not_an_error():
    case = p1_case()
    del case['match10-stop']
    assert ar.primary1(case, idx(80), margin=0.25)['outcome'] == 'inconclusive'


def r_case(full_minus_removed=1.0, attr_retained=0.9, size_retained=0.5, n=80, noise=0.05, seed=2):
    rng = np.random.default_rng(seed)
    removed = -full_minus_removed  # delta of the removed condition relative to full
    stage = {'down-full-trace': dict(delta_trace=removed + rng.normal(0, noise, n))}
    for name, kept in [('attr', attr_retained), ('size', size_retained), ('rand1', 0.1), ('rand2', 0.0),
                       ('rand3', 0.2)]:
        stage[f'read64-{name}-trace'] = dict(delta_trace=removed * (1 - kept) + rng.normal(0, noise, n))
    return stage


def test_retained_effect_and_primary3_contrasts():
    out = ar.primary3(r_case(), dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3, resolution_floor=0.0)
    assert out['R']['attr']['estimate'] == pytest.approx(0.9, abs=0.05)
    assert out['classification'] == 'sparse' and out['classification_is_descriptive'] is True
    assert out['tests']['attr_minus_size']['interval_95'][0] > 0
    assert out['tests']['attr_minus_mean_random']['interval_95'][0] > 0


def test_r_point_above_threshold_but_lower_bound_below_is_inconclusive():
    out = ar.primary3(r_case(attr_retained=0.75, noise=0.6, seed=3), dict(index=7, layer=1), 'trace', idx(80),
                      sparse_lower=0.7, dense_upper=0.3, resolution_floor=0.0)
    assert out['R']['attr']['estimate'] > 0.70 and out['R']['attr']['interval_95'][0] < 0.70
    assert out['classification'] == 'inconclusive'


def test_r_unavailable_when_full_removal_effect_interval_includes_zero():
    out = ar.primary3(r_case(full_minus_removed=0.0), dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7,
                      dense_upper=0.3, resolution_floor=0.0)
    assert out['R']['attr']['estimate'] is None and out['R']['attr']['reason'] == 'denominator-interval-includes-0'
    assert out['classification'] == 'unavailable' and out['tests']['attr_minus_size']['estimate'] is None


def test_primary3_uses_top_single_when_down_proj_was_a_single():
    stage = r_case()
    stage['top2-trace'] = stage.pop('down-full-trace')
    out = ar.primary3(stage, dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3, resolution_floor=0.0,
                      top5=[3, 7])
    assert out['R']['attr']['estimate'] == pytest.approx(0.9, abs=0.05)


def test_dense_requires_upper_bound_below_threshold_in_every_arm():
    arms = {'a': r_case(attr_retained=0.05), 'b': r_case(attr_retained=0.1, seed=5)}
    results = {arm: ar.primary3(s, dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3, resolution_floor=0.0)
               for arm, s in arms.items()}
    assert ar.dense_everywhere(results) is True
    results['c'] = ar.primary3(r_case(attr_retained=0.5), dict(index=7, layer=1), 'trace', idx(80),
                               sparse_lower=0.7, dense_upper=0.3, resolution_floor=0.0)
    assert ar.dense_everywhere(results) is False


def discovery(seed=0, n=40, adapters=14):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=adapters)
    noise = lambda s: rng.normal(0, s, (n, adapters))  # noqa: E731
    return {'online-grpo': base + noise(0.3), 'offline-grpo': base + noise(0.3),
            'gold-sft': rng.normal(size=adapters) + noise(0.3),
            'sampling-sft-k0': base * 0.9 + noise(0.3), 'sampling-sft-k2': base * 0.5 + rng.normal(size=adapters)
            + noise(0.3), 'sampling-sft-k5': rng.normal(size=adapters) + noise(0.3)}


def test_primary2_paired_correlation_differences_and_exclusions():
    out = ar.primary2(discovery(), idx(40), excluded_arms=[])
    assert set(out['contrasts']) == {'a', 'b', 'c', 'd'}
    assert out['contrasts']['a']['estimate'] > 0 and 'p_two_sided' in out['contrasts']['a']
    excluded = ar.primary2(discovery(), idx(40), excluded_arms=['offline-grpo'])
    assert excluded['contrasts']['a']['estimate'] is None and excluded['excluded_arms'] == ['offline-grpo']


def test_family_holm_and_gate_failure_labels():
    gates = {('gold-sft', 'trace'): dict(**{'pass': False}), ('online-grpo', 'trace'): dict(**{'pass': True}),
             ('offline-grpo', 'trace'): dict(**{'pass': True})}
    p2 = dict(contrasts={k: dict(estimate=0.2, p_two_sided=0.01, arms=arms) for k, arms in
                         [('a', ['online-grpo', 'offline-grpo', 'gold-sft']), ('b', ['online-grpo', 'offline-grpo']),
                          ('c', ['sampling-sft-k0', 'sampling-sft-k2']), ('d', ['sampling-sft-k2', 'sampling-sft-k5'])]})
    labels = ar.gate_consequences(gates, p2, 'trace')
    assert labels['primary2']['a'] == 'unvalidated-exploratory' and labels['primary2']['b'] == 'confirmatory'
    assert labels['primary2']['c'] == 'unvalidated-exploratory'  # no gate recorded counts as not validated
    assert labels['primary1'] == 'rests only on directly measured contrasts'
    assert 'not causal sparsity' in labels['primary3']


def test_concordance_lists_disagreements():
    rng = np.random.default_rng(4)
    c = {'top10-trace': dict(delta_trace=-0.5 + rng.normal(0, 0.05, 50), delta_stop=0.3 + rng.normal(0, 0.05, 50))}
    items = 30
    full = [dict(id=f'i{k}', correct=True, tokens=200) for k in range(items)]
    cond = [dict(id=f'i{k}', correct=k % 3 != 0, tokens=150) for k in range(items)]
    out = ar.concordance(c, {'top10-trace': (full, cond)}, idx(50), idx(30, seed=99))
    row = out['conditions']['top10-trace']
    assert row['trace_vs_accuracy']['agree'] is True and row['stop_vs_length']['agree'] is True
    cond_up = [dict(id=f'i{k}', correct=True, tokens=260) for k in range(items)]
    out = ar.concordance(c, {'top10-trace': (full, cond_up)}, idx(50), idx(30, seed=99))
    assert out['disagreements'] and out['label'].startswith('exploratory')


def test_concentration_and_cancellation_unavailable_below_floor():
    attr = np.array([5.0, 1.0, -1.0, 0.5, 0.0])
    out = ar.concentration(attr, null_orders=200, null_seed=7, resolution_floor=1e-6)
    assert out['neurons_to_50'] == 1 and out['neurons_to_90'] == 3
    assert out['cancellation'] == pytest.approx(7.5 / 5.5)
    assert len(out['null']['mean_fraction']) == 5
    tiny = ar.concentration(np.full(5, 1e-9), null_orders=10, null_seed=7, resolution_floor=1e-6)
    assert tiny['available'] is False and tiny['neurons_to_50'] is None


def test_build_and_render_report_end_to_end(tmp_path):
    stage, freeze = gate_case()
    stage.update(p1_case())
    stage.update(r_case())
    inputs = dict(
        settings=SETTINGS, scope='base-correct', provenance={'online-grpo': dict(status='partial')},
        discovery=dict(trace=discovery(), stop=discovery(seed=1)), unresolved_arms=[],
        freeze={arm: dict(trace=dict(freeze, down_proj=dict(index=7, layer=1)),
                          stop=dict(freeze, down_proj=dict(index=7, layer=1)))
                for arm in ['gold-sft', 'online-grpo']},
        stage_c={arm: dict(trace=stage, stop={k.replace('trace', 'stop'): dict(delta_stop=v.get('delta_trace'),
                                                                                delta_trace=v.get('delta_trace'))
                                              for k, v in stage.items()} | p1_case())
                 for arm in ['gold-sft', 'online-grpo']},
        stage_d={}, neuron={}, compute=dict(attribution=dict(busy=None)), a_vs_a0=None)
    report = ar.build(inputs)
    assert report['primary1']['outcome'] in ('supported', 'contradicted', 'inconclusive')
    assert set(report['holm']) == {'trace', 'stop'}
    md = ar.render(report)
    assert 'questions the untuned model solves' in md and 'limited check on selected removals' in md
    assert 'partially independent' in md and 'no overlap with available sources' in md
    json.dumps(report, default=str)


def test_primary2_support_needs_both_differences_holm_rejected_and_positive():
    contrasts = {k: dict(estimate=0.3, p_two_sided=0.001) for k in 'abcd'}
    holm = {f'primary2.{k}': dict(reject=True) for k in 'abcd'}
    assert ar.primary2_support(contrasts, holm) == dict(grpo_sharing='supported', k_ordering='supported',
                                                        rule='both differences > 0 and rejected by Holm')
    holm['primary2.b'] = dict(reject=False)
    contrasts['d'] = dict(estimate=-0.3, p_two_sided=0.001)
    out = ar.primary2_support(contrasts, holm)
    assert out['grpo_sharing'] == 'not supported' and out['k_ordering'] == 'not supported'
    contrasts['c'] = dict(estimate=None, p_two_sided=None)
    assert ar.primary2_support(contrasts, holm)['k_ordering'] == 'unavailable'


def test_r4_10_r_unavailable_when_the_full_removal_effect_is_below_the_resolution_floor():
    stage = r_case(full_minus_removed=1e-12, noise=1e-15)
    out = ar.primary3(stage, dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3,
                      resolution_floor=1e-3)
    assert out['classification'] == 'unavailable'
    assert out['R']['attr']['reason'] == 'denominator-below-resolution-floor'
    ok = ar.primary3(r_case(), dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3,
                     resolution_floor=1e-3)
    assert ok['classification'] == 'sparse'
    with pytest.raises(TypeError):
        ar.primary3(r_case(), dict(index=7, layer=1), 'trace', idx(80), sparse_lower=0.7, dense_upper=0.3)


def test_r4_10_primary1_headline_is_the_family_adjusted_inference():
    raw = dict(outcome='supported', p=0.0056)
    holm = {'trace': {'primary1': dict(adjusted=0.0952, reject=False)},
            'stop': {'primary1': dict(adjusted=0.03, reject=True)}}
    out = ar.primary1_adjusted(raw, holm)
    assert out['outcome'] == 'inconclusive' and out['outcome_unadjusted'] == 'supported'
    assert out['holm'] == {'trace': dict(adjusted=0.0952, reject=False), 'stop': dict(adjusted=0.03, reject=True)}
    assert 'not supported after Holm' in out['adjustment_note']
    both = ar.primary1_adjusted(raw, {m: {'primary1': dict(adjusted=0.01, reject=True)} for m in ('trace', 'stop')})
    assert both['outcome'] == 'supported'
    for raw_outcome in ('contradicted', 'inconclusive'):
        assert ar.primary1_adjusted(dict(outcome=raw_outcome, p=0.5), holm)['outcome'] == raw_outcome
