"""Ticket 04: uncertainty and stability (descriptive; never a gate)."""
import numpy as np
import pytest

from pilot_eval import c_attribution as ca
from pilot_eval import c_attribution_stability as cs


def _sums(signal, noise, *, q=150, a=40, layers=(0, 1, 2), width=200, seed=0):
    """QuestionSums whose per-question S = signal + noise; halves built from two noisy draws of S."""
    rng = np.random.default_rng(seed)
    ids = [f'gsm8k:test:{i}' for i in range(q)]
    sums = ca.QuestionSums(ids, n_adapters=a, layers=layers, width=width)
    base_a = rng.normal(size=a) * signal
    base_r = rng.normal(size=(len(layers), width)) * signal
    sums.adapters = base_a + rng.normal(size=(q, a)) * noise
    sums.reads = base_r + rng.normal(size=(q, len(layers), width)) * noise
    sums.adapter_halves = np.stack([q * base_a / 2 + rng.normal(size=a) * noise * np.sqrt(q)
                                    for _ in range(2)])
    sums.read_halves = np.stack([q * base_r / 2 + rng.normal(size=(len(layers), width)) * noise * np.sqrt(q)
                                 for _ in range(2)])
    return sums


def test_question_split_is_seeded_balanced_and_shared():
    ids = [f'q{i}' for i in range(150)]
    s = cs.question_split(ids)
    assert s['seed'] == 20261118 and len(s['A']) == len(s['B']) == 75
    assert sorted(s['A'] + s['B']) == sorted(ids)
    assert cs.question_split(ids) == s
    assert cs.question_split(ids, seed=1) != s


def test_spearman_matches_known_values_and_handles_ties():
    assert cs.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cs.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Hand value with ties: ranks a = [1, 2.5, 2.5, 4], b = [1, 2, 3, 4]
    ra = np.array([1, 2.5, 2.5, 4]) - 2.5
    rb = np.array([1, 2, 3, 4]) - 2.5
    expected = (ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    assert cs.spearman([0, 1, 1, 2], [0, 1, 2, 3]) == pytest.approx(expected)


def test_overlap_and_sign_agreement():
    a = np.array([5.0, 4.0, -9.0, 1.0, 0.5])
    b = np.array([4.0, -1.0, -8.0, 3.0, 0.1])
    assert cs.overlap(a, b, 2) == 1                   # top-2 by dJdλ: {0, 1} and {0, 3}
    agree = cs.sign_agreement(a, b, 2)               # signed top-2: {0, 1} and {0, 3}; index 1 disagrees
    assert agree == {'share': pytest.approx(2 / 3), 'n': 3}
    assert cs.sign_agreement(a, b, 3)['share'] == pytest.approx(3 / 4)  # union {0,1,3,4}: index 1 disagrees


def test_sign_agreement_uses_the_signed_top_sets_review_counterexample():
    # Review P2: by |dJdλ| the large negatives {2, 3} dominate and agree 100%; the helpful sets {0,1} disagree.
    a, b = np.array([4.0, 3.0, -100.0, -90.0]), np.array([-1.0, -2.0, -100.0, -90.0])
    assert cs.sign_agreement(a, b, 2)['share'] == 0.0


def test_planted_signal_agrees_and_noise_does_not():
    planted = cs.stability(_sums(signal=1.0, noise=0.5), hot_layers=[1])
    noise = cs.stability(_sums(signal=0.0, noise=1.0, seed=1), hot_layers=[1])
    for result, high in ((planted, True), (noise, False)):
        q = result['question_split']
        rho = q['adapters']['spearman']
        assert (rho > 0.9) if high else (abs(rho) < 0.4)
        mean = q['reads']['per_layer_mean']['overlap']
        assert (mean > 40) if high else (mean < 25)
        d = result['draw_split']
        assert d['label'] == cs.DRAW_SPLIT_LABEL
        assert (d['adapters']['spearman'] > 0.9) if high else (abs(d['adapters']['spearman']) < 0.4)
    assert set(planted['question_split']['reads']['hot_layers']) == {1}
    assert sum(planted['question_split']['reads']['global']['layer_distribution']['A'].values()) == 64


def test_streamed_bootstrap_equals_unchunked_and_is_deterministic():
    rng = np.random.default_rng(4)
    S = rng.normal(size=(30, 3, 50))
    counts = cs.resample_counts(30, B=500, seed=9)
    assert np.all(counts.sum(1) == 30)
    np.testing.assert_array_equal(counts, cs.resample_counts(30, B=500, seed=9))
    streamed = cs.bootstrap_dJdlambda(S, counts, chunk=7)
    whole = -(counts @ S.reshape(30, -1)) / 30
    for p in cs.PERCENTILES:
        np.testing.assert_allclose(streamed[f'p{p}'].ravel(), np.percentile(whole, p, axis=0), rtol=1e-12)
    # Same chunking twice is bit-exact; across chunk sizes the BLAS summation order
    # may differ, so compare to the same rtol as the percentile check above.
    np.testing.assert_array_equal(streamed['p2.5'], cs.bootstrap_dJdlambda(S, counts, chunk=7)['p2.5'])
    np.testing.assert_allclose(streamed['p2.5'], cs.bootstrap_dJdlambda(S, counts, chunk=1000)['p2.5'], rtol=1e-12)


def test_bootstrap_on_S_equals_recomputed_leave_one_out_inside_a_resample():
    rng = np.random.default_rng(7)
    rewards = {f'q{i}': list(rng.integers(0, 2, 8)) for i in range(6)}
    attr = {q: rng.normal(size=(8, 3)) for q in rewards}
    sums = ca.QuestionSums(list(rewards), n_adapters=3)
    for q in rewards:
        req = ca.reduction_request([{'question_id': q, 'draw_index': d} for d in range(8)], rewards)
        sums.add_adapters(req['question_ids'], range(8), attr[q], req['coefficients'], req['half_coefficients'])
    counts = np.zeros((1, 6))
    counts[0] = [2, 0, 1, 0, 3, 0]
    via_counts = -(counts @ sums.adapters) / 6
    picked = ['q0', 'q0', 'q2', 'q4', 'q4', 'q4']
    w = ca.loo_weights({f'{q}#{i}': rewards[q] for i, q in enumerate(picked)})
    direct = -sum((w['weights'][f'{q}#{i}'][:, None] * attr[q]).sum(0) for i, q in enumerate(picked))
    np.testing.assert_allclose(via_counts[0], direct, rtol=1e-13)


def test_interval_coverage_is_near_nominal_in_a_small_simulation():
    rng = np.random.default_rng(11)
    covered, trials, mu = 0, 200, 0.3
    counts = cs.resample_counts(150, B=1000, seed=3)
    for _ in range(trials):
        S = (-mu + rng.normal(size=(150, 1)))  # dJdλ = -mean(S) has true value mu
        iv = cs.bootstrap_dJdlambda(S, counts)
        covered += iv['p2.5'][0] <= mu <= iv['p97.5'][0]
    assert 0.88 <= covered / trials <= 0.99


def test_adapter_bootstrap_pairs_arms_and_reports_ranks():
    rng = np.random.default_rng(2)
    ids = [f'q{i}' for i in range(40)]
    a, b = rng.normal(size=(40, 5)), rng.normal(size=(40, 5))
    a[:, 0] -= 3.0  # adapter 0 clearly most helpful in arm a
    out = cs.adapter_bootstrap({'online': (ids, a), 'offline': (ids, b)}, B=300)
    assert out['label'] == cs.INTERVAL_LABEL
    assert out['arms']['online']['rank']['p97.5'][0] == 1
    counts = cs.resample_counts(40, 300, cs.BOOTSTRAP_SEED)
    np.testing.assert_allclose(out['paired']['dJdlambda']['p2.5'],
                               cs.bootstrap_dJdlambda(a - b, counts)['p2.5'])
    with pytest.raises(ValueError):
        cs.adapter_bootstrap({'online': (ids, a), 'offline': (ids[::-1], b)}, B=10)


def test_neuron_bootstrap_tiers_and_benchmark():
    rng = np.random.default_rng(5)
    S = rng.normal(size=(20, 2, 30))
    counts = cs.resample_counts(20, B=200)
    out = cs.neuron_bootstrap(S, counts, selected=[(0, 3), (1, 7)], tiers=('selected', 'per_layer', 'all'),
                              benchmark=True)
    np.testing.assert_allclose(out['selected']['p2.5'], [out['all']['p2.5'][0, 3], out['all']['p2.5'][1, 7]])
    assert out['per_layer']['p97.5'].shape == (2,)
    for tier in ('selected', 'per_layer', 'all'):
        bench = out['benchmark']['tiers'][tier]
        assert bench['seconds'] >= 0 and bench['tracemalloc_peak_bytes'] > 0 and bench['max_rss_bytes'] > 0


def test_descriptives():
    rewards = {'a': [1, 1], 'b': [0, 1], 'c': [0, 0]}
    d = np.array([3.0, -1.0, 0.5])
    out = cs.descriptives(rewards, d, k=1)
    assert out['J_bar'] == pytest.approx(0.5) and out['informative_questions'] == 1
    assert out['top1_adapter_share_of_gross'] == pytest.approx(3 / 4.5)
