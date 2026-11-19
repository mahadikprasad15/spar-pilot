"""Ticket 03: leave-one-out weights, aggregation, totals and signs."""
import itertools

import numpy as np
import pytest

from pilot_eval import c_attribution as ca


# --- weights ------------------------------------------------------------------------------------

def test_hand_computed_weights():
    # R = [1, 0, 0, 1]: b = [1/3, 2/3, 2/3, 1/3]; R - b = [2/3, -2/3, -2/3, 2/3]; / n = 4
    coef = ca.loo_coefficients([1, 0, 0, 1])
    np.testing.assert_allclose(coef, [1 / 6, -1 / 6, -1 / 6, 1 / 6])
    w = ca.loo_weights({'a': [1, 0, 0, 1], 'b': [1, 1, 1, 1], 'c': [0, 1, 0]})
    assert w['Q'] == 3
    np.testing.assert_allclose(w['weights']['a'], coef / 3)
    # R = [0, 1, 0]: b = [1/2, 0, 1/2]; (R - b)/3 = [-1/6, 1/3, -1/6]
    np.testing.assert_allclose(w['coefficients']['c'], [-1 / 6, 1 / 3, -1 / 6])
    for value in w['weights'].values():
        assert abs(value.sum()) < 1e-15


@pytest.mark.parametrize('rewards', [[1] * 8, [0] * 8, [1, 1]])
def test_all_same_groups_are_exactly_zero(rewards):
    assert np.all(ca.loo_coefficients(rewards) == 0.0)


def test_one_draw_is_refused():
    with pytest.raises(ca.AggregationError):
        ca.loo_coefficients([1])
    with pytest.raises(ca.AggregationError):
        ca.loo_weights({'a': [1, 0], 'b': [0]})


def test_draw_half_coefficients_use_within_half_baselines():
    r = [1, 0, 0, 0, 1, 1, 1, 0]
    h = ca.half_coefficients(r, range(8))
    # Half A: R = [1, 0, 0, 0]: b = [0, 1/3, 1/3, 1/3]; (R - b)/4 = [1/4, -1/12, -1/12, -1/12]
    np.testing.assert_allclose(h[:4, 0], [1 / 4, -1 / 12, -1 / 12, -1 / 12])
    assert np.all(h[:4, 1] == 0) and np.all(h[4:, 0] == 0)
    # Half B: R = [1, 1, 1, 0]: b = [2/3, 2/3, 2/3, 1]; (R - b)/4 = [1/12, 1/12, 1/12, -1/4]
    np.testing.assert_allclose(h[4:, 1], [1 / 12, 1 / 12, 1 / 12, -1 / 4])
    with pytest.raises(ca.AggregationError):
        ca.half_coefficients(r[:7], range(7))


def test_reduction_request_matches_weights_and_checks_rewards():
    rewards = {'q': [1, 0, 0, 0, 1, 1, 1, 0]}
    rows = [{'question_id': 'q', 'draw_index': d, 'reward': rewards['q'][d]} for d in (6, 1, 3)]
    req = ca.reduction_request(rows, rewards)
    full = ca.loo_coefficients(rewards['q'])
    np.testing.assert_allclose(req['coefficients'], full[[6, 1, 3]])
    np.testing.assert_allclose(req['half_coefficients'], ca.half_coefficients(rewards['q'], range(8))[[6, 1, 3]])
    rows[0]['reward'] = 0
    with pytest.raises(ca.AggregationError):
        ca.reduction_request(rows, rewards)


# --- pre-aggregation ------------------------------------------------------------------------------

def _synthetic(seed=0, q=5, n=8, c=7):
    rng = np.random.default_rng(seed)
    rewards = {f'q{i}': list(rng.integers(0, 2, n)) for i in range(q)}
    rewards['q0'] = [1] * n  # an all-same group stays in Q
    attr = {qid: rng.normal(size=(n, c)) for qid in rewards}
    return rewards, attr


def test_pre_aggregated_equals_direct_aggregation_with_split_batches():
    rewards, attr = _synthetic()
    w = ca.loo_weights(rewards)
    direct = sum((w['weights'][q][:, None] * attr[q]).sum(0) for q in rewards)
    sums = ca.QuestionSums(list(rewards), n_adapters=7)
    flat = [(q, d) for q in rewards for d in range(8)]
    for start in range(0, len(flat), 5):  # micro-batches that split questions
        batch = flat[start:start + 5]
        rows = [{'question_id': q, 'draw_index': d} for q, d in batch]
        req = ca.reduction_request(rows, rewards)
        sums.add_adapters(req['question_ids'], [d for _, d in batch], np.stack([attr[q][d] for q, d in batch]),
                          req['coefficients'], req['half_coefficients'])
    sums.complete(len(flat))
    agg = ca.aggregate(sums.adapters)
    np.testing.assert_allclose(agg['removal_J'], direct, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(agg['dJdlambda'], -direct, rtol=1e-13, atol=1e-15)
    assert np.all(sums.adapters[0] == 0)
    # Draw halves recomputed independently: b from the other 3 draws of the half, / 4.
    expected = np.zeros((2, 7))
    for q, r in rewards.items():
        for h, members in enumerate(ca.HALVES):
            rr = np.array([r[d] for d in members], float)
            coef = (rr - (rr.sum() - rr) / 3) / 4
            expected[h] += (coef[:, None] * attr[q][list(members)]).sum(0)
    np.testing.assert_allclose(sums.adapter_halves, expected, rtol=1e-13, atol=1e-15)
    with pytest.raises(ca.AggregationError):
        sums.add_adapters(['q1'], [0], attr['q1'][:1], [0.0], [[0.0, 0.0]])


def test_question_bootstrap_on_S_equals_recomputing_weights_in_the_resample():
    rewards, attr = _synthetic(seed=3)
    sums = ca.QuestionSums(list(rewards), n_adapters=7)
    for q in rewards:
        rows = [{'question_id': q, 'draw_index': d} for d in range(8)]
        req = ca.reduction_request(rows, rewards)
        sums.add_adapters(req['question_ids'], range(8), attr[q], req['coefficients'], req['half_coefficients'])
    pick = [2, 2, 0, 4, 1]
    resampled = ca.aggregate(sums.adapters[pick])['dJdlambda']
    keys = [list(rewards)[i] for i in pick]
    w = ca.loo_weights({f'{k}#{i}': rewards[k] for i, k in enumerate(keys)})
    direct = -sum((w['weights'][f'{k}#{i}'][:, None] * attr[k]).sum(0) for i, k in enumerate(keys))
    np.testing.assert_allclose(resampled, direct, rtol=1e-13, atol=1e-15)


# --- exact enumeration: the estimator's expectation is the exact gradient ----------------------------

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Toy:
    """2-token vocabulary, length-2 responses; logits theta + lambda * u at each position."""

    def __init__(self, seed, reward):
        rng = np.random.default_rng(seed)
        self.t1, self.u1 = rng.normal(size=2), rng.normal(size=2)
        self.t2, self.u2 = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        self.reward = reward

    def probs(self, lam=1.0):
        p1 = _softmax(self.t1 + lam * self.u1)
        return {(a, b): p1[a] * _softmax(self.t2[a] + lam * self.u2[a])[b] for a in (0, 1) for b in (0, 1)}

    def dlogp(self, y, lam=1.0):
        a, b = y
        p1 = _softmax(self.t1 + lam * self.u1)
        p2 = _softmax(self.t2[a] + lam * self.u2[a])
        return (self.u1[a] - p1 @ self.u1) + (self.u2[a][b] - p2 @ self.u2[a])

    def J(self, lam=1.0):
        return sum(p * self.reward(y) for y, p in self.probs(lam).items())


def _expected_estimate(toys, n, variant='estimator'):
    """E over all n-draw joint samples per question of the estimator's dJdlambda (exact enumeration)."""
    total = 0.0
    for toy in toys:
        probs = toy.probs()
        expectation = 0.0
        for draws in itertools.product(list(probs), repeat=n):
            p = np.prod([probs[y] for y in draws])
            r = np.array([toy.reward(y) for y in draws], float)
            attr = np.array([-toy.dlogp(y) for y in draws])   # attr = -dM/dlambda, M = sum log p
            if variant == 'token-mean':
                attr = attr / 2
            if variant == 'std-normalized':
                std = r.std()
                coef = np.zeros(n) if std == 0 else (r - r.mean()) / std / n
            else:
                coef = ca.loo_coefficients(r)
            removal = float((coef * attr).sum())
            d = removal if variant == 'sign-flip' else -removal
            expectation += p * d
        total += expectation
    return total / len(toys)


TOYS = [Toy(1, lambda y: float(y[1] == 1)), Toy(2, lambda y: float(y == (0, 0) or y == (1, 1)))]


def _exact_gradient(toys, eps=1e-6):
    return np.mean([(t.J(1 + eps) - t.J(1 - eps)) / (2 * eps) for t in toys])


def test_exact_enumeration_expectation_equals_exact_gradient():
    exact = _exact_gradient(TOYS)
    analytic = np.mean([sum(p * t.reward(y) * t.dlogp(y) for y, p in t.probs().items()) for t in TOYS])
    np.testing.assert_allclose(analytic, exact, rtol=1e-7)
    for n in (2, 3, 4):
        np.testing.assert_allclose(_expected_estimate(TOYS, n), analytic, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('variant', ['sign-flip', 'std-normalized', 'token-mean'])
def test_mutated_estimators_miss_the_exact_gradient(variant):
    exact = np.mean([sum(p * t.reward(y) * t.dlogp(y) for y, p in t.probs().items()) for t in TOYS])
    assert abs(_expected_estimate(TOYS, 3, variant) - exact) > 1e-3 * max(abs(exact), 1e-3)


def test_a_helpful_adapter_has_positive_dJdlambda_and_negative_removal():
    toy = Toy(5, lambda y: float(y == (1, 1)))
    toy.u1, toy.u2 = np.array([-1.0, 1.0]), np.array([[0.0, 0.0], [-1.0, 1.0]])  # lambda pushes towards (1, 1)
    assert toy.J(1.1) > toy.J(1.0)
    expected = _expected_estimate([toy], 3)
    assert expected > 0
    # One concrete group: draws (1,1) correct, others wrong.
    draws = [(1, 1), (0, 1), (1, 0)]
    r = [toy.reward(y) for y in draws]
    attr = np.array([[-toy.dlogp(y)] for y in draws])
    sums = ca.QuestionSums(['q'], n_adapters=1)
    sums.add_adapters(['q'] * 3, range(3), attr, ca.loo_coefficients(r), np.zeros((3, 2)))
    agg = ca.aggregate(sums.adapters)
    assert agg['dJdlambda'][0] > 0 and agg['removal_J'][0] < 0


# --- totals, shares, predictions, identity --------------------------------------------------------

def test_cancellation_keeps_gross_shares_finite_and_correct():
    d = np.array([5.0, -5.0 + 1e-12, 2.0, -2.0])
    t = ca.totals(d)
    assert t['net'] < 1e-10 and t['gross'] == pytest.approx(14.0)
    np.testing.assert_allclose(t['shares_of_gross'], np.abs(d) / 14.0)
    assert np.isfinite(t['shares_of_gross']).all() and t['denominator'] == 'gross'
    assert ca.share(d, [0, 2])['share_of_gross'] == pytest.approx(0.5)
    assert ca.totals(np.zeros(3))['net_over_gross'] is None


def test_group_prediction_and_per_component_cancellation():
    removal = np.array([-0.01, 0.002, -0.003])
    g = ca.group_prediction(removal, [0, 2])
    assert g['J'] == pytest.approx(-0.013) and g['pp'] == pytest.approx(-1.3)
    assert g['label'] == ca.GROUP_PREDICTION_LABEL
    S = np.array([[1.0, 2.0], [-1.0, 2.0]])
    np.testing.assert_allclose(ca.per_component_gross_net(S), [0.0, 1.0])


def test_aggregated_neuron_identity_and_sign_convention():
    reads = np.array([[0.5, -0.2, 0.1], [1.0, 1.0, -3.0]])
    adapters = np.array([9.0, 0.4, 7.0, -1.0])
    down = {0: 1, 1: 3}
    ok = ca.neuron_identity(reads, adapters, down)
    assert ok['passed'] and ok['layers']['1']['error'] == pytest.approx(0.0)
    assert ok['sign_convention'] == ca.SIGN_CONVENTION
    bad = ca.neuron_identity(reads, adapters + np.array([0, 1e-3, 0, 0]), down)
    assert not bad['passed'] and not bad['layers']['0']['passed']
    for entry in ('attr_c(e)', 'removal_J_c', 'dJdλ_c'):
        assert entry in ca.__doc__


# --- the instrument's reduced reads feed the same aggregate ---------------------------------------

def _tiny_question_rows(q=2, n=8):
    rows, rewards = [], {}
    for i in range(q):
        rewards[f'q{i}'] = [int((3 * i + d) % 3 == 0) for d in range(n)]
        for d in range(n):
            size = 7 + (d + i) % 4
            ids = [(5 * i + 3 * d + 7 * k) % 29 + 1 for k in range(size)]
            rows.append({'id': f'q{i}:{d}', 'question_id': f'q{i}', 'draw_index': d, 'reward': rewards[f'q{i}'][d],
                         'input_ids': ids, 'masks': {'seq_targets': list(range(4, size)), 'length': size}})
    return rows, rewards


def test_instrument_reduced_reads_equal_direct_aggregation_and_identity_holds():
    pytest.importorskip('torch')
    from test_attribution_instrument import instrument, tiny
    model = tiny()
    inst = instrument(model, metric_order=('seqlogp',))
    rows, rewards = _tiny_question_rows()
    layers = sorted(inst.down_index)
    width = model.config.intermediate_size
    sums = ca.QuestionSums(list(rewards), n_adapters=len(inst.adapters), layers=layers, width=width)
    direct_reads = np.zeros((len(layers), width))
    direct_adapters = np.zeros(len(inst.adapters))
    w = ca.loo_weights(rewards)
    for start in range(0, len(rows), 5):
        batch = rows[start:start + 5]
        req = ca.reduction_request(batch, rewards)
        out = inst.attribute(batch, neuron_reduction=req)
        sums.add_adapters(req['question_ids'], [r['draw_index'] for r in batch], out['attr']['seqlogp'],
                          req['coefficients'], req['half_coefficients'])
        sums.add_reads(out['neuron']['seqlogp'], out['neuron_question_ids'])
        plain = inst.attribute(batch)
        for k, r in enumerate(batch):
            weight = w['weights'][r['question_id']][r['draw_index']]
            direct_adapters += weight * plain['attr']['seqlogp'][k]
            for li, layer in enumerate(layers):
                direct_reads[li] += weight * plain['neuron']['seqlogp'][layer][k]
    sums.complete(len(rows))
    result = ca.summary(sums, adapters=inst.adapters, down_index=inst.down_index)
    np.testing.assert_allclose(result['adapters']['removal_J'], direct_adapters, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(result['reads']['removal_J'], direct_reads, rtol=1e-10, atol=1e-13)
    assert result['identity']['passed'], result['identity']
    assert result['sign_convention'] == ca.SIGN_CONVENTION
