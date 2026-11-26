"""Ticket 03a pure checks, gauge and matched null (NumPy only)."""
import numpy as np
import pytest

from pilot_eval import neuron_audit_math as nm

TOL = 1e-5


def rng(seed=0):
    return np.random.default_rng(seed)


def test_reconstruction_passes_rejects_wrong_layer_and_handles_zero_reference():
    z0, z1 = rng(0).normal(size=(6, 12)), rng(1).normal(size=(6, 12))
    assert nm.check_reconstruction(z0, z0 + 1e-9, rel_to_layer_max=1e-6)['status'] == 'pass'
    assert nm.check_reconstruction(z1, z0, rel_to_layer_max=1e-6)['status'] == 'fail'  # wrong layer
    assert nm.check_reconstruction(np.zeros((2, 3)), np.zeros((2, 3)), rel_to_layer_max=1e-6)['status'] == 'pass'
    assert nm.check_reconstruction(np.zeros((2, 3)), np.full((2, 3), 1e-30),
                                   rel_to_layer_max=1e-6)['status'] == 'unresolved-zero-reference'


def test_read_check_wrong_sign_fails_and_cancellation_uses_sum_of_abs_terms():
    z, a = rng(0).normal(size=(5, 12)), rng(1).normal(size=12)
    c = z @ a
    assert nm.check_read(c, a, z)['status'] == 'pass'
    assert nm.check_read(c, -a, z)['status'] == 'fail'  # wrong sign
    # Exact cancellation: a.z = 0 but terms are large; tiny absolute error must still pass.
    z2, a2 = np.array([[1e3, 1e3]]), np.array([1.0, -1.0])
    assert nm.check_read(np.array([1e-4]), a2, z2)['status'] == 'pass'
    # Zero reference scale: exact zero passes, anything else is unresolved, never divided by zero.
    assert nm.check_read(np.array([0.0]), a2, np.zeros((1, 2)))['status'] == 'pass'
    assert nm.check_read(np.array([1e-9]), a2, np.zeros((1, 2)))['status'] == 'unresolved-zero-reference'


def test_branch_check_scale_and_sign():
    z, a, b = rng(0).normal(size=(4, 12)), rng(1).normal(size=12), rng(2).normal(size=8)
    good = 2.0 * np.outer(z @ a, b)
    assert nm.check_branch(good, a, b, z, 2.0)['status'] == 'pass'
    assert nm.check_branch(good, a, b, z, 1.0)['status'] == 'fail'  # wrong scale
    assert nm.check_branch(good, a, -b, z, 2.0)['status'] == 'fail'


def test_reduction_check_detects_shifted_mask_and_count():
    z = rng(0).normal(size=(10, 12)).astype(np.float32)
    positions = [3, 4, 5]
    good_sum, good_abs = z[positions].astype(np.float64).sum(0), np.abs(z[positions].astype(np.float64)).sum(0)
    assert nm.check_reduction(z, positions, good_sum, good_abs, 3)['status'] == 'pass'
    shifted = [4, 5, 6]
    assert nm.check_reduction(z, shifted, good_sum, good_abs, 3)['status'] == 'fail'
    assert nm.check_reduction(z, positions, good_sum, good_abs, 4)['status'] == 'fail'


@pytest.mark.parametrize('k', [3.0, -0.5])
def test_gauge_preserves_physical_write_and_within_arm_ranking(k):
    a, b, z = rng(0).normal(size=12), rng(1).normal(size=8), rng(2).normal(size=(5, 12))
    a2, b2 = nm.gauge_transform(a, b, k)
    assert np.allclose(nm.physical_write(b, z @ a, 2.0), nm.physical_write(b2, z @ a2, 2.0))
    ranks = nm.within_arm_rank(np.abs(z * a).mean(0))
    assert np.array_equal(ranks, nm.within_arm_rank(np.abs(z * a2).mean(0)))
    # Signed terms flip with a negative gauge factor: not comparable across arms without alignment.
    assert np.sign((z * a2).mean(0)[0]) == np.sign(k) * np.sign((z * a).mean(0)[0])
    with pytest.raises(ValueError):
        nm.gauge_transform(a, b, 0)


def test_rank_ties_broken_by_neuron_index():
    assert nm.within_arm_rank([1.0, 2.0, 2.0, 0.5]).tolist() == [1, 2, 0, 3]


BINS = {'low': [0, 1, 2, 3, 4], 'mid': [5, 6, 7, 8], 'high': [9, 10, 11]}
X = [1, 2, 6, 9, 10]
Y_COUNTS = {'low': 2, 'mid': 2, 'high': 1}


def test_exact_distribution_matches_brute_force_enumeration():
    exact = nm.exact_overlap_distribution(X, Y_COUNTS, BINS)
    brute = nm.enumerate_overlap_distribution(X, Y_COUNTS, BINS)
    assert np.allclose(exact[:len(brute)], brute) and np.isclose(exact.sum(), 1)


def test_monte_carlo_null_is_seeded_without_replacement_and_matches_exact():
    a = nm.matched_null(X, Y_COUNTS, BINS, draws=4000, seed=10000)
    assert np.array_equal(a, nm.matched_null(X, Y_COUNTS, BINS, draws=4000, seed=10000))
    assert not np.array_equal(a, nm.matched_null(X, Y_COUNTS, BINS, draws=4000, seed=10001))
    exact = nm.exact_overlap_distribution(X, Y_COUNTS, BINS)
    mean, var = (np.arange(exact.size) * exact).sum(), ((np.arange(exact.size) ** 2) * exact).sum()
    var -= mean ** 2
    assert abs(a.mean() - mean) < 4 * np.sqrt(var / a.size)
    # Without replacement: drawing a whole bin always returns every member exactly once.
    full = nm.matched_null(X, {'low': 5, 'mid': 4, 'high': 3}, BINS, draws=20, seed=1)
    assert (full == len(X)).all()


def test_discrete_interval_coverage_is_conservative_within_monte_carlo_noise():
    exact = nm.exact_overlap_distribution(X, Y_COUNTS, BINS)
    interval = nm.reference_interval(nm.matched_null(X, Y_COUNTS, BINS, draws=4000, seed=10000))
    lo, hi = int(np.floor(interval['p2_5'])), int(np.ceil(interval['p97_5']))
    assert exact[lo:hi + 1].sum() >= 0.95 - 0.02  # discrete: coverage need not equal 95%
    assert 'not a confidence interval' in interval['note']


def test_underfilled_bin_is_unavailable_not_switched():
    with pytest.raises(nm.Unavailable):
        nm.matched_null(X, {'high': 4}, BINS, draws=10, seed=1)
    with pytest.raises(nm.Unavailable):
        nm.exact_overlap_distribution(X, {'high': 4}, BINS)


def test_derived_seed_streams_are_stable_and_distinct():
    s = nm.derive_seed(10000, 'A/D', 5, 'D-fixed')
    assert s == nm.derive_seed(10000, 'A/D', 5, 'D-fixed')
    assert len({s, nm.derive_seed(10000, 'A/D', 6, 'D-fixed'), nm.derive_seed(10000, 'A/D', 5, 'B-fixed')}) == 3


def test_adr0015_bound_at_realistic_width_passes_fp32_and_fails_each_mutation():
    """d = 8960 like Qwen2.5-1.5B down_proj; judge mutations over a 64-token batch, never one token."""
    r = rng(7)
    d, tokens = 8960, 64
    z = r.normal(size=(tokens, d)).astype(np.float32)
    z_other_layer = r.normal(size=(tokens, d)).astype(np.float32)
    a = (r.normal(size=d) * 0.01).astype(np.float32)
    c_fp32 = (z @ a).astype(np.float32)  # what a correct FP32 GPU/CPU path produces
    ok = nm.check_read(c_fp32, a, z)
    assert ok['status'] == 'pass' and 0 < ok['gamma_d'] < 6e-4
    assert nm.check_read(-c_fp32, a, z)['status'] == 'fail'                       # flipped sign of a
    assert nm.check_read((z_other_layer @ a).astype(np.float32), a, z)['status'] == 'fail'  # wrong layer
    assert nm.check_read(np.roll(c_fp32, 1), a, z)['status'] == 'fail'            # mask shifted by one token
    b = r.normal(size=24).astype(np.float32)
    branch = (2.0 * np.outer(c_fp32, b)).astype(np.float32)
    assert nm.check_branch(branch, a, b, z, 2.0)['status'] == 'pass'
    assert nm.check_branch(-branch, a, b, z, 2.0)['status'] == 'fail'


def test_sign_flip_can_hide_on_a_single_cancelling_token():
    """Why mutations are judged over a batch: with c close to 0 a flip stays within the bound."""
    z, a = np.array([[1.0, 1.0]]), np.array([0.5, -0.5])
    assert nm.check_read(np.array([0.0]), -a, z)['status'] == 'pass'
