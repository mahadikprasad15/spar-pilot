"""Ticket 03a pure math for the joint neuron/adapter audit (NumPy only, no model).

Independent identity checks, per-example masked reductions, rank-one gauge helpers and
the activity-bin-matched overlap null. Tolerances are always passed in by the caller
from a plan; the values in ``candidate-plan-v1.yaml`` are proposals, not approvals.

Notation (rank one): z_t is the ``down_proj`` input at token t (J neurons), a is the saved
``lora_A`` row (J), b the saved ``lora_B`` column, s the LoRA scale. The read is
c_t = a.z_t, the signed term is a_j z_j[t], and the adapter write is s*b*c_t.
"""
import hashlib
import itertools
from math import comb

import numpy as np


class Unavailable(ValueError):
    """A requested statistic or null cannot be formed under the frozen rules."""


def _result(passed, ratio, rule, **extra):
    return dict(status='pass' if passed else 'fail', max_ratio=ratio, rule=rule, **extra)


def check_reconstruction(z, z_rebuilt, *, rel_to_layer_max):
    """Captured z vs act(gate)*up rebuilt from separately hooked branches, one layer."""
    z, z_rebuilt = np.asarray(z, np.float64), np.asarray(z_rebuilt, np.float64)
    if z.shape != z_rebuilt.shape:
        return _result(False, None, 'shape', detail='shape mismatch')
    error = float(np.max(np.abs(z - z_rebuilt))) if z.size else 0.0
    scale = float(np.max(np.abs(z))) if z.size else 0.0
    if scale == 0.0:
        return dict(status='pass' if error == 0.0 else 'unresolved-zero-reference', max_ratio=None,
                    rule='exact_zero_or_unresolved', max_error=error)
    ratio = error / (rel_to_layer_max * scale)
    return _result(ratio <= 1.0, ratio, f'max|dz| <= {rel_to_layer_max} * layer max|z|', max_error=error)


def _token_rows(values):
    values = np.asarray(values, np.float64)
    return values.reshape(-1, values.shape[-1])


def gamma(d):
    """Deterministic FP32 dot-product error factor gamma_d = d*u/(1-d*u), u = 2^-24 (ADR 0015)."""
    u = 2.0 ** -24
    if d * u >= 1:
        raise ValueError('dot-product length too large for the FP32 bound')
    return d * u / (1 - d * u)


def _adr0015(observed, expected, scale, *, atol, rtol, d, rule):
    """Allowance max(atol + rtol*|expected|, gamma_d * scale); scale = sum of |rank-one terms|."""
    error = np.abs(observed - expected)
    zero = scale == 0
    if np.any(error[zero] != 0):
        return dict(status='unresolved-zero-reference', max_ratio=None, rule='exact_zero_or_unresolved',
                    zero_scale=int(zero.sum()))
    legacy = atol + rtol * np.abs(expected)
    allowance = np.maximum(legacy, gamma(d) * scale)
    ratio = float(np.max(error[~zero] / allowance[~zero])) if np.any(~zero) else 0.0
    legacy_ratio = float(np.max(error / legacy)) if error.size else 0.0
    return _result(ratio <= 1.0, ratio, rule, gamma_d=gamma(d), legacy_max_fraction=legacy_ratio,
                   zero_scale=int(zero.sum()))


def check_read(c_observed, a, z, *, atol=1e-6, rtol=1e-5):
    """Independently captured lora_A output vs a.z per token (ADR 0015 bound, d = len(a))."""
    z2 = _token_rows(z)
    c = np.asarray(c_observed, np.float64).reshape(-1)
    terms = z2 * np.asarray(a, np.float64)[None, :]
    return _adr0015(c, terms.sum(axis=1), np.abs(terms).sum(axis=1), atol=atol, rtol=rtol, d=terms.shape[1],
                    rule='|c - a.z| <= max(atol + rtol*|a.z|, gamma_d * sum|a_j z_j|)')


def check_branch(branch_observed, a, b, z, s, *, atol=1e-6, rtol=1e-5):
    """Actual adapter-branch output (live-scaled) vs s*b_i*(a.z), per output coordinate (ADR 0015 form)."""
    z2 = _token_rows(z)
    out = _token_rows(branch_observed)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    terms = z2 * a[None, :]
    expected = s * np.outer(terms.sum(axis=1), b)
    scale = abs(s) * np.outer(np.abs(terms).sum(axis=1), np.abs(b))
    return _adr0015(out, expected, scale, atol=atol, rtol=rtol, d=terms.shape[1],
                    rule='|branch - s b_i c| <= max(atol + rtol*|s b_i c|, gamma_d * |s||b_i| sum|a_j z_j|)')


def check_reduction(z_example, positions, reported_sum, reported_abs_sum, reported_count):
    """Recompute one example's masked sums from a full [T, J] capture and compare exactly-ish."""
    z = np.asarray(z_example, np.float64)
    picked = z[np.asarray(positions, int)] if len(positions) else np.zeros((0, z.shape[1]))
    ok_count = int(reported_count) == len(positions)
    scale = np.maximum(np.abs(picked).sum(axis=0), np.finfo(np.float32).tiny)
    sum_ratio = np.max(np.abs(picked.sum(axis=0) - reported_sum) / scale)
    abs_ratio = np.max(np.abs(np.abs(picked).sum(axis=0) - reported_abs_sum) / scale)
    # FP64 accumulation of FP32 values: allow a few FP32 ulps of the absolute-sum scale.
    limit = 8 * np.finfo(np.float32).eps
    passed = ok_count and sum_ratio <= limit and abs_ratio <= limit
    return _result(bool(passed), float(max(sum_ratio, abs_ratio) / limit), 'masked sums within 8 FP32 eps of sum|z|',
                   count_matches=ok_count)


# --- gauge -----------------------------------------------------------------------------------

def gauge_transform(a, b, k):
    """(a, b) -> (k a, b / k); leaves the rank-one update b a^T unchanged (k != 0)."""
    if k == 0:
        raise ValueError('gauge factor must be nonzero')
    return np.asarray(a) * k, np.asarray(b) / k


def physical_write(b, c, s):
    return s * np.outer(np.asarray(c, np.float64).reshape(-1), np.asarray(b, np.float64))


def within_arm_rank(mean_abs_terms):
    """Descending rank by mean |a_j z_j|, ties broken by neuron index (frozen convention)."""
    values = np.asarray(mean_abs_terms, np.float64)
    return np.lexsort((np.arange(values.size), -values))


# --- matched overlap null ------------------------------------------------------------------

def derive_seed(base_seed, *labels):
    """Deterministic, independent stream per (comparison, layer, direction, ...)."""
    digest = hashlib.sha256(repr((int(base_seed),) + tuple(map(str, labels))).encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def _pools(universe_bins):
    return {k: np.asarray(sorted(v), int) for k, v in sorted(universe_bins.items())}


def matched_null(observed, y_bin_counts, universe_bins, *, draws, seed):
    """Overlap of fixed X with Y-sized sets drawn without replacement inside each base-activity bin.

    ``universe_bins`` maps each bin to its eligible neuron indices (the layer's declared
    universe, including X where eligible); ``y_bin_counts`` is Y's actual count per bin.
    Raises ``Unavailable`` when a bin cannot supply the matched count (no fallback null).
    """
    pools = _pools(universe_bins)
    observed = set(int(i) for i in observed)
    for name, count in y_bin_counts.items():
        if count < 0 or name not in pools or count > len(pools[name]):
            raise Unavailable(f'bin {name!r} cannot supply {count} matched draws')
    rng = np.random.Generator(np.random.PCG64(seed))
    overlaps = np.zeros(draws, dtype=np.int64)
    members = {k: np.isin(pools[k], list(observed)) for k in pools}
    for d in range(draws):
        total = 0
        for name, count in sorted(y_bin_counts.items()):
            if count:
                picked = rng.choice(len(pools[name]), size=count, replace=False)
                total += int(members[name][picked].sum())
        overlaps[d] = total
    return overlaps


def exact_overlap_distribution(observed, y_bin_counts, universe_bins):
    """Exact pmf of the null overlap: convolution of per-bin hypergeometric counts."""
    pools = _pools(universe_bins)
    observed = set(int(i) for i in observed)
    pmf = np.array([1.0])
    for name, count in sorted(y_bin_counts.items()):
        pool = pools[name]
        if count > len(pool):
            raise Unavailable(f'bin {name!r} cannot supply {count} matched draws')
        hits = int(np.isin(pool, list(observed)).sum())
        total = comb(len(pool), count)
        local = np.array([comb(hits, x) * comb(len(pool) - hits, count - x) / total
                          for x in range(min(hits, count) + 1)])
        pmf = np.convolve(pmf, local)
    return pmf


def enumerate_overlap_distribution(observed, y_bin_counts, universe_bins):
    """Brute-force enumeration for tiny fixtures (validates the exact convolution)."""
    pools = _pools(universe_bins)
    observed = set(int(i) for i in observed)
    per_bin = []
    for name, count in sorted(y_bin_counts.items()):
        per_bin.append([sum(int(i) in observed for i in subset)
                        for subset in itertools.combinations(pools[name].tolist(), count)])
    counts = {}
    for combo in itertools.product(*per_bin):
        counts[sum(combo)] = counts.get(sum(combo), 0) + 1
    total = sum(counts.values())
    return np.array([counts.get(x, 0) / total for x in range(max(counts) + 1)])


def reference_interval(overlaps):
    """Null median and 2.5/97.5 percentiles: a randomization reference, not a confidence interval."""
    values = np.asarray(overlaps, np.float64)
    return dict(median=float(np.median(values)), p2_5=float(np.percentile(values, 2.5)),
                p97_5=float(np.percentile(values, 97.5)), draws=int(values.size),
                note='conditional randomization reference interval; not a confidence interval')
