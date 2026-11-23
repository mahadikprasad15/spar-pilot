"""Sparse rewiring ticket 11 (D21): Rank 1 vs Rank 0 comparison. CPU, NumPy only.

Descriptive comparison of ``sparse-gold-sft-r1`` and ``sparse-gold-sft-r0``:
- Survivor counts, global and per-layer Jaccard overlap, unique neurons, and a layer-matched null
  (reusing ``neuron_audit_math.matched_null``, ``exact_overlap_distribution``, ``reference_interval``).
- Three-way overlap of rank 1, rank 0, and named C-attribution sets on the same universe and null.
- Paired accuracy (McNemar exact p via ``c_attribution_generation.mcnemar_exact`` and paired
  bootstrap on question IDs via ``c_attribution_stability.resample_counts``), with ``gold-sft`` and
  ``untuned`` as references.
- Validation CE per checkpoint and hardening CE change across the freeze per arm.

Note: one seed per arm, and the two arms differ in read LR (3e-3 vs 3e-4, D1) as well as rank.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_stability as cs
from pilot_eval import neuron_audit_math as nm

SCHEMA: str = 'sparse-rewiring-rank-comparison-v1'
DEFAULT_SHAPE: tuple[int, int] = (28, 8960)
NULL_DRAWS: int = 10_000
NULL_SEED: int = 20261124
BOOTSTRAP_DRAWS: int = 10_000
BOOTSTRAP_SEED: int = 20261124
LABELS: dict[str, str] = {
    'descriptive': 'One seed per arm; descriptive comparison only.',
    'lr-differs': 'Rank 1 (read LR 3e-3) and Rank 0 (read LR 3e-4) differ in read learning rate as well as rank (D1).',
    'cohort150-exploratory': 'Cohort150 is evaluation only and has prior uses; accuracy comparisons are exploratory (D12).',
}


def _extract_pairs(value: Any, *, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Extract sorted unique zero-based ``(layer, neuron)`` pairs from an export dict or pair sequence."""
    raw = value['survivors'] if isinstance(value, Mapping) and 'survivors' in value else value
    n_layers, width = shape
    seen: set[tuple[int, int]] = set()
    for item in raw:
        if isinstance(item, Mapping):
            layer, neuron = int(item['layer']), int(item['neuron'])
        else:
            layer, neuron = int(item[0]), int(item[1])
        if not (0 <= layer < n_layers and 0 <= neuron < width):
            raise ValueError(f'neuron ({layer}, {neuron}) out of bounds for shape {shape}')
        seen.add((layer, neuron))
    return sorted(seen)


def _to_flat(pairs: Sequence[tuple[int, int]], width: int) -> list[int]:
    return sorted(layer * width + neuron for layer, neuron in pairs)


def _layer_counts_and_bins(
    flat_indices: Sequence[int],
    universe: np.ndarray,
) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    n_layers, width = universe.shape
    counts: dict[int, int] = {}
    for t in flat_indices:
        layer = t // width
        counts[layer] = counts.get(layer, 0) + 1
    bins: dict[int, np.ndarray] = {}
    for layer in counts:
        bins[layer] = np.flatnonzero(universe[layer]) + layer * width
    return counts, bins


def _chunked_matched_null(
    observed_flat: Sequence[int],
    y_counts: Mapping[int, int],
    bins: Mapping[int, np.ndarray],
    *,
    draws: int,
    seed: int,
    chunk: int = 512,
) -> np.ndarray:
    """Run ``neuron_audit_math.matched_null`` in memory-bounded chunks."""
    if draws <= 0:
        return np.zeros(0, dtype=np.int64)
    if not y_counts:
        return np.zeros(draws, dtype=np.int64)
    parts: list[np.ndarray] = []
    remaining = int(draws)
    chunk_idx = 0
    while remaining > 0:
        k = min(remaining, chunk)
        c_seed = nm.derive_seed(seed, 'chunk', chunk_idx)
        parts.append(nm.matched_null(observed_flat, y_counts, bins, draws=k, seed=c_seed))
        remaining -= k
        chunk_idx += 1
    return np.concatenate(parts)


def _overlap_with_null(
    fixed_flat: Sequence[int],
    matched_flat: Sequence[int],
    universe: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    u_flat = universe.ravel()
    fixed_in = [t for t in fixed_flat if u_flat[t]]
    matched_in = [t for t in matched_flat if u_flat[t]]
    counts, bins = _layer_counts_and_bins(matched_in, universe)
    observed = len(set(fixed_in) & set(matched_in))
    pmf = nm.exact_overlap_distribution(fixed_in, counts, bins)
    null_draws = _chunked_matched_null(fixed_in, counts, bins, draws=draws, seed=seed)
    return {
        'fixed_size': len(fixed_in),
        'matched_size': len(matched_in),
        'observed': observed,
        'null': nm.reference_interval(null_draws) | {'mean': float(np.mean(null_draws)) if null_draws.size else None},
        'exact': {
            'expected': float((np.arange(pmf.size) * pmf).sum()),
            'p_upper': float(min(1.0, pmf[observed:].sum())) if observed < pmf.size else 0.0,
            'rule': 'P(null overlap >= observed), exact conditional pmf',
        },
    }


def _export_counts_summary(value: Any, fallback_size: int) -> dict[str, Any]:
    if isinstance(value, Mapping) and 'counts' in value and isinstance(value['counts'], Mapping):
        total = value['counts'].get('total', {})
        return {
            'expected_active': total.get('expected_active'),
            'nonzero_z_det': total.get('nonzero_z_det'),
            'binary_survivors': total.get('binary_survivors'),
            'set_size': fallback_size,
        }
    return {
        'expected_active': None,
        'nonzero_z_det': None,
        'binary_survivors': fallback_size,
        'set_size': fallback_size,
    }


def compare_survivors(
    r1_export: Any,
    r0_export: Any,
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    draws: int = NULL_DRAWS,
    seed: int = NULL_SEED,
) -> dict[str, Any]:
    """Compare rank-1 and rank-0 final survivor sets: counts, Jaccard, per-layer overlap, unique sets, null."""
    n_layers, width = shape
    s1 = _extract_pairs(r1_export, shape=shape)
    s0 = _extract_pairs(r0_export, shape=shape)
    set1, set0 = set(s1), set(s0)
    shared = sorted(set1 & set0)
    union = set1 | set0
    jaccard = float(len(shared) / len(union)) if union else 1.0

    per_layer: list[dict[str, Any]] = []
    for layer in range(n_layers):
        l1 = {n for l, n in set1 if l == layer}
        l0 = {n for l, n in set0 if l == layer}
        lu = l1 | l0
        li = l1 & l0
        per_layer.append(
            {
                'layer': layer,
                'r1_count': len(l1),
                'r0_count': len(l0),
                'intersection': len(li),
                'union': len(lu),
                'jaccard': float(len(li) / len(lu)) if lu else 1.0,
            }
        )

    universe = np.ones(shape, dtype=bool)
    f1, f0 = _to_flat(s1, width), _to_flat(s0, width)
    null_seed = nm.derive_seed(seed, 'r1_vs_r0')
    overlap_null = _overlap_with_null(f1, f0, universe, draws=draws, seed=null_seed)

    return {
        'shape': list(shape),
        'counts': {
            'r1': _export_counts_summary(r1_export, len(s1)),
            'r0': _export_counts_summary(r0_export, len(s0)),
        },
        'intersection_count': len(shared),
        'union_count': len(union),
        'jaccard': jaccard,
        'layer_matched_null': overlap_null,
        'per_layer': per_layer,
        'shared': [{'layer': l, 'neuron': n} for l, n in shared],
        'unique_to_r1': [{'layer': l, 'neuron': n} for l, n in sorted(set1 - set0)],
        'unique_to_r0': [{'layer': l, 'neuron': n} for l, n in sorted(set0 - set1)],
    }


def compare_three_way_c_sets(
    r1_export: Any,
    r0_export: Any,
    c_sets: Mapping[str, Sequence[Any]],
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    universe: np.ndarray | None = None,
    draws: int = NULL_DRAWS,
    seed: int = NULL_SEED,
) -> dict[str, Any]:
    """Compare r1, r0, and their intersection (r1 ∩ r0) against named C-attribution sets on the same universe."""
    _, width = shape
    uni = np.ones(shape, dtype=bool) if universe is None else np.asarray(universe, dtype=bool)
    if uni.shape != shape:
        raise ValueError(f'universe shape {uni.shape} != {shape}')

    s1 = _extract_pairs(r1_export, shape=shape)
    s0 = _extract_pairs(r0_export, shape=shape)
    inter = sorted(set(s1) & set(s0))
    f1, f0, fi = _to_flat(s1, width), _to_flat(s0, width), _to_flat(inter, width)

    results: dict[str, Any] = {}
    for set_name, raw_members in sorted(c_sets.items()):
        c_pairs = _extract_pairs(raw_members, shape=shape)
        fc = _to_flat(c_pairs, width)
        results[set_name] = {
            'c_set_size': len(fc),
            'r1_vs_c': _overlap_with_null(fc, f1, uni, draws=draws, seed=nm.derive_seed(seed, set_name, 'r1')),
            'r0_vs_c': _overlap_with_null(fc, f0, uni, draws=draws, seed=nm.derive_seed(seed, set_name, 'r0')),
            'three_way_vs_c': _overlap_with_null(fc, fi, uni, draws=draws, seed=nm.derive_seed(seed, set_name, 'both')),
        }
    return {
        'universe_size': int(uni.sum()),
        'sets': results,
    }


def _reward_vector(responses: Mapping[str, Any], question_ids: Sequence[str]) -> np.ndarray:
    out = []
    for qid in question_ids:
        if qid not in responses:
            raise KeyError(f'question_id {qid!r} missing from responses')
        val = responses[qid]
        reward = val['reward'] if isinstance(val, Mapping) else val
        out.append(float(reward))
    return np.asarray(out, dtype=np.float64)


def _paired_block(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    counts: np.ndarray,
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Paired bootstrap interval of ``(b - a)`` in percentage points plus 2x2 table and McNemar exact p."""
    q = vec_a.size
    diff = vec_b - vec_a
    if q == 0:
        interval = {'estimate': 0.0, 'p2.5': 0.0, 'p97.5': 0.0}
    else:
        boot = (counts @ diff) / q
        lo, hi = np.percentile(boot, [2.5, 97.5])
        interval = {'estimate': float(np.mean(diff)), 'p2.5': float(lo), 'p97.5': float(hi)}
    a_ok = vec_a == 1.0
    b_ok = vec_b == 1.0
    both = int((a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    neither = int((~a_ok & ~b_ok).sum())
    return {
        'label_a': label_a,
        'label_b': label_b,
        'delta_pp_b_minus_a': {k: 100.0 * v for k, v in interval.items()},
        'both_correct': both,
        f'{label_a}_only': a_only,
        f'{label_b}_only': b_only,
        'neither': neither,
        'mcnemar_exact_p': float(cg.mcnemar_exact(b_only, a_only)),
    }


def compare_paired_accuracy(
    r1_responses: Mapping[str, Any],
    r0_responses: Mapping[str, Any],
    question_ids: Sequence[str],
    *,
    references: Mapping[str, Mapping[str, Any]] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired accuracy comparison between rank 1, rank 0, and optional references (gold-sft, untuned)."""
    ids = list(question_ids)
    if not ids:
        raise ValueError('question_ids must not be empty')
    v_r1 = _reward_vector(r1_responses, ids)
    v_r0 = _reward_vector(r0_responses, ids)
    counts = cs.resample_counts(len(ids), B=int(bootstrap_draws), seed=int(seed))

    out: dict[str, Any] = {
        'questions': len(ids),
        'accuracy': {'r1': float(v_r1.mean()), 'r0': float(v_r0.mean())},
        'r0_vs_r1': _paired_block(v_r1, v_r0, counts, label_a='r1', label_b='r0'),
        'vs_references': {},
    }
    for ref_name, ref_map in sorted((references or {}).items()):
        v_ref = _reward_vector(ref_map, ids)
        out['accuracy'][ref_name] = float(v_ref.mean())
        out['vs_references'][ref_name] = {
            'r1_vs_ref': _paired_block(v_ref, v_r1, counts, label_a=ref_name, label_b='r1'),
            'r0_vs_ref': _paired_block(v_ref, v_r0, counts, label_a=ref_name, label_b='r0'),
        }
    return out


def compare_val_ce_and_hardening(
    r1_val_ce: Mapping[int, float] | Sequence[Mapping[str, Any]],
    r0_val_ce: Mapping[int, float] | Sequence[Mapping[str, Any]],
    *,
    r1_hardening: Mapping[str, Any] | None = None,
    r0_hardening: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Align validation CE trajectories and summarize hardening CE deltas for both ranks."""
    def _to_map(data: Mapping[int, float] | Sequence[Mapping[str, Any]]) -> dict[int, float]:
        if isinstance(data, Mapping):
            return {int(k): float(v) for k, v in data.items()}
        return {int(row['step']): float(row['val_ce']) for row in data}

    m1, m0 = _to_map(r1_val_ce), _to_map(r0_val_ce)
    steps = sorted(set(m1) | set(m0))
    trajectory = [
        {
            'step': s,
            'r1_val_ce': m1.get(s),
            'r0_val_ce': m0.get(s),
            'delta_r0_minus_r1': (m0[s] - m1[s]) if (s in m1 and s in m0) else None,
        }
        for s in steps
    ]

    def _hardening_summary(h: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if h is None:
            return None
        before = float(h['val_ce_before'])
        after = float(h['val_ce_after'])
        return {
            'val_ce_before': before,
            'val_ce_after': after,
            'delta_val_ce': after - before,
            'pre_nonzero': h.get('pre_nonzero'),
            'post_binary': h.get('post_binary'),
        }

    return {
        'val_ce_trajectory': trajectory,
        'hardening': {
            'r1': _hardening_summary(r1_hardening),
            'r0': _hardening_summary(r0_hardening),
        },
    }


def compare_ranks(
    *,
    r1_export: Any,
    r0_export: Any,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    c_sets: Mapping[str, Sequence[Any]] | None = None,
    c_universe: np.ndarray | None = None,
    val_ce: tuple[Any, Any] | None = None,
    hardening: tuple[Mapping[str, Any] | None, Mapping[str, Any] | None] = (None, None),
    paired_evals: Mapping[str, Mapping[str, Any]] | None = None,
    null_draws: int = NULL_DRAWS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = NULL_SEED,
) -> dict[str, Any]:
    """Assemble the complete D21 rank-1 vs rank-0 comparison record."""
    out: dict[str, Any] = {
        'schema': SCHEMA,
        'labels': dict(LABELS),
        'survivors': compare_survivors(r1_export, r0_export, shape=shape, draws=null_draws, seed=seed),
    }
    if c_sets is not None:
        out['three_way_c_sets'] = compare_three_way_c_sets(
            r1_export, r0_export, c_sets, shape=shape, universe=c_universe, draws=null_draws, seed=seed
        )
    if val_ce is not None:
        out['val_ce_and_hardening'] = compare_val_ce_and_hardening(
            val_ce[0], val_ce[1], r1_hardening=hardening[0], r0_hardening=hardening[1]
        )
    if paired_evals is not None:
        out['paired_accuracy'] = {
            name: compare_paired_accuracy(
                spec['r1'],
                spec['r0'],
                spec['question_ids'],
                references=spec.get('references'),
                bootstrap_draws=bootstrap_draws,
                seed=nm.derive_seed(seed, 'paired', name),
            )
            for name, spec in sorted(paired_evals.items())
        }
    return out


def seal_rank_comparison(
    directory: str | Path,
    comparison: Mapping[str, Any],
    *,
    filename: str = 'rank_comparison.json',
) -> str:
    """Seal a rank-comparison record using ``attribution_units.seal_run``."""
    return au.seal_run(Path(directory), {filename: dict(comparison)})
