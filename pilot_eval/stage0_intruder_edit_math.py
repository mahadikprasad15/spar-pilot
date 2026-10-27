"""Deterministic singular-value matching and full merged-matrix construction."""
from fractions import Fraction
from functools import lru_cache

import numpy as np

MODES = ('unhalved', 'full', 'matched', 'control', 'ratio', 'ratio-control')


def match_controls(sigma, flagged, controls):
    """Minimum absolute-distance assignment, with lexicographic control-index ties.

    Singular values are sorted. Absolute distance has the Monge property, so an
    optimum exists without crossing assignments. Dynamic programming selects the
    control subsequence. Exact rational costs of the stored FP64 values avoid
    arbitrary epsilon tie-breaking; each flagged index gets a distinct control.
    """
    values = np.asarray(sigma, dtype=np.float64)
    flags, candidates = sorted(flagged), sorted(controls)
    if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values) > 0)
            or len(set(flags)) != len(flags) or len(set(candidates)) != len(candidates)
            or set(flags) & set(candidates)
            or any(type(i) is not int or not 0 <= i < len(values) for i in flags+candidates)):
        raise ValueError('Matching requires a sorted nonnegative spectrum and distinct disjoint indices.')
    if len(candidates) < len(flags):
        return dict(status='insufficient-controls', pairs=[], total_absolute_difference=None,
                    flagged_indices=flags, control_pool=candidates)
    exact = [Fraction.from_float(float(v)) for v in values]

    @lru_cache(None)
    def solve(fi, ci):
        if fi == len(flags): return Fraction(0), ()
        if len(candidates)-ci < len(flags)-fi: return None
        options = []
        for j in range(ci, len(candidates)-(len(flags)-fi)+1):
            rest = solve(fi+1, j+1)
            if rest is not None:
                options.append((abs(exact[flags[fi]]-exact[candidates[j]])+rest[0], (candidates[j],)+rest[1]))
        return min(options)

    cost, assignment = solve(0, 0)
    pairs = []
    for flagged_index, control_index in zip(flags, assignment):
        sf, sc = float(values[flagged_index]), float(values[control_index])
        ratio = None if sf == 0 else sc/sf
        pairs.append(dict(flagged_index=flagged_index, control_index=control_index,
            sigma_flagged=sf, sigma_control=sc, absolute_difference=float(abs(exact[flagged_index]-exact[control_index])),
            sigma_ratio=ratio, ratio_status='undefined-zero-flagged-singular-value' if sf == 0 else 'finite',
            ratio_retained=ratio is not None and .8 <= ratio <= 1.25))
    return dict(status='matched' if flags else 'no-flags', pairs=pairs, total_absolute_difference=float(cost),
                flagged_indices=flags, control_pool=candidates)


def module_selection(row):
    candidates = row['candidates'][:40]
    chance = row['levels']['50']['chance_norm']
    flags = [c['index'] for c in candidates[:20] if c['overlap_norm']['50'] < .5] if .5 > chance else []
    if flags != row['primary']['eligible_indices']:
        raise ValueError('Geometry primary eligibility disagrees with its candidate overlap/chance rule.')
    controls = [c['index'] for c in candidates if c['overlap_norm']['50'] >= .5] if .5 > chance else []
    sigma = np.asarray([c['sigma'] for c in candidates], dtype=float)
    matching = match_controls(sigma, flags, controls)
    matched = matching['status'] == 'matched'
    retained = [p for p in matching['pairs'] if p['ratio_retained']]
    selections = dict(full=flags, matched=flags if matched else [],
        control=[p['control_index'] for p in matching['pairs']] if matched else [],
        ratio=[p['flagged_index'] for p in retained],
        **{'ratio-control':[p['control_index'] for p in retained]})
    return dict(primary_status=row['primary']['status'], flagged_indices=flags, chance_norm=chance,
        m=row['levels']['50']['m'], control_pool=controls, matching=matching, selections=selections,
        skipped_full_reason=None if flags else row['primary']['status'],
        excluded_matched_reason='insufficient-controls' if flags and not matched else None,
        sensitivity_rule='same original matched pairs; inclusive ratio [0.8,1.25]; no rematching',
        degeneracy_note=row.get('degeneracy_note'))


def edit_matrix(merged, u, v, sigma, selected):
    matrix = np.asarray(merged, dtype=np.float64)
    left, right, values = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    indices = sorted(selected)
    if (matrix.ndim != 2 or left.shape[0] != matrix.shape[0] or right.shape[0] != matrix.shape[1]
            or values.ndim != 1 or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < min(left.shape[1], right.shape[1], len(values)) for i in indices)
            or not all(np.isfinite(a).all() for a in [matrix,left,right,values])):
        raise ValueError('Invalid merged matrix/singular-vector edit mapping.')
    target = matrix.copy()
    if indices:
        target -= (left[:,indices]*(.5*values[indices]))@right[:,indices].T
    result = target.astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite full edited matrix.')
    denominator = float(np.linalg.norm(target))
    error = float(np.linalg.norm(result.astype(float)-target))
    relative = None if denominator == 0 else error/denominator
    if (denominator == 0 and error != 0) or (denominator > 0 and not relative < 1e-6):
        raise ValueError('Full edited-matrix reconstruction exceeds inherited 1e-6 gate.')
    return result, dict(selected_indices=indices, halving_factor=.5,
        singular_values=[dict(index=i,before=float(values[i]),after=float(.5*values[i]),
            measured_coordinate=float(left[:,i]@result.astype(float)@right[:,i])) for i in indices],
        serialized_reconstruction_relative_error=relative, serialized_reconstruction_absolute_error=error,
        zero_target=denominator==0, representation='full-FP32-merged-weight',
        definition='FP64 merged matrix minus half the selected saved singular components, then FP32 storage')
