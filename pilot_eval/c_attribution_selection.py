"""C-attribution selection, matched controls, hot layers and the freeze (ticket 06).

Pure numpy except for reading files. Selection runs on an arm's aggregate (``c_attribution.summary``) and
its size measures; the freeze is sealed before any generation unit exists.

Groups per arm (positive dJdλ only; never padded, never dropped; a shortfall is frozen at its actual size):

| Group | Rule |
|---|---|
| ``top10-adapters`` | the 10 largest positive dJdλ among the adapters |
| ``global64-reads`` | the 64 largest positive dJdλ_j across every ``down_proj`` read |
| ``hot64-reads`` | the 64 largest positive dJdλ_j within each hot layer, removed together |
| ``layer64-reads`` | the 64 largest positive dJdλ_j within each ``down_proj`` layer, removed together |

Ties: layer, then module order or neuron index (the flat index order). Controls: adapters by module type,
reads by layer; size within ±tolerance of the target's; every same-kind target of the arm is excluded; no
rescaling; seeded group-wise maximum-cardinality matching (``match_group``).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval.c_attribution import SIGN_CONVENTION, group_prediction

FREEZE_DIR = 'c-attribution/freeze'
GENERATION_DIR = 'c-attribution/generation'
GROUPS = ('top10-adapters', 'global64-reads', 'hot64-reads', 'layer64-reads')
CONTROL = {'top10-adapters': 'match10-adapters', 'global64-reads': 'match-global64',
           'hot64-reads': 'match-hot64', 'layer64-reads': 'match-layer64'}


class SelectionError(ValueError):
    pass


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- matching ------------------------------------------------------------------------------------

def match_group(targets, size, strata, *, exclude, tolerance, seed):
    """Seeded group-wise maximum-cardinality matching of controls to ``targets`` (in rank order).

    ``size`` and ``strata`` are indexed by candidate id. A candidate is eligible for a target if it is in the
    same stratum, not in ``exclude``, and |size - target size| <= tolerance * |target size|. Candidates are
    permuted with ``seed`` (per stratum, strata in sorted order), then augmenting paths run in target order.
    The control is unavailable only if the maximum matching is smaller than the group.
    """
    targets = [int(t) for t in targets]
    size = np.asarray(size, dtype=np.float64)
    strata_arr = np.asarray(list(strata))
    strata = strata_arr.tolist()
    excluded = set(int(e) for e in exclude) | set(targets)
    keep = np.ones(len(size), dtype=bool)
    keep[list(excluded)] = False
    rng = np.random.default_rng(seed)
    by_stratum = {}
    for t in targets:
        by_stratum.setdefault(strata[t], []).append(t)
    per_target, eligible_count, deficient, per_stratum = {}, {}, [], {}
    for stratum in sorted(by_stratum, key=str):
        group = by_stratum[stratum]
        candidates = np.flatnonzero((strata_arr == strata_arr[group[0]]) & keep)
        candidates = candidates[rng.permutation(len(candidates))]
        adjacency = {t: [int(c) for c in candidates[np.abs(size[candidates] - size[t]) <= tolerance * abs(size[t])]]
                     for t in group}
        owner = {}

        def augment(t, seen):
            for c in adjacency[t]:
                if c in seen:
                    continue
                seen.add(c)
                if c not in owner or augment(owner[c], seen):
                    owner[c] = t
                    return True
            return False

        for t in group:
            augment(t, set())
        matched = {t: c for c, t in owner.items()}
        for t in group:
            eligible_count[str(t)] = len(adjacency[t])
            per_target[str(t)] = matched.get(t)
            if t not in matched:
                deficient.append(t)
        per_stratum[str(stratum)] = {'targets': len(group), 'matched': len(matched),
                                     'available': len(matched) == len(group)}
    available = not deficient
    ratios = [size[per_target[str(t)]] / size[t] for t in targets if per_target[str(t)] is not None and size[t] != 0]
    return {
        'available': available,
        'group': [per_target[str(t)] for t in targets] if available else None,
        'per_target': per_target,
        'deficient_targets': deficient,
        'eligible_candidates': eligible_count,
        'largest_feasible_matching': sum(v is not None for v in per_target.values()),
        'per_stratum': per_stratum,
        'deficient_strata': sorted({str(strata[t]) for t in deficient}),
        'match_ratio': ({'min': float(np.min(ratios)), 'median': float(np.median(ratios)), 'max': float(np.max(ratios))}
                        if ratios else None),
        'tolerance': tolerance, 'seed': seed,
        'rule': 'same stratum; |size - target| <= tolerance * |target|; excludes all same-kind targets; '
                'seeded maximum-cardinality matching; never loosened; no rescaling',
    }


# --- selection -----------------------------------------------------------------------------------

def top_positive(values, k, *, within=None):
    """The k largest positive values (ties by index), restricted to ``within`` indices if given."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise SelectionError('dJdλ must be finite')
    pool = np.arange(len(values)) if within is None else np.asarray(sorted(within))
    positive = [int(i) for i in pool if values[i] > 0]
    chosen = sorted(positive, key=lambda i: (-values[i], i))[:k]
    return chosen, {'requested': k, 'selected': len(chosen), 'shortfall': k - len(chosen), 'positive_available': len(positive)}


def adapter_measures(*, b_norms, scale, sum_abs_c, token_total):
    """Adapter write ‖s·b‖ · mean_t |c_t| over the arm's unpadded positions."""
    return np.abs(np.asarray(scale, dtype=np.float64)) * np.asarray(b_norms, dtype=np.float64) * \
        np.asarray(sum_abs_c, dtype=np.float64) / float(token_total)


def read_measures(*, term_size_total, token_total):
    """Read size |a_j| · mean_t |z_{j,t}| per (layer, j) from Σ |a_j||z_j| over the arm's positions."""
    return np.asarray(term_size_total, dtype=np.float64) / float(token_total)


def select_arm(summary, *, adapters, layers, adapter_size, read_size, hot_layers, seeds, tolerance=0.20,
               k_adapters=10, k_reads=64, k_harmful=10):
    """All four groups and their matched controls for one arm.

    ``summary`` is ``c_attribution.summary``; ``adapters`` = [(layer, projection)]; ``layers`` = the
    ``down_proj`` layers in read order; ``adapter_size`` [A]; ``read_size`` [L, W]; ``seeds`` = {group: int}.
    """
    a_d = np.asarray(summary['adapters']['dJdlambda'], dtype=np.float64)
    a_r = np.asarray(summary['adapters']['removal_J'], dtype=np.float64)
    r_d = np.asarray(summary['reads']['dJdlambda'], dtype=np.float64)
    r_r = np.asarray(summary['reads']['removal_J'], dtype=np.float64)
    L, W = r_d.shape
    if list(layers) != list(summary['reads']['layers']) or np.shape(read_size) != (L, W) or len(adapter_size) != len(a_d):
        raise SelectionError('measures do not match the aggregate')
    flat_d, flat_r = r_d.ravel(), r_r.ravel()
    read_size = np.asarray(read_size, dtype=np.float64).ravel()
    layer_of = np.repeat(np.asarray(layers), W)
    index_of_layer = {int(l): i for i, l in enumerate(layers)}
    missing = [h for h in hot_layers if int(h) not in index_of_layer]
    if missing:
        raise SelectionError(f'hot layers {missing} have no down_proj reads')

    groups = {}
    chosen, short = top_positive(a_d, k_adapters)
    groups['top10-adapters'] = {'kind': 'adapters', 'targets': chosen, 'shortfall': short}
    chosen, short = top_positive(flat_d, k_reads)
    groups['global64-reads'] = {'kind': 'reads', 'targets': chosen, 'shortfall': short}
    for name, scope in (('hot64-reads', [int(h) for h in hot_layers]), ('layer64-reads', [int(l) for l in layers])):
        targets, shortfalls = [], {}
        for layer in scope:
            li = index_of_layer[layer]
            picked, short = top_positive(flat_d, k_reads, within=range(li * W, (li + 1) * W))
            targets.extend(picked)
            shortfalls[str(layer)] = short
        groups[name] = {'kind': 'reads', 'targets': targets, 'shortfall': shortfalls, 'layers': scope}

    read_targets = set(t for g in groups.values() if g['kind'] == 'reads' for t in g['targets'])
    adapter_targets = set(groups['top10-adapters']['targets'])
    projections = [p for _, p in adapters]
    for name, g in groups.items():
        if g['kind'] == 'adapters':
            control = match_group(g['targets'], adapter_size, projections, exclude=adapter_targets,
                                  tolerance=tolerance, seed=seeds[name])
            removal, labels = a_r, [_adapter_label(adapters, i) for i in g['targets']]
        else:
            control = match_group(g['targets'], read_size, layer_of, exclude=read_targets,
                                  tolerance=tolerance, seed=seeds[name])
            removal, labels = flat_r, None
        g['prediction'] = group_prediction(removal, g['targets'])
        g['control'] = control
        if control['available']:
            control['prediction'] = group_prediction(removal, control['group'])
        if g['kind'] == 'reads':
            g['reads'] = _reads(g['targets'], layer_of, W)
            g['layer_distribution'] = _counts(int(layer_of[t]) for t in g['targets'])
            if control['available']:
                control['reads'] = _reads(control['group'], layer_of, W)
                control['layer_distribution'] = _counts(int(layer_of[t]) for t in control['group'])
        else:
            g['adapters'] = labels
            g['module_distribution'] = _counts(projections[t] for t in g['targets'])
            if control['available']:
                control['adapters'] = [_adapter_label(adapters, i) for i in control['group']]
                control['module_distribution'] = _counts(projections[t] for t in control['group'])
    harmful = sorted([int(i) for i in range(len(a_d)) if a_d[i] < 0], key=lambda i: (a_d[i], i))[:k_harmful]
    return {'sign_convention': SIGN_CONVENTION, 'groups': groups, 'tolerance': tolerance, 'seeds': dict(seeds),
            'harmful_adapters_reported_only': [_adapter_label(adapters, i) | {'dJdlambda': float(a_d[i])} for i in harmful]}


def _adapter_label(adapters, i):
    layer, projection = adapters[i]
    return {'index': int(i), 'layer': int(layer), 'projection': projection}


def _reads(flat, layer_of, width):
    out = {}
    for t in flat:
        out.setdefault(str(int(layer_of[t])), []).append(int(t % width))
    return {k: sorted(v) for k, v in out.items()}


def _counts(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


# --- hot layers ------------------------------------------------------------------------------------

def hot_layers(source, sha256, *, context='gsm8k_solution', step=64, k=10, module='down',
               arms=('online', 'offline'), combine='union'):
    """Layers whose ``module`` is in each arm's top-k modules by effective write, combined across arms."""
    if file_sha256(source) != sha256:
        raise SelectionError(f'hot-layer source {source} does not match sha256 {sha256}')
    rows = json.loads(Path(source).read_text())['rows']
    per_arm = {}
    for arm in arms:
        ranked = sorted([r for r in rows if r['arm'] == arm and r['context'] == context and r['step'] == step],
                        key=lambda r: (-r['write'], r['layer'], r['module']))
        if not ranked:
            raise SelectionError(f'no {arm} rows for context {context} step {step}')
        per_arm[arm] = {int(r['layer']): rank for rank, r in enumerate(ranked[:k], 1) if r['module'] == module}
    sets = [set(v) for v in per_arm.values()]
    if combine == 'union':
        layers = sorted(set().union(*sets))
    elif combine == 'intersection':
        layers = sorted(set.intersection(*sets))
    else:
        raise SelectionError(f'unknown combine rule {combine!r}')
    return {'layers': layers, 'per_arm_ranks': {a: dict(sorted(v.items())) for a, v in per_arm.items()},
            'source': {'path': str(source), 'sha256': sha256}, 'measurement': 'effective write ‖ΔW·x‖ / ‖W·x‖',
            'context': context, 'step': step, 'k': k, 'module': module, 'arms': list(arms), 'combine': combine}


def hot_layer_sensitivity(source, sha256, *, ks=(5, 10, 15, 20), **kw):
    out = []
    for k in ks:
        union = hot_layers(source, sha256, k=k, combine='union', **kw)
        inter = hot_layers(source, sha256, k=k, combine='intersection', **kw)
        out.append({'k': k, 'per_arm': {a: sorted(v) for a, v in union['per_arm_ranks'].items()},
                    'union': union['layers'], 'intersection': inter['layers']})
    return out


def check_hot_layer_binding(launch_record, summary, *, launch_record_sha256):
    """Refuse a hot-layer list that is missing, not derived from its bound source, or bound after aggregation.

    The aggregate must name the launch record it ran under, so the hot layers were sealed before any
    aggregation output existed; the list must equal the recomputation from the effective-write source.
    """
    bound = (launch_record or {}).get('hot_layers')
    if not bound:
        raise SelectionError('the launch record has no hot-layer list')
    if summary.get('launch_record_sha256') != launch_record_sha256:
        raise SelectionError('the aggregate was not produced under this launch record; the hot-layer list may be late')
    params = {k: bound[k] for k in ('context', 'step', 'k', 'module', 'arms', 'combine')}
    recomputed = hot_layers(bound['source']['path'], bound['source']['sha256'], **params)
    if recomputed['layers'] != list(bound['layers']):
        raise SelectionError(f"hot layers {bound['layers']} differ from the effective-write rule {recomputed['layers']}")
    return recomputed


# --- freeze ----------------------------------------------------------------------------------------

def build_freeze(arm, selection, *, launch_record_sha256, hot, split, inputs):
    return {'schema': 'c-attribution-freeze-v1', 'arm': arm, 'launch_record_sha256': launch_record_sha256,
            'hot_layers': hot, 'question_split': split, 'inputs': inputs, **selection}


def seal_freeze(root, arm, freeze):
    """Seal ``freeze/<arm>`` before any generation unit exists."""
    root = Path(root)
    generation = root / GENERATION_DIR
    if generation.exists() and any(generation.iterdir()):
        raise SelectionError('generation outputs already exist; the freeze must precede them')
    directory = root / FREEZE_DIR / arm
    au.seal_run(directory, {f'{arm}.json': freeze})
    return directory


def read_freeze(root, arm):
    """Verify and return a sealed freeze; generation calls this and refuses to start without it."""
    directory = Path(root) / FREEZE_DIR / arm
    au.verify_run(directory)
    return json.loads((directory / f'{arm}.json').read_text())
