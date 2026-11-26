"""Ticket 05: freeze independent candidate lists, matched controls and overlap nulls (discovery only).

Reads sealed block units (ticket 03a/04 format) and the frozen sequences, never held-out or
calibration rows, and writes a sealed selection under
``<output_root>/runs/joint-neuron-adapter-audit/selection/<run_id>/``.

Lists per layer (plan ``selection``): A and A-nb (top-decile base-activity gate, then signed
standardized math-minus-background difference), B per arm (mean |a_j z_j| on math), D per arm
(|mean paired change in mean |z||, signed change kept) and relative D (secondary). Lists are
built independently: B and D are never restricted to A. Ties go to the lower neuron index.
Underfilled lists keep their real size; undefined statistics are reported, never set to zero.

Matched controls and the overlap null use rank-based base-activity bins on the layer's eligible
universe. Null summaries are randomization reference intervals, not confidence intervals.
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pilot_eval import neuron_audit_math as nm
from pilot_eval.neuron_audit_blocks import BlockStore

RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/selection'
GROUP = 'response_content'


class SelectionRefused(ValueError):
    pass


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- pure list math -----------------------------------------------------------------------------

def rank_bins(activity, eligible, edges):
    """Rank-based bins over eligible neurons (ties by index); returns {bin: indices}, -1 for ineligible."""
    idx = np.flatnonzero(eligible)
    order = idx[np.lexsort((idx, activity[idx]))]  # ascending activity, then index
    n = len(order)
    cuts = [int(math.floor(e / 100 * n)) for e in edges]
    cuts[-1] = n
    labels = np.full(activity.shape, -1, dtype=int)
    bins = {}
    for b in range(len(edges) - 1):
        members = order[cuts[b]:cuts[b + 1]]
        labels[members] = b
        bins[b] = np.sort(members)
    return bins, labels


def top_k(scores, candidates, k):
    """Descending by score, ties by neuron index; never pads an underfilled list."""
    candidates = np.flatnonzero(candidates)
    order = candidates[np.lexsort((candidates, -scores[candidates]))]
    chosen = order[:k]
    return dict(indices=sorted(int(i) for i in chosen), size=int(len(chosen)), budget=int(k),
                underfilled=bool(len(chosen) < k), candidates=int(len(candidates)))


def pooled_sd(x, y):
    """sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)) over per-example values (sample variance)."""
    n1, n2 = x.shape[0], y.shape[0]
    if n1 + n2 - 2 <= 0:
        raise ValueError('pooled SD needs at least two examples in total')
    return np.sqrt(((n1 - 1) * x.var(axis=0, ddof=1) + (n2 - 1) * y.var(axis=0, ddof=1)) / (n1 + n2 - 2))


def list_a(base_math, base_background, eligible, *, budget, gate_fraction, low_spread_factor):
    activity = base_math.mean(axis=0)
    layer_median = float(np.median(activity[eligible])) if eligible.any() else 0.0
    if layer_median == 0.0:
        return dict(status='unresolved', reason='layer median of base mean |z| is zero')
    n_gate = int(math.ceil(gate_fraction * eligible.sum()))
    gate = np.zeros_like(eligible)
    gate[top_k(activity, eligible, n_gate)['indices']] = True
    spread = pooled_sd(base_math, base_background)
    low = gate & (spread < low_spread_factor * layer_median)
    score = np.full(activity.shape, np.nan)
    ok = gate & ~low
    score[ok] = (base_math.mean(axis=0)[ok] - base_background.mean(axis=0)[ok]) / spread[ok]
    chosen = top_k(np.nan_to_num(score, nan=-np.inf), ok, budget)
    return dict(status='ok', **chosen, gate_size=n_gate, low_spread=sorted(int(i) for i in np.flatnonzero(low)),
                scores=[float(score[i]) for i in chosen['indices']], layer_median=layer_median)


def list_b(terms_math, eligible, *, budget):
    score = terms_math.mean(axis=0)
    chosen = top_k(score, eligible, budget)
    return dict(status='ok', **chosen, scores=[float(score[i]) for i in chosen['indices']])


def list_d(base_math, arm_math, base_mean_z, arm_mean_z, eligible, *, budget, floor_percentile):
    d = arm_math - base_math
    signed = d.mean(axis=0)
    primary = np.abs(signed)
    chosen = top_k(primary, eligible, budget)
    activity = base_math.mean(axis=0)
    floor = float(np.percentile(activity[eligible], floor_percentile)) if eligible.any() else 0.0
    out = dict(status='ok', **chosen, scores=[float(primary[i]) for i in chosen['indices']],
               signed=[float(signed[i]) for i in chosen['indices']],
               mean_z_change=[float((arm_mean_z - base_mean_z).mean(axis=0)[i]) for i in chosen['indices']])
    if floor == 0.0:
        relative = dict(status='unresolved', reason='activity floor (layer percentile) is zero')
    else:
        meets = eligible & (activity >= floor) & (activity > 0)
        rel = np.full(activity.shape, np.nan)
        rel[meets] = primary[meets] / activity[meets]
        relative = dict(status='ok', floor=floor, **top_k(np.nan_to_num(rel, nan=-np.inf), meets, budget))
        relative['scores'] = [float(rel[i]) for i in relative['indices']]
    return out, relative


def matched_controls(members, labels, bins, *, k, seed_base, label):
    """k bin-matched control sets per list, without replacement, excluding the list's own members."""
    members = set(members)
    counts = Counter(int(labels[i]) for i in members)
    controls = []
    for j in range(k):
        rng = np.random.Generator(np.random.PCG64(nm.derive_seed(seed_base, *label, j)))
        chosen, problem = [], None
        for b, c in sorted(counts.items()):
            pool = np.array([i for i in bins[b] if i not in members])
            if c > len(pool):
                problem = f'bin {b} has {len(pool)} non-member neurons for {c} matched draws'
                break
            chosen += sorted(int(i) for i in rng.choice(pool, size=c, replace=False))
        controls.append(dict(index=j, status='unavailable', reason=problem) if problem
                        else dict(index=j, status='ok', indices=sorted(chosen)))
    return controls


def overlap(fixed, randomized, labels, bins, *, draws, seed):
    """Observed |X & Y| / |Y| and the bin-matched null for Y (fixed X), per layer."""
    if not fixed or not randomized:
        return dict(status='undefined', reason='empty list')
    counts = Counter(int(labels[i]) for i in randomized)
    observed = len(set(fixed) & set(randomized))
    try:
        null = nm.matched_null(fixed, counts, {b: v for b, v in bins.items()}, draws=draws, seed=seed)
    except nm.Unavailable as exc:
        return dict(status='unavailable', reason=str(exc), observed=observed, size=len(randomized))
    return dict(status='ok', observed=observed, size=len(randomized), fraction=observed / len(randomized),
                null_fraction=(null / len(randomized)).tolist(), seed=seed,
                reference=nm.reference_interval(null / len(randomized)))


def pooled_overlap(per_layer):
    """Equal-layer average of overlap fractions over defined layers, null aligned by draw index."""
    defined = {l: v for l, v in per_layer.items() if v.get('status') == 'ok'}
    if not defined:
        return dict(status='undefined', layers=0)
    observed = float(np.mean([v['fraction'] for v in defined.values()]))
    null = np.mean(np.stack([np.asarray(v['null_fraction']) for v in defined.values()]), axis=0)
    return dict(status='ok', observed=observed, layers=len(defined), excluded_layers=sorted(
        str(l) for l in per_layer if l not in defined), reference=nm.reference_interval(null))


# --- data access --------------------------------------------------------------------------------

class Discovery:
    """Discovery-only matrices [examples, J] gathered from sealed block units, by example ID order."""

    split = 'discovery'  # selection is locked to discovery rows; the 06 report subclasses this for held-out

    CONTEXT = ['sequences_sha256', 'batch_manifest_sha256', 'plan_core_sha256', 'qualification', 'rehearsal']

    def __init__(self, store_roots, sequences, *, sequences_sha256=None, prototype=False, available_only=False,
                 expected_context=None):
        """``expected_context`` (production): identity fields every unit must carry (e.g. plan core and
        qualification). Units must also agree with each other on ``CONTEXT`` and never repeat an (arm, example)."""
        self.rows = {r['id']: r for r in sequences}
        self.prototype, self.available_only = prototype, available_only
        if available_only and not prototype:
            raise SelectionRefused('available_only (rehearsal subsets) requires prototype mode')
        self.where, self.context = {}, None
        for root in store_roots:
            store = BlockStore(root)
            for key, state in store.status().items():
                if state != 'sealed':
                    continue
                index = store.index(key)
                ident = index['identity']
                if not prototype and ident.get('rehearsal'):  # rehearsal units only in prototype mode
                    raise SelectionRefused(f'unit {key} comes from a rehearsal run; not valid for selection')
                if not prototype and ident.get('sequences_sha256') != sequences_sha256:
                    raise SelectionRefused(f'unit {key} was collected from a different frozen sequences manifest')
                context = {f: ident.get(f) for f in self.CONTEXT}
                if self.context is None:
                    self.context = context
                elif context != self.context:
                    raise SelectionRefused(f'unit {key} comes from a different run context ({context} vs '
                                           f'{self.context}); never mix runs')
                for field, value in (expected_context or {}).items():
                    if ident.get(field) != value:
                        raise SelectionRefused(f'unit {key}: {field} is {ident.get(field)}, expected {value}')
                arm_rows = self.where.setdefault(ident['arm'], {})
                for row, example in enumerate(index['example_ids']):
                    if example in arm_rows:
                        raise SelectionRefused(f"arm {ident['arm']} example {example} appears in both "
                                               f'{arm_rows[example][1]} and {key}; refusing duplicates')
                    arm_rows[example] = (store, key, row)
        self._cache, self._cache_layer = {}, None  # one layer's blocks at a time: memory ~ 1/28 of the run

    def ids(self, group, split=None):
        split = split or self.split
        chosen = sorted(i for i, r in self.rows.items() if r['group'] == group and r['split'] == split)
        if split != self.split:
            raise SelectionRefused(f'this reader is limited to {self.split} rows')
        if self.available_only:  # rehearsal: only rows every arm actually collected
            chosen = [i for i in chosen if all(i in rows for rows in self.where.values())]
        return chosen

    def counts(self, arm, ids, group=GROUP):
        """Per-example counted tokens (for token-weighted summaries)."""
        out = []
        for i in ids:
            store, key, row = self.where[arm][i]
            out.append(store.index(key)['counts'][group][row])
        return out

    def matrix(self, arm, layer, stat, ids):
        if any(self.rows[i]['split'] != self.split for i in ids):
            raise SelectionRefused(f'rows outside the {self.split} split requested (held-out or calibration rows '
                                   'are never read by selection)')
        if arm not in self.where:
            raise SelectionRefused(f'arm {arm} has no sealed units')
        if not ids:
            raise SelectionRefused(f'no {self.split} rows requested for arm {arm}')
        missing = [i for i in ids if i not in self.where[arm]]
        if missing:
            raise SelectionRefused(f'arm {arm} lacks sealed rows for {len(missing)} {self.split} examples')
        if layer != self._cache_layer:  # 05/06 loop layer by layer; never hold the whole run in memory
            self._cache.clear()
            self._cache_layer = layer
        out = []
        for i in ids:
            store, key, row = self.where[arm][i]
            cache_key = (arm, key, layer, stat)
            if cache_key not in self._cache:
                self._cache[cache_key] = store.read_block(key, GROUP, layer, stat)
            out.append(self._cache[cache_key][row])
        return np.stack(out).astype(np.float64)


# --- orchestration --------------------------------------------------------------------------------

def _split_halves(ids, sizes, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    order = [ids[i] for i in rng.permutation(len(ids))]
    first = sizes[0] if sum(sizes) == len(ids) else len(ids) // 2
    return sorted(order[:first]), sorted(order[first:])


def build_lists(data, ids, arms, layers, plan):
    stats, sel = plan['statistics'], plan.get('selection', {})
    budgets = dict(primary=int(stats['primary_list_budget']), secondary=int(stats['secondary_list_budget']))
    low_spread, gate = float(stats['low_spread_factor']), float(stats['a_gate_fraction'])
    out = {}
    for layer in layers:
        base_math = data.matrix('untuned', layer, 'mean_abs_z', ids['math'])
        base_gen = data.matrix('untuned', layer, 'mean_abs_z', ids['general'])
        base_nb = data.matrix('untuned', layer, 'mean_abs_z', ids['near_miss'])
        base_mz = data.matrix('untuned', layer, 'mean_z', ids['math'])
        eligible = np.isfinite(base_math).all(0) & np.isfinite(base_gen).all(0) & np.isfinite(base_nb).all(0)
        per_arm = {}
        for arm in arms:
            arm_math = data.matrix(arm, layer, 'mean_abs_z', ids['math'])
            terms = data.matrix(arm, layer, 'mean_abs_term', ids['math'])
            arm_mz = data.matrix(arm, layer, 'mean_z', ids['math'])
            per_arm[arm] = (arm_math, terms, arm_mz)
            eligible &= np.isfinite(arm_math).all(0) & np.isfinite(terms).all(0)
        layer_out = dict(eligible=int(eligible.sum()), activity=base_math.mean(axis=0))
        for name, k in budgets.items():
            entry = {}
            entry['A'] = list_a(base_math, base_gen, eligible, budget=k, gate_fraction=gate, low_spread_factor=low_spread)
            entry['A-nb'] = list_a(base_math, base_nb, eligible, budget=k, gate_fraction=gate,
                                   low_spread_factor=low_spread)
            for arm, (arm_math, terms, arm_mz) in per_arm.items():
                entry[f'B/{arm}'] = list_b(terms, eligible, budget=k)
                entry[f'D/{arm}'], entry[f'D-relative/{arm}'] = list_d(
                    base_math, arm_math, base_mz, arm_mz, eligible, budget=k,
                    floor_percentile=float(stats.get('activity_floor_percentile', 50)))
            layer_out[name] = entry
        layer_out['eligible_mask'] = eligible
        out[layer] = layer_out
    return out


def _comparisons(arms):
    pairs = [('A', f'B/{a}') for a in arms] + [('A', f'D/{a}') for a in arms] + [(f'B/{a}', f'D/{a}') for a in arms]
    for i, a in enumerate(sorted(arms)):
        for b in sorted(arms)[i + 1:]:
            pairs += [(f'D/{a}', f'D/{b}'), (f'D/{b}', f'D/{a}')]  # both directions; first fixed
    return pairs


def select(plan, *, store_roots, sequences, output_root, run_id, layers, arms, sequences_sha256=None,
           prototype=False, available_only=False, expected_context=None, partial_label=None):
    run_dir = Path(output_root) / RUN_SUBDIR / run_id
    if (run_dir / 'complete.json').exists():
        seal = json.loads((run_dir / 'complete.json').read_text())
        if any(_sha(run_dir / n) != h for n, h in seal['files'].items()):
            raise SelectionRefused('sealed selection failed its integrity check')
        return json.loads((run_dir / 'summary.json').read_text())
    data = Discovery(store_roots, sequences, sequences_sha256=sequences_sha256, prototype=prototype,
                     available_only=available_only, expected_context=expected_context)
    stats = plan['statistics']
    groups = ['math', 'general', 'near_miss']
    ids = {g: data.ids(g) for g in groups}
    too_few = {g: len(v) for g, v in ids.items() if len(v) < 2}
    if too_few:
        raise SelectionRefused(f'too few discovery rows collected for {too_few} (need at least 2 per group); '
                               'a rehearsal must collect batches that cover every group (rehearsal_batch_choice=cover)')
    edges = stats['activity_bins']['percentile_edges']
    null_cfg = stats['overlap_null']
    draws, null_base = int(null_cfg['draws']), int(null_cfg['seeds'][0])
    controls_cfg = plan['selection']['matched_controls']
    lists = build_lists(data, ids, arms, layers, plan)
    bins_by_layer = {l: rank_bins(lists[l]['activity'], lists[l]['eligible_mask'], edges) for l in layers}
    frozen_lists, controls, overlaps = {}, {}, {}
    for budget in ['primary', 'secondary']:
        for l in layers:
            bins, labels = bins_by_layer[l]
            for name, entry in lists[l][budget].items():
                frozen_lists.setdefault(budget, {}).setdefault(name, {})[str(l)] = entry
                if budget == 'primary' and entry.get('status') == 'ok':
                    controls.setdefault(name, {})[str(l)] = matched_controls(
                        entry['indices'], labels, bins, k=int(controls_cfg['per_list']),
                        seed_base=int(controls_cfg['seed_base']), label=(name, l))
        if budget != 'primary':
            continue
        for fixed, rand in _comparisons(arms):
            per_layer = {}
            for l in layers:
                bins, labels = bins_by_layer[l]
                X, Y = lists[l][budget][fixed], lists[l][budget][rand]
                if X.get('status') != 'ok' or Y.get('status') != 'ok':
                    per_layer[l] = dict(status='undefined', reason='list unresolved')
                    continue
                per_layer[l] = overlap(X['indices'], Y['indices'], labels, bins, draws=draws,
                                       seed=nm.derive_seed(null_base, f'{fixed}|{rand}', l, 'fixed-first'))
            overlaps[f'{fixed} vs {rand}'] = dict(fixed=fixed, randomized=rand,
                                                  layers={str(l): {k: v for k, v in r.items() if k != 'null_fraction'}
                                                          for l, r in per_layer.items()},
                                                  pooled=pooled_overlap(per_layer))
    halves_cfg = stats['split_half']
    halves = {g: _split_halves(ids[g], halves_cfg.get(g, [len(ids[g]) // 2] * 2), int(halves_cfg['seed']))
              for g in groups}
    half_lists = [build_lists(data, {g: halves[g][h] for g in groups}, arms, layers, plan) for h in (0, 1)]
    stability = {}
    for name in lists[layers[0]]['primary']:
        per_layer = {}
        for l in layers:
            a, b = half_lists[0][l]['primary'][name], half_lists[1][l]['primary'][name]
            if a.get('status') == 'ok' and b.get('status') == 'ok' and a['indices']:
                per_layer[str(l)] = len(set(a['indices']) & set(b['indices'])) / len(a['indices'])
        stability[name] = per_layer
    bins_out = {str(l): {str(b): v.tolist() for b, v in bins_by_layer[l][0].items()} for l in layers}
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / 'lists.json', frozen_lists)
    _write_json(run_dir / 'controls.json', controls)
    _write_json(run_dir / 'overlaps.json', overlaps)
    _write_json(run_dir / 'bins.json', bins_out)
    _write_json(run_dir / 'split_half.json', dict(halves=halves, stability=stability, seed=halves_cfg['seed']))
    summary = dict(run_id=run_id, prototype=prototype, plan_sha256=plan.get('_plan_sha256'),
                   sequences_sha256=sequences_sha256, arms=list(arms), layers=list(layers),
                   discovery_counts={g: len(v) for g, v in ids.items()}, comparisons=sorted(overlaps),
                   unit_context=data.context, partial_label=partial_label,
                   primary_family={k: v['pooled'] for k, v in overlaps.items()},
                   note='Overlap intervals are bin-conditional randomization references, not confidence intervals. '
                        'All comparisons are exploratory with one training seed.',
                   label='PROTOTYPE (fixture input)' if prototype else None, created_unix=time.time())
    _write_json(run_dir / 'summary.json', summary)
    files = ['lists.json', 'controls.json', 'overlaps.json', 'bins.json', 'split_half.json', 'summary.json']
    _write_json(run_dir / 'complete.json', dict(files={n: _sha(run_dir / n) for n in files}))
    return summary


def main(argv=None):
    from pilot_eval import neuron_audit_plan as plan_mod
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stores', nargs='+', required=True, help='block store roots (contain units/)')
    parser.add_argument('--sequences_run', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--layers', required=True, help='e.g. 0-27')
    parser.add_argument('--rehearsal', action='store_true',
                        help='Colab rehearsal: prototype output over the discovery rows every arm collected')
    parser.add_argument('--qualification_dir', required=True,
                        help='03b run dir: its qualified arms define the (possibly partial) study')
    args = parser.parse_args(argv)
    plan = plan_mod.load(args.plan, '03b' if args.rehearsal else '04', rehearsal=args.rehearsal)
    seq_dir = Path(args.sequences_run)
    sequences = [json.loads(l) for l in (seq_dir / 'sequences.jsonl').read_text().splitlines()]
    manifest = json.loads((seq_dir / 'manifest.json').read_text())
    lo, hi = (int(x) for x in args.layers.split('-'))
    qualification = json.loads((Path(args.qualification_dir) / 'qualification.json').read_text())
    arms = [a for a in plan['arms'] if a != 'untuned' and a in qualification['arms_qualified']]
    expected = None if args.rehearsal else dict(plan_core_sha256=plan['_plan_core_sha256'],
                                                batch_manifest_sha256=qualification['batch_manifest_sha256'],
                                                qualification=qualification['run_id'], rehearsal=False)
    summary = select(plan, store_roots=args.stores, sequences=sequences, output_root=args.output_root,
                     run_id=args.run_id, layers=list(range(lo, hi + 1)), arms=arms,
                     sequences_sha256=manifest['sequences_sha256'], prototype=args.rehearsal,
                     available_only=args.rehearsal, expected_context=expected,
                     partial_label=qualification.get('partial_label'))
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
