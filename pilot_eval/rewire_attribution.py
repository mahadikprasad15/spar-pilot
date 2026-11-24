"""Sparse rewiring ticket 10 (D19): the attribution source manifest and the survivor comparison. CPU, NumPy only.

The comparison never knows which attribution it reads. An **attribution source manifest** lists one entry per
(source, arm, ranking): the score file and array, the measured universe as a [28, 8960] boolean mask, the sign
convention and any named frozen selections of zero-based ``(layer, neuron)``. Adding an arm or a source (gold-sft
or Sampling SFT C attribution, other production-1 layers, joint-audit lists) is a new entry, not new code.

Rules (spec §Comparison, D19):

- Files are read in place and never written. A file whose directory is sealed (``complete.json``) is verified
  with ``attribution_units.verify_run`` first; every input's sha256 goes into the sealed comparison manifest.
- Joins are on the flat index ``layer * 8960 + neuron``, the C freeze's convention; the loader asserts the shape
  and checks every C freeze group's ``targets`` against its ``reads`` before using it.
- A finite score outside the declared universe is refused. A NaN inside it is *unmeasured*: excluded from the
  universe and counted, never read as zero.
- Everything (overlaps, top-K, sign strata, per-layer percentile, null) is computed inside the entry's universe.
  Survivors outside it are counted and reported, not compared.
- The null is layer-matched: random sets with the in-universe survivors' per-layer sizes, drawn without
  replacement inside each layer's universe (``neuron_audit_math.matched_null`` with bins = layers; all sets of one
  comparison share the same draws via ``matched_null_stats``). The exact conditional pmf
  (``exact_overlap_distribution``) gives the upper-tail p; the draws give the reference interval.

Labels carried by every result: exploratory; a different object from attribution (attribution scores how the
``down_proj`` adapter reads neuron j's activation, survivors are neurons whose gate/up *inputs* were changed);
and circular when combined with cohort150 accuracy for entries selected on cohort150. The sign of a score is
local helpfulness, never a survivor probability.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import neuron_audit_math as nm
from pilot_eval.c_attribution_stability import _ranks, spearman, top

SHAPE = (28, 8960)
MANIFEST_SCHEMA = 'rewire-attribution-manifest-v1'
RESULT_SCHEMA = 'rewire-attribution-comparison-v1'
NULL_DRAWS = 10_000
NULL_SEED = 20261123
TOP_K = (64, 256, 1024, 4096)
RANK_BY = ('signed', 'abs')
LABELS = {
    'exploratory': 'Predeclared but exploratory: one seed, one λ, no correction across comparisons.',
    'different-object': ('Attribution scores how the down_proj adapter reads neuron j\'s activation; survivors are '
                         'neurons whose gate/up inputs were changed. Same join key, different measured object.'),
    'circular-with-cohort150-accuracy': ('This source computed its scores and picked its sets on cohort150: '
                                         'any result combining this overlap with cohort150 accuracy is circular.'),
}
SIGN_LABEL = 'sign of the score is local helpfulness under the entry\'s sign convention; never a survivor probability'
PERCENTILE_LABEL = ('rank within the layer\'s measured universe by the entry\'s ranking, (rank - 0.5) / n, ties '
                    'averaged; 0.5 is the layer-matched expectation')


class ManifestError(ValueError):
    pass


# --- loading ---------------------------------------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input(root, spec, seen):
    path = Path(spec['file'])
    path = path if path.is_absolute() else Path(root) / path
    if not path.is_file():
        raise ManifestError(f'input {path} does not exist')
    sealed = (path.parent / 'complete.json').is_file()
    if sealed:
        au.verify_run(path.parent)  # raises on any hash mismatch in the sealed directory
    digest = _sha256(path)
    if spec.get('sha256') and spec['sha256'] != digest:
        raise ManifestError(f'{path}: sha256 {digest} differs from the manifest {spec["sha256"]}')
    if str(path) not in seen:
        seen[str(path)] = dict(path=str(path), sha256=digest, sealed=sealed)
    return path


def _array(path, name):
    with np.load(path, allow_pickle=False) as npz:
        if name not in npz.files:
            raise ManifestError(f'{path} has no array {name!r}')
        return np.asarray(npz[name])


def _layers(values, what):
    layers = [int(x) for x in values]
    if not layers or len(set(layers)) != len(layers) or not all(0 <= x < SHAPE[0] for x in layers):
        raise ManifestError(f'{what}: layers must be distinct and in [0, {SHAPE[0]})')
    return layers


def _scores(root, entry, seen):
    path = _input(root, entry['scores'], seen)
    raw = _array(path, entry['scores']['array']).astype(np.float64)
    layers = entry.get('layers')
    if layers is None:
        if raw.shape != SHAPE:
            raise ManifestError(f'{path}: score shape {raw.shape} is not {SHAPE}')
        return raw
    layers = _layers(layers, 'scores')
    if raw.shape != (len(layers), SHAPE[1]):
        raise ManifestError(f'{path}: score shape {raw.shape} is not {(len(layers), SHAPE[1])} for layers {layers}')
    full = np.full(SHAPE, np.nan)
    full[layers] = raw
    return full


def _universe(root, spec, seen):
    kind = spec.get('kind')
    if kind == 'full':
        return np.ones(SHAPE, bool)
    if kind == 'layers':
        mask = np.zeros(SHAPE, bool)
        mask[_layers(spec['layers'], 'universe')] = True
        return mask
    if kind == 'mask':
        mask = _array(_input(root, spec, seen), spec['array'])
        if mask.shape != SHAPE or mask.dtype != np.bool_:
            raise ManifestError(f'universe mask must be a boolean {SHAPE} array, got {mask.dtype} {mask.shape}')
        return mask.copy()
    raise ManifestError(f'unknown universe kind {kind!r}')


def _c_freeze_group(path, group):
    record = json.loads(Path(path).read_text())
    groups = record.get('groups', {})
    if group not in groups or groups[group].get('kind') != 'reads':
        raise ManifestError(f'{path}: no read group {group!r}')
    g = groups[group]
    targets = sorted(int(t) for t in g['targets'])
    reads = sorted(int(layer) * SHAPE[1] + int(n) for layer, ns in g['reads'].items() for n in ns)
    if targets != reads or len(set(targets)) != len(targets) or \
            any(not 0 <= t < SHAPE[0] * SHAPE[1] for t in targets):
        raise ManifestError(f'{path}: group {group!r} targets do not follow the index convention '
                            f'layer * {SHAPE[1]} + neuron of its reads')
    return targets


def _frozen(root, name, spec, seen):
    kind = spec.get('kind')
    if kind == 'c-freeze':
        return _c_freeze_group(_input(root, spec, seen), spec['group'])
    if kind == 'pairs':
        return sorted(_flat(spec['pairs'], f'frozen set {name!r}'))
    raise ManifestError(f'frozen set {name!r}: unknown kind {kind!r}')


def _flat(pairs, what):
    out = []
    for pair in pairs:
        layer, neuron = (pair['layer'], pair['neuron']) if isinstance(pair, dict) else pair
        layer, neuron = int(layer), int(neuron)
        if not (0 <= layer < SHAPE[0] and 0 <= neuron < SHAPE[1]):
            raise ManifestError(f'{what}: ({layer}, {neuron}) is outside {SHAPE}')
        out.append(layer * SHAPE[1] + neuron)
    if len(set(out)) != len(out):
        raise ManifestError(f'{what}: duplicate (layer, neuron)')
    return out


def load_entry(root, entry):
    seen = {}
    for field in ('source', 'arm', 'ranking', 'scores', 'universe', 'sign_convention'):
        if field not in entry:
            raise ManifestError(f'manifest entry is missing {field!r}')
    rank_by = entry.get('rank_by', 'signed')
    if rank_by not in RANK_BY:
        raise ManifestError(f'rank_by must be one of {RANK_BY}')
    scores = _scores(root, entry, seen)
    declared = _universe(root, entry['universe'], seen)
    finite = np.isfinite(scores)
    if (finite & ~declared).any():
        raise ManifestError(f'{entry["source"]}/{entry["arm"]}: {int((finite & ~declared).sum())} scores lie '
                            'outside the declared universe')
    universe = declared & finite
    frozen, frozen_excluded = {}, {}
    for name, spec in sorted((entry.get('frozen') or {}).items()):
        members = _frozen(root, name, spec, seen)
        outside = [t for t in members if not declared.flat[t]]
        if outside:
            raise ManifestError(f'frozen set {name!r}: {len(outside)} members lie outside the declared universe')
        frozen[name] = [t for t in members if universe.flat[t]]
        frozen_excluded[name] = len(members) - len(frozen[name])
    return dict(source=str(entry['source']), arm=str(entry['arm']), ranking=str(entry['ranking']), rank_by=rank_by,
                sign_convention=entry['sign_convention'], selected_on=entry.get('selected_on'),
                scores=scores, declared=declared, universe=universe,
                excluded_missing=int(declared.sum() - universe.sum()), frozen=frozen,
                frozen_excluded_missing=frozen_excluded, inputs=list(seen.values()), spec=entry)


def load_manifest(path, root=None):
    """Load and check every entry; returns {'entries': [...], ...} with arrays in memory."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ManifestError(f'manifest schema must be {MANIFEST_SCHEMA!r}')
    root = Path(root or manifest.get('root') or path.parent)
    entries, keys = [], set()
    for entry in manifest.get('entries', []):
        loaded = load_entry(root, entry)
        key = (loaded['source'], loaded['arm'], loaded['ranking'])
        if key in keys:
            raise ManifestError(f'duplicate manifest entry {key}')
        keys.add(key)
        entries.append(loaded)
    return dict(schema=MANIFEST_SCHEMA, path=str(path), sha256=_sha256(path), root=str(root), entries=entries)


SURVIVOR_EXPORT_SCHEMA = 'sparse-rewiring-survivors-v1'


def load_survivors_with_provenance(path, *, require_final_sealed=True):
    """Load a survivor export and return ``(pairs, provenance)``.

    Refuses any export with ``survivors_omitted=True`` or ``frozen=False`` so an early checkpoint
    (e.g. step 0 where 250,880 records were omitted) can never silently become a zero-overlap report.
    When ``require_final_sealed=True``, requires a hash-verified sealed directory (``complete.json``)
    and ``schema == 'sparse-rewiring-survivors-v1'`` with ``frozen is True``.
    """
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f'survivor export {p} does not exist')
    sealed = (p.parent / 'complete.json').is_file()
    if sealed:
        au.verify_run(p.parent)
    elif require_final_sealed:
        raise ManifestError(f'survivor export {p} is not in a sealed directory (complete.json required)')
    digest = _sha256(p)
    value = json.loads(p.read_text())
    if isinstance(value, dict):
        if value.get('survivors_omitted') is True:
            raise ManifestError(f'survivor export {p} has survivors_omitted=True (incomplete pre-freeze export)')
        if value.get('frozen') is False:
            raise ManifestError(f'survivor export {p} is unfrozen (pre-freeze export cannot be used as final survivors)')
        if require_final_sealed:
            if value.get('schema') != SURVIVOR_EXPORT_SCHEMA:
                raise ManifestError(
                    f'survivor export {p} schema {value.get("schema")!r} != {SURVIVOR_EXPORT_SCHEMA!r}'
                )
            if value.get('frozen') is not True:
                raise ManifestError(f'survivor export {p} must have frozen=True for final attribution comparison')
        pairs = value['survivors']
        step = value.get('step')
        meta = value.get('meta') if isinstance(value.get('meta'), dict) else {}
    else:
        if require_final_sealed:
            raise ManifestError(f'survivor export {p} must be a sealed {SURVIVOR_EXPORT_SCHEMA!r} dict, not a bare list')
        pairs = value
        step = None
        meta = {}
    parsed = [(p_['layer'], p_['neuron']) if isinstance(p_, dict) else tuple(p_) for p_ in pairs]
    provenance = dict(path=str(p), sha256=digest, sealed=sealed, step=step, arm=meta.get('arm'), rank=meta.get('rank'))
    return parsed, provenance


def load_survivors(path, *, require_final_sealed=False):
    """A survivor export: refuses ``survivors_omitted=True`` and ``frozen=False`` exports."""
    pairs, _ = load_survivors_with_provenance(path, require_final_sealed=require_final_sealed)
    return pairs


def c_production_entries(arms=('online-grpo', 'offline-grpo'), *, groups=('global64-reads', 'layer64-reads')):
    """Manifest entries for ``c-production-1`` (D19), paths relative to the C-attribution artifact root.

    Production-1 adapter-attribution entries need ticket 01 (each GRPO arm's top ``down_proj`` layer and its
    trace/stop ranking files): they are ``{'layers': [layer], 'universe': {'kind': 'layers', ...}}`` entries.
    """
    return [{'source': 'c-production-1', 'arm': arm, 'ranking': 'dJdlambda',
             'scores': {'file': f'c-attribution/aggregate/{arm}/neurons.npz', 'array': 'dJdlambda'},
             'universe': {'kind': 'full'}, 'rank_by': 'signed', 'selected_on': 'gsm8k-test-cohort150',
             'sign_convention': 'dJdλ_j = ∂J/∂λ_j at λ = 1; positive = strengthening the read is locally helpful',
             'frozen': {g: {'kind': 'c-freeze', 'file': f'c-attribution/freeze/{arm}/{arm}.json', 'group': g}
                        for g in groups}} for arm in arms]


# --- comparison --------------------------------------------------------------------------------------

def _bins(survivors, universe):
    """Per-layer counts of in-universe survivors and each counted layer's universe (flat indices)."""
    counts, bins = {}, {}
    for t in survivors:
        counts[t // SHAPE[1]] = counts.get(t // SHAPE[1], 0) + 1
    for layer in counts:
        bins[layer] = np.flatnonzero(universe[layer]) + layer * SHAPE[1]
    return counts, bins


def layer_matched_null(values, survivors, universe, *, draws, seed, chunk=1024):
    """Sums of ``values`` over random sets with the survivors' per-layer sizes, drawn inside the universe."""
    flat = [t for t in _flat(survivors, 'survivors') if universe.flat[t]]
    counts, bins = _bins(flat, universe)
    return nm.matched_null_stats(values, counts, bins, draws=draws, seed=seed, chunk=chunk)


def _key(entry):
    s = entry['scores']
    return np.abs(s) if entry['rank_by'] == 'abs' else s


def _percentiles(entry):
    key, universe = _key(entry), entry['universe']
    out = np.full(SHAPE, np.nan)
    for layer in range(SHAPE[0]):
        idx = np.flatnonzero(universe[layer])
        if idx.size:
            out[layer, idx] = (_ranks(key[layer, idx]) - 0.5) / idx.size
    return out


def _indicator(members):
    v = np.zeros(SHAPE[0] * SHAPE[1])
    v[list(members)] = 1.0
    return v


def _overlap(S, members, draws, counts, bins):
    observed = len(set(S) & set(members))
    pmf = nm.exact_overlap_distribution(members, counts, bins)
    return dict(size=len(members), observed=observed,
                null=nm.reference_interval(draws) | {'mean': float(np.mean(draws)) if draws.size else None},
                exact=dict(expected=float((np.arange(pmf.size) * pmf).sum()),
                           p_upper=float(min(1.0, pmf[observed:].sum())) if observed < pmf.size else 0.0,
                           rule='P(null overlap >= observed), exact conditional pmf'))


def compare(survivors, entry, *, label, draws=NULL_DRAWS, seed=NULL_SEED, top_k=TOP_K, gates=None, chunk=1024):
    """One survivor set against one manifest entry. Returns a JSON-safe dict; null draws under ``_draws``."""
    S_all = _flat(survivors, f'survivors {label!r}')
    declared, universe = entry['declared'].ravel(), entry['universe'].ravel()
    S = sorted(t for t in S_all if universe[t])
    unmeasured = sum(1 for t in S_all if declared[t] and not universe[t])
    counts, bins = _bins(S, entry['universe'])
    pool = np.flatnonzero(universe)
    key = _key(entry).ravel()
    scores = entry['scores'].ravel()

    sets = {('frozen', name): members for name, members in entry['frozen'].items()}
    topk_unavailable = {}
    for k in top_k:
        if k <= pool.size:
            sets[('topk', str(k))] = sorted(pool[top(key[pool], k)].tolist())
        else:
            topk_unavailable[str(k)] = f'universe has {pool.size} measured neurons'
    for name, mask in (('positive', scores > 0), ('negative', scores < 0), ('zero', scores == 0)):
        sets[('sign', name)] = sorted(np.flatnonzero(mask & universe).tolist())
    pct = _percentiles(entry).ravel()

    values = {f'{kind}|{name}': _indicator(members) for (kind, name), members in sets.items()}
    values['percentile|sum'] = np.where(universe, pct, 0.0)
    comparison_seed = nm.derive_seed(seed, label, entry['source'], entry['arm'], entry['ranking'])
    stats = nm.matched_null_stats(values, counts, bins, draws=draws, seed=comparison_seed, chunk=chunk)
    n = len(S)
    out = dict(survivors_label=label, source=entry['source'], arm=entry['arm'], ranking=entry['ranking'],
               rank_by=entry['rank_by'], sign_convention=entry['sign_convention'],
               labels=sorted(_labels(entry)), seed=comparison_seed,
               universe=dict(declared=int(declared.sum()), measured=int(pool.size),
                             excluded_missing=entry['excluded_missing']),
               survivors=dict(total=len(S_all), in_universe=n, unmeasured=unmeasured,
                              outside_universe=len(S_all) - n - unmeasured,
                              per_layer={str(k): v for k, v in sorted(counts.items())},
                              per_layer_all={str(k): v for k, v in sorted(_bins(S_all, np.ones(SHAPE, bool))[0].items())}),
               frozen={}, topk={}, topk_unavailable=topk_unavailable, sign={'label': SIGN_LABEL},
               frozen_excluded_missing=entry['frozen_excluded_missing'])
    for (kind, name), members in sets.items():
        block = {'frozen': out['frozen'], 'topk': out['topk'], 'sign': out['sign']}[kind]
        block[name] = _overlap(S, members, stats[f'{kind}|{name}'], counts, bins)
    null_mean = stats['percentile|sum'] / n if n else stats['percentile|sum']
    observed_mean = float(pct[S].mean()) if n else None
    out['percentile'] = dict(
        label=PERCENTILE_LABEL, mean=observed_mean, median=float(np.median(pct[S])) if n else None,
        null=nm.reference_interval(null_mean) if n else None,
        p_upper=(float((1 + (null_mean >= observed_mean - 1e-12).sum()) / (1 + draws)) if n else None),
        p_rule='(1 + #draws with mean >= observed) / (1 + draws)',
        survivors=[dict(layer=int(t // SHAPE[1]), neuron=int(t % SHAPE[1]), score=float(scores[t]),
                        percentile=float(pct[t])) for t in S])
    if gates is not None:
        g = np.asarray(gates, dtype=np.float64)
        if g.shape != SHAPE:
            raise ManifestError(f'gates must be {SHAPE}, got {g.shape}')
        inside = pool[np.isfinite(g.ravel()[pool])]
        out['gates'] = dict(n=int(inside.size), spearman=spearman(g.ravel()[inside], key[inside]) if inside.size else None,
                            label='secondary: continuous gates at the freeze step vs the ranking, inside the universe')
    out['_draws'] = {k: stats[k] for k in stats if not k.startswith('percentile')} | {'percentile|mean': null_mean}
    return out


def _labels(entry):
    labels = {'exploratory', 'different-object'}
    if entry.get('selected_on') and 'cohort150' in str(entry['selected_on']):
        labels.add('circular-with-cohort150-accuracy')
    return labels


def run(manifest, survivor_sets, *, draws=NULL_DRAWS, seed=NULL_SEED, top_k=TOP_K, gates=None):
    """Every survivor set ({label: pairs}) against every manifest entry."""
    comparisons, all_draws = {}, {}
    for label, survivors in sorted(survivor_sets.items()):
        comparisons[label] = []
        for entry in manifest['entries']:
            result = compare(survivors, entry, label=label, draws=draws, seed=seed, top_k=top_k,
                             gates=(gates or {}).get(label))
            for name, values in result.pop('_draws').items():
                all_draws['|'.join((label, entry['source'], entry['arm'], entry['ranking'], name))] = values
            comparisons[label].append(result)
    return dict(schema=RESULT_SCHEMA, comparisons=comparisons, draws=all_draws, labels=LABELS,
                null=dict(draws=draws, seed=seed, kind='layer-matched (bins = layers, inside each universe)',
                          seed_rule='neuron_audit_math.derive_seed(seed, survivors, source, arm, ranking); '
                                    'all sets of one comparison share its draws'),
                top_k=list(top_k))


def _rows(result):
    for label, comparisons in result['comparisons'].items():
        for c in comparisons:
            for kind in ('frozen', 'topk', 'sign'):
                for name, o in c[kind].items():
                    if name == 'label':
                        continue
                    yield dict(survivors=label, source=c['source'], arm=c['arm'], ranking=c['ranking'], kind=kind,
                               set=name, size=o['size'], observed=o['observed'], in_universe=c['survivors']['in_universe'],
                               null_median=o['null']['median'], null_p2_5=o['null']['p2_5'],
                               null_p97_5=o['null']['p97_5'], expected=o['exact']['expected'],
                               p_upper=o['exact']['p_upper'], labels=';'.join(c['labels']))


def seal(directory, manifest, result, *, survivor_sources=None):
    """Seal compare/: the resolved manifest (with input hashes & survivor export digests), overlap tables and null draws."""
    resolved = dict(schema=MANIFEST_SCHEMA, path=manifest['path'], sha256=manifest['sha256'], root=manifest['root'],
                    survivor_sources=dict(survivor_sources or {}),
                    entries=[dict(source=e['source'], arm=e['arm'], ranking=e['ranking'], rank_by=e['rank_by'],
                                  sign_convention=e['sign_convention'], selected_on=e['selected_on'],
                                  declared=int(e['declared'].sum()), measured=int(e['universe'].sum()),
                                  excluded_missing=e['excluded_missing'],
                                  frozen={k: len(v) for k, v in e['frozen'].items()},
                                  inputs=e['inputs'], spec=e['spec']) for e in manifest['entries']])
    rows = list(_rows(result))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]) if rows else ['survivors'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    buffer = io.BytesIO()
    np.savez(buffer, **result['draws'])
    return au.seal_run(directory, {'manifest.json': resolved,
                                   'overlaps.json': {k: v for k, v in result.items() if k != 'draws'},
                                   'overlaps.csv': text.getvalue(), 'null_draws.npz': buffer.getvalue()})


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare sparse-rewiring survivors with attribution sources (D19).')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', help='attribution artifact root (default: the manifest\'s root or its folder)')
    parser.add_argument('--survivors', nargs=2, action='append', metavar=('LABEL', 'FILE'), required=True)
    parser.add_argument('--out', required=True, help='the run\'s compare/ directory')
    parser.add_argument('--draws', type=int, default=NULL_DRAWS)
    parser.add_argument('--seed', type=int, default=NULL_SEED)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest, args.root)
    survivor_sets, survivor_sources = {}, {}
    for label, path in args.survivors:
        pairs, prov = load_survivors_with_provenance(path, require_final_sealed=True)
        survivor_sets[label] = pairs
        survivor_sources[label] = prov
    result = run(manifest, survivor_sets, draws=args.draws, seed=args.seed)
    print(seal(args.out, manifest, result, survivor_sources=survivor_sources), args.out)


if __name__ == '__main__':
    main()
