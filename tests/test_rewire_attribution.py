"""Sparse rewiring ticket 10: attribution source manifest and comparison (spec tests 13 and 15). CPU, NumPy only."""
import json

import numpy as np
import pytest

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_selection as sel
from pilot_eval import neuron_audit_math as nm
from pilot_eval import rewire_attribution as ra

L, W = ra.SHAPE
DRAWS = 300


def flat(layer, neuron):
    return layer * W + neuron


# --- fixtures --------------------------------------------------------------------------------------

def c_root(tmp_path, arm='online-grpo', seed=0, nan_at=()):
    """A sealed C-attribution aggregate and freeze shaped like c-production-1 (full 28 x 8960)."""
    root = tmp_path / 'c'
    d = np.random.default_rng(seed).normal(size=(L, W))
    for layer, neuron in nan_at:
        d[layer, neuron] = np.nan
    finite = np.where(np.isfinite(d), d, -np.inf)
    globals_ = sel.top_positive(np.where(np.isfinite(d), d, 0.0).ravel(), 64)[0]
    layer64 = []
    for layer in range(L):
        layer64 += sel.top_positive(np.where(np.isfinite(d), d, 0.0).ravel(), 64, within=range(layer * W, (layer + 1) * W))[0]
    groups = {name: {'kind': 'reads', 'targets': t, 'reads': sel._reads(t, np.repeat(np.arange(L), W), W)}
              for name, t in (('global64-reads', globals_), ('layer64-reads', layer64))}
    import io
    buffer = io.BytesIO()
    np.savez(buffer, dJdlambda=d, removal_J=-d)
    au.seal_run(root / 'c-attribution/aggregate' / arm, {'neurons.npz': buffer.getvalue()})
    au.seal_run(root / sel.FREEZE_DIR / arm, {f'{arm}.json': {'schema': 'c-attribution-freeze-v1', 'arm': arm,
                                                              'groups': groups}})
    return root, d, finite


def c_entry(arm='online-grpo'):
    return {'source': 'c-production-1', 'arm': arm, 'ranking': 'dJdlambda',
            'scores': {'file': f'c-attribution/aggregate/{arm}/neurons.npz', 'array': 'dJdlambda'},
            'universe': {'kind': 'full'}, 'rank_by': 'signed', 'sign_convention': 'positive dJdλ = locally helpful',
            'selected_on': 'gsm8k-test-cohort150',
            'frozen': {name: {'kind': 'c-freeze', 'file': f'{sel.FREEZE_DIR}/{arm}/{arm}.json', 'group': name}
                       for name in ('global64-reads', 'layer64-reads')}}


def write_manifest(tmp_path, entries, root):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema': ra.MANIFEST_SCHEMA, 'root': str(root), 'entries': entries}))
    return path


def single_layer(tmp_path, layer, values, name='p1.npz'):
    path = tmp_path / 'c' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, trace=values[None, :], stop=-values[None, :])
    return name


# --- the matched null --------------------------------------------------------------------------------

def test_matched_null_stats_equals_matched_null_for_each_indicator():
    bins = {0: range(0, 40), 1: range(40, 70), 2: range(70, 100)}
    counts = {0: 5, 1: 0, 2: 7}
    xs = {'a': [1, 2, 3, 45, 71, 72, 99], 'b': list(range(60, 90))}
    values = {k: np.isin(np.arange(100), v).astype(float) for k, v in xs.items()}
    stats = nm.matched_null_stats(values, counts, bins, draws=500, seed=7, chunk=64)
    for k, v in xs.items():
        assert np.array_equal(stats[k], nm.matched_null(v, counts, bins, draws=500, seed=7))


def test_layer_matched_null_preserves_per_layer_sizes():
    universe = np.zeros(ra.SHAPE, bool)
    universe[[2, 5, 9]] = True
    universe[5, 100:] = False  # a partial layer
    survivors = [(2, 1), (2, 7), (5, 3), (9, 0), (9, 1), (9, 2)]
    per_layer = {layer: np.zeros(L * W) for layer in (2, 5, 9)}
    for layer in per_layer:
        per_layer[layer][layer * W:(layer + 1) * W] = 1.0
    stats = ra.layer_matched_null(per_layer, survivors, universe, draws=DRAWS, seed=1)
    assert np.all(stats[2] == 2) and np.all(stats[5] == 1) and np.all(stats[9] == 3)
    # Draws stay inside the universe: layer 5 beyond neuron 100 is never drawn.
    outside = np.zeros(L * W)
    outside[5 * W + 100:6 * W] = 1.0
    assert ra.layer_matched_null({'x': outside}, survivors, universe, draws=DRAWS, seed=1)['x'].max() == 0


# --- spec test 13: planted overlap, missing values ------------------------------------------------------

def test_planted_overlap_with_frozen_and_topk_is_recovered(tmp_path):
    root, d, _ = c_root(tmp_path)
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    planted = [tuple(int(x) for x in divmod(t, W)) for t in entry['frozen']['global64-reads'][:20]]
    random = [(3, 11), (3, 12), (17, 4000)]
    random = [p for p in random if flat(*p) not in set(entry['frozen']['global64-reads'])]
    out = ra.compare(planted + random, entry, label='r1', draws=DRAWS)
    g = out['frozen']['global64-reads']
    assert g['observed'] == 20 and g['size'] == 64
    assert g['exact']['p_upper'] < 1e-10 and g['null']['median'] <= 1
    assert out['survivors']['in_universe'] == len(planted) + len(random)
    assert sum(out['survivors']['per_layer'].values()) == len(planted) + len(random)
    assert out['topk']['64']['observed'] >= 20
    assert set(out['labels']) >= {'exploratory', 'different-object', 'circular-with-cohort150-accuracy'}


def test_sign_strata_and_percentile_rank(tmp_path):
    root, d, _ = c_root(tmp_path)
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    layer = 4
    order = np.argsort(d[layer])
    survivors = [(layer, int(j)) for j in order[-10:]] + [(layer, int(j)) for j in order[:3]]
    out = ra.compare(survivors, entry, label='r0', draws=DRAWS)
    assert out['sign']['positive']['observed'] == 10 and out['sign']['negative']['observed'] == 3
    assert out['sign']['zero']['observed'] == 0
    ranks = {(r['layer'], r['neuron']): r['percentile'] for r in out['percentile']['survivors']}
    assert ranks[(layer, int(order[-1]))] == pytest.approx(1 - 0.5 / W)
    assert ranks[(layer, int(order[0]))] == pytest.approx(0.5 / W)
    assert out['percentile']['label'].startswith('rank within the layer')


def test_missing_attribution_values_are_excluded_not_zero(tmp_path):
    missing = [(6, 10), (6, 11)]
    root, d, _ = c_root(tmp_path, nan_at=missing)
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    assert entry['excluded_missing'] == 2 and not entry['universe'][6, 10]
    out = ra.compare(missing + [(6, 12)], entry, label='r1', draws=DRAWS)
    assert out['survivors']['in_universe'] == 1 and out['survivors']['unmeasured'] == 2
    assert out['sign']['zero']['observed'] == 0  # not counted as a zero score
    assert all(r['neuron'] == 12 for r in out['percentile']['survivors'])


def test_empty_survivor_set_is_defined(tmp_path):
    root, _, _ = c_root(tmp_path)
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    out = ra.compare([], entry, label='r1', draws=50)
    assert out['frozen']['global64-reads']['observed'] == 0
    assert out['frozen']['global64-reads']['exact']['p_upper'] == 1.0
    assert out['percentile']['mean'] is None


# --- spec test 15: the manifest is generic ------------------------------------------------------------

def test_third_arm_and_partial_universe_run_without_code_change(tmp_path):
    root, d, _ = c_root(tmp_path)
    c_root(tmp_path, arm='offline-grpo', seed=1)
    layer = 11
    values = np.random.default_rng(3).normal(size=W)
    name = single_layer(tmp_path, layer, values)
    partial = np.zeros(ra.SHAPE, bool)
    partial[[0, 1], :4480] = True
    mask_path = tmp_path / 'c' / 'mask.npz'
    gold = np.where(partial, np.random.default_rng(4).normal(size=ra.SHAPE), np.nan)
    np.savez(mask_path, mask=partial, scores=gold)
    entries = [c_entry(), c_entry('offline-grpo'),
               {'source': 'production-1', 'arm': 'online-grpo', 'ranking': 'trace', 'layers': [layer],
                'scores': {'file': name, 'array': 'trace'}, 'universe': {'kind': 'layers', 'layers': [layer]},
                'rank_by': 'signed', 'sign_convention': 'fixture'},
               {'source': 'c-future', 'arm': 'gold-sft', 'ranking': 'dJdlambda',
                'scores': {'file': 'mask.npz', 'array': 'scores'},
                'universe': {'kind': 'mask', 'file': 'mask.npz', 'array': 'mask'},
                'rank_by': 'abs', 'sign_convention': 'fixture',
                'frozen': {'planted': {'kind': 'pairs', 'pairs': [[0, 1], [0, 2], [1, 3]]}}}]
    manifest = ra.load_manifest(write_manifest(tmp_path, entries, root))
    survivors = [(0, 1), (0, 2), (1, 3), (1, 4000), (1, 5000), (layer, 3), (20, 20)]
    result = ra.run(manifest, {'r1': survivors}, draws=DRAWS)
    rows = {(r['source'], r['arm'], r['ranking']): r for r in result['comparisons']['r1']}
    assert len(rows) == 4
    gold_row = rows[('c-future', 'gold-sft', 'dJdlambda')]
    assert gold_row['survivors']['in_universe'] == 4 and gold_row['survivors']['outside_universe'] == 3
    assert gold_row['frozen']['planted']['observed'] == 3
    assert gold_row['frozen']['planted']['null']['median'] == 0
    p1 = rows[('production-1', 'online-grpo', 'trace')]
    assert p1['survivors']['in_universe'] == 1 and p1['survivors']['per_layer'] == {str(layer): 1}
    assert 'circular-with-cohort150-accuracy' not in p1['labels']


def test_score_outside_declared_universe_is_refused(tmp_path):
    root, _, _ = c_root(tmp_path)
    entry = c_entry()
    entry['universe'] = {'kind': 'layers', 'layers': [0, 1]}
    with pytest.raises(ra.ManifestError, match='outside the declared universe'):
        ra.load_manifest(write_manifest(tmp_path, [entry], root))


def test_loader_refuses_wrong_shape_bad_index_convention_and_tampering(tmp_path):
    root, d, _ = c_root(tmp_path)
    np.savez(tmp_path / 'c' / 'small.npz', x=np.zeros((27, W)))
    bad = c_entry() | {'scores': {'file': 'small.npz', 'array': 'x'}, 'frozen': {}}
    with pytest.raises(ra.ManifestError, match='shape'):
        ra.load_manifest(write_manifest(tmp_path, [bad], root))
    freeze = root / sel.FREEZE_DIR / 'online-grpo' / 'online-grpo.json'
    record = json.loads(freeze.read_text())
    record['groups']['global64-reads']['reads'] = {'0': [1]}
    au.seal_run(tmp_path / 'other' / 'f', {'f.json': record})
    shifted = c_entry() | {'frozen': {'global64-reads': {'kind': 'c-freeze', 'file': str(tmp_path / 'other/f/f.json'),
                                                         'group': 'global64-reads'}}}
    with pytest.raises(ra.ManifestError, match='index convention'):
        ra.load_manifest(write_manifest(tmp_path, [shifted], root))
    npz = root / 'c-attribution/aggregate/online-grpo/neurons.npz'
    np.savez(npz, dJdlambda=d * 2)  # np.savez appends nothing: path already ends in .npz
    with pytest.raises(ValueError, match='failed its hash'):
        ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))


def test_duplicate_entry_and_bad_survivors_refused(tmp_path):
    root, _, _ = c_root(tmp_path)
    with pytest.raises(ra.ManifestError, match='duplicate'):
        ra.load_manifest(write_manifest(tmp_path, [c_entry(), c_entry()], root))
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    for bad in ([(28, 0)], [(0, 8960)], [(0, 1), (0, 1)]):
        with pytest.raises(ra.ManifestError):
            ra.compare(bad, entry, label='r1', draws=10)


def test_secondary_gate_correlation_inside_universe(tmp_path):
    root, d, _ = c_root(tmp_path)
    entry = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))['entries'][0]
    out = ra.compare([(0, 0)], entry, label='r1', draws=10, gates=d.copy())
    assert out['gates']['spearman'] == pytest.approx(1.0)
    assert out['gates']['n'] == L * W


def test_seal_writes_tables_draws_and_resolved_manifest(tmp_path):
    root, _, _ = c_root(tmp_path)
    manifest = ra.load_manifest(write_manifest(tmp_path, [c_entry()], root))
    result = ra.run(manifest, {'r1': [(0, 0), (1, 1)]}, draws=20)
    out = tmp_path / 'compare'
    ra.seal(out, manifest, result)
    names = au.verify_run(out)
    assert {'manifest.json', 'overlaps.json', 'overlaps.csv', 'null_draws.npz'} <= names
    sealed = json.loads((out / 'manifest.json').read_text())
    assert all(len(f['sha256']) == 64 for f in sealed['entries'][0]['inputs'])
    with np.load(out / 'null_draws.npz') as npz:
        assert npz['r1|c-production-1|online-grpo|dJdlambda|frozen|global64-reads'].shape == (20,)
    csv = (out / 'overlaps.csv').read_text().splitlines()
    assert csv[0].startswith('survivors,source,arm,ranking,kind,set')
    assert json.loads((out / 'overlaps.json').read_text())['null']['seed'] == ra.NULL_SEED


def test_cli_reads_survivor_exports_and_seals(tmp_path):
    root, _, _ = c_root(tmp_path)
    manifest = write_manifest(tmp_path, ra.c_production_entries(arms=('online-grpo',)), root)
    r1_dir = tmp_path / 'r1_export'
    r0_dir = tmp_path / 'r0_export'
    au.seal_run(
        r1_dir,
        {
            '1024.json': {
                'schema': ra.SURVIVOR_EXPORT_SCHEMA,
                'step': 1024,
                'frozen': True,
                'survivors_omitted': False,
                'meta': {'arm': 'sparse-gold-sft-r1', 'rank': 1},
                'survivors': [{'layer': 0, 'neuron': 5}, {'layer': 3, 'neuron': 9}],
            }
        },
    )
    au.seal_run(
        r0_dir,
        {
            '1024.json': {
                'schema': ra.SURVIVOR_EXPORT_SCHEMA,
                'step': 1024,
                'frozen': True,
                'survivors_omitted': False,
                'meta': {'arm': 'sparse-gold-sft-r0', 'rank': 0},
                'survivors': [{'layer': 0, 'neuron': 5}],
            }
        },
    )
    out = tmp_path / 'compare'
    ra.main(['--manifest', str(manifest), '--survivors', 'r1', str(r1_dir / '1024.json'),
             '--survivors', 'r0', str(r0_dir / '1024.json'), '--out', str(out), '--draws', '20'])
    result = json.loads((out / 'overlaps.json').read_text())
    assert [c['survivors']['in_universe'] for c in result['comparisons']['r1']] == [2]
    assert [c['survivors']['in_universe'] for c in result['comparisons']['r0']] == [1]
    sealed_manifest = json.loads((out / 'manifest.json').read_text())
    assert set(sealed_manifest['survivor_sources']) == {'r1', 'r0'}
    assert sealed_manifest['survivor_sources']['r1']['sealed'] is True
    assert sealed_manifest['survivor_sources']['r1']['step'] == 1024
    assert len(sealed_manifest['survivor_sources']['r1']['sha256']) == 64


def test_p2_2_survivor_loader_refuses_omitted_unfrozen_and_unsealed_exports(tmp_path):
    root, _, _ = c_root(tmp_path)
    manifest = write_manifest(tmp_path, ra.c_production_entries(arms=('online-grpo',)), root)

    # 1. Step-0 export with survivors_omitted=True must ALWAYS be refused (even non-strict load_survivors)
    omitted_dir = tmp_path / 'step0'
    au.seal_run(
        omitted_dir,
        {'0.json': {'schema': ra.SURVIVOR_EXPORT_SCHEMA, 'step': 0, 'frozen': False, 'survivors_omitted': True, 'survivors': []}},
    )
    with pytest.raises(ra.ManifestError, match='survivors_omitted'):
        ra.load_survivors(omitted_dir / '0.json')
    with pytest.raises(ra.ManifestError, match='survivors_omitted'):
        ra.main(['--manifest', str(manifest), '--survivors', 'r1', str(omitted_dir / '0.json'), '--out', str(tmp_path / 'o1')])

    # 2. Pre-freeze export (e.g. step 960, frozen=False) must ALWAYS be refused
    unfrozen_dir = tmp_path / 'step960'
    au.seal_run(
        unfrozen_dir,
        {'960.json': {'schema': ra.SURVIVOR_EXPORT_SCHEMA, 'step': 960, 'frozen': False, 'survivors_omitted': False,
                      'survivors': [{'layer': 0, 'neuron': 1}]}},
    )
    with pytest.raises(ra.ManifestError, match='unfrozen'):
        ra.load_survivors(unfrozen_dir / '960.json')
    with pytest.raises(ra.ManifestError, match='unfrozen'):
        ra.main(['--manifest', str(manifest), '--survivors', 'r1', str(unfrozen_dir / '960.json'), '--out', str(tmp_path / 'o2')])

    # 3. Unsealed export (no complete.json) must be refused by CLI / require_final_sealed=True
    unsealed = tmp_path / 'loose.json'
    unsealed.write_text(json.dumps({'schema': ra.SURVIVOR_EXPORT_SCHEMA, 'step': 1024, 'frozen': True,
                                    'survivors_omitted': False, 'survivors': [{'layer': 0, 'neuron': 1}]}))
    with pytest.raises(ra.ManifestError, match='sealed'):
        ra.load_survivors_with_provenance(unsealed, require_final_sealed=True)
    with pytest.raises(ra.ManifestError, match='sealed'):
        ra.main(['--manifest', str(manifest), '--survivors', 'r1', str(unsealed), '--out', str(tmp_path / 'o3')])
