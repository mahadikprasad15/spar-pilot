"""Ticket 05 on a labelled fixture: planted lists, independence, ties, controls, nulls, no held-out influence."""
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from pilot_eval import neuron_audit_selection as sel
from pilot_eval.neuron_audit_blocks import BlockStore

REAL_PLAN = Path(__file__).resolve().parents[1] / '.scratch/joint-neuron-adapter-audit/frozen_plan_v1.yaml'
J, LAYERS = 40, [0, 1]
ARMS = ['gold-sft', 'online-grpo']


def plan():
    value = yaml.safe_load(REAL_PLAN.read_text())
    value['statistics'].update(primary_list_budget=2, secondary_list_budget=1)
    value['statistics']['overlap_null']['draws'] = 200
    value['statistics']['split_half'].update(math=[10, 10], general=[10, 10], near_miss=[5, 5])
    return value


def sequences():
    rows = []
    for group, n_disc, n_held in [('math', 20, 6), ('general', 20, 6), ('near_miss', 10, 4)]:
        for i in range(n_disc + n_held):
            rows.append(dict(id=f'{group}:{i:03d}', group=group, split='discovery' if i < n_disc else 'heldout'))
    return rows


def values(rows, arm, layer, stat, heldout_shift=0.0):
    rng = np.random.default_rng(zlib.crc32(f'{arm}|{layer}|{stat}'.encode()))
    base = np.linspace(0.1, 4.0, J)
    out = np.zeros((len(rows), J), np.float32)
    for r, row in enumerate(rows):
        v = base + rng.normal(0, 0.05, J)
        if row['group'] == 'math':
            v[[37, 38]] += 1.0                      # math-selective, inside the top-decile gate
        if arm == 'gold-sft':
            v[[5, 6]] += 2.0 if row['group'] == 'math' else 0.0   # paired change
        if arm == 'online-grpo':
            v[[6, 7]] += 2.0 if row['group'] == 'math' else 0.0
        if stat == 'mean_abs_term':
            v = np.full(J, 0.01) + rng.normal(0, 0.001, J)
            v[[10, 11] if arm == 'gold-sft' else [12, 13]] = 1.0   # reads outside the gate
        if stat == 'mean_z':
            v = v * 0.1
        if row['split'] == 'heldout':
            v = v + heldout_shift
        out[r] = v
    return out


def write_stores(root, rows, *, heldout_shift=0.0, sequences_sha256='seq-sha'):
    store = BlockStore(root)
    for arm in ['untuned'] + ARMS:
        stats = ['mean_z', 'mean_abs_z'] + (['mean_term', 'mean_abs_term'] if arm != 'untuned' else [])
        blocks = {('response_content', l, s): values(rows, arm, l, s, heldout_shift) for l in LAYERS for s in stats}
        store.write_unit(f'{arm}-rows-all', identity=dict(arm=arm, sequences_sha256=sequences_sha256),
                         example_ids=[r['id'] for r in rows], counts={'response_content': [5] * len(rows)},
                         blocks=blocks)
    return root


def run(tmp_path, name='s1', **kw):
    rows = sequences()
    stores = write_stores(tmp_path / f'blocks-{name}', rows, **kw)
    return sel.select(plan(), store_roots=[stores], sequences=rows, output_root=tmp_path / f'out-{name}',
                      run_id='sel', layers=LAYERS, arms=ARMS, prototype=True), tmp_path / f'out-{name}' / sel.RUN_SUBDIR / 'sel'


def lists(run_dir):
    return json.loads((run_dir / 'lists.json').read_text())['primary']


def test_planted_neurons_recovered_by_independent_lists(tmp_path):
    summary, run_dir = run(tmp_path)
    got = lists(run_dir)
    assert got['A']['0']['indices'] == [37, 38]
    assert got['B/gold-sft']['0']['indices'] == [10, 11]          # not restricted to A's gate
    assert got['B/online-grpo']['0']['indices'] == [12, 13]
    assert got['D/gold-sft']['0']['indices'] == [5, 6]
    assert got['D/online-grpo']['1']['indices'] == [6, 7]
    assert got['D/gold-sft']['0']['signed'][0] > 0
    assert summary['label'].startswith('PROTOTYPE')
    assert got['A']['0']['gate_size'] == 4


def test_overlaps_both_directions_pooled_and_labelled(tmp_path):
    _, run_dir = run(tmp_path)
    overlaps = json.loads((run_dir / 'overlaps.json').read_text())
    forward, backward = overlaps['D/gold-sft vs D/online-grpo'], overlaps['D/online-grpo vs D/gold-sft']
    assert forward['layers']['0']['fraction'] == 0.5 and backward['layers']['0']['fraction'] == 0.5
    assert forward['pooled']['layers'] == 2 and forward['pooled']['observed'] == 0.5
    assert 'not a confidence interval' in forward['pooled']['reference']['note']
    assert forward['layers']['0']['seed'] != backward['layers']['0']['seed']
    assert set(overlaps) >= {'A vs B/gold-sft', 'A vs D/gold-sft', 'B/gold-sft vs D/gold-sft'}


def test_deterministic_and_unaffected_by_held_out_values(tmp_path):
    _, a = run(tmp_path, 'a')
    _, b = run(tmp_path, 'b', heldout_shift=5.0)
    for name in ['lists.json', 'controls.json', 'overlaps.json', 'split_half.json']:
        assert (a / name).read_text() == (b / name).read_text(), name


def test_controls_bin_matched_exclude_members_and_reproducible(tmp_path):
    _, run_dir = run(tmp_path)
    controls = json.loads((run_dir / 'controls.json').read_text())
    bins = {int(b): set(v) for b, v in json.loads((run_dir / 'bins.json').read_text())['0'].items()}
    label = {i: b for b, members in bins.items() for i in members}
    members = lists(run_dir)['D/gold-sft']['0']['indices']
    sets = controls['D/gold-sft']['0']
    assert len(sets) == 5 and all(s['status'] == 'ok' for s in sets)
    for s in sets:
        assert not set(s['indices']) & set(members)
        assert sorted(label[i] for i in s['indices']) == sorted(label[i] for i in members)
    assert len({tuple(s['indices']) for s in sets}) > 1


def test_sealed_selection_reused_and_corruption_refused(tmp_path):
    summary, run_dir = run(tmp_path)
    rows = sequences()
    again = sel.select(plan(), store_roots=[tmp_path / 'blocks-s1'], sequences=rows, output_root=tmp_path / 'out-s1',
                       run_id='sel', layers=LAYERS, arms=ARMS, prototype=True)
    assert again == summary
    (run_dir / 'lists.json').write_text('{}')
    with pytest.raises(sel.SelectionRefused, match='integrity'):
        sel.select(plan(), store_roots=[tmp_path / 'blocks-s1'], sequences=rows, output_root=tmp_path / 'out-s1',
                   run_id='sel', layers=LAYERS, arms=ARMS, prototype=True)


def test_refuses_mismatched_manifest_missing_arm_and_held_out_rows(tmp_path):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows, sequences_sha256='other')
    with pytest.raises(sel.SelectionRefused, match='different frozen sequences'):
        sel.Discovery([stores], rows, sequences_sha256='seq-sha')
    data = sel.Discovery([stores], rows, prototype=True)
    with pytest.raises(sel.SelectionRefused, match='held-out or calibration rows are never read'):
        data.matrix('untuned', 0, 'mean_abs_z', ['math:025'])
    with pytest.raises(sel.SelectionRefused, match='no sealed units'):
        data.matrix('offline-grpo', 0, 'mean_abs_z', ['math:000'])
    with pytest.raises(sel.SelectionRefused, match='limited to discovery rows'):
        data.ids('math', split='heldout')


def test_top_k_ties_by_index_and_underfilled_not_padded():
    scores = np.array([1.0, 3.0, 3.0, 2.0, 3.0])
    assert sel.top_k(scores, np.ones(5, bool), 2)['indices'] == [1, 2]
    small = sel.top_k(scores, np.array([1, 0, 0, 0, 1], bool), 3)
    assert small['indices'] == [0, 4] and small['underfilled'] is True and small['size'] == 2


def test_rank_bins_follow_percentile_edges_and_ties():
    activity = np.array([1.0] * 10 + list(range(10)), float)
    bins, labels = sel.rank_bins(activity, np.ones(20, bool), [0, 50, 75, 90, 95, 99, 100])
    assert [len(bins[b]) for b in range(6)] == [10, 5, 3, 1, 0, 1]
    assert labels[:10].tolist().count(labels[0]) >= 1 and (labels >= 0).all()


def test_zero_layer_median_and_zero_floor_are_unresolved():
    zeros = np.zeros((4, 6))
    assert sel.list_a(zeros, zeros, np.ones(6, bool), budget=2, gate_fraction=0.1,
                      low_spread_factor=1e-6)['status'] == 'unresolved'
    primary, relative = sel.list_d(zeros, zeros + 1, zeros, zeros, np.ones(6, bool), budget=2, floor_percentile=50)
    assert primary['status'] == 'ok' and relative['status'] == 'unresolved'


def test_low_spread_neurons_excluded_from_a():
    math_vals = np.tile(np.linspace(1, 2, 10), (5, 1))
    math_vals[:, 9] = 3.0                              # top activity, zero spread
    background = math_vals.copy()
    background[:, :9] -= np.random.default_rng(0).normal(0, 0.1, (5, 9))
    out = sel.list_a(math_vals, background, np.ones(10, bool), budget=1, gate_fraction=0.2, low_spread_factor=1e-6)
    assert 9 in out['low_spread'] and 9 not in out['indices']


def test_controls_unavailable_when_bin_too_small():
    bins = {0: np.array([0, 1]), 1: np.array([2])}
    labels = np.array([0, 0, 1])
    out = sel.matched_controls([2], labels, bins, k=2, seed_base=20000, label=('X', 0))
    assert all(c['status'] == 'unavailable' for c in out)


def test_pooled_overlap_excludes_undefined_layers():
    per_layer = {0: dict(status='ok', fraction=0.5, null_fraction=[0.0, 0.5]),
                 1: dict(status='undefined'), 2: dict(status='ok', fraction=1.0, null_fraction=[0.5, 0.5])}
    pooled = sel.pooled_overlap(per_layer)
    assert pooled['observed'] == 0.75 and pooled['layers'] == 2 and pooled['excluded_layers'] == ['1']


def test_rehearsal_subset_uses_only_rows_every_arm_collected(tmp_path):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows)
    with pytest.raises(sel.SelectionRefused, match='prototype'):
        sel.Discovery([stores], rows, sequences_sha256='seq-sha', available_only=True)
    data = sel.Discovery([stores], rows + [dict(id='math:999', group='math', split='discovery')],
                         prototype=True, available_only=True)
    assert 'math:999' not in data.ids('math') and len(data.ids('math')) == 20
