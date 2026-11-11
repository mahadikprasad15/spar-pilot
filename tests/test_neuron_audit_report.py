"""Ticket 06 on the 05 fixture: held-out estimands vs frozen controls, no re-ranking, floors, labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_neuron_audit_selection import ARMS, LAYERS, plan, sequences, write_stores  # noqa: E402

from pilot_eval import neuron_audit_report as rep  # noqa: E402
from pilot_eval import neuron_audit_selection as sel  # noqa: E402

QUAL = dict(partial_label=None, review_flags_raised=[], unresolved_floor_layers={},
            fp64_floors={'gold-sft': dict(layers={'0': dict(status='measured', p99=0.5), '1': dict(status='unresolved', p99=None)})})


def build(tmp_path, *, prototype=True, qualification=QUAL, name='r'):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows) if not (tmp_path / 'blocks').exists() else tmp_path / 'blocks'
    selection_root = tmp_path / 'sel'
    if not (selection_root / sel.RUN_SUBDIR / 's').exists():
        sel.select(plan(), store_roots=[stores], sequences=rows, output_root=selection_root, run_id='s',
                   layers=LAYERS, arms=ARMS, prototype=True)
    sel_dir = selection_root / sel.RUN_SUBDIR / 's'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sel_dir.iterdir()}
    results = rep.build(plan(), selection_dir=sel_dir, store_roots=[stores], sequences=rows, qualification=qualification,
                        output_root=tmp_path / 'out', run_id=name, layers=LAYERS, prototype=prototype)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sel_dir.iterdir()}
    assert before == after  # the report never rewrites or re-ranks the frozen selection
    return results, tmp_path / 'out' / rep.RUN_SUBDIR / name


def test_held_out_estimates_beat_frozen_controls(tmp_path):
    results, run_dir = build(tmp_path)
    layer = results['layers']['0']
    a = layer['A']
    assert a['value'] == pytest.approx(1.0, abs=0.1) and a['interval']['low'] > 0.5
    available = [c['value'] for c in a['controls'] if c['value'] is not None]
    assert all(v < 0.3 for v in available)  # tiny top bins can leave A's controls unavailable: reported, not padded
    assert all(c.get('status') == 'unavailable' for c in a['controls'] if c['value'] is None)
    d = layer['D/gold-sft']
    assert d['mean_change'] == pytest.approx(2.0, abs=0.1)
    assert d['sign_agreement'] == dict(agree=2, defined=2, zero_or_undefined=0, fraction=1.0)
    assert all(c['value'] < 0.5 for c in d['controls'])
    b = layer['B/gold-sft']
    assert b['math'] == pytest.approx(1.0, abs=0.01) and all(c['value'] < 0.1 for c in b['controls'])
    assert layer['B_intersect_D/gold-sft'] == dict(primary=0, relative=0)
    assert results['label'].startswith('PROTOTYPE')


def test_floor_fraction_measured_and_unresolved(tmp_path):
    results, _ = build(tmp_path)
    assert results['layers']['0']['D_floor/gold-sft']['status'] == 'ok'
    assert results['layers']['1']['D_floor/gold-sft']['status'] == 'unresolved'
    assert results['layers']['0']['D_floor/online-grpo']['status'] == 'unresolved'  # no floor recorded


def test_report_files_sealed_and_reused(tmp_path):
    results, run_dir = build(tmp_path)
    seal = json.loads((run_dir / 'complete.json').read_text())
    assert {'results.json', 'report.md'} <= set(seal['files'])
    text = (run_dir / 'report.md').read_text()
    for heading in ['## Coverage', '## Primary overlap family', '## Held-out estimates', '## Split-half', '## Limitations']:
        assert heading in text
    assert 'not a CI' in text and 'PROTOTYPE' in text
    again, _ = build(tmp_path)
    assert again == json.loads((run_dir / 'results.json').read_text())


def test_partial_label_rendered(tmp_path):
    qual = dict(QUAL, partial_label='partial study; missing arms: offline-grpo')
    _, run_dir = build(tmp_path, qualification=qual, name='p')
    assert '**Partial study:** partial study; missing arms: offline-grpo' in (run_dir / 'report.md').read_text()


def test_production_report_refuses_prototype_selection(tmp_path):
    build(tmp_path)
    with pytest.raises(sel.SelectionRefused, match='prototype'):
        rep.build(plan(), selection_dir=tmp_path / 'sel' / sel.RUN_SUBDIR / 's', store_roots=[tmp_path / 'blocks'],
                  sequences=sequences(), qualification=QUAL, output_root=tmp_path / 'out', run_id='prod',
                  layers=LAYERS, prototype=False)


def test_held_out_reader_cannot_read_discovery_rows(tmp_path):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows)
    held = rep.HeldOut([stores], rows, prototype=True)
    assert all(r.endswith(('020', '021', '022', '023', '024', '025')) for r in held.ids('math'))
    with pytest.raises(sel.SelectionRefused, match='outside the heldout split'):
        held.matrix('untuned', 0, 'mean_abs_z', ['math:000'])


def test_bootstrap_is_seeded_and_stratified():
    x, y = np.arange(10.0), np.arange(5.0) + 100
    stat = lambda d: float(x[d['a']].mean() - y[d['b']].mean())  # noqa: E731
    one = rep.bootstrap(stat, dict(a=10, b=5), resamples=200, seed=1234)
    assert one == rep.bootstrap(stat, dict(a=10, b=5), resamples=200, seed=1234)
    assert one['low'] < -97.5 < one['high'] and one['method'].startswith('percentile')


def test_sign_agreement_marks_zero_and_undefined():
    base = np.zeros((3, 4))
    arm = np.array([[1.0, -1.0, 0.0, 2.0]] * 3)
    out = rep.sign_agreement(base, arm, [0, 1, 2, 3], [1.0, 1.0, 1.0, 0.0])
    assert out == dict(agree=1, defined=2, zero_or_undefined=2, fraction=0.5)


def test_report_keeps_interpretation_evidence(tmp_path):
    qual = dict(QUAL, fp64_floor_methods={'gold-sft': 'proxy floor'},
                fp64_floors={'gold-sft': dict(layers={'0': dict(status='measured', p99=0.5, kind='proxy floor')})})
    results, run_dir = build(tmp_path, qualification=qual, name='e')
    floor = results['layers']['0']['D_floor/gold-sft']
    assert floor['kind'] == 'proxy floor' and 'PROXY' in floor['note']
    d = results['layers']['0']['D/gold-sft']
    assert d['mean_abs_change'] >= abs(d['mean_change']) - 1e-9 and d['interval_abs']['status'] == 'ok'
    assert all(v['layers'] for v in results['overlaps'].values())  # per-layer overlaps are kept
    text = (run_dir / 'report.md').read_text()
    assert '### Per-layer overlaps' in text and '[proxy floor]' in text and 'mean |change|' in text
    assert 'controls |mean change|' in text and 'controls (math)' in text


def test_d_magnitude_does_not_cancel_opposite_changes():
    base = np.zeros((2, 2))
    arm = np.array([[1.0, -1.0], [1.0, -1.0]])
    assert rep.d_change(base, arm, [0, 1]) == 0.0 and rep.d_magnitude(base, arm, [0, 1]) == 1.0
