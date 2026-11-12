"""Compact per-text extract from sealed block units (for off-Pod plotting); never touches the units."""
import json

import numpy as np

from test_neuron_audit_selection import ARMS, J, LAYERS, sequences, values, write_stores  # noqa: E402

from pilot_eval import neuron_audit_extract as ex  # noqa: E402


def _run(tmp_path, **kw):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows)
    out = tmp_path / 'extract'
    status = ex.extract([stores], rows, out_dir=out, arms=['untuned'] + ARMS, layers=LAYERS,
                        stats=['mean_abs_z'], **kw)
    return rows, out, status


def test_extract_matches_blocks_in_frozen_row_order(tmp_path):
    rows, out, status = _run(tmp_path)
    assert status['complete'] is True
    index = json.loads((out / 'index.json').read_text())
    ids = [r['id'] for r in rows]  # discovery + heldout, all three groups, sequences order
    assert index['example_ids'] == ids
    assert index['groups'] == [r['group'] for r in rows] and index['splits'] == [r['split'] for r in rows]
    for arm in ['untuned'] + ARMS:
        for layer in LAYERS:
            got = np.load(out / arm / f'mean_abs_z_layer{layer:02d}.npy')
            assert got.shape == (len(rows), J) and got.dtype == np.float32
            np.testing.assert_array_equal(got, values(rows, arm, layer, 'mean_abs_z'))
        assert json.loads((out / arm / 'counts.json').read_text()) == [5] * len(rows)


def test_extract_skips_calibration_rows_and_resumes(tmp_path):
    rows = sequences() + [dict(id='math:cal', group='math', split='calibration')]
    stores = write_stores(tmp_path / 'blocks', rows)
    out = tmp_path / 'extract'
    first = ex.extract([stores], rows, out_dir=out, arms=['untuned'], layers=LAYERS, stats=['mean_abs_z'])
    assert 'math:cal' not in json.loads((out / 'index.json').read_text())['example_ids']
    stamp = (out / 'untuned' / 'mean_abs_z_layer00.npy').stat().st_mtime_ns
    again = ex.extract([stores], rows, out_dir=out, arms=['untuned'], layers=LAYERS, stats=['mean_abs_z'])
    assert again['reused'] == first['written'] and again['written'] == 0
    assert (out / 'untuned' / 'mean_abs_z_layer00.npy').stat().st_mtime_ns == stamp


def test_adapter_term_stats_only_for_trained_arms(tmp_path):
    rows = sequences()
    stores = write_stores(tmp_path / 'blocks', rows)
    out = tmp_path / 'extract'
    ex.extract([stores], rows, out_dir=out, arms=['untuned', 'gold-sft'], layers=[0],
               stats=['mean_abs_z', 'mean_abs_term'])
    assert not (out / 'untuned' / 'mean_abs_term_layer00.npy').exists()
    np.testing.assert_array_equal(np.load(out / 'gold-sft' / 'mean_abs_term_layer00.npy'),
                                  values(rows, 'gold-sft', 0, 'mean_abs_term'))
