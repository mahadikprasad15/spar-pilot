"""Ticket 03a dense FP32 block units: sealing, reuse, NaN rules, interruption and corruption."""
import json

import numpy as np
import pytest

from pilot_eval.neuron_audit_blocks import BlockStore, CorruptUnit


def unit(rows=3, width=5, empty_row=None):
    rng = np.random.default_rng(0)
    counts = {'response_content': [4] * rows, 'termination': [2] * rows}
    blocks = {(g, l, s): rng.normal(size=(rows, width)).astype(np.float32)
              for g in counts for l in range(2) for s in ['mean_z', 'mean_abs_z']}
    if empty_row is not None:
        counts['termination'][empty_row] = 0
        for key in blocks:
            if key[0] == 'termination':
                blocks[key][empty_row] = np.nan
    tokens = dict(ct=rng.normal(size=(2, 7)).astype(np.float32), token_rows=np.arange(7) % rows,
                  token_positions=np.arange(7), token_groups=np.zeros(7))
    return dict(identity={'arm': 'x', 'plan': 'p'}, example_ids=[f'e{i}' for i in range(rows)],
                counts=counts, blocks=blocks, tokens=tokens)


def test_roundtrip_blocks_tokens_and_index(tmp_path):
    store, u = BlockStore(tmp_path), unit()
    assert store.write_unit('arm-0', **u) == 'written'
    for key, array in u['blocks'].items():
        assert np.array_equal(store.read_block('arm-0', *key), array)
    tokens = store.read_tokens('arm-0')
    assert np.array_equal(tokens['ct'], u['tokens']['ct']) and tokens['token_rows'].dtype == np.int32
    index = store.index('arm-0')
    assert index['example_ids'] == ['e0', 'e1', 'e2'] and len(index['blocks']) == 8
    assert all(e['shape'] == [3, 5] for e in index['blocks'])


def test_zero_count_rows_must_be_nan_and_others_finite(tmp_path):
    store = BlockStore(tmp_path)
    assert store.write_unit('ok', **unit(empty_row=1)) == 'written'
    assert np.isnan(store.read_block('ok', 'termination', 0, 'mean_z')[1]).all()
    bad = unit(empty_row=1)
    bad['blocks'][('termination', 0, 'mean_z')][1] = 0.0  # undefined labelled zero
    with pytest.raises(ValueError, match='NaN'):
        store.write_unit('bad', **bad)
    bad = unit()
    bad['blocks'][('response_content', 0, 'mean_z')][0, 0] = np.nan
    with pytest.raises(ValueError):
        store.write_unit('bad2', **bad)
    bad = unit()
    bad['blocks'][('response_content', 0, 'mean_z')] = bad['blocks'][('response_content', 0, 'mean_z')].astype(np.float16)
    with pytest.raises(ValueError, match='FP32'):
        store.write_unit('bad3', **bad)


def test_sealed_unit_reused_never_rewritten(tmp_path):
    store, u = BlockStore(tmp_path), unit()
    store.write_unit('arm-0', **u)
    assert BlockStore(tmp_path).write_unit('arm-0', **u) == 'reused'
    other = unit()
    other['identity'] = {'arm': 'x', 'plan': 'changed'}
    with pytest.raises(ValueError, match='different identity'):
        store.write_unit('arm-0', **other)


def test_corrupt_unit_detected_distinctly(tmp_path):
    BlockStore(tmp_path).write_unit('arm-0', **unit())
    path = tmp_path / 'units/arm-0/blocks.f32'
    data = bytearray(path.read_bytes())
    data[10] ^= 0xFF
    path.write_bytes(bytes(data))
    fresh = BlockStore(tmp_path)  # new process: hashes once
    with pytest.raises(CorruptUnit):
        fresh.verify_unit('arm-0')
    with pytest.raises(CorruptUnit):
        fresh.write_unit('arm-0', **unit())


def test_interrupted_unit_preserved_as_evidence_then_rewritten(tmp_path):
    store = BlockStore(tmp_path)
    (tmp_path / 'units/arm-0').mkdir(parents=True)
    (tmp_path / 'units/arm-0/blocks.f32').write_bytes(b'partial')
    assert store.state('arm-0') == 'interrupted'
    assert store.write_unit('arm-0', **unit()) == 'written'
    evidence = list((tmp_path / 'interrupted').iterdir())
    assert len(evidence) == 1 and (evidence[0] / 'blocks.f32').read_bytes() == b'partial'
    assert store.status() == {'arm-0': 'sealed'}


def test_extra_file_in_sealed_unit_is_corruption(tmp_path):
    BlockStore(tmp_path).write_unit('arm-0', **unit())
    (tmp_path / 'units/arm-0/stray.bin').write_bytes(b'x')
    with pytest.raises(CorruptUnit, match='file set'):
        BlockStore(tmp_path).verify_unit('arm-0')
    seal = json.loads((tmp_path / 'units/arm-0/seal.json').read_text())
    assert 'stray.bin' not in seal['files']
