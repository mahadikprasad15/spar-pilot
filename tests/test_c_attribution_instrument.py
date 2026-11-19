"""CPU checks for the sequence-log-probability path in the shared instrument."""

import numpy as np
import pytest
import torch

from pilot_eval import attribution_instrument as ai
from pilot_eval import attribution_metrics as am
from test_attribution_instrument import STOP, instrument, rows, tiny


def seq_rows():
    out = rows()[:2]
    for row in out:
        row['masks'] = {'seq_targets': list(range(4, len(row['input_ids']))),
                        'length': len(row['input_ids'])}
    return out


def test_seq_only_values_and_removal_attribution_match_independent_gate_oracle():
    model = tiny()
    batch = seq_rows()
    inst = instrument(model, metric_order=('seqlogp',))
    out = inst.attribute(batch, raw=True)
    ids, attention, positions = inst.batch(batch)
    gate = torch.ones((len(batch), 1, 1), requires_grad=True)
    with ai.branch_gates(model, {0: gate}):
        logits = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                       use_cache=False).logits
        expected = am.seqlogp(logits, ids, [r['masks'] for r in batch])
        oracle = -torch.autograd.grad(expected.sum(), gate)[0].reshape(-1).detach().numpy()
    assert set(out['metrics']) == {'seqlogp'}
    assert np.allclose(out['metrics']['seqlogp'], expected.detach().numpy(), rtol=1e-6)
    assert np.allclose(out['attr']['seqlogp'][:, 0], oracle, rtol=1e-5, atol=1e-7)
    assert out['raw']['g']['seqlogp'][0][:, :4].abs().sum() > 0
    for layer, index in inst.down_index.items():
        assert np.allclose(out['neuron']['seqlogp'][layer].sum(1),
                           out['attr']['seqlogp'][:, index], rtol=1e-4, atol=1e-10)


def test_adding_seqlogp_preserves_legacy_metric_outputs():
    model = tiny()
    batch = rows()[:2]
    for row in batch:
        row['masks']['seq_targets'] = list(range(4, len(row['input_ids'])))
    old = instrument(model).attribute(batch)
    combined = instrument(model, metric_order=('trace', 'stop', 'seqlogp')).attribute(batch)
    for name in ('trace', 'stop'):
        assert np.array_equal(old['metrics'][name], combined['metrics'][name], equal_nan=True)
        assert np.array_equal(old['attr'][name], combined['attr'][name], equal_nan=True)
        for layer in old['neuron'][name]:
            assert np.array_equal(old['neuron'][name][layer], combined['neuron'][name][layer], equal_nan=True)


def test_metric_selection_rejects_empty_duplicate_and_unknown_names():
    model = tiny()
    for names in ((), ('seqlogp', 'seqlogp'), ('unknown',)):
        with pytest.raises(ai.InstrumentFailure):
            instrument(model, metric_order=names)


def test_seqlogp_fp64_finite_differences_for_adapter_and_neuron_reads():
    model = tiny().double()
    batch = seq_rows()[:1]
    inst = instrument(model, metric_order=('seqlogp',), precision='fp64')
    out = inst.attribute(batch)
    ids, attention, positions = inst.batch(batch)
    eps = 1e-5

    def measured(gate, *, neuron=False):
        value = torch.full((1, 1, 1), gate, dtype=torch.float64)
        adapter_gates = {} if neuron else {6: value}
        neuron_gate = (0, 1, value) if neuron else None
        with ai.full_fp64(model), ai.branch_gates(model, adapter_gates), ai.neuron_gate(model, neuron_gate):
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                               use_cache=False).logits
                return float(am.seqlogp(logits, ids, [r['masks'] for r in batch])[0])

    adapter_fd = -(measured(1 + eps) - measured(1 - eps)) / (2 * eps)
    neuron_fd = -(measured(1 + eps, neuron=True) - measured(1 - eps, neuron=True)) / (2 * eps)
    assert out['attr']['seqlogp'][0, 6] == pytest.approx(adapter_fd, rel=1e-5, abs=1e-8)
    assert out['neuron']['seqlogp'][0][0, 1] == pytest.approx(neuron_fd, rel=1e-5, abs=1e-8)


def test_stop_only_with_no_stop_positions_remains_undefined():
    model = tiny()
    one = rows()[-1:]
    one[0]['masks'] = {'stop_targets': [], 'length': len(one[0]['input_ids'])}
    out = instrument(model, metric_order=('stop',)).attribute(one)
    assert np.isnan(out['metrics']['stop'][0])
    assert np.isnan(out['attr']['stop'][0]).all()
    assert out['stop_reasons'] == ['empty-stop-pull-mask']
