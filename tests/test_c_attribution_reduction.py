"""The production neuron path reduces by question inside each layer."""

import numpy as np
import pytest

from pilot_eval import attribution_instrument as ai
from test_attribution_instrument import instrument, rows, tiny


def test_streaming_question_and_draw_half_reductions_match_per_response_reference():
    model = tiny()
    batch = rows()[:3]
    for row in batch:
        row['masks'] = {'seq_targets': list(range(4, len(row['input_ids'])))}
    inst = instrument(model, metric_order=('seqlogp',))
    direct = inst.attribute(batch)
    request = {'question_ids': ['q0', 'q0', 'q1'],
               'coefficients': [0.25, -0.25, 0.0],
               'half_coefficients': [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.2]]}
    reduced = inst.attribute(batch, neuron_reduction=request)
    assert reduced['neuron_question_ids'] == ['q0', 'q1']
    for layer, index in inst.down_index.items():
        per_response = direct['neuron']['seqlogp'][layer]
        got = reduced['neuron']['seqlogp'][layer]
        expected_q = np.stack([0.25 * per_response[0] - 0.25 * per_response[1],
                               np.zeros(per_response.shape[1])])
        expected_halves = np.stack([0.1 * per_response[0] - 0.1 * per_response[1],
                                    0.2 * per_response[2]])
        np.testing.assert_allclose(got['by_question'], expected_q, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(got['half_sums'], expected_halves, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(reduced['term_size'][layer], direct['term_size'][layer].sum(0))
        a = model.get_base_model().model.layers[layer].mlp.down_proj.lora_A['default'].weight[0].detach().abs().numpy()
        np.testing.assert_allclose(reduced['sum_abs_z'][layer] * a, direct['term_size'][layer].sum(0), rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(got['identity']['net'], per_response.sum(1))
        np.testing.assert_allclose(got['identity']['gross'], np.abs(per_response).sum(1))
        assert np.max(np.abs(got['identity']['error'])) <= 1e-4 * max(got['identity']['gross'])
        assert got['by_question'].dtype == np.float64


def test_reduction_rejects_bad_or_nonfinite_coefficients_and_other_metric_modes():
    model = tiny()
    batch = rows()[:2]
    for row in batch:
        row['masks']['seq_targets'] = list(range(4, len(row['input_ids'])))
    inst = instrument(model, metric_order=('seqlogp',))
    valid = {'question_ids': ['q0', 'q0'], 'coefficients': [0.1, -0.1],
             'half_coefficients': [[0.1, 0.0], [-0.1, 0.0]]}
    for change in ({'coefficients': [0.1]}, {'coefficients': [0.1, float('nan')]},
                   {'half_coefficients': [[0.1], [-0.1]]}, {'question_ids': ['q0']}):
        with pytest.raises((ValueError, ai.InstrumentFailure)):
            inst.attribute(batch, neuron_reduction={**valid, **change})
    with pytest.raises(ai.InstrumentFailure):
        instrument(model).attribute(batch, neuron_reduction=valid)


def test_reduction_still_refuses_a_broken_neuron_identity():
    model = tiny()
    batch = rows()[:2]
    for row in batch:
        row['masks']['seq_targets'] = list(range(4, len(row['input_ids'])))
    request = {'question_ids': ['q0', 'q0'], 'coefficients': [0.1, -0.1],
               'half_coefficients': [[0.1, 0.0], [-0.1, 0.0]]}
    with pytest.raises(ai.InstrumentFailure, match='identity'):
        instrument(model, metric_order=('seqlogp',)).attribute(
            batch, neuron_reduction=request, _corrupt_pass2=True)
