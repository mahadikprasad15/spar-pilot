"""CPU wiring for 06b-09: compute monitoring per stage, concentration + resolution floor, a vs a0 quote,
early-stop rate from the study's untuned run, and the item-level comparison with pilot results."""
import json

import numpy as np
import pytest

from pilot_eval import attribution_gpu as gpu
from pilot_eval import attribution_report as ar
from pilot_eval import attribution_units as au


# --- compute monitoring -------------------------------------------------------------------------

def test_monitor_records_time_tokens_padding_and_marks_unavailable_parts(tmp_path):
    clock = iter([0.0, 10.0])
    monitor = gpu.StageMonitor('attribution_and_freeze', nvml=None, sync=lambda: None, now=lambda: next(clock),
                               parameters=1000, peak_flops=None, backward_passes=2)
    monitor.start()
    monitor.add_batch(lengths=[3, 5], padded_width=5)
    monitor.add_batch(lengths=[4], padded_width=4)
    record = monitor.finish()
    assert record['seconds'] == pytest.approx(10.0) and record['tokens'] == 12
    assert record['padding_fraction'] == pytest.approx(1 - 12 / 14)
    assert record['tokens_per_second'] == pytest.approx(1.2)
    assert record['nvml']['available'] is False
    assert record['mfu'] == dict(available=False, reason='no FP32 peak FLOP/s in the launch record')
    with_peak = gpu.StageMonitor('x', nvml=None, sync=lambda: None, now=iter([0.0, 2.0]).__next__, parameters=1000,
                                 peak_flops=1e6, backward_passes=2)
    with_peak.start()
    with_peak.add_batch(lengths=[10], padded_width=10)
    assert with_peak.finish()['mfu']['mfu'] == pytest.approx(2 * 1000 * 10 * 3 / 2 / 1e6)


def test_monitor_profiles_only_the_first_unit(tmp_path):
    pytest.importorskip('torch', exc_type=ImportError)
    import torch
    monitor = gpu.StageMonitor('stage_c', nvml=None, sync=lambda: None, parameters=1, peak_flops=None,
                               profile_path=tmp_path / 'trace.json')
    monitor.start()
    for _ in range(3):
        with monitor.unit():
            torch.ones(3) @ torch.ones(3)
    record = monitor.finish()
    assert (tmp_path / 'trace.json').is_file() and record['profiled_units'] == 1


# --- resolution floor and concentration ---------------------------------------------------------

def test_resolution_floor_rule_is_evaluated_from_the_qualification_measurement():
    assert ar.resolution_floor('10x measured batch-invariance discrepancy', 3e-7) == pytest.approx(3e-6)
    with pytest.raises(ValueError):
        ar.resolution_floor('ten times', 3e-7)
    assert ar.resolution_floor('10x batch invariance', 0.0) == 0.0


def test_concentration_table_per_arm_and_metric():
    freeze = {'gold-sft': {'trace': dict(down_proj=dict(layer=1)), 'stop': dict(down_proj=dict(layer=0))}}
    neuron = {'gold-sft': {'trace': {1: np.array([5.0, 1.0, -1.0, 0.5])}, 'stop': {0: np.full(4, 1e-12)}}}
    table = ar.concentration_table(freeze, neuron, null_orders=50, null_seed=3, resolution_floor=1e-9)
    assert table['gold-sft']['trace']['neurons_to_50'] == 1 and table['gold-sft']['trace']['layer'] == 1
    assert table['gold-sft']['stop']['available'] is False


# --- a vs a0 ---------------------------------------------------------------------------------------

def test_a_vs_a0_quoted_raw_for_the_frozen_top_down_proj(tmp_path):
    rows = [dict(arm='gold-sft', layer=1, module='down_proj', cos_a=0.97, a_drift=0.21, extra=1),
            dict(arm='gold-sft', layer=0, module='down_proj', cos_a=0.5, a_drift=0.9)]
    path = tmp_path / 'rows.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    freeze = {'gold-sft': {'trace': dict(down_proj=dict(layer=1)), 'stop': dict(down_proj=dict(layer=1))}}
    quote = ar.a_vs_a0(path, freeze)
    assert quote['gold-sft']['trace'] == dict(layer=1, cos_a=0.97, a_drift=0.21)
    assert 'gauge-dependent' in quote['label']
    assert ar.a_vs_a0(None, freeze) is None


# --- early stop ---------------------------------------------------------------------------------

def items(tokens, stop='eos'):
    return [dict(id=f'i{k}', tokens=t, stop_reason=stop, correct=True) for k, t in enumerate(tokens)]


def test_early_stop_threshold_comes_from_the_study_untuned_run_and_the_record_rule():
    untuned = items([100, 200, 300, 400, 500])
    threshold = ar.early_stop_threshold(untuned, rule=dict(kind='quantile-of-untuned-eos-tokens', quantile=0.2))
    assert threshold == pytest.approx(np.quantile([100, 200, 300, 400, 500], 0.2))
    assert ar.early_stop_rate(items([50, 400, 150]), threshold) == pytest.approx(2 / 3)
    assert ar.early_stop_rate(items([50, 60], stop='cap'), threshold) == 0.0  # capped is its own outcome
    for bad in (None, dict(kind='quantile-of-untuned-eos-tokens', quantile=None), dict(kind='other', quantile=0.1)):
        with pytest.raises(ValueError):
            ar.early_stop_threshold(untuned, rule=bad)


def test_stage_d_outcomes_count_capped_and_invalid_separately():
    rows = [dict(id='a', correct=True, status='valid', tokens=10, stop_reason='eos'),
            dict(id='b', correct=False, status='invalid', tokens=12, stop_reason='eos'),
            dict(id='c', correct=False, status='invalid', tokens=1024, stop_reason='cap')]
    out = ar.stage_d_outcomes(rows, early_threshold=11)
    assert out == dict(n=3, accuracy=pytest.approx(1 / 3), mean_tokens=pytest.approx(1046 / 3), capped=1, invalid=2,
                       capped_rate=pytest.approx(1 / 3), early_stop_rate=pytest.approx(1 / 3))


# --- pilot comparison ------------------------------------------------------------------------------

def test_item_level_comparison_with_pilot_results():
    study = [dict(id='gsm8k:test:1', correct=True), dict(id='gsm8k:test:2', correct=False),
             dict(id='gsm8k:test:3', correct=True)]
    pilot = [dict(id='gsm8k:test:1', correct=True), dict(id='gsm8k:test:2', score=dict(correct=True)),
             dict(id='gsm8k:test:3', correct=True)]
    out = ar.pilot_comparison(study, pilot)
    assert out['n'] == 3 and out['agree'] == 2 and out['flips'] == [dict(id='gsm8k:test:2', pilot=True, study=False)]
    with pytest.raises(ValueError, match='items'):
        ar.pilot_comparison(study[:2], pilot)
    with pytest.raises(ValueError, match='correct'):
        ar.pilot_comparison(study, [dict(id=r['id']) for r in pilot])
