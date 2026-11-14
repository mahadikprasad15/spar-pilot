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


# --- early stop (rule B: per-question fraction of the untuned response; user 2026-11-14) ---------------

def items(tokens, stop='eos'):
    return [dict(id=f'i{k}', tokens=t, stop_reason=stop, correct=True) for k, t in enumerate(tokens)]


RULE = dict(kind='fraction-of-untuned-same-item', ratio=0.5)


def test_early_stop_reference_is_the_untuned_response_to_the_same_question():
    untuned = items([100, 400, 200]) + [dict(id='i3', tokens=1024, stop_reason='cap', correct=False)]
    reference = ar.early_stop_reference(untuned, rule=RULE)
    assert reference == dict(rule=RULE, tokens={'i0': 100, 'i1': 400, 'i2': 200}, excluded_capped=['i3'],
                             source="this study's untuned run")
    run = items([40, 300, 99]) + [dict(id='i3', tokens=10, stop_reason='eos', correct=True)]
    rate = ar.early_stop_rate(run, reference)
    # i0: 40 < 50 early; i1: 300 >= 200 not; i2: 99 < 100 early; i3 excluded (untuned capped)
    assert rate == dict(rate=pytest.approx(2 / 3), early=['i0', 'i2'], counted=3, excluded=['i3'])
    capped = ar.early_stop_rate(items([10, 10, 10], stop='cap'), reference)
    assert capped['rate'] == 0.0  # a capped response is its own outcome, never an early stop
    for bad in (None, dict(kind='fraction-of-untuned-same-item', ratio=None), dict(kind='quantile', ratio=0.5),
                dict(kind='fraction-of-untuned-same-item', ratio=1.5)):
        with pytest.raises(ValueError):
            ar.early_stop_reference(untuned, rule=bad)


def test_stage_d_outcomes_count_capped_and_invalid_separately():
    untuned = [dict(id=i, tokens=20, stop_reason='eos', correct=True) for i in 'abc']
    reference = ar.early_stop_reference(untuned, rule=RULE)
    rows = [dict(id='a', correct=True, status='valid', tokens=5, stop_reason='eos'),
            dict(id='b', correct=False, status='invalid', tokens=12, stop_reason='eos'),
            dict(id='c', correct=False, status='invalid', tokens=1024, stop_reason='cap')]
    out = ar.stage_d_outcomes(rows, early_reference=reference)
    assert out['n'] == 3 and out['accuracy'] == pytest.approx(1 / 3) and out['capped'] == 1 and out['invalid'] == 2
    assert out['capped_rate'] == pytest.approx(1 / 3) and out['early_stop']['rate'] == pytest.approx(1 / 3)


# --- pilot comparison (pilot responses re-scored with flexible v3; stored scores audited) -------------

def test_item_level_comparison_rescores_pilot_responses_with_v3_and_records_disagreements():
    gold = {f'gsm8k:test:{i}': 'w\n#### 7' for i in (1, 2, 3)}
    study = [dict(id='gsm8k:test:1', correct=True), dict(id='gsm8k:test:2', correct=False),
             dict(id='gsm8k:test:3', correct=True)]
    pilot = [dict(id='gsm8k:test:1', text='so\n#### 7', correct=True, stop_reason='eos'),
             dict(id='gsm8k:test:2', text='so\n#### 7', score=dict(correct=False), stop_reason='eos'),  # stale
             dict(id='gsm8k:test:3', text='so\n#### 8', correct=False, stop_reason='eos')]
    out = ar.pilot_comparison(study, pilot, gold)
    assert out['scorer'] == 'gsm8k_flexible_v3 (re-scored from pilot text)'
    assert out['stored_score_disagreements'] == [dict(id='gsm8k:test:2', stored=False, v3=True)]
    assert out['n'] == 3 and out['agree'] == 1
    assert out['flips'] == [dict(id='gsm8k:test:2', pilot=True, study=False), dict(id='gsm8k:test:3', pilot=False,
                                                                                    study=True)]
    with pytest.raises(ValueError, match='items'):
        ar.pilot_comparison(study[:2], pilot, gold)
    with pytest.raises(ValueError, match='text'):
        ar.pilot_comparison(study, [dict(id=r['id'], correct=True) for r in pilot], gold)


def test_pilot_rows_filtered_by_the_record_and_capped_rows_scored_as_capped(tmp_path):
    path = tmp_path / 'p.jsonl'
    rows = [dict(id='gsm8k:test:1', step=0, text='#### 1'), dict(id='gsm8k:test:1', step=64, text='#### 7'),
            dict(id='gsm8k:test:2', step=64, text='#### 7', stop_reason='cap')]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    selected = ar.load_pilot_rows(path, filters=dict(step=64))
    assert [r['text'] for r in selected] == ['#### 7', '#### 7']
    out = ar.pilot_comparison([dict(id='gsm8k:test:1', correct=True), dict(id='gsm8k:test:2', correct=False)],
                              selected, {'gsm8k:test:1': '#### 7', 'gsm8k:test:2': '#### 7'})
    assert out['agree'] == 2  # the capped pilot row is scored capped (v3 refuses it), so it agrees with False
    with pytest.raises(ValueError, match='duplicate'):
        ar.load_pilot_rows(path, filters={})
