import json
import pytest

from pilot_eval import sampling_sft_pilot_v2 as pilot
from pilot_eval import sampling_sft_generation_v2 as generation
from pilot_eval.cli import main
from test_sampling_sft_pilot import Dependencies, prepared
from test_sampling_sft_instrument import source_template


def test_buffered_pilot_exact_replay_cpu_only(tmp_path, source_template):
    old = prepared(tmp_path, source_template)
    instrument = tmp_path/json.loads(old.read_text())['instrument_plan']
    plan = pilot.prepare(instrument, tmp_path, 'buffered-pilot')
    pilot.run(plan, tmp_path, 'uninterrupted', dependencies=Dependencies())
    pilot.run(plan, tmp_path, 'resumed', dependencies=Dependencies(), stop_after_blocks=2)
    pilot.run(plan, tmp_path, 'resumed', dependencies=Dependencies())
    result = pilot.compare(plan, tmp_path)
    assert result['scientific_equal'] is True
    assert result['gpu_qualified'] is False
    with pytest.raises(ValueError, match='real-CUDA'):
        generation.check_writer_replay(tmp_path, {'writer_pilot': str(plan.relative_to(tmp_path)),
            'runtime_hash': json.loads(plan.read_text())['runtime_hash'],
            'pilot_plan': str(old.relative_to(tmp_path))})


def test_buffered_pilot_cli_prepares(tmp_path, source_template):
    old = prepared(tmp_path, source_template)
    instrument = tmp_path/json.loads(old.read_text())['instrument_plan']
    assert main(['sampling-sft-buffered-pilot-prepare', '--instrument', str(instrument),
                 '--output-root', str(tmp_path), '--name', 'cli-buffered-pilot']) == 0


def test_cross_writer_comparison_detects_different_scientific_records(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pilot as old
    plan={'runtime_hash':'same','instrument_plan':'inst','source_plan':'src',
          'settings':{'k':5},'problems':[{'problem_id':1}]}
    monkeypatch.setattr(old,'load',lambda *a:(tmp_path,plan,tmp_path/'old'))
    monkeypatch.setattr(pilot,'load',lambda *a:(tmp_path,plan,tmp_path/'new'))
    monkeypatch.setattr(old,'verify_lane',lambda *a:{'traces':[{'id':1}],'execution_kind':'real-cuda'})
    monkeypatch.setattr(pilot,'verify_lane',lambda *a:{'traces':[{'id':1}],'execution_kind':'real-cuda'})
    monkeypatch.setattr(old,'scientific',lambda *a:[{'score':-1.0}])
    monkeypatch.setattr(pilot,'scientific',lambda *a:[{'score':-1.0}])
    assert pilot.compare_legacy(tmp_path/'old.json',tmp_path/'new.json',tmp_path)['scientific_equal']
    monkeypatch.setattr(pilot,'scientific',lambda *a:[{'score':-1.1}])
    assert not pilot.compare_legacy(tmp_path/'old.json',tmp_path/'new.json',tmp_path)['scientific_equal']
