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
