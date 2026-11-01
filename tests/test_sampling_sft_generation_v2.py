"""Offline fixture integration; does not qualify a real GPU run."""
import json
import pytest

from pilot_eval import sampling_sft_generation_v2 as v2
from pilot_eval import sampling_sft_generation as v1
from pilot_eval.cli import main
from test_sampling_sft_instrument import source_template
from test_sampling_sft_pilot import prepared as pilot_plan
from test_sampling_sft_generation import Dependencies


def test_v2_is_distinct_plan_and_completes_cpu_generation(tmp_path, source_template):
    pilot = pilot_plan(tmp_path, source_template)
    config = v2.prepare(pilot, tmp_path, 'buffered-generation')
    plan = json.loads(config.read_text())
    assert plan['protocol'] == 'sampling-sft-generation-buffered-v2'
    assert plan['writer']['mode'] == 'buffered-before-seal-v1'
    result = v2.run(config, tmp_path, 0, dependencies=Dependencies())
    assert result['state'] == 'completed'
    assert v2.verify(config, tmp_path, k=0)['execution_kind'] == 'cpu-fixture'


def test_v2_cli_prepares_versioned_plan(tmp_path, source_template):
    pilot = pilot_plan(tmp_path, source_template)
    assert main(['sampling-sft-buffered-generation-prepare', '--pilot', str(pilot),
                 '--name', 'cli-v2', '--output-root', str(tmp_path)]) == 0
    plan = json.loads((tmp_path/'plans/cli-v2/sampling-sft.generation.json').read_text())
    assert plan['protocol'] == 'sampling-sft-generation-buffered-v2'


def test_real_execution_refuses_missing_v2_replay_qualification(tmp_path):
    with pytest.raises(ValueError, match='v2 real-CUDA replay qualification'):
        v2.authorize(tmp_path, {}, tmp_path/'plan', tmp_path, 5, None)


def test_v2_block_matches_v1_and_resume_preserves_seal(tmp_path, source_template):
    pilot = pilot_plan(tmp_path, source_template)
    old = v1.prepare(pilot, tmp_path, 'old-fixture')
    new = v2.prepare(pilot, tmp_path, 'new-fixture')
    v1.run(old, tmp_path, 0, dependencies=Dependencies(), stop_after_blocks=1)
    v2.run(new, tmp_path, 0, dependencies=Dependencies(), stop_after_blocks=1)
    base = tmp_path/'runs/sampling-sft/sample/generation'
    old_block = base/'old-fixture/k0/rounds/0/blocks/000/scientific.jsonl'
    new_block = base/'new-fixture/k0/rounds/0/blocks/000/scientific.jsonl'
    original = new_block.read_bytes()
    assert original == old_block.read_bytes()
    v2.run(new, tmp_path, 0, dependencies=Dependencies(), stop_after_blocks=1)
    assert new_block.read_bytes() == original
    assert (new_block.parent.parent/'001/complete.json').exists()
