"""Pod driver helpers that do not need a GPU: configs and the small rehearsal plan."""
import json
from pathlib import Path

import yaml

from pilot_eval import neuron_audit_plan as plan_mod
from pilot_eval import neuron_audit_pod as pod

INPUTS = dict(prep='/w/prep', model_dir='/w/hf/model', model_bytes=10, tokenizer_dir='/w/prep/tok',
              adapter_dirs={'gold-sft': '/w/prep/a'}, hf_home='/w/hf')


def test_rehearsal_and_production_configs(tmp_path):
    _, p_path, qualify, production = pod.write_configs(tmp_path, '/vol/audit', inputs=INPUTS, tag='t', rehearsal=True,
                                                       max_batches=2, interrupt_after=3)
    assert qualify['run_id'] == 'rehearsal-q-t' and production['run_id'] == 'rehearsal-p-t'
    assert production['max_batches_per_arm'] == 2 and production['rehearsal_interrupt_after_units'] == 3
    assert 'stop_command' not in production and production['volume_root'] == qualify['volume_dir'] == '/vol/audit'
    _, _, _, real = pod.write_configs(tmp_path, '/vol/audit', inputs=INPUTS, tag='v1', rehearsal=False,
                                      stop_command=['runpodctl', 'pod', 'stop', 'abc'])
    assert real['run_id'] == 'p-v1' and real['stop_command'][-1] == 'abc' and 'max_batches_per_arm' not in real
    assert json.loads(Path(p_path).read_text())['label'] == 'pod-t'


def test_small_rehearsal_plan_keeps_fixed_rules(tmp_path):
    out = plan_mod.make_rehearsal_plan(pod.PLAN, tmp_path / 'r.yaml', small=True)
    value = yaml.safe_load(Path(out).read_text())
    assert value['status'] == 'rehearsal' and value['operations']['qualification_max_gpu_hours'] == 1
    assert value['calibration_allocation']['fp64_reference'] == dict(math=2, general=2)
    assert value['benchmark']['token_budgets_padded'] == [4096, 8192]
    assert plan_mod.problems(value, '03b', rehearsal=True) == []
    real = yaml.safe_load(pod.PLAN.read_text())
    assert real['operations']['qualification_max_gpu_hours'] == 5
    assert real['operations']['stop_after_verified_partial_backup'] is True


def test_disk_problems_flags_a_default_50_gb_container_disk():
    assert pod.disk_problems(dict(total_bytes=50 * 10**9, free_bytes=43 * 10**9))
    assert pod.disk_problems(dict(total_bytes=120 * 10**9, free_bytes=113 * 10**9)) == []
