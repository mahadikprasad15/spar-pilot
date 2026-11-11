"""Frozen run plan: 03b/04 refuse unapproved or incomplete plans and unready inputs."""
from pathlib import Path

import pytest
import yaml

from pilot_eval import neuron_audit_plan as plan_mod

REAL_PLAN = Path(__file__).resolve().parents[1] / '.scratch/joint-neuron-adapter-audit/frozen_plan_v1.yaml'


def approved(tmp_path, **changes):
    value = yaml.safe_load(REAL_PLAN.read_text())
    value['status'] = 'approved'
    value['approval'] = dict(approved_by='reviewer', approved_on='2026-11-11', notes=None)
    value['runtime']['code_commit'] = 'a' * 40
    value['operations']['qualification_max_gpu_hours'] = 4
    value['preflight']['expected_sys_executable'] = '/workspace/venv/bin/python'
    value['preflight']['expected_sys_prefix'] = '/workspace/venv'
    for dotted, new in changes.items():
        node = value
        *parents, last = dotted.split('.')
        for key in parents:
            node = node[key]
        node[last] = new
    path = tmp_path / 'plan.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def test_checked_in_plan_is_unapproved_and_lists_every_gap():
    problems = plan_mod.problems(plan_mod.read(REAL_PLAN), '03b')
    assert 'status is unapproved, not approved' in problems
    for field in ['approval.approved_by', 'runtime.code_commit', 'operations.qualification_max_gpu_hours',
                  'preflight.expected_sys_executable']:
        assert f'{field} is null' in problems
    with pytest.raises(plan_mod.PlanError, match='unapproved'):
        plan_mod.load(REAL_PLAN, '03b')


def test_checked_in_plan_records_user_decisions():
    plan = plan_mod.read(REAL_PLAN)
    assert plan['fp64_floor']['zero_floor'] == 'unresolved'
    assert plan['hard_gates']['batched_vs_singleton_logits']['bound'] == 1e-3
    amendment = plan['spec_amendments'][0]
    assert amendment['id'] == 'reuse-adr-0014-logit-gate' and amendment['decided_by'] == 'user'
    assert 'Do not import' in amendment['spec_text']
    assert plan['review_flags']['early_layer']['layers'] == [0, 1, 2]
    assert plan['review_flags']['signoff_record_required_before'] == '04'
    assert plan['precision']['legacy_allow_tf32_setters'] == 'forbidden'


def test_approved_complete_plan_loads_with_hash(tmp_path):
    loaded = plan_mod.load(approved(tmp_path), '03b')
    assert len(loaded['_plan_sha256']) == 64
    with pytest.raises(plan_mod.PlanError, match='production_wall_time_cap_hours'):
        plan_mod.load(approved(tmp_path), '04')


def test_missing_section_or_null_required_field_refused(tmp_path):
    with pytest.raises(plan_mod.PlanError, match='code_commit'):
        plan_mod.load(approved(tmp_path, **{'runtime.code_commit': None}), '03b')
    with pytest.raises(plan_mod.PlanError, match='approved_on'):
        plan_mod.load(approved(tmp_path, **{'approval.approved_on': None}), '03b')
    with pytest.raises(plan_mod.PlanError, match='hard_gates'):
        plan_mod.load(approved(tmp_path, hard_gates=None), '03b')


def test_forbidden_changes_refused(tmp_path):
    with pytest.raises(plan_mod.PlanError, match='zero_floor'):
        plan_mod.load(approved(tmp_path, **{'fp64_floor.zero_floor': 'fp32_spacing'}), '03b')
    with pytest.raises(plan_mod.PlanError, match='legacy'):
        plan_mod.load(approved(tmp_path, **{'precision.legacy_allow_tf32_setters': 'allowed'}), '03b')
    with pytest.raises(plan_mod.PlanError, match='applies_to'):
        plan_mod.load(approved(tmp_path, **{'hard_gates.batched_vs_singleton_logits.applies_to': 'everything'}), '03b')


def test_preconditions_require_sealed_inputs_and_accepted_review(tmp_path):
    import json
    root = tmp_path / 'artifacts'
    plan = plan_mod.load(approved(tmp_path), '03b')
    problems = plan_mod.input_problems(plan, root)
    assert any('inventory' in p for p in problems) and any('sequences' in p for p in problems)
    inv = root / plan['inputs']['inventory_run']
    inv.mkdir(parents=True)
    (inv / 'inventory.json').write_text('{}')
    import hashlib
    (inv / 'complete.json').write_text(json.dumps(dict(
        inventory_sha256=hashlib.sha256(b'{}').hexdigest(), sources_sha256='x', key_lists={})))
    seq = root / plan['inputs']['sequences_run']
    seq.mkdir(parents=True)
    (seq / 'manifest.json').write_text('{}')
    (seq / 'complete.json').write_text(json.dumps(dict(files={
        'manifest.json': hashlib.sha256(b'{}').hexdigest()})))
    problems = plan_mod.input_problems(plan, root)
    assert problems == ['background review is pending, plan requires accept']
