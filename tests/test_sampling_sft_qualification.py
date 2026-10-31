"""Public qualification handoff, offline preparation never launches a GPU."""
import json
import pytest
from pilot_eval.cli import main
from test_sampling_sft_instrument import source_template, setup
from pilot_eval.workflow import _hash


def test_public_qualification_prepares_explicit_job_without_running(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    args=['sampling-sft-qualification-stage','--config',str(source),'--settings',str(settings),
          '--volume-id','fixture-volume','--rate','0.60','--name','q','--stage','fixtures','--reviewer','Prasad','--output-root',str(tmp_path)]
    assert main(args)==0
    path=tmp_path/'runs/sampling-sft/sample/jobs/q-fixtures/request.json'
    request=json.loads(path.read_text())
    assert request['command'][1:3]==['-m','pilot_eval']
    assert request['command'][3]=='sampling-sft-instrument-run'
    assert not (path.parent/'launch.json').exists()
    approval=json.loads((tmp_path/'runs/sampling-sft/sample/qualification/q/fixtures/approval.json').read_text())
    assert approval['request_sha256']==_hash(request)
    assert approval['waive_budget_limits'] is True
    plan=json.loads((tmp_path/'plans/q-fixtures/sampling-sft.instrument.json').read_text())
    assert plan['settings']['mode']=='fixtures'
    assert plan['settings']['problem_ids']==[14,41,63,75,95,97,101,103,142,143,150,209,223,256,265,266,288,310,317,323]
    assert main(args)==0


def test_score_stage_refuses_missing_fixture_evidence(tmp_path,source_template):
    source,settings=setup(tmp_path,source_template)
    assert main(['sampling-sft-qualification-stage','--config',str(source),'--settings',str(settings),
        '--volume-id','fixture-volume','--rate','0.60','--name','q','--stage','scores','--reviewer','Prasad','--output-root',str(tmp_path)])==1
    assert not (tmp_path/'runs/sampling-sft/sample/jobs/q-scores/launch.json').exists()
