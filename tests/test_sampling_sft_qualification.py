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


def test_new_runtime_reuses_sealed_old_fixtures_for_both_scorers(tmp_path,source_template):
    from test_sampling_sft_instrument import cli, Dependencies
    from test_sampling_sft_sampler import Engine, generation
    from pilot_eval.training import file_hash
    source,settings=setup(tmp_path,source_template)
    data=json.loads(settings.read_text());data.update(mode='fixtures',k=0,problem_ids=list(range(20)))
    settings.write_text(json.dumps(data))
    assert cli(tmp_path,'prepare','--config',source,'--settings',settings,'--name','old-fixtures')==0
    old=tmp_path/'plans/old-fixtures/sampling-sft.instrument.json'
    deps=Dependencies();deps.engine=Engine([generation([65,3],'stop',3)]*19+[generation([65]*1024)])
    assert cli(tmp_path,'run','--config',old,deps=deps)==0
    # Simulate a capture produced by an earlier implementation release while preserving its seals.
    oldplan=json.loads(old.read_text());oldplan['implementation']={'previous_release':'historical'}
    old.write_text(json.dumps(oldplan));folder=tmp_path/oldplan['run_path']
    (folder/'plan-seal.json').write_text(json.dumps({'config_sha256':_hash(oldplan)}))
    seal=json.loads((folder/'complete.json').read_text());seal['config_sha256']=_hash(oldplan)
    seal['files']['plan-seal.json']=file_hash(folder/'plan-seal.json')
    (folder/'complete.json').write_text(json.dumps(seal))
    data['runtime']['prefix_caching']=False;data['runtime']['v1_multiprocessing']=False
    settings.write_text(json.dumps(data))
    for stage in ('scores','oracle'):
        assert main(['sampling-sft-qualification-stage','--config',str(source),'--settings',str(settings),
            '--volume-id','fixture-volume','--rate','0.60','--name','new','--stage',stage,
            '--fixture-source',str(old),'--reviewer','Prasad','--output-root',str(tmp_path)])==0
        plan=json.loads((tmp_path/f'plans/new-{stage}/sampling-sft.instrument.json').read_text())
        assert plan['settings']['fixtures']==str((folder/'fixtures.json').relative_to(tmp_path))
        assert plan['settings']['runtime']['v1_multiprocessing'] is False
        assert plan['settings']['runtime']['prefix_caching'] is False
