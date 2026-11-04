"""CPU-only DAG and artifact verification tests."""
from pathlib import Path

import pytest


def test_dag_rejects_unknown_or_forward_dependencies():
    from pilot_eval.offline_grpo_overnight import validate_graph
    with pytest.raises(ValueError,match='dependency'):
        validate_graph([dict(id='a',depends_on=['b']),dict(id='b',depends_on=[])])


def make_spec(root):
    from pilot_eval import offline_grpo_overnight as overnight
    source=root/'source.json';source.write_text('{}')
    from pilot_eval.training import file_hash
    stages=[]
    for name,operations in overnight.STAGES.items():
        plan=str(root/f'{name}.json')
        commands=[]
        for operation in operations:
            command=[operation,'--output-root',str(root)]
            if not operation.endswith('prepare'):command+=['--config',plan]
            if operation=='offline-grpo-internals-collect':command+=['--kind',name]
            if operation=='offline-grpo-train-prepare':command+=['--checkpoint-zero',str(source)]
            if operation=='offline-grpo-train-check-diagnostics':command+=['--fidelity-tolerance','0.5']
            commands.append(command)
        stages.append(dict(id=name,depends_on=overnight.DEPENDENCIES[name],plan=plan,commands=commands))
    for group in [('diagnostics','train'),('geometry','push','drift','internals_report'),('cut_build','cut_agreement','cut_evaluation','cut_report')]:
        shared=next(stage['plan'] for stage in stages if stage['id']==group[0])
        for stage in stages:
            if stage['id'] in group:
                stage['plan']=shared
                for command in stage['commands']:
                    if '--config' in command:command[command.index('--config')+1]=shared
    return dict(protocol='offline-grpo-overnight-v1',name='fixture',stages=stages,
        export=dict(repo='fixture/private',paths=['source.json']),stage0_runtime=dict(session_id='fixture',execution_kind='synthetic-cpu-fixture'),training_runtime=dict(dtype='float32'),
        model_identity=dict(model='fixture',model_revision='rev',tokenizer_revision='rev',base_sha256='base'),eos_token_id=99,snapshot_path='source.json',snapshot_hashes={'source.json':'fixture'},hf_cache_root=str(root),minimum_cache_free_bytes=0,software_versions={'trl':'0.26.2'},predictions_path='source.json',reporting_protocol_path='source.json',source_hashes={'source.json':file_hash(source)},minimum_free_bytes=0)


def test_public_run_continues_independent_stage_and_resumes(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from pilot_eval import offline_grpo_overnight as workflow,stage0_budget
    from pilot_eval.workflow import _hash
    from pilot_eval.cli import main
    spec=make_spec(tmp_path);spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    assert main(['offline-grpo-overnight-prepare','--output-root',str(tmp_path),'--spec',str(spec_path)])==0
    path=tmp_path/'plans/fixture/offline.overnight.json';plan=json.loads(path.read_text())
    directory=tmp_path/'runs/offline-grpo/fixture/overnight';preflight=dict(ok=True,evidence={},issues=[])
    (directory/'meta/preflight.json').write_text(json.dumps(preflight))
    allocation=dict(reviewer='test',notes='fixture',session_id='fixture',source_reviewed=True,max_units_per_hour=1.,max_seconds=60.)
    auth=dict(protocol='offline-grpo-prelaunch-v1',approved=True,plan_sha256=_hash(plan),stage0_runtime=spec['stage0_runtime'],training_runtime=spec['training_runtime'],repo=spec['export']['repo'],reviewer='test',preflight_sha256=_hash(preflight),max_total_units=1.,allocation=allocation)
    auth_path=tmp_path/'auth.json';auth_path.write_text(json.dumps(auth))
    row=dict(reservation_id='fixture',state='active',job_sha256=_hash(plan),approval=allocation)
    monkeypatch.setattr(stage0_budget,'reserve',lambda *args:row)
    monkeypatch.setattr(stage0_budget,'reservations',lambda *args:{'fixture':row})
    done=set();attempts=[]
    def native(stage,root):
        if stage['id'] not in done:raise FileNotFoundError(stage['id'])
        return True
    monkeypatch.setattr(workflow,'_native_verify',native)
    monkeypatch.setattr(workflow,'_snapshot_report',lambda *args:tmp_path)
    monkeypatch.setattr(workflow,'_export',lambda *args,**kwargs:dict(record=dict(revision='fixture',manifest_sha256='manifest'),manifest_path='fixture',partial=kwargs['partial']))
    class Deps:
        fail=True
        def runtime(self):return spec['stage0_runtime']
        def hub(self):return SimpleNamespace(info=lambda repo:SimpleNamespace(private=True))
        def invoke(self,command,*,reservation_provider):
            name=command[command.index('--kind')+1] if command[0]=='offline-grpo-internals-collect' else next(stage['id'] for stage in spec['stages'] if command[0] in workflow.STAGES[stage['id']])
            attempts.append((name,command[0]))
            if name=='train' and self.fail:return 1
            if command[0]==workflow.STAGES[name][-1]:done.add(name)
            if name in ['pool','dataset','push','drift','cut_agreement','cut_evaluation']:
                got,left=reservation_provider({},None,None,tmp_path,spec['stage0_runtime'],None)
                assert got['reservation_id']=='fixture' and left>0
            return 0
    deps=Deps()
    state=workflow.run(path,tmp_path,auth_path,dependencies=deps)
    assert state['stages']['train']['state']=='failed'
    assert state['stages']['dataset']['state']=='verified'
    assert state['stages']['behaviour']['state']=='blocked_dependency'
    assert state['state']=='incomplete' and state['export']['state']=='partial_verified'
    deps.fail=False
    state=workflow.run(path,tmp_path,auth_path,dependencies=deps)
    assert state['state']=='completed' and state['export']['state']=='verified'
    assert sum(name=='pool' and command==workflow.STAGES['pool'][-1] for name,command in attempts)==1


def test_fresh_readback_detects_member_and_manifest_mismatch(tmp_path):
    import hashlib
    import io
    import json
    import shutil
    import tarfile
    from pilot_eval.offline_grpo_overnight import _fresh_readback
    from pilot_eval.training import file_hash
    payload=b'hello'
    manifest=dict(files={'runs/result.json':dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))})
    archive=tmp_path/'archive.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for name,data in [('runs/result.json',payload),('manifest.json',json.dumps(manifest).encode())]:
            info=tarfile.TarInfo(name);info.size=len(data);stream.addfile(info,io.BytesIO(data))
    record=dict(repo='private',archive='archive',revision='rev',archive_sha256=file_hash(archive),manifest_path='manifest.json')
    class Deps:
        def fresh_fetch(self,repo,name,revision,cache):
            target=cache/'fresh.tar.gz';shutil.copyfile(archive,target);return target
    _fresh_readback(record,manifest,Deps())
    manifest['files']['runs/result.json']['sha256']='0'*64
    with pytest.raises(ValueError,match='member differs'):
        _fresh_readback(record,manifest,Deps())


def test_preflight_aggregates_missing_credentials_runtime_and_source(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_overnight as workflow
    from pilot_eval.cli import main
    spec=make_spec(tmp_path);(tmp_path/'source.json').write_text('changed')
    spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    assert main(['offline-grpo-overnight-prepare','--output-root',str(tmp_path),'--spec',str(spec_path)])==0
    monkeypatch.setenv('HF_HUB_ENABLE_HF_TRANSFER','0')
    monkeypatch.setenv('HF_TOKEN','fixture-secret')
    class Deps:
        def runtime(self):raise ValueError('runtime unavailable')
        def parity(self):raise ValueError('skipped')
        def hub(self):raise ValueError('missing fixture-secret')
    path=tmp_path/'plans/fixture/offline.overnight.json'
    result=workflow.preflight(path,tmp_path,dependencies=Deps())
    checks={r['check'] for r in result['issues']}
    assert {'sources','runtime','parity','hf'}<=checks and not result['ok']
    assert 'fixture-secret' not in json.dumps(result)


def test_partial_manifest_lists_unavailable_stage_plans(tmp_path):
    import json
    from pilot_eval import offline_grpo_overnight as workflow
    spec=make_spec(tmp_path)
    plan=dict(spec=spec)
    directory=tmp_path/'runs/offline-grpo/fixture/overnight'
    snapshot=directory/'results/snapshots/test';snapshot.mkdir(parents=True)
    (snapshot/'final-report.json').write_text(json.dumps(dict(stages={'train':dict(state='failed')})))
    (tmp_path/'plans/fixture').mkdir(parents=True)
    (tmp_path/'plans/fixture/offline.overnight.json').write_text('{}')
    path=workflow._artifact_manifest(plan,tmp_path,directory,snapshot,partial=True)
    manifest=json.loads(path.read_text())
    assert manifest['partial'] is True and 'stage:train:failed' in manifest['missing']
    assert any(value.endswith(':plan') for value in manifest['missing'])


def test_model_dispatch_scopes_offline_environment(monkeypatch):
    import os
    from pilot_eval.offline_grpo_overnight import Dependencies
    from pilot_eval import cli
    seen={}
    def fake_main(arguments,**kwargs):
        seen.update(hf=os.environ.get('HF_HUB_OFFLINE'),transformers=os.environ.get('TRANSFORMERS_OFFLINE'))
        return 0
    monkeypatch.setattr(cli,'main',fake_main)
    monkeypatch.delenv('HF_HUB_OFFLINE',raising=False)
    monkeypatch.delenv('TRANSFORMERS_OFFLINE',raising=False)
    assert Dependencies().invoke(['offline-grpo-train-run'])==0
    assert seen==dict(hf='1',transformers='1')
    assert 'HF_HUB_OFFLINE' not in os.environ and 'TRANSFORMERS_OFFLINE' not in os.environ


def test_model_cache_uses_pinned_revisions_and_rejects_changed_bytes(tmp_path):
    from pilot_eval.offline_grpo_overnight import _verify_model_cache
    from pilot_eval.training import file_hash
    model=tmp_path/'model.safetensors';model.write_bytes(b'weights')
    tokenizer=tmp_path/'tokenizer.json';tokenizer.write_bytes(b'tokens')
    spec=dict(hf_cache_root=str(tmp_path),minimum_cache_free_bytes=0,
              model_identity=dict(model='m',model_revision='model-rev',tokenizer_revision='tokenizer-rev'),
              snapshot_hashes={model.name:file_hash(model),tokenizer.name:file_hash(tokenizer)})
    calls=[]
    class Deps:
        def cache_root(self):return tmp_path
        def cache_lookup(self,model,name,revision):calls.append((name,revision));return tmp_path/name
    _verify_model_cache(spec,Deps())
    assert calls==[('model.safetensors','model-rev'),('tokenizer.json','tokenizer-rev')]
    tokenizer.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache file missing or differs'):
        _verify_model_cache(spec,Deps())


def test_budget_expiry_stops_later_stages_and_export(tmp_path,monkeypatch):
    import json,time
    from types import SimpleNamespace
    from pilot_eval import offline_grpo_overnight as workflow,stage0_budget
    from pilot_eval.workflow import _hash
    spec=make_spec(tmp_path);spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    path=workflow.prepare(spec_path,tmp_path);plan=json.loads(path.read_text())
    directory=tmp_path/'runs/offline-grpo/fixture/overnight';preflight=dict(ok=True,evidence={},issues=[])
    (directory/'meta/preflight.json').write_text(json.dumps(preflight))
    allocation=dict(reviewer='test',notes='fixture',session_id='fixture',source_reviewed=True,max_units_per_hour=1.,max_seconds=1.)
    auth=dict(protocol='offline-grpo-prelaunch-v1',approved=True,plan_sha256=_hash(plan),stage0_runtime=spec['stage0_runtime'],training_runtime=spec['training_runtime'],repo=spec['export']['repo'],reviewer='test',preflight_sha256=_hash(preflight),max_total_units=1.,allocation=allocation)
    auth_path=tmp_path/'auth.json';auth_path.write_text(json.dumps(auth))
    row=dict(reservation_id='fixture',state='active',job_sha256=_hash(plan),approval=allocation)
    monkeypatch.setattr(stage0_budget,'reserve',lambda *args:row)
    monkeypatch.setattr(workflow,'_native_verify',lambda *args:(_ for _ in ()).throw(FileNotFoundError()))
    clock={'now':0.}
    monkeypatch.setattr(time,'monotonic',lambda:clock['now'])
    class Deps:
        calls=0
        def runtime(self):return spec['stage0_runtime']
        def hub(self):return SimpleNamespace(info=lambda repo:SimpleNamespace(private=True))
        def invoke(self,command,*,reservation_provider):
            self.calls+=1;clock['now']=2.;return 0
    deps=Deps()
    with pytest.raises(TimeoutError,match='allocation expired'):
        workflow.run(path,tmp_path,auth_path,dependencies=deps)
    assert deps.calls==1
    status=json.loads((directory/'meta/status.json').read_text())
    assert status['state']=='interrupted' and status['export']['state']=='not_started'


def test_generated_recipe_binds_complete_stage_matrix(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_overnight as workflow
    from pilot_eval.training import file_hash
    (tmp_path/'stage0.json').write_text(json.dumps(dict(session_id='fixture',execution_kind='synthetic-cpu-fixture')))
    (tmp_path/'training.json').write_text(json.dumps(dict(dtype='float32')))
    (tmp_path/'source.json').write_text('{}')
    inputs=dict(audit='source.json',historical_baseline='source.json',frozen='source.json',weights_dir='source.json',
        stage0_runtime_file='stage0.json',training_runtime_file='training.json',checkpoint_zero='source.json',
        checkpoint_weights_sha256='a'*64,checkpoint_config_sha256='b'*64,dataset_source='source.json',dataset_root=str(tmp_path),
        spectral='source.json',fixed_inputs='source.json',qualification='source.json',cut_reference='source.json',cut_reference_root=str(tmp_path),
        long_min_tokens=10,prompt_count=1,fidelity_tolerance_nats=.5,model_identity=dict(model='m',model_revision='r',tokenizer_revision='r',base_sha256='h'),eos_token_id=99,
        snapshot_hashes={'model.safetensors':'h'},hf_cache_root=str(tmp_path),minimum_cache_free_bytes=0,
        source_hashes={'source.json':file_hash(tmp_path/'source.json')},predictions_path='source.json',reporting_protocol_path='source.json',software_versions={'trl':'0.26.2'},minimum_free_bytes=0,
        hf_repo='owner/private',export_paths=['source.json'])
    path=tmp_path/'inputs.json';path.write_text(json.dumps(inputs))
    output=workflow.template(path,tmp_path,'example')
    spec=json.loads(output.read_text())
    assert {stage['id'] for stage in spec['stages']}==set(workflow.STAGES)
    assert spec['stages'][0]['commands'][0][0]=='offline-pool-inventory'
    assert workflow.prepare(output,tmp_path).is_file()
    from pilot_eval import cli,offline_grpo_pool,offline_grpo_fresh,offline_grpo_training,offline_grpo_dataset,offline_grpo_behaviour,offline_grpo_internals,offline_grpo_cuts
    for module,names in [(offline_grpo_pool,['inventory','prepare','import_rollouts']),
        (offline_grpo_fresh,['prepare','run']),(offline_grpo_training,['prepare','run','check_diagnostics']),
        (offline_grpo_dataset,['prepare','run','report']),(offline_grpo_behaviour,['prepare','run','report']),
        (offline_grpo_internals,['prepare','run_geometry','collect','report']),
        (offline_grpo_cuts,['prepare','build','agreement','evaluate','report'])]:
        for name in names:monkeypatch.setattr(module,name,lambda *args,**kwargs:{'fixture':True})
    for stage in spec['stages']:
        for command in stage['commands']:
            assert cli.main(command)==0


def real_export_run(tmp_path,monkeypatch,*,tamper=False):
    """Run the real report, manifest, upload and readback against a local fake Hub."""
    import json,shutil
    from types import SimpleNamespace
    from pilot_eval import offline_grpo_overnight as workflow,stage0_budget
    from pilot_eval.workflow import _hash
    spec=make_spec(tmp_path)
    runs=tmp_path/'runs/child';(runs/'checkpoints/checkpoint-0064').mkdir(parents=True)
    (runs/'checkpoints/checkpoint-0064/adapter_model.safetensors').write_bytes(b'adapter')
    (tmp_path/'models/base').mkdir(parents=True);(tmp_path/'models/base/model.safetensors').write_bytes(b'base')
    (runs/'handoff.json').write_text(json.dumps(dict(r0_path='rollout.json')))
    (tmp_path/'rollout.json').write_text(json.dumps(dict(run_path='runs/child',parents={})))
    for stage in spec['stages']:
        (tmp_path/f"{stage['id']}.json").write_text(json.dumps(dict(run_path='runs/child')))
    (tmp_path/'dataset.json').write_text(json.dumps(dict(run_path='runs/child',sft_root='/elsewhere/sources',sft_path='sft.json',sft_sha256='a'*64)))
    (tmp_path/'cut_build.json').write_text(json.dumps(dict(run_path='runs/child',reference_root='.',reference_path='source.json',reference_sha256='b'*64)))
    for stage in spec['stages']:
        if stage['id']=='behaviour':stage['report']='runs/child/behaviour-report'
    (runs/'behaviour-report/results').mkdir(parents=True)
    (runs/'behaviour-report/results/results.json').write_text(json.dumps(dict(summaries={'greedy-64':1},changes={'greedy-64':2},online_comparison=None,online_comparison_status=dict(state='not-selected'),anchor=dict(state='passed'))))
    (runs/'results').mkdir()
    (runs/'results/results.json').write_text(json.dumps(dict(full=1,paired_common=2,common_question_ids=[])))
    (runs/'report').mkdir()
    (runs/'report/results.json').write_text(json.dumps(dict(adapter_frobenius_norms=1,drift=2,preupdate=3,unavailable=[],descriptive_similarities=6,push_alignment=7,limitations=[],paired=4,summaries=5,note='n')))
    spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    path=workflow.prepare(spec_path,tmp_path);plan=json.loads(path.read_text())
    directory=tmp_path/'runs/offline-grpo/fixture/overnight';preflight=dict(ok=True,evidence={},issues=[])
    (directory/'meta/preflight.json').write_text(json.dumps(preflight))
    allocation=dict(reviewer='test',notes='fixture',session_id='fixture',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.)
    auth=dict(protocol='offline-grpo-prelaunch-v1',approved=True,plan_sha256=_hash(plan),stage0_runtime=spec['stage0_runtime'],training_runtime=spec['training_runtime'],repo=spec['export']['repo'],reviewer='test',preflight_sha256=_hash(preflight),max_total_units=1.,allocation=allocation)
    auth_path=tmp_path/'auth.json';auth_path.write_text(json.dumps(auth))
    row=dict(reservation_id='fixture',state='active',job_sha256=_hash(plan),approval=allocation)
    monkeypatch.setattr(stage0_budget,'reserve',lambda *args:row)
    monkeypatch.setattr(stage0_budget,'reservations',lambda *args:{'fixture':row})
    done=set()
    def native(stage,root):
        if stage['id'] not in done:raise FileNotFoundError(stage['id'])
        return True
    monkeypatch.setattr(workflow,'_native_verify',native)
    class Hub:
        def __init__(self):self.published={};self.commits=0
        def info(self,repo):return SimpleNamespace(private=True)
        def publish(self,repo,files):
            self.commits+=1;revision=f'{self.commits:040x}'
            for name,source in files.items():
                target=tmp_path.parent/(tmp_path.name+'-hub')/revision/name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);self.published[name,revision]=target
            return revision
        def fetch(self,repo,name,revision):return self.published[name,revision]
    hub=Hub()
    class Deps:
        fail=True;corrupt=tamper
        def runtime(self):return spec['stage0_runtime']
        def hub(self):return hub
        def fresh_fetch(self,repo,name,revision,cache):
            target=Path(cache)/'fresh.tar.gz'
            if self.corrupt:target.write_bytes(b'tampered')
            else:shutil.copyfile(hub.published[name,revision],target)
            return target
        def invoke(self,command,*,reservation_provider):
            name=command[command.index('--kind')+1] if command[0]=='offline-grpo-internals-collect' else next(s['id'] for s in spec['stages'] if command[0] in workflow.STAGES[s['id']])
            print('child output for',name)
            if name=='train' and self.fail:return 1
            if command==next(s for s in spec['stages'] if s['id']==name)['commands'][-1]:done.add(name)
            return 0
    return workflow,path,auth_path,Deps(),hub,directory


def latest_manifest(directory):
    import json
    return json.loads(max((directory/'exports').glob('*/manifest.json'),key=lambda p:p.stat().st_mtime_ns).read_text())


def test_real_partial_export_lists_gaps_and_keeps_evidence(tmp_path,monkeypatch):
    workflow,path,auth,deps,hub,directory=real_export_run(tmp_path,monkeypatch)
    state=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert state['state']=='incomplete' and state['export']['state']=='partial_verified'
    manifest=latest_manifest(directory)
    assert manifest['partial'] is True and 'stage:train:failed' in manifest['missing']
    assert 'stage:behaviour:blocked_dependency' in manifest['missing']
    files=set(manifest['files'])
    assert 'runs/child/checkpoints/checkpoint-0064/adapter_model.safetensors' in files
    assert not any(name.startswith('models/') for name in files)
    snapshot=next(name.rsplit('/',1)[0] for name in files if name.endswith('final-report.md'))
    for name in ['meta/preflight.json','meta/authorization.json','meta/umbrella.json','config.json','logs/train.log','logs/events.jsonl']:
        assert snapshot+'/'+name in files
    assert manifest['external_sources']==[dict(stage='dataset',root='/elsewhere/sources',path='sft.json',sha256='a'*64)]
    report=(tmp_path/snapshot/'final-report.md').read_text()
    assert state['export']['external_sources']==1
    assert '| train | failed |' in report and '| behaviour | blocked_dependency |' in report
    assert 'not a pure effect of staleness' in report
    with pytest.raises(ValueError,match='Completed verified export'):
        workflow.verify(path,tmp_path,dependencies=deps)


def test_readback_mismatch_is_not_called_verified_and_can_recover(tmp_path,monkeypatch):
    workflow,path,auth,deps,hub,directory=real_export_run(tmp_path,monkeypatch,tamper=True)
    deps.fail=False
    state=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert all(entry['state']=='verified' for entry in state['stages'].values())
    assert state['state']=='incomplete' and state['export']['state']=='failed'
    with pytest.raises(ValueError,match='Completed verified export'):
        workflow.verify(path,tmp_path,dependencies=deps)
    deps.corrupt=False
    state=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert state['state']=='completed' and state['export']['state']=='verified'
    assert workflow.verify(path,tmp_path,dependencies=deps)['complete'] is True


def test_completed_run_is_not_exported_twice(tmp_path,monkeypatch):
    import json
    workflow,path,auth,deps,hub,directory=real_export_run(tmp_path,monkeypatch)
    deps.fail=False
    first=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert first['state']=='completed' and hub.commits==1
    final=json.loads(next((directory/'results/snapshots').glob('*/final-report.json')).read_text())
    assert final['measurements']['behaviour']['summary']['online_comparison_status']==dict(state='not-selected')
    assert final['measurements']['internals_report']['summary']['push_alignment']==7
    second=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert second['state']=='completed' and hub.commits==1
    assert second['export']['manifest_sha256']==first['export']['manifest_sha256']


def test_interruption_preserves_budget_and_resumes_only_unverified_stages(tmp_path,monkeypatch):
    import json
    workflow,path,auth,deps,hub,directory=real_export_run(tmp_path,monkeypatch)
    deps.fail=False;calls=[];original=deps.invoke
    class Stop(KeyboardInterrupt):pass
    def invoke(command,*,reservation_provider):
        calls.append(command[0])
        if command[0]=='offline-grpo-train-run' and calls.count(command[0])==1:raise Stop()
        return original(command,reservation_provider=reservation_provider)
    deps.invoke=invoke
    with pytest.raises(Stop):workflow.run(path,tmp_path,auth,dependencies=deps)
    status=json.loads((directory/'meta/status.json').read_text())
    assert status['state']=='interrupted' and status['stages']['train']['state']=='running'
    assert status['error_type']=='Stop' and hub.commits==0
    assert all((p.parent/'timing.complete.json').exists() for p in (directory/'attempts').glob('*/start.json'))
    before=calls.count('offline-pool-import')
    state=workflow.run(path,tmp_path,auth,dependencies=deps)
    assert state['state']=='completed' and calls.count('offline-pool-import')==before==1
    assert calls.count('offline-grpo-train-run')==2


def test_killed_attempt_without_timing_refuses_resume(tmp_path,monkeypatch):
    workflow,path,auth,deps,hub,directory=real_export_run(tmp_path,monkeypatch)
    workflow.run(path,tmp_path,auth,dependencies=deps)
    killed=directory/'attempts/killed';killed.mkdir()
    (killed/'start.json').write_text('{"reservation_id":"fixture"}')
    with pytest.raises(ValueError,match='unmeasured elapsed time'):
        workflow.run(path,tmp_path,auth,dependencies=deps)


def test_run_requires_passing_preflight_and_reports_missing_source(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_overnight as workflow
    spec=make_spec(tmp_path);spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps(spec))
    path=workflow.prepare(spec_path,tmp_path)
    auth=tmp_path/'auth.json';auth.write_text('{}')
    with pytest.raises(ValueError,match='Run preflight'):
        workflow.run(path,tmp_path,auth,dependencies=object())
    (tmp_path/'source.json').unlink()
    class Deps:
        def runtime(self):raise ValueError('unavailable')
        def parity(self):raise ValueError('skipped')
        def hub(self):raise ValueError('no credentials')
    result=workflow.preflight(path,tmp_path,dependencies=Deps())
    assert not result['ok']
    assert any(i['check']=='sources' and 'missing' in i['message'] for i in result['issues'])
    assert any(i['check']=='hf' for i in result['issues'])
    with pytest.raises(ValueError,match='unresolved failures'):
        workflow.run(path,tmp_path,auth,dependencies=Deps())


def test_dataset_without_report_resumes_instead_of_failing(tmp_path,monkeypatch):
    import json
    from pilot_eval import offline_grpo_dataset,offline_grpo_overnight as workflow
    run=tmp_path/'runs/dataset';run.mkdir(parents=True)
    plan=tmp_path/'dataset.json';plan.write_text(json.dumps(dict(run_path='runs/dataset')))
    monkeypatch.setattr(offline_grpo_dataset,'verify',lambda *args:{})
    with pytest.raises(FileNotFoundError,match='Dataset report missing'):
        workflow._native_verify(dict(id='dataset',plan=str(plan)),tmp_path)


def test_parity_gate_runs_from_project_root(monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path as P
    from types import SimpleNamespace
    from pilot_eval import offline_grpo_overnight as workflow
    seen={}
    def fake(command,**kwargs):
        seen.update(kwargs);return SimpleNamespace(returncode=0,stdout='1 passed in 1s\n')
    monkeypatch.setattr(subprocess,'run',fake)
    monkeypatch.chdir(tmp_path)
    assert workflow.Dependencies().parity()['passed'] is True
    assert P(seen['cwd'])==P(workflow.__file__).parent.parent


def test_spec_requires_explicit_fidelity_tolerance(tmp_path):
    import json
    from pilot_eval import offline_grpo_overnight as workflow
    spec=make_spec(tmp_path)
    for stage in spec['stages']:
        for command in stage['commands']:
            if '--fidelity-tolerance' in command:del command[command.index('--fidelity-tolerance'):]
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='fidelity tolerance'):workflow.prepare(path,tmp_path)
