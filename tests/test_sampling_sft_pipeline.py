import pytest
from pathlib import Path
from pilot_eval.sampling_sft_pipeline import next_action


def test_pipeline_sequential_resumable_and_stops_at_review_boundaries():
    s={'generation':{'0':'completed','2':'pending','5':'pending'},'pending_human_labels':None,
       'coverage_blocked':None,'gate':'pending','readiness':'pending','training':{'0':'pending','2':'pending','5':'pending'}}
    assert next_action(s)=={'action':'generate','arm':2}
    s['generation']['2']='failed'
    assert next_action(s)=={'action':'inspect_failed_generation','arm':2}
    s['generation']['2']='completed';s['generation']['5']='completed';s['pending_human_labels']=3
    assert next_action(s)=={'action':'human_labels','count':3}
    s['pending_human_labels']=0;s['coverage_blocked']=True
    assert next_action(s)=={'action':'coverage_blocked'}
    s['coverage_blocked']=False
    assert next_action(s)=={'action':'gate'}
    s['gate']='completed';assert next_action(s)=={'action':'readiness'}
    s['readiness']='completed';s['training']['0']='completed'
    assert next_action(s)=={'action':'train','arm':2}
    s['training']['2']='failed';assert next_action(s)=={'action':'inspect_failed_training','arm':2}
    s['training']['2']='completed';s['training']['5']='completed'
    assert next_action(s)=={'action':'verify_export'}


def test_sequential_executor_resumes_without_double_launch_and_stops_on_failure():
    from pilot_eval.sampling_sft_pipeline import advance_generation
    completed=set(); launched=[]
    def verify(arm): return arm in completed
    def execute(arm):
        launched.append(arm)
        if arm==2:raise RuntimeError('retained failure')
        completed.add(arm)
    with pytest.raises(RuntimeError,match='retained failure'):
        advance_generation(verify,execute)
    assert launched==[0,2] and completed=={0}
    def resume(arm):launched.append(arm);completed.add(arm)
    with pytest.raises(ValueError,match='reconcile'):
        advance_generation(verify,resume,lambda arm:arm==2)
    assert launched==[0,2]
    advance_generation(verify,resume,lambda arm:False)
    assert launched==[0,2,2,5] and completed=={0,2,5}


def test_run_prepared_jobs_never_relaunches_failed_or_incomplete_work(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p
    jobs=[];statuses={}
    for arm in (0,2,5):
        job=tmp_path/f'job{arm}.json';job.write_text('{}')
        approval=tmp_path/f'approval{arm}.json';approval.write_text('{}')
        jobs.append({'arm':arm,'job':str(job),'approval':str(approval)})
        statuses[str(job)]={'work_complete':False}
    calls=[]
    def run(job,root,approval):
        calls.append(str(job));statuses[str(job)]={'work_complete':True,'workload_success':True}
        return statuses[str(job)]
    p.run_prepared_jobs(tmp_path,jobs,lambda job:statuses[str(job)],run)
    p.run_prepared_jobs(tmp_path,jobs,lambda job:statuses[str(job)],run)
    assert calls==[j['job'] for j in jobs]
    statuses[jobs[1]['job']]={'work_complete':True,'workload_success':False}
    with pytest.raises(ValueError,match='failed'):p.run_prepared_jobs(tmp_path,jobs,lambda job:statuses[str(job)],run)
    assert len(calls)==3


def test_prepared_job_manifest_binds_arm_plan_and_required_payloads(tmp_path):
    import json
    from pilot_eval.sampling_sft_pipeline import validate_generation_job
    root=tmp_path; plan={'run_path':'runs/source/generation/g'}
    source='plans/source.json';config=root/'plans/g.json';config.parent.mkdir();config.write_text('{}')
    job=root/'runs/source/jobs/j/job.plan.json';job.parent.mkdir(parents=True)
    base=plan['run_path']+'/k0'
    request={'kind':'gpu','mode':'runpod','stage':'sampling','command':['python','-m','pilot_eval','sampling-sft-generation-run','--config',str(config),'--k','0','--job',str(job)],
             'outputs':[base+'/'+x for x in ['complete.json','results.json','attempts.jsonl','dispositions.jsonl']]}
    job.write_text(json.dumps({'source_plan':source,'request':request}))
    validate_generation_job(root,config,plan,source,0,job)
    request['outputs'].remove(base+'/attempts.jsonl');job.write_text(json.dumps({'source_plan':source,'request':request}))
    with pytest.raises(ValueError,match='inventory'):validate_generation_job(root,config,plan,source,0,job)


def test_dynamic_scientific_inventory_tracks_all_round_files_and_rejects_symlink(tmp_path):
    from pilot_eval.sampling_sft_pipeline import scientific_inventory
    root=tmp_path; folder=root/'runs/source/generation/g/k0';(folder/'rounds/0/blocks/000').mkdir(parents=True)
    (folder/'complete.json').write_text('{}');(folder/'rounds/0/blocks/000/scientific.jsonl').write_text('record\n')
    inventory=scientific_inventory(root,folder)
    assert 'runs/source/generation/g/k0/rounds/0/blocks/000/scientific.jsonl' in inventory
    (folder/'secret-link').symlink_to(root/'outside')
    with pytest.raises(ValueError,match='symlink'):scientific_inventory(root,folder)


def test_waiver_receipt_is_request_and_live_ledger_bound(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p
    from pilot_eval import sampling_sft_budget as budget
    from pilot_eval.workflow import _hash
    root=tmp_path;job=root/'runs/source/jobs/j/job.plan.json';job.parent.mkdir(parents=True)
    request={'job_id':'j','stage':'sampling'};job.write_text(json.dumps({'source_plan':'plans/source.json','request':request}))
    source=root/'plans/source.json';source.parent.mkdir();source.write_text(json.dumps({'run_path':'runs/source'}))
    audit=root/'runs/source/results/source-audit.json';audit.parent.mkdir();audit.write_text('{}')
    monkeypatch.setattr(budget,'status',lambda root:{'sha256':'current'})
    target=p.create_job_approval(root,job,'Prasad')
    record=json.loads(target.read_text())
    assert record['request_sha256']==_hash(request) and record['budget_sha256']=='current'
    assert record['waive_budget_limits'] is True and record['paid_execution_approved'] is True
    assert p.create_job_approval(root,job,'Prasad')==target
    monkeypatch.setattr(budget,'status',lambda root:{'sha256':'changed'})
    with pytest.raises(ValueError,match='stale'):p.create_job_approval(root,job,'Prasad')


def test_pipeline_runs_real_lifecycle_sequentially_and_inventory_includes_blocks(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p
    from pilot_eval import sampling_sft_generation as g, sampling_sft_jobs as j
    root=tmp_path;folder=root/'runs/source/generation/g';folder.mkdir(parents=True)
    config=root/'plans/g.json';config.parent.mkdir();config.write_text('{}')
    plan={'run_path':str(folder.relative_to(root)),'source_plan':'plans/source.json'}
    entries=[]
    for k in (0,2,5):
        job=root/f'runs/source/jobs/k{k}/job.plan.json';job.parent.mkdir(parents=True)
        base=plan['run_path']+f'/k{k}'
        request={'kind':'gpu','mode':'runpod','stage':'sampling','command':['python','-m','pilot_eval','sampling-sft-generation-run','--config',str(config),'--k',str(k),'--job',str(job)],
                 'outputs':[base+'/'+x for x in ['complete.json','results.json','attempts.jsonl','dispositions.jsonl']]}
        job.write_text(json.dumps({'source_plan':plan['source_plan'],'request':request}))
        entries.append({'arm':k,'job':str(job),'reviewer':'Prasad'})
    ep=root/'entries.json';ep.write_text(json.dumps(entries));calls=[];completed=set()
    monkeypatch.setattr(g,'load',lambda *a:(root,plan,folder))
    monkeypatch.setattr(j,'_load',lambda *a:None)
    monkeypatch.setattr(j,'status',lambda job,root:dict(work_complete=str(job) in completed,workload_success=str(job) in completed))
    monkeypatch.setattr(p,'create_job_approval',lambda root,job,reviewer:job.parent/'approval.json')
    def run(job,root,approval):
        k=int(job.parent.name[1:]);calls.append(k);completed.add(str(job))
        path=folder/f'k{k}/rounds/0/blocks/000';path.mkdir(parents=True)
        (path/'scientific.jsonl').write_text('saved\n')
        return dict(work_complete=True,workload_success=True)
    monkeypatch.setattr(j,'run',run)
    (folder/'complete.json').write_text('{}')
    monkeypatch.setattr(g,'verify',lambda *a:{'execution_kind':'real-cuda'})
    result=p.generation(config,root,ep)
    inventory=json.loads(Path(result['inventory']).read_text())
    assert calls==[0,2,5] and len(inventory['files'])==4
    assert sum('/rounds/0/blocks/000/scientific.jsonl' in name for name in inventory['files'])==3
    assert str((folder/'complete.json').relative_to(root)) in inventory['files']
    p.generation(config,root,ep);assert calls==[0,2,5]


def test_pipeline_stops_before_next_job_when_output_verification_fails(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p
    entries=[{'arm':k,'job':str(tmp_path/f'k{k}.json'),'approval':str(tmp_path/f'k{k}.approval')} for k in (0,2,5)]
    calls=[]
    def run(job,root,approval):calls.append(job);return {'work_complete':True,'workload_success':False}
    with pytest.raises(ValueError,match='failed'):p.run_prepared_jobs(tmp_path,entries,lambda job:{'work_complete':False},run)
    assert len(calls)==1


def test_complete_inventory_includes_top_level_generation_evidence(tmp_path):
    from pilot_eval.sampling_sft_pipeline import scientific_inventory
    root=tmp_path;folder=root/'generation';folder.mkdir()
    for file in ['complete.json','generation.review.json','run_k0.json']:(folder/file).write_text('{}')
    inv=scientific_inventory(root,folder)
    assert {'generation/complete.json','generation/generation.review.json','generation/run_k0.json'}<=set(inv)


def test_auto_training_gate_only_allows_zero_uncovered_and_no_human_flags():
    from pilot_eval.sampling_sft_pipeline import training_decision_action
    request={'required_acknowledgments':[], 'coverage_blocked':False}
    report={'coverage':{'blocked':False,'uncovered_ids':{'0':[],'2':[],'5':[]}}}
    assert training_decision_action(request,report)=='approve_three'
    request['required_acknowledgments']=['k2:omissions-require-T16']
    assert training_decision_action(request,report)=='human_review'
    request['required_acknowledgments']=[];report['coverage']['uncovered_ids']['2']=[123]
    assert training_decision_action(request,report)=='human_review'
    report['coverage']['blocked']=True
    assert training_decision_action(request,report)=='blocked'


def test_user_selected_all_three_jobs_train_sequentially_only_when_ready(tmp_path):
    from pilot_eval.sampling_sft_pipeline import run_prepared_jobs
    entries=[];statuses={};calls=[]
    for k in (0,2,5):
        path=tmp_path/f'train{k}.json';path.write_text('{}')
        entries.append({'arm':k,'job':str(path),'approval':str(path)+'.approval'})
        statuses[str(path)]={'work_complete':False}
    def run(job,root,approval):
        calls.append(job.name);statuses[str(job)]={'work_complete':True,'workload_success':True}
        return statuses[str(job)]
    run_prepared_jobs(tmp_path,entries,lambda job:statuses[str(job)],run)
    assert calls==['train0.json','train2.json','train5.json']


def test_review_stage_pauses_for_real_labels_and_freezes_empty_packet(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_review as review
    plan=tmp_path/'plans/review/sampling-sft.review.json'
    monkeypatch.setattr(review,'prepare',lambda *a:plan)
    monkeypatch.setattr(review,'status',lambda *a:{'required':2,'human_reviewed':0,'page':'review.html','frozen':False})
    monkeypatch.setattr(review,'freeze',lambda *a: (_ for _ in ()).throw(AssertionError('must pause')))
    result=p.review_stage(tmp_path/'generation.json',tmp_path,'review')
    assert result['state']=='human-labels-required' and result['remaining']==2
    monkeypatch.setattr(review,'status',lambda *a:{'required':0,'human_reviewed':0,'page':'review.html','frozen':False})
    monkeypatch.setattr(review,'freeze',lambda *a:{'coverage_blocked':False,'arms':{k:{'uncovered_count':0} for k in ['0','2','5']}})
    result=p.review_stage(tmp_path/'generation.json',tmp_path,'review')
    assert result['state']=='coverage-ready'


def test_training_pipeline_rejects_unready_evidence_and_uses_user_three_arm_selection(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training_data as data
    monkeypatch.setattr(data,'status',lambda *a:{'readiness':{'training_ready':False,'t16_passed':False}})
    with pytest.raises(ValueError,match='T15/T16'):p.training_ready('data',tmp_path)
    monkeypatch.setattr(data,'status',lambda *a:{'readiness':{'training_ready':True,'t15_passed':True,'t16_passed':True}})
    assert p.training_ready('data',tmp_path)['training_ready']


def test_training_job_validation_requires_selected_arm_and_complete_exports(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training as training
    root=tmp_path;plan=root/'plans/k2.json';plan.parent.mkdir();plan.write_text('{}')
    job=root/'runs/source/jobs/k2/job.plan.json';job.parent.mkdir(parents=True);job.write_text('{}')
    config={'k':2,'source_plan':'plans/source.json'}
    monkeypatch.setattr(training,'load',lambda *a:(root,config,root))
    monkeypatch.setattr(training,'required_outputs',lambda config:['required/checkpoint'])
    record={'source_plan':'plans/source.json','request':{'mode':'runpod','kind':'gpu','stage':'training',
        'command':['python','-m','pilot_eval','sampling-sft-train-run','--config',str(plan),'--job',str(job)],'outputs':['required/checkpoint']}}
    p.validate_training_job(root,2,plan,job,record)
    record['request']['outputs']=[]
    with pytest.raises(ValueError,match='inventory'):p.validate_training_job(root,2,plan,job,record)
    with pytest.raises(ValueError):p.validate_training_job(root,0,plan,job,record)


def test_training_pipeline_executes_verified_selected_arms_once(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training as t, sampling_sft_jobs as j
    root=tmp_path; data=root/'plans/data.json';data.parent.mkdir();data.write_text(json.dumps({'run_path':'data'}))
    entries=[]; completed=set();calls=[]
    for k in (0,2,5):
        job=root/f'jobs/k{k}/job.plan.json';job.parent.mkdir(parents=True);job.write_text('{}')
        cfg=root/f'plans/k{k}.json';cfg.write_text('{}')
        entries.append(dict(arm=k,job=str(job),config=str(cfg),reviewer='Prasad'))
    ep=root/'jobs.json';ep.write_text(json.dumps(entries))
    monkeypatch.setattr(p,'training_ready',lambda *a:{'training_ready':True})
    monkeypatch.setattr(j,'_load',lambda job,root:(root,{},job.parent))
    monkeypatch.setattr(p,'validate_training_job',lambda root,k,cfg,job,record:dict(data_plan='plans/data.json',selection='selection.json',run_path=f'training/k{k}'))
    monkeypatch.setattr(t,'selection',lambda *a:({'arms':[0,2,5]},{}))
    monkeypatch.setattr(j,'status',lambda job,root:dict(work_complete=str(job) in completed,workload_success=str(job) in completed))
    monkeypatch.setattr(p,'create_job_approval',lambda root,job,reviewer:job.parent/'approval.json')
    def run(job,root,approval):
        calls.append(job.parent.name);completed.add(str(job));k=job.parent.name
        folder=root/'training'/k;folder.mkdir(parents=True);(folder/'complete.json').write_text('{}')
        return {'work_complete':True,'workload_success':True}
    monkeypatch.setattr(j,'run',run)
    monkeypatch.setattr(t,'verify',lambda *a:{'execution_kind':'real-cuda','successful_steps':64})
    result=p.training(data,root,ep)
    assert result['state']=='training-verified' and calls==['k0','k2','k5']
    p.training(data,root,ep);assert calls==['k0','k2','k5']


def test_generation_job_preparation_saves_three_distinct_bounded_requests(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_generation as g, sampling_sft_jobs as j
    root=tmp_path;config=root/'plans/g.json';config.parent.mkdir();config.write_text('{}')
    folder=root/'runs/source/generation/g';folder.mkdir(parents=True)
    plan={'name':'g','source_plan':'plans/source.json','run_path':'runs/source/generation/g','pilot_report':'report.json'}
    (root/'plans/source.json').write_text(json.dumps({'run_path':'runs/source'}))
    (root/'report.json').write_text('{}')
    monkeypatch.setattr(g,'load',lambda *a:(root,plan,folder))
    monkeypatch.setattr(g,'qualification',lambda *a:({},{}))
    from pilot_eval.workflow import _hash
    (folder/'generation.review.json').write_text(json.dumps({'config_sha256':_hash(plan),'alarm_assessment':{},'review':{'generation_approved':True}}))
    monkeypatch.setattr(g,'plan_approval',lambda *a:None)
    monkeypatch.setattr(j,'prepare',lambda source,root,request:root/'runs/source/jobs'/json.loads(request.read_text())['job_id']/'job.plan.json')
    path=p.prepare_generation_jobs(config,root,'volume','0.60','Prasad',{'0':'3','2':'4','5':'5'},'0')
    entries=json.loads(path.read_text());assert [x['arm'] for x in entries]==[0,2,5]
    for e in entries:
        request=json.loads((folder/'pipeline'/'requests'/f"k{e['arm']}.json").read_text())
        assert request['estimated_hours']=={'0':'3','2':'4','5':'5'}[str(e['arm'])]
        assert request['network_volume_id']=='volume' and request['stage']=='sampling'
        assert request['command'][request['command'].index('--k')+1]==str(e['arm'])
    assert p.prepare_generation_jobs(config,root,'volume','0.60','Prasad',{'0':'3','2':'4','5':'5'},'0')==path
    with pytest.raises(ValueError):p.prepare_generation_jobs(config,root,'volume','0.60','Prasad',{'0':'3','2':'4','5':'6'},'0')


def test_gate_preparation_needs_frozen_review_and_bound_word_policy(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_review as review, sampling_sft_gate as gate
    monkeypatch.setattr(review,'verify',lambda *a:{'coverage_blocked':False})
    monkeypatch.setattr(gate,'prepare',lambda review,root,word,name:tmp_path/'plans'/name/'sampling-sft.gate.json')
    result=p.gate_prepare('review',tmp_path,'policy','gate')
    assert result['state']=='gate-prepared' and result['training_authorized'] is False


def test_gate_report_runs_only_after_sealed_independent_scores(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_gate as gate
    monkeypatch.setattr(gate,'load',lambda *a:(tmp_path,{'run_path':'gate'},tmp_path/'gate'))
    monkeypatch.setattr(gate,'verify_scores',lambda *a: (_ for _ in ()).throw(ValueError('missing scores')))
    with pytest.raises(ValueError,match='missing scores'):p.gate_report('gate',tmp_path)
    monkeypatch.setattr(gate,'verify_scores',lambda *a:{'execution_kind':'real-cuda'})
    monkeypatch.setattr(gate,'report',lambda *a:{'execution_kind':'real-cuda','coverage':{'blocked':False}})
    assert p.gate_report('gate',tmp_path)['state']=='gate-reported'


def test_gate_score_job_requires_full_completion_output(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_gate as g
    root=tmp_path;cfg=root/'plans/gate.json';cfg.parent.mkdir();cfg.write_text('{}')
    job=root/'jobs/gate/job.plan.json';job.parent.mkdir(parents=True);job.write_text('{}')
    plan={'source_plan':'plans/source.json','run_path':'runs/source/gate/g'}
    monkeypatch.setattr(g,'load',lambda *a:(root,plan,root/'runs/source/gate/g'))
    record={'source_plan':plan['source_plan'],'request':{'mode':'runpod','kind':'gpu','stage':'sampling',
      'command':['python','-m','pilot_eval','sampling-sft-gate-score','--config',str(cfg),'--job',str(job)],
      'outputs':['runs/source/gate/g/scores/complete.json','runs/source/gate/g/scores/results.json']}}
    p.validate_gate_job(root,cfg,job,record)
    record['request']['outputs']=[]
    with pytest.raises(ValueError,match='inventory'):p.validate_gate_job(root,cfg,job,record)


def test_data_stage_requires_frozen_coverage_and_runs_full_readiness(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training_data as d
    calls=[];plan=tmp_path/'plans/data/sampling-sft.training-data.json'
    monkeypatch.setattr(d,'prepare',lambda *a:plan)
    monkeypatch.setattr(d,'export',lambda *a:calls.append('export'))
    monkeypatch.setattr(d,'status',lambda *a:{'readiness':None})
    monkeypatch.setattr(d,'check',lambda *a,**kw:(calls.append(kw['mode']) or {'training_ready':True}))
    result=p.data_stage('review',tmp_path,'data','/python')
    assert result['state']=='training-data-ready' and calls==['export','full']


def test_gate_decision_automation_stops_on_coverage_and_binds_current_request(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_gate as g
    report={'coverage':{'blocked':False,'uncovered_ids':{'0':[],'2':[],'5':[]}}}
    request={'required_acknowledgments':[],'coverage_blocked':False,'request_path':'runs/request.json',
      'request_sha256':'request','report_sha256':'report','budget_snapshot':{'sha256':'ledger'},
      'selection':{'arms':[0,2,5]}}
    monkeypatch.setattr(g,'verify',lambda *a:report)
    monkeypatch.setattr(g,'request',lambda *a:request)
    def approve(config,root,path):
        value=json.loads(path.read_text())
        assert value['waive_budget_limits'] and value['budget_sha256']=='ledger'
        assert value['acknowledged_coverage_flags']==[]
        return {'selection_path':'runs/selection.json'}
    monkeypatch.setattr(g,'approve',approve)
    result=p.gate_select('gate',tmp_path,'forecast','Prasad','decision')
    assert result['state']=='selection-recorded'
    request['required_acknowledgments']=['k2:omissions-require-T16']
    assert p.gate_select('gate',tmp_path,'forecast','Prasad','other')['state']=='human-review-required'


def test_readiness_reuses_passed_check_without_running_again(tmp_path,monkeypatch):
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training_data as d
    monkeypatch.setattr(d,'prepare',lambda *a:tmp_path/'plan')
    monkeypatch.setattr(d,'export',lambda *a:None)
    monkeypatch.setattr(d,'status',lambda *a:{'readiness':{'training_ready':True,'t15_passed':True,'t16_passed':True}})
    monkeypatch.setattr(d,'check',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must reuse')))
    assert p.data_stage('review',tmp_path,'data','/python')['state']=='training-data-ready'


def test_prepare_training_builds_three_named_plans_and_jobs_without_launch(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_training as t, sampling_sft_jobs as j
    data=tmp_path/'plans/data.json';data.parent.mkdir();data.write_text(json.dumps({'source_plan':'plans/source.json','run_path':'runs/source/data/d'}))
    selection=tmp_path/'selection.json';selection.write_text('{}')
    monkeypatch.setattr(p,'training_ready',lambda *a:{'training_ready':True})
    monkeypatch.setattr(t,'selection',lambda *a:({'arms':[0,2,5]},{}))
    def prepare(data,selection,root,name,k):
        target=root/'plans'/name/'sampling-sft.training.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}');return target
    monkeypatch.setattr(t,'prepare',prepare)
    def job_request(plan,root,template):
        req=json.loads(template.read_text());target=plan.parent/'request.json';target.write_text(json.dumps(req));return target
    monkeypatch.setattr(t,'job_request',job_request)
    monkeypatch.setattr(j,'prepare',lambda source,root,request:root/'jobs'/json.loads(request.read_text())['job_id']/'job.plan.json')
    target=p.prepare_training_jobs(data,selection,tmp_path,'sample','volume','0.6','Prasad',{'0':'1','2':'2','5':'3'},'0')
    entries=json.loads(target.read_text());assert [e['arm'] for e in entries]==[0,2,5]


def test_prepare_gate_job_requires_explicit_estimate_and_full_score_inventory(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_sft_pipeline as p, sampling_sft_gate as g, sampling_sft_jobs as j
    root=tmp_path;config=root/'plans/gate.json';config.parent.mkdir();config.write_text('{}')
    folder=root/'runs/source/gate/g';folder.mkdir(parents=True)
    (root/'plans/source.json').write_text(json.dumps({'run_path':'runs/source'}))
    plan={'source_plan':'plans/source.json','run_path':'runs/source/gate/g'}
    monkeypatch.setattr(g,'load',lambda *a:(root,plan,folder))
    monkeypatch.setattr(j,'prepare',lambda source,root,request:root/'runs/source/jobs'/json.loads(request.read_text())['job_id']/'job.plan.json')
    result=p.prepare_gate_job(config,root,'gscore','volume','0.6','Prasad','4','0')
    request=json.loads((folder/'pipeline/gscore-request.json').read_text())
    assert request['command'][3]=='sampling-sft-gate-score' and len(request['outputs'])==2


def test_inventory_ignores_persistent_lock_but_rejects_writer(tmp_path):
    from pilot_eval.sampling_sft_pipeline import scientific_inventory
    folder=tmp_path/'arm';folder.mkdir();(folder/'.lock').write_text('');(folder/'complete.json').write_text('{}')
    assert list(scientific_inventory(tmp_path,folder))==['arm/complete.json']
    (folder/'writer.json').write_text('{}')
    with pytest.raises(ValueError,match='writer'):scientific_inventory(tmp_path,folder)


def test_job_runner_stops_when_prior_success_does_not_verify_science(tmp_path):
    from pilot_eval.sampling_sft_pipeline import advance_generation
    with pytest.raises(ValueError,match='returned without'):
        advance_generation(lambda arm:False,lambda arm:None)
