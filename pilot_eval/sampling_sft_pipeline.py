"""Resume-first sequential three-arm workflow decision engine.

This module does not treat human labels, coverage, source verification, capacity,
T15/T16 or paid job authorization as inferred success.
"""
ARMS=('0','2','5')


def next_action(state):
    for arm in ARMS:
        status=state['generation'][arm]
        if status in ('failed','interrupted'):return dict(action='inspect_failed_generation',arm=int(arm))
        if status!='completed':return dict(action='generate',arm=int(arm))
    pending=state['pending_human_labels']
    if pending is None:return dict(action='prepare_review')
    if pending:return dict(action='human_labels',count=pending)
    if state['coverage_blocked'] is None:return dict(action='freeze_coverage')
    if state['coverage_blocked']:return dict(action='coverage_blocked')
    if state['gate']!='completed':return dict(action='gate')
    if state['readiness']!='completed':return dict(action='readiness')
    for arm in ARMS:
        status=state['training'][arm]
        if status in ('failed','interrupted'):return dict(action='inspect_failed_training',arm=int(arm))
        if status!='completed':return dict(action='train',arm=int(arm))
    return dict(action='verify_export')


def advance_generation(verified, execute, uncertain=lambda arm:False):
    """Sequentially invoke independently guarded jobs; never retry uncertain work."""
    for raw in ARMS:
        arm=int(raw)
        if verified(arm):continue
        if uncertain(arm):raise ValueError('Prior job requires inspection and reconcile before a new launch.')
        execute(arm)
        if not verified(arm):raise ValueError('Job returned without a verified completed arm.')


def run_prepared_jobs(root, entries, status, run):
    """Run three already-reviewed lifecycle jobs, serially; no implicit retries."""
    from pathlib import Path
    if [e['arm'] for e in entries] != [0,2,5]:raise ValueError('Require ordered k0/k2/k5 jobs.')
    for entry in entries:
        job=Path(entry['job']); approval=Path(entry['approval']) if entry.get('approval') else None
        state=status(job)
        if state['work_complete']:
            if not state['workload_success']:raise ValueError('Prior job failed; inspect and reconcile before continuation.')
            continue
        if (job.parent/'launch.json').exists():raise ValueError('Prior launch uncertain; inspect and reconcile.')
        if approval is None:approval=create_job_approval(root,job,entry['reviewer'])
        outcome=run(job,root,approval)
        if not outcome.get('work_complete') or not outcome.get('workload_success'):
            raise ValueError('Job failed or did not seal work; inspect before continuation.')


def validate_generation_job(root,config,plan,source_plan,arm,job_path):
    import json
    from pathlib import Path
    from pilot_eval.sampling_sft_sources import inside
    job=json.loads(Path(job_path).read_text());request=job['request'];argv=request['command']
    if (request.get('kind')!='gpu' or request.get('mode')!='runpod' or request.get('stage')!='sampling'
        or job['source_plan']!=source_plan or len(argv)<4 or argv[3]!='sampling-sft-generation-run'
        or '--config' not in argv or inside(root,argv[argv.index('--config')+1])!=inside(root,config)
        or '--job' not in argv or inside(root,argv[argv.index('--job')+1])!=inside(root,job_path)
        or '--k' not in argv or argv[argv.index('--k')+1]!=str(arm)):
        raise ValueError('Job does not match this generation plan and arm.')
    base=plan['run_path']+'/k'+str(arm)
    required={base+'/'+name for name in ['complete.json','results.json','attempts.jsonl','dispositions.jsonl']}
    if not required <= set(request['outputs']):raise ValueError('Generation job output inventory incomplete.')


def scientific_inventory(root,folder):
    """Hash the complete dynamic arm tree after its scientific verifier succeeds."""
    from pathlib import Path
    from pilot_eval.sampling_sft_exports import inventory
    root=Path(root).resolve();folder=Path(folder)
    if not folder.resolve().is_relative_to(root) or not folder.is_dir():raise ValueError('Arm is outside artifact root.')
    files=[]
    for path in sorted(folder.rglob('*')):
        if path.name == 'writer.json':
            raise ValueError('Active or unrecovered writer in scientific inventory.')
        if path.name == '.lock':continue
        if path.is_symlink():raise ValueError('Scientific inventory refuses symlink.')
        if path.is_file():files.append(str(path.relative_to(root)))
    if not files:raise ValueError('Empty scientific inventory.')
    return inventory(root,files)


def generation(config, output_root, entries_path):
    """Execute three previously prepared jobs in order and seal full arm inventories."""
    import json
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_generation as gen, sampling_sft_jobs as jobs
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.workflow import _hash
    from pilot_eval.training import file_hash
    root,plan,folder=gen.load(config,output_root)
    entries=json.loads(Path(entries_path).read_text())
    if not isinstance(entries,list) or [x.get('arm') for x in entries]!=[0,2,5]:
        raise ValueError('Supply exactly ordered k0/k2/k5 preapproved job entries.')
    if len({str(inside(root,e['job'])) for e in entries})!=3:
        raise ValueError('Each arm needs a distinct job.')
    for entry in entries:
        if set(entry)!={'arm','job','reviewer'}:raise ValueError('Each entry needs arm, job and reviewer only.')
        if not isinstance(entry['reviewer'],str) or not entry['reviewer'].strip():raise ValueError('Reviewer required.')
        job_path=inside(root,entry['job'])
        jobs._load(job_path,root)
        validate_generation_job(root,config,plan,plan['source_plan'],entry['arm'],job_path)
    entries=[{**e,'job':str(inside(root,e['job']))} for e in entries]
    def status(job):return jobs.status(job,root)
    def run(job,unused,approval):
        import threading
        done=threading.Event(); log=job.parent/'logs/child.console.log'
        def follow():
            offset=0
            while not done.wait(.3):
                if log.exists():
                    with log.open(errors='replace') as stream:
                        stream.seek(offset);chunk=stream.read();offset=stream.tell()
                    if chunk:print(chunk,end='',flush=True)
            if log.exists():
                with log.open(errors='replace') as stream:
                    stream.seek(offset);print(stream.read(),end='',flush=True)
        watcher=threading.Thread(target=follow,daemon=True);watcher.start()
        try:return jobs.run(job,root,approval)
        finally:done.set();watcher.join(timeout=5)
    run_prepared_jobs(root,entries,status,run)
    result=gen.verify(config,root)
    if result['execution_kind']!='real-cuda':raise ValueError('Real CUDA generation evidence required.')
    # Inventory all generation controls and all dynamic arm trees. Exclude only
    # this inventory itself on re-verification to avoid self-reference.
    scientific=scientific_inventory(root,folder)
    scientific.pop(str((folder/'results/full-scientific-inventory.json').relative_to(root)),None)
    value=dict(protocol='sampling-sft-full-generation-inventory-v1',config_sha256=_hash(plan),
        generation_complete_sha256=file_hash(folder/'complete.json'),files=scientific,
        scope='dynamic arm trees after verified generation; external copy still required',training_authorized=False)
    value['sha256']=_hash(value)
    source.frozen(folder/'results/full-scientific-inventory.json',value)
    return dict(state='generation-verified',inventory=str(folder/'results/full-scientific-inventory.json'),
                files=len(scientific),next_step='prepare eligibility review; human labels are required if packet is nonempty')


def create_job_approval(root,job_path,reviewer):
    """Record user's explicitly authorized waiver against current ledger and request."""
    import json
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_budget as budget
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    if not reviewer.strip():raise ValueError('Reviewer required.')
    root=Path(root);job_path=Path(job_path);job=json.loads(job_path.read_text())
    current=budget.status(root);request=job['request']
    source_plan=json.loads((root/job['source_plan']).read_text())
    value=dict(reviewer=reviewer,notes='User explicitly approved sequential three-arm sampling and training and waived budget limits; retain actual cost accounting.',
        paid_execution_approved=True,preflight_reviewed=True,waive_budget_limits=True,
        budget_sha256=current['sha256'],request_sha256=_hash(request),
        source_audit_sha256=file_hash(root/source_plan['run_path']/'results/source-audit.json'))
    target=job_path.parent/'pipeline-approval.json'
    if target.exists() and json.loads(target.read_text())!=value:raise ValueError('Existing pipeline approval is stale; inspect before retry.')
    source.frozen(target,value)
    return target


def training_decision_action(request,report):
    """No implicit human acknowledgment for omissions or cohort differences."""
    coverage=report['coverage']
    if request['coverage_blocked'] or coverage['blocked']:return 'blocked'
    if (request['required_acknowledgments'] or set(coverage['uncovered_ids'])!=set(ARMS)
        or any(coverage['uncovered_ids'][k] for k in ARMS)):
        return 'human_review'
    return 'approve_three'


def review_stage(config,output_root,name):
    """Create/reuse review and stop rather than invent intended-answer labels."""
    from pilot_eval import sampling_sft_review as review
    path=review.prepare(config,output_root,name)
    current=review.status(path,output_root)
    remaining=current['required']-current['human_reviewed']
    if remaining:
        return dict(state='human-labels-required',remaining=remaining,page=current['page'],review_plan=str(path))
    coverage=review.freeze(path,output_root)
    counts={k:coverage['arms'][k]['uncovered_count'] for k in ARMS}
    if coverage['coverage_blocked']:state='coverage-blocked'
    elif any(counts.values()):state='coverage-acknowledgment-required'
    else:state='coverage-ready'
    return dict(state=state,uncovered=counts,review_plan=str(path))


def training_ready(data_config,root):
    from pilot_eval import sampling_sft_training_data as data
    readiness=data.status(data_config,root)['readiness']
    if not readiness or not all(readiness.get(k) is True for k in ['training_ready','t15_passed','t16_passed']):
        raise ValueError('Full verified T15/T16 training readiness is required.')
    return readiness


def validate_training_job(root,arm,config,job_path,job):
    from pilot_eval import sampling_sft_training as training
    from pilot_eval.sampling_sft_sources import inside
    _,plan,_=training.load(config,root)
    request=job['request'];argv=request.get('command',[])
    if (plan['k']!=arm or job['source_plan']!=plan['source_plan'] or request.get('mode')!='runpod'
        or request.get('kind')!='gpu' or request.get('stage')!='training' or len(argv)<4
        or argv[3]!='sampling-sft-train-run' or '--config' not in argv or '--job' not in argv
        or inside(root,argv[argv.index('--config')+1])!=inside(root,config)
        or inside(root,argv[argv.index('--job')+1])!=inside(root,job_path)):
        raise ValueError('Training job does not match the selected plan and arm.')
    if not set(training.required_outputs(plan))<=set(request.get('outputs',[])):
        raise ValueError('Training job output inventory incomplete.')
    return plan


def training(data_config,output_root,entries_path):
    """Execute selected, fully ready k0/k2/k5 training jobs sequentially."""
    import json
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_jobs as jobs, sampling_sft_training as train
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.workflow import _hash
    root=Path(output_root).resolve();data_config=inside(root,data_config)
    training_ready(data_config,root)
    entries=json.loads(Path(entries_path).read_text())
    if not isinstance(entries,list) or [e.get('arm') for e in entries]!=[0,2,5]:
        raise ValueError('Supply ordered k0/k2/k5 training jobs.')
    data=json.loads(data_config.read_text()); selection_paths=set();folders=[]
    for e in entries:
        if set(e)!={'arm','job','config','reviewer'} or not e['reviewer'].strip():
            raise ValueError('Training entry requires arm, job, config, reviewer.')
        job_path=inside(root,e['job']);cfg=inside(root,e['config']);_,job,_=jobs._load(job_path,root)
        plan=validate_training_job(root,e['arm'],cfg,job_path,job)
        if inside(root,plan['data_plan'])!=data_config:raise ValueError('Training data mismatch.')
        decision,_=train.selection(root,root/plan['selection'],data)
        if decision['arms']!=[0,2,5]:raise ValueError('Post-gate selection must explicitly include all three arms.')
        selection_paths.add(plan['selection']);folders.append(root/plan['run_path'])
    if len(selection_paths)!=1 or len({e['job'] for e in entries})!=3:
        raise ValueError('Require one shared selection and distinct training jobs.')
    normalized=[dict(arm=e['arm'],job=str(inside(root,e['job'])),reviewer=e['reviewer']) for e in entries]
    run_prepared_jobs(root,normalized,lambda job:jobs.status(job,root),lambda job,unused,approval:jobs.run(job,root,approval))
    results=[train.verify(inside(root,e['config']),root) for e in entries]
    if any(r.get('execution_kind')!='real-cuda' or r.get('successful_steps')!=64 for r in results):
        raise ValueError('All three verified real training runs must complete 64 steps.')
    files={}
    for folder in folders:files.update(scientific_inventory(root,folder))
    result=dict(protocol='sampling-sft-three-training-inventory-v1',selection=next(iter(selection_paths)),
        arms=[0,2,5],files=files,scope='verified training trees; external copy and cost reconciliation pending')
    result['sha256']=_hash(result)
    target=inside(root,data['run_path'])/'results/three-training-inventory.json'
    source.frozen(target,result)
    return dict(state='training-verified',inventory=str(target),files=len(files),external_copy_verified=False)


def prepare_generation_jobs(config,output_root,volume_id,rate,reviewer,estimated_hours,setup_hours):
    """Freeze explicit bounded jobs; never launch them during preparation."""
    import json, sys
    from pilot_eval import sampling_sft as source, sampling_sft_generation as gen, sampling_sft_jobs as jobs, sampling_sft_budget as budget
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    root,plan,folder=gen.load(config,output_root)
    if not volume_id.strip() or not reviewer.strip():raise ValueError('Volume and reviewer required.')
    budget.number(rate,True);budget.number(setup_hours)
    if set(estimated_hours)!=set(ARMS):raise ValueError('Explicit estimate required for each arm.')
    for value in estimated_hours.values():budget.number(value,True)
    receipt=json.loads((folder/'generation.review.json').read_text())
    gen.plan_approval(plan,receipt['review'],file_hash(root/plan['pilot_report']))
    _,assessment=gen.qualification(root,plan,receipt['review'])
    if receipt.get('config_sha256')!=_hash(plan) or receipt.get('alarm_assessment')!=assessment:
        raise ValueError('Generation review or alarm assessment changed.')
    entries=[]
    for k in (0,2,5):
        job_id=plan['name']+'-k'+str(k)
        source_plan=json.loads((root/plan['source_plan']).read_text())
        job=root/source_plan['run_path']/'jobs'/job_id/'job.plan.json'
        base=plan['run_path']+'/k'+str(k)
        request=dict(job_id=job_id,kind='gpu',mode='runpod',stage='sampling',
            rate_usd_per_hour=str(rate),estimated_hours=str(estimated_hours[str(k)]),setup_hours=str(setup_hours),
            network_volume_id=volume_id,command=[sys.executable,'-m','pilot_eval','sampling-sft-generation-run',
                '--config',str(inside(root,config)),'--k',str(k),'--job',str(job),'--output-root',str(root)],
            outputs=[base+'/'+n for n in ['complete.json','results.json','attempts.jsonl','dispositions.jsonl']])
        req=folder/'pipeline/requests'/f'k{k}.json';source.frozen(req,request)
        prepared=jobs.prepare(root/plan['source_plan'],root,req)
        entries.append(dict(arm=k,job=str(prepared.relative_to(root)),reviewer=reviewer))
    target=folder/'pipeline/generation-jobs.json';source.frozen(target,entries)
    return target


def gate_prepare(review_config,root,word_policy,name):
    """Prepare independent scoring only after verified coverage; do not label."""
    from pilot_eval import sampling_sft_review as review, sampling_sft_gate as gate
    coverage=review.verify(review_config,root)
    if coverage['coverage_blocked']:raise ValueError('Coverage exceeds the unchanged omission gate.')
    path=gate.prepare(review_config,root,word_policy,name)
    return dict(state='gate-prepared',config=str(path),training_authorized=False)


def gate_report(config,root):
    """Verify sealed independent scoring before computing the CPU gate report."""
    from pilot_eval import sampling_sft_gate as gate
    root,plan,folder=gate.load(config,root)
    gate.verify_scores(plan,folder)
    report=gate.report(config,root)
    return dict(state='gate-reported',execution_kind=report['execution_kind'],
        coverage_blocked=report['coverage']['blocked'],training_authorized=False)


def validate_gate_job(root,config,job_path,job):
    from pilot_eval import sampling_sft_gate as gate
    from pilot_eval.sampling_sft_sources import inside
    _,plan,_=gate.load(config,root);req=job['request'];argv=req.get('command',[])
    if (job['source_plan']!=plan['source_plan'] or req.get('mode')!='runpod' or req.get('kind')!='gpu'
        or req.get('stage')!='sampling' or len(argv)<4 or argv[3]!='sampling-sft-gate-score'
        or '--config' not in argv or '--job' not in argv or '--max-new-units' in argv
        or inside(root,argv[argv.index('--config')+1])!=inside(root,config)
        or inside(root,argv[argv.index('--job')+1])!=inside(root,job_path)):
        raise ValueError('Independent score job does not match full gate plan.')
    required={plan['run_path']+'/scores/'+name for name in ['complete.json','results.json']}
    if not required<=set(req.get('outputs',[])):raise ValueError('Gate score job output inventory incomplete.')
    return plan


def gate_score(config,root,job_path,reviewer):
    """Run one already prepared independent HF scoring job and preserve its tree."""
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_gate as gate, sampling_sft_jobs as jobs
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.workflow import _hash
    root=Path(root).resolve();config=inside(root,config);job_path=inside(root,job_path)
    _,job,_=jobs._load(job_path,root);plan=validate_gate_job(root,config,job_path,job)
    state=jobs.status(job_path,root)
    if state['work_complete']:
        if not state['workload_success']:raise ValueError('Prior gate job failed; inspect and reconcile.')
    else:
        if (job_path.parent/'launch.json').exists():raise ValueError('Prior gate launch uncertain; inspect and reconcile.')
        approval=create_job_approval(root,job_path,reviewer)
        state=jobs.run(job_path,root,approval)
        if not state['work_complete'] or not state['workload_success']:raise ValueError('Independent score job failed.')
    _,_,folder=gate.load(config,root)
    score_result=gate.verify_scores(plan,folder)
    if score_result.get('execution_kind')!='real-cuda':raise ValueError('Real independent score evidence required.')
    files=scientific_inventory(root,folder/'scores')
    value=dict(protocol='sampling-sft-gate-score-inventory-v1',config_sha256=_hash(plan),files=files,
        scope='complete dynamic HF score tree; external copy still required',training_authorized=False)
    value['sha256']=_hash(value)
    target=folder/'score-scientific-inventory.json';source.frozen(target,value)
    return dict(state='scores-verified',inventory=str(target),files=len(files))


def data_stage(review_config,root,name,python):
    """Build exact training data and run full, not kernel-only, T15/T16."""
    from pilot_eval import sampling_sft_training_data as data
    path=data.prepare(review_config,root,name)
    data.export(path,root)
    current=data.status(path,root)['readiness']
    readiness=current if current and current.get('training_ready') is True else data.check(path,root,python=python,mode='full')
    if readiness.get('training_ready') is not True:
        raise ValueError('Full T15/T16 failed or is incomplete; inspect retained evidence.')
    return dict(state='training-data-ready',config=str(path),readiness=readiness)


def gate_select(config,root,forecast,reviewer,name):
    """Record the user's all-arm choice only where no human coverage decision is pending."""
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_gate as gate
    from pilot_eval.sft import safe_name
    safe_name(name)
    if not reviewer.strip():raise ValueError('Reviewer required.')
    report=gate.verify(config,root);request=gate.request(config,root,forecast)
    if request['selection']['arms']!=[0,2,5]:raise ValueError('User-selected scope requires all three arms.')
    action=training_decision_action(request,report)
    if action!='approve_three':return dict(state='coverage-blocked' if action=='blocked' else 'human-review-required',
        request_path=request['request_path'],required_acknowledgments=request['required_acknowledgments'])
    approval=dict(request_path=request['request_path'],request_sha256=request['request_sha256'],
        report_sha256=request['report_sha256'],budget_sha256=request['budget_snapshot']['sha256'],
        reviewer=reviewer,notes='User selected all three arms and waived budget limits; no coverage acknowledgments were needed.',
        gate_approved=True,acknowledged_coverage_flags=[],review_80_percent=True,waive_budget_limits=True)
    target=Path(root)/'plans'/name/'gate-approval.json';source.frozen(target,approval)
    selection=gate.approve(config,root,target)
    return dict(state='selection-recorded',selection_path=selection['selection_path'],training_authorized=False)


def prepare_training_jobs(data_config,selection,output_root,name,volume_id,rate,reviewer,estimated_hours,setup_hours):
    """Prepare separately bounded selected jobs after full readiness; no launch."""
    import json
    from pathlib import Path
    from pilot_eval import sampling_sft as source, sampling_sft_training as train, sampling_sft_jobs as jobs, sampling_sft_budget as budget
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.sft import safe_name
    root=Path(output_root).resolve();data_config=inside(root,data_config);selection=inside(root,selection)
    safe_name(name)
    if not volume_id.strip() or not reviewer.strip() or set(estimated_hours)!=set(ARMS):
        raise ValueError('Explicit volume, reviewer and per-arm estimates required.')
    budget.number(rate,True);budget.number(setup_hours)
    for value in estimated_hours.values():budget.number(value,True)
    training_ready(data_config,root)
    dc=json.loads(data_config.read_text());decision,_=train.selection(root,selection,dc)
    if decision['arms']!=[0,2,5]:raise ValueError('Selection must include all three user-selected arms.')
    entries=[];folder=inside(root,dc['run_path'])/'pipeline'/name
    for arm in (0,2,5):
        arm_name=name+'-k'+str(arm)
        plan=train.prepare(data_config,selection,root,arm_name,arm)
        template=dict(job_id=arm_name,kind='gpu',mode='runpod',stage='training',network_volume_id=volume_id,
            rate_usd_per_hour=str(rate),estimated_hours=str(estimated_hours[str(arm)]),setup_hours=str(setup_hours))
        path=folder/'templates'/f'k{arm}.json';source.frozen(path,template)
        request=train.job_request(plan,root,path)
        job=jobs.prepare(root/dc['source_plan'],root,request)
        entries.append(dict(arm=arm,config=str(plan.relative_to(root)),job=str(job.relative_to(root)),reviewer=reviewer))
    target=folder/'training-jobs.json';source.frozen(target,entries)
    return target


def prepare_gate_job(config,output_root,name,volume_id,rate,reviewer,estimated_hours,setup_hours):
    """Prepare an independently scored gate job with explicit bounded estimate."""
    import json,sys
    from pilot_eval import sampling_sft as source, sampling_sft_gate as gate, sampling_sft_jobs as jobs, sampling_sft_budget as budget
    from pilot_eval.sampling_sft_sources import inside
    from pilot_eval.sft import safe_name
    root,plan,folder=gate.load(config,output_root);safe_name(name)
    if not volume_id.strip() or not reviewer.strip():raise ValueError('Volume and reviewer required.')
    budget.number(rate,True);budget.number(estimated_hours,True);budget.number(setup_hours)
    source_plan=json.loads((root/plan['source_plan']).read_text())
    job=root/source_plan['run_path']/'jobs'/name/'job.plan.json'
    request=dict(job_id=name,kind='gpu',mode='runpod',stage='sampling',network_volume_id=volume_id,
        rate_usd_per_hour=str(rate),estimated_hours=str(estimated_hours),setup_hours=str(setup_hours),
        command=[sys.executable,'-m','pilot_eval','sampling-sft-gate-score','--config',str(inside(root,config)),
                 '--job',str(job),'--output-root',str(root)],
        outputs=[plan['run_path']+'/scores/'+n for n in ['complete.json','results.json']])
    path=folder/'pipeline'/(name+'-request.json');source.frozen(path,request)
    job=jobs.prepare(root/plan['source_plan'],root,path)
    return dict(job=str(job),reviewer=reviewer)
