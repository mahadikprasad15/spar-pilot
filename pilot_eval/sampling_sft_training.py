"""Post-gate endpoint training plans, immutable checkpoints and paid-job gates.

The optional notebook and RunPod child call this same public workflow. CPU fixture
injection tests orchestration only; it cannot establish real L4 qualification.
"""
import json
import math
import os
import shutil
import time
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_gate as gate
from pilot_eval import sampling_sft_generation as generation
from pilot_eval import sampling_sft_training_data as data
from pilot_eval import sampling_sft_budget as budget
from pilot_eval.sampling_sft_sources import inside,verify_parents
from pilot_eval.sft import safe_name
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import CHECKPOINT_FILES,file_hash
from pilot_eval.run import _write_state
from pilot_eval.workflow import _hash

ADAPTER_FILES={'adapter_config.json','adapter_model.safetensors'}
REQUIRED_STEPS=[0,8,16,32,64]


def implementation():
    return {n:file_hash(Path(__file__).with_name(n)) for n in
        ['sampling_sft_training.py','sampling_sft_training_backend.py','sft_backend.py']}


def seal(folder,step,recovery,plan_sha,runtime_sha):
    required=CHECKPOINT_FILES if recovery else ADAPTER_FILES
    if any(not (folder/n).is_file() for n in required):raise ValueError('Incomplete adapter/optimizer recovery state.')
    if recovery and json.loads((folder/'trainer_state.json').read_text()).get('global_step')!=step:
        raise ValueError('Recovery progress does not match checkpoint step.')
    source.frozen(folder/'complete.json',dict(step=step,recoverable=recovery,config_sha256=plan_sha,
        runtime_sha256=runtime_sha,files={n:file_hash(folder/n) for n in sorted(required)}))


def checkpoint(folder,plan_sha,runtime_sha=None,require_recovery=False):
    value=json.loads((folder/'complete.json').read_text())
    required=CHECKPOINT_FILES if value['recoverable'] else ADAPTER_FILES
    if (set(value['files'])!=required or value['config_sha256']!=plan_sha
            or value['step']!=int(folder.name.removeprefix('checkpoint-'))
            or (runtime_sha is not None and value['runtime_sha256']!=runtime_sha)):
        raise ValueError('Checkpoint plan/runtime/file identity changed.')
    for n,h in value['files'].items():
        if file_hash(inside(folder,n))!=h:raise ValueError('Checkpoint content changed.')
    if value['recoverable']:
        state=json.loads((folder/'trainer_state.json').read_text())
        if state['global_step']!=value['step']:raise ValueError('Checkpoint progress changed.')
    if require_recovery and not value['recoverable']:
        raise ValueError('Adapter-only state cannot recover optimizer/scheduler/progress/RNG; prepare a new restart attempt.')
    return value


def selection(root,path,data_config):
    path=inside(root,path);receipt=json.loads(path.read_text())
    if path.stem!=_hash(receipt) or receipt.get('selection_confirmed') is not True:
        raise ValueError('Invalid immutable post-gate selection receipt.')
    request_path=inside(root,receipt['request_path']);request=json.loads(request_path.read_text())
    # The gate plan is explicitly bound in its own folder, rather than inferred
    # from user-supplied names. Locate its one matching saved plan identity.
    matches=[]
    for candidate in (root/'plans').glob('*/sampling-sft.gate.json'):
        candidate_config=json.loads(candidate.read_text())
        if _hash(candidate_config)==request['config_sha256']:matches.append(candidate)
    if len(matches)!=1:raise ValueError('Expected one source-bound saved gate plan.')
    _,config,folder=gate.load(matches[0],root)
    # Restore the already reviewed immutable report; recomputing its 10,000
    # bootstrap resamples is the separate gate-verify command, not training.
    report_seal=json.loads((folder/'complete.json').read_text())
    if (report_seal['config_sha256']!=_hash(config)
            or report_seal['scores_sha256']!=file_hash(folder/'scores/complete.json')):
        raise ValueError('Reviewed report/score seal changed.')
    for n,h in report_seal['files'].items():
        if file_hash(inside(folder,n))!=h:raise ValueError('Reviewed report/plot/row payload changed.')
    gate.verify_scores(config,folder)
    report=json.loads((folder/'results/report.json').read_text())
    if (path.parent!=folder/'training-selections' or config['review_plan']!=data_config['review_plan']
            or receipt['request_sha256']!=file_hash(request_path)
            or receipt['report_sha256']!=file_hash(folder/'results/report.json')
            or request['report_sha256']!=receipt['report_sha256']
            or request['score_evidence_sha256']!=file_hash(folder/'scores/complete.json')
            or receipt['arms']!=request['selection']['arms']
            or receipt['execution_kind']!=report['execution_kind'] or report['coverage']['blocked']
            or receipt['approval']['gate_approved'] is not True
            or receipt['approval']['request_sha256']!=receipt['request_sha256']
            or receipt['approval']['report_sha256']!=receipt['report_sha256']
            or receipt['approval']['budget_sha256']!=request['budget_snapshot']['sha256']
            or sorted(receipt['approval']['acknowledged_coverage_flags'])!=request['required_acknowledgments']):
        raise ValueError('Post-gate selection/report/coverage evidence changed.')
    return receipt,request


def prepare(data_path,selection_path,output_root,name,k,capture_recovery=True,resume_from=None,restart_of=None):
    if k not in [0,5]:raise ValueError('Only approved k0/k5 endpoints; k2 needs separate scope approval.')
    if resume_from and restart_of:raise ValueError('Choose resume-from or restart-of, not both.')
    root=Path(output_root).resolve();data_path=inside(root,data_path);data.verify(data_path,root)
    dc=json.loads(data_path.read_text());df=inside(root,dc['run_path'])
    decision,_=selection(root,selection_path,dc)
    if k not in decision['arms']:raise ValueError('This arm was not selected after the gate.')
    safe_name(name);_,sp,_=source.load(root/dc['source_plan'],root)
    folder=root/sp['run_path']/'training'/name
    parents={str(inside(root,data_path).relative_to(root)):file_hash(inside(root,data_path)),
        str((df/'export.complete.json').relative_to(root)):file_hash(df/'export.complete.json'),
        str(inside(root,selection_path).relative_to(root)):file_hash(inside(root,selection_path))}
    continuation=None
    if resume_from or restart_of:
        oldpath=inside(root,resume_from or restart_of);_,old,oldfolder=load(oldpath,root)
        if old['k']!=k or old['data_plan']!=str(inside(root,data_path).relative_to(root)):
            raise ValueError('Continuation needs the same arm and training data.')
        if not (oldfolder/'started.json').exists() or (oldfolder/'complete.json').exists():
            raise ValueError('Only an interrupted attempt can be resumed/restarted; reuse completed results.')
        if (oldfolder/'writer.json').exists():raise ValueError('Prior writer ownership exists; inspect and recover it before continuing.')
        candidates=checkpoints(oldfolder,old)
        continuation=dict(kind='resume' if resume_from else 'restart',plan=str(oldpath.relative_to(root)),step=0)
        parents[str(oldpath.relative_to(root))]=file_hash(oldpath)
        if resume_from:
            if not candidates:raise ValueError('No sealed optimizer recovery checkpoint; use an approved restart.')
            step=max(candidates);cp=oldfolder/'checkpoints'/f'checkpoint-{step}'
            checkpoint(cp,_hash(old),require_recovery=True)
            continuation.update(step=step,checkpoint=str(cp.relative_to(root)))
            parents[str((cp/'complete.json').relative_to(root))]=file_hash(cp/'complete.json')
    config=dict(protocol='sampling-sft-training-v1',name=name,k=k,data_plan=str(inside(root,data_path).relative_to(root)),
        selection=str(inside(root,selection_path).relative_to(root)),source_plan=dc['source_plan'],gold_config=dc['gold_config'],
        parents=parents,implementation=implementation(),capture_recovery=bool(capture_recovery),continuation=continuation,
        run_path=str(folder.relative_to(root)),seed=42,optimizer_steps=64,checkpoint_steps=REQUIRED_STEPS,
        length_policy='preserve all retained IDs; arm-specific maximum; stop at model context limit',
        expected_base_sha256=json.loads((root/sp['run_path']/'results/source-audit.json').read_text())['training']['base_sha256'],
        step_policy='one original batch per optimizer step; microbatch1; actual supervised-token denominator',
        training_authorized=False)
    path=root/'plans'/name/'sampling-sft.training.json';source.frozen(path,config)
    source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(config),config=config))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);config=json.loads(path.read_text());folder=inside(root,config['run_path'])
    if (config.get('protocol')!='sampling-sft-training-v1' or config['k'] not in [0,5]
            or config['implementation']!=implementation()
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(config),'config':config}):
        raise ValueError('Training plan/code/manifest changed.')
    verify_parents(root,config['parents'])
    data.verify(root/config['data_plan'],root);dc=json.loads((root/config['data_plan']).read_text())
    selection(root,root/config['selection'],dc)
    return root,config,folder


def checkpoints(folder,config):
    result={}
    for p in (folder/'checkpoints').glob('checkpoint-*'):
        if (p/'complete.json').exists():result[int(p.name.removeprefix('checkpoint-'))]=checkpoint(p,_hash(config))
    return result


def event(folder,kind,**values):
    target=folder/'logs/events.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
    from datetime import datetime,timezone
    with target.open('a') as f:
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),kind=kind,**values),allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())


def required_outputs(config):
    names=['complete.json','results/results.json','meta/runtime.json','meta/status.json','meta/run_manifest.json',
        'started.json','inputs/order.json','inputs/readiness.json','checkpoints/progress.json','logs/run.log','meta/resolved_training.json','results/initialization.json',
        'initialization-loaded/adapter_config.json','initialization-loaded/adapter_model.safetensors','logs/steps.jsonl','logs/steps.csv','logs/events.jsonl']
    steps=REQUIRED_STEPS
    files=CHECKPOINT_FILES if config['capture_recovery'] else ADAPTER_FILES
    return [config['run_path']+'/'+n for n in names]+[
        f'{config["run_path"]}/checkpoints/checkpoint-{s}/{n}' for s in steps for n in sorted(files|{'complete.json'})]


def job_request(path,output_root,template_path):
    root,config,folder=load(path,output_root)
    request=json.loads(Path(template_path).read_text());safe_name(request['job_id'])
    if request.get('mode')!='runpod' or request.get('kind')!='gpu' or request.get('stage')!='training':
        raise ValueError('Use an explicit RunPod training estimate and volume identity.')
    _,sp,_=source.load(root/config['source_plan'],root)
    job=root/sp['run_path']/'jobs'/request['job_id']/'job.plan.json'
    request.update(command=['python','-m','pilot_eval','sampling-sft-train-run','--config',str(inside(root,path).relative_to(root)),
        '--job',str(job.relative_to(root)),'--output-root',str(root)],outputs=required_outputs(config))
    budget.estimate(request)
    target=folder/'job-requests'/(request['job_id']+'.json');source.frozen(target,request)
    return target


def authorize_job(root,job):
    """Pre-launch check: no model import; the lifecycle still reserves and approves cost."""
    request=job['request'];argv=request.get('command',[])
    if len(argv)<4 or argv[3]!='sampling-sft-train-run' or '--config' not in argv or '--job' not in argv:
        raise ValueError('Training jobs must invoke the approved endpoint workflow.')
    path=inside(root,argv[argv.index('--config')+1]);_,config,folder=load(path,root)
    if (job['source_plan']!=config['source_plan'] or request['stage']!='training'
            or inside(root,argv[argv.index('--job')+1])!=root/job['directory']/'job.plan.json'
            or not set(required_outputs(config)).issubset(request['outputs'])):
        raise ValueError('Training job/source/config/export inventory mismatch.')
    if (folder/'complete.json').exists():raise ValueError('Completed arm is immutable; verify/reuse it without a new paid launch.')
    if (folder/'started.json').exists():raise ValueError('Interrupted attempt preserved; prepare a new resume/restart plan and job.')
    ready=data.status(root/config['data_plan'],root)['readiness']
    if not ready or ready.get('training_ready') is not True or ready.get('t16_passed') is not True:
        raise ValueError('Full verified T15/T16 readiness is required before paid training.')
    _,dc,df=data.load(root/config['data_plan'],root)
    gold=json.loads((root/config['gold_config']).read_text())
    exported=json.loads((df/'results/export.json').read_text())
    gen=json.loads((root/dc['generation_plan']).read_text())
    model_config=inside(root,gen['settings']['local_model'])/'config.json'
    context=json.loads(model_config.read_text()).get('max_position_embeddings')
    if type(context) is not int or context<=0:raise ValueError('Verified local model context limit is unavailable; inspect before launch.')
    if exported['arms'][str(config['k'])]['max_length']>context:
        raise ValueError('Retained target exceeds the verified model context limit; no truncation occurs.')
    _,dc,_=data.load(root/config['data_plan'],root);receipt,request_gate=selection(root,root/config['selection'],dc)
    if receipt['execution_kind']!='real-cuda':raise ValueError('CPU fixture gate evidence cannot authorize real training.')
    # Fresh reservations are enforced by the lifecycle ledger, not an obsolete
    # gate-time ledger hash. A larger per-job estimate needs a new gate selection.
    approved=request_gate['selection']['forecast']['per_arm'][str(config['k'])]
    ceiling=budget.micros(budget.number(approved['rate_usd_per_hour'],True)*
        (budget.number(approved['estimated_hours'],True)*budget.number('1.5')+budget.number(approved['setup_hours'])))
    if budget.estimate(request)>ceiling:raise ValueError('Job exceeds the gate-reviewed arm forecast; obtain an updated selection.')
    return path,config


def authorize_active_job(root,path,job_path):
    """Stage-specific child check; leave the hash-bound sampler module unchanged."""
    from pilot_eval import sampling_sft_jobs as jobs
    _,job,folder=jobs._load(job_path,root);jobs._state(folder,job)
    authorize_job(root,job)
    launch=json.loads((folder/'launch.json').read_text());request=job['request'];argv=request['command']
    if (request['mode']!='runpod' or request['stage']!='training'
            or launch['pod_id']!=os.environ.get('RUNPOD_POD_ID')
            or inside(root,argv[argv.index('--config')+1])!=inside(root,path)
            or (folder/'execution.json').exists()):
        raise ValueError('Training child does not match the active approved job.')
    ledger=budget.status(root);current=ledger['jobs'].get(request['job_id'],{})
    if (ledger['cap_exceeded'] or not current.get('launch_recorded') or current.get('state')!='reserved'
            or current.get('request_sha256')!=job['request_sha256']):
        raise ValueError('Training job has no active approved reservation.')


def run(path,output_root,job_path=None,*,dependencies=None):
    root,config,folder=load(path,output_root)
    if (folder/'complete.json').exists():return verify(path,root)
    real=dependencies is None
    if real:
        from pilot_eval import sampling_sft_jobs as jobs
        if not job_path:raise ValueError('A fresh approved reserved training job is required; no model imported.')
        authorize_active_job(root,path,job_path)
    with _writer(folder):
        if (folder/'started.json').exists():raise ValueError('Interrupted attempt preserved; prepare a new resume/restart plan.')
        dc=json.loads((root/config['data_plan']).read_text());df=inside(root,dc['run_path'])
        rows=generation.read_lines(df/f'arms/k{config["k"]}/training.jsonl')
        batches=json.loads((df/f'arms/k{config["k"]}/batches.json').read_text())
        gold=json.loads((root/config['gold_config']).read_text())
        engine=None;started=time.monotonic()
        if real:
            readiness=json.loads((df/'results/readiness.json').read_text())
        else:
            readiness=dict(scope='cpu-fixture; full T16 deliberately not claimed',t16_passed=False,training_ready=False)
        source.frozen(folder/'inputs/readiness.json',readiness)
        source.frozen(folder/'started.json',dict(config_sha256=_hash(config),execution_kind='real-cuda' if real else 'cpu-fixture',
            job=str(inside(root,job_path).relative_to(root)) if job_path else None,readiness_sha256=_hash(readiness)))
        source.frozen(folder/'inputs/order.json',batches)
        source.frozen(folder/'meta/resolved_training.json',dict(gold=gold,gold_max_length=gold['max_length'],
            arm_max_length=max(len(r['input_ids']) for r in rows),microbatch_size=1,
            step_policy=config['step_policy'],max_grad_norm=1.0,weight_decay=0,lr_schedule='constant',
            bf16=False,fp16=False,tf32=False,autocast=False,seed=42,optimizer_foreach=False))
        event(folder,'started',k=config['k'],continuation=config['continuation'],budget=generation.snapshot(root))
        try:
            from pilot_eval.sampling_sft_training_backend import train
            if real:
                from pilot_eval.sampling_sft_training_backend import open_engine
                engine,runtime=open_engine(gold,max(len(r['input_ids']) for r in rows))
            else:
                engine,runtime=dependencies.open(gold,rows,df/'initialization/checkpoint-0')
                runtime={**runtime,'execution_kind':'cpu-fixture'}
            source.frozen(folder/'meta/runtime.json',runtime)
            result=train(engine,rows,batches,gold,config,folder,root,df/'initialization/checkpoint-0',runtime,
                after_step=getattr(dependencies,'after_step',None) if dependencies else None)
            result.update(execution_kind=runtime['execution_kind'],k=config['k'],training_seed=42,
                config_sha256=_hash(config),runtime_sha256=_hash(runtime),budget_snapshot=generation.snapshot(root),
                seconds=time.monotonic()-started)
            source.frozen(folder/'results/results.json',result)
            event(folder,'completed',steps=64,seconds=result['seconds'])
            _write_state(folder,'completed',64,64)
            # Seal exactly the declared scientific export. PEFT may also write
            # README/model-card files; they must not become hidden dependencies
            # of a downloaded bundle whose inventory never requested them.
            files={n.removeprefix(config['run_path']+'/'):file_hash(inside(root,n))
                for n in required_outputs(config) if n!=config['run_path']+'/complete.json'}
            source.frozen(folder/'complete.json',dict(config_sha256=_hash(config),files=files))
            return verify(path,root)
        except BaseException as exc:
            event(folder,'interrupted',error=str(exc),seconds=time.monotonic()-started,budget=generation.snapshot(root))
            _write_state(folder,'failed',max(checkpoints(folder,config),default=0),64,str(exc));raise
        finally:
            if engine is not None and hasattr(engine,'close'):engine.close()


def verify(path,output_root):
    root,config,folder=load(path,output_root);complete=json.loads((folder/'complete.json').read_text())
    if complete['config_sha256']!=_hash(config):raise ValueError('Completed training plan identity changed.')
    for n,h in complete['files'].items():
        if file_hash(inside(folder,n))!=h:raise ValueError('Completed arm/export payload changed.')
    runtime=json.loads((folder/'meta/runtime.json').read_text());available=checkpoints(folder,config)
    expected_files={n.removeprefix(config['run_path']+'/') for n in required_outputs(config) if n!=config['run_path']+'/complete.json'}
    if not expected_files.issubset(complete['files']):raise ValueError('Training export seal omits required products.')
    if not set(REQUIRED_STEPS).issubset(available):raise ValueError('Missing required adapter checkpoints.')
    if any(v['runtime_sha256']!=_hash(runtime) for v in available.values()):raise ValueError('Checkpoint runtime differs from the completed arm.')
    rows=generation.read_lines(folder/'logs/steps.jsonl');dc=json.loads((root/config['data_plan']).read_text());df=inside(root,dc['run_path'])
    batches=json.loads((df/f'arms/k{config["k"]}/batches.json').read_text())
    exported={r['id']:r for r in generation.read_lines(df/f'arms/k{config["k"]}/training.jsonl')}
    gold=json.loads((root/config['gold_config']).read_text())
    if [r['step'] for r in rows]!=list(range(1,65)):raise ValueError('Missing/duplicated optimizer logs.')
    for row,batch in zip(rows,batches):
        ids=[s['source_id'] for s in batch['slots'] if not s['missing']]
        if row['example_ids']!=ids or row['missing_slots']!=[s for s in batch['slots'] if s['missing']]:
            raise ValueError('Step exposures/missing slots changed.')
        count=sum(sum(v!=-100 for v in exported[i]['labels'][1:]) for i in ids)
        if (row['supervised_tokens']!=count or row['optimizer_steps']!=row['step']
                or row['scheduler_steps']!=row['step'] or row['learning_rate']!=gold['optimizer']['learning_rate']):
            raise ValueError('Step denominator, optimizer/scheduler or schedule changed.')
        if any(not math.isfinite(float(row[n])) for n in ['loss','grad_norm','learning_rate']):
            raise ValueError('Nonfinite training log.')
    result=json.loads((folder/'results/results.json').read_text())
    ready=json.loads((folder/'inputs/readiness.json').read_text())
    started=json.loads((folder/'started.json').read_text())
    if started['readiness_sha256']!=_hash(ready):raise ValueError('Launch readiness evidence changed.')
    if result['execution_kind']=='real-cuda' and (not ready['t16_passed'] or not ready['training_ready'] or ready['config_sha256']!=_hash(dc)):
        raise ValueError('Real training lacks its verified T16 launch evidence.')
    if (result['config_sha256']!=_hash(config) or result['target_count']!=196 or result['successful_steps']!=64 or result['example_exposures']!=sum(b['nonmissing_count'] for b in batches)
            or result['runtime_sha256']!=_hash(runtime) or result['base_sha256_before']!=result['base_sha256_after']):
        raise ValueError('Training summary does not reconcile.')
    return result


def status(path,output_root):
    _,config,folder=load(path,output_root)
    if (folder/'complete.json').exists():return verify(path,output_root)
    values=checkpoints(folder,config)
    return dict(state='interrupted' if (folder/'started.json').exists() else 'prepared',k=config['k'],
        sealed_steps=sorted(values),recoverable_steps=[s for s,v in values.items() if v['recoverable']],
        training_authorized=False,full_t16_required=True)


def recover_writer(path,output_root,owner_token,review_notes):
    import platform
    from pilot_eval.training import run_lock
    root,config,folder=load(path,output_root)
    if not review_notes.strip():raise ValueError('Record writer inspection notes.')
    with run_lock(folder):
        marker=folder/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Writer owner token changed.')
        if owner.get('host')==platform.node():
            try:os.kill(int(owner['pid']),0)
            except ProcessLookupError:pass
            else:raise ValueError('Recorded training writer is still active.')
        event(folder,'writer-recovered',owner=owner,review_notes=review_notes)
        marker.unlink()
    return status(path,root)
