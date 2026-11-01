"""Versioned lifecycle for mixed training, with immutable legacy helpers."""
import json
import re
from pathlib import Path
from pilot_eval import sampling_sft as sources, sampling_sft_budget as budget, sampling_sft_exports as copies
from pilot_eval import sampling_sft_mixed_training as training
from pilot_eval.sampling_sft_jobs import now,_state,_event,_work,_approval
from pilot_eval.sampling_sft_provider import RunPodProvider
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash


def _implementation():
    folder=Path(__file__).parent
    return {name:file_hash(folder/name) for name in ['sampling_sft_mixed_jobs.py','sampling_sft_jobs.py',
        'sampling_sft_mixed_training.py','sampling_sft_budget.py','sampling_sft_exports.py','sampling_sft_provider.py','sampling_sft_hf.py']}


def authorize_training(root,job,validate_only=False):
    request=job['request'];argv=request.get('command',[])
    if (request.get('mode')!='runpod' or request.get('kind')!='gpu' or request.get('stage')!='training'
        or len(argv)<4 or argv[3]!='sampling-sft-mixed-training-run'):
        raise ValueError('Mixed job must match the approved mixed training stage.')
    if not validate_only:training.authorize_job(root,job)


def prepare(config,output_root,request_path):
    root,plan,_=sources.load(config,output_root)
    request=json.loads(Path(request_path).read_text())
    budget.estimate(request)
    authorize_training(root,{'request':request},validate_only=True)
    mode=request.get('mode')
    if mode not in ['noop','runpod'] or (mode=='noop')!=(request['kind']=='noop'):
        raise ValueError('Use either the built-in zero-cost noop or an explicitly approved RunPod GPU job.')
    if mode=='runpod' and (request['kind']!='gpu' or not request.get('network_volume_id')
            or not isinstance(request.get('command'),list) or not request['command']
            or any(not isinstance(x,str) or not x for x in request['command'])):
        raise ValueError('RunPod jobs need a GPU estimate, network-volume identity and explicit argv command.')
    if mode=='runpod':
        command=request['command']
        python=bool(re.fullmatch(r'python(?:3(?:\.\d+)?)?',Path(command[0]).name))
        module=command[2] if len(command)>2 and command[1]=='-m' else None
        scientific=(module=='pilot_eval' and len(command)>3 and command[3].startswith('sampling-sft-')
                    and not command[3].startswith(('sampling-sft-job-','sampling-sft-budget-'))
                    and 'recover-writer' not in command[3])
        if not python or not (module=='pytest' or scientific):
            raise ValueError('Use a reviewed project scientific CLI/test entrypoint; shell and cloud-management launchers are not accepted.')
    outputs=request.get('outputs')
    if not isinstance(outputs,list) or not outputs:raise ValueError('Declare required job output paths.')
    for name in outputs:
        target=inside(root,copies.checked_relative(name))
        base=root/plan['run_path']
        if not target.is_relative_to(base) or target.is_relative_to(base/'jobs'):
            raise ValueError('Job outputs must be in this plan’s canonical run tree, outside lifecycle metadata.')
    directory=root/plan['run_path']/'jobs'/safe_name(request['job_id'])
    job=dict(schema_version=1,source_plan=str(Path(config).resolve().relative_to(root)),
        source_plan_sha256=_hash(plan),request=request,request_sha256=_hash(request),
        directory=str(directory.relative_to(root)),implementation=_implementation())
    path=directory/'job.plan.json'
    with run_lock(directory):
        sources.frozen(path,job)
        sources.frozen(directory/'job-seal.json',dict(plan_file_sha256=file_hash(path),job_sha256=_hash(job)))
        sources.frozen(directory/'request.json',request)
        if not (directory/'state.json').exists():_event(directory,job,'prepared')
    return path


def _load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path)
    job=json.loads(path.read_text())
    seal=json.loads((path.parent/'job-seal.json').read_text())
    if (seal!=dict(plan_file_sha256=file_hash(path),job_sha256=_hash(job))
            or job.get('implementation')!=_implementation()
            or job.get('request_sha256')!=_hash(job['request'])
            or inside(root,job['directory'])!=path.parent):
        raise ValueError('Job identity, implementation or request changed; preserve this attempt.')
    return root,job,path.parent


def run(path,output_root,approval_path=None,*,provider=None):
    root,job,directory=_load(path,output_root)
    provider=provider or RunPodProvider();request=job['request']
    approval=json.loads(Path(approval_path).read_text()) if approval_path else None
    with run_lock(directory):
        _state(directory,job)
        if _work(directory,root,job):return status(path,output_root)
        if (directory/'launch.json').exists():
            raise ValueError('Interrupted/uncertain job launch: inspect and reconcile; never relaunch automatically.')
        if request['mode']=='runpod':
            if request['stage']=='training':
                authorize_training(root,job)
            _approval(job,root,approval)
            pod_id=provider.own_pod_id();observed=provider.inspect(pod_id)
            if observed.get('id')!=pod_id or observed.get('storageVolumeId',observed.get('networkVolumeId'))!=request['network_volume_id']:
                raise ValueError('Own Pod/network-volume identity mismatch.')
            if observed.get('storageMountPath') and not root.is_relative_to(Path(observed['storageMountPath'])):
                raise ValueError('Artifact root is outside the verified persistent storage mount.')
            if budget.number(observed.get('costPerHr'),True)>budget.number(request['rate_usd_per_hour'],True):
                raise ValueError('Provider hourly quote exceeds the reserved rate; re-estimate before launch.')
        else:pod_id=None
        budget.initialize(root/job['source_plan'],root)
        budget.reserve(root/job['source_plan'],root,directory/'request.json',approval_path)
        snapshot=budget.authorize_launch(root,request['job_id'],request,approval,pod_id=pod_id)
        launch=dict(pod_id=pod_id,host=copies.host(provider),started_at=now(),
                    job_sha256=_hash(job),budget_sha256=snapshot['sha256'])
        sources.frozen(directory/'launch.json',launch)
        _event(directory,job,'launch-recorded',pod_id=pod_id,budget=snapshot)
        log=directory/'logs/child.console.log'
        try:
            if request['mode']=='noop':
                for name in request['outputs']:
                    if inside(root,name).exists():raise ValueError('Noop output already exists; refuse to overwrite it.')
                    sources.save(inside(root,name),dict(scope='offline-noop',paid_execution=False))
                log.parent.mkdir(parents=True,exist_ok=True);log.write_text('Offline no-op; no provider or model calls.\n')
                result=dict(returncode=0,elapsed_seconds=0)
            else:
                timeout=float(budget.number(request['estimated_hours'],True)*3600*budget.number('1.5'))
                result=provider.run_job(request,root,log,timeout)
        except Exception as exc:
            result=dict(returncode=None,error_type=type(exc).__name__,error=str(exc),elapsed_seconds=None)
            if not log.exists():log.parent.mkdir(parents=True,exist_ok=True);log.write_text('Job failed; see execution receipt.\n')
        missing=[name for name in request['outputs'] if not inside(root,name).is_file()]
        result.update(finished_at=now(),missing_expected_outputs=missing,
                      success=result.get('returncode')==0 and not missing,
                      actual_cost='unreconciled; reservation retained until explicit billing observation')
        sources.save(directory/'execution.json',result)
        names=[str(p.relative_to(root)) for p in [path,directory/'job-seal.json',directory/'request.json',directory/'launch.json',directory/'execution.json',log]]
        names += [name for name in request['outputs'] if name not in missing]
        manifest=dict(schema_version=1,job_sha256=_hash(job),files=copies.inventory(root,names),
            simulation=request['mode']=='noop',worker_host=launch['host'],pod_id=pod_id,
            workload_success=result['success'],missing_expected_outputs=missing)
        manifest['sha256']=_hash(manifest)
        sources.save(directory/'export-manifest.json',manifest)
        sources.save(directory/'work-complete.json',dict(manifest_sha256=manifest['sha256'],job_sha256=_hash(job)))
        _event(directory,job,'work-ended',success=result['success'],reservation='retained pending actual billing')
        return status(path,output_root)


def status(path,output_root):
    root,job,directory=_load(path,output_root)
    state=_state(directory,job);manifest=_work(directory,root,job)
    return dict(job_id=job['request']['job_id'],mode=job['request']['mode'],
        phase=state['events'][-1]['kind'] if state['events'] else 'prepared',
        work_complete=manifest is not None,workload_success=manifest['workload_success'] if manifest else None,
        job_sha256=_hash(job),request_sha256=job['request_sha256'],
        budget=budget.status(root) if (budget.folder(root)/'budget.json').exists() else None)

