"""Saved bounded-job lifecycle shared by RunPod scripts and the optional notebook."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from pilot_eval import sampling_sft as sources
from pilot_eval import sampling_sft_budget as budget
from pilot_eval import sampling_sft_exports as copies
from pilot_eval.sampling_sft_provider import RunPodProvider
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash


def now():return datetime.now(timezone.utc).isoformat()


def _implementation():
    parent=Path(__file__).parent
    return {name:file_hash(parent/name) for name in ['sampling_sft_jobs.py','sampling_sft_budget.py',
        'sampling_sft_exports.py','sampling_sft_provider.py']}


def prepare(config,output_root,request_path):
    root,plan,_=sources.load(config,output_root)
    request=json.loads(Path(request_path).read_text())
    budget.estimate(request)
    mode=request.get('mode')
    if mode not in ['noop','runpod'] or (mode=='noop')!=(request['kind']=='noop'):
        raise ValueError('Use either the built-in zero-cost noop or an explicitly approved RunPod GPU job.')
    if mode=='runpod' and (request['kind']!='gpu' or not request.get('network_volume_id')
            or not isinstance(request.get('command'),list) or not request['command']
            or any(not isinstance(x,str) or not x for x in request['command'])):
        raise ValueError('RunPod jobs need a GPU estimate, network-volume identity and explicit argv command.')
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


def _state(directory,job):
    path=directory/'state.json'
    if not path.exists():
        if (directory/'events.jsonl').exists():raise ValueError('Lifecycle state missing; do not reset its event history.')
        return dict(schema_version=1,job_sha256=_hash(job),events=[])
    value=json.loads(path.read_text());payload={k:v for k,v in value.items() if k!='sha256'}
    if value.get('sha256')!=_hash(payload) or value.get('job_sha256')!=_hash(job):
        raise ValueError('Job lifecycle state integrity failure.')
    encoded=''.join(json.dumps(e,sort_keys=True)+'\n' for e in value['events'])
    log=directory/'events.jsonl'
    existing=log.read_text() if log.exists() else ''
    if not encoded.startswith(existing):raise ValueError('Lifecycle event log changed outside the committed state.')
    if existing!=encoded:
        log.write_text(encoded) # Recover a truncated projection of committed atomic state.
    return value


def _event(directory,job,kind,**extra):
    value=_state(directory,job)
    value={k:v for k,v in value.items() if k!='sha256'}
    value['events']=[*value['events'],dict(sequence=len(value['events']),kind=kind,at=now(),**extra)]
    value['sha256']=_hash(value)
    sources.save(directory/'state.json',value)
    log=directory/'events.jsonl'
    with log.open('w') as stream:
        stream.write(''.join(json.dumps(e,sort_keys=True)+'\n' for e in value['events']))
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return value


def _work(directory,root,job):
    marker=directory/'work-complete.json'
    if not marker.exists():return None
    manifest=copies.verify_manifest(directory/'export-manifest.json',root)
    if json.loads(marker.read_text())!=dict(manifest_sha256=manifest['sha256'],job_sha256=_hash(job)):
        raise ValueError('Job work seal differs from export manifest.')
    return manifest


def _approval(job,root,approval):
    source=sources.status(root/job['source_plan'],root)
    if _hash(json.loads((root/job['source_plan']).read_text()))!=job['source_plan_sha256']:
        raise ValueError('Job source-plan identity changed.')
    if not source['audit'] or not source['audit']['source_ready']:
        raise ValueError('Paid job source preflight is not ready.')
    audit_sha=file_hash(root/json.loads((root/job['source_plan']).read_text())['run_path']/'results/source-audit.json')
    if (not isinstance(approval,dict) or approval.get('paid_execution_approved') is not True
            or approval.get('preflight_reviewed') is not True
            or approval.get('request_sha256')!=job['request_sha256']
            or approval.get('source_audit_sha256')!=audit_sha
            or not all(isinstance(approval.get(k),str) and approval[k].strip() for k in ['reviewer','notes'])):
        raise ValueError('Explicit paid-job approval bound to this request and source preflight is required.')


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
            _approval(job,root,approval)
            pod_id=provider.own_pod_id();observed=provider.inspect(pod_id)
            if observed.get('id')!=pod_id or observed.get('networkVolumeId')!=request['network_volume_id']:
                raise ValueError('Own Pod/network-volume identity mismatch.')
        else:pod_id=None
        budget.initialize(root/job['source_plan'],root)
        budget.reserve(root/job['source_plan'],root,directory/'request.json',approval_path)
        snapshot=budget.authorize_launch(root,request['job_id'],request,approval)
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


def export(path,output_root,copy_root,role,receipt_path,*,provider=None):
    root,job,directory=_load(path,output_root)
    with run_lock(directory):
        manifest=_work(directory,root,job)
        if not manifest:raise ValueError('Job has no sealed work to export.')
        for other in (directory/'receipts').glob('*.json') if (directory/'receipts').exists() else []:
            existing=json.loads(other.read_text())
            if existing['role']!=role and (existing['host'],existing['copy_root'])==(copies.host(provider),str(Path(copy_root).resolve())):
                raise ValueError('Two export roles cannot designate the same copy.')
        result=copies.export(directory/'export-manifest.json',root,copy_root,role,receipt_path,provider=provider)
        sources.save(directory/'receipts'/f'{role}.json',result)
        _event(directory,job,'export-verified',role=role,receipt_sha256=result['sha256'])
        return result


def finish(path,output_root,network_receipt,download_receipt,*,provider=None):
    root,job,directory=_load(path,output_root);provider=provider or RunPodProvider()
    with run_lock(directory):
        manifest=_work(directory,root,job)
        if not manifest:raise ValueError('No sealed work/export inventory.')
        copies.check_receipts(manifest,network_receipt,download_receipt,root)
        state=_state(directory,job)
        if state['events'][-1]['kind'] in ['stop-confirmed','noop-completed']:
            return status(path,output_root)
        if job['request']['mode']=='noop':
            _event(directory,job,'noop-completed',scope='Offline copies only; no Pod stopped.')
            return status(path,output_root)
        launch=json.loads((directory/'launch.json').read_text())
        if provider.own_pod_id()!=launch['pod_id']:
            raise ValueError('Automatic shutdown is restricted to the job’s own Pod.')
        _event(directory,job,'stop-requested',pod_id=launch['pod_id'])
        try:
            provider.stop(launch['pod_id']) # Last external mutation; confirmation is a separate client command.
        except Exception as exc:
            _event(directory,job,'stop-request-failed',error_type=type(exc).__name__)
            raise RuntimeError('Stop request failed; Pod status and cost remain unresolved. Inspect immediately.') from exc
        return dict(state='stop-requested',pod_id=launch['pod_id'],confirmed=False,workload_success=manifest['workload_success'],
                    budget='reservation remains until actual billing is reconciled')


def confirm(path,output_root,*,provider=None):
    root,job,directory=_load(path,output_root);provider=provider or RunPodProvider()
    with run_lock(directory):
        state=_state(directory,job)
        if not any(e['kind']=='stop-requested' for e in state['events']):raise ValueError('No verified stop request to confirm.')
        pod_id=json.loads((directory/'launch.json').read_text())['pod_id']
        if copies.host(provider)==json.loads((directory/'launch.json').read_text())['host']:
            raise ValueError('Confirm shutdown from the external controller, not the self-stopping worker.')
        observed=provider.inspect(pod_id)
        if observed.get('id')!=pod_id or observed.get('desiredStatus')!='EXITED':
            raise ValueError('Provider does not report this Pod stopped; costs remain uncertain.')
        _event(directory,job,'stop-confirmed',pod_id=pod_id,scope='external provider-reported EXITED; billing reconciled separately')
        return status(path,output_root)


def execute(path,output_root,network_root,download_receipt,approval_path=None,max_wait_seconds=0,*,provider=None):
    """Run script path: work, network export, bounded external-ack wait, own stop."""
    wait=budget.number(max_wait_seconds)
    root,job,directory=_load(path,output_root)
    maximum=(budget.number(job['request']['estimated_hours'],True)*3600*budget.number('1.5')) if job['request']['mode']=='runpod' else 0
    if wait>maximum:raise ValueError('Download-ack wait exceeds this job’s bounded estimate.')
    began=time.monotonic()
    run(path,root,approval_path,provider=provider)
    network=directory/'network-receipt.json'
    export(path,root,network_root,'network',network,provider=provider)
    while not Path(download_receipt).is_file():
        remaining=float(wait)-(time.monotonic()-began)
        if remaining<=0:
            _event(directory,job,'export-blocked',reason='External download receipt missing; reservation retained; operator action required.')
            raise ValueError('Downloaded-copy verification is missing. No stop/delete attempted; inspect immediately, costs remain reserved.')
        print('Waiting for external downloaded-copy receipt; not a completed job.',flush=True)
        time.sleep(min(10,remaining))
    return finish(path,root,network,download_receipt,provider=provider)
