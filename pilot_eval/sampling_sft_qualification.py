"""Explicit stages for the bounded qualification handoff; never full generation."""
import json
import random
import sys
import threading
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval import sampling_sft_pilot as pilot
from pilot_eval import sampling_sft_jobs as jobs
from pilot_eval import sampling_sft_budget as budget
from pilot_eval.sampling_sft_sources import inside, verify_parents
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

STAGES=['fixtures','refinement','scores','oracle','profile','select',
        'pilot-reference','pilot-kill','pilot-resume','compare']


def historical_fixtures(root, capture_config, source_config):
    """Verify a frozen old capture without asserting its code equals this release."""
    root=Path(root).resolve(); capture_config=inside(root,capture_config)
    old=json.loads(capture_config.read_text()); folder=inside(root,old['run_path'])
    if (old.get('protocol')!='sampling-sft-bounded-instrument-v1'
        or old['settings']['mode']!='fixtures'
        or inside(root,old['source_plan'])!=inside(root,source_config)
        or old['runtime_hash']!=_hash(old['settings']['runtime'])
        or json.loads((folder/'plan-seal.json').read_text())!={'config_sha256':_hash(old)}):
        raise ValueError('Historical fixture capture plan identity mismatch.')
    source_path=inside(root,source_config)
    source_plan=json.loads(source_path.read_text())
    if 'run_path' not in source_plan:raise ValueError('Historical source plan is incomplete.')
    audit=inside(root,source_plan['run_path'])/'results/source-audit.json'
    required={str(source_path.relative_to(root)),str(audit.relative_to(root))}
    if not required <= set(old.get('parents',{})):
        raise ValueError('Historical fixture capture lacks source/audit parent bindings.')
    verify_parents(root,old['parents'])
    seal=json.loads((folder/'complete.json').read_text())
    if seal.get('config_sha256')!=_hash(old) or not {'fixtures.json','requests.jsonl'} <= set(seal.get('files',{})):
        raise ValueError('Historical fixture capture seal is incomplete.')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Historical fixture capture seal mismatch.')
    return folder/'fixtures.json'


def stage(config,output_root,settings_path,name,operation,reviewer,volume_id,rate,*,execute=False,estimated_hours='0.5',fixture_source=None):
    root,parent,_=source.load(config,output_root)
    safe_name(name)
    if operation not in STAGES or not reviewer.strip() or not volume_id.strip():
        raise ValueError('Explicit qualification stage, reviewer and persistent volume required.')
    budget.number(rate,True);budget.number(estimated_hours,True)
    control=root/parent['run_path']/'qualification'/name/operation
    control.mkdir(parents=True,exist_ok=True)
    def path(s):return root/'plans'/f'{name}-{s}'/'sampling-sft.instrument.json'
    pp=root/'plans'/f'{name}-pilot'/'sampling-sft.pilot.json'
    if operation in ['select','pilot-reference','pilot-kill','pilot-resume','compare']:
        receipt=instrument.compare(path('scores'),root,path('oracle'))
        if not receipt['gpu_qualified']:
            raise ValueError('Real independent score qualification is required before runtime selection/pilot.')
    if operation=='select':
        review=dict(candidates=[str(path('profile').relative_to(root))],
            score_config=str(path('scores').relative_to(root)),oracle_config=str(path('oracle').relative_to(root)),
            reviewer=reviewer,notes='Qualify the fixed supplied scheduler only; no claim of fastest hardware or scheduler sweep.')
        source.frozen(control/'review.json',review)
        return instrument.select(path('profile'),root,control/'review.json')
    if operation=='compare':
        result=pilot.compare(pp,root)
        if not result['gpu_qualified']:
            raise ValueError('Scientific equality alone is insufficient; real kill/restart evidence is missing.')
        return result
    if operation.startswith('pilot-'):
        if not pp.exists():pilot.prepare(path('profile'),root,f'{name}-pilot')
        _,plan,pilot_dir=pilot.load(pp,root)
        if operation in ['pilot-kill','pilot-resume']:
            pilot.verify_lane(pilot_dir/'uninterrupted',plan)
        lane='uninterrupted' if operation=='pilot-reference' else 'resumed'
        if operation=='pilot-resume':
            killed=root/parent['run_path']/'jobs'/f'{name}-pilot-kill'/'execution.json'
            if json.loads(killed.read_text()).get('returncode')!=-9:
                raise ValueError('Restore requires external job evidence of the controlled SIGKILL.')
            marker=pilot_dir/'resumed/writer.json'
            if marker.exists() and execute:
                owner=json.loads(marker.read_text())
                pilot.recover_writer(pp,root,'resumed',owner['token'],
                    'Operator requested qualified restore; external execution receipt records SIGKILL; local PID liveness checked.')
        target=pp
        child=['sampling-sft-pilot-run','--config',str(target),'--lane',lane]
        if operation=='pilot-kill':child+=['--interrupt-after-block','2']
        base=str((pilot_dir/lane).relative_to(root))
        names=['blocks/001/complete.json','writer.json'] if operation=='pilot-kill' else ['complete.json','results.json','responses.jsonl']
    else:
        settings=json.loads(Path(settings_path).read_text())
        settings.pop('fixtures',None)
        settings.update(mode='bounded',k=2,problem_ids=[0,1,2,3])
        if operation=='fixtures':settings.update(mode='fixtures',k=0,
            problem_ids=sorted(random.Random(0).sample(range(512),32))[:20])
        if operation in ['scores','oracle']:
            if fixture_source:
                capture=inside(root,fixture_source)
                fixture_file=historical_fixtures(root,capture,config)
                fixture_plan=json.loads(capture.read_text())
                source.frozen(control/'fixture-source.json',dict(plan=str(capture.relative_to(root)),
                    plan_sha256=file_hash(capture),complete_sha256=file_hash(fixture_file.parent/'complete.json'),
                    fixture_sha256=file_hash(fixture_file)))
            else:
                _,fixture_plan,fixtures_dir=instrument.load(path('fixtures'),root)
                instrument.verify(path('fixtures'),root)
                fixture_file=fixtures_dir/'fixtures.json'
            settings.update(mode=operation,k=0,problem_ids=[p['problem_id'] for p in fixture_plan['problems']],
                fixtures=str(fixture_file.relative_to(root)))
        if operation=='profile':
            instrument.compare(path('scores'),root,path('oracle'))
            settings.update(mode='profile',k=2,problem_ids=list(range(64)))
        source.frozen(control/'settings.json',settings)
        target=instrument.prepare(config,root,control/'settings.json',f'{name}-{operation}')
        plan=json.loads(target.read_text());base=plan['run_path']
        child=['sampling-sft-instrument-run','--config',str(target)]
        names=['complete.json','results.json','env.json']
        if operation not in ['scores','oracle']:
            names+=['responses.jsonl','requests.jsonl','timings.jsonl']
        if operation=='fixtures':names+=['fixtures.json']
    job_id=f'{name}-{operation}'
    job_path=root/parent['run_path']/'jobs'/job_id/'job.plan.json'
    request=dict(job_id=job_id,kind='gpu',stage='sampling',mode='runpod',
        rate_usd_per_hour=str(rate),estimated_hours=str(estimated_hours),setup_hours='0',
        network_volume_id=volume_id,
        command=[sys.executable,'-m','pilot_eval',*child,'--job',str(job_path),'--output-root',str(root)],
        outputs=[base+'/'+filename for filename in names])
    source.frozen(control/'request.json',request)
    job_path=jobs.prepare(config,root,control/'request.json')
    current=budget.initialize(config,root)
    approval_path=control/'approval.json'
    if not approval_path.exists():
        approval=dict(reviewer=reviewer,notes='User approved bounded qualification and withdrew cost-cap blocker. '
            'Historical accounting incomplete. No production generation or training approval.',
            paid_execution_approved=True,preflight_reviewed=True,waive_budget_limits=True,
            budget_sha256=current['sha256'],request_sha256=_hash(request),
            source_audit_sha256=file_hash(root/parent['run_path']/'results/source-audit.json'))
        source.frozen(approval_path,approval)
    if not execute:
        return dict(state='prepared',stage=operation,job=str(job_path),approval=str(approval_path),production_ready=False)
    directory=job_path.parent
    done=threading.Event()
    def follow():
        log=directory/'logs/child.console.log';offset=0
        while True:
            if log.exists():
                with log.open(errors='replace') as stream:
                    stream.seek(offset);chunk=stream.read();offset=stream.tell()
                if chunk:print(chunk,end='',flush=True)
            if done.wait(0.3):break
        if log.exists():
            with log.open(errors='replace') as stream:
                stream.seek(offset);print(stream.read(),end='',flush=True)
    watcher=threading.Thread(target=follow,daemon=True);watcher.start()
    try:result=jobs.run(job_path,root,approval_path)
    finally:done.set();watcher.join(timeout=5)
    execution=json.loads((directory/'execution.json').read_text())
    if operation=='pilot-kill':
        if execution.get('returncode')!=-9:
            raise ValueError('Expected controlled SIGKILL was not recorded; preserve the attempt.')
        return dict(state='controlled-kill-recorded',stage=operation,job=str(job_path),production_ready=False)
    if not result['workload_success']:
        raise ValueError('Qualification workload failed; see '+str(directory/'logs/child.console.log'))
    if operation.startswith('pilot-'):
        pilot.verify_lane(pilot_dir/lane,plan)
    else:
        evidence=instrument.verify(target,root)['result']
        if evidence['execution_kind']!='real-cuda':raise ValueError('Real qualification cannot use CPU fixtures.')
        if operation=='refinement' and not evidence.get('steps'):
            raise ValueError('k2 run did not exercise any refinement decisions.')
    return dict(state='completed',stage=operation,job=str(job_path),production_ready=False,
        next_step='Verify/export the complete qualification tree and independent downloaded copy before Pod stop.')
