"""Atomic experiment-wide USD ledger. Reservations are not permission to spend."""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from pilot_eval.sampling_sft import save, frozen, load
from pilot_eval.sft import safe_name
from pilot_eval.training import run_lock
from pilot_eval.workflow import _hash

SCALE = 1000000
CAPS = {'sampling':10000000, 'training':6000000, 'storage_shared':9000000}
POLICY = dict(total_microusd=25000000, stage_caps_microusd=CAPS,
              review_fraction='0.8', gpu_buffer='1.5', currency='USD')


def number(value, positive=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a cost/time estimate.')
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('Invalid numeric cost/time estimate.') from exc
    if not amount.is_finite() or amount < 0 or (positive and amount <= 0):
        raise ValueError('Cost/time must be finite and nonnegative (rates/durations positive).')
    return amount


def micros(value):
    return int((number(value)*SCALE).to_integral_value(rounding=ROUND_CEILING))


def usd(value):
    return f'{Decimal(value)/SCALE:.6f}'


def folder(root):
    return Path(root).resolve()/'runs/sampling-sft/budget'


def _read(root):
    directory=folder(root)
    origin=json.loads((directory/'origin.json').read_text())
    value=json.loads((directory/'budget.json').read_text())
    payload={k:v for k,v in value.items() if k!='sha256'}
    if (value.get('sha256')!=_hash(payload) or value.get('policy')!=POLICY
            or value.get('origin')!=origin or value.get('schema_version')!=1):
        raise ValueError('Sampling SFT budget integrity/policy mismatch.')
    previous=None
    for i,event in enumerate(value['events']):
        row={k:v for k,v in event.items() if k!='sha256'}
        if event.get('sequence')!=i or event.get('previous_sha256')!=previous or event.get('sha256')!=_hash(row):
            raise ValueError('Sampling SFT budget event chain mismatch.')
        previous=event['sha256']
    _jobs(value) # Validate the transaction state machine before accepting a ledger.
    return value


def _write(root,value,event=None):
    if event is not None:
        row={**event,'at':datetime.now(timezone.utc).isoformat(),'sequence':len(value['events']),
             'previous_sha256':value['events'][-1]['sha256'] if value['events'] else None}
        row['sha256']=_hash(row)
        value={**value,'events':[*value['events'],row]}
    value={k:v for k,v in value.items() if k!='sha256'}
    value['sha256']=_hash(value)
    save(folder(root)/'budget.json',value)
    return _summary(value)


def initialize(config,output_root):
    load(config,output_root) # Bind this workflow to a verified source plan, not a new cap.
    directory=folder(output_root)
    with run_lock(directory):
        if (directory/'budget.json').exists():
            return _summary(_read(output_root))
        if (directory/'origin.json').exists():
            raise ValueError('Budget origin exists but ledger is missing; never reset spending.')
        origin=dict(id=uuid.uuid4().hex,policy_sha256=_hash(POLICY))
        frozen(directory/'origin.json',origin)
        return _write(output_root,dict(schema_version=1,origin=origin,policy=POLICY,events=[]))


def _jobs(value):
    jobs={}
    for event in value['events']:
        kind=event.get('kind');job_id=event.get('job_id')
        if kind=='reserve':
            if (job_id in jobs or event['stage'] not in CAPS or type(event['reserved_microusd']) is not int
                    or event['reserved_microusd']!=estimate(event['request'])
                    or event['request_sha256']!=_hash(event['request']) or event['stage']!=event['request']['stage']):
                raise ValueError('Invalid duplicate/stage/amount reservation event.')
            jobs[job_id]=dict(event,state='reserved',actual_microusd=0)
            if event['request']['kind']=='storage':
                jobs[job_id]['remaining_months']=str(number(event['request']['months'],True))
        elif kind=='extend':
            if job_id not in jobs or jobs[job_id]['request']['kind']!='storage' or type(event['additional_microusd']) is not int or event['additional_microusd']<=0:
                raise ValueError('Invalid storage extension event.')
            jobs[job_id]['reserved_microusd']+=event['additional_microusd']
            jobs[job_id]['remaining_months']=str(number(jobs[job_id]['remaining_months'])+number(event['request']['additional_months'],True))
            jobs[job_id]['state']='reserved'
            jobs[job_id].setdefault('extensions',[]).append(event['request_sha256'])
        elif kind=='launch':
            if job_id not in jobs or jobs[job_id]['state']!='reserved' or jobs[job_id].get('launched'):
                raise ValueError('Invalid or repeated launch transaction.')
            jobs[job_id]['launched']=True
        elif kind=='settle':
            if (job_id not in jobs or jobs[job_id]['state']!='reserved' or type(event['actual_microusd']) is not int
                    or event['actual_microusd']!=micros(event['observation']['actual_usd'])
                    or event['observation_sha256']!=_hash(event['observation'])):
                raise ValueError('Invalid settlement event.')
            job=jobs[job_id]
            if job['request']['kind']=='storage':
                covered=number(event['observation']['covered_months'],True)
                remaining=number(job['remaining_months'])-covered
                if remaining<0:raise ValueError('Storage bill covers more than the reserved lifetime.')
                job.update(remaining_months=str(remaining),
                    reserved_microusd=micros(number(job['request']['rate_usd_per_month'])*remaining),
                    actual_microusd=job['actual_microusd']+event['actual_microusd'],
                    state='reserved' if remaining else 'settled',observation_sha256=event['observation_sha256'])
            else:
                job.update(state='settled',actual_microusd=event['actual_microusd'],reserved_microusd=0,observation_sha256=event['observation_sha256'])
        else:
            raise ValueError('Unknown budget transaction.')
    return jobs


def _summary(value):
    jobs=_jobs(value)
    stages={name:dict(spent=0,reserved=0) for name in CAPS}
    for job in jobs.values():
        stages[job['stage']]['spent']+=job['actual_microusd']
        if job['state']=='reserved':stages[job['stage']]['reserved']+=job['reserved_microusd']
    spent=sum(s['spent'] for s in stages.values());reserved=sum(s['reserved'] for s in stages.values())
    return dict(schema_version=1,sha256=value['sha256'],spent_usd=usd(spent),reserved_usd=usd(reserved),
        committed_usd=usd(spent+reserved),cap_exceeded=spent+reserved>POLICY['total_microusd'] or any(sum(s.values())>CAPS[k] for k,s in stages.items()),
        review_required=(spent+reserved)*5>=POLICY['total_microusd']*4 or any(sum(s.values())*5>=CAPS[k]*4 for k,s in stages.items()),
        stage_totals={name:dict(spent_usd=usd(s['spent']),reserved_usd=usd(s['reserved']),committed_usd=usd(sum(s.values())),cap_usd=usd(CAPS[name])) for name,s in stages.items()},
        jobs={name:dict(state=job['state'],stage=job['stage'],kind=job['request']['kind'],request_sha256=job['request_sha256'],
                       reserved_usd=usd(job['reserved_microusd']),actual_usd=usd(job['actual_microusd']) if job['state']=='settled' or job['actual_microusd'] else None) for name,job in jobs.items()})


def status(output_root):
    with run_lock(folder(output_root)):
        return _summary(_read(output_root))


def estimate(request):
    if not isinstance(request,dict):raise ValueError('Budget request must be an object.')
    safe_name(request.get('job_id',''))
    kind=request.get('kind');stage=request.get('stage')
    if stage not in CAPS:raise ValueError('Unknown budget stage.')
    if kind=='noop':
        if request.get('mode')!='noop' or request.get('command')!=[]:
            raise ValueError('Noop reservations cannot run a command or provider.')
        return 0
    if kind=='gpu':
        if stage not in ['sampling','training']:raise ValueError('GPU work/reruns must use the served stage, not shared overhead.')
        rate=number(request['rate_usd_per_hour'],True)
        hours=number(request['estimated_hours'],True)
        setup=number(request['setup_hours'])
        return micros(rate*(hours*Decimal('1.5')+setup))
    if kind=='storage':
        if stage!='storage_shared' or not request.get('volume_id'):raise ValueError('Storage needs a named volume and storage/shared stage.')
        return micros(number(request['rate_usd_per_month'],True)*number(request['months'],True))
    if kind=='shared-overhead':
        if stage!='storage_shared' or not request.get('notes'):raise ValueError('Shared overhead needs explicit allocation notes.')
        return micros(request['upper_bound_usd'])
    raise ValueError('Unknown budget request kind.')


def _check(value,stage,additional,request,approval):
    current=_summary(value)
    committed=micros(current['committed_usd']);stage_committed=micros(current['stage_totals'][stage]['committed_usd'])
    if current['cap_exceeded'] or committed+additional>POLICY['total_microusd'] or stage_committed+additional>CAPS[stage]:
        raise ValueError('Reservation exceeds a hard stage or total budget cap.')
    review=current['review_required'] or (committed+additional)*5>=POLICY['total_microusd']*4 or (stage_committed+additional)*5>=CAPS[stage]*4
    if review:
        if (not isinstance(approval,dict) or approval.get('budget_sha256')!=value['sha256']
                or approval.get('request_sha256')!=_hash(request) or approval.get('review_80_percent') is not True
                or not all(isinstance(approval.get(k),str) and approval[k].strip() for k in ['reviewer','notes'])):
            raise ValueError('80% budget checkpoint: explicit review bound to the current ledger and request is required.')


def reserve(config,output_root,request_path,approval_path=None):
    _,plan,_=load(config,output_root)
    request=json.loads(Path(request_path).read_text());amount=estimate(request)
    approval=json.loads(Path(approval_path).read_text()) if approval_path else None
    with run_lock(folder(output_root)):
        value=_read(output_root);jobs=_jobs(value)
        if request['job_id'] in jobs:
            old=jobs[request['job_id']]
            if old['request_sha256']!=_hash(request) or old['plan_sha256']!=_hash(plan):
                raise ValueError('Reserved job identity/estimate changed; preserve it and use another job ID.')
            return _summary(value)
        _check(value,request['stage'],amount,request,approval)
        return _write(output_root,value,dict(kind='reserve',job_id=request['job_id'],stage=request['stage'],
            request=request,request_sha256=_hash(request),plan_sha256=_hash(plan),reserved_microusd=amount,
            approval=approval,permission_to_launch=False))


def settle(output_root,job_id,observation_path):
    observation=json.loads(Path(observation_path).read_text())
    amount=micros(observation['actual_usd'])
    if not all(isinstance(observation.get(k),str) and observation[k].strip() for k in ['event_id','evidence','reviewer','notes']):
        raise ValueError('Actual cost needs an explicit evidenced observation; never fabricate a bill.')
    with run_lock(folder(output_root)):
        value=_read(output_root);jobs=_jobs(value)
        if job_id not in jobs:raise ValueError('Unknown reservation.')
        existing=[e for e in value['events'] if e.get('event_id')==observation['event_id']]
        if existing:
            if existing[0]['job_id']!=job_id or existing[0]['observation_sha256']!=_hash(observation):
                raise ValueError('Billing event changed or was reused by another job.')
            return _summary(value)
        if jobs[job_id]['state']=='settled':
            raise ValueError('Reservation already settled; preserve the recorded bill.')
        if jobs[job_id]['request']['kind']=='storage':
            covered=number(observation.get('covered_months'),True)
            if covered>number(jobs[job_id]['remaining_months']):
                raise ValueError('Storage bill must name an unbilled portion of the reserved lifetime.')
        # Real overruns must be recorded, not rejected to make the ledger look under cap.
        return _write(output_root,value,dict(kind='settle',job_id=job_id,actual_microusd=amount,
            event_id=observation['event_id'],observation=observation,observation_sha256=_hash(observation)))


def extend(output_root,job_id,request_path,approval_path=None):
    request=json.loads(Path(request_path).read_text())
    months=number(request['additional_months'],True)
    approval=json.loads(Path(approval_path).read_text()) if approval_path else None
    with run_lock(folder(output_root)):
        value=_read(output_root);jobs=_jobs(value)
        job=jobs.get(job_id)
        if not job or job['request']['kind']!='storage':raise ValueError('Only storage lifetime reservations can be extended.')
        if _hash(request) in job.get('extensions',[]):return _summary(value)
        amount=micros(number(job['request']['rate_usd_per_month'],True)*months)
        _check(value,'storage_shared',amount,request,approval)
        return _write(output_root,value,dict(kind='extend',job_id=job_id,additional_microusd=amount,
            request=request,request_sha256=_hash(request),approval=approval))


def authorize_launch(output_root,job_id,request,approval=None):
    """One launch authorization, atomically recorded with current budget review."""
    with run_lock(folder(output_root)):
        value=_read(output_root);job=_jobs(value).get(job_id)
        if not job or job['state']!='reserved' or job['request_sha256']!=_hash(request):
            raise ValueError('Job needs its matching outstanding reservation before launch.')
        if job.get('launched'):
            raise ValueError('Launch already authorized; reconcile interrupted work rather than relaunch.')
        last=value['events'][-1]
        reviewed_own_reservation=(last['kind']=='reserve' and last['job_id']==job_id
                                 and bool(last.get('approval',{}))
                                 and last['approval'].get('review_80_percent') is True)
        if not reviewed_own_reservation:
            _check(value,job['stage'],0,request,approval)
        return _write(output_root,value,dict(kind='launch',job_id=job_id,
            request_sha256=_hash(request),review=approval))
