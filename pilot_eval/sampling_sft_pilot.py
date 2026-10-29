"""Fixed32 k5 pilot, block-sealed recovery, exact replay and CPU alarm report.

Only completed block directories are scientific progress. Partial directories
move to operational history on recovery. Neither a fixture nor graceful pause
can certify the separate real-CUDA kill/restart gate.
"""
import json
import os
import platform
import random
import signal
import statistics
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sampler import sample, diagnostic
from pilot_eval.sampling_sft_sources import inside, observe, verify_parents
from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v3
from pilot_eval.sft import load_sft, safe_name
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash


LANES=['uninterrupted','resumed']


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
        ['sampling_sft_pilot.py','sampling_sft_sampler.py','sampling_sft_engine.py',
         'sampling_sft_instrument.py','scoring.py']}


def prepare(instrument_path,output_root,name):
    root,parent,directory=instrument.load(instrument_path,output_root)
    safe_name(name)
    audit=source.verify(root/parent['source_plan'],root)['audit']
    _,rows,_=load_sft(root/audit['training']['config_path'],root)
    from pilot_eval.protocol import gsm8k_messages
    pids=sorted(random.Random(0).sample(range(512),32))
    problems=[dict(problem_id=i,source_id=rows[i]['id'],gold=rows[i]['gold'],
        user_message=gsm8k_messages(rows[i]['question'])[0]['content'],
        scorer_prompt_ids=rows[i]['input_ids'][:rows[i]['prompt_tokens']]) for i in pids]
    parents={str(inside(root,instrument_path).relative_to(root)):file_hash(inside(root,instrument_path))}
    selection=observe(root,directory/'runtime-selection.json',parents)
    selected=json.loads(selection.read_text()) if selection.exists() else None
    if selected:
        verify_parents(root,selected['parents'])
        if selected['runtime_hash']!=parent['runtime_hash'] or selected['selected']!=str(inside(root,instrument_path).relative_to(root)):
            raise ValueError('Use the selected Stage0.5 candidate as the pilot parent.')
    folder=root/json.loads((root/parent['source_plan']).read_text())['run_path']/'pilot'/name
    settings={**parent['settings'],'mode':'bounded','k':5,'problem_ids':pids}
    settings.pop('fixtures',None)
    plan=dict(protocol='sampling-sft-pilot-v1',name=name,source_plan=parent['source_plan'],
        instrument_plan=str(inside(root,instrument_path).relative_to(root)),settings=settings,
        runtime_hash=parent['runtime_hash'],problems=problems,pilot_subset_seed=0,
        selection=str(selection.relative_to(root)) if selected else None,
        parents=parents,implementation=implementation(),run_path=str(folder.relative_to(root)),
        production_allowed=False)
    path=root/'plans'/name/'sampling-sft.pilot.json'
    source.frozen(path,plan);source.frozen(folder/'plan-seal.json',dict(config_sha256=_hash(plan)))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path)
    plan=json.loads(path.read_text());folder=inside(root,plan['run_path'])
    instrument.load(root/plan['instrument_plan'],root)
    if (plan.get('protocol')!='sampling-sft-pilot-v1' or plan['implementation']!=implementation()
            or _hash(plan['settings']['runtime'])!=plan['runtime_hash']
            or json.loads((folder/'plan-seal.json').read_text())!={'config_sha256':_hash(plan)}):
        raise ValueError('Pilot runtime/implementation/plan changed; use a new attempt.')
    verify_parents(root,plan['parents'])
    return root,plan,folder


def event(folder,kind,**data):
    instrument.append(folder/'events.jsonl',dict(event=kind,at=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),pid=os.getpid(),**diagnostic(data)))


def lane_directory(folder,lane):
    if lane not in LANES:raise ValueError('Unknown pilot lane.')
    return folder/lane


def blocks(folder,plan):
    """Verify the contiguous chain before reading a resume boundary."""
    products=sorted((folder/'blocks').glob('*'))
    previous=None;values=[]
    for index,path in enumerate(products):
        if path.name!=f'{index:03d}' or not path.is_dir():raise ValueError('Pilot block chain is not contiguous.')
        seal=json.loads((path/'complete.json').read_text())
        if (seal['config_sha256']!=_hash(plan) or seal['runtime_hash']!=plan['runtime_hash']
                or seal['previous']!=previous or seal['block']!=index):
            raise ValueError('Pilot sealed block identity changed.')
        env=folder/f'env.{seal["attempt"]}.json'
        if file_hash(env)!=seal['env_sha256']:
            raise ValueError('Pilot worker environment evidence changed.')
        for name,digest in seal['files'].items():
            if file_hash(inside(path,name))!=digest:raise ValueError('Pilot sealed scientific block changed.')
        state=json.loads((path/'state.json').read_text())
        if state['next_block']!=index+1:raise ValueError('Pilot state boundary changed.')
        values.append((path,state));previous=file_hash(path/'complete.json')
    return values


def scientific(folder,plan):
    rows=[];verified=blocks(folder,plan)
    for directory,_ in verified:
        rows.extend(json.loads(line) for line in (directory/'scientific.jsonl').read_text().splitlines())
    return rows


def run(path,output_root,lane,job_path=None,*,stop_after_blocks=None,interrupt_after_block=None,dependencies=None):
    root,plan,parent=load(path,output_root);folder=lane_directory(parent,lane)
    if (folder/'complete.json').exists():return verify_lane(folder,plan)
    existing=blocks(folder,plan)
    if existing and all(s['end_state'] is not None for s in existing[-1][1]['states'].values()):
        with _writer(folder):finalize(folder,plan,blocks(folder,plan))
        return verify_lane(folder,plan)
    if stop_after_blocks is not None and (type(stop_after_blocks) is not int or stop_after_blocks<=0):
        raise ValueError('Stop-after-blocks must be a positive completed-block count.')
    if interrupt_after_block is not None and (lane!='resumed' or interrupt_after_block!=2):
        raise ValueError('The qualification interruption is fixed at completed block2 of the resumed lane.')
    if dependencies is None:
        if not plan['selection']:raise ValueError('Real pilot requires the measured Stage0.5 runtime selection first.')
        # Recompute/verify the parent's approved choice and its real score evidence.
        selection=json.loads((root/plan['selection']).read_text())
        review_path=parent/'inputs/runtime-review.json';source.frozen(review_path,selection['review'])
        instrument.select(root/plan['instrument_plan'],root,review_path)
        instrument.authorize(root,path,job_path,expected_command='sampling-sft-pilot-run')
    with _writer(folder):
        verified=blocks(folder,plan)
        if (folder/'terminal-error.json').exists():raise ValueError('Engine failure blocks this pilot attempt; inspect and create a new plan.')
        resume=verified[-1][1] if verified else None
        if (folder/'partial').exists():
            abandoned=folder/'abandoned'/uuid.uuid4().hex;abandoned.parent.mkdir(parents=True,exist_ok=True)
            (folder/'partial').rename(abandoned)
            event(folder,'partial-science-discarded',resume_block=len(verified),retained_as=str(abandoned.relative_to(folder)))
        attempt=uuid.uuid4().hex
        event(folder,'started',attempt=attempt,resume_block=len(verified),job=str(inside(root,job_path).relative_to(root)) if job_path else None)
        _write_state(folder,'loading-engine',sum(s['end_state'] is not None for s in resume['states'].values()) if resume else 0,32)
        started=time.monotonic()
        try:
            if dependencies is None:
                from pilot_eval.sampling_sft_engine import open_engine
                engine,tokenizer,runtime=open_engine(plan,root)
                for problem in plan['problems']:
                    ids=tokenizer.apply_chat_template([dict(role='user',content=problem['user_message'])],
                        tokenize=True,add_generation_prompt=True)
                    if list(ids)!=problem['scorer_prompt_ids']:raise ValueError('Pilot scorer retokenization changed.')
            else:
                engine,tokenizer,runtime=dependencies.open(plan,root)
                runtime={**runtime,'execution_kind':'cpu-fixture'}
            source.save(folder/f'env.{attempt}.json',runtime)
            engine.clear_cache() # identical cold-cache start for uninterrupted and resumed work
            if hasattr(engine,'measurement_start'):engine.measurement_start()
            partial=folder/'partial'

            def record(value):
                if 'engine_error' in value:
                    event(folder,'engine-replay',attempt=attempt,details=value['engine_error']);return
                partial.mkdir(exist_ok=True)
                # Cache counters are operational and may differ across engine launches.
                value=json.loads(json.dumps(value))
                value.pop('replay',None)
                for output in value.get('outputs',[]):output.pop('num_cached_tokens',None)
                instrument.append(partial/'scientific.jsonl',value)
                if dependencies is not None and hasattr(dependencies,'after_record'):dependencies.after_record(value)

            def timing(value):
                partial.mkdir(exist_ok=True);instrument.append(partial/'timings.jsonl',value)

            def checkpoint(block,states,rng):
                partial.mkdir(exist_ok=True)
                saved={pid:{**row,**(dict(text=tokenizer.decode(row['response_ids'],skip_special_tokens=True))
                        if row['end_state'] is not None else {})} for pid,row in states.items()}
                source.save(partial/'state.json',dict(next_block=block+1,states=saved,rng=rng))
                previous=verified[-1][0]/'complete.json' if verified else None
                seal=dict(block=block,config_sha256=_hash(plan),runtime_hash=plan['runtime_hash'],
                    attempt=attempt,execution_kind=runtime['execution_kind'],
                    env_sha256=file_hash(folder/f'env.{attempt}.json'),
                    previous=file_hash(previous) if previous else None,
                    files={p.name:file_hash(p) for p in partial.iterdir() if p.is_file()})
                source.save(partial/'complete.json',seal)
                target=folder/'blocks'/f'{block:03d}';target.parent.mkdir(exist_ok=True)
                if target.exists():raise ValueError('Refuse to overwrite a sealed block.')
                partial.rename(target);verified.append((target,dict(next_block=block+1,states=saved,rng=rng)))
                count=sum(s['end_state'] is not None for s in states.values())
                _write_state(folder,'running',count,32)
                event(folder,'block-sealed',attempt=attempt,block=block+1,completed=count,
                    seal_sha256=file_hash(target/'complete.json'))
                print(f'Pilot {lane}: {count}/32 terminal; {block+1}/32 blocks sealed.',flush=True)
                if dependencies is not None and hasattr(dependencies,'after_checkpoint'):dependencies.after_checkpoint(block)
                if interrupt_after_block==block+1:
                    event(folder,'qualification-kill-after-block2',attempt=attempt,job=str(inside(root,job_path).relative_to(root)) if job_path else None)
                    os.kill(os.getpid(),signal.SIGKILL)

            result=sample(plan['problems'],5,engine,tokenizer,plan['settings']['runtime']['actual_stop_ids'],
                resume=resume,stop_after_blocks=stop_after_blocks,on_record=record,on_timing=timing,on_block=checkpoint)
            elapsed=time.monotonic()-started
            memory=engine.measurement_end() if hasattr(engine,'measurement_end') else None
            event(folder,'attempt-ended',attempt=attempt,state=result['state'],seconds=elapsed,memory=memory)
            if result['state']=='engine-error':
                source.save(folder/'terminal-error.json',result)
                raise ValueError('Pilot engine replay exhausted; diagnostics preserved, no completed dataset.')
            if result['state']=='paused':
                _write_state(folder,'paused',sum(s['end_state'] is not None for s in result['traces']),32)
                return dict(state='paused',sealed_blocks=len(verified),production_allowed=False)
            finalize(folder,plan,verified)
        except BaseException as exc:
            event(folder,'interrupted',attempt=attempt,error=str(exc),type=type(exc).__name__)
            _write_state(folder,'failed',sum(s['end_state'] is not None for s in verified[-1][1]['states'].values()) if verified else 0,32,str(exc))
            raise
    return verify_lane(folder,plan)



def finalize(folder,plan,verified):
    traces=sorted((dict(r) for r in verified[-1][1]['states'].values()),key=lambda row:row['problem_id'])
    by_pid={p['problem_id']:p for p in plan['problems']}
    for trace in traces:
        text=trace['text']
        problem=by_pid[trace['problem_id']]
        trace.update(text=text,automatic_flexible_v3=score_gsm8k_flexible_v3(text,problem['gold'],
            capped=trace['end_state']!='finished'),automatic_strict=score_gsm8k(text,problem['gold'],
            capped=trace['end_state']!='finished')['strict'])
    source.save(folder/'results.json',dict(state='completed',traces=traces,
        execution_kind=json.loads((verified[-1][0]/'complete.json').read_text())['execution_kind'],runtime_hash=plan['runtime_hash'],
        sealed_blocks=len(verified),production_allowed=False))
    event(folder,'finalized-from-sealed-blocks',sealed_blocks=len(verified))
    responses=folder/'responses.jsonl'
    source.save(folder/'responses.json',traces)
    responses.write_text(''.join(json.dumps(row)+'\n' for row in traces))
    _write_state(folder,'completed',32,32)
    source.save(folder/'complete.json',dict(config_sha256=_hash(plan),
        last_block=file_hash(verified[-1][0]/'complete.json'),
        files={name:file_hash(folder/name) for name in ['results.json','responses.json','responses.jsonl','events.jsonl']}))

def verify_lane(folder,plan):
    verified=blocks(folder,plan);marker=folder/'complete.json'
    if not marker.exists():return dict(state='partial',sealed_blocks=len(verified),production_allowed=False)
    seal=json.loads(marker.read_text())
    if seal['config_sha256']!=_hash(plan) or not verified or seal['last_block']!=file_hash(verified[-1][0]/'complete.json'):
        raise ValueError('Pilot completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Pilot final result changed.')
    result=json.loads((folder/'results.json').read_text())
    if len(result['traces'])!=32 or any(r['end_state'] not in ['finished','capped','block_budget_exhausted'] for r in result['traces']):
        raise ValueError('Pilot final dispositions incomplete.')
    return result


def compare(path,output_root):
    root,plan,folder=load(path,output_root)
    left,right=[verify_lane(folder/lane,plan) for lane in LANES]
    if any(r['state']!='completed' for r in [left,right]):raise ValueError('Both pilot lanes must finish before replay comparison.')
    a,b=[scientific(folder/lane,plan) for lane in LANES]
    equal=a==b and left['traces']==right['traces']
    records=[json.loads(line) for line in (folder/'resumed/events.jsonl').read_text().splitlines()]
    kill=next((r for r in records if r['event']=='qualification-kill-after-block2'),None)
    restarted=any(r['event']=='started' and r['resume_block']==2 and (not kill or r['attempt']!=kill['attempt']) for r in records)
    external_kill=False
    if kill and kill.get('job'):
        from pilot_eval.sampling_sft_jobs import _load
        _,_,job_folder=_load(root/kill['job'],root)
        receipt=json.loads((job_folder/'execution.json').read_text())
        external_kill=receipt.get('returncode')==-signal.SIGKILL
    real=all(r['execution_kind']=='real-cuda' for r in [left,right]) and all(
        json.loads((directory/'complete.json').read_text())['execution_kind']=='real-cuda'
        for lane in LANES for directory,_ in blocks(folder/lane,plan))
    result=dict(scientific_equal=equal,compared_blocks=left['sealed_blocks'],
        first_difference=first_difference(dict(scientific=a,traces=left['traces']),dict(scientific=b,traces=right['traces'])),
        uninterrupted_records=len(a),resumed_records=len(b),
        kill_after_block2_recorded=bool(kill),restart_from_block2=restarted,external_sigkill_confirmed=external_kill,
        gpu_qualified=real and equal and restarted and external_kill,production_allowed=False,
        parents={str((folder/lane/'complete.json').relative_to(root)):file_hash(folder/lane/'complete.json') for lane in LANES},
        requalification_order=[] if equal else ['disable prefix caching','constrain scheduler/batch splitting',
            'enable supported deterministic modes'],
        scope='exact scientific replay; every runtime change requires a new plan and measured qualification')
    source.frozen(folder/'replay.json',result)
    if not equal:raise ValueError('Exact pilot replay failed; Stage2 blocked, no tolerance relaxation.')
    return result


def first_difference(left,right,path=''):
    if type(left) is not type(right):return dict(path=path,left_type=type(left).__name__,right_type=type(right).__name__)
    if isinstance(left,dict):
        if set(left)!=set(right):return dict(path=path,left_keys=sorted(left),right_keys=sorted(right))
        for key in sorted(left):
            difference=first_difference(left[key],right[key],path+'.'+str(key))
            if difference:return difference
    elif isinstance(left,list):
        if len(left)!=len(right):return dict(path=path,left_length=len(left),right_length=len(right))
        for i,(a,b) in enumerate(zip(left,right)):
            difference=first_difference(a,b,f'{path}[{i}]')
            if difference:return difference
    elif left!=right:return dict(path=path,left=left,right=right)
    return None


def status(path,output_root):
    _,plan,folder=load(path,output_root)
    return dict(runtime_hash=plan['runtime_hash'],lanes={lane:verify_lane(folder/lane,plan) for lane in LANES},production_allowed=False)


def recover_writer(path,output_root,lane,owner_token,review_notes):
    _,_,parent=load(path,output_root);folder=lane_directory(parent,lane)
    if not owner_token or not review_notes.strip():raise ValueError('Recovery requires recorded token and review notes.')
    with run_lock(folder):
        marker=folder/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Pilot writer owner token mismatch.')
        if owner.get('host')==platform.node():
            pid=owner.get('pid')
            if type(pid) is not int or pid<=0:raise ValueError('Invalid writer PID.')
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise ValueError('Writer process is still active.')
        event(folder,'ownership-released',owner=owner,review_notes=review_notes);marker.unlink()
    return dict(state='ownership-released')


def report(path,output_root,forecast_path):
    root,plan,folder=load(path,output_root)
    replay=compare(path,root)
    result=verify_lane(folder/'uninterrupted',plan);traces=result['traces']
    lengths=sorted(len(r['response_ids']) for r in traces)
    correct=sum(r['automatic_flexible_v3']['correct'] for r in traces)
    unfinished=sum(r['end_state'] in ['capped','block_budget_exhausted'] for r in traces)
    forecast=json.loads(Path(forecast_path).read_text())
    from pilot_eval.sampling_sft_budget import initialize, status as budget_status, number, micros,usd
    if (not isinstance(forecast,dict) or not {'rate_usd_per_hour','estimated_hours','setup_hours','notes'}<=set(forecast)
            or not isinstance(forecast.get('notes'),str) or not forecast['notes'].strip()):
        raise ValueError('Forecast needs quoted rate, estimated_hours, setup_hours and measured-basis notes.')
    rate=number(forecast['rate_usd_per_hour'],True);hours=number(forecast['estimated_hours'],True)
    setup=number(forecast['setup_hours'])
    reservation=micros(rate*(hours*Decimal('1.5')+setup))
    initialize(root/plan['source_plan'],root);ledger=budget_status(root)
    headroom=min(Decimal('10')-Decimal(ledger['stage_totals']['sampling']['committed_usd']),
                 Decimal('25')-Decimal(ledger['committed_usd']))
    timings=[]
    for directory,_ in blocks(folder/'uninterrupted',plan):
        timings.extend(json.loads(line) for line in (directory/'timings.jsonl').read_text().splitlines())
    seconds=sum(t['seconds'] for t in timings)
    generated=sum(t['generated_tokens'] for t in timings)
    decisions=[r['decision'] for r in scientific(folder/'uninterrupted',plan) if 'decision' in r]
    stats=length_statistics(lengths)
    events=[json.loads(line) for line in (folder/'uninterrupted/events.jsonl').read_text().splitlines()]
    alarms=dict(unfinished=dict(value=unfinished/32,numerator=unfinished,denominator=32,threshold=0.02,
                relation='>',triggered=unfinished/32>0.02),
        initial_correctness=dict(value=correct/32,numerator=correct,denominator=32,threshold=0.9,
                relation='<',triggered=correct/32<0.9),
        budget=dict(reservation_usd=usd(reservation),headroom_usd=str(headroom),triggered=Decimal(usd(reservation))>headroom))
    report=dict(n_problems=32,k=5,initial_correct=correct,initial_unfinished=unfinished,
        strict_correct=sum(r['automatic_strict']['correct'] for r in traces),
        strict_format_valid=sum(r['automatic_strict']['status']=='valid' for r in traces),
        length_median=stats['median'],length_p90=stats['p90'],
        quantile_method='linear interpolation, index=(n-1)*p',
        acceptance_rate=sum(r['accepted'] for r in decisions)/len(decisions) if decisions else None,
        acceptance_by_step={str(i):dict(accepted=sum(d['accepted'] for d in decisions if d['step']==i),
            attempts=sum(d['step']==i for d in decisions)) for i in range(5)},
        acceptance_by_cut_decile={str(i):dict(accepted=sum(d['accepted'] for d in decisions
            if min(9,int(d['cut']/len(d['old_ids'])*10))==i),attempts=sum(
            min(9,int(d['cut']/len(d['old_ids'])*10))==i for d in decisions)) for i in range(10)},
        accepted_mean_score_gain_per_generated_token=sum(d['proposal_score']-d['old_score'] for d in decisions if d['accepted'])/generated if generated else None,
        engine_seconds=seconds,generated_tokens=generated,generated_tokens_per_engine_wall_second=generated/seconds if seconds else None,
        uncached_prefill_tokens=sum(t['input_tokens']-t['cached_tokens'] for t in timings),
        attempt_wall_seconds=[e['seconds'] for e in events if e['event']=='attempt-ended'],
        peak_gpu_memory=[e['memory'] for e in events if e['event']=='attempt-ended'],
        cold_cache_policy='clear at every completed block and process start; no un-cleared counterfactual claimed',
        forecast=forecast,forecast_sha256=file_hash(forecast_path),forecast_scope='explicit measured-basis estimate, not a reservation or proof of an upper bound',
        alarms=alarms,replay=replay,execution_kind=result['execution_kind'],
        ready_for_generation_review=replay['gpu_qualified'] and not any(a['triggered'] for a in alarms.values()),
        production_allowed=False,budget_snapshot_sha256=ledger['sha256'],runtime_hash=plan['runtime_hash'])
    source.save(folder/'report.json',report)
    return report


def length_statistics(values):
    return dict(median=statistics.median(values),p90=statistics.quantiles(values,n=10,method='inclusive')[8])


def console(value):
    if 'lanes' in value:
        return dict(runtime_hash=value['runtime_hash'],lanes={lane:dict(state=r['state'],
            sealed_blocks=r.get('sealed_blocks',0),terminal=len(r.get('traces',[]))) for lane,r in value['lanes'].items()},
            production_allowed=False)
    keys=['state','execution_kind','runtime_hash','sealed_blocks','scientific_equal','compared_blocks',
          'kill_after_block2_recorded','restart_from_block2','external_sigkill_confirmed','gpu_qualified',
          'n_problems','initial_correct','initial_unfinished','length_median','length_p90','alarms',
          'ready_for_generation_review','production_allowed']
    return {key:value[key] for key in keys if key in value}
