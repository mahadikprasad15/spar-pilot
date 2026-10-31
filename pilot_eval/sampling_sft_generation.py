"""Three full sampling arms, resumable attempt rounds and automatic dispositions.

Capture eligibility is provisional flexible-v3 correctness. Ticket06 can add
separate human decisions; this module never edits text or authorizes training.
"""
import json
import os
import platform
import uuid
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_budget as budget
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval import sampling_sft_pilot as pilot
from pilot_eval.protocol import gsm8k_messages
from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sampler import sample
from pilot_eval.sampling_sft_sources import inside, observe, verify_parents
from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v3
from pilot_eval.sft import load_sft, safe_name
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash


ARMS=(0,2,5)
END_STATES=('finished','capped','block_budget_exhausted','engine_error')


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
        ['sampling_sft_generation.py','sampling_sft_pilot.py','sampling_sft_sampler.py',
         'sampling_sft_engine.py','sampling_sft_ieee_diagnostic.py','scoring.py']}


def retry_seed(index):
    if type(index) is not int or not 0<=index<=3:raise ValueError('Initial attempt plus at most3 model retries.')
    return 0 if index==0 else 1000+index


def snapshot(root):
    value=budget.status(root)
    return {k:value[k] for k in ['sha256','spent_usd','reserved_usd','committed_usd','review_required','cap_exceeded']}


def prepare(pilot_path,output_root,name):
    root,parent,directory=pilot.load(pilot_path,output_root);safe_name(name)
    audit=source.verify(root/parent['source_plan'],root)['audit']
    if not audit['source_ready']:raise ValueError('Verified source preflight required.')
    _,rows,_=load_sft(root/audit['training']['config_path'],root)
    problems=[dict(problem_id=i,source_id=row['id'],gold=row['gold'],
        user_message=gsm8k_messages(row['question'])[0]['content'],
        scorer_prompt_ids=row['input_ids'][:row['prompt_tokens']],
        gold_end_token_id=audit['training']['prompt_mask_contract']['eos_token_id']) for i,row in enumerate(rows)]
    parents={str(inside(root,pilot_path).relative_to(root)):file_hash(inside(root,pilot_path))}
    report=observe(root,directory/'report.json',parents)
    settings={**parent['settings'],'mode':'generation','problem_ids':list(range(512))}
    folder=root/json.loads((root/parent['source_plan']).read_text())['run_path']/'generation'/name
    plan=dict(protocol='sampling-sft-generation-v1',name=name,source_plan=parent['source_plan'],
        pilot_plan=str(inside(root,pilot_path).relative_to(root)),pilot_report=str(report.relative_to(root)),
        settings=settings,problems=problems,arms=list(ARMS),data_seed=0,
        retry_indices=[0,1,2,3],retry_data_seeds=[retry_seed(i) for i in range(4)],
        algorithm_hash=_hash(source.recipe()),runtime_hash=parent['runtime_hash'],
        implementation=implementation(),parents=parents,run_path=str(folder.relative_to(root)),
        baseline='deferred',eligibility='finished, gold ending, and automatic flexible-v3 correct; human review cannot override ending',training_allowed=False)
    path=root/'plans'/name/'sampling-sft.generation.json'
    source.frozen(path,plan);source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(plan)))
    budget.initialize(root/plan['source_plan'],root)
    if not (folder/'meta/status.json').exists():_write_state(folder,'prepared',0,3)
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);plan=json.loads(path.read_text())
    pilot.load(root/plan['pilot_plan'],root)
    folder=inside(root,plan['run_path'])
    if (plan.get('protocol')!='sampling-sft-generation-v1' or plan['implementation']!=implementation()
            or plan['runtime_hash']!=_hash(plan['settings']['runtime']) or plan['arms']!=list(ARMS)
            or plan['algorithm_hash']!=_hash(source.recipe())
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(plan)}):
        raise ValueError('Generation source/runtime/algorithm/implementation changed; use a new attempt.')
    verify_parents(root,plan['parents'])
    return root,plan,folder


def qualification(root,plan):
    """Recompute real evidence, rather than trusting a report's ready flag."""
    _,parent,pilot_folder=pilot.load(root/plan['pilot_plan'],root)
    replay=pilot.compare(root/plan['pilot_plan'],root)
    if not replay['gpu_qualified']:raise ValueError('Real qualified pilot kill/restart replay is required.')
    if not parent['selection']:raise ValueError('Missing selected runtime/T3 evidence.')
    selection=json.loads((root/parent['selection']).read_text())
    review=pilot_folder/'inputs/runtime-review.json'
    if not review.exists() or json.loads(review.read_text())!=selection['review']:
        raise ValueError('Missing reviewed runtime selection handoff.')
    instrument.select(root/parent['instrument_plan'],root,review)
    report=json.loads((root/plan['pilot_report']).read_text())
    left=pilot.verify_lane(pilot_folder/'uninterrupted',parent)
    correct=sum(t['automatic_flexible_v3']['correct'] for t in left['traces'])
    unfinished=sum(t['end_state'] in ['capped','block_budget_exhausted'] for t in left['traces'])
    if (report['replay']!=replay or report['runtime_hash']!=plan['runtime_hash']
            or report['n_problems']!=32 or report['initial_correct']!=correct
            or report['initial_unfinished']!=unfinished or unfinished/32>0.02 or correct/32<0.9
            or any(a['triggered'] for a in report['alarms'].values())
            or not report['ready_for_generation_review']):
        raise ValueError('Pilot alarms or inconsistent report block Stage2; do not retune automatically.')
    return report


def freeze(path,output_root,review_path):
    root,plan,folder=load(path,output_root)
    report=qualification(root,plan)
    review=json.loads(Path(review_path).read_text())
    plan_approval(plan,review,file_hash(root/plan['pilot_report']))
    target=folder/'generation.review.json'
    if target.exists():
        saved=json.loads(target.read_text())
        if (saved['config_sha256']!=_hash(plan) or saved['review']!=review
                or saved['pilot_report_sha256']!=file_hash(root/plan['pilot_report'])):
            raise ValueError('Generation review changed; preserve the existing attempt.')
        return saved
    ledger=budget.status(root)
    if not reservation_fits(ledger,report['alarms']['budget']['reservation_usd']):
        raise ValueError('Stage2 forecast no longer fits current budget headroom.')
    receipt=dict(config_sha256=_hash(plan),review=review,budget_snapshot=snapshot(root),
                 pilot_report_sha256=file_hash(root/plan['pilot_report']),scope='generation only; per-job paid approval still required')
    source.frozen(target,receipt)
    return receipt


def plan_approval(plan,review,report_sha256):
    if (not isinstance(review,dict) or review.get('generation_approved') is not True
            or review.get('plan_sha256')!=_hash(plan) or review.get('pilot_report_sha256')!=report_sha256
            or not all(isinstance(review.get(k),str) and review[k].strip() for k in ['reviewer','notes'])):
        raise ValueError('Approve this exact generation plan and measured pilot report before Stage2.')


def reservation_fits(ledger,reservation):
    # Decimal/microdollar policy: never round a near-cap amount through a float.
    headroom=min(budget.number(10)-budget.number(ledger['stage_totals']['sampling']['committed_usd']),
                 budget.number(25)-budget.number(ledger['committed_usd']))
    return not ledger['cap_exceeded'] and budget.number(reservation)<=headroom


def arm_folder(folder,k):
    if type(k) is not int or k not in ARMS:raise ValueError('Only approved k0/k2/k5 arms; no baseline.')
    return folder/f'k{k}'


def jsonl(path,values):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        for value in values:stream.write(json.dumps(value,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines()]


def attempt(k,index,state,problem):
    ids=state['response_ids'];end=state['end_state']
    if end not in END_STATES:raise ValueError('Attempt is not terminal.')
    text=state.get('text','')
    flexible=score_gsm8k_flexible_v3(text,problem['gold'],capped=end!='finished')
    strict=score_gsm8k(text,problem['gold'],capped=end!='finished')['strict']
    ending_ok=bool(ids) and ids[-1]==problem['gold_end_token_id'] and ids.count(problem['gold_end_token_id'])==1
    eligible=end=='finished' and ending_ok and flexible['correct']
    return dict(attempt_id=f'k{k}:p{problem["problem_id"]}:r{index}',k=k,
        problem_id=problem['problem_id'],source_id=problem['source_id'],retry_index=index,data_seed=retry_seed(index),
        end_state=end,response_ids=ids,text=text,mean_logprob=state['mean_logprob'],blocks=state['blocks'],
        finish=dict(stop_token_id=ids[-1] if end=='finished' else None,tokens=len(ids),capped=end=='capped'),
        automatic_flexible_v3=flexible,automatic_strict=strict,eligible=eligible,
        rejection_reason=None if eligible else end if end!='finished' else 'wrong-end-token' if not ending_ok else 'automatic-answer-incorrect-or-unextractable',
        eligibility_scope='automatic-only; human overrides are a separate product')


def finalize_round(root,plan,k,index,round_folder,problems,verified):
    states=verified[-1][1]['states'];by_pid={p['problem_id']:p for p in problems}
    if set(map(int,states))!=set(by_pid):raise ValueError('Attempt-round membership changed.')
    rows=[attempt(k,index,state,by_pid[int(pid)]) for pid,state in sorted(states.items(),key=lambda x:int(x[0]))]
    if any(a['end_state']=='engine_error' for a in rows):raise ValueError('Engine failure cannot complete a round.')
    jsonl(round_folder/'attempts.jsonl',rows)
    env=[json.loads((d/'complete.json').read_text())['execution_kind'] for d,_ in verified]
    source.save(round_folder/'complete.json',dict(config_sha256=_hash(plan),k=k,retry_index=index,
        problem_ids=sorted(by_pid),last_block=file_hash(verified[-1][0]/'complete.json'),
        files={'attempts.jsonl':file_hash(round_folder/'attempts.jsonl')},
        execution_kind='real-cuda' if all(e=='real-cuda' for e in env) else 'cpu-fixture',budget_snapshot=snapshot(root)))
    return rows


def verify_round(plan,k,index,folder,problems):
    verified=pilot.blocks(folder,plan)
    if not (folder/'complete.json').exists():return None
    seal=json.loads((folder/'complete.json').read_text())
    if (seal['config_sha256']!=_hash(plan) or seal['k']!=k or seal['retry_index']!=index
            or seal['problem_ids']!=[p['problem_id'] for p in problems]
            or not verified or seal['last_block']!=file_hash(verified[-1][0]/'complete.json')):
        raise ValueError('Attempt round identity changed.')
    if file_hash(folder/'attempts.jsonl')!=seal['files']['attempts.jsonl']:
        raise ValueError('Attempt payload changed.')
    states=verified[-1][1]['states'];by_pid={p['problem_id']:p for p in problems}
    rebuilt=[attempt(k,index,state,by_pid[int(pid)]) for pid,state in sorted(states.items(),key=lambda x:int(x[0]))]
    rows=read_lines(folder/'attempts.jsonl')
    if rows!=rebuilt:raise ValueError('Saved attempts do not reconcile with sealed response states.')
    if any(r['end_state']=='engine_error' for r in rows):raise ValueError('Engine-error round cannot be complete.')
    return rows


def collect_rounds(plan,k,folder):
    problems=plan['problems'];attempts=[];completed=0
    if any(not p.is_dir() or p.name not in ['0','1','2','3'] for p in (folder/'rounds').glob('*')):
        raise ValueError('Unexpected attempt-round artifact.')
    for index in range(4):
        current=folder/'rounds'/str(index)
        if not problems:
            if current.exists():raise ValueError('Generation continued after every problem was covered.')
            continue
        rows=verify_round(plan,k,index,current,problems) if current.exists() else None
        if rows is None:
            if any((folder/'rounds'/str(j)).exists() for j in range(index+1,4)):
                raise ValueError('Attempt rounds are not contiguous.')
            return attempts,problems,index
        attempts.extend(rows);completed=index+1
        successful={a['problem_id'] for a in rows if a['eligible']}
        problems=[p for p in problems if p['problem_id'] not in successful]
    return attempts,problems,completed


def authorize(root,plan,path,folder,k,job_path):
    qualification(root,plan)
    receipt=json.loads((folder/'generation.review.json').read_text())
    plan_approval(plan,receipt['review'],file_hash(root/plan['pilot_report']))
    if (receipt['config_sha256']!=_hash(plan)
            or receipt['pilot_report_sha256']!=file_hash(root/plan['pilot_report'])):
        raise ValueError('Missing exact Stage2 generation approval.')
    instrument.authorize(root,path,job_path,expected_command='sampling-sft-generation-run')
    from pilot_eval.sampling_sft_jobs import _load
    _,job,_=_load(job_path,root)
    argv=job['request']['command']
    if (job['source_plan']!=plan['source_plan'] or '--k' not in argv or argv[argv.index('--k')+1]!=str(k)):
        raise ValueError('Source/arm differs from approved job argv.')


def provisional(plan,k,index,prior,states):
    known={a['problem_id'] for a in prior if a['eligible']}
    by_pid={p['problem_id']:p for p in plan['problems']}
    for pid,state in states.items():
        if state['end_state'] is not None and attempt(k,index,state,by_pid[int(pid)])['eligible']:known.add(int(pid))
    return len(known)


def run(path,output_root,k,job_path=None,*,stop_after_blocks=None,dependencies=None):
    root,plan,parent=load(path,output_root);folder=arm_folder(parent,k)
    if (folder/'complete.json').exists():
        result=verify_arm(root,plan,k,folder);finish_all(root,plan,parent);return result
    if stop_after_blocks is not None and (type(stop_after_blocks) is not int or stop_after_blocks<=0):
        raise ValueError('Stop-after-blocks must be positive.')
    if dependencies is None:authorize(root,plan,path,parent,k,job_path)
    with _writer(folder):
        if (folder/'terminal-error.json').exists():raise ValueError('Engine-error attempt is blocked; inspect before a new plan.')
        prior,problems,index=collect_rounds(plan,k,folder)
        if not problems or index==4:
            result=finish_arm(root,plan,k,folder,prior);finish_all(root,plan,parent);return result
        run_id=uuid.uuid4().hex;engine=tokenizer=runtime=None
        pilot.event(folder,'started',run_id=run_id,retry_index=index,job=str(inside(root,job_path).relative_to(root)) if job_path else None)
        _write_state(folder,'restoring',len({a['problem_id'] for a in prior if a['eligible']}),512)
        new_blocks=0
        try:
            while problems and index<4:
                current=folder/'rounds'/str(index);current.mkdir(parents=True,exist_ok=True)
                verified=pilot.blocks(current,plan)
                if dependencies is None and any(json.loads((d/'complete.json').read_text())['execution_kind']!='real-cuda' for d,_ in verified):
                    raise ValueError('CPU fixtures cannot be resumed as real generation.')
                if (current/'partial').exists():
                    abandoned=current/'abandoned'/uuid.uuid4().hex;abandoned.parent.mkdir(exist_ok=True)
                    (current/'partial').rename(abandoned)
                    pilot.event(folder,'partial-block-abandoned',retry_index=index,resume_block=len(verified),history=str(abandoned.relative_to(folder)))
                resume=verified[-1][1] if verified else None
                if resume and all(s['end_state'] is not None for s in resume['states'].values()):
                    rows=finalize_round(root,plan,k,index,current,problems,verified)
                else:
                    if engine is None:
                        engine_plan={**plan,'settings':{**plan['settings'],'k':k}}
                        if dependencies is None:
                            from pilot_eval.sampling_sft_engine import open_engine
                            engine,tokenizer,runtime=open_engine(engine_plan,root)
                            for p in plan['problems']:
                                ids=tokenizer.apply_chat_template([dict(role='user',content=p['user_message'])],tokenize=True,add_generation_prompt=True)
                                if list(ids)!=p['scorer_prompt_ids']:raise ValueError('Generation scorer retokenization changed.')
                        else:
                            engine,tokenizer,runtime=dependencies.open(engine_plan,root)
                            runtime={**runtime,'execution_kind':'cpu-fixture'}
                    env=current/f'env.{run_id}.json';source.save(env,runtime)
                    engine.clear_cache()
                    partial=current/'partial'

                    def record(value):
                        if 'engine_error' in value:
                            pilot.event(folder,'engine-replay',k=k,retry_index=index,details=value['engine_error']);return
                        partial.mkdir(exist_ok=True)
                        value=json.loads(json.dumps(value));value.pop('replay',None)
                        for out in value.get('outputs',[]):out.pop('num_cached_tokens',None)
                        value.update(k=k,retry_index=index,data_seed=retry_seed(index))
                        instrument.append(partial/'scientific.jsonl',value)
                        if dependencies is not None and hasattr(dependencies,'after_record'):dependencies.after_record(value)

                    def timing(value):
                        partial.mkdir(exist_ok=True);instrument.append(partial/'timings.jsonl',value)

                    def checkpoint(block,states,rng):
                        nonlocal new_blocks
                        partial.mkdir(exist_ok=True)
                        saved={pid:{**s,**(dict(text=tokenizer.decode(s['response_ids'],skip_special_tokens=True))
                            if s['end_state'] is not None else {})} for pid,s in states.items()}
                        source.save(partial/'state.json',dict(next_block=block+1,states=saved,rng=rng))
                        seal=dict(config_sha256=_hash(plan),runtime_hash=plan['runtime_hash'],block=block,
                            attempt=run_id,execution_kind=runtime['execution_kind'],env_sha256=file_hash(env),
                            previous=file_hash(verified[-1][0]/'complete.json') if verified else None,
                            files={p.name:file_hash(p) for p in partial.iterdir() if p.is_file()})
                        source.save(partial/'complete.json',seal)
                        target=current/'blocks'/f'{block:03d}';target.parent.mkdir(exist_ok=True)
                        if target.exists():raise ValueError('Refuse to overwrite a sealed generation block.')
                        partial.rename(target);verified.append((target,dict(next_block=block+1,states=saved,rng=rng)))
                        new_blocks+=1
                        good_before=len({a['problem_id'] for a in prior if a['eligible']})
                        resolved=provisional(plan,k,index,prior,saved)
                        if index==3:resolved=good_before+sum(s['end_state'] is not None for s in saved.values())
                        _write_state(folder,'running',resolved,512)
                        source.save(parent/f'run_k{k}.json',dict(state='running',k=k,retry_index=index,
                            resolved=resolved,total=512,round_terminal=sum(s['end_state'] is not None for s in saved.values()),
                            runtime_hash=plan['runtime_hash'],budget_snapshot=snapshot(root)))
                        print(f'Generation k{k}: {resolved}/512 resolved; attempt {index}/3 retries; block {block+1}/32 sealed.',flush=True)
                        if dependencies is not None and hasattr(dependencies,'after_checkpoint'):dependencies.after_checkpoint(k,index,block)

                    remaining=stop_after_blocks-new_blocks if stop_after_blocks is not None else None
                    result=sample(problems,k,engine,tokenizer,plan['settings']['runtime']['actual_stop_ids'],
                        data_seed=retry_seed(index),resume=resume,on_record=record,on_timing=timing,on_block=checkpoint,
                        stop_after_blocks=remaining)
                    if result['state']=='engine-error':
                        by_pid={p['problem_id']:p for p in problems}
                        failed=[attempt(k,index,s,by_pid[s['problem_id']]) for s in result['traces']]
                        jsonl(current/'failed-attempts.jsonl',failed)
                        source.save(folder/'terminal-error.json',dict(k=k,retry_index=index,errors=result['errors'],
                            failed_attempts_sha256=file_hash(current/'failed-attempts.jsonl'),budget_snapshot=snapshot(root)))
                        raise ValueError('Two engine/protocol failures block this arm; no model retry or completed dataset.')
                    if result['state']=='paused':
                        count=provisional(plan,k,index,prior,verified[-1][1]['states'])
                        _write_state(folder,'paused',count,512)
                        return dict(state='paused',k=k,retry_index=index,production_complete=False)
                    rows=finalize_round(root,plan,k,index,current,problems,verified)
                prior.extend(rows)
                good={a['problem_id'] for a in rows if a['eligible']}
                problems=[p for p in problems if p['problem_id'] not in good];index+=1
                if stop_after_blocks is not None and new_blocks>=stop_after_blocks and problems and index<4:
                    _write_state(folder,'paused',len({a['problem_id'] for a in prior if a['eligible']}),512)
                    return dict(state='paused',k=k,retry_index=index,production_complete=False)
            result=finish_arm(root,plan,k,folder,prior)
            finish_all(root,plan,parent)
            return result
        except BaseException as exc:
            pilot.event(folder,'interrupted',run_id=run_id,k=k,retry_index=index,error=str(exc),type=type(exc).__name__)
            states=verified[-1][1]['states'] if verified else {}
            _write_state(folder,'failed',provisional(plan,k,min(index,3),prior,states),512,str(exc))
            raise


def products(plan,k,attempts):
    by_pid={p['problem_id']:[] for p in plan['problems']}
    for row in attempts:
        if row['k']!=k or row['problem_id'] not in by_pid or row['end_state'] not in END_STATES:
            raise ValueError('Invalid attempt identity/state.')
        by_pid[row['problem_id']].append(row)
    dispositions=[]
    for pid,rows in by_pid.items():
        if not rows or [a['retry_index'] for a in rows]!=list(range(len(rows))) or len(rows)>4:
            raise ValueError('Missing, duplicated or unbounded model attempts.')
        eligible=[a for a in rows if a['eligible']]
        if any(a['end_state']=='engine_error' for a in rows):raise ValueError('Engine-error attempts block completion.')
        if eligible and (len(eligible)!=1 or eligible[0]!=rows[-1]):raise ValueError('Did not stop at first eligible attempt.')
        if not eligible and len(rows)!=4:raise ValueError('Uncovered problem has not exhausted the declared retries.')
        dispositions.append(dict(problem_id=pid,source_id=rows[0]['source_id'],k=k,
            disposition='covered' if eligible else 'uncovered',selected_attempt=eligible[0]['attempt_id'] if eligible else None,
            attempt_ids=[a['attempt_id'] for a in rows],reason=None if eligible else 'bounded-model-retries-exhausted',
            eligibility_scope='automatic-only; human review pending'))
    return dispositions


def summary(plan,k,attempts,dispositions,kind):
    covered=sum(d['disposition']=='covered' for d in dispositions)
    return dict(state='completed',k=k,example_count=512,attempt_count=len(attempts),covered=covered,uncovered=512-covered,
        end_state_counts={state:sum(a['end_state']==state for a in attempts) for state in END_STATES},
        response_tokens=sum(len(a['response_ids']) for a in attempts),runtime_hash=plan['runtime_hash'],
        execution_kind=kind,human_review_pending=True,training_allowed=False,baseline='deferred')


def finish_arm(root,plan,k,folder,attempts):
    dispositions=products(plan,k,attempts)
    kinds=[json.loads(p.read_text())['execution_kind'] for p in (folder/'rounds').glob('*/complete.json')]
    kind='real-cuda' if kinds and all(v=='real-cuda' for v in kinds) else 'cpu-fixture'
    result=summary(plan,k,attempts,dispositions,kind)
    jsonl(folder/'attempts.jsonl',attempts);jsonl(folder/'dispositions.jsonl',dispositions)
    source.save(folder/'results.json',result)
    round_seals={str(p.relative_to(folder)):file_hash(p) for p in (folder/'rounds').glob('*/complete.json')}
    source.save(folder.parent/f'run_k{k}.json',dict(**result,budget_snapshot=snapshot(root),
        cost_scope='cumulative experiment ledger snapshot at collection; active job cost may remain reserved'))
    _write_state(folder,'completed',512,512)
    source.save(folder/'complete.json',dict(config_sha256=_hash(plan),rounds=round_seals,
        files={name:file_hash(folder/name) for name in ['attempts.jsonl','dispositions.jsonl','results.json']},
        run_snapshot_sha256=file_hash(folder.parent/f'run_k{k}.json')))
    return verify_arm(root,plan,k,folder)


def verify_arm(root,plan,k,folder):
    attempts,remaining,index=collect_rounds(plan,k,folder)
    if not (folder/'complete.json').exists():
        current=folder/'rounds'/str(index)
        verified=pilot.blocks(current,plan) if current.exists() else []
        count=provisional(plan,k,min(index,3),attempts,verified[-1][1]['states'] if verified else {})
        return dict(state='blocked' if (folder/'terminal-error.json').exists() else 'partial',k=k,
            attempt_count=len(attempts),covered=len({a['problem_id'] for a in attempts if a['eligible']}),
            provisional_covered=count,remaining=len(remaining),retry_index=index)
    seal=json.loads((folder/'complete.json').read_text())
    if seal['config_sha256']!=_hash(plan):raise ValueError('Arm completion configuration changed.')
    for name,digest in seal['rounds'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Round completion changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Arm scientific result changed.')
    if file_hash(folder.parent/f'run_k{k}.json')!=seal['run_snapshot_sha256']:raise ValueError('Arm cost snapshot changed.')
    dispositions=products(plan,k,attempts)
    if read_lines(folder/'attempts.jsonl')!=attempts or read_lines(folder/'dispositions.jsonl')!=dispositions:
        raise ValueError('Attempt and final-disposition reconciliation failed.')
    result=json.loads((folder/'results.json').read_text())
    if result!=summary(plan,k,attempts,dispositions,result['execution_kind']):raise ValueError('Arm summary does not reconcile.')
    return result


def finish_all(root,plan,folder):
    results=[verify_arm(root,plan,k,arm_folder(folder,k)) for k in ARMS]
    if any(r['state']!='completed' for r in results):
        _write_state(folder,'partial',sum(r['state']=='completed' for r in results),3)
        return None
    result=dict(state='completed',arms=results,runtime_hash=plan['runtime_hash'],training_allowed=False,
        execution_kind='real-cuda' if all(r['execution_kind']=='real-cuda' for r in results) else 'cpu-fixture')
    source.frozen(folder/'results/results.json',result)
    source.frozen(folder/'complete.json',dict(config_sha256=_hash(plan),
        arms={str(k):file_hash(folder/f'k{k}/complete.json') for k in ARMS},
        result_sha256=file_hash(folder/'results/results.json')))
    _write_state(folder,'completed',3,3)
    return result


def verify(path,output_root,k=None):
    root,plan,folder=load(path,output_root)
    if k is not None:
        result=verify_arm(root,plan,k,arm_folder(folder,k))
        if result['state']!='completed':raise ValueError('Arm is not complete.')
        return result
    result=finish_all(root,plan,folder)
    if result is None:raise ValueError('All three512-problem arms must complete.')
    return result


def status(path,output_root):
    root,plan,folder=load(path,output_root)
    return dict(arms={str(k):verify_arm(root,plan,k,arm_folder(folder,k)) for k in ARMS},
        plan_sha256=_hash(plan),pilot_report_sha256=file_hash(root/plan['pilot_report']) if (root/plan['pilot_report']).exists() else None,
        generation_reviewed=(folder/'generation.review.json').exists(),
        budget=snapshot(root),runtime_hash=plan['runtime_hash'],training_allowed=False)


def recover_writer(path,output_root,k,owner_token,review_notes):
    _,_,parent=load(path,output_root);folder=arm_folder(parent,k)
    if not owner_token or not review_notes.strip():raise ValueError('Explicit writer token and reviewed notes required.')
    with run_lock(folder):
        marker=folder/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Generation writer token mismatch.')
        if owner.get('host')==platform.node():
            pid=owner.get('pid')
            if type(pid) is not int or pid<=0:raise ValueError('Invalid writer PID.')
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise ValueError('Writer is still active.')
        pilot.event(folder,'ownership-released',owner=owner,review_notes=review_notes);marker.unlink()
    return dict(state='ownership-released')
