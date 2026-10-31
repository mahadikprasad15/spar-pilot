"""Saved, bounded sampling/score/profile workflow. Public CLI and notebook seam.

This product is not the full-cohort pilot or a production freeze. Incomplete
attempts are preserved for diagnosis; exact restart qualification is Ticket04.
"""
import json
import os
import time
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval.protocol import gsm8k_messages
from pilot_eval.sampling_sft_sampler import sample, score_receipt, token_ids
from pilot_eval.sampling_sft_sources import inside, observe, verify_parents
from pilot_eval.sft import load_sft, safe_name
from pilot_eval.stage0_workflow import _writer
from pilot_eval.run import _write_state
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


RUNTIME_FIELDS = {'vllm_version','torch_version','cuda','driver','gpu','dtype','tf32',
    'worker_numeric_flags','max_num_seqs','max_num_batched_tokens','gpu_memory_utilization',
    'prefix_caching','max_model_len','top_k','min_p','presence_penalty','frequency_penalty','actual_stop_ids',
    'triton_precision'}


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
            ['sampling_sft_instrument.py','sampling_sft_sampler.py','sampling_sft_engine.py']}


def validate_runtime(value):
    if not isinstance(value,dict) or set(value)!=RUNTIME_FIELDS:
        raise ValueError('Resolve every engine control explicitly; no implicit runtime defaults.')
    precision=value['triton_precision']
    if (not isinstance(precision,dict) or set(precision)!={'mode','cache_policy','sources'} or
        precision['mode']!='ieee' or precision['cache_policy']!='isolated-per-engine'):
        raise ValueError('Explicit IEEE precision and isolated-per-engine cache required.')
    sources=precision['sources']
    if (not isinstance(sources,dict) or set(sources)!={'vllm_version','triton_version','unqualified_dots','files'}
        or sources['vllm_version']!='0.11.1' or not isinstance(sources['triton_version'],str)
        or not sources['triton_version'] or type(sources['unqualified_dots']) is not int or sources['unqualified_dots']<1
        or not isinstance(sources['files'],dict) or set(sources['files'])!={'vllm_attention','triton_core','triton_knobs'}):
        raise ValueError('Explicit installed precision source identity required.')
    for item in sources['files'].values():
        if (not isinstance(item,dict) or set(item)!={'path','sha256'} or not isinstance(item['path'],str)
            or not item['path'] or not isinstance(item['sha256'],str) or len(item['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Invalid precision source fingerprint.')
    if (value['vllm_version']!='0.11.1' or not isinstance(value['gpu'], str) or not value['gpu'].strip()
            or value['dtype']!='float32' or value['tf32'] is not False
            or value['worker_numeric_flags']!=dict(matmul_tf32=False,cudnn_tf32=False,matmul_precision='highest')):
        raise ValueError('Supported adapter: vLLM0.11.1, explicit GPU identity, FP32, worker TF32off.')
    for name in ['torch_version','cuda','driver']:
        if not isinstance(value[name],str) or not value[name].strip():
            raise ValueError('Record exact software/driver versions.')
    for name in ['max_num_seqs','max_num_batched_tokens','max_model_len']:
        if type(value[name]) is not int or value[name]<=0:raise ValueError('Invalid scheduler/context control.')
    if (type(value['prefix_caching']) is not bool or not 0<float(value['gpu_memory_utilization'])<1
            or type(value['top_k']) is not int or value['top_k'] not in [-1,0] and value['top_k']<1
            or not 0<=float(value['min_p'])<=1
            or any(not -2<=float(value[n])<=2 for n in ['presence_penalty','frequency_penalty'])):
        raise ValueError('Invalid explicit decoding controls.')
    stops=token_ids(value['actual_stop_ids'])
    if len(stops)!=2 or len(set(stops))!=2:raise ValueError('Resolve both distinct actual stop IDs.')


def inventory(root,folder,parents):
    folder=inside(root,folder)
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    if not files:raise ValueError('Supply an existing local model/tokenizer snapshot; downloads are separate.')
    for path in files:observe(root,path,parents)
    return str(folder.relative_to(root))


def prepare(config_path, output_root, settings_path, name):
    root,config,parent=source.load(config_path,output_root)
    audit=source.verify(config_path,root)['audit']
    if not audit['source_ready']:raise ValueError('Complete source preflight first.')
    safe_name(name)
    settings=json.loads(Path(settings_path).read_text())
    required={'runtime','mode','k','problem_ids','local_model','local_tokenizer'}
    if set(settings) not in [required,required|{'fixtures'}]:raise ValueError('Unexpected bounded settings schema.')
    validate_runtime(settings['runtime'])
    mode=settings['mode']
    if mode not in ['bounded','profile','scores','oracle','fixtures']:raise ValueError('Unknown bounded mode.')
    if settings['k'] not in [0,2,5]:raise ValueError('Only approved k0/2/5.')
    pids=settings['problem_ids']
    if (not isinstance(pids,list) or not 1<=len(pids)<=64 or pids!=sorted(set(pids))
            or any(type(i) is not int or not 0<=i<512 for i in pids)):
        raise ValueError('Predeclare1–64 ordered problem ordinals within the saved512 cohort.')
    if mode=='fixtures' and (settings['k']!=0 or len(pids)!=20):
        raise ValueError('Score-fixture capture requires20 predeclared problems at k0.')
    if mode=='profile' and (len(pids)!=64 or settings['k']!=2):
        raise ValueError('Throughput profile requires64 fixed problems and k2, one block.')
    parents={str(inside(root,config_path).relative_to(root)):file_hash(inside(root,config_path)),
             str((parent/'results/source-audit.json').relative_to(root)):file_hash(parent/'results/source-audit.json')}
    settings=dict(settings)
    for key in ['local_model','local_tokenizer']:
        settings[key]=inventory(root,settings[key],parents)
    gold_path=root/audit['training']['config_path']
    _,rows,_=load_sft(gold_path,root)
    problems=[dict(problem_id=i,source_id=rows[i]['id'],gold=rows[i]['gold'],
        user_message=gsm8k_messages(rows[i]['question'])[0]['content'],
        scorer_prompt_ids=rows[i]['input_ids'][:rows[i]['prompt_tokens']]) for i in pids]
    if mode in ['scores','oracle']:
        fixture_path=observe(root,settings.get('fixtures',''),parents)
        fixtures=json.loads(fixture_path.read_text())
        validate_fixtures(fixtures,settings['runtime']['actual_stop_ids'],problems)
        coordinates=set()
        from pilot_eval.sampling_sft_sampler import generated
        for fixture in fixtures:
            ref=fixture.get('generation_record',{})
            log=observe(root,ref.get('path',''),parents)
            seal=observe(root,log.parent/'complete.json',parents)
            sealed=json.loads(seal.read_text())
            if log.name!='requests.jsonl' or sealed.get('files',{}).get(log.name)!=file_hash(log):
                raise ValueError('T3 traces require a sealed saved generation log.')
            row=json.loads(log.read_text().splitlines()[ref['line']])
            idx=ref['output_index'];request=row['requests'][idx];out=row['outputs'][idx]
            coordinate=(str(log),ref['line'],idx)
            if coordinate in coordinates:raise ValueError('Do not duplicate one saved trace as distinct T3 fixtures.')
            coordinates.add(coordinate)
            if (row['kind']!='generate' or request!=fixture['generation_request']
                    or request['problem_id']!=fixture['problem_id']
                    or request['prefix_ids']+generated(out,request['max_tokens'],settings['runtime']['actual_stop_ids'])!=fixture['response_ids']):
                raise ValueError('T3 IDs/provenance differ from saved generated trace.')
        settings['fixtures']=str(fixture_path.relative_to(root))
    elif 'fixtures' in settings:raise ValueError('Fixtures only apply to scoring modes.')
    plan=dict(protocol='sampling-sft-bounded-instrument-v1',name=name,
        source_plan=str(inside(root,config_path).relative_to(root)),parents=parents,settings=settings,
        problems=problems,implementation=implementation(),runtime_hash=_hash(settings['runtime']),
        run_path=str((parent/'instrument'/name).relative_to(root)),production_ready=False)
    target=root/'plans'/name/'sampling-sft.instrument.json'
    source.frozen(target,plan)
    source.frozen(root/plan['run_path']/'plan-seal.json',dict(config_sha256=_hash(plan)))
    if not (root/plan['run_path']/'meta/status.json').exists():
        _write_state(root/plan['run_path'],'prepared',0,len(problems))
    return target


def validate_fixtures(fixtures,stops,problems):
    if not isinstance(fixtures,list) or len(fixtures)!=20 or len({r['trace_id'] for r in fixtures})!=20:
        raise ValueError('Predeclare exactly20 score traces before calling either engine.')
    by_pid={p['problem_id']:p for p in problems}
    lengths={'short':[],'long':[]};categories=set()
    for row in fixtures:
        token_ids(row['prompt_ids']);ids=token_ids(row['response_ids'])
        if not ids or len(ids)>1024 or row['prompt_ids']!=by_pid[row['problem_id']]['scorer_prompt_ids']:
            raise ValueError('Fixture response/context differs from declared saved source.')
        cats=set(row['categories']);categories.update(cats)
        if 'finished' in cats and ids[-1] not in stops:raise ValueError('Finished fixture lacks actual stop.')
        if 'capped' in cats and (len(ids)!=1024 or any(i in stops for i in ids)):
            raise ValueError('Capped fixture must have1024 tokens and no stop.')
        if any(i in stops for i in ids[:-1]):raise ValueError('Fixture contains a premature stop.')
        # Provenance is a saved successful generation request, never a hand-edited list.
        if row.get('generation_request',{}).get('prefix_ids')!=[] and 'empty-prefix' in cats:
            raise ValueError('Empty-prefix category requires an actual empty generation prefix.')
        for key in lengths:
            if key in cats:lengths[key].append(len(ids))
    if not {'short','long','finished','capped','empty-prefix'}<=categories:
        raise ValueError('T3 fixture categories incomplete; do not substitute categories after scoring.')
    if max(lengths['short'])>=min(lengths['long']):raise ValueError('Short/long fixture labels overlap.')


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path)
    plan=json.loads(path.read_text());folder=inside(root,plan['run_path'])
    source.verify(root/plan['source_plan'],root)
    if (plan.get('protocol')!='sampling-sft-bounded-instrument-v1'
            or plan.get('implementation')!=implementation()
            or json.loads((folder/'plan-seal.json').read_text())!={'config_sha256':_hash(plan)}
            or plan['runtime_hash']!=_hash(plan['settings']['runtime'])):
        raise ValueError('Bounded instrument plan/code/seal changed; use a new name.')
    verify_parents(root,plan['parents'])
    for key in ['local_model','local_tokenizer']:
        current={str(p.relative_to(root)) for p in inside(root,plan['settings'][key]).rglob('*') if p.is_file()}
        frozen={p for p in plan['parents'] if inside(root,p).is_relative_to(inside(root,plan['settings'][key]))}
        if current!=frozen:raise ValueError('Local snapshot file inventory changed.')
    return root,plan,folder


def authorize(root,path,job_path,*,expected_command='sampling-sft-instrument-run'):
    """A real child must match its already reserved/launched trusted job argv."""
    if not job_path:raise ValueError('Real engine work requires a launched budgeted job; no model imported.')
    from pilot_eval.sampling_sft_jobs import _load, _state
    from pilot_eval.sampling_sft_budget import status as budget_status
    _,job,folder=_load(job_path,root)
    _state(folder,job)
    launch=json.loads((folder/'launch.json').read_text())
    request=job['request'];argv=request['command']
    if (request['mode']!='runpod' or request['stage']!='sampling'
            or launch['pod_id']!=os.environ.get('RUNPOD_POD_ID')
            or len(argv)<4 or argv[3]!=expected_command
            or '--config' not in argv or '--job' not in argv
            or inside(root,argv[argv.index('--config')+1])!=inside(root,path)
            or inside(root,argv[argv.index('--job')+1])!=inside(root,job_path)
            or (folder/'execution.json').exists()):
        raise ValueError('Scientific child does not match the active approved job.')
    ledger=budget_status(root)
    current=ledger['jobs'].get(request['job_id'],{})
    if ((ledger['cap_exceeded'] and not current.get('budget_limits_waived')) or not current.get('launch_recorded') or current.get('state')!='reserved'
            or current.get('request_sha256')!=job['request_sha256']):
        raise ValueError('Job has no active approved cost reservation.')


def append(path,value):
    with path.open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(path,output_root,job_path=None,*,dependencies=None):
    root,plan,folder=load(path,output_root)
    if (folder/'complete.json').exists():return verify(path,root)
    if dependencies is None:authorize(root,path,job_path)
    with _writer(folder):
        if (folder/'started.json').exists():
            raise ValueError('Interrupted/failed bounded attempt preserved; use a new name. Pilot resume is Ticket04.')
        source.frozen(folder/'started.json',dict(config_sha256=_hash(plan),simulation=dependencies is not None))
        _write_state(folder,'loading-engine',0,len(plan['problems']))
        start=time.monotonic()
        try:
            if dependencies is None:
                from pilot_eval.sampling_sft_engine import open_engine
                engine,tokenizer,runtime=open_engine(plan,root)
            else:
                engine,tokenizer,runtime=dependencies.open(plan,root)
                runtime={**runtime,'execution_kind':'cpu-fixture'}
            source.save(folder/'env.json',runtime)
            settings=plan['settings'];mode=settings['mode']
            if dependencies is None:
                for problem in plan['problems']:
                    actual=tokenizer.apply_chat_template([dict(role='user',content=problem['user_message'])],
                        tokenize=True,add_generation_prompt=True)
                    if list(actual)!=problem['scorer_prompt_ids']:
                        raise ValueError('Pinned scorer prompt retokenization mismatch.')
            if mode=='fixtures':
                result=capture_fixtures(plan,root,folder,engine,tokenizer)
            elif mode in ['scores','oracle']:
                fixtures=json.loads((root/settings['fixtures']).read_text())
                values=engine.score(fixtures)
                # Validate now; malformed scores cannot become a comparison receipt later.
                from pilot_eval.sampling_sft_sampler import scored
                for value,request in zip(values,fixtures):scored(value,request)
                if len(values)!=20:raise ValueError('Score engine cardinality mismatch.')
                result=dict(state='completed',scores=values,fixture_sha256=file_hash(root/settings['fixtures']))
            else:
                timings=[]
                if hasattr(engine,'measurement_start'):engine.measurement_start()
                def timing(value):
                    timings.append(value);append(folder/'timings.jsonl',value)
                def checkpoint(block,states,rng):
                    source.save(folder/'blocks'/f'{block:03d}.json',dict(states=states,rng=rng))
                    completed=sum(s['end_state'] is not None for s in states.values())
                    _write_state(folder,'running',completed,len(states))
                    print(f'Bounded k{settings["k"]}: block {block+1}; {completed}/{len(states)} terminal; '
                          f'{sum(len(s["response_ids"]) for s in states.values())} retained tokens.',flush=True)
                workload_start=time.monotonic()
                result=sample(plan['problems'],settings['k'],engine,tokenizer,
                    settings['runtime']['actual_stop_ids'],
                    on_block=checkpoint,
                    on_record=lambda record:append(folder/'requests.jsonl',record),
                    on_timing=timing,
                    block_limit=1 if mode=='profile' else 32)
                if mode=='profile':
                    engine_seconds=sum(t['seconds'] for t in timings)
                    generation_seconds=sum(t['seconds'] for t in timings if t['kind']=='generate')
                    generated_tokens=sum(t['generated_tokens'] for t in timings)
                    prefill_tokens=sum(t['input_tokens']-t['cached_tokens'] for t in timings)
                    result['profile']=dict(workload_wall_seconds=time.monotonic()-workload_start,
                        generation_seconds=generation_seconds,engine_seconds=engine_seconds,
                        generated_tokens=generated_tokens,uncached_prefill_tokens=prefill_tokens,
                        decode_tokens_per_second=generated_tokens/generation_seconds if generation_seconds else None,
                        prefill_tokens_per_second=prefill_tokens/engine_seconds if engine_seconds else None,
                        rate_scope='request-wall throughput proxies, not isolated CUDA kernel rates; auxiliary scoring decode excluded',
                        peak_gpu_memory=engine.measurement_end() if hasattr(engine,'measurement_end') else None)
                    result['batch_dependence']=batch_dependence(engine,result)
                    result['trace_scope']='one-block throughput fixture; nonterminal traces are not completed attempts'
                for trace in result['traces']:
                    append(folder/'responses.jsonl',dict(**trace,k=settings['k'],data_seed=0,
                        text=tokenizer.decode(trace['response_ids'],skip_special_tokens=False)))
            result.update(execution_kind=runtime['execution_kind'],gpu_qualified=False,
                runtime_hash=plan['runtime_hash'],seconds=time.monotonic()-start,production_ready=False)
            source.save(folder/'results.json',result)
            if result['state']=='engine-error':raise ValueError('Engine protocol replay exhausted; diagnostics saved.')
            _write_state(folder,'completed',len(plan['problems']),len(plan['problems']))
            files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*')
                   if p.is_file() and p.name not in ['writer.json','complete.json']}
            source.save(folder/'complete.json',dict(config_sha256=_hash(plan),files=files))
        except Exception as exc:
            source.save(folder/'failure.json',dict(state='failed',type=type(exc).__name__,error=str(exc)))
            _write_state(folder,'failed',0,len(plan['problems']),str(exc))
            raise
    return verify(path,root)



def capture_fixtures(plan,root,folder,engine,tokenizer):
    """Twenty fixed scorer fixtures; one explicitly forced cap, never training data."""
    from pilot_eval.sampling_sft_sampler import proposer, generated, request_seed, ProtocolError, diagnostic
    problems=plan['problems'];stops=plan['settings']['runtime']['actual_stop_ids']
    requests=[dict(problem_id=p['problem_id'],block=0,step=-1,
        seed=request_seed(0,p['problem_id'],0,-1),prefix_ids=[],
        prompt_ids=proposer(p,[],tokenizer),max_tokens=1024) for p in problems]
    requests[-1]['qualification_min_tokens']=1024
    print('Score-fixture capture:20 fixed requests; final request forced to1024 tokens; not training data.',flush=True)
    if hasattr(engine,'clear_cache'):engine.clear_cache()
    errors=[]
    for replay in range(2):
        outputs=None
        try:
            started=time.monotonic();outputs=engine.generate(requests)
            if not isinstance(outputs,list) or len(outputs)!=20:
                raise ProtocolError('Fixture capture requires20 aligned generation outputs.')
            ids=[generated(out,1024,stops) for out in outputs]
            if len(ids[-1])!=1024 or any(token in stops for token in ids[-1]):
                raise ProtocolError('Predeclared forced-cap fixture did not reach1024 without stops.')
            append(folder/'requests.jsonl',dict(kind='generate',requests=requests,outputs=outputs,replay=replay))
            append(folder/'timings.jsonl',dict(kind='generate',seconds=time.monotonic()-started))
            break
        except (RuntimeError,ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
            error=dict(replay=replay,type=type(exc).__name__,error=str(exc),raw_outputs=diagnostic(outputs))
            errors.append(error);append(folder/'requests.jsonl',dict(engine_error=error))
    else:raise ProtocolError('Same fixed fixture generation failed twice; stop.')
    low=min(map(len,ids));high=max(map(len,ids))
    if low==high:raise ValueError('Fixed fixture capture lacks distinct short/long traces; stop before scoring.')
    # Length-only labels are fixed before either scorer runs. Never select on score agreement.
    line=len((folder/'requests.jsonl').read_text().splitlines())-1
    fixtures=[]
    for index,(problem,request,response) in enumerate(zip(problems,requests,ids)):
        cats=['empty-prefix']
        if len(response)==low:cats.append('short')
        if len(response)==high:cats.append('long')
        if response[-1] in stops:cats.append('finished')
        elif len(response)==1024:cats.append('capped')
        fixture=dict(trace_id=f"fixed-{problem['problem_id']}",problem_id=problem['problem_id'],
            prompt_ids=problem['scorer_prompt_ids'],response_ids=response,categories=cats,
            generation_request=request,generation_record=dict(path=str((folder/'requests.jsonl').relative_to(root)),
                                                              line=line,output_index=index))
        fixtures.append(fixture)
        append(folder/'responses.jsonl',dict(**fixture,training_eligible=False,
            text=tokenizer.decode(response,skip_special_tokens=False)))
    validate_fixtures(fixtures,stops,problems)
    source.frozen(folder/'fixtures.json',fixtures)
    print(f'Score-fixture capture complete:20 traces; lengths {low}–{high}; frozen before score comparison.',flush=True)
    return dict(state='completed',traces=fixtures,errors=errors,training_eligible=False,
        scope='Fixed scoring fixtures; final trace forced to1024 tokens by min_tokens, not a model outcome.')


def batch_dependence(engine,result):
    batch=next(row for row in result['requests'] if row['kind']=='generate')
    engine.clear_cache()
    singleton=[]
    for request in batch['requests']:
        singleton.extend(engine.generate([request]));engine.clear_cache()
    differences=[]
    for req,left,right in zip(batch['requests'],batch['outputs'],singleton):
        a,b=left['token_ids'],right['token_ids']
        differences.append(dict(problem_id=req['problem_id'],different_tokens=
            sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b)),batch_ids=a,singleton_ids=b))
    return differences


def status(path,output_root):
    root,plan,folder=load(path,output_root)
    complete=folder/'complete.json'
    if complete.exists():
        seal=json.loads(complete.read_text())
        if seal['config_sha256']!=_hash(plan):raise ValueError('Bounded completion seal changed.')
        for name,digest in seal['files'].items():
            if file_hash(inside(folder,name))!=digest:raise ValueError('Bounded scientific artifact changed.')
    return dict(state='completed' if complete.exists() else 'failed' if (folder/'failure.json').exists()
                else 'interrupted' if (folder/'started.json').exists() else 'prepared',
                result=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else None,
                runtime_hash=plan['runtime_hash'],production_ready=False)


def verify(path,output_root):
    value=status(path,output_root)
    if value['state']!='completed':raise ValueError('Bounded instrument has no verified completion.')
    return value


def compare(path,output_root,oracle_path):
    root,plan,folder=load(path,output_root)
    _,other,_=load(oracle_path,root)
    left,right=verify(path,root)['result'],verify(oracle_path,root)['result']
    if (plan['settings']['mode']!='scores' or other['settings']['mode']!='oracle'
            or plan['source_plan']!=other['source_plan'] or plan['runtime_hash']!=other['runtime_hash']
            or left['fixture_sha256']!=right['fixture_sha256']):
        raise ValueError('Compare the same frozen fixtures and sampling runtime only.')
    receipt=score_receipt(json.loads((root/plan['settings']['fixtures']).read_text()),left['scores'],right['scores'])
    receipt.update(execution_kind=left['execution_kind'],gpu_qualified=receipt['passed'] and
        left['execution_kind']=='real-cuda' and right['execution_kind']=='real-cuda',
        parents={str(inside(root,p).relative_to(root)):file_hash(p) for p in [path,oracle_path]},
        fixture_sha256=left['fixture_sha256'],runtime_hash=plan['runtime_hash'],production_ready=False)
    source.frozen(folder/'score-receipt.json',receipt)
    if not receipt['passed']:raise ValueError('T3 failed; receipt saved, thresholds unchanged.')
    return receipt


def select(path,output_root,review_path):
    """Freeze a measured Stage0.5 choice; this does not certify exact replay."""
    root,plan,folder=load(path,output_root)
    review=json.loads(Path(review_path).read_text())
    if not all(isinstance(review.get(k),str) and review[k].strip() for k in ['reviewer','notes']):
        raise ValueError('Record candidate review and reason before selecting a runtime.')
    candidates=review.get('candidates')
    if not isinstance(candidates,list) or not candidates:raise ValueError('Review every declared measured candidate.')
    profiles=[];parents={}
    common=None;corpus=None
    for candidate in candidates:
        _,cfg,directory=load(candidate,root);result=verify(candidate,root)['result']
        if cfg['settings']['mode']!='profile' or result['execution_kind']!='real-cuda' or result['state']=='engine-error':
            raise ValueError('Real worker/profile evidence required; CPU fixtures cannot qualify a runtime.')
        comparable={k:v for k,v in cfg['settings']['runtime'].items()
                    if k not in ['max_num_seqs','max_num_batched_tokens']}
        if common is None:common,corpus=comparable,cfg['problems']
        if comparable!=common or cfg['problems']!=corpus:raise ValueError('Profile candidate inputs/non-scheduler settings differ.')
        if cfg['settings']['runtime']['max_num_seqs'] not in [32,64,128,256,512]:
            raise ValueError('Profile scheduler candidate outside declared sweep.')
        if not result.get('batch_dependence') or result['profile']['peak_gpu_memory'] is None:
            raise ValueError('Candidate lacks batching/worker memory evidence.')
        parents[str((directory/'complete.json').relative_to(root))]=file_hash(directory/'complete.json')
        profiles.append((result['profile']['workload_wall_seconds'],str(inside(root,candidate).relative_to(root)),cfg))
    _,winner,chosen=min(profiles,key=lambda x:(x[0],x[1]))
    if chosen['runtime_hash']!=plan['runtime_hash']:raise ValueError('Select via the fastest candidate configuration.')
    score_path=inside(root,review.get('score_config',''))
    oracle_path=inside(root,review.get('oracle_config',''))
    receipt=compare(score_path,root,oracle_path)
    if not receipt['gpu_qualified'] or receipt['runtime_hash']!=chosen['runtime_hash']:
        raise ValueError('Selected runtime lacks passing real T3 evidence.')
    for p in [score_path,oracle_path]:parents[str(p.relative_to(root))]=file_hash(p)
    result=dict(state='stage0.5-selected',selected=winner,runtime=chosen['settings']['runtime'],
        runtime_hash=chosen['runtime_hash'],parents=parents,review=review,
        scope='bounded score/worker/profile checks; pilot exact replay remains pending',production_ready=False)
    source.frozen(folder/'runtime-selection.json',result)
    return result


def console(value):
    """Keep raw prompts, IDs and proposals in files, not megabytes of cell output."""
    result=value.get('result',value)
    return {key:result[key] for key in ['state','execution_kind','runtime_hash','gpu_qualified',
            'production_ready','seconds','profile','passed','scope','selected'] if key in result} | dict(
        traces=len(result.get('traces',[])),refinement_decisions=len(result.get('steps',[])),
        engine_errors=len(result.get('errors',[])))
