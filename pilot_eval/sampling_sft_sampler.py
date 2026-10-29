"""Engine-independent likelihood search. No imports of a model or cloud client.

Engines accept ordered request lists. Every engine answer is checked before it
enters scientific state. The caller owns durable completed-block checkpoints.
"""
import math
import random
import re
import time

from pilot_eval.sampling_sft import PROPOSER_TEMPLATE


class ProtocolError(ValueError):
    """An engine response cannot be interpreted as a model outcome."""


def clean_gold(text):
    text = re.sub(r'<<[^>]*>>', '', text)
    match = re.search(r'####\s*([^\n]+)\s*$', text)
    if not match:
        raise ValueError('Gold answer has no final #### marker.')
    return text[:match.start()] + 'The final answer is ' + match.group(1).strip() + '.'


def request_seed(data_seed, pid, block, step):
    return (((data_seed*1_000_003+pid)*1009+block)*1013+step+1) % 2**31


def proposer(problem, prefix, tokenizer):
    text = PROPOSER_TEMPLATE.format(TRAINING_USER_MESSAGE=problem['user_message'],
        CLEANED_GOLD=clean_gold(problem['gold']),
        PARTIAL_RESPONSE_TEXT=tokenizer.decode(prefix, skip_special_tokens=True))
    # Duplication is intentional: text in user message AND original assistant IDs.
    ids = tokenizer.apply_chat_template([dict(role='user', content=text)],
                                        tokenize=True, add_generation_prompt=True)
    return list(ids) + prefix


def token_ids(ids):
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ProtocolError('Malformed token IDs.')
    return ids


def generated(output, maximum, stops):
    ids = list(token_ids(output.get('token_ids')))
    reason, stop = output.get('finish_reason'), output.get('stop_reason')
    if reason not in ['stop', 'length']:
        raise ProtocolError('Missing/unknown generation finish reason.')
    # Append only a *reported* stop ID; never infer a missing EOS from text.
    if reason == 'stop' and (not ids or ids[-1] not in stops):
        if type(stop) is not int or stop not in stops:
            raise ProtocolError('Stopped generation omitted an identifiable stop token.')
        ids.append(stop)
    if not ids or len(ids) > maximum:
        raise ProtocolError('Empty generation or token budget overflow.')
    if reason == 'length' and len(ids) != maximum:
        raise ProtocolError('Length-limited generation returned fewer than requested tokens without a stop.')
    positions = [i for i, value in enumerate(ids) if value in stops]
    if positions and (positions != [len(ids)-1] or reason != 'stop'):
        raise ProtocolError('Inconsistent first-stop accounting.')
    if reason == 'stop' and not positions:
        raise ProtocolError('Stop reason without a stop token.')
    return ids


def scored(output, request):
    ids = request['response_ids']
    expected = list(range(len(request['prompt_ids']), len(request['prompt_ids'])+len(ids)))
    values = output.get('logprobs')
    if (output.get('token_ids') != ids or output.get('positions') != expected
            or not ids or not isinstance(values, list) or len(values) != len(ids)
            or any(type(x) not in [float, int] or not math.isfinite(x) or x > 0 for x in values)):
        raise ProtocolError('Malformed/misaligned response-token log probabilities.')
    return math.fsum(values)/len(values)


def diagnostic(value):
    """Retain malformed numerical evidence as valid JSON, including NaN/Inf."""
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':repr(value)}
    if isinstance(value,dict):return {str(k):diagnostic(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'representation':repr(value)}


def sample(problems, k, engine, tokenizer, stops, *, data_seed=0, on_block=None, on_record=None,
           on_timing=None, block_limit=32, resume=None, stop_after_blocks=None):
    """One bounded attempt per problem; retries/production resume belong upstream.

    Errors replay the identical ordered logical batch once. An exhausted replay
    stops the run, including all still-active traces; it never advances a seed.
    """
    if k not in [0, 2, 5]:
        raise ValueError('Unsupported search amount.')
    if not problems or len({p['problem_id'] for p in problems}) != len(problems):
        raise ValueError('Require distinct problems.')
    problems = sorted(problems, key=lambda p:p['problem_id'])
    states = {p['problem_id']:dict(problem_id=p['problem_id'], source_id=p.get('source_id'), response_ids=[],
                                   mean_logprob=None, end_state=None, blocks=0) for p in problems}
    rng = {p['problem_id']:random.Random((data_seed*1_000_003+p['problem_id']) % 2**31) for p in problems}
    first_block=0
    if resume is not None:
        first_block=resume['next_block']
        if type(first_block) is not int or not 0<=first_block<=32:
            raise ValueError('Invalid completed-block resume boundary.')
        def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        if set(map(int,resume['states']))!=set(states) or set(map(int,resume['rng']))!=set(rng):
            raise ValueError('Resume cohort identity differs.')
        states={int(pid):dict(value) for pid,value in sorted(resume['states'].items(),key=lambda item:int(item[0]))}
        for pid,value in resume['rng'].items():rng[int(pid)].setstate(tuples(value))
    records, decisions, errors = [], [], []

    def call(kind, requests, validator):
        for replay in range(2):
            outputs=None
            try:
                start=time.monotonic()
                outputs = getattr(engine,kind)(requests)
                elapsed=time.monotonic()-start
                if not isinstance(outputs,list) or len(outputs) != len(requests):
                    raise ProtocolError('Engine batch cardinality changed.')
                checked = [validator(out, req) for out,req in zip(outputs,requests)]
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                error = dict(kind=kind, requests=requests, replay=replay,
                             error=str(exc), type=type(exc).__name__,
                             raw_outputs=diagnostic(outputs))
                errors.append(error)
                if on_record: on_record(dict(engine_error=error))
                continue
            entry = dict(kind=kind, requests=requests, outputs=outputs, replay=replay)
            records.append(entry)
            if on_record: on_record(entry)
            if on_timing:on_timing(dict(kind=kind,seconds=elapsed,
                input_tokens=sum(len(r['prompt_ids'])+(len(r['response_ids']) if kind=='score' else 0) for r in requests),
                cached_tokens=sum(o.get('num_cached_tokens',0) or 0 for o in outputs),
                generated_tokens=sum(len(o['token_ids']) for o in outputs) if kind=='generate' else 0))
            return checked
        raise ProtocolError('Same logical request failed twice; stop the run.')

    try:
        for block in range(first_block,block_limit):
            active = [p for p in problems if states[p['problem_id']]['end_state'] is None]
            if not active: break
            prefixes = [states[p['problem_id']]['response_ids'] for p in active]

            def generate(prefixes, maxima, step):
                requests = [dict(problem_id=p['problem_id'],block=block,step=step,
                    seed=request_seed(data_seed,p['problem_id'],block,step),
                    prefix_ids=prefix, prompt_ids=proposer(p,prefix,tokenizer), max_tokens=maximum)
                    for p,prefix,maximum in zip(active,prefixes,maxima)]
                return call('generate',requests,lambda out,req:generated(out,req['max_tokens'],stops))

            def score(traces, step):
                requests = [dict(problem_id=p['problem_id'],block=block,step=step,
                    prompt_ids=p['scorer_prompt_ids'],response_ids=ids) for p,ids in zip(active,traces)]
                return call('score',requests,scored)

            extensions = generate(prefixes,[min(64,1024-len(ids)) for ids in prefixes],-1)
            current = [prefix+extra for prefix,extra in zip(prefixes,extensions)]
            scores = score(current,-1)
            for step in range(k):
                cuts = [rng[p['problem_id']].randint(0,len(ids)-1) for p,ids in zip(active,current)]
                retained = [ids[:cut] for ids,cut in zip(current,cuts)]
                extra = generate(retained,[len(ids)-cut for ids,cut in zip(current,cuts)],step)
                proposals = [prefix+new for prefix,new in zip(retained,extra)]
                new_scores = score(proposals,step)
                for i,p in enumerate(active):
                    accept = new_scores[i] > scores[i]
                    entry = dict(problem_id=p['problem_id'],block=block,step=step,cut=cuts[i],
                        old_ids=list(current[i]),proposal_ids=proposals[i],old_score=scores[i],
                        proposal_score=new_scores[i],accepted=accept)
                    decisions.append(entry)
                    if on_record:on_record(dict(decision=entry))
                    if accept:current[i],scores[i]=proposals[i],new_scores[i]
            for p,ids,value in zip(active,current,scores):
                terminal = ('finished' if ids[-1] in stops else 'capped' if len(ids)==1024
                            else 'block_budget_exhausted' if block==31 else None)
                states[p['problem_id']].update(response_ids=ids,mean_logprob=value,
                                               end_state=terminal,blocks=block+1)
            engine.clear_cache()
            if on_block:on_block(block,states,{pid:r.getstate() for pid,r in rng.items()})
            if stop_after_blocks is not None and block-first_block+1>=stop_after_blocks:break
    except ProtocolError:
        for state in states.values():
            if state['end_state'] is None:state['end_state']='engine_error'
    return dict(state='engine-error' if any(s['end_state']=='engine_error' for s in states.values())
                else 'paused' if any(s['end_state'] is None for s in states.values()) and block_limit==32 else 'completed',
                traces=list(states.values()),steps=decisions,requests=records,errors=errors,
                gpu_qualified=False)


def score_receipt(traces, actual, oracle):
    """T3. Frozen fixtures supply categories before either scorer is run."""
    if len(traces)!=20 or len({r['trace_id'] for r in traces})!=20:
        raise ValueError('T3 requires exactly20 distinct predeclared traces.')
    if len(actual)!=20 or len(oracle)!=20:
        raise ValueError('T3 score cardinality changed.')
    required={'short','long','finished','capped','empty-prefix'}
    if not required <= {c for r in traces for c in r['categories']}:
        raise ValueError('T3 categories incomplete.')
    rows=[]
    for trace,left,right in zip(traces,actual,oracle):
        try:
            lm,rm=scored(left,trace),scored(right,trace)
            errors=[abs(a-b) for a,b in zip(left['logprobs'],right['logprobs'])]
            worst=max(range(len(errors)),key=errors.__getitem__)
            row=dict(trace_id=trace['trace_id'],aligned=True,mean_error=abs(lm-rm),
                max_token_error=errors[worst],worst_position=left['positions'][worst],
                token_ids=left['token_ids'],positions=left['positions'],
                engine_logprobs=left['logprobs'],oracle_logprobs=right['logprobs'],
                passed=abs(lm-rm)<=0.001 and errors[worst]<=0.01)
        except (ProtocolError,TypeError,KeyError) as exc:
            row=dict(trace_id=trace['trace_id'],aligned=False,passed=False,error=str(exc),actual=left,oracle=right)
        rows.append(row)
    return dict(passed=all(r['passed'] for r in rows),traces=rows,
                mean_allowance=0.001,per_token_allowance=0.01,scope='score correctness only')
