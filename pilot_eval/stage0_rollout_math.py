"""Untuned draw validation and cohort membership; no model/runtime dependencies."""
import math

import numpy as np

from pilot_eval.scoring import score_gsm8k,score_gsm8k_flexible_v2,score_gsm8k_flexible_v3
from pilot_eval.workflow import _hash


def normalize(row,output,draw,*,origin,seed,limit,eos,historical_reward=None):
    content,raw=output.get('token_ids'),output.get('generated_token_ids')
    if (type(draw) is not int or not 0<=draw<8 or not isinstance(output.get('text'),str)
            or any(not isinstance(v,list) for v in [content,raw]) or not raw
            or any(type(t) is not int or t<0 for t in content+raw) or len(raw)>limit):
        raise ValueError('Invalid eight-draw token/text identity.')
    reason=output.get('stop_reason');capped=reason=='cap'
    if capped:
        if content!=raw or len(raw)!=limit or (eos is not None and eos in raw):raise ValueError('Invalid capped sampled continuation.')
    elif reason=='eos':
        if content!=raw[:-1] or (eos is not None and (raw[-1]!=eos or eos in raw[:-1])):
            raise ValueError('Sampled EOS/content metadata disagree.')
    else:raise ValueError('Unknown sampled stopping reason.')
    scores=dict(strict=score_gsm8k(output['text'],row['gold'],capped)['strict'],
        flexible_v2=score_gsm8k_flexible_v2(output['text'],row['gold'],capped),
        flexible_v3=score_gsm8k_flexible_v3(output['text'],row['gold'],capped))
    if scores['strict']['correct'] and not scores['flexible_v3']['correct']:raise ValueError('Strict/v3 scoring invariant failed.')
    reward=0 if capped else int(scores['flexible_v3']['correct'])
    if historical_reward is not None and (type(historical_reward) is not int or historical_reward not in [0,1]):
        raise ValueError('Invalid historical training reward.')
    return dict(group_id=row['id'],draw_id=f"{row['id']}:draw:{draw}",draw_index=draw,origin=origin,seed=seed,
        prompt=row['prompt'],prompt_sha256=_hash(row['prompt']),gold=row['gold'],text=output['text'],token_ids=content,
        generated_token_ids=raw,token_count=len(content),stop_reason=reason,capped=capped,scores=scores,reward_v3=reward,
        historical_training_reward=historical_reward,stop_identity_verified=eos is not None,
        answer_sha256=_hash([row['prompt'],raw]))


def cohorts(records,order,*,selection_seed=42):
    groups={};identities=set()
    for row in records:
        key=(row['group_id'],row['draw_index'])
        if key in identities:raise ValueError('Duplicate eight-draw identity.')
        identities.add(key);groups.setdefault(row['group_id'],[]).append(row)
    if not groups or not set(groups)<=set(order) or len(set(order))!=len(order):raise ValueError('Cohort question order differs.')
    ids=[i for i in order if i in groups];rng=np.random.default_rng(selection_seed)
    all_,correct,one,dropped=[],[],[],[];probabilities={};negative_logs={}
    for identifier in ids:
        rows=sorted(groups[identifier],key=lambda r:r['draw_index'])
        if [r['draw_index'] for r in rows]!=list(range(8)):raise ValueError('Expected exactly eight draws per question.')
        eligible=[r['draw_index'] for r in rows if r['reward_v3']==1]
        all_.extend([identifier,r['draw_index']] for r in rows);correct.extend([identifier,d] for d in eligible)
        if eligible:one.append([identifier,int(rng.choice(eligible))])
        else:dropped.append(identifier)
        probabilities[identifier]=len(eligible)/8
        negative_logs[identifier]=-math.log(len(eligible)/8) if eligible else None
    return dict(gold=ids,own_all=all_,own_correct=correct,own_correct_one=one,dropped=dropped,p_correct=probabilities,
        neg_log_p_correct=negative_logs,selection_seed=selection_seed,selection_order=ids,
        selection_rule='numpy.default_rng/PCG64; source question order, sorted correct draw indices; uniform one per eligible question',
        zero_success_note='Zero of eight observed successes is not zero true correctness probability.',
        units=dict(p_correct='observed k/8',neg_log_p_correct='nats per whole answer; own-correct only'),
        counts=dict(questions=len(ids),answers=len(records),correct_answers=len(correct),eligible_questions=len(one),zero_success_questions=len(dropped)))


def replay_comparison(reference,current):
    fields=['group_id','draw_index','text','token_ids','generated_token_ids','stop_reason','capped','token_count','seed']
    if len(reference)!=16 or len(current)!=16:raise ValueError('Replay requires two recorded questions times eight draws.')
    differences=[]
    for a,b in zip(reference,current):
        delta=[key for key in fields if a[key]!=b[key]]
        if delta:differences.append(dict(draw_id=a['draw_id'],fields=delta))
    return dict(draws=16,exact=not differences,differences=differences,
                criterion='exact token IDs, text, stopping metadata, identities and recorded group seed; no reward-rate tolerance')
