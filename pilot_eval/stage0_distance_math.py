"""Unique answer mapping and full-vocabulary teacher-forced statistics."""
import numpy as np

from pilot_eval.workflow import _hash


def token_statistics(logits,targets):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(targets)
    if x.ndim!=2 or not x.size or y.shape!=(len(x),) or not np.issubdtype(y.dtype,np.integer) or np.any(y<0) or np.any(y>=x.shape[1]) or not np.isfinite(x).all():
        raise ValueError('Expected finite full-vocabulary logits and valid integer target IDs.')
    shifted=x-x.max(-1,keepdims=True);lp=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
    return dict(surprise=-lp[np.arange(len(y)),y],entropy=-(np.exp(lp)*lp).sum(-1),
                rank=1+(x>x[np.arange(len(y)),y,None]).sum(-1))


def prepare_answers(training,draws,datasets,eos):
    answers={};mapping={k:[] for k in ['gold','own_all','own_correct','own_correct_one']};by_id={r['id']:r for r in training}
    def add(row,targets,label):
        prompt=row['input_ids'][:row['prompt_tokens']]
        if not prompt or not targets or any(type(t) is not int or t<0 for t in prompt+targets):raise ValueError('Invalid answer token mapping.')
        identifier=_hash([prompt,targets]);value=dict(answer_id=identifier,problem_id=row['id'],prompt=row['prompt'],
            input_ids=prompt+targets,prompt_tokens=len(prompt),target_ids=targets,positions=list(range(len(prompt)-1,len(prompt)+len(targets)-1)),
            answer_length=len(targets),example_label=label)
        if identifier not in answers:answers[identifier]=value
        return identifier
    for row in training:
        boundary=row['prompt_tokens'];targets=row['input_ids'][boundary:]
        if row['labels']!=[-100]*boundary+targets or not targets or targets[-1]!=eos or eos in targets[:-1]:raise ValueError('Gold SFT answer mask/end-turn contract differs.')
        mapping['gold'].append(dict(problem_id=row['id'],answer_id=add(row,targets,'gold'),draw_index=None))
    refs={}
    for r in draws:
        targets=r['generated_token_ids'];capped=r['capped']
        if (r['stop_reason']=='eos' and (not targets or targets[-1]!=eos or eos in targets[:-1])) or (capped and eos in targets):
            raise ValueError('Sampled stopping token treatment differs.')
        key=(r['group_id'],r['draw_index'])
        if key in refs:raise ValueError('Duplicate answer membership identity.')
        refs[key]=dict(problem_id=r['group_id'],answer_id=add(by_id[r['group_id']],targets,'draw'),draw_index=r['draw_index'])
    for name in ['own_all','own_correct','own_correct_one']:
        mapping[name]=[refs[tuple(key)] for key in datasets[name]]
    if [r['problem_id'] for r in mapping['gold']]!=datasets['gold']:raise ValueError('Gold membership order changed.')
    return list(answers.values()),mapping


def check_arrays(arrays,answer):
    if set(arrays)!={'surprise','entropy','rank'}:raise ValueError('Per-token score fields changed.')
    n=len(answer['target_ids'])
    if any(np.asarray(a).shape!=(n,) for a in arrays.values()) or any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Per-token score mapping/finiteness mismatch.')
    if arrays['surprise'].dtype!=np.float64 or arrays['entropy'].dtype!=np.float64 or not np.issubdtype(arrays['rank'].dtype,np.integer):raise ValueError('Token score precision differs.')
    if np.any(arrays['surprise']<0) or np.any(arrays['entropy']<0) or np.any(arrays['rank']<1):raise ValueError('Invalid token surprise, entropy or rank.')
    return arrays


def position_windows(size):
    """Disjoint summaries; first32 takes precedence for sequences shorter than64."""
    if type(size) is not int or size<1:raise ValueError('Positive answer length required.')
    return ['first32' if i<32 else ('last32' if i>=size-32 else 'middle') for i in range(size)]
