"""Gold-independent fixed-context KL and explicitly shifted view membership."""
import numpy as np
from pilot_eval.activation_prepare import VIEWS


def prediction_positions(rows):
    """A view belongs to target t; paired logits come from context t-1."""
    result=[]
    for example,row in enumerate(rows):
        size=len(row['input_ids'])
        if row['attention_mask'] != [1]*size or any(len(row['masks'][v])!=size for v in VIEWS):
            raise ValueError('fixed token/mask alignment mismatch')
        for target in range(1,size):
            views=[v for v in VIEWS if row['masks'][v][target]]
            if len(views)>1: raise ValueError('overlapping target views')
            if views:
                result.append(dict(example=example,context=target-1,target=target,
                                   view=views[0],token_id=row['input_ids'][target]))
    return result


def full_vocabulary_kl(tuned_logits,untuned_logits):
    """CPU known-answer seam; stable FP64 KL(tuned || untuned), in nats."""
    p,q=np.asarray(tuned_logits,dtype=np.float64),np.asarray(untuned_logits,dtype=np.float64)
    if p.ndim!=2 or p.shape!=q.shape or p.shape[1]<2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('invalid paired full-vocabulary logits')
    def log_prob(x):
        x=x-x.max(axis=-1,keepdims=True)
        return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
    lp,lq=log_prob(p),log_prob(q)
    return (np.exp(lp)*(lp-lq)).sum(axis=-1)


def summarize_kl(positions,values,example_ids):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(len(positions),) or not np.isfinite(values).all():
        raise ValueError('KL position/value identity mismatch')
    result=[]
    for view in VIEWS:
        per=[]
        for example,identifier in enumerate(example_ids):
            selected=values[[i for i,p in enumerate(positions) if p['example']==example and p['view']==view]]
            if len(selected): per.append(dict(id=identifier,count=len(selected),sum=float(selected.sum()),mean=float(selected.mean())))
        tokens=sum(r['count'] for r in per)
        result.append(dict(view=view,tokens=tokens,examples=len(per),
            token_mean=sum(r['sum'] for r in per)/tokens if tokens else None,
            example_mean=sum(r['mean'] for r in per)/len(per) if per else None,
            per_example=per,undefined_reason=None if tokens else 'no_target_contexts'))
    return result
