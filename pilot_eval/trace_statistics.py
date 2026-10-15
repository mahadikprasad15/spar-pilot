"""Approved document-level trace inference procedures; no model dependencies."""
import math
import numpy as np


def null_pvalue(observed, null):
    null=np.asarray(null,dtype=np.float64)
    if null.ndim!=1 or not len(null) or not np.isfinite(null).all():
        raise ValueError('The declared null sample must contain finite scores.')
    if not math.isfinite(observed):return dict(p_value=math.nan,null_percentile=math.nan)
    return dict(p_value=float((1+np.count_nonzero(null>=observed))/(1+len(null))),
                null_percentile=float(100*np.mean(null<=observed)))


def c3_pvalue(differences):
    values=np.asarray(differences,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('C3 requires a complete finite paired-bootstrap sample.')
    return float(min(1,2*min((1+np.count_nonzero(values<=0))/(len(values)+1),
                             (1+np.count_nonzero(values>=0))/(len(values)+1))))


def holm(p_values):
    values=np.asarray(p_values,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Holm requires the complete declared family of finite p-values.')
    order=np.argsort(values,kind='stable');result=np.empty(len(values));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,float(values[index]*(len(values)-rank))))
        result[index]=previous
    return result.tolist()


def snr(document_vectors):
    values=np.asarray(document_vectors,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('SNR needs finite per-document vectors.')
    n=len(values)
    if n<2:return math.nan
    mean=values.mean(axis=0)
    se=math.sqrt(float(np.square(values-mean).sum())/(n*(n-1)))
    return float(np.linalg.norm(mean)/se) if se else math.nan


def _examples(examples):
    if not isinstance(examples,dict) or not examples or not all(isinstance(key,str) for key in examples):
        raise ValueError('Supply nonempty explicit document IDs and their vectors.')
    ids=sorted(examples);values=np.asarray([examples[key] for key in ids],dtype=np.float64)
    if values.ndim not in (2,3) or values.ndim==3 and values.shape[1]!=5 or not np.isfinite(values).all():
        raise ValueError('Expected finite document vectors or five positions per document.')
    return ids,values


def _condition(values,scorer,form):
    if form=='pooled':
        pooled=values.mean(axis=1) if values.ndim==3 else values
        return float(scorer(pooled.mean(axis=0)))
    if form=='max-five' and values.ndim==3:
        return max(float(scorer(values[:,index,:].mean(axis=0))) for index in range(5))
    raise ValueError('max-five requires all five per-document positions; unsupported form.')


def bootstrap_scores(examples,scorer,*,draws=1000,seed=20261016,form='pooled'):
    ids,values=_examples(examples)
    if type(draws) is not int or draws<1:raise ValueError('Positive bootstrap draw count required.')
    rng=np.random.default_rng(seed);indices=[];scores=[]
    for _ in range(draws):
        chosen=rng.integers(0,len(ids),size=len(ids));indices.append(chosen.tolist())
        scores.append(_condition(values[chosen],scorer,form))
    interval=np.quantile(scores,[.025,.975],method='linear').tolist() if np.isfinite(scores).all() else [math.nan,math.nan]
    pooled=values.mean(axis=1) if values.ndim==3 else values
    return dict(document_ids=ids,indices=indices,scores=scores,observed=_condition(values,scorer,form),
                interval=interval,snr=snr(pooled),form=form,seed=seed,quantile_method='linear')


def bootstrap_compare(left,right,scorer,*,draws=2000,seed=20261016,form='pooled'):
    ids,a=_examples(left);other,b=_examples(right)
    if ids!=other or a.shape!=b.shape:raise ValueError('Paired bootstrap requires matching document IDs and shapes.')
    if type(draws) is not int or draws<1:raise ValueError('Positive bootstrap draw count required.')
    rng=np.random.default_rng(seed);indices=[];differences=[]
    for _ in range(draws):
        chosen=rng.integers(0,len(ids),size=len(ids));indices.append(chosen.tolist())
        differences.append(_condition(a[chosen],scorer,form)-_condition(b[chosen],scorer,form))
    valid=np.isfinite(differences).all()
    return dict(document_ids=ids,indices=indices,differences=differences,
        observed=_condition(a,scorer,form)-_condition(b,scorer,form),
        p_value=c3_pvalue(differences) if valid else math.nan,
        interval=np.quantile(differences,[.025,.975],method='linear').tolist() if valid else [math.nan,math.nan],
        form=form,seed=seed,quantile_method='linear',interpretation='declared bootstrap-tail statistic; not an exact null test')


def ncurves(examples,scorer,*,sizes=(25,50,100,200),draws=200,seed=20261017,strata,form='pooled'):
    ids,values=_examples(examples)
    if set(strata)!=set(ids):raise ValueError('Every document needs its preserved stratum.')
    if type(draws) is not int or draws<1:raise ValueError('Positive subset draw count required.')
    rng=np.random.default_rng(seed);rows=[]
    from collections import Counter
    for size in sizes:
        if type(size) is not int or not 1<=size<=len(ids):raise ValueError('Invalid subset size.')
        for draw in range(draws):
            chosen=rng.choice(len(ids),size=size,replace=False)
            selected=[ids[index] for index in chosen]
            rows.append(dict(n=size,draw=draw,document_ids=selected,counts=dict(Counter(strata[key] for key in selected)),
                             score=_condition(values[chosen],scorer,form),full_cohort=size==len(ids)))
    return rows
