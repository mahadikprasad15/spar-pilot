"""Rank-based trace scoring. Real readouts must come from qualified later stages."""
import math
import numpy as np
from pilot_eval.trace_vocabulary import load_references


def _ranks(values):
    order=np.argsort(values,kind='stable');ranks=np.empty(len(values),dtype=np.float64)
    start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and values[order[stop]]==values[order[start]]:stop+=1
        ranks[order[start:stop]]=(start+stop-1)/2
        start=stop
    return ranks


def _spearman(left,right):
    if len(left)<2:return math.nan
    a,b=_ranks(np.asarray(left)),_ranks(np.asarray(right))
    a-=a.mean();b-=b.mean();denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator else math.nan


def score_vector(vector, readout, inventory, *, references=None, freeze_path=None, synthetic=False):
    """Score one vector; synthetic bypass is explicit and recorded in the result.

    The production command consumer must also qualify the readout/capture source.
    This low-level function does not establish lens compatibility from a callback.
    """
    inference_status='synthetic-only'
    if not synthetic:
        if freeze_path is None:raise ValueError('A valid reference freeze is required before scoring.')
        if references is not None:raise ValueError('Production references must come only from the verified freeze.')
        payload=load_references(freeze_path,require_review=True)
        references=payload['references'];inference_status=payload['review']['inference_status']
    if not references:raise ValueError('No reference sets supplied.')
    vector=np.asarray(vector,dtype=np.float64)
    if vector.ndim!=1 or not np.isfinite(vector).all():raise ValueError('Vector must be finite and one-dimensional.')
    norm=float(np.linalg.norm(vector))
    if norm==0:return dict(state='undefined',reason='zero-vector',norm=0.,synthetic=synthetic,inference_status=inference_status)
    logits=np.asarray(readout(vector),dtype=np.float64)
    ids=sorted(inventory.decoded)
    if logits.shape!=(len(ids),) or not np.isfinite(logits).all():
        raise ValueError('Readout must provide one finite score per pinned token ID, in sorted-ID order.')
    order=sorted(range(len(ids)),key=lambda i:(-logits[i],ids[i]))
    chosen=[];seen=set()
    for index in order[:1000]:
        value=inventory.normalize(ids[index])
        if value is None or value in seen:continue
        chosen.append(dict(token_id=ids[index],decoded_repr=repr(inventory.decoded[ids[index]]),string=value))
        seen.add(value)
        if len(chosen)==20:break
    strings=[row['string'] for row in chosen];count=len(strings);values={}
    collapsed={}
    for index,token_id in enumerate(ids):
        value=inventory.normalize(token_id)
        if value is not None:collapsed[value]=max(collapsed.get(value,-math.inf),float(logits[index]))
    for name,reference in references.items():
        score={field:len(set(strings)&set(reference[key]))/count if count else math.nan
            for field,key in [('overlap','strings'),('overlap_word','word'),('overlap_format','format_number')]}
        support=[row for row in reference['scores'] if row['y_a']+row['y_b']>=5 and row['string'] in collapsed]
        score['soft_rank']=_spearman([collapsed[row['string']] for row in support],[row['z'] for row in support])
        values[name]=score
    return dict(state='scored' if count else 'undefined',reason=None if count else 'no-valid-strings',
                scores=values,strings=strings,list_ids=[row['token_id'] for row in chosen],
                raw_tokens=chosen,valid_count=count,short_list=count<20,norm=norm,synthetic=synthetic,inference_status=inference_status)


def random_vectors(draws, dimension, *, seed=20261015, positions=1, covariance=None):
    """Declared unit-vector null; covariance is supplied only by qualified capture."""
    if any(type(n) is not int or n<1 for n in (draws,dimension,positions)):
        raise ValueError('Positive integer draw, dimension and position counts required.')
    vectors=np.random.default_rng(seed).standard_normal((draws,positions,dimension))
    if covariance is not None:
        covariance=np.asarray(covariance,dtype=np.float64)
        if (covariance.shape!=(dimension,dimension) or not np.isfinite(covariance).all()
                or not np.array_equal(covariance,covariance.T)):
            raise ValueError('Invalid qualified covariance shape/symmetry.')
        values,axes=np.linalg.eigh(covariance)
        if np.any(values<0) or not np.any(values>0):
            raise ValueError('Invalid covariance spectrum; no silent PSD repair.')
        vectors=vectors@(axes*np.sqrt(values)).T
    norms=np.linalg.norm(vectors,axis=-1,keepdims=True)
    if np.any(norms==0) or not np.isfinite(norms).all():raise ValueError('Undefined null direction.')
    return vectors/norms


def null_scores(scorer, *, dimension, draws=1000, seed=20261015, positions=1, covariance=None):
    """Apply the same max-over-positions rule as the secondary learned statistic.

    scorer accepts one vector and returns a name-to-number mapping. Real lens and
    freeze qualification belongs to the enclosing saved-workflow operation.
    """
    vectors=random_vectors(draws,dimension,seed=seed,positions=positions,covariance=covariance)
    result={}
    for sample in vectors:
        values=[scorer(vector) for vector in sample]
        if any(set(row)!=set(values[0]) for row in values):raise ValueError('Null score names changed.')
        for name in values[0]:
            result.setdefault(name,[]).append(float(np.max([row[name] for row in values])))
    return {name:np.asarray(values,dtype=np.float64) for name,values in result.items()}


def check_zero_adapter(delta):
    """Exact step-zero assertion; no numerical allowance is substituted."""
    values=np.asarray(delta)
    if not values.size or not np.isfinite(values).all() or np.any(values!=0):
        raise ValueError('Step-0 adapter delta must be finite and exactly zero.')
    return dict(exact_zero=True,shape=list(values.shape),action='skip-undefined-readout')
