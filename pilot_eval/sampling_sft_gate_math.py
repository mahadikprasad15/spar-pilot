"""CPU descriptive estimands; one eligible answer is one problem observation."""
import math
import statistics


def quantile(values,p):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*p;lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def summarize_rows(rows):
    if len({r['problem_id'] for r in rows})!=len(rows):raise ValueError('One selected answer per problem required.')
    nll=[]
    for row in rows:
        values=row['logprobs']
        if (not values or len(values)!=len(row['response_ids'])
                or any(not math.isfinite(x) or x>0 for x in values)):
            raise ValueError('Missing, nonfinite or misaligned token log probabilities.')
        nll.append(-math.fsum(values)/len(values))
    count=len(rows);lengths=[len(r['response_ids']) for r in rows]
    overlap=[r['word4_overlap'] for r in rows if r['word4_overlap'] is not None]
    marker=sum(r['calculator_marker'] for r in rows)
    return dict(n_answers=count,n_tokens=sum(lengths),nll=statistics.mean(nll) if nll else None,
        length_median=statistics.median(lengths) if lengths else None,length_p90=quantile(lengths,0.9),
        calculator_marker=dict(numerator=marker,denominator=count,rate=marker/count if count else None),
        word4_overlap=dict(mean=statistics.mean(overlap) if overlap else None,defined_answers=len(overlap),
                           undefined_answers=count-len(overlap)),
        nll_units='nats per response token, actual stop included; equal-problem mean',
        length_units='response tokens, actual stop included')


def paired_bootstrap(left,right,metric):
    import numpy as np
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    if (left.ndim!=1 or right.ndim!=1 or left.shape!=right.shape or metric not in ['length','nll']
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))):
        raise ValueError('Bootstrap needs aligned finite whole-problem vectors and a declared metric.')
    result=dict(metric=metric,n_problems=len(left),resamples=10000,seed=42,
        generator='numpy.default_rng PCG64',interval='percentile95; linear quantile',
        estimand='difference of arm medians' if metric=='length' else 'difference of equal-problem means')
    if not len(left):return dict(**result,status='undefined-empty-common-cohort',estimate=None,ci_low=None,ci_high=None)
    reduce=np.median if metric=='length' else np.mean
    rng=np.random.default_rng(42);samples=[]
    for start in range(0,10000,128):
        indices=rng.integers(0,len(left),size=(min(128,10000-start),len(left)))
        samples.extend((reduce(left[indices],axis=1)-reduce(right[indices],axis=1)).tolist())
    return dict(**result,status='computed',estimate=float(reduce(left)-reduce(right)),
                ci_low=float(np.quantile(samples,0.025)),ci_high=float(np.quantile(samples,0.975)))
