"""Document-level drift uncertainty and conservative runtime grouping."""
import numpy as np

from pilot_eval.workflow import _hash


def comparison_group(identity,product):
    # GPU type or approximate versions never stand in for a session identity.
    runtime=identity['runtime']
    return _hash(dict(identity,unknown_session_product=None if runtime.get('session_id') else product))


def interval(sums,counts,draws):
    sums,counts=np.asarray(sums,dtype=float),np.asarray(counts)
    if sums.ndim!=1 or sums.shape!=counts.shape or not len(sums) or np.any(counts<=0) or not np.isfinite(sums).all() or draws.ndim!=2 or draws.shape[1]!=len(sums) or not np.issubdtype(draws.dtype,np.integer) or draws.min()<0 or draws.max()>=len(sums):raise ValueError('Matched whole-document sums/counts and resampling indices required.')
    means=sums/counts;replicates=means[draws].mean(1)
    return dict(equal_example_mean=float(means.mean()),token_mean=float(sums.sum()/counts.sum()),
        interval_95=np.percentile(replicates,[2.5,97.5]).tolist(),token_interval_95=np.percentile(sums[draws].sum(1)/counts[draws].sum(1),[2.5,97.5]).tolist(),documents=len(sums),tokens=int(counts.sum()),bootstrap_unit='whole document')


def log_slopes(points):
    """Measured positive points only; zeros never become arbitrary log epsilons."""
    valid=sorted([r for r in points if r['size']>0 and r['kl']>0],key=lambda r:r['step'])
    adjacent=[]
    for a,b in zip(valid,valid[1:]):
        value=None if a['size']==b['size'] else float(np.log(b['kl']/a['kl'])/np.log(b['size']/a['size']))
        adjacent.append(dict(from_step=a['step'],to_step=b['step'],slope=value))
    x=np.log([r['size'] for r in valid]);y=np.log([r['kl'] for r in valid])
    fitted=None if len(valid)<2 or np.ptp(x)==0 else float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
    return dict(fitted_slope=fitted,adjacent_slopes=adjacent,positive_points=len(valid),excluded_nonpositive_steps=[r['step'] for r in points if r['size']<=0 or r['kl']<=0],near_two='unclassified by approval',direction_sensitivity_claim=False,
        interpretation='KL/size² is descriptive. No numerical near-2 criterion or small-update validity claim has been declared.')


def lengths(reference,edited,draws):
    a={r['id']:r for r in reference};b={r['id']:r for r in edited};ids=sorted(a)
    if not ids or set(a)!=set(b) or len(a)!=len(reference) or len(b)!=len(edited) or any(a[i]['prompt_sha256']!=b[i]['prompt_sha256'] for i in ids):raise ValueError('Length comparison requires identical question/prompt identities.')
    before=np.array([a[i]['tokens'] for i in ids]);after=np.array([b[i]['tokens'] for i in ids])
    if np.any(before<0) or np.any(after<0) or draws.shape[1]!=len(ids) or draws.min()<0 or draws.max()>=len(ids):raise ValueError('Length/count/bootstrap mapping changed.')
    delta=after-before
    return dict(n=len(ids),ids=ids,median_paired_change=float(np.median(delta)),interval_95=np.percentile(np.median(delta[draws],axis=1),[2.5,97.5]).tolist(),
        shorter_share=float(np.mean(delta<0)),reference_cap_rate=float(np.mean([a[i]['capped'] for i in ids])),edited_cap_rate=float(np.mean([b[i]['capped'] for i in ids])),
        statistic='median of paired content-token changes; not difference of marginal medians',reasoning_claim=False)
