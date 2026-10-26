"""Rank-one intervention math, keeping saved factors in their original gauge."""
import numpy as np


def _norm(a,b):
    return float(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))


def _remove(vector,basis):
    # k=0 is an exact identity including original FP32 bytes.
    if basis.shape[1]==0:return vector.copy()
    flat=vector.astype(np.float64).reshape(-1)
    q=basis.astype(np.float64)
    return (flat-q@(q.T@flat)).reshape(vector.shape).astype(np.float32)


def _matched(a,b,target):
    original=_norm(a,b)
    if original==0:
        if target!=0:raise ValueError('Cannot scale a zero update to nonzero magnitude.')
        return a.copy(),b.copy()
    return a.copy(),(b.astype(np.float64)*(target/original)).astype(np.float32)


def edit_module(a,b,u,v):
    """Return factor edits, checks and exact random bases; no trained-overlap report."""
    if (a.dtype!=np.float32 or b.dtype!=np.float32 or a.ndim!=2 or b.ndim!=2
            or a.shape[0]!=1 or b.shape[1]!=1 or u.shape[0]!=b.shape[0]
            or v.shape[0]!=a.shape[1] or u.shape[1]!=v.shape[1]
            or not all(np.isfinite(x).all() for x in [a,b,u,v])):
        raise ValueError('Invalid edit factors or principal bases.')
    k=u.shape[1]
    for q in [u,v]:
        if k and np.max(np.abs(q.astype(np.float64).T@q-np.eye(k)))>1e-6:
            raise ValueError('Edit basis orthogonality gate failed.')
    output={'V1':(a.copy(),_remove(b,u)),'V2':(_remove(a,v),b.copy())}
    bases={}
    for draw in range(3):
        seed=1000+draw
        if k:
            rng=np.random.default_rng(seed)
            q,_=np.linalg.qr(rng.normal(size=(b.shape[0],k)),mode='reduced')
            q=q.astype(np.float32)
        else:q=np.zeros((b.shape[0],0),dtype=np.float32)
        bases[f'V3-{draw}']=q
        output[f'V3-{draw}']=(a.copy(),_remove(b,q))
    output['V4']=_matched(a,b,_norm(*output['V1']))
    output['V4-input']=_matched(a,b,_norm(*output['V2']))
    checks={}
    for variant,(va,vb) in output.items():
        if not np.isfinite(va).all() or not np.isfinite(vb).all():
            raise ValueError('Nonfinite edited factor.')
        checks[variant]=dict(update_norm=_norm(va,vb),a_norm=float(np.linalg.norm(va.astype(np.float64))),
                             b_norm=float(np.linalg.norm(vb.astype(np.float64))))
    for variant,vector,q in [('V1',output['V1'][1],u),('V2',output['V2'][0],v)]+[
            (key,output[key][1],q) for key,q in bases.items()]:
        norm=float(np.linalg.norm(vector.astype(np.float64)))
        residual=float(np.linalg.norm(q.astype(np.float64).T@vector.reshape(-1).astype(np.float64)))
        checks[variant]['removed_component_norm']=residual
        checks[variant]['removed_component_relative']=None if norm==0 else residual/norm
        if residual>1e-6:
            raise ValueError(f'{variant} removed-component gate failed: {residual}.')
    for control,target in [('V4','V1'),('V4-input','V2')]:
        goal=checks[target]['update_norm'];actual=checks[control]['update_norm']
        error=abs(actual-goal)
        checks[control]['matched_variant']=target
        checks[control]['matched_norm_absolute_error']=error
        if not np.isclose(actual,goal,rtol=1e-6,atol=0):
            raise ValueError('Per-module magnitude control gate failed.')
    return output,checks,bases
