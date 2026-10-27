"""FP64 sufficient energies on untuned inputs; signed interference is retained."""
import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.stage0_drift_math import positions

FIELDS=['parallel_squared_sum','rest_squared_sum','twice_cross_sum','total_squared_sum']


def parallel_factors(a,basis):
    a=np.asarray(a,dtype=np.float64);basis=np.asarray(basis,dtype=np.float64)
    if a.ndim!=2 or basis.ndim!=2 or a.shape[1]!=basis.shape[0] or not np.isfinite(a).all() or not np.isfinite(basis).all():raise ValueError('Finite input factors and compatible principal basis required.')
    return (a@basis)@basis.T


def energy_sums(x,a,parallel):
    x,a,parallel=(np.asarray(v,dtype=np.float64) for v in [x,a,parallel])
    if x.ndim!=2 or a.shape!=(x.shape[1],) or parallel.shape!=a.shape or any(not np.isfinite(v).all() for v in [x,a,parallel]):raise ValueError('Invalid FP64 input-energy dimensions/finiteness.')
    p=x@parallel;r=x@(a-parallel);total=x@a
    return np.array([p@p,r@r,2*(p@r),total@total],dtype=np.float64)


def view_sums(x,a,parallel,row):
    positions([row]);x=np.asarray(x,dtype=np.float64)
    if x.shape!=(len(row['input_ids']),a.shape[1]) or parallel.shape[0]!=len(a) or parallel.shape[2]!=a.shape[1]:raise ValueError('Hooked inputs differ from the frozen token/factor mapping.')
    result=np.zeros((len(a),parallel.shape[1],len(VIEWS),4),dtype=np.float64)
    for vi,view in enumerate(VIEWS):
        selected=x[np.array(row['masks'][view],dtype=bool)]
        for fi in range(len(a)):
            for li in range(parallel.shape[1]):result[fi,li,vi]=energy_sums(selected,a[fi],parallel[fi,li])
    return result


def ratios(sums):
    p,r,c,t=map(float,sums);parts=p+r
    return dict(principal_fraction=None if parts==0 else p/parts,
        parallel_over_total=None if t==0 else p/t,rest_over_total=None if t==0 else r/t,
        cross_over_total=None if t==0 else c/t,interference_over_parts=None if parts==0 else c/parts,
        zero_total=t==0,zero_parts=parts==0)


def validate(arrays,rows,config):
    shape=(len(rows),len(config['factor_labels']),len(config['energy_levels']),len(VIEWS),len(config['modules']),4)
    if set(arrays)!={'sums'} or arrays['sums'].shape!=shape or arrays['sums'].dtype!=np.float64 or not np.isfinite(arrays['sums']).all():raise ValueError('Energy sufficient-sum shape/precision/finiteness changed.')
    values=arrays['sums'];p,r,c,t=(values[...,i] for i in range(4))
    if any(np.any(v<0) for v in [p,r,t]):raise ValueError('Squared energy cannot be negative.')
    if not np.allclose(p+r+c,t,**config['agreement']):raise ValueError('Principal/rest/cross reconstruction gate failed; preserve diagnostics.')
    # Cauchy-Schwarz is a mathematical consistency condition, not an effect-size gate.
    if not np.allclose(np.minimum(c*c,4*p*r),c*c,**config['agreement']):raise ValueError('Cross term exceeds Cauchy-Schwarz allowance.')
    for ei,row in enumerate(rows):
        positions([row])
        for vi,view in enumerate(VIEWS):
            if not any(row['masks'][view]) and np.any(values[ei,:,:,vi]):raise ValueError('Uncounted token view has nonzero energy.')
    # Total uses the same unprojected factor at every cutoff.
    if not np.allclose(t,t[:,:,0:1],**config['agreement']):raise ValueError('Unprojected total changes with the principal cutoff.')
    return arrays
