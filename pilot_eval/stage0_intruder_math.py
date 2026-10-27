"""Exact CPU updated-weight SVD and basis-invariant subspace diagnostics."""
import numpy as np

LEVELS = (30, 50, 70)
THRESHOLDS = (.3, .4, .5, .6, .7)


def _basis(value):
    basis=np.asarray(value,dtype=np.float64)
    if basis.ndim!=2 or min(basis.shape)<1 or basis.shape[1]>basis.shape[0] or not np.isfinite(basis).all():
        raise ValueError('Invalid subspace basis dimensions or values.')
    error=float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1]))))
    if error>1e-6:raise ValueError('Subspace basis exceeds inherited 1e-6 orthogonality gate.')
    # Orthonormalize the retained FP32 columns for subspace-only calculations.
    return np.linalg.qr(basis,mode='reduced')[0]


def principal_angles(old,new):
    left,right=_basis(old),_basis(new)
    if left.shape!=right.shape:raise ValueError('Compare equal-dimension subspaces in the same ambient space.')
    # Singular values are cosines; clipping only addresses bounded cosine roundoff.
    cosines=np.linalg.svd(left.T@right,compute_uv=False)
    return np.arccos(np.clip(cosines,0.,1.))


def overlap_norms(old,candidates):
    basis=_basis(old);candidates=np.asarray(candidates,dtype=np.float64)
    if candidates.ndim!=2 or candidates.shape[0]!=basis.shape[0] or not np.isfinite(candidates).all():
        raise ValueError('Candidate/output subspace mapping differs.')
    if np.max(np.abs(np.sum(candidates*candidates,axis=0)-1))>1e-6:
        raise ValueError('Candidate directions are not unit vectors.')
    return np.linalg.norm(basis.T@candidates,axis=0)


def ratio(numerator,denominator):
    if not np.isfinite([numerator,denominator]).all() or min(numerator,denominator)<0:
        raise ValueError('Spectral ratio requires finite nonnegative terms.')
    if denominator==0:
        return dict(value=None,status='infinite-positive-over-zero' if numerator>0 else 'undefined-zero-over-zero')
    return dict(value=float(numerator/denominator),status='finite')


def classify(overlaps,*,chance,threshold):
    values=np.asarray(overlaps,dtype=np.float64)
    if (values.ndim!=1 or not np.isfinite(values).all() or np.any(values<0)
            or not np.isfinite([chance,threshold]).all() or not 0<=chance<=1 or not 0<threshold<1):
        raise ValueError('Invalid intruder overlap/chance/threshold.')
    raw=np.flatnonzero(values<threshold).tolist()
    decidable=threshold>chance
    return dict(threshold=float(threshold),chance_norm=float(chance),raw_below_threshold_indices=raw,
        status='undecidable-at-or-below-chance' if not decidable else ('eligible-intruders' if raw else 'no-intruders'),
        eligible_indices=raw if decidable else [])


def _boundary(sigma,k):
    if k==len(sigma):return dict(gap=None,status='no-next-singular-value',numerically_degenerate=None)
    gap=float(sigma[k-1]-sigma[k])
    # A descriptive machine-precision indicator, not an edit-acceptance tolerance.
    scale=float(np.max(sigma))
    precision_bound=float(np.finfo(np.float64).eps*len(sigma)*scale)
    return dict(gap=gap,status='exact-degeneracy' if gap==0 else 'positive-gap',
        numerically_degenerate=bool(gap<=precision_bound),fp64_resolution_indicator=precision_bound)


def diagnostics(a,b,old,new,ranks,*,zero_update=False):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    sigma=np.asarray(old['sigma'],dtype=np.float64);updated=np.asarray(new['new_sigma'],dtype=np.float64)
    output,input_=ranks['d_out'],ranks['d_in'];rank=min(output,input_)
    if (a.shape!=(1,input_) or b.shape!=(output,1) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or len(sigma)!=rank or len(updated)!=rank or rank<40
            or old['u'].shape[0]!=output or old['v'].shape[0]!=input_
            or new['new_u'].shape[0]!=output or new['new_v'].shape[0]!=input_
            or not np.isfinite(sigma).all() or not np.isfinite(updated).all()
            or np.any(sigma<0) or np.any(updated<0) or np.any(np.diff(sigma)>0) or np.any(np.diff(updated)>0)):
        raise ValueError('Updated/base SVD and factor dimensions or spectrum invalid.')
    norm=float(np.linalg.norm(a)*np.linalg.norm(b));levels={}
    candidate_u=new['new_u'][:,:40]
    overlaps={}
    for level in LEVELS:
        k=ranks[f'k{level}'];m=max(k,40)
        if type(k) is not int or not 1<=k<=rank or m>rank or min(old['u'].shape[1],old['v'].shape[1],new['new_u'].shape[1],new['new_v'].shape[1])<m:
            raise ValueError('Requested old/new principal and floored intruder subspaces are unavailable; clipping forbidden.')
        left=principal_angles(old['u'][:,:k],new['new_u'][:,:k])
        right=principal_angles(old['v'][:,:k],new['new_v'][:,:k])
        if zero_update:left=np.zeros(k);right=np.zeros(k)
        overlap=overlap_norms(old['u'][:,:m],candidate_u)
        overlaps[str(level)]=overlap
        chance=float(np.sqrt(m/output));boundary=_boundary(sigma,k);new_boundary=_boundary(updated,k)
        levels[str(level)]=dict(k=k,m=m,chance_norm=chance,base_boundary=boundary,updated_boundary=new_boundary,
            intruder_base_boundary=_boundary(sigma,m),
            update_over_sigma=ratio(norm,float(sigma[k-1])),
            update_over_gap=ratio(norm,boundary['gap']) if boundary['gap'] is not None else dict(value=None,status='undefined-no-next-singular-value'),
            rotation=dict(left_angles_radians=left.tolist(),right_angles_radians=right.tolist(),
                max_left_degrees=float(np.degrees(left).max()),max_right_degrees=float(np.degrees(right).max()),
                unique_boundary=not (boundary.get('numerically_degenerate') or new_boundary.get('numerically_degenerate')),
                limitation='Tied or numerically unresolved boundary subspaces are not uniquely selected; within-subspace mixing is ignored.'),
            threshold_sweep={str(t):classify(overlap[:20],chance=chance,threshold=t) for t in THRESHOLDS},
            control_threshold_sweep={str(t):classify(overlap,chance=chance,threshold=t) for t in THRESHOLDS})
    candidates=[]
    resolution=float(np.finfo(np.float64).eps*rank*float(updated[0]))
    for i in range(40):
        left_gap=float(updated[i-1]-updated[i]) if i else None
        right_gap=float(updated[i]-updated[i+1]) if i+1<rank else None
        candidates.append(dict(index=i,rank=i+1,sigma=float(updated[i]),gap_before=left_gap,gap_after=right_gap,
            numerically_degenerate=bool((left_gap is not None and left_gap<=resolution) or (right_gap is not None and right_gap<=resolution)),
            overlap_norm={str(level):float(overlaps[str(level)][i]) for level in LEVELS},
            chance_norm={str(level):levels[str(level)]['chance_norm'] for level in LEVELS},
            detection_candidate=i<20,primary_below_threshold=bool(overlaps['50'][i]<.5),
            primary_control_candidate=bool(overlaps['50'][i]>=.5 and .5>levels['50']['chance_norm'])))
    primary=levels['50']['threshold_sweep']['0.5']
    return dict(update_operator_norm=norm,update_frobenius_norm=norm,
        update_representation='rank-one alpha-one b a-transpose; operator and Frobenius norms agree',
        zero_update=bool(zero_update),small_update_skip=False,levels=levels,candidates=candidates,
        detection_indices=list(range(20)),control_candidate_indices=[r['index'] for r in candidates if r['primary_control_candidate']],
        primary=primary,
        degeneracy_note='Vector flags inside tied singular clusters depend on the SVD basis. Raw flags follow the approved rule; they do not establish stable individual directions.')


def measure(matrix,a,b,old,ranks):
    w=np.asarray(matrix,dtype=np.float64);a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if w.shape!=(ranks['d_out'],ranks['d_in']) or a.shape!=(1,w.shape[1]) or b.shape!=(w.shape[0],1):
        raise ValueError('Base matrix and rank-one factors differ in dimensions.')
    if not all(np.isfinite(x).all() for x in [w,a,b]):raise ValueError('Nonfinite geometry input.')
    zero=not np.any(a) or not np.any(b)
    updated=w+b@a
    u,sigma,vt=np.linalg.svd(updated,full_matrices=False)
    norm=float(np.linalg.norm(updated))
    error=float(np.linalg.norm(updated-(u*sigma)@vt)/norm) if norm else 0.
    if error>=1e-6:raise ValueError('Updated full-SVD reconstruction exceeds inherited gate.')
    orth=max(float(np.max(np.abs(q.T@q-np.eye(len(sigma))))) for q in [u,vt.T])
    if orth>1e-6:raise ValueError('Updated full-SVD orthogonality exceeds inherited gate.')
    keep=max(40,*[ranks[f'k{level}'] for level in LEVELS])
    new=dict(new_sigma=sigma,new_u=u[:,:keep].astype(np.float32),new_v=vt[:keep,:].T.astype(np.float32))
    retained=max(float(np.max(np.abs(q.astype(float).T@q-np.eye(keep)))) for q in [new['new_u'],new['new_v']])
    if retained>1e-6:raise ValueError('Updated retained-basis orthogonality exceeds inherited gate.')
    row=diagnostics(a,b,old,new,ranks,zero_update=zero)
    row['decomposition']=dict(full_reconstruction_relative_error=error,full_orthogonality_error=orth,
        retained_orthogonality_error=retained,retained_columns=keep,full_reconstruction_defined=norm!=0,
        all_singular_values_saved=True,full_matrix_saved=False,
        zero_matrix_note='Zero updated matrix has nonunique singular bases.' if norm==0 else None)
    return new,row
