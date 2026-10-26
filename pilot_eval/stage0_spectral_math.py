"""Float64 spectral computations; retained bases are checked again after storage conversion."""
import numpy as np

LEVELS = (30, 50, 70)


def decompose(matrix):
    w = np.asarray(matrix, dtype=np.float64)
    if w.ndim != 2 or min(w.shape) < 40 or not np.isfinite(w).all():
        raise ValueError('Expected a finite matrix with at least 40 singular directions.')
    norm = np.linalg.norm(w)
    if norm == 0:
        raise ValueError('Zero base-weight spectrum has undefined energy ranks.')
    u, sigma, vt = np.linalg.svd(w, full_matrices=False)
    error = float(np.linalg.norm(w - (u*sigma)@vt)/norm)
    if not error < 1e-6:
        raise ValueError(f'SVD reconstruction gate failed: {error}.')
    full_orth=max(float(np.max(np.abs(b.T@b-np.eye(len(sigma))))) for b in [u,vt.T])
    if full_orth>1e-6:
        raise ValueError(f'Full-SVD orthogonality gate failed: {full_orth}.')
    energy = np.cumsum(sigma**2)
    ranks = {f'k{level}': min(len(sigma), int(np.searchsorted(energy, energy[-1]*level/100)+1))
             for level in LEVELS}
    retained = max(40, ranks['k70'])
    left = u[:,:retained].astype(np.float32)
    right = vt[:retained,:].T.astype(np.float32)
    orth = max(float(np.max(np.abs(b.astype(np.float64).T@b-np.eye(retained)))) for b in [left,right])
    if orth > 1e-6:
        raise ValueError(f'Retained-basis orthogonality gate failed: {orth}.')
    return dict(sigma=sigma, u=left, v=right), dict(**ranks, d_out=w.shape[0], d_in=w.shape[1],
        retained_columns=retained, reconstruction_relative_error=error,
        full_orthogonality_error=full_orth,retained_orthogonality_error=orth,
        energy_top1=float(sigma[0]**2/energy[-1]),
        gaps={str(level): float(sigma[k-1]-sigma[k]) if k<len(sigma) else None
              for level,k in [(level,ranks[f'k{level}']) for level in LEVELS]})


def projection(vector, basis):
    value = np.asarray(vector,dtype=np.float64).reshape(-1)
    basis = np.asarray(basis,dtype=np.float64)
    if value.shape != (basis.shape[0],) or not np.isfinite(value).all():
        raise ValueError('Factor/basis dimensions or finite-value contract differ.')
    norm2=float(value@value)
    return None if norm2==0 else float(np.sum((basis.T@value)**2)/norm2)


def chance_band(basis):
    # A fixed local generator; no global RNG mutation or retry-until-pass.
    rng=np.random.default_rng(42)
    vectors=rng.normal(size=(100,basis.shape[0]))
    values=np.sum((vectors@basis.astype(np.float64))**2,axis=1)/np.sum(vectors**2,axis=1)
    expected=basis.shape[1]/basis.shape[0]
    mean=float(values.mean())
    if abs(mean-expected)>0.1*expected:
        raise ValueError(f'100-Gaussian chance gate failed: mean={mean}, chance={expected}.')
    return dict(mean=mean,p05=float(np.percentile(values,5)),p95=float(np.percentile(values,95)),
                chance=expected,draws=100,seed=42,generator='numpy-PCG64')
