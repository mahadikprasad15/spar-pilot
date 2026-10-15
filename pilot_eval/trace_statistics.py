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
