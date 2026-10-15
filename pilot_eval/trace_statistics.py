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
            subset=values[chosen]
            pooled=subset.mean(axis=1) if subset.ndim==3 else subset
            rows.append(dict(n=size,draw=draw,document_ids=selected,counts=dict(Counter(strata[key] for key in selected)),
                             score=_condition(subset,scorer,form),snr=snr(pooled),
                             norm=float(np.linalg.norm(pooled.mean(axis=0))),
                             position_snr=[snr(subset[:,i,:]) for i in range(5)] if subset.ndim==3 else None,
                             full_cohort=size==len(ids)))
    return rows


def run_draws(directory,config,unit_ids,compute,*,chunk_size=25):
    """Saved-workflow runner for costly null/resampling units; seals whole chunks.

    config must bind the vector, lens, reference and procedure identities at the
    calling stage. Unsealed chunk work may repeat; sealed rows never recompute.
    """
    import json
    from pathlib import Path
    from pilot_eval.training import file_hash,run_lock
    from pilot_eval.workflow import _save_frozen
    from pilot_eval.run import _write_json,_write_state
    directory=Path(directory).resolve()
    if type(chunk_size) is not int or chunk_size<1 or len(set(unit_ids))!=len(unit_ids):
        raise ValueError('Distinct draw IDs and positive chunk size required.')
    identity=dict(parameters=config,unit_ids=unit_ids,chunk_size=chunk_size)
    with run_lock(directory):
        _save_frozen(directory/'config.json',identity)
        complete=directory/'complete.json'
        if complete.exists():
            for relative,expected in json.loads(complete.read_text())['files'].items():
                if file_hash(directory/relative)!=expected:raise ValueError('Draw payload hash mismatch.')
            return json.loads((directory/'results/results.json').read_text())
        chunks=directory/'chunks';chunks.mkdir(exist_ok=True);rows=[]
        try:
            for start in range(0,len(unit_ids),chunk_size):
                path=chunks/f'{start:06d}.json';seal=path.with_suffix('.seal.json')
                selected=unit_ids[start:start+chunk_size]
                if seal.exists():
                    if file_hash(path)!=json.loads(seal.read_text())['sha256']:
                        raise ValueError('Draw chunk hash mismatch.')
                    block=json.loads(path.read_text())
                    if block['unit_ids']!=selected:raise ValueError('Draw chunk identities changed.')
                else:
                    block=dict(unit_ids=selected,rows=[compute(unit) for unit in selected])
                    _write_json(path,block);_write_json(seal,{'sha256':file_hash(path)})
                rows.extend(block['rows'])
                _write_state(directory,'running',len(rows),len(unit_ids))
                print(f'Resampling: {len(rows)}/{len(unit_ids)} durable units.',flush=True)
            _write_json(directory/'results/results.json',rows)
            paths=[directory/'config.json',directory/'results/results.json',*sorted(chunks.glob('*.json'))]
            _write_json(complete,{'files':{str(p.relative_to(directory)):file_hash(p) for p in paths}})
            _write_state(directory,'completed',len(rows),len(unit_ids))
            return rows
        except BaseException as exc:
            _write_state(directory,'failed',len(rows),len(unit_ids),str(exc));raise


def stratum_robustness(examples,scorer,*,strata,draws=2000,matched_draws=1000,
                       matched_size=50,seed=20261016,matched_seed=20261020,form='pooled'):
    ids,_=_examples(examples)
    if set(strata)!=set(ids) or set(strata.values())!={'legacy-150','new-50'}:
        raise ValueError('Preserve both legacy-150 and new-50 document strata.')
    groups={group:{key:examples[key] for key in ids if strata[key]==group} for group in ('legacy-150','new-50')}
    within={group:bootstrap_scores(values,scorer,draws=draws,seed=seed,form=form) for group,values in groups.items()}
    legacy_ids,legacy=_examples(groups['legacy-150'])
    if (type(matched_size) is not int or not 1<=matched_size<=len(legacy_ids)
            or type(matched_draws) is not int or matched_draws<1):raise ValueError('Invalid matched legacy draw request.')
    rng=np.random.default_rng(matched_seed);scores=[];chosen_ids=[]
    for _ in range(matched_draws):
        chosen=rng.choice(len(legacy_ids),size=matched_size,replace=False)
        scores.append(_condition(legacy[chosen],scorer,form));chosen_ids.append([legacy_ids[i] for i in chosen])
    new=within['new-50']['observed']
    percentile=float(100*np.mean(np.asarray(scores)<=new)) if math.isfinite(new) and np.isfinite(scores).all() else math.nan
    return dict(groups=within,matched_legacy_scores=scores,matched_document_ids=chosen_ids,
                new_score_percentile=percentile,percentile_rule='upper empirical CDF; ties included',
                interpretation='robustness only; no exposure-effect inference')


def calibration_diagnostics(scorer,*,dimension,n_documents,trials,draws,seed,shift):
    """Explicit bounded synthetic diagnostic, not a type-I calibration certificate."""
    if any(type(n) is not int or n<1 for n in (dimension,n_documents,trials,draws)):
        raise ValueError('Declare positive bounded diagnostic counts.')
    shift=np.asarray(shift,dtype=np.float64)
    if shift.shape!=(dimension,) or not np.isfinite(shift).all():raise ValueError('Invalid diagnostic shift vector.')
    rng=np.random.default_rng(seed);null=[];shifted=[]
    for _ in range(trials):
        left={str(i):rng.standard_normal(dimension) for i in range(n_documents)}
        right={str(i):rng.standard_normal(dimension) for i in range(n_documents)}
        draw_seed=int(rng.integers(0,2**32))
        null.append(bootstrap_compare(left,right,scorer,draws=draws,seed=draw_seed)['p_value'])
        shifted.append(bootstrap_compare({k:v+shift for k,v in left.items()},right,scorer,draws=draws,seed=draw_seed)['p_value'])
    return dict(seed=seed,trials=trials,draws=draws,n_documents=n_documents,dimension=dimension,
                null_pvalues=null,shifted_pvalues=shifted,
                null_rejection_rate=float(np.mean(np.asarray(null)<.05)),
                shift_detection_rate=float(np.mean(np.asarray(shifted)<.05)),
                establishes_nominal_calibration=False,
                limitation='Synthetic case-specific diagnostic; assess actual ranking statistic and Monte Carlo uncertainty before inferential claims.')
