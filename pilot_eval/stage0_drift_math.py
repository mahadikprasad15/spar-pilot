"""CPU reconstruction and numeric oracles for fixed-context drift measurements."""
import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.token_measurement_math import full_vocabulary_kl, prediction_positions

SUM_FIELDS = ['update_squared_sum', 'base_squared_sum', 'ratio_sum']


def coefficient_fractions(dot, actual, sum_abs, length):
    """Coefficient reconstruction use of its allowance (ADR 0015).

    The legacy allowance 1e-6+1e-5|dot| is kept. An FP32 dot product of length
    d may also err by the a-priori bound gamma_d*sum|a_i x_i|,
    gamma_d=d*u/(1-d*u), u=2**-24: under heavy cancellation the legacy floor is
    finer than FP32 can resolve. Returns (fraction, legacy_fraction).
    """
    dot,actual,sum_abs=(np.asarray(v,dtype=np.float64) for v in [dot,actual,sum_abs])
    if (type(length) is not int or length<1 or dot.shape!=actual.shape or dot.shape!=sum_abs.shape
            or not all(np.isfinite(v).all() for v in [dot,actual,sum_abs]) or np.any(sum_abs<0)):
        raise ValueError('Malformed coefficient reconstruction evidence.')
    if not dot.size:return 0.,0.
    unit=2.0**-24;gamma=length*unit/(1-length*unit)
    error=np.abs(dot-actual);legacy=1e-6+1e-5*np.abs(dot)
    return float((error/np.maximum(legacy,gamma*sum_abs)).max()),float((error/legacy).max())


def positions(rows):
    writes=[]
    for example,row in enumerate(rows):
        ids=row['input_ids'];size=len(ids)
        if (not size or row['attention_mask']!=[1]*size
                or any(type(t) is not int or t<0 for t in ids)
                or set(row['masks'])!=set(VIEWS)
                or any(len(row['masks'][v])!=size or any(type(b) is not bool for b in row['masks'][v]) for v in VIEWS)):
            raise ValueError('Fixed input/mask mapping mismatch.')
        for t in range(size):
            views=[v for v in VIEWS if row['masks'][v][t]]
            if len(views)>1:raise ValueError('Counted views overlap.')
            if views:writes.append(dict(example=example,position=t,view=views[0],token_id=ids[t]))
    return dict(writes=writes,predictions=prediction_positions(rows))


def kl_values(tuned_logits,base_logits):
    """Full vocabulary KL(variant || untuned); retain signed roundoff residuals."""
    result=full_vocabulary_kl(tuned_logits,base_logits)
    # Same analytic FP64 bound as the Pilot 4 token instrument; no clipping.
    def logs(value):
        value=np.asarray(value,dtype=np.float64)
        value=value-value.max(axis=-1,keepdims=True)
        return value-np.log(np.exp(value).sum(axis=-1,keepdims=True))
    lp,lq=logs(tuned_logits),logs(base_logits)
    bound=64*np.finfo(np.float64).eps*(1+np.max(np.abs(lp)+np.abs(lq)))
    if np.any(result < -bound):raise ValueError('Negative KL exceeds FP64 arithmetic bound.')
    return result


def write_sums(base_output,update_output):
    """Direct dense-output oracle; undefined zero denominators fail explicitly."""
    base,update=np.asarray(base_output,dtype=np.float64),np.asarray(update_output,dtype=np.float64)
    if base.ndim!=2 or base.shape!=update.shape or not np.isfinite(base).all() or not np.isfinite(update).all():
        raise ValueError('Invalid paired module outputs.')
    bn=np.linalg.norm(base,axis=-1);un=np.linalg.norm(update,axis=-1)
    if np.any(bn==0):raise ValueError('Effective-write denominator is zero; ratio undefined.')
    return np.array([np.sum(un**2),np.sum(bn**2),np.sum(un/bn)],dtype=np.float64)


def check(result,rows,config,variant,diagnostic):
    """Recheck every persisted mapping, shape, declared gate and zero control."""
    arrays,meta,gates=result['arrays'],result['metadata'],result['validation']
    mapping=positions(rows);module_count=len(config['modules'])
    if (meta!=dict(example_ids=[r['id'] for r in rows],**mapping,modules=config['modules'],
                   variant=variant,kl_direction='variant||untuned',write_inputs='adapted-module-input',
                   write_fields=SUM_FIELDS)
            or set(arrays)!={'kl','write_sums'}
            or arrays['kl'].shape!=(len(mapping['predictions']),)
            or arrays['write_sums'].shape!=(len(rows),len(VIEWS),module_count,len(SUM_FIELDS))
            or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('Drift token/view/module/value mapping mismatch.')
    if np.any(arrays['write_sums']<0):raise ValueError('Negative effective-write sums.')
    for e,row in enumerate(rows):
        for v,view in enumerate(VIEWS):
            if not any(row['masks'][view]) and np.any(arrays['write_sums'][e,v]):
                raise ValueError('Empty view has effective-write values.')
    operator_gate='operator_passed' if config.get('collection_kind')=='conditional-intruders' else 'rank1_passed'
    required={'reference_invariant',operator_gate,'finite','mapping_exact'}
    if any(gates.get(key) is not True for key in required):raise ValueError('Drift numerical/integrity gate failed.')
    if not 0<=gates.get('reconstruction_max_fraction',float('inf'))<=1:
        raise ValueError('Drift rank-one reconstruction gate failed.')
    if (gates.get('diagnostic') is not diagnostic or gates.get('context_chunk')!=config['context_chunk']
            or gates.get('max_logit_contexts',float('inf'))>config['context_chunk']
            or gates.get('workspace_bytes')!=config['workspace_bytes']):
        raise ValueError('Drift workspace/diagnostic identity mismatch.')
    if diagnostic and any(gates.get(k) is not True for k in ['repeat_passed','chunk_passed']):
        raise ValueError('Strict repeat/chunk agreement gate failed.')
    if (not np.isfinite(gates.get('max_abs_log_sum',float('inf'))) or gates.get('max_abs_log_sum',-1)<0
            or np.any(arrays['kl'] < -gates.get('kl_roundoff_bound',-1))
            or not np.isfinite(gates.get('kl_roundoff_bound',float('inf')))
            or not 0<=gates['kl_roundoff_bound']<=64*np.finfo(float).eps*(1+gates.get('max_abs_log_sum',-1))):
        raise ValueError('Invalid KL arithmetic bound or negative residual.')
    if variant in config['controls']:
        if np.any(arrays['kl']) or np.any(arrays['write_sums'][...,[0,2]]):
            raise ValueError('Zero/no-operator control has nonzero KL or writes.')
    return result


def summarize(config,batches,rows):
    """Documents are the units; keep token weighting and RMS ratio distinct."""
    by_id={r['id']:r for r in rows};groups={};controls={};documents=[]
    for unit,result in zip(config['units'],batches):
        variant=unit['variant'];meta=result['metadata'];arrays=result['arrays']
        if variant in config['controls']:
            controls[variant]=dict(documents=len(unit['ids']),exact_zero=True)
            continue
        for e,identifier in enumerate(meta['example_ids']):
            row=by_id[identifier]
            for vi,view in enumerate(VIEWS):
                selected=[i for i,p in enumerate(meta['predictions']) if p['example']==e and p['view']==view]
                count=sum(row['masks'][view])
                if not count:continue
                values=arrays['kl'][selected];sums=arrays['write_sums'][e,vi]
                record=dict(variant=variant,id=identifier,corpus=row['corpus'],view=view,
                    kl_count=len(values),kl_sum=float(values.sum()),kl_mean=float(values.mean()) if len(values) else None,
                    write_count=count,write_sums=sums.tolist())
                documents.append(record);groups.setdefault((variant,row['corpus'],view),[]).append(record)
    cells=[]
    for (variant,corpus,view),docs in sorted(groups.items()):
        valid=[r for r in docs if r['kl_count']];n=sum(r['kl_count'] for r in valid)
        summed=np.sum([r['write_sums'] for r in docs],axis=0);tokens=sum(r['write_count'] for r in docs)
        module_stats=[]
        for mi,(module,s) in enumerate(zip(config['modules'],summed)):
            if s[1]<=0:raise ValueError('Effective-write corpus denominator is zero.')
            per=[r['write_sums'][mi][2]/r['write_count'] for r in docs]
            module_stats.append(dict(module=module,update_squared_sum=float(s[0]),base_squared_sum=float(s[1]),
                ratio_sum=float(s[2]),token_mean_ratio=float(s[2]/tokens),equal_example_mean_ratio=float(np.mean(per)),
                rms_ratio=float(np.sqrt(s[0]/s[1]))))
        cells.append(dict(variant=variant,corpus=corpus,view=view,
            kl=dict(documents=len(valid),tokens=n,token_mean=sum(r['kl_sum'] for r in valid)/n if n else None,
                    equal_example_mean=float(np.mean([r['kl_mean'] for r in valid])) if valid else None),
            writes=dict(documents=len(docs),tokens=tokens,modules=module_stats),
            sum_factor_norm_products=config['factor_norm_sums'][variant]))
    observed=[r['validation'].get('chunk_comparison') for r in batches]
    observed=[v for v in observed if v is not None]
    differences=[]
    for corpus,view in sorted({(c['corpus'],c['view']) for c in cells}):
        group=[c for c in cells if (c['corpus'],c['view'])==(corpus,view) and c['kl']['token_mean'] is not None]
        pairs=[dict(left=a['variant'],right=b['variant'],left_kl=a['kl']['token_mean'],right_kl=b['kl']['token_mean'],
            absolute_difference=abs(a['kl']['token_mean']-b['kl']['token_mean']))
            for i,a in enumerate(group) for b in group[i+1:]]
        if pairs:differences.append(dict(corpus=corpus,view=view,**min(pairs,key=lambda p:p['absolute_difference'])))
    resolution=dict(observed_max_chunk_disagreement_nats=max((v['max_absolute_error'] for v in observed),default=None),
        diagnostics=observed,smallest_token_mean_variant_differences=differences,
        scope='Observed numerical resolution only; not statistical uncertainty or a universal error bound.')
    return dict(schema_version=1,measurement_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',
        execution_kind=config['execution_kind'],kl_direction='variant||untuned',units='nats',main_documents_per_variant=300,
        modules=config['modules'],controls=controls,cells=cells,per_document=documents,
        historical_values_used=False,numerical_resolution=resolution,comparison_provenance=dict(runtime=config['runtime'],execution_sha256=config['execution_sha256']),
        interpretation='Fixed supplied contexts; KL alone is not behavioral harm. No cross-session/historical join or non-inferiority claim.')
