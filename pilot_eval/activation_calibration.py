"""Versioned empirical agreement calibration; CPU rule fitting, held-out validation."""
from contextlib import nullcontext
import json
from pathlib import Path
import random
import re

import numpy as np

from pilot_eval.activation_prepare import load_prepared, measurement_steps, STEPS, VIEWS
from pilot_eval.activation_profile import (ARRAYS, BATCHES, HFProfileDependencies,
    _relative, _seal, _verified, _load_arrays, _check_gate)
from pilot_eval.activation_workflow import _save_arrays
from pilot_eval.training import run_lock, file_hash
from pilot_eval.run import _write_json, _write_state
from pilot_eval.workflow import _hash, _save_frozen

PROTOCOL = 'pilot3-agreement-v2'
VARIANTS = ['batch-1', 'repeat-1', 'padded-1', 'batch-2', 'batch-4', 'batch-8', 'batch-16']


def _expanded(values):
    """Per-example, token-weighted and equal-example sufficient means."""
    out = {}
    for key, value in values.items():
        prefix, field = key.split('_', 1)
        count = values[prefix + ('_defined_count' if field == 'ratio_sum' else '_count')]
        valid = count > 0
        denominator = count.sum(axis=0)
        divisor = count[..., None] if value.ndim == count.ndim + 1 else count
        denom = denominator[..., None] if value.ndim == count.ndim + 1 else denominator
        token = np.divide(value.sum(axis=0), denom, out=np.zeros_like(value[0]), where=denom != 0)
        per = np.divide(value, divisor, out=np.zeros_like(value), where=divisor != 0)
        n = valid.sum(axis=0)
        ndiv = n[..., None] if value.ndim == count.ndim + 1 else n
        equal = np.divide(per.sum(axis=0), ndiv, out=np.zeros_like(value[0]), where=ndiv != 0)
        if field.endswith('count'):
            token = equal = (denominator > 0).astype(float)
        out[key] = np.concatenate([value, token[None], equal[None]], axis=0)
    return out


def comparison_metrics(reference, candidate, *, resolution=None):
    """Maximum errors over examples, views/layers/modules and both pooled weightings.

    Vector and activation-magnitude errors use mean ordinary activation magnitude
    as their physical scale. Dimensionless ratio errors use max(1, |a|, |b|).
    Direction error is 1-cosine; weak/zero vectors are reported unresolved.
    """
    if set(reference) != set(candidate):
        raise ValueError('checkpoint identity mismatch')
    errors, unresolved = {}, {'directions': 0, 'direction_comparisons': 0}
    def retain(name, value):
        value = float(np.max(value, initial=0))
        if not np.isfinite(value):
            raise ValueError('nonfinite calibration metric')
        errors[name] = max(errors.get(name, 0.), value)
    for step in reference:
        raw_a, raw_b = reference[step], candidate[step]
        if set(raw_a) != ARRAYS or set(raw_b) != ARRAYS:
            raise ValueError('calibration summary schema mismatch')
        for key in ARRAYS:
            a, b = raw_a[key], raw_b[key]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite or misaligned calibration arrays')
            if key.endswith('count') and not np.array_equal(a, b):
                raise ValueError('count/defined coverage mismatch')
            if key.endswith('base_norm_sum') and not np.array_equal(a == 0, b == 0):
                raise ValueError('undefined baseline coverage mismatch')
            if step == 0 and 'delta' in key and (np.any(a) or np.any(b)):
                raise ValueError('exact-zero write failed')
        expanded_a, expanded_b = _expanded(raw_a), _expanded(raw_b)
        for key in sorted(ARRAYS):
            if key.endswith('count'):
                continue
            a, b = expanded_a[key], expanded_b[key]
            prefix, field = key.split('_', 1)
            scale = np.maximum(expanded_a[prefix + '_base_norm_sum'], expanded_b[prefix + '_base_norm_sum'])
            if field in ('base_sum', 'delta_sum'):
                error = np.linalg.norm(b-a, axis=-1)
                name = prefix + ('_base_vector' if field == 'base_sum' else '_delta_vector')
                normalized = np.divide(error, scale, out=np.zeros_like(error), where=scale != 0)
                if np.any((scale == 0) & (error != 0)):
                    raise ValueError('nonzero difference with zero ordinary magnitude')
                retain(name, normalized)
                an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
                floor = (resolution or {}).get(name, 0.) * scale
                defined = (an > floor) & (bn > floor)
                unresolved['directions'] += int((~defined).sum())
                unresolved['direction_comparisons'] += int(defined.sum())
                cosine = np.divide((a*b).sum(axis=-1), an*bn, out=np.ones_like(an), where=defined)
                retain(name + '_direction', np.where(defined & (error != 0), 1-np.clip(cosine, -1, 1), 0))
                # Alternative normalization by norm of the mean baseline vector.
                if field == 'delta_sum':
                    av = np.linalg.norm(expanded_a['block_base_sum'], axis=-1)
                    bv = np.linalg.norm(expanded_b['block_base_sum'], axis=-1)
                    if not np.array_equal(av == 0, bv == 0):
                        raise ValueError('undefined alternative-ratio coverage mismatch')
                    ar = np.divide(an, av, out=np.zeros_like(an), where=av != 0)
                    br = np.divide(bn, bv, out=np.zeros_like(bn), where=bv != 0)
                    retain('block_alternative_ratio', abs(ar-br)/np.maximum(1., np.maximum(abs(ar), abs(br))))
            else:
                normalizer = np.maximum(1., np.maximum(abs(a), abs(b))) if field == 'ratio_sum' else scale
                error = abs(b-a)
                if np.any((normalizer == 0) & (error != 0)):
                    raise ValueError('undefined scalar normalization')
                retain(key, np.divide(error, normalizer, out=np.zeros_like(error), where=normalizer != 0))
    return {'errors': errors, 'coverage': unresolved}


def fit_rule(metrics, *, margin):
    if margin != 3 or not metrics:
        raise ValueError('approved protocol requires a 3x empirical margin and calibration evidence')
    envelope = {}
    for measurement in metrics:
        for key, value in measurement['errors'].items():
            envelope[key] = max(envelope.get(key, 0.), value)
    return {'protocol': PROTOCOL, 'margin': margin, 'envelope': envelope,
            'thresholds': {key: margin*value for key, value in envelope.items()},
            'resolution': {key: margin*value for key, value in envelope.items() if key.endswith('_vector')},
            'interpretation': 'empirical engineering envelope; not a confidence guarantee'}


def check_agreement(reference, candidate, rule):
    if rule.get('protocol') != PROTOCOL or rule.get('margin') != 3:
        raise ValueError('unsupported calibrated rule')
    if (set(rule['envelope']) != set(rule['thresholds']) or any(
            not np.isfinite(v) or v < 0 or rule['thresholds'][k] != 3*v
            for k,v in rule['envelope'].items())):
        raise ValueError('thresholds must equal the frozen 3x envelope')
    metrics = comparison_metrics(reference, candidate, resolution=rule['resolution'])
    if set(metrics['errors']) != set(rule['thresholds']):
        raise ValueError('calibrated metric schema mismatch')
    differences = [{'measurement': key, 'error': value, 'limit': rule['thresholds'][key]}
                   for key, value in metrics['errors'].items() if value > rule['thresholds'][key]]
    return {'passed': not differences, 'differences': differences, 'thresholds': rule['thresholds'],
            'coverage': metrics['coverage'], 'protocol': PROTOCOL}


def prepare_calibration(config_path, output_root, *, name, dependencies=None):
    root = Path(output_root).resolve()
    config, rows = load_prepared(config_path, root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('safe calibration name required')
    excluded, cohorts = [], {'calibration': [], 'validation': []}
    rng = random.Random(config['seed'])
    for corpus in ['gsm8k', 'fineweb']:
        ordered = sorted((r for r in rows if r['corpus']==corpus), key=lambda r:(-len(r['input_ids']), r['id']))
        excluded.extend(r['id'] for r in ordered[:8])
        eligible = ordered[8:]
        # Sample both cohorts from four length strata, preserving paired length order.
        strata = np.array_split(np.arange(len(eligible)), 4)
        for stratum in strata:
            if len(stratum) < 4:
                raise ValueError('insufficient independent calibration examples')
            selected = rng.sample(list(map(int, stratum)), 4)
            cohorts['calibration'].extend(eligible[i]['id'] for i in selected[:2])
            cohorts['validation'].extend(eligible[i]['id'] for i in selected[2:])
    for phase in cohorts:
        cohorts[phase].sort(key=lambda item: (next(r['corpus'] for r in rows if r['id']==item),
                                            -len(next(r['input_ids'] for r in rows if r['id']==item)), item))
    runtime = (dependencies or HFProfileDependencies()).runtime()
    plan = {'protocol': PROTOCOL, 'prepared_path': str(_relative(config_path, root).relative_to(root)),
            'prepared_sha256': _hash(config), 'inputs_sha256': config['items_sha256'],
            'runtime': runtime, 'cohorts': cohorts, 'excluded_profile_ids': excluded,
            'margin': 3, 'seed': config['seed'], 'variants': VARIANTS, 'checkpoint_steps': measurement_steps(config),
            'run_path': config['run_path'] + '/calibration/' + name}
    path = root / 'plans' / name / 'activation.calibration.json'
    with run_lock(path.parent):
        _save_frozen(path, plan)
    return path


def _plan(path, root, runtime=None):
    plan = json.loads(_relative(path, root).read_text())
    config, rows = load_prepared(root / plan['prepared_path'], root)
    if (plan['protocol'] != PROTOCOL or plan['margin'] != 3 or plan['prepared_sha256'] != _hash(config)
            or plan['inputs_sha256'] != config['items_sha256']
            or (runtime is not None and plan['runtime'] != runtime)):
        raise ValueError('calibration identity/runtime mismatch')
    if (set(plan['cohorts']) != {'calibration','validation'} or plan['variants'] != VARIANTS
            or plan['checkpoint_steps'] != measurement_steps(config) or plan['seed'] != config['seed']):
        raise ValueError('calibration protocol schema mismatch')
    all_ids = sum(plan['cohorts'].values(), [])
    by_id = {r['id']:r for r in rows}
    if any(i not in by_id for i in all_ids):
        raise ValueError('unknown calibration input')
    if any(sum(by_id[i]['corpus']==corpus for i in ids)!=8
           for ids in plan['cohorts'].values() for corpus in ['gsm8k','fineweb']):
        raise ValueError('calibration cohort balance mismatch')
    if (any(len(ids)!=16 for ids in plan['cohorts'].values()) or len(set(all_ids))!=32
            or set(all_ids)&set(plan['excluded_profile_ids'])):
        raise ValueError('calibration and validation cohorts must be independent')
    return plan, config, rows


def _collect(path, root, phase, deps):
    plan, config, all_rows = _plan(path, root, deps.runtime())
    directory = root / plan['run_path'] / phase
    identity = {'plan_sha256': _hash(plan), 'phase': phase}
    required = ['config.json', 'results.json'] + [v+'/complete.json' for v in VARIANTS]
    with run_lock(directory):
        if (directory/'complete.json').exists():
            return _verified(directory, identity, required)
        _save_frozen(directory/'config.json', identity)
        selected = [{r['id']:r for r in all_rows}[item] for item in plan['cohorts'][phase]]
        records, files, engine = [], [directory/'config.json'], None
        _write_state(directory, 'running', 0, len(VARIANTS))
        try:
            for variant in VARIANTS:
                target = directory/variant
                unit_identity = {'phase_identity': identity, 'variant': variant, 'example_ids': plan['cohorts'][phase]}
                if (target/'complete.json').exists():
                    record = _verified(target, unit_identity, ['results.json'])
                else:
                    batch = int(variant.split('-')[-1])
                    try:
                        if engine is None:
                            engine = deps.activation_engine(config, root)
                        scope = engine.integrity_scope() if hasattr(engine,'integrity_scope') else nullcontext({})
                        collected = {s:{k:[] for k in ARRAYS} for s in plan['checkpoint_steps']}
                        with scope:
                            for start in range(0,len(selected),batch):
                                chunk = selected[start:start+batch]
                                print(f'activation calibration: {phase} {variant} examples {start}/{len(selected)}', flush=True)
                                if variant == 'padded-1':
                                    reference = engine.capture_reference(chunk, padded_width=max(len(r['input_ids']) for r in selected))
                                else:
                                    reference = engine.capture_reference(chunk)
                                try:
                                    for step in plan['checkpoint_steps']:
                                        print(f'activation calibration: {phase} {variant} checkpoint {step}', flush=True)
                                        source = config['source_evidence']['checkpoints'][str(step)]
                                        checkpoint = root/source['path']
                                        if (file_hash(checkpoint/'adapter_model.safetensors')!=source['adapter_sha256']
                                                or file_hash(checkpoint/'complete.json')!=source['complete_sha256']):
                                            raise ValueError('source checkpoint hash mismatch')
                                        measured = engine.measure(checkpoint, reference, step=step)
                                        _check_gate(measured['validation'], config, step)
                                        count = np.repeat(np.array([[sum(r['masks'][v]) for v in VIEWS] for r in chunk])[...,None],28,axis=2)
                                        for key,value in measured['arrays'].items():
                                            if not np.isfinite(value).all() or value.shape[0]!=len(chunk):
                                                raise ValueError('nonfinite/misaligned calibration summary')
                                            if key=='block_count' and not np.array_equal(value,count):
                                                raise ValueError('frozen token mapping/count mismatch')
                                            collected[step][key].append(value)
                                finally:
                                    if hasattr(engine,'release_reference'): engine.release_reference(reference)
                                    del reference
                        for step in plan['checkpoint_steps']:
                            _save_arrays(target/f'step-{step}.npz', {k:np.concatenate(v) for k,v in collected[step].items()})
                        record = {'variant':variant,'status':'completed','example_ids':plan['cohorts'][phase]}
                    except Exception as exc:
                        if not deps.is_oom(exc) or batch==1:
                            raise
                        record = {'variant':variant,'status':'oom','error':str(exc),'example_ids':plan['cohorts'][phase]}
                        if engine is not None: engine.close()
                        engine=None
                        deps.cleanup()
                    _write_json(target/'results.json',record)
                    _seal(target,unit_identity,[target/'results.json']+list(target.glob('step-*.npz')))
                records.append(record)
                files.extend(p for p in target.iterdir() if p.suffix in ['.json','.npz'])
                _write_state(directory,'running',len(records),len(VARIANTS))
            result={'phase':phase,'records':records,'example_ids':plan['cohorts'][phase]}
            _write_json(directory/'results.json',result)
            _seal(directory,identity,files+[directory/'results.json'])
            _write_state(directory,'completed',len(records),len(VARIANTS))
            return result
        except BaseException as exc:
            _write_state(directory,'failed',len(records),len(VARIANTS),str(exc))
            _write_json(directory/'logs/error.json',{'error':str(exc)})
            raise
        finally:
            if engine is not None: engine.close()
            deps.cleanup()


def collect_calibration(path, output_root, *, dependencies=None):
    return _collect(path,Path(output_root).resolve(),'calibration',dependencies or HFProfileDependencies())


def _arrays(directory, variant, steps=STEPS):
    return {step:_load_arrays(directory/variant/f'step-{step}.npz') for step in steps}


def _verify_phase(plan, root, phase):
    directory=root/plan['run_path']/phase
    result=_verified(directory,{'plan_sha256':_hash(plan),'phase':phase},['config.json','results.json']+[v+'/complete.json' for v in VARIANTS])
    return directory,result


def freeze_rule(path, output_root, *, review_notes):
    root=Path(output_root).resolve()
    if not review_notes.strip(): raise ValueError('review notes required')
    plan,_,_=_plan(path,root)
    directory,result=_verify_phase(plan,root,'calibration')
    reference=_arrays(directory,'batch-1',plan['checkpoint_steps'])
    candidates=[r['variant'] for r in result['records'] if r['status']=='completed' and r['variant']!='batch-1']
    raw=[comparison_metrics(reference,_arrays(directory,v,plan['checkpoint_steps'])) for v in candidates]
    rule=fit_rule(raw,margin=plan['margin'])
    # Refit directional envelope using only directions above measured resolution.
    filtered=[comparison_metrics(reference,_arrays(directory,v,plan['checkpoint_steps']),resolution=rule['resolution']) for v in candidates]
    refined=fit_rule(filtered,margin=plan['margin'])
    refined['resolution']=rule['resolution']
    refined.update(plan_sha256=_hash(plan),review_notes=review_notes,
                   calibration_complete_sha256=file_hash(directory/'complete.json'))
    target=root/plan['run_path']/'rule.json'
    with run_lock(target.parent): _save_frozen(target,refined)
    return target


def _negative_controls(reference, rule):
    controls={}
    for label in ['wrong-sign','wrong-scale']:
        corrupted={s:{k:v.copy() for k,v in data.items()} for s,data in reference.items()}
        for step in reference:
            if step==0: continue
            for key in ['block_delta_sum','block_delta_norm_sum','module_delta_norm_sum','module_ratio_sum']:
                if label=='wrong-sign':
                    if key=='block_delta_sum': corrupted[step][key]*=-1
                else: corrupted[step][key]*=2
        controls[label]=not check_agreement(reference,corrupted,rule)['passed']
    # Deliberate token/module mapping corruption must fail the exact count gate.
    bad={s:{k:v.copy() for k,v in data.items()} for s,data in reference.items()}
    bad[next(s for s in reference if s != 0)]['block_count'][0,0,0]+=1
    try: comparison_metrics(reference,bad)
    except ValueError: controls['wrong-mapping']=True
    else: controls['wrong-mapping']=False
    if not all(controls.values()): raise ValueError('calibrated gate accepted a negative control')
    return controls


def validate_rule(path, output_root, *, dependencies=None):
    root=Path(output_root).resolve()
    plan,_,_=_plan(path,root)
    rule_path=root/plan['run_path']/'rule.json'
    if not rule_path.exists(): raise ValueError('freeze reviewed calibration rule before validation')
    rule=json.loads(rule_path.read_text())
    if rule['plan_sha256']!=_hash(plan): raise ValueError('frozen rule identity mismatch')
    calibration_directory, calibration=_verify_phase(plan,root,'calibration')
    if rule['calibration_complete_sha256'] != file_hash(calibration_directory/'complete.json'):
        raise ValueError('calibration evidence identity mismatch')
    _collect(path,root,'validation',dependencies or HFProfileDependencies())
    directory,result=_verify_phase(plan,root,'validation')
    target=root/plan['run_path']/'validated'
    identity={'plan_sha256':_hash(plan),'rule_sha256':file_hash(rule_path),
              'calibration_complete_sha256':file_hash(root/plan['run_path']/'calibration/complete.json'),
              'validation_complete_sha256':file_hash(directory/'complete.json')}
    with run_lock(target):
        if (target/'complete.json').exists(): return _verified(target,identity,['results.json','config.json'])
        _save_frozen(target/'config.json',identity)
        agreements, controls = {}, {}
        try:
            reference=_arrays(directory,'batch-1',plan['checkpoint_steps'])
            controls=_negative_controls(reference,rule)
            agreements={r['variant']:check_agreement(reference,_arrays(directory,r['variant'],plan['checkpoint_steps']),rule)
                        for r in result['records'] if r['status']=='completed'}
            for required in ['repeat-1','padded-1']:
                if required not in agreements or not agreements[required]['passed']:
                    raise ValueError('repeatability/padding validation failed; stop and inspect evidence')
            eligible=[b for b in BATCHES if 'batch-'+str(b) in agreements and agreements['batch-'+str(b)]['passed']]
            # Failed candidates remain rejected; they never authorize execution.
            saved={'protocol':PROTOCOL,'rule':rule,'validated_batches':eligible,'negative_controls':controls,
                   'agreements':agreements,'runtime':plan['runtime'],'prepared_sha256':plan['prepared_sha256'],
                   'calibration_plan':str(_relative(path,root).relative_to(root))}
            _write_json(target/'results.json',saved)
            _seal(target,identity,[target/'config.json',target/'results.json'])
            _write_state(target,'completed',len(eligible),len(BATCHES))
            return saved
        except Exception as exc:
            _write_json(target/'results.json', {'status':'failed','error':str(exc),
                'agreements':agreements,'negative_controls':controls,'protocol':PROTOCOL})
            _write_state(target,'failed',0,len(BATCHES),str(exc))
            raise



def load_validated_rule(path, output_root, *, prepared, runtime):
    root=Path(output_root).resolve()
    plan,_,_=_plan(path,root,runtime)
    if plan['prepared_sha256']!=_hash(prepared): raise ValueError('calibration/preparation mismatch')
    for phase in ['calibration','validation']: _verify_phase(plan,root,phase)
    rule_path=root/plan['run_path']/'rule.json'
    directory=root/plan['run_path']/'validated'
    identity={'plan_sha256':_hash(plan),'rule_sha256':file_hash(rule_path),
              'calibration_complete_sha256':file_hash(root/plan['run_path']/'calibration/complete.json'),
              'validation_complete_sha256':file_hash(root/plan['run_path']/'validation/complete.json')}
    result=_verified(directory,identity,['config.json','results.json'])
    if result['rule']!=json.loads(rule_path.read_text()): raise ValueError('validated rule mismatch')
    return result
