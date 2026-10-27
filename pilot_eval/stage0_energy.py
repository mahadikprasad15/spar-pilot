"""Bounded, source-verified base-input energy collection and durable resume."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.stage0_energy_plan import prepare,load
from pilot_eval.stage0_energy_math import validate
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_now,budget_status
from pilot_eval.stage0_evaluation import _reservation,_attempt_timing,_recover_owner
from pilot_eval.stage0_budget import reserve
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def _deps():
    from pilot_eval.stage0_energy_backend import EnergyDependencies
    return EnergyDependencies()


def runtime(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,directory,_=load(path,root);deps=dependencies or _deps();info=deps.runtime();frozen=directory/'meta/runtime.json'
    return dict(runtime=info,model_loaded=False,runtime_frozen=frozen.exists(),matches_frozen_runtime=None if not frozen.exists() else info==json.loads(frozen.read_text()),
        matches_qualification_runtime=info==config['qualification_runtime'],budget=budget_status(root),
        storage=deps.preflight(config,root) if hasattr(deps,'preflight') else dict(scope='fixture'))


def predict(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_=load(path,root);review=json.loads(Path(review_path).read_text())
    if set(review)!={'reviewer','notes','predictions','prior_exposure'} or type(review['prior_exposure']) is not bool or any(not str(review[k]).strip() for k in ['reviewer','notes','predictions']):raise ValueError('Freeze predictions and declare prior exposure before energy collection.')
    target=directory/'predictions.json';marker=directory/'predictions.complete.json'
    with _writer(directory):
        if target.exists():
            value=_prediction(config,directory)
            if value['review']!=review:raise ValueError('Energy predictions immutable.')
            return value
        spectral_digest=_hash(json.loads(inside(root,config['spectral_path']).read_text()))
        exposure={str(p.relative_to(root)):file_hash(p) for p in (root/'reports').glob('*/exposure.json') if json.loads(p.read_text()).get('config_sha256')==spectral_digest}
        value=dict(config_sha256=_hash(config),review=review,frozen_at=_now(),exposure_receipts=exposure,
            prediction_status='exploratory' if review['prior_exposure'] or exposure or list((directory/'units').glob('*/complete.json')) else 'frozen-before-measurement')
        _save_frozen(target,value);_save_frozen(marker,dict(sha256=file_hash(target)));return value


def _prediction(config,directory):
    value=json.loads((directory/'predictions.json').read_text())
    if value['config_sha256']!=_hash(config) or json.loads((directory/'predictions.complete.json').read_text())!={'sha256':file_hash(directory/'predictions.json')}:raise ValueError('Prediction receipt changed.')
    return value


def read_unit(config,directory,unit,by_id):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),unit_sha256=_hash(unit),runtime_sha256=file_hash(directory/'meta/runtime.json')) or set(seal.get('files',{}))!={'arrays.npz','metadata.json'}:raise ValueError('Energy unit seal identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Energy unit payload changed.')
    meta=json.loads((folder/'metadata.json').read_text());attempt=directory/'attempts'/meta['attempt_id'];start=json.loads((attempt/'start.json').read_text())
    if file_hash(attempt/'start.json')!=meta['attempt_start_sha256'] or start['runtime_sha256']!=file_hash(directory/'meta/runtime.json') or not all(meta['integrity'].get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Energy attempt integrity changed.')
    final=attempt/'final-integrity.json';timing=_attempt_timing(attempt,required=False)
    if final.exists() and not all(json.loads(final.read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full energy integrity failed.')
    if timing and timing['final_integrity_sha256']!=(file_hash(final) if final.exists() else None):raise ValueError('Energy timing/final integrity seal changed.')
    proof=meta['proof'];n=len(unit['ids'])
    if proof!=dict(base_input_scope=True,sequence_forwards=n,all_factors_shared_inputs=True,hook_calls_per_module=n):raise ValueError('Energy forward/input scope changed.')
    with np.load(folder/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    return validate(arrays,[by_id[i] for i in unit['ids']],config)


def _execution(config,directory,root,path):
    value=json.loads(path.read_text())
    if json.loads((directory/'production-freeze.json').read_text())!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)) or value['config_sha256']!=_hash(config):raise ValueError('Energy execution freeze changed.')
    for name,digest in value['forecast']['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Energy profile evidence changed.')
    return value


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,rows=load(path,root);_prediction(config,directory);by_id={r['id']:r for r in rows};units=config['units']
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid new-unit limit.')
    with _writer(directory):
        if (directory/'measurement.complete.json').exists():return verify_measurement(path,root)
        saved={u['key']:read_unit(config,directory,u,by_id) for u in units};count=lambda:sum(v is not None for v in saved.values())
        if max_new_units==0:return _state(directory,'incomplete',count(),len(units))
        target=inside(root,path).parent/'stage0.energy.execution.json';production=target.exists()
        execution=_execution(config,directory,root,target) if production else {'reservation':{'job_sha256':_hash(config)}}
        pending=[u for u in units if saved[u['key']] is None and (production or u['key'] in config['profile_keys'])]
        if not pending:return dict(state='profile-complete',production_frozen=False)
        deps=dependencies or _deps();info=deps.runtime()
        if (directory/'meta/runtime.json').exists() and info!=json.loads((directory/'meta/runtime.json').read_text()):raise ValueError('Energy session/GPU changed; use a separately named comparison.')
        if approval_path is None and not production and not (directory/'meta/active-reservation.json').exists():raise ValueError('Approve a separately bounded energy profile first.')
        reservation,remaining=_reservation(config,execution,directory,root,info,approval_path);_save_frozen(directory/'meta/runtime.json',info)
        attempt=directory/'attempts'/uuid.uuid4().hex;_save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],runtime_sha256=file_hash(directory/'meta/runtime.json'),started_at=_now()))
        start=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-start>=remaining:raise TimeoutError('Energy allocation exhausted; reconcile before renewing.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                deadline();engine=deps.engine(config,root)
                for unit in pending:
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic();selected=[by_id[i] for i in unit['ids']];arrays,proof=engine.measure(selected)
                    try:validate(arrays,selected,config)
                    except Exception as exc:
                        diag=directory/'diagnostics'/attempt.name/unit['key'];diag.mkdir(parents=True,exist_ok=True);np.savez(diag/'failed-arrays.npz',**arrays)
                        _write_json(diag/'failure.json',dict(error=str(exc),unit=unit,proof=proof));raise
                    if proof!=dict(base_input_scope=True,sequence_forwards=len(selected),all_factors_shared_inputs=True,hook_calls_per_module=len(selected)):raise ValueError('Actual energy capture scope differs.')
                    integrity=engine.check_integrity()
                    if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Energy collection changed model weights.')
                    deadline();folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                    temporary=folder/'arrays.tmp.npz';np.savez(temporary,**arrays);temporary.replace(folder/'arrays.npz')
                    _write_json(folder/'metadata.json',dict(proof=proof,integrity=integrity,seconds=time.monotonic()-begin,attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json')))
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),unit_sha256=_hash(unit),runtime_sha256=file_hash(directory/'meta/runtime.json')),files={n:file_hash(folder/n) for n in ['arrays.npz','metadata.json']}))
                    saved[unit['key']]=arrays;new+=1;_state(directory,'running',count(),len(units))
                    if progress:print(f'Stage0 energy: {count()}/{len(units)} groups sealed; {unit["corpus"]}; {len(selected)} base forwards, all factors shared.',flush=True)
                deadline();integrity=engine.check_integrity(full=True);_write_json(attempt/'final-integrity.json',integrity)
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full energy weight integrity failed.')
                load(path,root);deadline()
                if count()==len(units):
                    files=['config.json','inputs/items.jsonl','inputs/geometry.json','checks/projections.json','meta/runtime.json','predictions.json','predictions.complete.json']+[str((directory/'units'/u['key']/'complete.json').relative_to(directory)) for u in units]
                    _write_json(directory/'measurement.complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in files}))
                return _state(directory,'measurement-complete' if count()==len(units) else 'incomplete',count(),len(units))
        except (Exception,KeyboardInterrupt) as exc:
            if engine is not None and not (attempt/'final-integrity.json').exists():
                try:
                    left=remaining-(time.monotonic()-start)
                    if left>0:
                        with deadline_guard(left,config['execution_kind']=='real-cuda'):_write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except (Exception,KeyboardInterrupt):pass
            _state(directory,'failed',count(),len(units),error=str(exc));log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-start,reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def forecast(path,output_root):
    root=Path(output_root).resolve();config,directory,rows=load(path,root);by_id={r['id']:r for r in rows};rates={};evidence={};overhead=0.;numerics={}
    for unit in config['units']:
        if unit['key'] not in config['profile_keys']:continue
        arrays=read_unit(config,directory,unit,by_id)
        if arrays is None:raise ValueError('Measure the bounded GSM8K and FineWeb profile groups first.')
        values=arrays['sums'];residual=np.abs(values[...,:3].sum(-1)-values[...,3])
        allowance=config['agreement']['atol']+config['agreement']['rtol']*np.abs(values[...,3])
        numerics[unit['key']]=dict(max_absolute_reconstruction_error=float(residual.max()),max_reconstruction_allowance_fraction=float((residual/allowance).max()))
        folder=directory/'units'/unit['key'];meta=json.loads((folder/'metadata.json').read_text());attempt=directory/'attempts'/meta['attempt_id']
        timing=_attempt_timing(attempt);overhead=max(overhead,timing['seconds'])
        if not all(json.loads((attempt/'final-integrity.json').read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Profile final integrity failed.')
        rates[unit['corpus']]=meta['seconds']/sum(len(by_id[i]['input_ids'])**2 for i in unit['ids'])
        for p in [folder/'complete.json',attempt/'timing.complete.json']:evidence[str(p.relative_to(directory))]=file_hash(p)
    missing=[u for u in config['units'] if read_unit(config,directory,u,by_id) is None]
    result=dict(config_sha256=_hash(config),remaining_groups=len(missing),remaining_base_forwards=sum(len(u['ids']) for u in missing),rates=rates,
        conservative_total_seconds=overhead+sum(rates[u['corpus']]*sum(len(by_id[i]['input_ids'])**2 for i in u['ids']) for u in missing),
        evidence=evidence,profile_numerics=numerics,agreement=config['agreement'],method='per-corpus measured seconds / sum squared sequence lengths; plus worst whole-attempt overhead',completion_guaranteed=False)
    _write_json(directory/'results/forecast.json',result);return result


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_=load(path,root);review=json.loads(Path(review_path).read_text());measured=json.loads((directory/'results/forecast.json').read_text())
    required={'reviewer','notes','session_id','source_reviewed','max_units_per_hour','max_seconds','forecast_sha256','numerical_reviewed','agreement'}
    if set(review)!=required or review['numerical_reviewed'] is not True or review['agreement']!=config['agreement'] or review['session_id']!=json.loads((directory/'meta/runtime.json').read_text())['session_id'] or review['forecast_sha256']!=file_hash(directory/'results/forecast.json') or measured['config_sha256']!=_hash(config) or review['max_seconds']<measured['conservative_total_seconds']:raise ValueError('Review exact energy gates and the separately bounded measured production forecast.')
    for name,digest in measured['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Measured energy evidence changed.')
    target=inside(root,path).parent/'stage0.energy.execution.json'
    with _writer(directory):
        if target.exists():
            if _execution(config,directory,root,target)['review']!=review:raise ValueError('Frozen energy review changed.')
            return target
        reservation=reserve(root,_hash(config),{k:v for k,v in review.items() if k not in ['forecast_sha256','numerical_reviewed','agreement']})
        _save_frozen(target,dict(config_sha256=_hash(config),review=review,reservation=reservation,forecast=measured));_save_frozen(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
        _write_json(directory/'meta/active-reservation.json',dict(reservation_id=reservation['reservation_id']))
    return target


def verify_measurement(path,output_root):
    root=Path(output_root).resolve();config,directory,rows=load(path,root);by_id={r['id']:r for r in rows};_prediction(config,directory)
    seal=json.loads((directory/'measurement.complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Energy measurement completion changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Completed energy evidence changed.')
    if any(read_unit(config,directory,u,by_id) is None for u in config['units']):raise ValueError('Energy batches incomplete.')
    return dict(measurement_complete=True,base_forwards=len(rows),factor_conditions=len(config['factor_labels']),real_gpu_measurement=config['execution_kind']=='real-cuda')


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,rows=load(path,root);by_id={r['id']:r for r in rows}
    return dict(completed=sum(read_unit(config,directory,u,by_id) is not None for u in config['units']),total=len(config['units']),budget=budget_status(root),measurement=verify_measurement(path,root) if (directory/'measurement.complete.json').exists() else None)


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_=load(path,root);return _recover_owner(directory,owner_token,review_notes)
