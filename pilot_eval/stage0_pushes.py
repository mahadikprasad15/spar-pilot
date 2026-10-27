"""Saved public gradient workflow: bounded profile, separate freeze, resumable run."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.stage0_push_plan import load,prepare
from pilot_eval.stage0_push_math import instrument_check
from pilot_eval.stage0_sources import inside,factors
from pilot_eval.stage0_evaluation import _reservation,_attempt_timing,_recover_owner
from pilot_eval.stage0_workflow import _writer,_state,_now,budget_status
from pilot_eval.stage0_budget import reserve
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def check(path,output_root):
    root=Path(output_root).resolve();config,directory,_=load(path,root);result=instrument_check()
    _write_json(directory/'checks/instrument.json',dict(config_sha256=_hash(config),**result))
    if not result['passed']:raise ValueError('Initial-gradient independent instrument gate failed; thresholds unchanged.')
    return result


def _deps():
    from pilot_eval.stage0_push_backend import PushDependencies
    return PushDependencies()


def runtime(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,directory,_=load(path,root);deps=dependencies or _deps();info=deps.runtime()
    frozen=directory/'meta/runtime.json'
    return dict(runtime=info,matches_sampled_runtime=info==config['sampled_runtime'],gradient_runtime_frozen=frozen.exists(),
        matches_gradient_runtime=None if not frozen.exists() else info==json.loads(frozen.read_text()),model_loaded=False,budget=budget_status(root),
        storage=deps.preflight(config,root) if hasattr(deps,'preflight') else {'scope':'fixture'})


def validate_arrays(config,unit,arrays,proof,root):
    expected=set();errors={}
    for i,name in enumerate(config['modules']):
        for prefix in ['gradient_a','gradient_b','push','reconstruction']:expected.add(f'{prefix}_{i}')
        shape=config['module_shapes'][name];ga=arrays[f'gradient_a_{i}'];gb=arrays[f'gradient_b_{i}'];push=arrays[f'push_{i}'];oracle=arrays[f'reconstruction_{i}']
        if list(ga.shape)!=shape['a'] or any(list(v.shape)!=shape['b'] for v in [gb,push,oracle]) or any(not np.isfinite(v).all() for v in [ga,gb,push,oracle]):raise ValueError('Gradient array dimensions/finiteness changed.')
        if any(v.dtype!=np.float32 for v in [ga,gb,push]) or oracle.dtype!=np.float64 or np.any(ga) or not np.array_equal(push,-gb):raise ValueError('Raw gradient precision/sign/A-zero gate failed.')
        numerator=float(np.linalg.norm(gb.astype(float)-oracle));den=float(np.linalg.norm(oracle))
        error=(0. if numerator==0 else None) if den==0 else numerator/den
        if error is None or not error<config['reconstruction_gate']:raise ValueError('Independent gradient reconstruction gate failed.')
        errors[name]=error
    if set(arrays)!=expected or proof.get('a_gradient_exact_zero') is not True or proof.get('optimizer_steps')!=0 or proof.get('advantages_detached') is not True or proof.get('global_token_denominator')!=unit['token_count'] or proof.get('reconstruction_relative_errors')!=errors:
        raise ValueError('Gradient proof/denominator mapping changed.')
    active=sum(s['advantage']!=0 for s in unit['sequences'])
    if proof.get('active_sequences')!=active or proof.get('zero_advantage_sequences')!=len(unit['sequences'])-active:raise ValueError('Dead group sequence count changed.')
    if not active and any(np.any(a) for a in arrays.values()):raise ValueError('A dead objective must produce exact zero gradients.')
    return arrays


def read_unit(config,directory,unit,root):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)) or set(seal.get('files',{}))!={'gradients.npz','metadata.json'}:raise ValueError('Gradient seal identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Saved gradient evidence changed.')
    meta=json.loads((folder/'metadata.json').read_text());attempt=directory/'attempts'/meta['attempt_id']
    if file_hash(attempt/'start.json')!=meta['attempt_start_sha256'] or json.loads((attempt/'start.json').read_text()).get('runtime_sha256')!=file_hash(directory/'meta/runtime.json') or not all(meta['integrity'].get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Gradient attempt/runtime/integrity changed.')
    final=attempt/'final-integrity.json'
    if final.exists() and not all(json.loads(final.read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full gradient integrity failed.')
    timing=_attempt_timing(attempt,required=False)
    if timing is not None and timing['final_integrity_sha256']!=(file_hash(final) if final.exists() else None):raise ValueError('Gradient attempt timing seal changed.')
    with np.load(folder/'gradients.npz',allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    return validate_arrays(config,unit,arrays,meta['proof'],root)


def profile_keys(units):
    groups={}
    for u in units:groups.setdefault((u['case'],u['objective']),[]).append(u)
    return [next((u for u in rows if not u['partial'] and any(s['advantage']!=0 for s in u['sequences'])),rows[0])['key'] for rows in groups.values()]


def _execution(config,directory,root,path):
    value=json.loads(path.read_text());seal=json.loads((directory/'production-freeze.json').read_text())
    if seal!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)) or value['config_sha256']!=_hash(config):raise ValueError('Frozen gradient execution changed.')
    for name,digest in value['forecast']['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Frozen gradient profile evidence changed.')
    return value


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,units=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid batch limit.')
    receipt=json.loads((directory/'checks/instrument.json').read_text())
    if receipt.get('config_sha256')!=_hash(config) or receipt.get('passed') is not True:raise ValueError('Run the independent CPU instrument check first.')
    with _writer(directory):
        if (directory/'measurement.complete.json').exists():return verify_measurement(path,root)
        saved={u['key']:read_unit(config,directory,u,root) for u in units};done=lambda:sum(v is not None for v in saved.values())
        if max_new_units==0:return _state(directory,'incomplete',done(),len(units))
        deps=dependencies or _deps();info=deps.runtime()
        if (directory/'meta/runtime.json').exists() and info!=json.loads((directory/'meta/runtime.json').read_text()):raise ValueError('Gradient comparisons require their frozen session/runtime; use a separately named comparison after a change.')
        target=inside(root,path).parent/'stage0.pushes.execution.json';production=target.exists()
        execution=_execution(config,directory,root,target) if production else {'reservation':{'job_sha256':_hash(config)}}
        allowed=set(config['unit_keys'] if production else profile_keys(units))
        pending=[u for u in units if saved[u['key']] is None and u['key'] in allowed]
        if not pending:return dict(state='profile-complete',production_frozen=False,completed=done(),total=len(units))
        if approval_path is None and not production and not (directory/'meta/active-reservation.json').exists():
            raise ValueError('Approve a separately bounded initial-gradient profile allocation first.')
        reservation,remaining=_reservation(config,execution,directory,root,info,approval_path)
        _save_frozen(directory/'meta/runtime.json',info)
        attempt=directory/'attempts'/uuid.uuid4().hex;_save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],started_at=_now(),phase='production' if production else 'profile',runtime_sha256=file_hash(directory/'meta/runtime.json')))
        start=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-start>=remaining:raise TimeoutError('Gradient allocation exhausted; reconcile usage before renewal.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                deadline()
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',info)
                if progress:print('Stage 0 pushes: loading verified initialization; no optimizer.',flush=True)
                engine=deps.engine(config,root)
                for unit in pending:
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic();arrays,proof=engine.measure(unit);validate_arrays(config,unit,arrays,proof,root)
                    integrity=engine.check_integrity()
                    if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Initial weights changed during gradients.')
                    deadline();folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                    temporary=folder/'gradients.tmp.npz';np.savez(temporary,**arrays);temporary.replace(folder/'gradients.npz')
                    _write_json(folder/'metadata.json',dict(seconds=time.monotonic()-begin,proof=proof,integrity=integrity,attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json')))
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)),files={n:file_hash(folder/n) for n in ['gradients.npz','metadata.json']}))
                    saved[unit['key']]=arrays;new+=1;_state(directory,'running',done(),len(units))
                    if progress:print(f'Stage 0 pushes: {done()}/{len(units)} batches sealed; {unit["case"]} {unit["objective"]}; {unit["question_count"]} questions; {unit["token_count"]} tokens.',flush=True)
                deadline();integrity=engine.check_integrity(full=True);_write_json(attempt/'final-integrity.json',integrity)
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full initial-weight integrity failed.')
                load(path,root);deadline()
                if done()==len(units):
                    files=['config.json','inputs/batches.jsonl','inputs/initialization-audit.json','meta/runtime.json','checks/instrument.json']+[str((directory/'units'/u['key']/'complete.json').relative_to(directory)) for u in units]
                    _write_json(directory/'measurement.complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in files}))
                return _state(directory,'measurement-complete' if done()==len(units) else 'incomplete',done(),len(units),production_frozen=production)
        except (Exception,KeyboardInterrupt) as exc:
            # A graceful failure can still seal the immutable-weight evidence for
            # completed batches. A hard kill remains visibly unreconciled.
            if engine is not None and not (attempt/'final-integrity.json').exists():
                try:
                    left=remaining-(time.monotonic()-start)
                    if left>0:
                        with deadline_guard(left,config['execution_kind']=='real-cuda'):
                            _write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except (Exception,KeyboardInterrupt):pass
            _state(directory,'failed',done(),len(units),error=str(exc));log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-start,reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def forecast(path,output_root):
    root=Path(output_root).resolve();config,directory,units=load(path,root);evidence={};rates={};attempts=set()
    for unit in units:
        if unit['key'] not in profile_keys(units):continue
        if read_unit(config,directory,unit,root) is None:raise ValueError('Profile one full batch per A0/objective before production.')
        folder=directory/'units'/unit['key'];meta=json.loads((folder/'metadata.json').read_text());attempts.add(meta['attempt_id'])
        work=sum(len(s['input_ids'])**2 for s in unit['sequences'])
        rates[unit['case']+':'+unit['objective']]=meta['seconds']/work
        evidence[str((folder/'complete.json').relative_to(directory))]=file_hash(folder/'complete.json')
    overhead=0.
    for key in attempts:
        attempt=directory/'attempts'/key;timing=_attempt_timing(attempt)
        if not all(json.loads((attempt/'final-integrity.json').read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Profile full integrity failed.')
        overhead=max(overhead,timing['seconds']);evidence[str((attempt/'timing.complete.json').relative_to(directory))]=file_hash(attempt/'timing.complete.json')
    missing=[u for u in units if read_unit(config,directory,u,root) is None]
    result=dict(config_sha256=_hash(config),remaining_batches=len(missing),profile_batches=len(rates),rates_seconds_per_squared_sequence_token=rates,
        conservative_total_seconds=overhead+sum(rates[u['case']+':'+u['objective']]*sum(len(s['input_ids'])**2 for s in u['sequences']) for u in missing),
        measured_whole_attempt_overhead=overhead,evidence=evidence,completion_guaranteed=False,method='condition-specific measured batch time / summed squared sequence lengths; plus worst measured whole attempt; inspect before allocating')
    _write_json(directory/'results/forecast.json',result);return result


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_=load(path,root);review=json.loads(Path(review_path).read_text())
    expected={'reviewer','notes','session_id','source_reviewed','max_units_per_hour','max_seconds','forecast_sha256'}
    measured=json.loads((directory/'results/forecast.json').read_text())
    frozen_runtime=json.loads((directory/'meta/runtime.json').read_text())
    if set(review)!=expected or review['session_id']!=frozen_runtime['session_id'] or review['forecast_sha256']!=file_hash(directory/'results/forecast.json') or measured['config_sha256']!=_hash(config) or review['max_seconds']<measured['conservative_total_seconds']:raise ValueError('Approve a separately bounded same-session gradient production forecast.')
    for name,digest in measured['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Measured gradient evidence changed.')
    target=inside(root,path).parent/'stage0.pushes.execution.json'
    with _writer(directory):
        if target.exists():
            if _execution(config,directory,root,target)['review']!=review:raise ValueError('Production review changed.')
            return target
        reservation=reserve(root,_hash(config),{k:v for k,v in review.items() if k!='forecast_sha256'})
        _save_frozen(target,dict(config_sha256=_hash(config),review=review,reservation=reservation,forecast=measured))
        _save_frozen(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
        _write_json(directory/'meta/active-reservation.json',dict(reservation_id=reservation['reservation_id']))
    return target


def verify_measurement(path,output_root):
    root=Path(output_root).resolve();config,directory,units=load(path,root);seal=json.loads((directory/'measurement.complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Gradient measurement completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Gradient completed evidence changed.')
    if any(read_unit(config,directory,u,root) is None for u in units):raise ValueError('Missing sealed gradient batch.')
    return dict(measurement_complete=True,batches=len(units),conditions=len(config['cases'])*3,real_gpu_measurement=config['execution_kind']=='real-cuda',cpu_report_pending=not (directory/'complete.json').exists())


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,units=load(path,root)
    return dict(completed=sum(read_unit(config,directory,u,root) is not None for u in units),total=len(units),
        measurement=verify_measurement(path,root) if (directory/'measurement.complete.json').exists() else None,budget=budget_status(root))


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_=load(path,root);return _recover_owner(directory,owner_token,review_notes)
