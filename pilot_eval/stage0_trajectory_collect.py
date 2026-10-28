"""Optional missing-only KL/write collection, reusing the strict Ticket06 engine."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.activation_profile import AGREEMENT
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_trajectory import load as load_parent,verify_cpu
from pilot_eval.stage0_trajectory_sources import identity,merge_products,drift_records
from pilot_eval.stage0_qualification import _load as qualification_load,_verify as qualification_verify
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_drift import _read_unit as read_drift
from pilot_eval.stage0_drift_math import check
from pilot_eval.stage0_evaluation import _reservation,_attempt_timing,_loaded_checks,_jsonl,_recover_owner
from pilot_eval.stage0_budget import reserve
from pilot_eval.stage0_workflow import _writer,_state,_now,_verify_parents,budget_status
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def _deps():
    from pilot_eval.stage0_drift_backend import DriftDependencies
    return DriftDependencies()


def runtime(parent_path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,_,_=load_parent(parent_path,root)
    return dict(runtime=(dependencies or _deps()).runtime(),model_loaded=False,budget=budget_status(root),source_execution_kind=config['execution_kind'])


def prepare(parent_path,qualification_path,output_root,name,*,dependencies=None,states=None,context_chunk=32,workspace_bytes=536870912):
    root=Path(output_root).resolve();safe_name(name);pp,qp=inside(root,parent_path),inside(root,qualification_path)
    parent,pd,results=verify_cpu(pp,root);_,_,rows=load_parent(pp,root)
    qualification,qd=qualification_load(qp,root);diagnostic=qualification_verify(qualification,qd)
    real=parent['execution_kind']=='real-cuda'
    if (diagnostic['execution_kind']=='real-cuda')!=real or qualification['model_identity']!=parent['model_identity']:raise ValueError('Trajectory collection requires corresponding successful qualification sources.')
    receipts=[]
    for p in sorted((qd/'reviews').glob('*/qualification.json')):
        r=json.loads(p.read_text());review=r.get('review',{})
        if r.get('qualified') is True and r.get('config_sha256')==_hash(qualification) and p.parent.name==_hash(review) and review.get('diagnostic_sha256')==file_hash(qd/'complete.json') and review.get('absolute_gate')==diagnostic['proposed_absolute_gate']:receipts.append(p)
    if not receipts:raise ValueError('Approve Ticket04 numerical qualification before optional trajectory collection.')
    for value in [context_chunk,workspace_bytes]:
        if type(value) is not int or value<1:raise ValueError('Positive bounded head chunk/workspace required.')
    info=(dependencies or _deps()).runtime()
    if info['execution_kind']!=parent['execution_kind']:raise ValueError('Real and fixture trajectory runtimes differ.')
    available={s['alias']:s for s in parent['states'] if s['state']=='verified-adapter-payload'}
    aliases=list(available) if states is None else list(states)
    if not aliases or len(set(aliases))!=len(aliases) or any(s not in available for s in aliases):raise ValueError('Select distinct, verified original checkpoint aliases only.')
    product='collection-'+name;ident=identity(parent,info,1,product)
    compatible=[r for u,r in zip(parent['units'],results) if u['kind']=='product' and r['identity']['comparison_group']==ident['comparison_group']]
    reusable=[r for r in merge_products(compatible) if r['alias'] in aliases and r['kl_values'] is not None and r['writes'] is not None and {w['module'] for w in r['writes']}==set(parent['modules'])]
    reused_keys={(r['alias'],r['id'],r['view']) for r in reusable};units=[];profile=[]
    for alias in aliases:
        first=True
        for corpus in ['gsm8k','fineweb']:
            missing=[r['id'] for r in rows if r['corpus']==corpus and not all((alias,r['id'],v) in reused_keys for v in VIEWS if any(r['masks'][v]))]
            groups=[missing[:2]]+[missing[i:i+8] for i in range(2,len(missing),8)] if missing else []
            for gi,ids in enumerate(groups):
                unit=dict(key=f'{alias}-{corpus}-{gi:03d}',variant=alias,ids=ids,diagnostic=first)
                units.append(unit)
                if gi==0:profile.append(unit['key'])
                first=False
    variants={a:dict(adapter_path=available[a]['path']) for a in aliases};adapter_files={a:{n:file_hash(inside(root,variants[a]['adapter_path'])/n) for n in ['adapter_config.json','adapter_model.safetensors','complete.json']} for a in aliases}
    parents={str(p.relative_to(root)):file_hash(p) for p in [pp,pd/'cpu.complete.json',qp,qd/'complete.json',receipts[0]]}
    directory=root/f'runs/stage0/{name}/trajectory-collection';path=root/'plans'/name/'stage0.trajectory.collection.json'
    from pilot_eval.stage0_trajectory import implementation
    config=dict(protocol='stage0-trajectory-collection-v1',schema_version=1,name=name,run_path=str(directory.relative_to(root)),parent_path=str(pp.relative_to(root)),parents=parents,
        qualification_path=str(qp.relative_to(root)),implementation=implementation(),numpy_version=np.__version__,runtime=info,
        execution_kind=parent['execution_kind'],model_identity=parent['model_identity'],weights_dir=parent['weights_dir'],modules=parent['modules'],
        variants=variants,variant_order=aliases,adapter_files=adapter_files,controls=[a for a in aliases if available[a]['step']==0],
        factor_norm_sums={a:available[a]['sum_factor_norm_products'] for a in aliases},items_sha256=parent['items_sha256'],
        units=units,profile_keys=profile,batch_size=1,batches_per_unit=8,context_chunk=context_chunk,workspace_bytes=workspace_bytes,
        decoding=qualification['decoding'],prompt_contract=qualification.get('evaluation_config',{}).get('prompt_template',{}),seed=42,agreement=dict(AGREEMENT),
        execution_sha256=_hash(dict(parent_sha256=file_hash(pp),runtime=info,aliases=aliases)),reuse_sha256=_hash(reusable),
        comparison_identity=ident,coefficient_policy=parent['raw_coefficient_policy'],budget_ceiling_units=20,
        measurements='new KL and effective writes in the same forward/session; full-vocabulary FP32 logits, FP64 KL; no new generation',
        model_call_note='missing sequences only; disabled-reference/switch-restoration plus strict diagnostic repeats/chunk checks add calls')
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config);_jsonl(directory/'inputs/reused.jsonl',reusable)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),missing_adapter_sequence_forwards=sum(len(u['ids']) for u in units),reused_document_view_records=len(reusable),runtime=info,resources='optional bounded CUDA run; CPU verify/report',profile_sequence_forwards=sum(len(u['ids']) for u in units if u['key'] in profile)))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(units))
    return path


def load(path,root):
    from pilot_eval.stage0_trajectory import implementation
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-trajectory-collection-v1' or config['implementation']!=implementation() or config['numpy_version']!=np.__version__:raise ValueError('Trajectory collection implementation/environment changed.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):raise ValueError('Collection plan identity changed.')
    _verify_parents(root,config['parents']);parent,_,_=verify_cpu(inside(root,config['parent_path']),root)
    _,_,rows=load_parent(inside(root,config['parent_path']),root)
    reused=[json.loads(x) for x in (directory/'inputs/reused.jsonl').read_text().splitlines()]
    if _hash(reused)!=config['reuse_sha256']:raise ValueError('Verified reuse mapping changed.')
    for alias,entry in config['variants'].items():
        for n,digest in config['adapter_files'][alias].items():
            if file_hash(inside(root,entry['adapter_path'])/n)!=digest:raise ValueError('Trajectory adapter changed.')
    return config,directory,parent,rows,reused


def _saved(config,directory,rows):
    by_id={r['id']:r for r in rows};cache={}
    return [read_drift(config,directory,u,by_id,cache) for u in config['units']]


def predict(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_,_,_=load(path,root);review=json.loads(Path(review_path).read_text())
    if set(review)!={'reviewer','notes','predictions','prior_exposure'} or type(review['prior_exposure']) is not bool or any(not str(review[k]).strip() for k in ['reviewer','notes','predictions']):raise ValueError('Freeze trajectory hypotheses and prior exposure before collection.')
    with _writer(directory):
        target=directory/'predictions.json'
        if target.exists():
            value=json.loads(target.read_text())
            if value['review']!=review or json.loads((directory/'predictions.complete.json').read_text())!={'sha256':file_hash(target)}:raise ValueError('Trajectory predictions changed.')
            return value
        value=dict(config_sha256=_hash(config),review=review,recorded_at=_now(),status='exploratory' if review['prior_exposure'] or config['reuse_sha256']!=_hash([]) or any((directory/'units').glob('*/complete.json')) else 'frozen-before-new-collection')
        _save_frozen(target,value);_save_frozen(directory/'predictions.complete.json',dict(sha256=file_hash(target)));return value


def _execution(config,directory,root,path):
    value=json.loads(path.read_text())
    if value['config_sha256']!=_hash(config) or json.loads((directory/'production-freeze.json').read_text())!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)):raise ValueError('Trajectory collection freeze changed.')
    for p,digest in value['forecast']['evidence'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Measured trajectory forecast evidence changed.')
    return value


def _finish(config,directory,parent,rows,reused,saved):
    documents=drift_records(config,saved,rows,parent,config['comparison_identity']['comparison_group'])
    # Reused positions are already source-verified. A partially reused document
    # is recomputed whole; replace its old views instead of duplicating them.
    new_keys={(r['alias'],r['id'],r['view']) for r in documents}
    documents+=[r for r in reused if (r['alias'],r['id'],r['view']) not in new_keys]
    documents=sorted(documents,key=lambda r:(r['alias'],r['id'],r['view']))
    _jsonl(directory/'results/per-document.jsonl',documents)
    _write_json(directory/'results/results.json',dict(measurement_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',document_view_records=len(documents),comparison_identity=config['comparison_identity'],coefficients_collected=False))
    files=['config.json','inputs/reused.jsonl','results/per-document.jsonl','results/results.json','predictions.json','predictions.complete.json']+[f'units/{u["key"]}/complete.json' for u in config['units']]
    if config['units']:files+=['meta/runtime.json','meta/loaded-adapters.json']
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}));return _state(directory,'complete',len(saved),len(saved))


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,parent,rows,reused=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid collection limit.')
    prediction=json.loads((directory/'predictions.json').read_text())
    if prediction['config_sha256']!=_hash(config) or json.loads((directory/'predictions.complete.json').read_text())!={'sha256':file_hash(directory/'predictions.json')}:raise ValueError('Prediction receipt changed.')
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        saved=_saved(config,directory,rows);done=lambda:sum(v is not None for v in saved)
        if not config['units']:return _finish(config,directory,parent,rows,reused,saved)
        if max_new_units==0:return _state(directory,'incomplete',done(),len(saved))
        target=inside(root,path).parent/'stage0.trajectory.execution.json';production=target.exists()
        pending=[(i,u) for i,u in enumerate(config['units']) if saved[i] is None and (production or u['key'] in config['profile_keys'])]
        if not pending:return dict(state='profile-complete',completed=done(),total=len(saved),production_frozen=False)
        deps=dependencies or _deps();info=deps.runtime()
        if info!=config['runtime']:raise ValueError('New KL/write pairs must stay in the frozen trajectory session; preserve incompatible runs separately.')
        if approval_path is None and not production and not (directory/'meta/active-reservation.json').exists():raise ValueError('Approve a separately bounded trajectory profile allocation.')
        execution=_execution(config,directory,root,target) if production else {'reservation':{'job_sha256':_hash(config)}}
        reservation,remaining=_reservation(config,execution,directory,root,info,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex;_save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],started_at=_now()))
        start=time.monotonic();engine=None;new=0;by_id={r['id']:r for r in rows}
        def deadline():
            if time.monotonic()-start>=remaining:raise TimeoutError('Trajectory allocation exhausted; reconcile before renewing.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',info);deadline();engine=deps.engine(config,root)
                _save_frozen(directory/'meta/loaded-adapters.json',dict(execution_kind=config['execution_kind'],checks=getattr(engine,'loaded_adapter_checks',[])));_loaded_checks(config,directory)
                for index,unit in pending:
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic();selected=[by_id[i] for i in unit['ids']];value=engine.measure(selected,unit['variant'],diagnostic=unit['diagnostic'])
                    evidence=attempt/'observations'/unit['key'];evidence.mkdir(parents=True,exist_ok=True);np.savez(evidence/'arrays.npz',**value['arrays']);_write_json(evidence/'result.json',{k:v for k,v in value.items() if k!='arrays'})
                    value=check(value,selected,config,unit['variant'],unit['diagnostic']);integrity=engine.check_integrity()
                    if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Trajectory model integrity failed.')
                    deadline();folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True);temporary=folder/'arrays.tmp.npz';np.savez(temporary,**value['arrays']);temporary.replace(folder/'arrays.npz')
                    _write_json(folder/'result.json',{k:v for k,v in value.items() if k!='arrays'});_write_json(folder/'attempt.json',dict(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),integrity=integrity,seconds=time.monotonic()-begin))
                    from pilot_eval.stage0_evaluation import _identity
                    _write_json(folder/'complete.json',dict(identity=_identity(config,unit,_hash(info),config['items_sha256']),files={p:file_hash(folder/p) for p in ['arrays.npz','result.json','attempt.json']}))
                    saved[index]=value;new+=1;_state(directory,'running',done(),len(saved))
                    if progress:print(f'Stage0 trajectory: {done()}/{len(saved)} missing-data groups sealed; {unit["variant"]}; {len(selected)} sequences.',flush=True)
                deadline();integrity=engine.check_integrity(full=True);_write_json(attempt/'final-integrity.json',integrity)
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full trajectory integrity failed.')
                load(path,root);deadline()
                return _finish(config,directory,parent,rows,reused,saved) if done()==len(saved) else _state(directory,'incomplete',done(),len(saved))
        except (Exception,KeyboardInterrupt) as exc:
            if engine is not None and not (attempt/'final-integrity.json').exists():
                try:
                    left=remaining-(time.monotonic()-start)
                    if left>0:
                        with deadline_guard(left,config['execution_kind']=='real-cuda'):_write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except (Exception,KeyboardInterrupt):pass
            _state(directory,'failed',done(),len(saved),error=str(exc));_write_json(attempt/'error.json',dict(error=str(exc),type=type(exc).__name__));raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-start,reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def forecast(path,output_root):
    root=Path(output_root).resolve();config,directory,_,rows,_=load(path,root);saved=_saved(config,directory,rows);by_id={r['id']:r for r in rows};rates={};evidence={};overhead=0.
    for unit,value in zip(config['units'],saved):
        if unit['key'] not in config['profile_keys']:continue
        if value is None:raise ValueError('Complete the bounded first group per missing checkpoint/corpus before production.')
        folder=directory/'units'/unit['key'];meta=json.loads((folder/'attempt.json').read_text());attempt=directory/'attempts'/meta['attempt_id'];timing=_attempt_timing(attempt)
        if not all(json.loads((attempt/'final-integrity.json').read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Trajectory profile full integrity failed.')
        corpus=by_id[unit['ids'][0]]['corpus'];rates[unit['variant']+':'+corpus]=meta['seconds']/sum(len(by_id[i]['input_ids'])**2 for i in unit['ids']);overhead=max(overhead,timing['seconds'])
        for p in [folder/'complete.json',attempt/'timing.complete.json']:evidence[str(p.relative_to(directory))]=file_hash(p)
    missing=[u for u,v in zip(config['units'],saved) if v is None]
    result=dict(config_sha256=_hash(config),remaining_groups=len(missing),remaining_adapter_sequence_forwards=sum(len(u['ids']) for u in missing),rates=rates,evidence=evidence,
        conservative_total_seconds=overhead+sum(rates[u['variant']+':'+by_id[u['ids'][0]]['corpus']]*sum(len(by_id[i]['input_ids'])**2 for i in u['ids']) for u in missing),
        extra_calls=config['model_call_note'],completion_guaranteed=False)
    _write_json(directory/'results/forecast.json',result);return result


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_,_,_=load(path,root);review=json.loads(Path(review_path).read_text());measured=json.loads((directory/'results/forecast.json').read_text())
    required={'reviewer','notes','session_id','source_reviewed','max_units_per_hour','max_seconds','forecast_sha256'}
    if set(review)!=required or review['session_id']!=config['runtime']['session_id'] or review['forecast_sha256']!=file_hash(directory/'results/forecast.json') or measured['config_sha256']!=_hash(config) or review['max_seconds']<measured['conservative_total_seconds']:raise ValueError('Approve a separate measured trajectory production budget in the frozen runtime.')
    for p,digest in measured['evidence'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Trajectory measured evidence changed.')
    target=inside(root,path).parent/'stage0.trajectory.execution.json'
    with _writer(directory):
        if target.exists():
            if _execution(config,directory,root,target)['review']!=review:raise ValueError('Trajectory production review changed.')
            return target
        reservation=reserve(root,_hash(config),{k:v for k,v in review.items() if k!='forecast_sha256'});_save_frozen(target,dict(config_sha256=_hash(config),review=review,reservation=reservation,forecast=measured))
        _save_frozen(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)));_write_json(directory/'meta/active-reservation.json',dict(reservation_id=reservation['reservation_id']))
    return target


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,parent,rows,reused=load(path,root);seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Trajectory collection completion changed.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Trajectory collection completed payload changed.')
    if config['units']:
        if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']:raise ValueError('Trajectory saved runtime changed.')
        _loaded_checks(config,directory)
    saved=_saved(config,directory,rows)
    if any(v is None for v in saved):raise ValueError('Trajectory missing data remains incomplete.')
    expected=drift_records(config,saved,rows,parent,config['comparison_identity']['comparison_group']);keys={(r['alias'],r['id'],r['view']) for r in expected};expected+=[r for r in reused if (r['alias'],r['id'],r['view']) not in keys]
    stored=[json.loads(s) for s in (directory/'results/per-document.jsonl').read_text().splitlines()]
    if sorted(expected,key=lambda r:(r['alias'],r['id'],r['view']))!=stored:raise ValueError('Trajectory raw-position reconstruction differs.')
    return dict(measurement_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',document_view_records=len(stored),comparison_identity=config['comparison_identity'])


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,_,rows,reused=load(path,root)
    return dict(completed=sum(v is not None for v in _saved(config,directory,rows)),total=len(config['units']),reused_records=len(reused),budget=budget_status(root),runtime=config['runtime'])


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_,_,_=load(path,root);return _recover_owner(directory,owner_token,review_notes)
