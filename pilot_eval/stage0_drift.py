"""Frozen fixed-context KL/write collection, bounded GPU attempts, CPU verification."""
import json
import os
import platform
import time
import uuid
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.activation_profile import AGREEMENT
from pilot_eval.run import _write_json
from pilot_eval.sft import read_artifact, safe_name
from pilot_eval.stage0_budget import reserve, reservations
from pilot_eval.stage0_edits import _load as load_edits, VARIANTS
from pilot_eval.stage0_evaluation import (_execution, _intact, _attempt_timing, _loaded_checks,
                                         _jsonl, _identity)
from pilot_eval.stage0_drift_math import check, positions, summarize
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _now, budget_status
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen



def _factor_norms(directory):
    """One read of a previously source-verified FP32 rank-one payload."""
    import math
    payload=(directory/'adapter_model.safetensors').read_bytes()
    length=int.from_bytes(payload[:8],'little')
    if not 1<=length<=len(payload)-8:raise ValueError('Invalid factor payload header.')
    header=json.loads(payload[8:8+length]);norms={};spans=[]
    for name,entry in header.items():
        if name=='__metadata__':continue
        shape=entry['shape'];left,right=entry['data_offsets']
        suffix=next((v for v in ['.lora_A.weight','.lora_B.weight'] if name.endswith(v)),None)
        if (entry['dtype']!='F32' or suffix is None or len(shape)!=2 or min(shape)<1
                or (shape[0] if suffix=='.lora_A.weight' else shape[1])!=1
                or not 0<=left<right<=len(payload)-8-length or right-left!=math.prod(shape)*4):
            raise ValueError('Invalid rank-one FP32 factor mapping.')
        value=np.frombuffer(payload,dtype='<f4',count=math.prod(shape),offset=8+length+left)
        if not np.isfinite(value).all():raise ValueError('Nonfinite factor payload.')
        module=name[:-len(suffix)];norms.setdefault(module,{})[suffix]=float(np.linalg.norm(value.astype(float)))
        spans.append((left,right))
    spans.sort()
    if any(a[1]>b[0] for a,b in zip(spans,spans[1:])) or any(len(v)!=2 for v in norms.values()):
        raise ValueError('Overlapping or incomplete factor mapping.')
    return {module:values['.lora_A.weight']*values['.lora_B.weight'] for module,values in sorted(norms.items())}


def _implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_drift.py','stage0_drift_math.py','stage0_drift_backend.py',
        'stage0_evaluation.py','stage0_evaluation_backend.py','token_measurement_math.py','activation_profile.py']}


def _configuration(execution_path,inputs_path,root,name,batch_size,batches_per_unit,context_chunk,workspace_bytes):
    safe_name(name);execution,qualification,_=_execution(execution_path,root)
    edits,_,audit,_=load_edits(inside(root,qualification['edits_path']),root)
    real=execution['real_gpu_authorized']
    inputs_path=inside(root,inputs_path);prepared=json.loads(inputs_path.read_text())
    if real and inputs_path!=inside(root,audit['sources']['sft']['path']):
        raise ValueError('Use the audited original SFT fixed-input product; no new cohort selection.')
    for k in ['model','model_revision','tokenizer_revision','dtype']:
        if prepared.get(k)!=qualification['model_identity'].get(k):raise ValueError('Fixed-context model/tokenizer identity changed.')
    rows=read_artifact(root,prepared['items_path'])
    if _hash(rows)!=prepared['items_sha256']:raise ValueError('Fixed input payload changed.')
    if (len(rows)!=300 or len({r['id'] for r in rows})!=300
            or {c:sum(r['corpus']==c for r in rows) for c in ['gsm8k','fineweb']}!={'gsm8k':150,'fineweb':150}):
        raise ValueError('Expected the original 150 GSM8K and 150 FineWeb measurement sequences.')
    positions(rows)
    for value in [batch_size,batches_per_unit,context_chunk,workspace_bytes]:
        if type(value) is not int or value<1:raise ValueError('Positive integer batch/chunk/workspace settings required.')
    if batch_size>300 or batches_per_unit>300:raise ValueError('Batch or save group exceeds the fixed cohort.')
    if real and batch_size!=1:
        raise ValueError('Only singleton fixed-context inference is qualified; larger batches need separate agreement evidence.')
    variants=dict(execution['variants']);controls=['untuned','sft-zero','grpo-zero']
    for arm in ['sft','grpo']:
        source=audit['sources'][arm]['checkpoints']['0']
        if source['state']!='verified-adapter-payload' or not source['b_exact_zero']:
            raise ValueError('Both verified zero adapters are required for bounded instrument controls.')
        variants[arm+'-zero']=dict(arm=arm,operation='zero-control',adapter_path=source['path'])
    adapter_files={};factor_norm_sums={};modules=None
    for variant,data in variants.items():
        folder=inside(root,data['adapter_path']);norms=_factor_norms(folder)
        current=list(norms)
        if modules is not None and modules!=current:raise ValueError('Adapter module mapping differs.')
        modules=current
        factor_norm_sums[variant]=float(sum(norms.values()))
        adapter_files[variant]={n:file_hash(folder/n) for n in ['adapter_config.json','adapter_model.safetensors','complete.json']}
    if not real:modules=edits['modules'] # tiny instrument inventory; never qualifies real inference
    factor_norm_sums['untuned']=0.
    diagnostic_ids=[r['id'] for c in ['gsm8k','fineweb'] for r in [r for r in rows if r['corpus']==c][:2]]
    units=[dict(key='control-'+v,variant=v,ids=diagnostic_ids,diagnostic=True) for v in controls]
    width=batch_size*batches_per_unit
    for start in range(0,len(rows),width):
        for variant in VARIANTS:
            units.append(dict(key=f'batch-{start//width:04d}-{variant}',variant=variant,
                ids=[r['id'] for r in rows[start:start+width]],diagnostic=start==0))
    config=dict(schema_version=1,protocol='stage0-fixed-context-v1',name=name,
        run_path=f'runs/stage0/{name}/fixed-context',execution_path=str(inside(root,execution_path).relative_to(root)),
        execution_sha256=file_hash(inside(root,execution_path)),implementation=_implementation(),
        inputs_path=str(inputs_path.relative_to(root)),inputs_sha256=file_hash(inputs_path),
        items_path=prepared['items_path'],items_file_sha256=file_hash(inside(root,prepared['items_path'])),
        items_sha256=prepared['items_sha256'],item_ids=[r['id'] for r in rows],runtime=execution['runtime'],
        execution_kind=execution['runtime']['execution_kind'],weights_dir=qualification['weights_dir'],
        model_identity=qualification['model_identity'],tokenizer_files=qualification['tokenizer_files'],
        decoding=qualification['decoding'],prompt_contract=qualification.get('evaluation_config',{}).get('prompt_template',{}),
        variants=variants,variant_order=list(variants),controls=controls,modules=modules,adapter_files=adapter_files,
        factor_norm_sums=factor_norm_sums,batch_size=batch_size,batches_per_unit=batches_per_unit,
        context_chunk=context_chunk,workspace_bytes=workspace_bytes,units=units,seed=42,agreement=AGREEMENT,
        kl_direction='variant||untuned',padding='right',position_ids='attention-cumsum-minus-one',
        write_definition='direct branch on adapted module inputs; sums of squared norms and per-token norm ratios',
        diagnostic_scope='all four zero-control documents; first document of the first group for each variant; strict repeat/chunk checks',
        coefficient_batch_policy='no batched production coefficients retained; prior descriptive policy unchanged',
        historical_values_used=False,budget_ceiling_units=20)
    return config


def prepare(execution_path,inputs_path,output_root,name,*,batch_size=1,batches_per_unit=8,context_chunk=32,workspace_bytes=536870912):
    root=Path(output_root).resolve()
    config=_configuration(execution_path,inputs_path,root,name,batch_size,batches_per_unit,context_chunk,workspace_bytes)
    path=root/'plans'/name/'stage0.drift.json';directory=inside(root,config['run_path'])
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),
            main_sequence_forwards=3000,control_sequence_forwards=12,total_units=len(config['units']),
            resource='one qualified CUDA GPU for run; CPU prepare/status/verify',
            extra_calls='reference/switch restoration and bounded repeat/chunk checks; forecast from measured attempts',
            full_vocabulary_arrays_saved=False,execution_kind=config['execution_kind']))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(config['units']))
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='stage0-fixed-context-v1' or config.get('implementation')!=_implementation():
        raise ValueError('Drift implementation changed; preserve evidence and use a new execution identity.')
    expected=_configuration(inside(root,config['execution_path']),inside(root,config['inputs_path']),root,config['name'],
        config['batch_size'],config['batches_per_unit'],config['context_chunk'],config['workspace_bytes'])
    directory=inside(root,config['run_path'])
    if (config!=expected or json.loads((directory/'config.json').read_text())!=config
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config)):
        raise ValueError('Frozen drift source/configuration changed.')
    return config,directory


def _rows(config,root):return read_artifact(root,config['items_path'])


def _read_unit(config,directory,unit,by_id,attempt_cache):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if (seal.get('identity')!=_identity(config,unit,_hash(config['runtime']),config['items_sha256'])
            or set(seal.get('files',{}))!={'arrays.npz','result.json','attempt.json'}):
        raise ValueError('Drift batch receipt identity mismatch.')
    for p,digest in seal['files'].items():
        if file_hash(folder/p)!=digest:raise ValueError('Drift sealed payload changed.')
    attempt_info=json.loads((folder/'attempt.json').read_text());key=attempt_info['attempt_id']
    if key not in attempt_cache:
        attempt=directory/'attempts'/key
        final=attempt/'final-integrity.json';timing=_attempt_timing(attempt,required=False)
        if timing and timing['final_integrity_sha256'] is not None:
            if not final.exists() or file_hash(final)!=timing['final_integrity_sha256']:
                raise ValueError('Drift attempt integrity payload changed.')
        if final.exists() and not _intact(config,json.loads(final.read_text())):
            raise ValueError('Failed attempt integrity invalidates its sealed measurements.')
        attempt_cache[key]=file_hash(attempt/'start.json')
    if attempt_info['attempt_start_sha256']!=attempt_cache[key] or not _intact(config,attempt_info['integrity']):
        raise ValueError('Drift attempt identity/integrity changed.')
    value=json.loads((folder/'result.json').read_text())
    with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:value['arrays']={k:arrays[k] for k in arrays.files}
    return check(value,[by_id[i] for i in unit['ids']],config,unit['variant'],unit['diagnostic'])


def _saved(config,directory,root):
    by_id={r['id']:r for r in _rows(config,root)};cache={}
    if any((directory/'units').glob('*/complete.json')):
        if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']:
            raise ValueError('Saved drift runtime changed.')
        _loaded_checks(config,directory)
    return [_read_unit(config,directory,u,by_id,cache) for u in config['units']]


def _reservation(config,directory,root,runtime,approval_path):
    usage=budget_status(root)
    if usage['state']!='observed' or usage['spent_units']>=20:raise ValueError('Observed cumulative Stage 0 usage must be below 20 units.')
    if approval_path is not None:
        approval=json.loads(Path(approval_path).read_text())
        if approval.get('session_id')!=runtime['session_id']:raise ValueError('Drift budget approval session differs.')
        row=reserve(root,_hash(config),approval)
        _write_json(directory/'meta/active-reservation.json',dict(reservation_id=row['reservation_id']))
    else:
        selected=directory/'meta/active-reservation.json'
        if not selected.exists():raise ValueError('Drift requires its own bounded --approval; greedy approval does not cover this workload.')
        row=reservations(root).get(json.loads(selected.read_text())['reservation_id'])
    if (not row or row['state']!='active' or row['job_sha256']!=_hash(config)
            or row['approval']['session_id']!=runtime['session_id']):
        raise ValueError('No active matching drift reservation; reconcile usage and approve a new bound.')
    used=sum(_attempt_timing(p.parent)['seconds'] for p in (directory/'attempts').glob('*/start.json')
             if json.loads(p.read_text())['reservation_id']==row['reservation_id'])
    remaining=row['approval']['max_seconds']-used
    if remaining<=0:raise ValueError('Drift reserved time exhausted; reconcile usage.')
    return row,remaining


def _progress(config,directory,saved,state,**extra):
    return _state(directory,state,sum(b is not None for b in saved),len(saved),**extra)


def _summary(config,directory,root,saved):
    result=summarize(config,saved,_rows(config,root))
    result['module_factor_norms_path']='inputs/module-factor-norms.json'
    return result


def _finish(config,directory,root,saved):
    result=_summary(config,directory,root,saved)
    _jsonl(directory/'results/per-document.jsonl',result['per_document'])
    _write_json(directory/'results/results.json',result)
    required=['config.json','meta/runtime.json','meta/loaded-adapters.json','meta/final-integrity.json',
              'inputs/module-factor-norms.json','results/results.json','results/per-document.jsonl']
    required += [f'units/{u["key"]}/complete.json' for u in config['units']]
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in required}))
    _progress(config,directory,saved,'complete')
    return result


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid unit limit.')
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory,root)
        saved=_saved(config,directory,root)
        if max_new_units==0:return _progress(config,directory,saved,'incomplete')
        from pilot_eval.stage0_drift_backend import DriftDependencies
        from pilot_eval.stage0_qualification_backend import deadline_guard
        deps=dependencies or DriftDependencies();runtime=deps.runtime()
        if runtime!=config['runtime']:raise ValueError('Drift requires the exact qualified runtime/session; do not silently join a new environment.')
        reservation,remaining=_reservation(config,directory,root,runtime,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],started_at=_now(),remaining_seconds=remaining))
        started=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-started>=remaining:raise TimeoutError('Reserved drift wall time exhausted; reconcile usage.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                deadline()
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',runtime)
                engine=deps.engine(config,root)
                _save_frozen(directory/'meta/loaded-adapters.json',dict(execution_kind=config['execution_kind'],checks=getattr(engine,'loaded_adapter_checks',[])))
                _loaded_checks(config,directory)
                norms={v:_factor_norms(inside(root,d['adapter_path'])) for v,d in config['variants'].items()}
                _save_frozen(directory/'inputs/module-factor-norms.json',norms)
                by_id={r['id']:r for r in _rows(config,root)}
                for i,unit in enumerate(config['units']):
                    if saved[i] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic()
                    rows=[by_id[j] for j in unit['ids']]
                    result=engine.measure(rows,unit['variant'],diagnostic=unit['diagnostic'])
                    # Failed observations stay inspectable; only validated payloads get seals.
                    evidence=attempt/'observations'/unit['key'];evidence.mkdir(parents=True,exist_ok=True)
                    with (evidence/'arrays.npz').open('wb') as stream:np.savez_compressed(stream,**result['arrays'])
                    _write_json(evidence/'result.json',{k:v for k,v in result.items() if k!='arrays'})
                    result=check(result,rows,config,unit['variant'],unit['diagnostic'])
                    deadline();integrity=engine.check_integrity()
                    if not _intact(config,integrity):raise ValueError('Drift model-integrity guard failed.')
                    folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'arrays.npz.tmp').open('wb') as stream:
                        np.savez_compressed(stream,**result['arrays'])
                    (folder/'arrays.npz.tmp').replace(folder/'arrays.npz')
                    _write_json(folder/'result.json',{k:v for k,v in result.items() if k!='arrays'})
                    _write_json(folder/'attempt.json',dict(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),
                        integrity=integrity,seconds=time.monotonic()-begin))
                    _write_json(folder/'complete.json',dict(identity=_identity(config,unit,_hash(runtime),config['items_sha256']),
                        files={p:file_hash(folder/p) for p in ['arrays.npz','result.json','attempt.json']}))
                    saved[i]=result;new+=1;_progress(config,directory,saved,'running',reservation_id=reservation['reservation_id'])
                    if progress:print(f'Stage 0 fixed context: {sum(b is not None for b in saved)}/{len(saved)} sealed; {unit["variant"]}; {len(rows)} documents.',flush=True)
                deadline();integrity=engine.check_integrity(full=True);deadline()
                _write_json(attempt/'final-integrity.json',integrity)
                if not _intact(config,integrity):raise ValueError('Full drift integrity failed; preserve this attempt.')
                if all(b is not None for b in saved):
                    _write_json(directory/'meta/final-integrity.json',integrity)
                    return _finish(config,directory,root,saved)
                return _progress(config,directory,saved,'incomplete',reservation_id=reservation['reservation_id'])
        except (Exception,KeyboardInterrupt) as exc:
            _progress(config,directory,saved,'failed',error=str(exc))
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__,attempt_id=attempt.name))+'\n')
            raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,newly_sealed_batches=new,
                    reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),
                    final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None,
                    usage_reconciled=False,reservation_stays_charged=True))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def _verify(config,directory,root):
    seal=json.loads((directory/'complete.json').read_text())
    required={'config.json','meta/runtime.json','meta/loaded-adapters.json','meta/final-integrity.json',
              'inputs/module-factor-norms.json','results/results.json','results/per-document.jsonl'}
    required|={f'units/{u["key"]}/complete.json' for u in config['units']}
    if seal.get('config_sha256')!=_hash(config) or set(seal.get('files',{}))!=required:raise ValueError('Drift completion identity mismatch.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Drift completion payload changed.')
    if not _intact(config,json.loads((directory/'meta/final-integrity.json').read_text())):raise ValueError('Missing final model integrity.')
    saved=_saved(config,directory,root)
    if any(b is None for b in saved):raise ValueError('Drift collection is incomplete.')
    result=_summary(config,directory,root,saved)
    records=[json.loads(s) for s in (directory/'results/per-document.jsonl').read_text().splitlines()]
    if result!=json.loads((directory/'results/results.json').read_text()) or records!=result['per_document']:
        raise ValueError('Drift aggregates disagree with sealed per-position measurements.')
    return result


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    return _verify(config,directory,root)


def status(path,output_root):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    saved=_saved(config,directory,root)
    complete=(directory/'complete.json').exists()
    if complete:_verify(config,directory,root)
    owner=directory/'writer.json'
    return dict(measurement_complete=complete,completed_units=len(saved) if complete else sum(b is not None for b in saved),
        total_units=len(saved),execution_kind=config['execution_kind'],result_directory=str(directory/'results'),
        budget=budget_status(root),writer=json.loads(owner.read_text()) if owner.exists() else None,
        active_reservations=[r for r in reservations(root).values() if r['state']=='active'])


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory=_load(path,root)
    if not owner_token or not review_notes.strip():raise ValueError('Explicit owner token and recovery notes required.')
    with run_lock(directory):
        marker=directory/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Writer token mismatch.')
        if owner['host']==platform.node():
            try:os.kill(owner['pid'],0)
            except ProcessLookupError:pass
            else:raise ValueError('Recorded writer is still active.')
        log=directory/'logs/ownership-recovery.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(owner=owner,notes=review_notes,updated_at=_now()))+'\n')
            stream.flush();os.fsync(stream.fileno())
        marker.unlink()
    return dict(state='ownership-released',scope='writer only; reservation and sealed measurements retained')
