"""Transactional raw capture, local staging and verified durable batch units."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from pilot_eval.trace_production_plan import load_execution
from pilot_eval.trace_capture import summarize_pairs
from pilot_eval.trace_corpus import _inside
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash,_save_frozen
from pilot_eval.run import _write_json,_write_state
from pilot_eval.activation_workflow import _save_arrays


def _unit(folder,identity):
    marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal['identity']!=identity or set(seal['files'])!={'arrays.npz','evidence.json'}:
        raise ValueError('Capture unit identity or required inventory mismatch.')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Capture unit payload hash mismatch; sealed corruption requires inspection.')
    return json.loads((folder/'evidence.json').read_text())


def _identity(config,runtime,index):
    return dict(config_sha256=_hash(config),runtime=runtime,batch=index,indices=config['batches'][index],conditions=config['conditions'])


def _space(path,required,deps):
    path.mkdir(parents=True,exist_ok=True)
    free=deps.free_bytes(path) if hasattr(deps,'free_bytes') else shutil.disk_usage(path).free
    if free<required:raise OSError(f'Insufficient free space at {path}: need {required} bytes, available {free}. Accepted units are preserved.')


def _publish(stage,destination,identity):
    if _unit(destination,identity) is not None:return
    if _unit(stage,identity) is None:raise ValueError('Cannot publish an incomplete staged capture.')
    destination.mkdir(parents=True,exist_ok=True)
    seal=json.loads((stage/'complete.json').read_text())
    for name,digest in seal['files'].items():
        target=destination/name
        if target.exists() and file_hash(target)==digest:continue
        temporary=destination/(name+'.partial')
        shutil.copyfile(stage/name,temporary)
        if file_hash(temporary)!=digest:raise OSError('Durable sync hash mismatch; staged evidence retained.')
        temporary.replace(target)
    _write_json(destination/'complete.json',seal)
    _unit(destination,identity)


def _pack(base,adapted,rows,condition,config):
    summary=summarize_pairs([dict(example_ids=[r['id'] for r in rows],content_positions=[r['content_positions'] for r in rows],
        base=base,adapted=adapted)],sites=config['sites'],retained_sites=config['retained_sites'])
    arrays={}
    for site in config['sites']:
        prefix=condition+'.'
        arrays[prefix+'position_sum.'+site]=summary['position_delta_sums'][site]
        arrays[prefix+'primary_delta.'+site]=summary['example_primary_delta'][site]
        arrays[prefix+'primary_base.'+site]=summary['example_primary_base'][site]
        arrays[prefix+'primary_delta_norm.'+site]=np.linalg.norm(summary['example_primary_delta'][site],axis=1)
        arrays[prefix+'primary_base_norm.'+site]=np.linalg.norm(summary['example_primary_base'][site],axis=1)
        positions=np.stack([adapted[site][i,r['content_positions']].astype(np.float64)-base[site][i,r['content_positions']]
                            for i,r in enumerate(rows)])
        arrays[prefix+'position_norm_sum.'+site]=np.linalg.norm(positions,axis=2).sum(axis=0)
    for site,values in summary['retained_delta'].items():
        converted=values.astype(config['storage_dtype'])
        if not np.isfinite(converted).all():raise ValueError(f'Retention conversion overflow at {condition}/{site}; no unit published.')
        arrays[condition+'.retained_delta.'+site]=converted
    if condition=='base':
        arrays['base.moment_sum']=summary['base_moments']['sum']
        arrays['base.moment_outer_sum']=summary['base_moments']['outer_sum']
    return arrays


def _read_arrays(folder):
    with np.load(folder/'arrays.npz',allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def _finish(config,directory,runtime,new,reused):
    aggregate={};ids=[]
    for index,indices in enumerate(config['batches']):
        folder=directory/'units'/f'{index:06d}'
        evidence=_unit(folder,_identity(config,runtime,index))
        if evidence is None:raise ValueError('Missing durable batch; cannot release GPU.')
        expected=[config['examples'][i]['id'] for i in indices]
        if evidence['example_ids']!=expected:raise ValueError('Durable capture example order changed.')
        ids.extend(expected)
        for key,value in _read_arrays(folder).items():
            if '.position_sum.' in key or '.position_norm_sum.' in key or key in ['base.moment_sum','base.moment_outer_sum']:
                aggregate[key]=aggregate.get(key,np.zeros_like(value))+value
            elif '.primary_delta.' in key or '.primary_base.' in key:
                target=key.replace('.primary_delta.','.primary_delta_sum.').replace('.primary_base.','.primary_base_sum.')
                aggregate[target]=aggregate.get(target,np.zeros(value.shape[1],dtype=np.float64))+value.sum(axis=0)
        print(f'Capture aggregate: {index+1}/{len(config["batches"])} durable batches.',flush=True)
    if ids!=[r['id'] for r in config['examples']]:raise ValueError('Accepted corpus inventory differs from frozen plan.')
    _save_arrays(directory/'results/aggregate.npz',aggregate)
    result=dict(capture_complete=True,durable_verified=True,gpu_release_safe=True,production_scientific_success=None,
        examples=len(ids),example_ids=ids,conditions=config['conditions'],sites=config['sites'],
        batches=len(config['batches']),new_batches=new,reused_batches=reused,base_moment_count=len(ids)*127,
        sequence_forwards=len(ids)*len(config['conditions']),model_calls=len(config['batches'])*len(config['conditions']),
        storage_dtype=config['storage_dtype'],capture_precision='float32',accumulation='float64',
        unit_policy=config['unit_policy'],aggregate='results/aggregate.npz',retained_shards='units/*/arrays.npz',
        execution_kind=runtime['execution_kind'],primary=config['primary'],review=config['review'],
        note='Capture completion is not a task-readability result; no learned token lists revealed.')
    _write_json(directory/'results/results.json',result)
    names=['meta/runtime.json','meta/run_manifest.json','results/aggregate.npz','results/results.json']
    names.extend(f'units/{i:06d}/complete.json' for i in range(len(config['batches'])))
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in names}))
    _write_state(directory,'completed',len(config['batches']),len(config['batches']))
    return result


def verify_capture(path,output_root):
    root=Path(output_root).resolve();config=load_execution(path,root);directory=_inside(root,config['run_path'])
    marker=json.loads((directory/'complete.json').read_text())
    required={'meta/runtime.json','meta/run_manifest.json','results/aggregate.npz','results/results.json'}
    required|={f'units/{i:06d}/complete.json' for i in range(len(config['batches']))}
    if marker['config_sha256']!=_hash(config) or set(marker['files'])!=required:
        raise ValueError('Capture completion seal omits or changes required inventory.')
    for name,digest in marker['files'].items():
        if file_hash(_inside(directory,name))!=digest:raise ValueError('Capture completion hash mismatch.')
    runtime=json.loads((directory/'meta/runtime.json').read_text())
    for i in range(len(config['batches'])):
        if _unit(directory/'units'/f'{i:06d}',_identity(config,runtime,i)) is None:raise ValueError('Missing durable capture unit.')
    return json.loads((directory/'results/results.json').read_text())


def load_aggregate(path,output_root):
    root=Path(output_root).resolve();verify_capture(path,root);config=load_execution(path,root)
    with np.load(root/config['run_path']/'results/aggregate.npz',allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def run_capture(path,output_root,*,staging_root=None,dependencies=None):
    root=Path(output_root).resolve();config=load_execution(path,root);directory=_inside(root,config['run_path'])
    stage_root=Path(staging_root).resolve() if staging_root else root/'staging/pilot-5'
    stage_root=stage_root/config['name'];stage_root.mkdir(parents=True,exist_ok=True)
    with run_lock(directory),run_lock(stage_root):
        if (directory/'complete.json').exists():return verify_capture(path,root)
        from pilot_eval.trace_production_backend import ProductionDependencies
        deps=dependencies or ProductionDependencies();engine=None;completed=0;phase='verify-runtime';start=time.monotonic()
        try:
            runtime=deps.runtime()
            if runtime['execution_kind']!=config['execution_kind']:raise ValueError('Qualification execution kind does not match requested runtime.')
            comparable={k:v for k,v in runtime.items() if k not in ['git_commit','profile_batches']}
            if comparable!=config['qualified_runtime']:raise ValueError('Production runtime differs from the qualified runtime; review a new execution or explicit bridge.')
            _save_frozen(directory/'meta/runtime.json',runtime)
            accepted={i:_unit(directory/'units'/f'{i:06d}',_identity(config,runtime,i)) for i in range(len(config['batches']))}
            completed=sum(v is not None for v in accepted.values());reused=completed
            bytes_used=sum(v['durable_payload_bytes'] for v in accepted.values() if v)
            for index,indices in enumerate(config['batches']):
                if accepted[index]:continue
                elapsed=time.monotonic()-start
                if elapsed>=config['review']['budget']['max_seconds']:raise RuntimeError('Reviewed time budget reached; sealed batches preserved for review.')
                rows=[config['examples'][i] for i in indices];identity=_identity(config,runtime,index)
                stage=stage_root/f'{index:06d}';destination=directory/'units'/f'{index:06d}'
                phase='space-preflight'
                # Raw staging plus reduced arrays and atomic copy allowance. No compression assumption.
                estimated=len(rows)*128*config['width']*len(config['sites'])*8*len(config['conditions'])+config['width']**2*8
                _space(stage_root,2*estimated,deps);_space(directory,2*estimated,deps)
                staged=_unit(stage,identity)
                if staged is None:
                    if engine is None:
                        phase='model-load';print('Capture phase: model load (once this attempt).',flush=True)
                        engine=deps.engine(config,root)
                    phase='base-hash-before';started=time.monotonic();before=engine.integrity()
                    if before!=config['expected_base_hash']:raise ValueError('Frozen base hash mismatch before batch.')
                    hash_seconds=time.monotonic()-started
                    arrays={};base=None;forward_seconds=0.
                    for condition in config['conditions']:
                        phase='forward';started=time.monotonic();values=engine.capture(rows,condition);forward_seconds+=time.monotonic()-started
                        if set(values)!=set(config['sites']):raise ValueError('Missing production capture site.')
                        for site,value in values.items():
                            if np.asarray(value).dtype!=np.float32 or value.shape!=(len(rows),128,config['width']) or not np.isfinite(value).all():
                                raise ValueError(f'Invalid capture at {condition}/{site}.')
                        if condition=='base':base=values
                        arrays.update(_pack(base,values,rows,condition,config))
                        print(f'Capture batch {index+1}/{len(config["batches"])}: {condition}; {len(rows)} documents; staged, not yet accepted.',flush=True)
                    phase='base-hash-after';started=time.monotonic()
                    if engine.integrity()!=before:raise ValueError('Frozen base changed; batch remains unpublished.')
                    hash_seconds+=time.monotonic()-started
                    phase='stage-save';started=time.monotonic();_save_arrays(stage/'arrays.npz',arrays)
                    evidence=dict(example_ids=[r['id'] for r in rows],conditions=config['conditions'],
                        base_unchanged=True,hash_seconds=hash_seconds,forward_capture_transfer_seconds=forward_seconds,
                        stage_save_seconds=time.monotonic()-started,durable_payload_bytes=(stage/'arrays.npz').stat().st_size,
                        storage_dtype=config['storage_dtype'],base_moment_count=len(rows)*127)
                    _write_json(stage/'evidence.json',evidence)
                    _write_json(stage/'complete.json',dict(identity=identity,files={n:file_hash(stage/n) for n in ['arrays.npz','evidence.json']}))
                    staged=evidence
                if bytes_used+staged['durable_payload_bytes']>config['review']['budget']['max_durable_bytes']:
                    raise RuntimeError('Reviewed durable storage budget reached; staged batch preserved; request review before continuing.')
                phase='durable-sync';started=time.monotonic();_publish(stage,destination,identity)
                bytes_used+=staged['durable_payload_bytes'];completed+=1
                _write_json(directory/'checkpoints/progress.json',dict(sealed_batches=completed,total_batches=len(config['batches']),last_batch=index))
                _write_state(directory,'running',completed,len(config['batches']))
                # This staging directory is owned by this workflow; verified durable files now own the evidence.
                for name in ['complete.json','arrays.npz','evidence.json']:(stage/name).unlink()
                elapsed=time.monotonic()-start;new=completed-reused
                eta=(len(config['batches'])-completed)*elapsed/new if new else None
                print(f'Capture sealed {completed}/{len(config["batches"])} batches; sync {time.monotonic()-started:.1f}s; estimated remaining {eta/60:.1f} min (observed this attempt).',flush=True)
            if engine:engine.close();engine=None
            phase='durable-aggregate';result=_finish(config,directory,runtime,completed-reused,reused)
            verify_capture(path,root)
            print('All required capture evidence is durably verified; GPU capture stage is safe to release.',flush=True)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',completed,len(config['batches']),str(exc))
            error=directory/'logs/errors.jsonl';error.parent.mkdir(parents=True,exist_ok=True)
            with error.open('a') as f:f.write(json.dumps(dict(phase=phase,error=str(exc),type=type(exc).__name__,sealed_batches=completed))+'\n')
            raise
        finally:
            if engine:engine.close()
