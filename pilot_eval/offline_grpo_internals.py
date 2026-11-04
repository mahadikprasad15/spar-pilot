"""Verified offline adapter sources and additive internal-measurement planning."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.grpo_training import CHECKPOINTS, _verified_checkpoints
from pilot_eval.offline_grpo_training import _load as load_training, _verify_complete
from pilot_eval.stage0_sources import inside, factors
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.stage0_workflow import _writer
from pilot_eval.sft import safe_name


def product_metrics(left,right):
    """Gauge-invariant Frobenius norms and cosine without dense products."""
    a,b=(np.asarray(v,dtype=np.float64).reshape(-1) for v in left)
    c,d=(np.asarray(v,dtype=np.float64).reshape(-1) for v in right)
    if a.shape!=c.shape or b.shape!=d.shape or not all(np.isfinite(x).all() for x in [a,b,c,d]):
        raise ValueError('Rank-one product dimensions or values differ')
    n=float(np.linalg.norm(a)*np.linalg.norm(b));m=float(np.linalg.norm(c)*np.linalg.norm(d))
    cosine=None if n==0 or m==0 else float((a@c)*(b@d)/(n*m))
    return dict(norm_left=n,norm_right=m,cosine=None if cosine is None else float(np.clip(cosine,-1.,1.)))


def _identity():
    from pilot_eval import stage0_drift_backend,stage0_drift_math,stage0_push_backend,stage0_push_math,stage0_spectral_math,stage0_intruder_math,stage0_weights
    return {k:file_hash(Path(v.__file__)) for k,v in dict(workflow=__import__(__name__,fromlist=['']),
        drift=stage0_drift_backend,drift_math=stage0_drift_math,push=stage0_push_backend,
        push_math=stage0_push_math,spectral_math=stage0_spectral_math,intruder_math=stage0_intruder_math,weights=stage0_weights).items()}


def source(path,root):
    path=inside(root,path);training,directory,frozen,prepared,windows,_=load_training(path,root)
    _verify_complete(directory,root,training,training['optimizer_groups'])
    checkpoints=_verified_checkpoints(directory)
    if sorted(checkpoints)!=CHECKPOINTS:raise ValueError('Offline training checkpoints are incomplete')
    evidence={}
    for step,checkpoint in checkpoints.items():
        evidence[str(step)]=dict(path=str(checkpoint.relative_to(root)),
            complete_sha256=file_hash(checkpoint/'complete.json'),
            adapter_sha256=file_hash(checkpoint/'adapter_model.safetensors'),
            config_sha256=file_hash(checkpoint/'adapter_config.json'))
    return dict(training_path=str(path.relative_to(root)),training_sha256=file_hash(path),
        training_complete_sha256=file_hash(directory/'complete.json'),
        model_identity={**{k:prepared[k] for k in ['model','model_revision','tokenizer_revision','dtype']},
                        'base_sha256':prepared['source_evidence']['base_sha256']},
        pool_path=training['pool_path'],pool_sha256=training['pool_sha256'],
        runtime=training['runtime'],prompt_contract=training['prompt_template'],checkpoints=evidence),windows


def prepare(training_path,spectral_path,inputs_path,runtime_path,output_root,name,*,qualification_path,online_path=None):
    from pilot_eval.stage0_spectral import _load as spectral_load,_verify_cache
    from pilot_eval.activation_prepare import load_prepared
    from pilot_eval.stage0_drift_math import positions
    root=Path(output_root).resolve();safe_name(name)
    training_path,spectral_path,inputs_path,runtime_path,qualification_path=[inside(root,p) for p in [training_path,spectral_path,inputs_path,runtime_path,qualification_path]]
    from pilot_eval.stage0_qualification import _load as qualification_load,_verify as qualification_verify
    qualification,qualification_dir=qualification_load(qualification_path,root)
    qualification_result=qualification_verify(qualification,qualification_dir)
    approved=[p for p in (qualification_dir/'reviews').glob('*/qualification.json') if json.loads(p.read_text()).get('qualified') is True and json.loads(p.read_text()).get('config_sha256')==_hash(qualification) and json.loads(p.read_text()).get('review',{}).get('diagnostic_sha256')==file_hash(qualification_dir/'complete.json') and p.parent.name==_hash(json.loads(p.read_text()).get('review',{}))]
    if not approved:raise ValueError('Approved Stage 0 runtime qualification required')
    offline,windows=source(training_path,root)
    spectral,audit=spectral_load(spectral_path,root);_verify_cache(spectral,inside(root,spectral['run_path']))
    if audit['sources']['sft']['identity']!=offline['model_identity'] or qualification['model_identity']!=offline['model_identity']:
        raise ValueError('Spectral or qualification model differs')
    prepared,rows=load_prepared(inputs_path,root);positions(rows)
    if (_hash(rows)!=prepared['items_sha256'] or len(rows)!=300 or len({r['id'] for r in rows})!=300
        or {c:sum(r['corpus']==c for r in rows) for c in ['gsm8k','fineweb']}!={'gsm8k':150,'fineweb':150}):
        raise ValueError('Expected verified 150 GSM8K and 150 FineWeb fixed inputs')
    if any(prepared[k]!=offline['model_identity'][k] for k in ['model','model_revision','tokenizer_revision','dtype']):raise ValueError('Fixed input model identity differs')
    if prepared['source_evidence']['base_sha256']!=offline['model_identity']['base_sha256']:raise ValueError('Fixed input base identity differs')
    online=None
    if online_path is not None:
        from pilot_eval.grpo_evaluation import _source as online_source
        online_path=inside(root,online_path)
        old,_,old_plan,_,old_checkpoints=online_source(online_path,root)
        if any(old_plan[k]!=offline['model_identity'][k] for k in ['model','model_revision','tokenizer_revision','dtype']) or old_plan['source_evidence']['base_sha256']!=offline['model_identity']['base_sha256']:
            raise ValueError('Online source model differs')
        online=dict(path=str(online_path.relative_to(root)),sha256=file_hash(online_path),
            complete_sha256=file_hash(root/old['run_path']/'complete.json'),
            checkpoints={str(k):dict(path=str(v.relative_to(root)),seal_sha256=file_hash(v/'complete.json'),adapter_sha256=file_hash(v/'adapter_model.safetensors')) for k,v in old_checkpoints.items()})
    runtime=json.loads(runtime_path.read_text())
    if not runtime.get('session_id'):raise ValueError('Explicit bounded runtime identity required')
    if runtime!=qualification_result['runtime'] or json.loads(approved[0].read_text()).get('runtime')!=runtime:
        raise ValueError('Qualification and collection runtime sessions differ')
    directory=root/'runs/offline-grpo'/name/'internals';path=root/'plans'/name/'offline.internals.json'
    config=dict(protocol='offline-grpo-internals-v1',name=name,run_path=str(directory.relative_to(root)),source=offline,online=online,
        spectral_path=str(spectral_path.relative_to(root)),spectral_sha256=file_hash(spectral_path),
        inputs_path=str(inputs_path.relative_to(root)),inputs_sha256=file_hash(inputs_path),
        items_sha256=_hash(rows),runtime_path=str(runtime_path.relative_to(root)),runtime_sha256=file_hash(runtime_path),runtime=runtime,
        execution_kind=runtime['execution_kind'],weights_dir=spectral['weights_dir'],modules=list(spectral['inventory']['modules']),
        qualification_path=str(qualification_path.relative_to(root)),qualification_sha256=file_hash(qualification_path),
        qualification_complete_sha256=file_hash(qualification_dir/'complete.json'),qualification_review_path=str(approved[0].relative_to(root)),qualification_review_sha256=file_hash(approved[0]),
        decoding=qualification['decoding'],tokenizer_files=qualification['tokenizer_files'],
        implementation=_identity(),note='Geometry and slopes are descriptive; matched-norm KL requires a separate scaled-adapter measurement.')
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _save_frozen(directory/'inputs/items.json',rows)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),pool_questions=len(windows)*8,measurement_items=len(rows)))
    return path


def load(path,root):
    from pilot_eval.stage0_spectral import _load as spectral_load,_verify_cache
    root=Path(root).resolve();path=inside(root,path);config=json.loads(path.read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='offline-grpo-internals-v1' or config['implementation']!=_identity() or json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Offline internals configuration or implementation changed')
    for k in ['spectral','inputs','runtime','qualification','qualification_review']:
        if file_hash(inside(root,config[k+'_path']))!=config[k+'_sha256']:raise ValueError('Offline internals source changed')
    from pilot_eval.stage0_qualification import _load as qualification_load,_verify as qualification_verify
    qualification,qualification_dir=qualification_load(root/config['qualification_path'],root)
    qualification_result=qualification_verify(qualification,qualification_dir)
    if qualification_result['runtime']!=config['runtime'] or json.loads((root/config['qualification_review_path']).read_text()).get('runtime')!=config['runtime']:
        raise ValueError('Qualification and collection runtime sessions differ')
    if file_hash(qualification_dir/'complete.json')!=config['qualification_complete_sha256']:
        raise ValueError('Qualification completion changed')
    if qualification['decoding']!=config['decoding'] or qualification['tokenizer_files']!=config['tokenizer_files']:
        raise ValueError('Qualification tokenizer or decoding changed')
    actual,windows=source(root/config['source']['training_path'],root)
    if config['online'] is not None:
        from pilot_eval.grpo_evaluation import _source as online_source
        oldpath=root/config['online']['path']
        if file_hash(oldpath)!=config['online']['sha256']:raise ValueError('Online source changed')
        old,_,old_plan,_,checkpoints=online_source(oldpath,root)
        if file_hash(root/old['run_path']/'complete.json')!=config['online']['complete_sha256']:
            raise ValueError('Online source completion changed')
        if {str(k):dict(path=str(v.relative_to(root)),seal_sha256=file_hash(v/'complete.json'),adapter_sha256=file_hash(v/'adapter_model.safetensors')) for k,v in checkpoints.items()}!=config['online']['checkpoints']:
            raise ValueError('Online checkpoint inventory changed')
    if actual!=config['source']:raise ValueError('Offline training source changed')
    spectral,audit=spectral_load(root/config['spectral_path'],root);cache=_verify_cache(spectral,inside(root,spectral['run_path']))
    if audit['sources']['sft']['identity']!=actual['model_identity']:raise ValueError('Spectral model identity changed')
    from pilot_eval.activation_prepare import load_prepared
    prepared,rows=load_prepared(root/config['inputs_path'],root)
    if _hash(rows)!=config['items_sha256'] or rows!=json.loads((directory/'inputs/items.json').read_text()):raise ValueError('Fixed input payload changed')
    return config,directory,windows,rows,spectral,cache


def run_geometry(path,output_root,*,max_new_units=None):
    """Compute verified product directions and inherited exact-SVD intruder geometry."""
    from pilot_eval.run import _write_json
    from pilot_eval.stage0_intruder_math import measure
    from pilot_eval.stage0_spectral_math import projection,LEVELS
    from pilot_eval.stage0_weights import LocalWeights
    root=Path(output_root).resolve();config,directory,_,_,spectral,cache=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid unit limit')
    off={int(step):factors(inside(root,ref['path'])) for step,ref in config['source']['checkpoints'].items()}
    online={} if config['online'] is None else {int(step):factors(inside(root,ref['path'])) for step,ref in config['online']['checkpoints'].items()}
    metadata={row['module']:row for row in cache['module_results']};new=0
    with _writer(directory/'geometry'):
        for index,module in enumerate(config['modules']):
            folder=directory/'geometry/units'/f'{index:03d}';marker=folder/'complete.json'
            if marker.exists():
                seal=json.loads(marker.read_text())
                if seal['config_sha256']!=_hash(config) or any(file_hash(folder/k)!=v for k,v in seal['files'].items()):raise ValueError('Geometry unit changed')
                continue
            if max_new_units is not None and new>=max_new_units:break
            row=metadata[module];base_path=inside(root,spectral['run_path'])/row['arrays_path']
            with np.load(base_path,allow_pickle=False) as arrays:old={k:arrays[k] for k in ['sigma','u','v']}
            entry=spectral['inventory']['modules'][module]
            matrix=LocalWeights().matrix({**entry,'path':str(inside(root,entry['path']))})
            projections={};norms={};comparisons={};geometry={};retained={}
            for step in CHECKPOINTS:
                a,b=off[step][module];metrics=product_metrics((a,b),(a,b));norms[str(step)]=metrics['norm_left']
                projections[str(step)]={str(level):dict(input=projection(a,old['v'][:,:row[f'k{level}']]),
                    output=projection(b,old['u'][:,:row[f'k{level}']])) for level in LEVELS}
                if step in online:comparisons[str(step)]=product_metrics((a,b),online[step][module])
                if step in [0,64]:
                    arrays,evidence=measure(matrix,a,b,old,row);geometry[str(step)]=evidence
                    retained.update({f'{k}_{step}':v for k,v in arrays.items()})
            folder.mkdir(parents=True,exist_ok=True);tmp=folder/'geometry.tmp.npz';np.savez_compressed(tmp,**retained);tmp.replace(folder/'geometry.npz')
            _write_json(folder/'result.json',dict(module=module,norms=norms,projections=projections,
                online_product_comparisons=comparisons,geometry=geometry))
            _write_json(marker,dict(config_sha256=_hash(config),files={n:file_hash(folder/n) for n in ['geometry.npz','result.json']}))
            new+=1;print(f'Offline geometry: {index+1}/{len(config["modules"])} modules sealed; {module}',flush=True)
        complete=all((directory/'geometry/units'/f'{i:03d}'/'complete.json').exists() for i in range(len(config['modules'])))
        if complete:
            _save_frozen(directory/'geometry/complete.json',dict(config_sha256=_hash(config),units={str(i):file_hash(directory/'geometry/units'/f'{i:03d}'/'complete.json') for i in range(len(config['modules']))}))
        return dict(complete=complete,new_units=new,total_modules=len(config['modules']))


def _push_units(windows):
    from pilot_eval.grpo_algorithm import group_advantages
    questions=[q for w in windows for q in w];byid={q['id']:q for q in questions}
    shuffled=[questions[i]['id'] for i in np.random.default_rng(42).permutation(len(questions))]
    groups=[shuffled[i:i+8] for i in range(0,len(shuffled),8)]+[[q['id'] for q in windows[0]]]
    units=[]
    for index,ids in enumerate(groups):
        sequences=[];dead=0
        for identifier in ids:
            q=byid[identifier];advantages=group_advantages([d['reward_v3'] for d in q['draws']]);dead+=int(not any(advantages))
            for draw,adv in zip(q['draws'],advantages):
                target=draw['generated_token_ids'];prompt=q['input_ids']
                sequences.append(dict(problem_id=identifier,input_ids=prompt+target,target_ids=target,
                    positions=list(range(len(prompt)-1,len(prompt)+len(target)-1)),advantage=adv))
        units.append(dict(key='window-1' if index==64 else f'push-{index:03d}',case='offline',objective='grpo',
            problem_ids=ids,question_count=len(ids),partial=False,sequences=sequences,
            token_count=sum(len(x['target_ids']) for x in sequences),dead_groups=dead))
    return units


def _collector(config,kind,windows,rows,root):
    from pilot_eval.stage0_push_math import a0_hash
    from pilot_eval.activation_profile import AGREEMENT
    variants={f'off{step}':dict(adapter_path=ref['path']) for step,ref in config['source']['checkpoints'].items()}
    common=dict(config,model_identity=config['source']['model_identity'],prompt_contract=config['source']['prompt_contract'],
        seed=42,variants=variants,variant_order=list(variants),batch_size=1,context_chunk=32,workspace_bytes=536870912,
        controls=['off0'],agreement=dict(AGREEMENT),batches_per_unit=8)
    if kind=='push':
        initial=factors(inside(root,variants['off0']['adapter_path']))
        return dict(common,variants={'off0':variants['off0']},variant_order=['off0'],cases={'offline':dict(initialization_arm='off0',a0_sha256=a0_hash(initial))},
            module_shapes={k:dict(a=list(a.shape),b=list(b.shape)) for k,(a,b) in initial.items()}),_push_units(windows)
    if kind!='drift':raise ValueError('Unknown collector')
    common['factor_norm_sums']={alias:sum(float(np.linalg.norm(a.astype(float))*np.linalg.norm(b.astype(float))) for a,b in factors(inside(root,value['adapter_path'])).values()) for alias,value in variants.items()}
    common['kl_direction']='variant||untuned'
    units=[]
    for step in CHECKPOINTS:
        for start in range(0,len(rows),8):
            units.append(dict(key=f'off{step}-{start//8:03d}',variant=f'off{step}',ids=[r['id'] for r in rows[start:start+8]],diagnostic=start==0))
    return common,units


def _read_collection(config,kind,directory,unit,root,collector,rows,cache):
    from pilot_eval.stage0_evaluation import _attempt_timing
    from pilot_eval.stage0_pushes import validate_arrays
    from pilot_eval.stage0_drift_math import check
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal['identity']!=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)) or set(seal['files'])!={'arrays.npz','result.json'}:
        raise ValueError('Collection unit identity changed')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Collection unit payload changed')
    saved=json.loads((folder/'result.json').read_text());attempt=inside(directory/'attempts',saved['attempt_id'])
    if saved['attempt_id'] not in cache:
        final=attempt/'final-integrity.json'
        if not final.exists() or not all(json.loads(final.read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged']):
            raise ValueError('Collection attempt lacks successful full integrity')
        timing=_attempt_timing(attempt)
        if timing['final_integrity_sha256']!=file_hash(final):raise ValueError('Collection attempt timing changed')
        cache[saved['attempt_id']]=file_hash(attempt/'start.json')
    if saved['attempt_start_sha256']!=cache[saved['attempt_id']] or not all(saved['integrity'].get(k) is True for k in ['base_unchanged','adapters_unchanged']):
        raise ValueError('Collection attempt identity changed')
    with np.load(folder/'arrays.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    if kind=='push':validate_arrays(collector,unit,arrays,saved['proof'],root)
    else:
        byid={r['id']:r for r in rows}
        check(dict(arrays=arrays,metadata=saved['metadata'],validation=saved['validation']),[byid[i] for i in unit['ids']],collector,unit['variant'],unit['diagnostic'])
    return arrays,saved


def _reserve(config,execution,directory,root,runtime,approval_path):
    from pilot_eval.stage0_evaluation import _reservation
    return _reservation(config,execution,directory,root,runtime,approval_path)


def collect(path,output_root,kind,*,approval_path=None,max_new_units=None,dependencies=None):
    import time,uuid
    from pilot_eval.run import _write_json
    from pilot_eval.stage0_qualification_backend import deadline_guard
    from pilot_eval.stage0_pushes import validate_arrays
    from pilot_eval.stage0_drift_math import check
    root=Path(output_root).resolve();config,base,windows,rows,_,_=load(path,root)
    collector,units=_collector(config,kind,windows,rows,root);directory=base/kind
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid unit bound')
    if dependencies is None:
        from pilot_eval.offline_grpo_training import _verify_preflight
        training=json.loads((root/config['source']['training_path']).read_text())
        _verify_preflight(root/training['run_path'],training)
        from pilot_eval.stage0_drift_backend import DriftDependencies
        from pilot_eval.stage0_push_backend import PushDependencies
        if kind=='push':
            from pilot_eval.stage0_push_math import instrument_check
            if not instrument_check()['passed']:raise ValueError('Independent initial push instrument gate failed')
        dependencies=PushDependencies() if kind=='push' else DriftDependencies()
    with _writer(directory):
        cache={};saved=[_read_collection(config,kind,directory,u,root,collector,rows,cache) for u in units]
        if (directory/'complete.json').exists():return verify_collection(path,output_root,kind)
        if max_new_units==0:return dict(complete=False,completed=sum(x is not None for x in saved),total=len(units))
        runtime=dependencies.runtime()
        if runtime!=config['runtime']:raise ValueError('Actual runtime differs from approved session')
        reservation,remaining=_reserve(collector,{'reservation':{'reservation_id':None,'job_sha256':_hash(collector)}},directory,root,runtime,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],runtime_sha256=_hash(runtime)))
        started=time.monotonic();engine=None;new=0
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                if hasattr(dependencies,'preflight'):dependencies.preflight(collector,root)
                _save_frozen(directory/'meta/runtime.json',runtime)
                engine=dependencies.engine(collector,root);byid={r['id']:r for r in rows}
                for index,unit in enumerate(units):
                    if saved[index] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    if time.monotonic()-started>=remaining:raise TimeoutError('Collection allocation exhausted')
                    if kind=='push':
                        arrays,proof=engine.measure(unit);validate_arrays(collector,unit,arrays,proof,root);result=dict(proof=proof)
                    else:
                        result=engine.measure([byid[i] for i in unit['ids']],unit['variant'],diagnostic=unit['diagnostic'])
                        check(result,[byid[i] for i in unit['ids']],collector,unit['variant'],unit['diagnostic'])
                        arrays=result.pop('arrays')
                    integrity=engine.check_integrity()
                    if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Collection weight integrity failed')
                    folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                    tmp=folder/'arrays.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(folder/'arrays.npz')
                    result.update(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),integrity=integrity)
                    _write_json(folder/'result.json',result)
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),unit_sha256=_hash(unit)),files={n:file_hash(folder/n) for n in ['arrays.npz','result.json']}))
                    saved[index]=(arrays,result);new+=1
                    from pilot_eval.stage0_workflow import _state
                    _state(directory,'running',sum(x is not None for x in saved),len(units),reservation_id=reservation['reservation_id'])
                    print(f'Offline {kind}: {sum(x is not None for x in saved)}/{len(units)} units sealed; {unit["key"]}',flush=True)
                integrity=engine.check_integrity(full=True);_write_json(attempt/'final-integrity.json',integrity)
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):raise ValueError('Full collection integrity failed')
                if any(x is None for x in saved):
                    from pilot_eval.stage0_workflow import _state
                    return _state(directory,'incomplete',sum(x is not None for x in saved),len(units))
                _save_frozen(directory/'complete.json',dict(config_sha256=_hash(config),units={u['key']:file_hash(directory/'units'/u['key']/'complete.json') for u in units}))
                from pilot_eval.stage0_workflow import _state
                _state(directory,'complete',len(units),len(units))
                return dict(complete=True,units=len(units),execution_kind=config['execution_kind'])
        except (Exception,KeyboardInterrupt) as exc:
            if engine is not None and 'integrity' not in str(exc).lower() and time.monotonic()-started<remaining:
                try:
                    with deadline_guard(remaining-(time.monotonic()-started),config['execution_kind']=='real-cuda'):
                        _write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except Exception:pass
            from pilot_eval.stage0_workflow import _state
            _state(directory,'failed',sum(x is not None for x in saved),len(units),error=str(exc))
            raise
        finally:
            if engine:engine.close()
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,reservation_id=reservation['reservation_id'],
                start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
            _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def verify_collection(path,output_root,kind):
    root=Path(output_root).resolve();config,base,windows,rows,_,_=load(path,root);collector,units=_collector(config,kind,windows,rows,root);directory=base/kind
    seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config) or set(seal['units'])!={u['key'] for u in units}:raise ValueError('Collection completion identity changed')
    if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']:raise ValueError('Collection runtime changed')
    cache={}
    for unit in units:
        if file_hash(directory/'units'/unit['key']/'complete.json')!=seal['units'][unit['key']] or _read_collection(config,kind,directory,unit,root,collector,rows,cache) is None:
            raise ValueError('Collection unit missing')
    return dict(complete=True,units=len(units),execution_kind=config['execution_kind'])


def verify_geometry(path,output_root):
    root=Path(output_root).resolve();config,base,_,_,_,_=load(path,root)
    seal=json.loads((base/'geometry/complete.json').read_text())
    if seal['config_sha256']!=_hash(config) or set(seal['units'])!={str(i) for i in range(len(config['modules']))}:
        raise ValueError('Geometry completion incomplete')
    for i,module in enumerate(config['modules']):
        folder=base/'geometry/units'/f'{i:03d}'
        if file_hash(folder/'complete.json')!=seal['units'][str(i)]:raise ValueError('Geometry unit seal changed')
        marker=json.loads((folder/'complete.json').read_text())
        if (marker['config_sha256']!=_hash(config) or set(marker['files'])!={'geometry.npz','result.json'}
                or any(file_hash(folder/n)!=h for n,h in marker['files'].items())):
            raise ValueError('Geometry payload changed')
        if json.loads((folder/'result.json').read_text())['module']!=module:raise ValueError('Geometry module ordering differs')
    return dict(complete=True,modules=len(config['modules']))


def report(path,output_root):
    from pilot_eval.activation_prepare import VIEWS
    from pilot_eval.run import _write_json
    from pilot_eval.offline_grpo_training import _accepted_history
    root=Path(output_root).resolve();config,base,windows,rows,_,_=load(path,root)
    for kind in ['push','drift']:verify_collection(path,root,kind)
    verify_geometry(path,root)
    geometry=json.loads((base/'geometry/complete.json').read_text())
    if geometry['config_sha256']!=_hash(config) or set(geometry['units'])!={str(i) for i in range(len(config['modules']))}:
        raise ValueError('Geometry completion incomplete')
    geometry_rows=[]
    for i in range(len(config['modules'])):
        folder=base/'geometry/units'/f'{i:03d}'
        if file_hash(folder/'complete.json')!=geometry['units'][str(i)]:raise ValueError('Geometry unit seal changed')
        marker=json.loads((folder/'complete.json').read_text())
        if marker['config_sha256']!=_hash(config) or any(file_hash(folder/n)!=h for n,h in marker['files'].items()):raise ValueError('Geometry payload changed')
        row=json.loads((folder/'result.json').read_text())
        if row['module']!=config['modules'][i]:raise ValueError('Geometry module ordering differs')
        geometry_rows.append(row)
    collector,units=_collector(config,'drift',windows,rows,root);cache={};sums={};rowmap={r['id']:r for r in rows}
    for unit in units:
        arrays,meta=_read_collection(config,'drift',base/'drift',unit,root,collector,rows,cache)
        for index,position in enumerate(meta['metadata']['predictions']):
            row=rowmap[unit['ids'][position['example']]];key=(unit['variant'],row['corpus'],position['view'])
            target=sums.setdefault(key,dict(kl_sum=0.,kl_tokens=0,write_sums=np.zeros((len(config['modules']),3)),write_tokens=0))
            target['kl_sum']+=float(arrays['kl'][index]);target['kl_tokens']+=1
        for index,identifier in enumerate(unit['ids']):
            row=rowmap[identifier]
            for vi,view in enumerate(VIEWS):
                count=sum(row['masks'][view]);key=(unit['variant'],row['corpus'],view)
                if not count:continue
                target=sums.setdefault(key,dict(kl_sum=0.,kl_tokens=0,write_sums=np.zeros((len(config['modules']),3)),write_tokens=0))
                target['write_sums']+=arrays['write_sums'][index,vi];target['write_tokens']+=count
    drift=[]
    for (variant,corpus,view),v in sorted(sums.items()):
        drift.append(dict(step=int(variant[3:]),corpus=corpus,view=view,kl_direction='adapter||untuned',
            kl_token_mean=v['kl_sum']/v['kl_tokens'] if v['kl_tokens'] else None,kl_tokens=v['kl_tokens'],
            module_effective_write_ratio={m:float(v['write_sums'][i,2]/v['write_tokens']) for i,m in enumerate(config['modules'])} if v['write_tokens'] else {},
            module_write_sums={m:dict(update_squared_sum=float(v['write_sums'][i,0]),base_squared_sum=float(v['write_sums'][i,1]),ratio_sum=float(v['write_sums'][i,2])) for i,m in enumerate(config['modules'])},write_tokens=v['write_tokens']))
    training=json.loads((root/config['source']['training_path']).read_text());checkpoints=_verified_checkpoints(root/training['run_path'])
    _,_,logs=_accepted_history(root,checkpoints[64],training['optimizer_groups'])
    diagnostics={}
    for step in [8,16,32,64]:
        ref=logs[step-1]['diagnostic_evidence']['logratio'];d=json.loads(inside(root,ref['path']).read_text())
        from pilot_eval.offline_grpo_diagnostics import summarize
        expected=[(q,draw) for q in windows[step-1] for draw in q['draws']]
        if (d.get('protocol')!='offline-logratio-v1' or d['timing']!='before-update'
            or d['draw_ids']!=[draw['draw_id'] for _,draw in expected]
            or d['question_ids']!=[q['id'] for q,_ in expected]
            or d['generated_token_ids']!=[draw['generated_token_ids'] for _,draw in expected]
            or d['masks']!=[draw['completion_mask'] for _,draw in expected]
            or len(d['log_mu'])!=len(expected) or len(d['log_pi'])!=len(expected)
            or any(len(mu)!=len(draw['generated_token_ids']) or len(pi)!=len(draw['generated_token_ids'])
                   for mu,pi,(_,draw) in zip(d['log_mu'],d['log_pi'],expected))):
            raise ValueError('Pre-update diagnostic window or token identity changed')
        rebuilt=summarize(d['log_mu'],d['log_pi'])
        if any(d['summary'].get(k)!=value for k,value in rebuilt.items()):
            raise ValueError('Pre-update diagnostic arithmetic changed')
        diagnostics[str(step)]=dict(sequence_mean=rebuilt['sequence_mean'],token_rate=rebuilt['token_rate'])
    correlations=[]
    for row in geometry_rows:
        points=[dict(checkpoint=step,preupdate_window=step,similarity=row['online_product_comparisons'][str(step)]['cosine'],
                     empirical_log_ratio=diagnostics[str(step)]['sequence_mean']) for step in [8,16,32,64] if str(step) in row['online_product_comparisons']]
        valid=[p for p in points if p['similarity'] is not None]
        slope=None
        if len(valid)>=2 and np.ptp([p['empirical_log_ratio'] for p in valid])>0:
            slope=float(np.polyfit([p['empirical_log_ratio'] for p in valid],[p['similarity'] for p in valid],1)[0])
        correlations.append(dict(module=row['module'],points=points,descriptive_slope=slope))
    push_config,push_units=_collector(config,'push',windows,rows,root);cache={};pushes=[]
    for unit in push_units:
        arrays,_=_read_collection(config,'push',base/'push',unit,root,push_config,rows,cache)
        pushes.append((unit,arrays))
    initial=factors(inside(root,config['source']['checkpoints']['0']['path']))
    endpoint=factors(inside(root,config['source']['checkpoints']['64']['path']))
    online=factors(inside(root,config['online']['checkpoints']['64']['path'])) if config['online'] else None
    first_ref=logs[0]['diagnostic_evidence']['window1_push']
    first_evidence=json.loads(inside(root,first_ref['path']).read_text())
    recorded=first_evidence['independent_push'];names=first_evidence['parameter_names'];window_arrays=pushes[-1][1]
    shapes={module:i for i,module in enumerate(config['modules'])};actual=[]
    for name in names:
        matched=False
        for module,index in shapes.items():
            if name.startswith(module+'.lora_A.'):
                actual.extend((-window_arrays[f'gradient_a_{index}']).reshape(-1).tolist());matched=True;break
            if name.startswith(module+'.lora_B.'):
                actual.extend(window_arrays[f'push_{index}'].reshape(-1).tolist());matched=True;break
        if not matched:raise ValueError('Window-1 push parameter mapping differs')
    criterion=first_evidence['criterion']
    if len(recorded)!=len(actual) or not np.allclose(actual,recorded,atol=criterion['atol'],rtol=criterion['rtol']):
        raise ValueError('Separately collected window-1 push differs from accepted training diagnostic')
    total_norms={str(step):float(np.linalg.norm([r['norms'][str(step)] for r in geometry_rows])) for step in CHECKPOINTS}
    push_alignment=[]
    for i,module in enumerate(config['modules']):
        mean=np.mean([arrays[f'push_{i}'] for unit,arrays in pushes if unit['key']!='window-1'],axis=0)
        push_alignment.append(dict(module=module,offline=product_metrics((initial[module][0],mean),endpoint[module]),
            online=None if online is None else product_metrics((initial[module][0],mean),online[module]),
            window1_push_norm=float(np.linalg.norm(pushes[-1][1][f'push_{i}']))))
    result=dict(protocol='offline-grpo-internals-report-v1',execution_kind=config['execution_kind'],
        geometry=geometry_rows,adapter_frobenius_norms=total_norms,drift=drift,preupdate=diagnostics,descriptive_similarities=correlations,push_alignment=push_alignment,
        window1='separately computed actual first-window negative gradient; training diagnostic independently checks negative gradient sign',
        unavailable=([] if config['online'] else ['online checkpoint comparisons not selected']) + ['Gold and Sampling SFT internal-source comparisons are not wired by this version'],
        limitations=['Geometry and slopes are descriptive, not behavioral causes.','Observed adapter norm and KL are not matched-norm measurements.',
                     'One seed cannot estimate run-to-run variability.','Fixed-context KL is adapter || untuned; pre-update empirical log ratio has the opposite direction.'])
    with _writer(base/'report'):
        _save_frozen(base/'report/results.json',result)
        lines=['# Offline GRPO internals','','Geometry and associations are descriptive. Fixed-context KL is adapter || untuned; pre-update log ratios use the opposite direction.','','| Checkpoint | Adapter Frobenius norm |','|---:|---:|']
        lines.extend(f'| {step} | {total_norms[str(step)]:.6g} |' for step in CHECKPOINTS)
        lines.extend(['','| Checkpoint | Corpus | View | KL nats/token |','|---:|---|---|---:|'])
        for row in drift:
            value='undefined' if row['kl_token_mean'] is None else f"{row['kl_token_mean']:.6g}"
            lines.append(f"| {row['step']} | {row['corpus']} | {row['view']} | {value} |")
        lines.extend(['','The actual window-1 push is independently checked against the accepted training diagnostic.',
                      'Matched-norm KL requires a separately measured scaled-adapter extension.',
                      'Gold and Sampling SFT internal-source comparisons are not wired in this version.'])
        content='\n'.join(lines)+'\n';report_path=base/'report/report.md'
        if report_path.exists() and report_path.read_text()!=content:raise ValueError('Internals report text changed')
        report_path.write_text(content)
        _save_frozen(base/'report/complete.json',dict(config_sha256=_hash(config),results_sha256=file_hash(base/'report/results.json'),
            markdown_sha256=file_hash(report_path),inputs={kind:file_hash(base/kind/'complete.json') for kind in ['geometry','drift','push']}))
    return result


def verify_report(path,output_root):
    root=Path(output_root).resolve();config,base,_,_,_,_=load(path,root)
    verify_geometry(path,root)
    for kind in ['push','drift']:verify_collection(path,root,kind)
    seal=json.loads((base/'report/complete.json').read_text())
    if (seal['config_sha256']!=_hash(config) or seal['results_sha256']!=file_hash(base/'report/results.json')
            or seal['markdown_sha256']!=file_hash(base/'report/report.md')
            or seal['inputs']!={kind:file_hash(base/kind/'complete.json') for kind in ['geometry','drift','push']}):
        raise ValueError('Internals report seal changed')
    return dict(complete=True,execution_kind=config['execution_kind'])
