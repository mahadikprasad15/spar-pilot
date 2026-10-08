"""Separately frozen, profiled and resumable supplemental SFT/GRPO measurements."""
import json
import time
from pathlib import Path
import numpy as np
from pilot_eval.activation_measurement import load_execution,_compatible_runtime
from pilot_eval.activation_prepare import STEPS,VIEWS,PROJECTIONS
from pilot_eval.activation_profile import AGREEMENT,_relative,_seal,_verified,HFProfileDependencies
from pilot_eval.activation_workflow import _save_arrays
from pilot_eval.run import _write_json,_write_state
from pilot_eval.training import file_hash,run_lock
from pilot_eval.sft import safe_name
from pilot_eval.workflow import _hash,_save_frozen
from pilot_eval.token_measurement_math import prediction_positions,summarize_kl

PROTOCOL='pilot4-token-measurements-v1'
CONVENTIONS=dict(coefficient='raw-Ax-adapted-input',kl='tuned||untuned',units='nats',
    view='next-target-token',exclude=['target-zero','padding','beyond-saved-sequence'],
    weightings=['token','example'],coefficient_dtype='float32',kl_dtype='float64')


class HFTokenDependencies(HFProfileDependencies):
    def token_engine(self,prepared,root,settings):
        from pilot_eval.token_measurement_engine import TokenMeasurementEngine
        return TokenMeasurementEngine(self.activation_engine(prepared,root),
            context_chunk=settings['context_chunk'],workspace_bytes=settings['workspace_bytes'])


def _sources(sources,root):
    products={}
    for arm,path in sources.items():
        execution,prepared,rows=load_execution(root/path,root)
        products[arm]=(execution,prepared,rows)
    sft,grpo=products['sft'],products['grpo']
    if (sft[1]['protocol_version']!='pilot3-write-v1'
            or grpo[1].get('source_contract',{}).get('kind')!='grpo'
            or sft[2]!=grpo[2] or sft[0]['batches']!=grpo[0]['batches']
            or sft[0]['batch_size']!=grpo[0]['batch_size']
            or sft[1]['source_evidence']['base_sha256']!=grpo[1]['source_evidence']['base_sha256']
            or any(sft[1][k]!=grpo[1][k] for k in ['model','model_revision','tokenizer_revision','dtype','items_sha256'])
            or not _compatible_runtime(sft[0]['runtime'],grpo[0]['runtime'])
            or any(p[0]['checkpoint_steps']!=STEPS for p in products.values())):
        raise ValueError('supplement requires matched verified SFT/GRPO executions')
    return products


def _implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
            ['token_measurement_workflow.py','token_measurement_math.py','token_measurement_engine.py','activation_engine.py']}


def prepare_tokens(sft_execution,grpo_execution,output_root,name,*,context_chunk,workspace_bytes):
    root=Path(output_root).resolve();name=safe_name(name)
    sources={arm:str(_relative(path,root).relative_to(root)) for arm,path in
             [('sft',sft_execution),('grpo',grpo_execution)]}
    products=_sources(sources,root)
    if type(context_chunk)!=int or type(workspace_bytes)!=int or min(context_chunk,workspace_bytes)<1:
        raise ValueError('positive explicit context chunk and workspace budget required')
    first=products['sft']
    config=dict(protocol=PROTOCOL,run_id=name,sources=sources,
        source_hashes={arm:file_hash(root/path) for arm,path in sources.items()},
        conventions=CONVENTIONS,settings=dict(context_chunk=context_chunk,workspace_bytes=workspace_bytes),
        batches=first[0]['batches'],batch_size=first[0]['batch_size'],checkpoint_steps=STEPS,
        inputs_sha256=first[1]['items_sha256'],runtime=first[0]['runtime'],implementation=_implementation(),
        comparison=AGREEMENT,run_path=str(Path('runs/pilot-4')/first[1]['model'].replace('/','--')/'supplemental-tokens'/name))
    path=root/'plans'/name/'tokens.prepared.json'
    with run_lock(path.parent):
        _save_frozen(path,config)
        _save_frozen(root/config['run_path']/'meta/run_manifest.json',config)
        if not (root/config['run_path']/'meta/status.json').exists():
            _write_state(root/config['run_path'],'prepared',0,len(config['batches'])*10)
        print(f'Supplement prepared; existing summaries unchanged; {path}',flush=True)
    return path


def _load(path,root):
    path=_relative(path,root);config=json.loads(path.read_text())
    if (config['protocol']!=PROTOCOL or config['conventions']!=CONVENTIONS
            or config['comparison']!=AGREEMENT or config['implementation']!=_implementation()
            or config['checkpoint_steps']!=STEPS):raise ValueError('supplement convention/code identity changed; use a new product')
    if path!=root/'plans'/config['run_id']/path.name or path.name not in ['tokens.prepared.json','tokens.execution.json']:
        raise ValueError('supplement plan path mismatch')
    if any(file_hash(root/p)!=config['source_hashes'][a] for a,p in config['sources'].items()):
        raise ValueError('supplement source config hash changed')
    products=_sources(config['sources'],root)
    source=products['sft']
    if (config['batches']!=source[0]['batches'] or config['batch_size']!=source[0]['batch_size']
            or config['inputs_sha256']!=source[1]['items_sha256'] or config['runtime']!=source[0]['runtime']):
        raise ValueError('supplement batch/input/runtime identity changed')
    if type(config['settings']['context_chunk'])!=int or type(config['settings']['workspace_bytes'])!=int or min(config['settings'].values())<1:
        raise ValueError('invalid frozen supplement resource settings')
    return config,products


def _positions(rows):
    return [dict(example=i,position=t,view=v,token_id=row['input_ids'][t])
            for i,row in enumerate(rows) for t in range(len(row['input_ids']))
            for v in VIEWS if row['masks'][v][t]]


def _check(result,rows,step,settings):
    arrays,meta,validation=result['arrays'],result['metadata'],result['validation']
    expected=[(i,p) for i in range(28) for p in PROJECTIONS]
    if (meta['example_ids']!=[r['id'] for r in rows] or meta['coefficient_positions']!=_positions(rows)
            or meta['predictions']!=prediction_positions(rows) or meta['step']!=step
            or [(m['layer'],m['projection']) for m in meta['modules']]!=expected
            or meta['coefficient_convention']!=CONVENTIONS['coefficient']
            or meta['kl_direction']!=CONVENTIONS['kl'] or meta['kl_units']!='nats'
            or meta['logit_precision']!='float32' or meta['reduction_precision']!='float64'
            or set(arrays)!={'coefficients','kl'}
            or arrays['coefficients'].shape!=(len(_positions(rows)),196) or arrays['coefficients'].dtype!=np.float32
            or arrays['kl'].shape!=(len(prediction_positions(rows)),) or arrays['kl'].dtype!=np.float64
            or any(not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('supplement token/module/value identity mismatch or nonfinite payload')
    if (validation['rank1_passed'] is not True or validation['reference_invariant'] is not True
            or validation['module_count']!=196 or not 0<=validation['rank1_max_fraction']<=1
            or validation['max_logit_contexts']>settings['context_chunk']
            or validation['bounded_workspace_estimate']>settings['workspace_bytes']):
        raise ValueError('supplement numerical/hook/workspace gate failed')
    for m in meta['modules']:
        if (m['scale']!=1 or any(len(m[k])!=64 or any(c not in '0123456789abcdef' for c in m[k]) for k in ['A_sha256','B_sha256'])
                or not np.isfinite([m['A_norm'],m['B_norm']]).all() or min(m['A_norm'],m['B_norm'])<0):
            raise ValueError('supplement factor/scaling provenance invalid')
    if step==0 and (np.count_nonzero(arrays['kl']) or validation['zero_contribution'] is not True):
        raise ValueError('supplement zero checkpoint gate failed')


def _canonical(result):
    meta=result['metadata']
    ids=meta['example_ids']
    keys=lambda positions,pos:[(ids[r['example']],r[pos],r['view'],r['token_id']) for r in positions]
    return dict(coefficient_keys=keys(meta['coefficient_positions'],'position'),
                prediction_keys=keys(meta['predictions'],'target'),modules=meta['modules'],**result['arrays'])


def _agree(reference,candidate):
    for key in ['coefficient_keys','prediction_keys','modules']:
        if reference[key]!=candidate[key]:raise ValueError('supplement comparison token/factor mapping differs')
    errors={key:float(np.max(np.abs(reference[key].astype(np.float64)-candidate[key].astype(np.float64)),initial=0))
            for key in ['coefficients','kl']}
    if any(not np.allclose(reference[key],candidate[key],**AGREEMENT) for key in errors):
        raise ValueError(f'supplement batch/chunk agreement gate failed: {errors}')
    return errors


def _diagnostics(rows,batch):
    # Independent held-out check, plus long examples for capacity; no resampling.
    longest=[max((r for r in rows if r['corpus']==c),key=lambda r:len(r['input_ids'])) for c in ['gsm8k','fineweb']]
    chosen={r['id'] for r in longest}
    capacity=longest+[r for r in rows if r['id'] not in chosen][:max(batch-2,0)]
    remaining=[r for r in rows if r['id'] not in {r['id'] for r in capacity}]
    heldout=[next(r for r in remaining if r['corpus']==c) for c in ['gsm8k','fineweb']]
    return ([capacity[i:i+batch] for i in range(0,len(capacity),batch)]
            +[heldout[i:i+batch] for i in range(0,len(heldout),batch)])


def profile_tokens(config_path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();started=time.perf_counter();config,products=_load(config_path,root)
    deps=dependencies or HFTokenDependencies();runtime=deps.runtime()
    if not _compatible_runtime(config['runtime'],runtime):raise ValueError('supplement runtime differs from validated sources')
    directory=root/config['run_path']/'profile'
    identity=dict(config_sha256=_hash(config),runtime=runtime,settings=config['settings'],kind='supplemental-profile')
    with run_lock(directory):
        if (directory/'complete.json').exists():return _verified(directory,identity,['config.json','results.json'])
        _save_frozen(directory/'config.json',identity)
        total=sum(len(_diagnostics(product[2],config['batch_size']))*len(STEPS) for product in products.values())
        _write_state(directory,'running',0,total)
        measurements=[];paths=[directory/'config.json'];saving_seconds=0.
        try:
            for arm,(_,prepared,rows) in products.items():
                for group,selected in enumerate(_diagnostics(rows,config['batch_size'])):
                    loading_started=time.perf_counter()
                    engine=deps.token_engine(prepared,root,config['settings'])
                    loading_seconds=time.perf_counter()-loading_started
                    deps.reset_peak_memory();group_started=time.perf_counter()
                    try:
                        with engine.integrity_scope():
                            for step in STEPS:
                                checkpoint=root/prepared['source_evidence']['checkpoints'][str(step)]['path']
                                deps.synchronize();t=time.perf_counter()
                                result=engine.measure(checkpoint,selected,step=step);deps.synchronize()
                                seconds=time.perf_counter()-t
                                _check(result,selected,step,config['settings'])
                                save_started=time.perf_counter()
                                evidence=directory/f'{arm}-group-{group}-step-{step}'
                                _save_arrays(evidence/'arrays.npz',result['arrays'])
                                _write_json(evidence/'metadata.json',result['metadata'])
                                paths.extend([evidence/'arrays.npz',evidence/'metadata.json'])
                                save_seconds=time.perf_counter()-save_started
                                saving_seconds+=save_seconds
                                reference=_canonical(result)
                                repeat=engine.measure(checkpoint,selected,step=step)
                                _check(repeat,selected,step,config['settings'])
                                repeat_error=_agree(reference,_canonical(repeat))
                                # Test batching against independent singleton passes.
                                singles=[]
                                for row in selected:
                                    single=engine.measure(checkpoint,[row],step=step)
                                    _check(single,[row],step,config['settings']);singles.append(_canonical(single))
                                candidate=dict(coefficient_keys=sum((v['coefficient_keys'] for v in singles),[]),
                                    prediction_keys=sum((v['prediction_keys'] for v in singles),[]),modules=singles[0]['modules'],
                                    coefficients=np.concatenate([v['coefficients'] for v in singles]),
                                    kl=np.concatenate([v['kl'] for v in singles]))
                                batch_error=_agree(reference,candidate)
                                alternate=dict(config['settings'],context_chunk=max(1,config['settings']['context_chunk']//2))
                                changed=engine.measure(checkpoint,selected,step=step,context_chunk=alternate['context_chunk'])
                                _check(changed,selected,step,alternate)
                                chunk_error=_agree(reference,_canonical(changed))
                                peak=deps.peak_memory()
                                measurements.append(dict(arm=arm,group=group,step=step,examples=len(selected),
                                    example_ids=[r['id'] for r in selected],seconds=seconds,**peak,
                                    repeat_errors=repeat_error,batch_errors=batch_error,chunk_errors=chunk_error,
                                    saving_wall_seconds=save_seconds,loading_wall_seconds=loading_seconds if step==0 else 0.,
                                    validation=result['validation'],saved_array_bytes=sum(v.nbytes for v in result['arrays'].values())))
                                _write_state(directory,'running',len(measurements),total)
                                print(f'Token profile {arm} group {group} step {step}: {seconds:.2f}s; checks passed',flush=True)
                    finally:
                        engine.close()
                        engine=None
                    measurements[-1]['group_wall_seconds']=time.perf_counter()-group_started
            # Approximate cost from measured example throughput; lengths/hooks/I/O
            # can differ across the full cohort. No runtime promise.
            groups=len(_diagnostics(products['sft'][2],config['batch_size']))
            estimate=sum(r['seconds']/r['examples']*300 for r in measurements)/groups
            result=dict(profile_passed=True,measurements=measurements,runtime=runtime,
                total_wall_seconds=time.perf_counter()-started,saving_wall_seconds=saving_seconds,estimated_main_forward_seconds=estimate,
                estimated_array_bytes=int(sum(r['saved_array_bytes']/r['examples']*300 for r in measurements)/groups),
                estimate_note='Each checkpoint/arm averaged across declared diagnostic groups; production includes sealing/I/O and can differ.',
                agreement=AGREEMENT)
            _write_json(directory/'results.json',result)
            _seal(directory,identity,paths+[directory/'results.json'])
            _write_state(directory,'completed',total,total)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',len(measurements),total,str(exc));raise


def freeze_tokens(config_path,output_root,*,review_notes):
    root=Path(output_root).resolve();config,products=_load(config_path,root)
    if not review_notes.strip():raise ValueError('review supplemental checks/cost and supply notes')
    profile=root/config['run_path']/'profile'
    profile_identity=json.loads((profile/'config.json').read_text())
    if profile_identity['config_sha256']!=_hash(config):raise ValueError('supplement profile/config mismatch')
    result=_verified(profile,profile_identity,['config.json','results.json'])
    if not result['profile_passed']:raise ValueError('supplement profile failed')
    frozen=dict(config,profile_complete_sha256=file_hash(profile/'complete.json'),
                profile_runtime=result['runtime'],review_notes=review_notes,state='frozen')
    path=root/'plans'/config['run_id']/'tokens.execution.json'
    with run_lock(path.parent):_save_frozen(path,frozen)
    return path


def _execution(config_path,root):
    config,products=_load(config_path,root)
    if config.get('state')!='frozen' or not config.get('review_notes','').strip():raise ValueError('supplement must be profiled and reviewed before execution')
    prepared=root/'plans'/config['run_id']/'tokens.prepared.json'
    original=json.loads(prepared.read_text())
    if any(config[k]!=v for k,v in original.items()):raise ValueError('supplement frozen settings changed')
    profile=root/config['run_path']/'profile'
    if file_hash(profile/'complete.json')!=config['profile_complete_sha256']:raise ValueError('supplement profile hash changed')
    identity=json.loads((profile/'config.json').read_text())
    result=_verified(profile,identity,['config.json','results.json'])
    if not result['profile_passed'] or result['runtime']!=config['profile_runtime'] or identity['config_sha256']!=_hash(original):
        raise ValueError('supplement profile identity/runtime changed')
    return config,products


def _unit(config,arm,batch,step):
    return dict(execution_sha256=_hash(config),arm=arm,batch=batch,step=step,
                source_execution_sha256=config['source_hashes'][arm],kind='token-shard')


def _read_unit(directory,identity,rows,step,settings):
    result=_verified(directory,identity,['arrays.npz','metadata.json','results.json'])
    metadata=json.loads((directory/'metadata.json').read_text())
    with np.load(directory/'arrays.npz',allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    record=dict(arrays=arrays,metadata=metadata,validation=result['validation'])
    _check(record,rows,step,settings)
    return record


def _units(config,products,root):
    for arm,(_,prepared,rows) in products.items():
        mapping={r['id']:r for r in rows}
        for batch in config['batches']:
            selected=[mapping[i] for i in batch['example_ids']]
            for step in STEPS:
                directory=root/config['run_path']/'shards'/arm/f"batch-{batch['index']:06d}-step-{step}"
                yield arm,prepared,batch['index'],step,selected,directory,_unit(config,arm,batch,step)


def measure_tokens(config_path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,products=_execution(config_path,root)
    directory=root/config['run_path'];deps=dependencies or HFTokenDependencies()
    if not _compatible_runtime(config['profile_runtime'],deps.runtime()):raise ValueError('supplement production runtime changed')
    units=list(_units(config,products,root));done=[]
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify_tokens(config_path,root)
        for arm,prepared,index,step,rows,path,identity in units:
            if (path/'complete.json').exists():
                _read_unit(path,identity,rows,step,config['settings']);done.append([arm,index,step])
        _write_json(directory/'checkpoints/progress.json',dict(completed_units=done,total=len(units)))
        _write_state(directory,'running',len(done),len(units))
        engine=None;current=None;started=time.perf_counter()
        try:
            groups={}
            for unit in units:
                if [unit[0],unit[2],unit[3]] not in done:
                    groups.setdefault((unit[0],unit[2]),[]).append(unit)
            on_entry=len(done)
            for (arm,index),batch_units in groups.items():
                prepared,rows=batch_units[0][1],batch_units[0][4]
                if current!=arm:
                    if engine:
                        engine.close()
                        engine=None
                    engine=deps.token_engine(prepared,root,config['settings']);current=arm
                pending=[];reference=None
                try:
                    # One exact boundary pair per bounded batch workload; no unit
                    # publishes until all pending measurements pass the final hash.
                    with engine.integrity_scope() as integrity:
                        reference=engine.reference(rows)
                        for _,_,_,step,_,path,identity in batch_units:
                            print(f'Token measurement {arm} batch {index} step {step}: {len(done)}/{len(units)} verified; batch {len(rows)}; pending {len(pending)}',flush=True)
                            t=time.perf_counter()
                            record=engine.measure(root/prepared['source_evidence']['checkpoints'][str(step)]['path'],rows,step=step,reference=reference)
                            _check(record,rows,step,config['settings'])
                            _save_arrays(path/'arrays.npz',record['arrays'])
                            _write_json(path/'metadata.json',record['metadata'])
                            _write_json(path/'results.json',dict(validation=record['validation'],wall_seconds=time.perf_counter()-t))
                            pending.append((step,path,identity))
                    for step,path,identity in pending:
                        result=json.loads((path/'results.json').read_text())
                        _write_json(path/'results.json',dict(result,integrity=integrity))
                        _seal(path,identity,[path/'arrays.npz',path/'metadata.json',path/'results.json'])
                        done.append([arm,index,step])
                    _write_json(directory/'checkpoints/progress.json',dict(completed_units=done,total=len(units)))
                    _write_state(directory,'running',len(done),len(units))
                finally:
                    if reference is not None:engine.release_reference(reference)
            result=dict(measurement_complete=True,completed_units=len(done),total_units=len(units),
                        inputs_sha256=config['inputs_sha256'],protocol=PROTOCOL)
            _write_json(directory/'results/results.json',result)
            _write_json(directory/'meta/runtime.json',deps.runtime())
            _write_json(directory/'meta/timing.json',dict(attempt_wall_seconds=time.perf_counter()-started,verified_on_entry=on_entry,
                note='Per-unit timing persists in each shard; this attempt excludes earlier attempts.'))
            identity=dict(execution_sha256=_hash(config))
            _seal(directory,identity,[directory/'results/results.json',directory/'meta/runtime.json',directory/'meta/timing.json',directory/'meta/run_manifest.json']+
                  [path/'complete.json' for *_,path,_ in units])
            _write_state(directory,'completed',len(done),len(units))
            return result
        except BaseException as exc:
            _write_state(directory,'failed',len(done),len(units),str(exc))
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:f.write(json.dumps(dict(type=type(exc).__name__,error=str(exc),completed=len(done)))+'\n')
            raise
        finally:
            if engine:engine.close()


def verify_tokens(config_path,output_root):
    root=Path(output_root).resolve();config,products=_execution(config_path,root)
    directory=root/config['run_path'];units=list(_units(config,products,root))
    required=['results/results.json','meta/runtime.json','meta/timing.json','meta/run_manifest.json']+[str((p/'complete.json').relative_to(directory)) for *_,p,_ in units]
    # _verified expects results.json; this product uses the canonical results tree.
    marker=json.loads((directory/'complete.json').read_text())
    if marker['identity']!=dict(execution_sha256=_hash(config)) or not set(required).issubset(marker['files']):
        raise ValueError('supplement completion inventory mismatch')
    for relative,digest in marker['files'].items():
        if file_hash(_relative(directory/relative,directory))!=digest:raise ValueError('supplement completion hash mismatch')
    if not _compatible_runtime(config['profile_runtime'],json.loads((directory/'meta/runtime.json').read_text())):
        raise ValueError('supplement saved runtime mismatch')
    for _,_,_,step,rows,path,identity in units:_read_unit(path,identity,rows,step,config['settings'])
    result=json.loads((directory/'results/results.json').read_text())
    if result.get('completed_units')!=len(units) or not result.get('measurement_complete'):
        raise ValueError('supplement incomplete result')
    return result


def report_tokens(config_path,output_root,name):
    root=Path(output_root).resolve();name=safe_name(name)
    verify_tokens(config_path,root);config,products=_execution(config_path,root)
    directory=root/'reports'/name
    identity=dict(protocol=PROTOCOL,execution_sha256=_hash(config),
                  measurement_complete_sha256=file_hash(root/config['run_path']/'complete.json'))
    with run_lock(directory):
        _save_frozen(directory/'config.json',identity)
        if (directory/'complete.json').exists():
            marker=json.loads((directory/'complete.json').read_text())
            if marker['identity']!=identity:raise ValueError('supplement report identity mismatch')
            for relative,digest in marker['files'].items():
                if file_hash(_relative(directory/relative,directory))!=digest:raise ValueError('supplement report hash mismatch')
            return json.loads((directory/'results/results.json').read_text())
        trajectory=[];coefficient_summary=[];raw_shards=[]
        aggregate={}
        for arm,prepared,batch,step,rows,path,unit in _units(config,products,root):
            record=_read_unit(path,unit,rows,step,config['settings']);meta=record['metadata'];arrays=record['arrays']
            raw_shards.append(dict(arm=arm,step=step,batch=batch,path=str(path.relative_to(root)),complete_sha256=file_hash(path/'complete.json')))
            summary=summarize_kl(meta['predictions'],arrays['kl'],meta['example_ids'])
            for cell in summary:
                aggregate.setdefault((arm,step,cell['view']),[]).extend(cell['per_example'])
            # Coefficients remain per-token in shards; report observed scalar and
            # reconstructed branch magnitudes in each module's own factorization.
            for view in VIEWS:
                selected=[i for i,p in enumerate(meta['coefficient_positions']) if p['view']==view]
                if not selected:continue
                c=arrays['coefficients'][selected].astype(np.float64)
                coefficient_summary.append(dict(arm=arm,step=step,batch=batch,view=view,tokens=len(selected),
                    signed_sum=c.sum(0).tolist(),absolute_sum=np.abs(c).sum(0).tolist(),squared_sum=(c*c).sum(0).tolist(),
                    branch_norm_sum=(np.abs(c).sum(0)*np.array([m['scale']*m['B_norm'] for m in meta['modules']])).tolist(),
                    modules=meta['modules']))
        for (arm,step,view),per in aggregate.items():
            count=sum(p['count'] for p in per)
            trajectory.append(dict(arm=arm,step=step,view=view,tokens=count,examples=len(per),
                token_mean=sum(p['sum'] for p in per)/count if count else None,
                example_mean=sum(p['mean'] for p in per)/len(per) if per else None,
                undefined_reason=None if count else 'no_target_contexts',per_example=per))
        result=dict(report_complete=True,protocol=PROTOCOL,provenance=identity,conventions=CONVENTIONS,
            kl_trajectory=trajectory,coefficient_summaries=coefficient_summary,raw_shards=raw_shards,
            limits=['Coefficients depend on saved factorization and are not standalone invariant write magnitudes.',
                    'KL uses fixed supplied contexts, not unconditional generated-policy distributions.',
                    'Positive KL does not imply loss of capability; no causal mechanism is established.',
                    'Supplemental forward passes are new evidence; old write summaries remain unchanged.'])
        _write_json(directory/'results/results.json',result)
        lines=['# Supplemental coefficients and fixed-context KL','',
               '| Arm | Step | View | Tokens | Examples | Token mean KL (nats) | Equal-example mean |',
               '|---|---:|---|---:|---:|---:|---:|']
        for r in trajectory:lines.append(f"| {r['arm']} | {r['step']} | {r['view']} | {r['tokens']} | {r['examples']} | {r['token_mean']} | {r['example_mean']} |")
        lines+=['','Raw per-token coefficients, KL, positions, factors and numerical evidence remain in the sealed shard index in results.json.','']+result['limits']
        (directory/'results/report.md').write_text('\n'.join(lines)+'\n')
        _seal(directory,identity,[directory/'config.json',directory/'results/results.json',directory/'results/report.md'])
        _write_state(directory,'completed',len(trajectory),len(trajectory))
        return result
