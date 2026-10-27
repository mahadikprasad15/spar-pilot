"""Bounded eight-draw group transactions, exact replay and frozen production route."""
import json
from pathlib import Path
import time
import uuid

from pilot_eval.run import _write_json
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_now,budget_status
from pilot_eval.stage0_budget import reserve
from pilot_eval.stage0_evaluation import _jsonl,_reservation,_attempt_timing
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.stage0_rollout_plan import load,replay_policy
from pilot_eval.stage0_rollout_math import normalize,replay_comparison
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def _read(config,directory,unit,runtime,training,specials):
    folder=directory/'units'/unit['key']
    if not (folder/'complete.json').exists():return None
    marker=json.loads((folder/'complete.json').read_text())
    expected=dict(config_sha256=_hash(config),unit=unit,runtime_sha256=_hash(runtime),specials=specials)
    if marker.get('identity')!=expected or set(marker.get('files',{}))!={'responses.jsonl','metadata.json'}:
        raise ValueError('Rollout unit/config/runtime identity changed.')
    for name,digest in marker['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Saved rollout group payload changed.')
    meta=json.loads((folder/'metadata.json').read_text());attempt=directory/'attempts'/meta['attempt_id']
    if file_hash(attempt/'start.json')!=meta['attempt_start_sha256'] or meta['integrity'].get('base_unchanged') is not True:
        raise ValueError('Rollout attempt identity/integrity changed.')
    timing=_attempt_timing(attempt,required=False);final=attempt/'final-integrity.json'
    if final.exists():
        if json.loads(final.read_text()).get('base_unchanged') is not True:raise ValueError('Failed full integrity invalidates this attempt.')
        if timing is not None and timing['final_integrity_sha256']!=file_hash(final):raise ValueError('Attempt final-integrity receipt changed.')
    saved=[json.loads(x) for x in (folder/'responses.jsonl').read_text().splitlines()]
    expected_rows=[normalize(training,value,d,origin=unit['origin'],seed=unit['seed'],limit=1024,eos=specials['eos_token_id']) for d,value in enumerate(saved)]
    if saved!=expected_rows or len(saved)!=8:raise ValueError('Saved draw identities/scores do not reconstruct.')
    return saved


def phase_saved(config,base,phase,units,runtime,training):
    directory=base/phase;path=directory/'inputs/special-tokens.json'
    if not path.exists():
        if list((directory/'units').glob('*/complete.json')):raise ValueError('Sealed groups lack tokenizer identity.')
        return [None]*len(units)
    specials=json.loads(path.read_text())
    if json.loads((directory/'meta/runtime.json').read_text())!=runtime:raise ValueError('Saved rollout runtime changed.')
    return [_read(dict(config,phase=phase),directory,u,runtime,training[u['id']],specials) for u in units]


def profile_verify(config,base,training,historical):
    directory=base/'profile';seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Profile completion identity changed.')
    for path,digest in seal['files'].items():
        if file_hash(inside(directory,path))!=digest:raise ValueError('Profile evidence changed.')
    result=json.loads((directory/'results/profile.json').read_text());runtime=json.loads((directory/'meta/runtime.json').read_text())
    policy=replay_policy(config,runtime)
    if policy!=result['policy']:raise ValueError('Historical replay policy changed.')
    rows=phase_saved(config,base,'profile',policy['units'],runtime,training)
    if any(r is None for r in rows):raise ValueError('Incomplete profile groups.')
    current=[r for group in rows for r in group]
    reference=[r for u in policy['units'] for r in historical if r['group_id']==u['id']]
    comparison=replay_comparison(reference,current) if policy['replay_eligible'] else dict(exact=False,draws=0,differences=[],criterion='historical replay unavailable; fresh uniform timing')
    route='reuse-historical-128' if policy['replay_eligible'] and comparison['exact'] else 'uniform-fresh-512'
    if result['comparison']!=comparison or result['route']!=route or result['runtime']!=runtime:
        raise ValueError('Replay result/route contradicts saved exact evidence.')
    if result['production_groups']!=(384 if route=='reuse-historical-128' else 512):raise ValueError('Route group count changed.')
    return result


def _execution(config,base,path,root,training,historical):
    result=profile_verify(config,base,training,historical);execution=json.loads(path.read_text())
    marker=json.loads((base/'production-freeze.json').read_text())
    if marker!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)):raise ValueError('Rollout production freeze changed.')
    units=config['extension_units'] if result['route']=='reuse-historical-128' else config['uniform_units']
    if (execution['config_sha256']!=_hash(config) or execution['profile_sha256']!=file_hash(base/'profile/complete.json')
            or execution['runtime']!=result['runtime'] or execution['route']!=result['route'] or execution['units']!=units
            or execution['real_gpu_authorized']!=(config['execution_kind']=='real-cuda')):
        raise ValueError('Frozen production route/units/runtime differ from approved evidence.')
    return execution


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,base,training,historical=load(path,root);by_id={r['id']:r for r in training}
    result=profile_verify(config,base,by_id,historical);review=json.loads(Path(review_path).read_text())
    required={'reviewer','notes','session_id','max_units_per_hour','max_seconds','source_reviewed','profile_sha256','route'}
    if (set(review)!=required or review['profile_sha256']!=file_hash(base/'profile/complete.json') or review['route']!=result['route']
            or review['session_id']!=result['runtime']['session_id'] or review['max_seconds']<result['forecast']['conservative_total_seconds']):
        raise ValueError('Review the exact replay/fallback route and cover its measured forecast in the same session.')
    target=Path(path).resolve().parent/'stage0.rollouts.execution.json'
    with _writer(base):
        if target.exists():
            old=_execution(config,base,target,root,by_id,historical)
            if old['review']!=review:raise ValueError('Frozen rollout selection changed; use a new plan.')
            return target
        units=config['extension_units'] if result['route']=='reuse-historical-128' else config['uniform_units']
        job=_hash(dict(config,phase='collection'))
        reservation=reserve(root,job,{k:v for k,v in review.items() if k not in ['profile_sha256','route']})
        execution=dict(protocol='stage0-rollouts-execution-v1',config_sha256=_hash(config),profile_sha256=review['profile_sha256'],
            route=result['route'],runtime=result['runtime'],units=units,review=review,reservation=reservation,
            real_gpu_authorized=config['execution_kind']=='real-cuda',measurement_complete=False)
        _save_frozen(target,execution);_save_frozen(base/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
    return target


def collect(path,output_root,phase,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,base,training,historical=load(path,root);by_id={r['id']:r for r in training}
    directory=base/phase;phase_config=dict(config,phase=phase)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid new-group limit.')
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    deps=dependencies or RolloutDependencies()
    with _writer(directory):
        if (directory/'complete.json').exists():
            if phase=='profile':return profile_verify(config,base,by_id,historical)
            from pilot_eval.stage0_rollout_results import verify
            return verify(path,root)
        runtime=deps.runtime()
        if runtime['execution_kind']!=config['execution_kind']:raise ValueError('Synthetic and real sampling sources/runtime cannot mix.')
        if phase=='profile':
            policy=replay_policy(config,runtime);units=policy['units'];execution={'reservation':{'job_sha256':_hash(phase_config)}}
        else:
            execution=_execution(config,base,Path(path).resolve().parent/'stage0.rollouts.execution.json',root,by_id,historical)
            units=execution['units']
            if runtime!=execution['runtime']:raise ValueError('R0 production requires the profiled runtime/session; retain and separately name a new cohort.')
        if (directory/'meta/runtime.json').exists() and json.loads((directory/'meta/runtime.json').read_text())!=runtime:
            raise ValueError('R0 resume runtime changed; preserve this product and prepare a new session plan.')
        saved=phase_saved(config,base,phase,units,runtime,by_id)
        if max_new_units==0:return dict(state='incomplete',completed=sum(x is not None for x in saved),total=len(units))
        if approval_path is None and not (directory/'meta/active-reservation.json').exists() and phase=='profile':
            raise ValueError('Approve a bounded profile allocation before loading a model.')
        reservation,remaining=_reservation(phase_config,execution,directory,root,runtime,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],remaining_seconds=remaining,started_at=_now()))
        started=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-started>=remaining:raise TimeoutError('R0 reserved wall time exhausted; reconcile before another bounded approval.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                deadline()
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',runtime)
                if progress:print(f'Stage 0 R0 {phase}: loading the pinned local FP32 untuned model; {len(units)} groups in this phase.',flush=True)
                load_start=time.monotonic();engine=deps.engine(config,root);load_seconds=time.monotonic()-load_start
                specials=engine.special_tokens() if hasattr(engine,'special_tokens') else {k:config['historical_runtime'][k] for k in ['eos_token_id','pad_token_id']}
                if any(type(specials.get(k)) is not int for k in ['eos_token_id','pad_token_id']):raise ValueError('Actual tokenizer EOS/PAD evidence is missing.')
                _save_frozen(directory/'inputs/special-tokens.json',specials)
                selected=[by_id[u['id']] for u in units]
                _save_frozen(directory/'inputs/tokenized-prompts.json',engine.prepare_inputs(selected))
                saved=phase_saved(config,base,phase,units,runtime,by_id)
                reusable={}
                if phase=='collection':
                    profile=profile_verify(config,base,by_id,historical)
                    old=phase_saved(config,base,'profile',profile['policy']['units'],runtime,by_id)
                    reusable={(u['id'],u['seed']):values for u,values in zip(profile['policy']['units'],old)}
                for n,unit in enumerate(units):
                    if saved[n] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic()
                    if engine.check_integrity().get('base_unchanged') is not True:raise ValueError('Sampling base integrity failed before group.')
                    reused=reusable.get((unit['id'],unit['seed']))
                    outputs=reused if reused is not None else engine.sample([by_id[unit['id']]],config['generation_settings'],unit['seed'])
                    deadline()
                    if len(outputs)!=8:raise ValueError('Expected exactly eight sampled continuations.')
                    rows=[normalize(by_id[unit['id']],value,d,origin=unit['origin'],seed=unit['seed'],limit=1024,eos=specials['eos_token_id']) for d,value in enumerate(outputs)]
                    integrity=engine.check_integrity()
                    if integrity.get('base_unchanged') is not True:raise ValueError('Sampling base integrity failed after group.')
                    folder=directory/'units'/unit['key'];_jsonl(folder/'responses.jsonl',rows)
                    _write_json(folder/'metadata.json',dict(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),
                        reservation_id=reservation['reservation_id'],seconds=time.monotonic()-begin,integrity=integrity,reused_profile=reused is not None))
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(phase_config),unit=unit,runtime_sha256=_hash(runtime),specials=specials),
                        files={k:file_hash(folder/k) for k in ['responses.jsonl','metadata.json']}))
                    saved[n]=rows;new+=1;_state(directory,'running',sum(v is not None for v in saved),len(units),reservation_id=reservation['reservation_id'])
                    if progress:print(f'Stage 0 R0 {phase}: {sum(v is not None for v in saved)}/{len(units)} sealed groups; {unit["id"]}; seed {unit["seed"]}.',flush=True)
                if progress:print(f'Stage 0 R0 {phase}: checking the full untuned base hash before publication.',flush=True)
                deadline();full_start=time.monotonic();integrity=engine.check_integrity(full=True);full_seconds=time.monotonic()-full_start;deadline()
                _write_json(attempt/'final-integrity.json',integrity)
                if integrity.get('base_unchanged') is not True:raise ValueError('Full untuned base hash failed; preserve this attempt.')
                if any(x is None for x in saved):return _state(directory,'incomplete',sum(v is not None for v in saved),len(units),reservation_id=reservation['reservation_id'])
                # Recheck immutable source payloads before publishing measurements.
                load(path,root)
                if phase=='profile':
                    values=[r for group in saved for r in group]
                    reference=[r for u in units for r in historical if r['group_id']==u['id']]
                    comparison=replay_comparison(reference,values) if policy['replay_eligible'] else dict(exact=False,draws=0,differences=[],criterion='historical replay unavailable; fresh uniform timing')
                    route='reuse-historical-128' if policy['replay_eligible'] and comparison['exact'] else 'uniform-fresh-512'
                    pending=config['extension_units'] if route=='reuse-historical-128' else config['uniform_units']
                    profile_keys={(u['id'],u['seed']) for u in units};reused_count=sum((u['id'],u['seed']) in profile_keys for u in pending)
                    costs=[]
                    for u,group in zip(units,saved):
                        meta=json.loads((directory/'units'/u['key']/'metadata.json').read_text())
                        costs.append(meta['seconds']*1024/max(len(r['generated_token_ids']) for r in group))
                    result=dict(state='awaiting-review',runtime=runtime,policy=policy,comparison=comparison,route=route,
                        production_groups=len(pending),profile_groups_reusable=reused_count,reservation_id=reservation['reservation_id'],
                        forecast=dict(conservative_total_seconds=load_seconds+full_seconds+max(costs)*(len(pending)-reused_count),
                            measured_load_seconds=load_seconds,measured_final_integrity_seconds=full_seconds,
                            cap_extrapolated_seconds_per_group=max(costs),method='worst measured group extrapolated to 1024 decoding steps, plus load/final integrity',
                            completion_guaranteed=False),real_gpu_measurement=config['execution_kind']=='real-cuda')
                    _write_json(directory/'results/profile.json',result)
                    files=['results/profile.json','meta/runtime.json','inputs/special-tokens.json','inputs/tokenized-prompts.json']
                    files += [str(p.relative_to(directory)) for p in (directory/'units').rglob('*') if p.is_file()]
                    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}))
                    _state(directory,'awaiting-review',2,2,reservation_id=reservation['reservation_id']);return result
                _write_json(directory/'meta/final-integrity.json',dict(**integrity,attempt_id=attempt.name))
                from pilot_eval.stage0_rollout_results import finish
                return finish(config,base,execution,saved,training,historical,root)
        except (Exception,KeyboardInterrupt) as exc:
            _state(directory,'failed',sum(v is not None for v in saved),len(units),error=str(exc),reservation_id=reservation['reservation_id'])
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,newly_sealed_groups=new,
                    reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),
                    final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None,
                    usage_reconciled=False,reservation_stays_charged=True))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))
