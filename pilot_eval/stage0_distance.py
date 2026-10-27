"""Teacher-forced dataset measurement using verified saved R0 memberships."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_now,budget_status
from pilot_eval.stage0_evaluation import _jsonl,_reservation,_attempt_timing,_recover_owner
from pilot_eval.stage0_budget import reserve
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.stage0_rollout_plan import load as load_rollouts
from pilot_eval.stage0_rollout_results import verify as verify_rollouts
from pilot_eval.stage0_distance_math import prepare_answers,check_arrays
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_distance.py','stage0_distance_math.py','stage0_distance_backend.py','stage0_distance_report.py']}


def prepare(rollouts_path,output_root,name,*,context_chunk=16,workspace_bytes=536870912):
    root=Path(output_root).resolve();safe_name(name);rp=inside(root,rollouts_path)
    result=verify_rollouts(rp,root);rollout,source,training,_=load_rollouts(rp,root)
    execution=json.loads((rp.parent/'stage0.rollouts.execution.json').read_text())
    if any(type(v) is not int or v<1 for v in [context_chunk,workspace_bytes]):raise ValueError('Positive scoring chunk/workspace required.')
    draws=[json.loads(x) for x in (source/'results/rollouts.jsonl').read_text().splitlines()]
    datasets=json.loads((source/'results/datasets.json').read_text())
    special=json.loads((source/'collection/inputs/special-tokens.json').read_text());eos=special['eos_token_id']
    answers,memberships=prepare_answers(training,draws,datasets,eos)
    path=root/'plans'/name/'stage0.distance.json';directory=root/f'runs/stage0/{name}/e06a_distance'
    config=dict(protocol='stage0-distance-v1',schema_version=1,name=name,run_path=str(directory.relative_to(root)),
        rollouts_path=str(rp.relative_to(root)),rollouts_sha256=file_hash(rp),rollouts_complete_sha256=file_hash(source/'complete.json'),
        model_identity=rollout['model_identity'],weights_dir=rollout['weights_dir'],runtime=execution['runtime'],execution_kind=rollout['execution_kind'],
        context_chunk=context_chunk,workspace_bytes=workspace_bytes,implementation=implementation(),numpy_version=np.__version__,
        unique_answer_ids=[a['answer_id'] for a in answers],answers_sha256=_hash(answers),memberships_sha256=_hash(memberships),
        mask='saved SFT prompt masked; gold through supervised EOS; sampled generated IDs including observed EOS, no artificial EOS on cap',
        units='nats per answer token',primary='nll',excess_surprise='nll minus model entropy; descriptive, not KL',
        rank='1 + vocabulary logits strictly greater than target; ties share rank',position_windows='disjoint first32/middle/last32; first32 precedence on answers shorter than64',own_all_gate=.02,budget_ceiling_units=20,
        statistics=dict(draws=10000,seed=42,percentiles=[2.5,97.5],unit='whole problem',weighting='ratio of summed token surprise to summed token count'),
        premise=dict(sampled_distribution='HF temperature1, topk0, topp1',score_model='identical audited untuned weights',
            runtime='same sampled production session',stopping='observed EOS included; cap gets no synthetic EOS',
            finite_sample_note='Token-pooled stopped-sequence ratios need not have exact zero finite-sample expectation; gate failure is retained, never widened.'))
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _jsonl(directory/'inputs/answers.jsonl',answers);_save_frozen(directory/'inputs/memberships.json',memberships)
        _save_frozen(directory/'inputs/datasets.json',datasets)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),unique_answers=len(answers),membership_counts={k:len(v) for k,v in memberships.items()},resources='GPU scoring; CPU summary/report'))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(answers))
    return path


def load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-distance-v1' or config['implementation']!=implementation() or config['numpy_version']!=np.__version__ or config['own_all_gate']!=.02:raise ValueError('Distance implementation/settings changed; preserve evidence and use a new name.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):raise ValueError('Distance plan identity changed.')
    rp=inside(root,config['rollouts_path'])
    if file_hash(rp)!=config['rollouts_sha256']:raise ValueError('R0 plan changed.')
    rollout,source,_,_=load_rollouts(rp,root)
    if file_hash(source/'complete.json')!=config['rollouts_complete_sha256']:raise ValueError('R0 completion changed.')
    # R0 verifier checks all response, token, cohort and parent identities.
    verify_rollouts(rp,root)
    answers=[json.loads(x) for x in (directory/'inputs/answers.jsonl').read_text().splitlines()]
    memberships=json.loads((directory/'inputs/memberships.json').read_text())
    if _hash(answers)!=config['answers_sha256'] or _hash(memberships)!=config['memberships_sha256'] or [a['answer_id'] for a in answers]!=config['unique_answer_ids']:raise ValueError('Frozen answer/token/membership mapping changed.')
    if json.loads((directory/'inputs/datasets.json').read_text())!=json.loads((source/'results/datasets.json').read_text()):raise ValueError('Cohort membership metadata changed.')
    return config,directory,answers,memberships


def read_unit(config,directory,answer):
    folder=directory/'units'/answer['answer_id'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),answer_sha256=_hash(answer)) or set(seal.get('files',{}))!={'tokens.npz','metadata.json'}:raise ValueError('Answer-scoring unit identity changed.')
    for n,digest in seal['files'].items():
        if file_hash(folder/n)!=digest:raise ValueError('Answer token evidence changed.')
    meta=json.loads((folder/'metadata.json').read_text());attempt=directory/'attempts'/meta['attempt_id']
    if file_hash(attempt/'start.json')!=meta['attempt_start_sha256'] or meta['integrity'].get('base_unchanged') is not True:raise ValueError('Scoring attempt identity/integrity failed.')
    timing=_attempt_timing(attempt,required=False);final=attempt/'final-integrity.json'
    if final.exists() and (json.loads(final.read_text()).get('base_unchanged') is not True or (timing is not None and timing['final_integrity_sha256']!=file_hash(final))):raise ValueError('Failed full scoring integrity invalidates this attempt.')
    with np.load(folder/'tokens.npz',allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    return check_arrays(arrays,answer)


def _deps():
    from pilot_eval.stage0_distance_backend import DistanceDependencies
    return DistanceDependencies()


def runtime(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,directory,_,_=load(path,root);deps=dependencies or _deps();info=deps.runtime()
    return dict(runtime=info,matches_sampled_runtime=info==config['runtime'],budget=budget_status(root),model_loaded=False,
        storage=deps.preflight(config,root) if hasattr(deps,'preflight') else dict(passed=True,scope='fixture'))


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    root=Path(output_root).resolve();config,directory,answers,_=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid scoring unit limit.')
    with _writer(directory):
        if (directory/'measurement.complete.json').exists():return verify_measurement(path,root)
        saved=[read_unit(config,directory,a) for a in answers]
        if max_new_units==0:return _state(directory,'incomplete',sum(v is not None for v in saved),len(answers))
        deps=dependencies or _deps();info=deps.runtime()
        if info!=config['runtime']:raise ValueError('Scoring requires the sampled production runtime/session; preserve this plan on a change.')
        execution_path=Path(path).resolve().parent/'stage0.distance.execution.json'
        execution=_execution(config,directory,root,execution_path) if execution_path.exists() else {'reservation':{'job_sha256':_hash(config)}}
        if not execution_path.exists() and max_new_units!=1:
            raise ValueError('Before production, run one bounded answer, inspect its forecast and freeze a separately reviewed measured budget.')
        if approval_path is None and not (directory/'meta/active-reservation.json').exists() and not execution_path.exists():raise ValueError('Approve a separately bounded scoring allocation first.')
        reservation,remaining=_reservation(config,execution,directory,root,info,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],remaining_seconds=remaining,started_at=_now()))
        started=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-started>=remaining:raise TimeoutError('Scoring allocation exhausted; reconcile before renewing.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                deadline();_save_frozen(directory/'meta/runtime.json',info)
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                if progress:print('Stage 0 distance: loading the local untuned FP32 model.',flush=True)
                engine=deps.engine(config,root)
                for n,answer in enumerate(answers):
                    if saved[n] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic()
                    arrays=check_arrays(engine.score(answer),answer);deadline();proof=engine.check_integrity()
                    if proof.get('base_unchanged') is not True:raise ValueError('Scoring model integrity failed.')
                    folder=directory/'units'/answer['answer_id'];folder.mkdir(parents=True,exist_ok=True)
                    temporary=folder/'tokens.tmp.npz';np.savez(temporary,**arrays);temporary.replace(folder/'tokens.npz')
                    _write_json(folder/'metadata.json',dict(answer_id=answer['answer_id'],seconds=time.monotonic()-begin,
                        attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),integrity=proof))
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),answer_sha256=_hash(answer)),files={k:file_hash(folder/k) for k in ['tokens.npz','metadata.json']}))
                    saved[n]=arrays;new+=1;_state(directory,'running',sum(v is not None for v in saved),len(answers),reservation_id=reservation['reservation_id'])
                    if progress:print(f'Stage 0 distance: {sum(v is not None for v in saved)}/{len(answers)} unique answers sealed; {len(answer["target_ids"])} supervised tokens.',flush=True)
                deadline();proof=engine.check_integrity(full=True);deadline();_write_json(attempt/'final-integrity.json',proof)
                if proof.get('base_unchanged') is not True:raise ValueError('Full scoring integrity failed; preserve attempt evidence.')
                load(path,root)
                if all(v is not None for v in saved):
                    _write_json(directory/'meta/final-integrity.json',dict(**proof,attempt_id=attempt.name))
                    files=['config.json','meta/runtime.json','inputs/answers.jsonl','inputs/memberships.json','inputs/datasets.json','meta/final-integrity.json']+[str(p.relative_to(directory)) for p in (directory/'units').rglob('complete.json')]
                    _write_json(directory/'measurement.complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}))
                    return _state(directory,'measurement-complete',len(answers),len(answers),reservation_id=reservation['reservation_id'],cpu_report_pending=True)
                return _state(directory,'incomplete',sum(v is not None for v in saved),len(answers),reservation_id=reservation['reservation_id'])
        except (Exception,KeyboardInterrupt) as exc:
            _state(directory,'failed',sum(v is not None for v in saved),len(answers),error=str(exc),reservation_id=reservation['reservation_id'])
            log=directory/'logs/errors.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__))+'\n')
            raise
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,reservation_id=reservation['reservation_id'],newly_sealed_answers=new,
                    start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
                _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def verify_measurement(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,_=load(path,root);seal=json.loads((directory/'measurement.complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Scoring completion identity differs.')
    for p,digest in seal['files'].items():
        if file_hash(inside(directory,p))!=digest:raise ValueError('Scoring completed payload changed.')
    if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime'] or json.loads((directory/'meta/final-integrity.json').read_text()).get('base_unchanged') is not True:raise ValueError('Scoring runtime/final integrity mismatch.')
    for answer in answers:
        if read_unit(config,directory,answer) is None:raise ValueError('Missing token-scoring unit.')
    return dict(measurement_complete=True,unique_answers=len(answers),real_gpu_measurement=config['execution_kind']=='real-cuda',cpu_report_pending=not (directory/'complete.json').exists())


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,_=load(path,root)
    result=verify_measurement(path,root) if (directory/'measurement.complete.json').exists() else None
    return dict(measurement=result,completed=sum(read_unit(config,directory,a) is not None for a in answers),total=len(answers),budget=budget_status(root),report=str(directory/'report.html'))


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory,_,_=load(path,root)
    return _recover_owner(directory,owner_token,review_notes)


def forecast(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,_=load(path,root)
    values=[];evidence={}
    for answer in answers:
        value=read_unit(config,directory,answer)
        if value is not None:
            folder=directory/'units'/answer['answer_id'];meta=json.loads((folder/'metadata.json').read_text())
            values.append((answer,meta));evidence[str((folder/'complete.json').relative_to(directory))]=file_hash(folder/'complete.json')
    if not values:raise ValueError('Measure at least one bounded unique answer before a production forecast.')
    attempts={v['attempt_id'] for _,v in values}
    overhead=0.
    for identifier in attempts:
        attempt=directory/'attempts'/identifier;timing=_attempt_timing(attempt)
        final=json.loads((attempt/'final-integrity.json').read_text())
        if final.get('base_unchanged') is not True:raise ValueError('Profile attempt full integrity failed.')
        overhead=max(overhead,timing['seconds'])
        evidence[str((attempt/'timing.complete.json').relative_to(directory))]=file_hash(attempt/'timing.complete.json')
    # Attention is quadratic in total sequence length. Extrapolate the worst
    # measured seconds per squared length; expose every term rather than an ETA guarantee.
    rate=max(meta['seconds']/len(answer['input_ids'])**2 for answer,meta in values)
    missing=[a for a in answers if read_unit(config,directory,a) is None]
    result=dict(config_sha256=_hash(config),measured_unique_answers=len(values),remaining_unique_answers=len(missing),
        seconds_per_squared_sequence_token=rate,measured_attempt_overhead_seconds=overhead,
        conservative_total_seconds=overhead+rate*sum(len(a['input_ids'])**2 for a in missing),
        evidence=evidence,method='worst measured per-answer seconds / total sequence length squared, extrapolated to missing sequences; plus worst measured whole attempt',completion_guaranteed=False)
    _write_json(directory/'results/forecast.json',result);return result


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,_,_=load(path,root);review=json.loads(Path(review_path).read_text())
    target=Path(path).resolve().parent/'stage0.distance.execution.json'
    required={'reviewer','notes','session_id','source_reviewed','max_units_per_hour','max_seconds','forecast_sha256'}
    if set(review)!=required or review['session_id']!=config['runtime']['session_id']:raise ValueError('Separate same-session measured scoring review required.')
    measured=json.loads((directory/'results/forecast.json').read_text())
    if review['forecast_sha256']!=file_hash(directory/'results/forecast.json') or measured['config_sha256']!=_hash(config) or review['max_seconds']<measured['conservative_total_seconds']:raise ValueError('Scoring production budget must bind and cover the measured forecast.')
    for name,digest in measured['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Measured scoring evidence changed.')
    with _writer(directory):
        if target.exists():
            frozen=_execution(config,directory,root,target)
            if frozen['review']!=review:raise ValueError('Scoring production review changed; use a new plan.')
            return target
        reservation=reserve(root,_hash(config),{k:v for k,v in review.items() if k!='forecast_sha256'})
        frozen=dict(config_sha256=_hash(config),review=review,reservation=reservation,forecast=measured,real_gpu_authorized=config['execution_kind']=='real-cuda')
        _save_frozen(target,frozen);_save_frozen(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
        _write_json(directory/'meta/active-reservation.json',dict(reservation_id=reservation['reservation_id']))
    return target


def _execution(config,directory,root,path):
    value=json.loads(path.read_text());marker=json.loads((directory/'production-freeze.json').read_text())
    if marker!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)) or value['config_sha256']!=_hash(config):raise ValueError('Frozen scoring execution changed.')
    for name,digest in value['forecast']['evidence'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Frozen scoring profile evidence changed.')
    return value
