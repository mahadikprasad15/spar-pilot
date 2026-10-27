"""Frozen paired greedy collection with sealed batches and CPU reconstruction."""
import csv
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid

from pilot_eval.run import _write_json
from pilot_eval.stage0_budget import reserve, reservations
from pilot_eval.stage0_edits import VARIANTS
from pilot_eval.stage0_evaluation_records import OUTPUT_FIELDS, records, summarize
from pilot_eval.stage0_qualification import _load as load_qualification, _verify as verify_qualification
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _now, _state, _writer, budget_status
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen


def _implementation():
    folder=Path(__file__).parent
    return {name:file_hash(folder/name) for name in [
        'stage0_evaluation.py','stage0_evaluation_records.py','stage0_evaluation_backend.py','scoring.py']}


def _execution(path,root):
    path=inside(root,path);execution=json.loads(path.read_text())
    if execution.get('protocol')!='stage0-paired-execution-v1':
        raise ValueError('Use a frozen Ticket 04 paired execution plan.')
    qpath=inside(root,execution['qualification_path'])
    if file_hash(qpath)!=execution['qualification_sha256']:
        raise ValueError('Qualification source changed.')
    qualification,directory=load_qualification(qpath,root)
    diagnostic=verify_qualification(qualification,directory)
    marker=json.loads((directory/'production-freeze.json').read_text())
    if marker!=dict(path=str(path.relative_to(root)),sha256=file_hash(path)):
        raise ValueError('Frozen production selection changed.')
    receipt_path=inside(root,execution['qualification_receipt'])
    if file_hash(receipt_path)!=execution['qualification_receipt_sha256']:
        raise ValueError('Numerical approval receipt changed.')
    receipt=json.loads(receipt_path.read_text());review=receipt['review']
    real=diagnostic['execution_kind']=='real-cuda'
    if (not diagnostic['eligible_for_review'] or receipt.get('qualified') is not True
            or receipt.get('real_gpu_qualified') is not real
            or execution.get('real_gpu_authorized') is not real
            or receipt['config_sha256']!=_hash(qualification)
            or review['diagnostic_sha256']!=file_hash(directory/'complete.json')
            or review['absolute_gate']!=diagnostic['proposed_absolute_gate']
            or execution['runtime']!=diagnostic['runtime']
            or execution['variants']!=qualification['variants']
            or execution['decoding']!=qualification['decoding']
            or execution['cohort_sha256']!=qualification['cohort_sha256']):
        raise ValueError('Frozen plan does not bind eligible approved qualification evidence.')
    return execution,qualification,directory


def _items(root,path):
    parsed=json.loads(inside(root,path).read_text())
    rows=parsed['items'] if isinstance(parsed,dict) else parsed
    if len(rows)!=150 or len({r['id'] for r in rows})!=150:
        raise ValueError('Expected the original 150 unique held-out items.')
    return rows


def _configuration(path,root,batch_size,batches_per_unit):
    execution,qualification,qdir=_execution(path,root)
    if type(batch_size) is not int or not 1<=batch_size<=150:
        raise ValueError('Batch size must be an integer between 1 and 150.')
    if type(batches_per_unit) is not int or not 1<=batches_per_unit<=150:
        raise ValueError('Batches per durable unit must be between 1 and 150.')
    kind=execution['runtime']['execution_kind']
    if kind=='real-cuda' and batch_size!=1:
        raise ValueError('Ticket 04 qualified singleton generation; larger real batches need separate agreement evidence.')
    items=_items(root,execution['cohort_path'])
    decoding=execution['decoding']
    if (decoding.get('do_sample') is not False or decoding.get('max_new_tokens')!=1024
            or decoding.get('num_beams')!=1 or decoding.get('num_return_sequences')!=1
            or any(type(decoding.get(k)) is not int or decoding[k]<0 for k in ['eos_token_id','pad_token_id'])):
        raise ValueError('Frozen greedy decoding/EOS/PAD settings are incomplete or differ from the approved protocol.')
    conditional=qualification.get('qualification_kind')=='conditional-intruders'
    variant_order=list(execution['variants']) if conditional else list(VARIANTS)
    if not conditional and set(execution['variants'])!=set(VARIANTS):
        raise ValueError('Paired evaluation requires the entire ten-variant catalogue including V0a/V0b.')
    units=[]
    unit_size=batch_size*batches_per_unit
    for start in range(0,len(items),unit_size):
        index=start//unit_size;selected=items[start:start+unit_size]
        generation_batches=[dict(ids=[r['id'] for r in selected[offset:offset+batch_size]],
                                 seed=42+(start+offset)//batch_size)
                            for offset in range(0,len(selected),batch_size)]
        for variant in variant_order:
            units.append(dict(key=f'batch-{index:04d}-{variant}',variant=variant,
                ids=[r['id'] for r in selected],seed=generation_batches[0]['seed'],
                generation_batches=generation_batches,
                item_seeds={i:g['seed'] for g in generation_batches for i in g['ids']}))
    adapter_files={variant:{name:file_hash(inside(root,row['adapter_path'])/name)
                           for name in ['adapter_config.json','adapter_model.safetensors','complete.json']}
                   for variant,row in execution['variants'].items()}
    config=dict(schema_version=1,protocol='stage0-paired-greedy-v1',name=qualification['name'],
        run_path=execution['run_path'],execution_path=str(inside(root,path).relative_to(root)),
        execution_sha256=file_hash(path),implementation=_implementation(),execution_kind=kind,
        runtime=execution['runtime'],cohort_path=execution['cohort_path'],cohort_sha256=execution['cohort_sha256'],
        item_ids=[r['id'] for r in items],variants=execution['variants'],variant_order=variant_order,
        adapter_files=adapter_files,model_identity=qualification['model_identity'],
        qualification_inputs_path=str((qdir/'meta/tokenized-inputs.json').relative_to(root)),
        qualification_inputs_sha256=file_hash(qdir/'meta/tokenized-inputs.json'),
        weights_dir=qualification['weights_dir'],tokenizer_files=qualification['tokenizer_files'],
        prompt_contract=qualification.get('evaluation_config',{}).get('prompt_template',{}),
        decoding={**execution['decoding'],'use_cache':True},scorer=execution['scorer'],strict_scorer='gsm8k-strict',
        cap_reward=0,batch_size=batch_size,batches_per_unit=batches_per_unit,units=units,seed=42,
        scheduling='ordered cohort batches; V0a,V0b then all edits within each batch; same 42+batch-index seed across variants',
        real_batch_qualification='singleton only; larger CPU fixture batches do not qualify CUDA batching',
        integrity_policy='base SHA256 at load and clean attempt end; parameter identity/version guards at each sealed batch',
        human_review_status='not-applied',budget_ceiling_units=20)
    if conditional:
        config.update(mixed_catalogue_path=qualification['mixed_catalogue_path'],
            mixed_catalogue_sha256=qualification['mixed_catalogue_sha256'],collection_kind='conditional-intruders')
    return config,execution,qdir


def prepare(execution_path,output_root,*,batch_size=1,batches_per_unit=8):
    root=Path(output_root).resolve();execution_path=inside(root,execution_path)
    config,_,_= _configuration(execution_path,root,batch_size,batches_per_unit)
    path=execution_path.parent/'stage0.evaluation.json';directory=inside(root,config['run_path'])
    with _writer(directory):
        _save_frozen(path,config)
        _save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(path.relative_to(root)),
            config_sha256=_hash(config),expected_responses=_count(config),total_batches=len(config['units']),
            resources='one qualified CUDA GPU for run; CPU for prepare/status/verify',
            execution_kind=config['execution_kind'],scope='frozen collection manifest; completion is recorded separately'))
        if not (directory/'meta/status.json').exists():
            _state(directory,'prepared',0,len(config['units']),completed_responses=0,total_responses=_count(config))
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='stage0-paired-greedy-v1' or config.get('implementation')!=_implementation():
        raise ValueError('Evaluation implementation changed; preserve evidence and use a new execution identity.')
    expected,execution,_=_configuration(inside(root,config['execution_path']),root,config['batch_size'],config['batches_per_unit'])
    if config!=expected:
        raise ValueError('Evaluation source/configuration changed.')
    directory=inside(root,config['run_path'])
    if (json.loads((directory/'config.json').read_text())!=config
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config)):
        raise ValueError('Evaluation plan/run manifest identity differs.')
    return config,execution,directory


def _jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _inputs(engine,config,root):
    items=_items(root,config['cohort_path']);inputs=engine.prepare_inputs(items)
    if not isinstance(inputs,list) or [r.get('id') for r in inputs]!=config['item_ids']:
        raise ValueError('Tokenized input order differs from frozen cohort.')
    for row,item in zip(inputs,items):
        ids=row['input_ids'];mask=row['attention_mask']
        if (not ids or any(type(t) is not int or t<0 for t in ids) or mask!=[1]*len(ids)
                or ('prompt_tokens' in item and len(ids)!=item['prompt_tokens'])):
            raise ValueError('Frozen prompt token IDs/mask/count disagree.')
        row.update(prompt_sha256=_hash(item['prompt']),position_ids=list(range(len(ids))))
    if config['execution_kind']=='real-cuda':
        qualified=json.loads(inside(root,config['qualification_inputs_path']).read_text())['sequences']
        by_id={row['id']:row for row in inputs}
        for row in qualified:
            if by_id[row['id']]['input_ids']!=row['input_ids'][:row['prompt_tokens']]:
                raise ValueError('Actual generation prompt IDs differ from the qualified common inputs.')
    return dict(scope='one unpadded prompt-token sequence per held-out item; shared by every variant',sequences=inputs)


def _identity(config,unit,runtime_digest,input_digest):
    return dict(config_sha256=_hash(config),runtime_sha256=runtime_digest,input_sha256=input_digest,key=unit['key'])


def _intact(config,proof):
    return proof.get('base_unchanged') is True and (config['execution_kind']!='real-cuda' or proof.get('adapters_unchanged') is True)


def _loaded_checks(config,directory):
    value=json.loads((directory/'meta/loaded-adapters.json').read_text())
    if value['execution_kind']!=config['execution_kind']:
        raise ValueError('Loaded-adapter evidence has the wrong execution kind.')
    if config['execution_kind']=='real-cuda':
        checks=value['checks']
        if (len(checks)!=len(config['variants']) or {r['variant'] for r in checks}!=set(config['variants'])
                or any(r['exact_payload_match'] is not True or r['tensor_count']!=392 for r in checks)):
            raise ValueError('Real collection requires exact in-memory checks for all 392 factors of every adapter.')


def _read_unit(config,directory,unit,items,runtime_digest,input_digest,attempt_cache):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if (seal.get('identity')!=_identity(config,unit,runtime_digest,input_digest)
            or set(seal.get('files',{}))!={'responses.jsonl','metadata.json'}):
        raise ValueError('Evaluation batch/config/runtime receipt changed.')
    for name,digest in seal['files'].items():
        if file_hash(folder/name)!=digest:raise ValueError('Evaluation batch payload changed.')
    metadata=json.loads((folder/'metadata.json').read_text())
    if not _intact(config,metadata['integrity']) or metadata['unit_key']!=unit['key']:
        raise ValueError('Evaluation batch lacks successful model integrity evidence.')
    attempt=directory/'attempts'/metadata['attempt_id']
    if metadata['attempt_id'] not in attempt_cache:
        start_hash=file_hash(attempt/'start.json')
        final=attempt/'final-integrity.json'
        timing=_attempt_timing(attempt,required=False)
        if timing is not None and timing['final_integrity_sha256'] is not None:
            if not final.exists() or file_hash(final)!=timing['final_integrity_sha256']:
                raise ValueError('Attempt final-integrity evidence changed.')
        if final.exists() and not _intact(config,json.loads(final.read_text())):
            raise ValueError('A full model-integrity failure invalidates batches from that attempt; preserve this comparison.')
        attempt_cache[metadata['attempt_id']]=start_hash
    if attempt_cache[metadata['attempt_id']]!=metadata['attempt_start_sha256']:
        raise ValueError('Batch attempt-start identity changed.')
    saved=[json.loads(line) for line in (folder/'responses.jsonl').read_text().splitlines()]
    expected=records(config,unit,[items[i] for i in unit['ids']],
        [{k:r[k] for k in OUTPUT_FIELDS} for r in saved],runtime_sha256=runtime_digest,input_sha256=input_digest)
    if saved!=expected:raise ValueError('Saved batch identity, token metadata or scores disagree with reconstruction.')
    return saved


def _attempt_timing(attempt,*,required=True):
    timing=attempt/'timing.json';seal=attempt/'timing.complete.json'
    if not timing.exists() or not seal.exists():
        if required:
            raise ValueError('Interrupted attempt has unmeasured elapsed time; reconcile its reservation and approve a new bound.')
        return None
    if json.loads(seal.read_text())!={'sha256':file_hash(timing)}:
        raise ValueError('Recorded attempt timing changed.')
    value=json.loads(timing.read_text())
    if type(value['seconds']) not in [int,float] or not math.isfinite(value['seconds']) or value['seconds']<0:
        raise ValueError('Recorded elapsed time must be finite and nonnegative.')
    if value['start_sha256']!=file_hash(attempt/'start.json'):
        raise ValueError('Recorded attempt start changed.')
    return value


def _saved(config,directory,root):
    runtime_path=directory/'meta/runtime.json';input_path=directory/'inputs/tokenized-prompts.json'
    if not runtime_path.exists() or not input_path.exists():
        if any((directory/'units').glob('*/complete.json')):
            raise ValueError('Sealed batches are missing their runtime or tokenized inputs.')
        return [None]*len(config['units'])
    if json.loads(runtime_path.read_text())!=config['runtime']:
        raise ValueError('Saved evaluation runtime identity changed.')
    _loaded_checks(config,directory)
    inputs=json.loads(input_path.read_text())['sequences']
    if [r['id'] for r in inputs]!=config['item_ids']:
        raise ValueError('Saved tokenized input order changed.')
    items={r['id']:r for r in _items(root,config['cohort_path'])}
    for row in inputs:
        ids=row['input_ids']
        if row['prompt_sha256']!=_hash(items[row['id']]['prompt']) or row['attention_mask']!=[1]*len(ids) or row['position_ids']!=list(range(len(ids))):
            raise ValueError('Saved prompt identity/mask/position mapping changed.')
    runtime_digest=file_hash(runtime_path);input_digest=file_hash(input_path)
    attempt_cache={}
    return [_read_unit(config,directory,u,items,runtime_digest,input_digest,attempt_cache) for u in config['units']]


def _reservation(config,execution,directory,root,runtime,approval_path):
    usage=budget_status(root)
    if usage['state']!='observed' or usage['spent_units']>=20:
        raise ValueError('Current observed cumulative Stage 0 usage must remain below the 20-unit ceiling.')
    if approval_path is not None:
        approval=json.loads(Path(approval_path).read_text())
        if approval.get('session_id')!=runtime['session_id']:
            raise ValueError('Resume budget approval belongs to another runtime session.')
        row=reserve(root,_hash(config),approval)
    else:
        selected=directory/'meta/active-reservation.json'
        rid=json.loads(selected.read_text())['reservation_id'] if selected.exists() else execution['reservation']['reservation_id']
        row=reservations(root).get(rid)
        if not row or row['state']!='active':
            raise ValueError('No active production reservation; reconcile usage and supply a new bounded --approval.')
        if row['job_sha256'] not in [execution['reservation']['job_sha256'],_hash(config)]:
            raise ValueError('Reservation belongs to another workload.')
    if row['approval']['session_id']!=runtime['session_id']:
        raise ValueError('Production reservation/runtime session differs.')
    _write_json(directory/'meta/active-reservation.json',dict(reservation_id=row['reservation_id']))
    used=0.
    for start in (directory/'attempts').glob('*/start.json'):
        if json.loads(start.read_text())['reservation_id']!=row['reservation_id']:continue
        used+=_attempt_timing(start.parent)['seconds']
    remaining=row['approval']['max_seconds']-used
    if remaining<=0:raise ValueError('Reserved wall time is exhausted; reconcile usage before another approval.')
    return row,remaining


def _finish(config,directory,root):
    saved=_saved(config,directory,root)
    if any(batch is None for batch in saved):raise ValueError('Evaluation is incomplete.')
    rows=[r for batch in saved for r in batch]
    if len(rows)!=_count(config) or len({r['record_id'] for r in rows})!=_count(config):
        raise ValueError('Missing or duplicate paired responses.')
    summary=summarize(config,rows)
    _jsonl(directory/'results/responses.jsonl',rows)
    _write_json(directory/'results/results.json',summary)
    columns=list(next(iter(summary['variants'].values())))
    with (directory/'results/summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['variant',*columns]);writer.writeheader()
        writer.writerows(dict(variant=k,**v) for k,v in summary['variants'].items())
    (directory/'results/results.md').write_text('# Stage 0 raw paired greedy results\n\n'
        f'Execution: {config["execution_kind"]}.\n\n'
        '| Variant | n | Flexible accuracy | Strict accuracy | Cap rate | Mean content tokens |\n|---|---:|---:|---:|---:|---:|\n'+
        '\n'.join(f'| {k} | {v["n"]} | {v["flexible_accuracy"]:.3f} | {v["strict_accuracy"]:.3f} | {v["cap_rate"]:.3f} | {v["mean_content_tokens"]:.1f} |'
                  for k,v in summary['variants'].items())+
        '\n\nRaw automatic scores; no human review overlay or non-inferiority claim. '+summary['interpretation']+'\n')
    paths=['config.json','meta/runtime.json','meta/loaded-adapters.json','inputs/tokenized-prompts.json','results/responses.jsonl',
           'results/results.json','results/summary.csv','results/results.md','meta/final-integrity.json']
    paths += [f'units/{u["key"]}/complete.json' for u in config['units']]
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in paths}))
    _state(directory,'complete',len(saved),len(saved),completed_responses=_count(config),total_responses=_count(config))
    return summary


def run(path,output_root,*,dependencies=None,max_new_units=None,approval_path=None,progress=True):
    root=Path(output_root).resolve();config,execution,directory=_load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):
        raise ValueError('Invalid new-batch limit.')
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory,root)
        saved=_saved(config,directory,root)
        if max_new_units==0:return _progress(directory,config,saved,'incomplete')
        from pilot_eval.stage0_evaluation_backend import EvaluationDependencies
        from pilot_eval.stage0_qualification_backend import deadline_guard
        deps=dependencies or _dependencies(config);runtime=deps.runtime()
        if runtime!=config['runtime']:
            raise ValueError('Paired evaluation requires the exact qualified runtime/session; preserve this comparison on a change.')
        reservation,remaining=_reservation(config,execution,directory,root,runtime,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],started_at=_now(),remaining_seconds=remaining))
        started=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-started>=remaining:
                raise TimeoutError('Reserved evaluation wall time exhausted; reconcile usage before resuming.')
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                deadline()
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',runtime)
                engine=deps.engine(config,root)
                _save_frozen(directory/'meta/loaded-adapters.json',dict(execution_kind=config['execution_kind'],
                    checks=getattr(engine,'loaded_adapter_checks',[]),
                    scope='actual in-memory FP32 payload comparison' if config['execution_kind']=='real-cuda' else 'synthetic fixture; no real model load'))
                _loaded_checks(config,directory)
                _save_frozen(directory/'inputs/tokenized-prompts.json',_inputs(engine,config,root))
                saved=_saved(config,directory,root)
                items={r['id']:r for r in _items(root,config['cohort_path'])}
                runtime_digest=file_hash(directory/'meta/runtime.json');input_digest=file_hash(directory/'inputs/tokenized-prompts.json')
                for index,unit in enumerate(config['units']):
                    if saved[index] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic()
                    before=engine.check_integrity()
                    if not _intact(config,before):
                        raise ValueError(f'Base/adapter parameter integrity failed before generation: {before}')
                    rows=[items[i] for i in unit['ids']]
                    outputs=[]
                    for generation in unit['generation_batches']:
                        deadline()
                        outputs.extend(engine.generate([items[i] for i in generation['ids']],unit['variant'],generation['seed']))
                        if progress:print(f'Stage 0 greedy: {unit["key"]}; {len(outputs)}/{len(rows)} generated; pending durable seal.',flush=True)
                    deadline()
                    batch=records(config,unit,rows,outputs,runtime_sha256=runtime_digest,input_sha256=input_digest)
                    integrity=engine.check_integrity()
                    if not _intact(config,integrity):
                        raise ValueError(f'Base/adapter parameter integrity failed after generation: {integrity}')
                    folder=directory/'units'/unit['key'];_jsonl(folder/'responses.jsonl',batch)
                    _write_json(folder/'metadata.json',dict(unit_key=unit['key'],attempt_id=attempt.name,
                        attempt_start_sha256=file_hash(attempt/'start.json'),
                        reservation_id=reservation['reservation_id'],seconds=time.monotonic()-begin,integrity=integrity))
                    _write_json(folder/'complete.json',dict(identity=_identity(config,unit,runtime_digest,input_digest),
                        files={p:file_hash(folder/p) for p in ['responses.jsonl','metadata.json']}))
                    saved[index]=batch;new+=1
                    state=_progress(directory,config,saved,'running',reservation_id=reservation['reservation_id'])
                    if progress:print(f'Stage 0 greedy: {state["completed"]}/{state["total"]} sealed batches; '
                        f'{state["completed_responses"]}/{_count(config)} responses; {unit["variant"]}; batch {config["batch_size"]}.',flush=True)
                deadline();integrity=engine.check_integrity(full=True);deadline()
                _write_json(attempt/'final-integrity.json',integrity)
                if not _intact(config,integrity):
                    raise ValueError(f'Full frozen-base/adapter integrity failed; preserve this attempt: {integrity}')
                if all(batch is not None for batch in saved):
                    _write_json(directory/'meta/final-integrity.json',dict(**integrity,attempt_id=attempt.name))
                    return _finish(config,directory,root)
                return _progress(directory,config,saved,'incomplete',reservation_id=reservation['reservation_id'])
        except (Exception,KeyboardInterrupt) as exc:
            _progress(directory,config,saved,'failed',error=str(exc),reservation_id=reservation['reservation_id'])
            with (directory/'logs/errors.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(error=str(exc),type=type(exc).__name__,attempt_id=attempt.name))+'\n')
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


def _progress(directory,config,saved,state,**extra):
    return _state(directory,state,sum(b is not None for b in saved),len(saved),
        completed_responses=sum(len(b) for b in saved if b is not None),total_responses=_count(config),**extra)


def _verify(config,directory,root):
    marker=json.loads((directory/'complete.json').read_text())
    required={'config.json','meta/runtime.json','meta/loaded-adapters.json','inputs/tokenized-prompts.json','results/responses.jsonl',
              'results/results.json','results/summary.csv','results/results.md','meta/final-integrity.json'}
    required|={f'units/{u["key"]}/complete.json' for u in config['units']}
    if marker.get('config_sha256')!=_hash(config) or set(marker.get('files',{}))!=required:
        raise ValueError('Paired evaluation completion receipt is inconsistent.')
    for relative,digest in marker['files'].items():
        if file_hash(inside(directory,relative))!=digest:raise ValueError('Paired evaluation completion payload changed.')
    if not _intact(config,json.loads((directory/'meta/final-integrity.json').read_text())):
        raise ValueError('Completion lacks a full final base-integrity check.')
    batches=_saved(config,directory,root)
    if any(b is None for b in batches):raise ValueError('Missing paired evaluation batches.')
    rows=[r for batch in batches for r in batch]
    aggregate=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
    if len(rows)!=_count(config) or len({r['record_id'] for r in rows})!=_count(config) or rows!=aggregate:
        raise ValueError('Aggregate response identities/counts do not reconcile with batches.')
    summary=json.loads((directory/'results/results.json').read_text())
    if summary!=summarize(config,rows):raise ValueError('Saved summary differs from raw-response reconstruction.')
    return summary


def verify(path,output_root):
    root=Path(output_root).resolve();config,_,directory=_load(path,root)
    return _verify(config,directory,root)


def status(path,output_root):
    root=Path(output_root).resolve();config,_,directory=_load(path,root);saved=_saved(config,directory,root)
    summary=_verify(config,directory,root) if (directory/'complete.json').exists() else None
    owner=directory/'writer.json'
    return dict(completed_batches=len(saved) if summary is not None else sum(b is not None for b in saved),total_batches=len(saved),
        completed_responses=summary['response_count'] if summary is not None else sum(len(b) for b in saved if b is not None),total_responses=_count(config),
        measurement_complete=summary is not None,execution_kind=config['execution_kind'],
        budget=budget_status(root),summary=summary,writer=json.loads(owner.read_text()) if owner.exists() else None,
        lifecycle=json.loads((directory/'meta/status.json').read_text()),
        active_reservations=[r for r in reservations(root).values() if r['state']=='active'],
        result_directory=str(directory/'results'),human_review_status='not-applied')


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,_,directory=_load(path,root)
    return _recover_owner(directory,owner_token,review_notes)


def _recover_owner(directory,owner_token,review_notes):
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
    return dict(state='ownership-released',scope='writer only; reservation and sealed responses retained')


def _count(config):return len(config['item_ids'])*len(config['variants'])


def _dependencies(config):
    if config.get('collection_kind')=='conditional-intruders':
        from pilot_eval.stage0_merged_backend import MergedEvaluationDependencies
        return MergedEvaluationDependencies()
    from pilot_eval.stage0_evaluation_backend import EvaluationDependencies
    return EvaluationDependencies()
