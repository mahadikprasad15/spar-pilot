"""Saved bounded qualification → explicit review → measured evaluation freeze.

Real inference lives behind a lazy backend. CPU fixtures cannot authorize CUDA.
"""
import json
from pathlib import Path
import time

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_budget import reserve, reservations
from pilot_eval.stage0_edits import _load as load_edits, _verify as verify_edits
from pilot_eval.stage0_qualification_math import compare_logits, assess
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _software, budget_status
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

PILOT4_COHORT_SHA256='ae9d4dfbec0e8121462e52ae8366436a8ad277dca0a652fdbf83b58d11a59b99'


def _implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_qualification.py','stage0_qualification_math.py',
                                           'stage0_budget.py','stage0_qualification_backend.py']}


def prepare(edits_path,cohort_path,output_root,name,*,catalogue_path=None):
    root=Path(output_root).resolve();safe_name(name);conditional={}
    if catalogue_path is not None:
        from pilot_eval.stage0_conditional import binding
        conditional=binding(catalogue_path,root);edits_path=conditional['edits_path']
    edits_path=inside(root,edits_path)
    edits,spectral,audit,_=load_edits(edits_path,root)
    catalogue=verify_edits(edits,inside(root,edits['run_path']),root)
    # The checked-in Pilot 4 evaluation config contains the exact ordered cohort.
    # A caller may select another copy under the artifact root, whose content is
    # bound below; this is not an invitation to substitute a new sample.
    default_cohort = root/'reports/pilot4-results-so-far-2026-10-13/inputs/evaluation-config.json'
    cohort=inside(root,cohort_path or default_cohort)
    text=cohort.read_text()
    parsed=json.loads(text) if text.lstrip().startswith(('[','{')) else [json.loads(x) for x in text.splitlines()]
    cohort_config=parsed if isinstance(parsed,dict) else {}
    rows=cohort_config.get('items',[]) if cohort_config else parsed
    fixture=edits['execution_kind']=='synthetic-cpu-fixture'
    if not fixture:
        if (not cohort_config or cohort_config.get('protocol_version')!='pilot4-behaviour-v1'
                or cohort_config.get('model')!=audit['sources']['sft']['identity']['model']
                or cohort_config.get('model_revision')!=audit['sources']['sft']['identity']['model_revision']
                or cohort_config.get('tokenizer_revision')!=audit['sources']['sft']['identity']['tokenizer_revision']
                or cohort_config.get('dtype')!='float32'
                or cohort_config.get('scorer')!='gsm8k-flexible-v3'
                or file_hash(cohort)!=PILOT4_COHORT_SHA256):
            raise ValueError('Use the pinned Pilot 4 evaluation config and its original held-out cohort.')
    if (len(rows)!=150 or len({r['id'] for r in rows})!=150
            or any(not isinstance(r.get('prompt'),str) or not r['prompt'] or not isinstance(r.get('gold'),str) for r in rows)):
        raise ValueError('Expected 150 unique held-out prompts and full gold answers.')
    tokenizer_files={}
    if not fixture:
        snapshot=inside(root,spectral['weights_dir'])
        for asset_name in ['tokenizer.json','tokenizer_config.json']:
            asset=snapshot/asset_name
            if not asset.is_file():raise FileNotFoundError(f'Pinned tokenizer asset is missing: {asset}')
            tokenizer_files[asset_name]=file_hash(asset)
    selected=rows[:16]
    config=dict(schema_version=1,protocol='stage0-loader-qualification-v1',name=name,
        run_path=f'runs/stage0/{name}/qualification',edits_path=str(edits_path.relative_to(root)),
        edits_sha256=file_hash(edits_path),cohort_path=str(cohort.relative_to(root)),cohort_sha256=file_hash(cohort),
        selection='first 16 in the pinned held-out cohort order',sequence_ids=[r['id'] for r in selected],
        model_identity=audit['sources']['sft']['identity'],weights_dir=spectral['weights_dir'],
        variants=catalogue['variants'],merged_modules=edits['modules'][::2],
        merge_selection='alternating catalogue modules; partial merge leaves remaining adapters active',
        implementation=_implementation(),fixture=fixture,tokenizer_files=tokenizer_files,seed=42,
        decoding=dict(do_sample=False,num_beams=1,max_new_tokens=1024,repetition_penalty=1.,no_repeat_ngram_size=0,
            top_k=0,top_p=1.,num_return_sequences=1,
            eos_token_id=(cohort_config.get('prompt_template',{}).get('decoding',{}).get('eos_token_id') if cohort_config else None),
            pad_token_id=(cohort_config.get('prompt_template',{}).get('decoding',{}).get('pad_token_id') if cohort_config else None)),
        scorer='gsm8k-flexible-v3',strict_alongside=True,hard_error_stop=1e-3,allowance_multiplier=3,
        negative_error_multiplier=1000,total_units=32,budget_ceiling_units=20)
    if cohort_config:
        config['evaluation_config']={key:cohort_config[key] for key in
            ['protocol_version','prompt_template','settings','seed','scorer','strict_scorer','cap_reward']
            if key in cohort_config}
        config['evaluation_config_sha256']=_hash(cohort_config)
    if conditional:
        from pilot_eval.stage0_conditional import implementation
        config.update(conditional);config['total_units']=16*len(config['arms'])
        config['conditional_implementation']=implementation()
        config['merged_modules']={arm:row['adapter_disable_modules'] for arm,row in config['diagnostic_variants'].items()}
        config['merge_selection']='exact catalogue module masks; unhalved saved matrices'
    path=root/'plans'/name/'stage0.qualification.json'
    with _writer(path.parent):
        _save_frozen(path,config)
        directory=root/config['run_path']
        _save_frozen(directory/'meta/manifest.json',dict(config_sha256=_hash(config),software=_software()))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,config['total_units'])
    return path


def _load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-loader-qualification-v1' or config['implementation']!=_implementation():
        raise ValueError('Qualification implementation changed; preserve evidence and use a new name.')
    if json.loads((directory/'meta/manifest.json').read_text())['config_sha256']!=_hash(config):
        raise ValueError('Qualification plan changed.')
    if file_hash(root/config['edits_path'])!=config['edits_sha256'] or file_hash(root/config['cohort_path'])!=config['cohort_sha256']:
        raise ValueError('Qualification source changed.')
    for name,digest in config['tokenizer_files'].items():
        if file_hash(inside(root,config['weights_dir'])/name)!=digest:
            raise ValueError('Pinned tokenizer payload changed.')
    edits,_,_,_=load_edits(root/config['edits_path'],root)
    verify_edits(edits,inside(root,edits['run_path']),root)
    if config.get('qualification_kind')=='conditional-intruders':
        from pilot_eval.stage0_conditional import validate,implementation
        validate(config,root)
        if config.get('conditional_implementation')!=implementation():raise ValueError('Conditional backend implementation changed.')
    return config,directory


def _rows(config,root):
    text=(root/config['cohort_path']).read_text()
    if text.lstrip().startswith(('[','{')):
        parsed=json.loads(text)
        rows=parsed.get('items',[]) if isinstance(parsed,dict) else parsed
    else:
        rows=[json.loads(x) for x in text.splitlines()]
    return [r for r in rows if r['id'] in config['sequence_ids']]


def _unit(directory,key,config,runtime):
    folder=directory/'units'/key
    if not (folder/'complete.json').exists():return None
    saved=json.loads((folder/'complete.json').read_text())
    if saved.get('identity')!=dict(config_sha256=_hash(config),runtime_sha256=_hash(runtime),key=key):
        raise ValueError('Diagnostic unit identity changed.')
    if saved.get('files')!={'result.json':file_hash(folder/'result.json')}:
        raise ValueError('Diagnostic result changed.')
    value=json.loads((folder/'result.json').read_text())
    if config.get('qualification_kind')=='conditional-intruders':
        from pilot_eval.stage0_conditional import validate_observation
        token_sequence=None
        if runtime['execution_kind']=='real-cuda':
            token_sequence=next(r for r in json.loads((directory/'meta/tokenized-inputs.json').read_text())['sequences'] if r['id']==value['id'])
        validate_observation(value,token_sequence)
    return value


def _verify(config,directory):
    seal=json.loads((directory/'complete.json').read_text())
    if seal.get('config_sha256')!=_hash(config) or set(seal.get('files',{}))!={
            'results/diagnostic.json','meta/runtime.json','meta/tokenized-inputs.json'}:
        raise ValueError('Diagnostic completion identity mismatch.')
    for p,digest in seal['files'].items():
        if file_hash(directory/p)!=digest:raise ValueError('Diagnostic payload changed.')
    runtime=json.loads((directory/'meta/runtime.json').read_text())
    token_manifest=json.loads((directory/'meta/tokenized-inputs.json').read_text())
    token_rows=token_manifest.get('sequences',[])
    if [row.get('id') for row in token_rows]!=config['sequence_ids']:
        raise ValueError('Tokenized diagnostic inputs do not match the frozen cohort selection.')
    if runtime['execution_kind']=='real-cuda':
        for row in token_rows:
            identity=[row['input_ids'],row['attention_mask']]
            if (len(row['input_ids'])!=len(row['attention_mask'])
                    or row.get('token_sha256')!=__import__('hashlib').sha256(
                        json.dumps(identity,separators=(',',':')).encode()).hexdigest()):
                raise ValueError('Tokenized input/mask identity changed.')
    rows=[]
    for arm in config.get('arms',['sft','grpo']):
        for i in range(16):
            value=_unit(directory,f'{arm}-{i:02d}',config,runtime)
            if value is None:raise ValueError('Incomplete 16-sequence diagnostic.')
            rows.append(value)
    result=json.loads((directory/'results/diagnostic.json').read_text())
    if result['assessment']!=assess(rows) or result['completed_units']!=config['total_units']:
        raise ValueError('Diagnostic summary disagrees with sealed units.')
    if config.get('qualification_kind')=='conditional-intruders' and runtime['execution_kind']=='real-cuda':
        proof=result.get('final_integrity') or {}
        if proof.get('base_unchanged') is not True or proof.get('adapters_unchanged') is not True:
            raise ValueError('Conditional qualification lacks a successful full final integrity check.')
    return result


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    return _verify(config,directory)


def runtime(path,output_root,*,dependencies=None):
    """Record session identity and artifact readback without loading the model."""
    root=Path(output_root).resolve();config,directory=_load(path,root)
    from pilot_eval.stage0_qualification_backend import QualificationDependencies
    deps=dependencies or _dependencies(config)
    info=deps.runtime()
    storage=deps.preflight(config,root) if hasattr(deps,'preflight') else dict(passed=True,scope='fixture')
    result=dict(runtime=info,storage=storage,budget=budget_status(root),
        active_reservations=[row for row in reservations(root).values() if row['state']=='active'],
        model_loaded=False,model_download_allowed=False)
    with _writer(directory):
        _write_json(directory/'meta/preflight.json',dict(config_sha256=_hash(config),**result))
    return result


def run(path,output_root,approval_path,*,dependencies=None,max_new_units=None):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    if max_new_units is not None and max_new_units<0:raise ValueError('Invalid new-unit limit.')
    from pilot_eval.stage0_qualification_backend import QualificationDependencies, deadline_guard
    deps=dependencies or _dependencies(config)
    with _writer(directory):
        if (directory/'complete.json').exists():return _verify(config,directory)
        runtime=deps.runtime()
        if (runtime['execution_kind']=='synthetic-cpu-fixture') != config['fixture']:
            raise ValueError('Real and synthetic qualification inputs must match their execution kind.')
        if runtime['execution_kind']=='real-cuda':
            if runtime.get('precision')!='float32' or runtime.get('tf32') is not False or runtime.get('autocast') is not False:
                raise ValueError('Qualification requires FP32, TF32 off and autocast off.')
        approval=json.loads(Path(approval_path).read_text())
        if approval.get('session_id')!=runtime['session_id']:raise ValueError('Budget approval session differs from current runtime.')
        _save_frozen(directory/'meta/runtime.json',runtime)
        units=[(arm,i,row) for arm in config.get('arms',['sft','grpo']) for i,row in enumerate(_rows(config,root))]
        completed=[_unit(directory,f'{arm}-{i:02d}',config,runtime) for arm,i,row in units]
        if max_new_units==0:return dict(state='incomplete',completed=sum(x is not None for x in completed),total=config['total_units'])
        if hasattr(deps,'preflight'):
            preflight=deps.preflight(config,root)
            _write_json(directory/'meta/preflight.json',dict(config_sha256=_hash(config),**preflight))
        reservation=reserve(root,_hash(config),approval)
        attempt=directory/'attempts'/reservation['reservation_id']
        _save_frozen(attempt/'reservation.json',reservation)
        started=time.monotonic();engine=None;new=0
        def deadline():
            if time.monotonic()-started>=approval['max_seconds']:raise RuntimeError('Reserved wall-time exhausted; stop and reconcile usage.')
        try:
            # Real backend installs a process deadline, including load and a stuck CUDA call.
            with deadline_guard(approval['max_seconds'],runtime['execution_kind']=='real-cuda'):
                deadline();load_started=time.monotonic();engine=deps.engine(config,root)
                model_load_seconds=time.monotonic()-load_started
                rows_by_id={row['id']:row for row in _rows(config,root)}
                if hasattr(engine,'prepare_inputs'):
                    token_manifest=engine.prepare_inputs([rows_by_id[i] for i in config['sequence_ids']])
                    _save_frozen(directory/'meta/tokenized-inputs.json',token_manifest)
                elif runtime['execution_kind']=='synthetic-cpu-fixture':
                    _save_frozen(directory/'meta/tokenized-inputs.json',dict(scope='synthetic fixture only',
                        sequences=[dict(id=i,prompt_sha256=_hash(rows_by_id[i]['prompt']),gold_sha256=_hash(rows_by_id[i]['gold']))
                                   for i in config['sequence_ids']]))
                for j,(arm,i,row) in enumerate(units):
                    if completed[j] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    deadline();begin=time.monotonic()
                    data=engine.diagnose(row,arm,deadline)
                    if data.get('integrity') is not True:raise ValueError('Unmerged base restoration/integrity failed.')
                    teacher=compare_logits(data['reference'],data['merged'],data['negative'],data['positions'])
                    value=dict(id=row['id'],arm=arm,teacher=teacher,greedy=data['greedy'],
                               seconds=time.monotonic()-begin,greedy_seconds=data.get('greedy_seconds'),
                               generated_tokens=data.get('generated_tokens'),integrity=True)
                    if config.get('qualification_kind')=='conditional-intruders':
                        from pilot_eval.stage0_conditional import validate_observation
                        validate_observation(value)
                    folder=directory/'units'/f'{arm}-{i:02d}'
                    _write_json(folder/'result.json',value)
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),runtime_sha256=_hash(runtime),key=f'{arm}-{i:02d}'),
                        files={'result.json':file_hash(folder/'result.json')}))
                    completed[j]=value;new+=1
                    count=sum(x is not None for x in completed)
                    _state(directory,'running',count,config['total_units'],reservation_id=reservation['reservation_id'])
                    print(f'Stage 0 loader diagnostic: {count}/{config["total_units"]} sealed; {arm}; {row["id"]}.',flush=True)
                    if teacher['max_absolute_error']>1e-3:raise ValueError('Merged-loader error exceeds hard 1e-3 stop; preserved failed evidence.')
                if any(x is None for x in completed):
                    return _state(directory,'incomplete',sum(x is not None for x in completed),config['total_units'],
                                  reservation_id=reservation['reservation_id'])
                final_integrity=None
                if config.get('qualification_kind')=='conditional-intruders' and runtime['execution_kind']=='real-cuda':
                    deadline();final_integrity=engine.check_integrity(full=True);deadline()
                    if not final_integrity['base_unchanged'] or not final_integrity['adapters_unchanged']:
                        raise ValueError('Conditional full base/adapter integrity failed before diagnostic publication.')
                assessment=assess(completed)
                # Conservative extrapolation; approval is still a hard wall-time bound, not a guarantee of completion.
                per_sequence=max(max(r['seconds'],(r.get('greedy_seconds') or 0)*1024/max(1,r.get('generated_tokens') or 1)) for r in completed)
                estimate=dict(workload='greedy paired evaluation only; KL/write workload needs its own measured budget',
                    variants=len(config['variants']),prompts=150,sequences=150*len(config['variants']),conservative_seconds=per_sequence*150*len(config['variants']),
                    model_load_seconds=model_load_seconds,
                    conservative_total_seconds=model_load_seconds+per_sequence*150*len(config['variants']),
                    method='one measured model load plus 150 times variant count times the worse of full diagnostic-unit seconds or 1,024-token greedy extrapolation',
                    storage_bytes_estimate=max(len(json.dumps(r).encode()) for r in completed)*150*len(config['variants']),
                    forecast_is_not_completion_guarantee=True)
                result=dict(**assessment,assessment=assessment,completed_units=config['total_units'],runtime=runtime,
                    execution_kind=runtime['execution_kind'],
                    production_estimate=estimate,reservation_id=reservation['reservation_id'])
                if config.get('qualification_kind')=='conditional-intruders':result['final_integrity']=final_integrity
                _write_json(directory/'results/diagnostic.json',result)
                required=['results/diagnostic.json','meta/runtime.json','meta/tokenized-inputs.json']
                _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in required}))
                _state(directory,'awaiting-review' if assessment['eligible_for_review'] else 'failed',config['total_units'],config['total_units'])
                if not assessment['eligible_for_review']:raise ValueError('Qualification evidence failed; review failed diagnostic.')
                return result
        except (ValueError,RuntimeError,OSError) as exc:
            _state(directory,'failed',sum(x is not None for x in completed),config['total_units'],error=str(exc),reservation_id=reservation['reservation_id'])
            with (directory/'logs/errors.jsonl').open('a') as stream:stream.write(json.dumps(dict(error=str(exc)))+'\n')
            raise
        finally:
            if engine is not None:engine.close()
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,
                newly_sealed_units=new,usage_reconciled=False,reservation_stays_charged=True))


def approve(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory=_load(path,root);result=_verify(config,directory)
    review=json.loads(Path(review_path).read_text())
    required={'reviewer','notes','diagnostic_sha256','absolute_gate'}
    if (set(review)!=required or not review['reviewer'].strip() or not review['notes'].strip()
            or review['diagnostic_sha256']!=file_hash(directory/'complete.json')
            or not result['eligible_for_review'] or review['absolute_gate']!=result['proposed_absolute_gate']):
        raise ValueError('Approve the exact evidence-derived 3E gate, with reviewer and diagnostic identity.')
    receipt=dict(config_sha256=_hash(config),review=review,runtime=result['runtime'],
        qualified=True,real_gpu_qualified=result['execution_kind']=='real-cuda',
        scope='bounded saved unhalved mixed-matrix diagnostic; calibration, not independent validation' if config.get('qualification_kind') else 'bounded partial-merge diagnostic; calibration, not independent validation')
    target=directory/'reviews'/_hash(review)/'qualification.json'
    with _writer(directory):_save_frozen(target,receipt)
    return target


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory=_load(path,root);result=_verify(config,directory)
    review=json.loads(Path(review_path).read_text())
    required={'reviewer','notes','diagnostic_sha256','max_units_per_hour','max_seconds','session_id','source_reviewed'}
    if set(review)!=required or review['diagnostic_sha256']!=file_hash(directory/'complete.json'):
        raise ValueError('Production budget review must be distinct and bind measured diagnostic evidence.')
    receipts=list((directory/'reviews').glob('*/qualification.json'))
    valid=[]
    for p in receipts:
        receipt=json.loads(p.read_text());r=receipt['review']
        if (p.parent.name==_hash(r) and receipt['config_sha256']==_hash(config)
                and r['diagnostic_sha256']==file_hash(directory/'complete.json')
                and r['absolute_gate']==result['proposed_absolute_gate']):valid.append(p)
    if not valid or not result['eligible_for_review']:raise ValueError('Record a valid separate numerical approval first.')
    if (review['session_id']!=result['runtime']['session_id']
            or review['max_seconds']<result['production_estimate']['conservative_total_seconds']):
        raise ValueError('Production approval must cover the measured forecast in the qualified session.')
    target=root/'plans'/config['name']/'stage0.execution.json'
    with _writer(directory):
        if target.exists():
            frozen=json.loads(target.read_text())
            if frozen['budget_review']!=review:raise ValueError('Frozen selection changed; use a new plan.')
            return target
        reservation=reserve(root,_hash(dict(config_sha256=_hash(config),budget_review=review)),
                            {k:v for k,v in review.items() if k!='diagnostic_sha256'})
        frozen=dict(schema_version=1,protocol='stage0-paired-execution-v1',qualification_path=str(inside(root,path).relative_to(root)),
            qualification_sha256=file_hash(inside(root,path)),qualification_receipt=str(valid[0].relative_to(root)),
            qualification_receipt_sha256=file_hash(valid[0]),budget_review=review,reservation=reservation,
            variants=config['variants'],cohort_path=config['cohort_path'],cohort_sha256=config['cohort_sha256'],
            runtime=result['runtime'],decoding=config['decoding'],scorer=config['scorer'],strict_alongside=True,
            seed=42,estimate=result['production_estimate'],absolute_gate=result['proposed_absolute_gate'],
            real_gpu_authorized=result['execution_kind']=='real-cuda',measurement_complete=False,
            run_path=f'runs/stage0/{config["name"]}/paired-evaluation')
        _save_frozen(target,frozen)
        _write_json(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
    return target


def _dependencies(config):
    if config.get('qualification_kind')=='conditional-intruders':
        from pilot_eval.stage0_mixed_qualification_backend import MixedQualificationDependencies
        return MixedQualificationDependencies()
    from pilot_eval.stage0_qualification_backend import QualificationDependencies
    return QualificationDependencies()


def status(path,output_root):
    root=Path(output_root).resolve();config,directory=_load(path,root)
    runtime_path=directory/'meta/runtime.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    if runtime is None and list((directory/'units').glob('*/complete.json')):
        raise ValueError('Sealed qualification units are missing their runtime.')
    completed=0
    if runtime is not None:
        completed=sum(_unit(directory,f'{case}-{i:02d}',config,runtime) is not None
                      for case in config.get('arms',['sft','grpo']) for i in range(16))
    result=_verify(config,directory) if (directory/'complete.json').exists() else None
    return dict(diagnostic_complete=result is not None,completed_units=completed,total_units=config['total_units'],
        eligible_for_review=result['eligible_for_review'] if result else False,qualified=False,
        budget=budget_status(root),writer=json.loads((directory/'writer.json').read_text()) if (directory/'writer.json').exists() else None,
        diagnostic=str(directory/'results/diagnostic.json'))


def recover_writer(path,output_root,owner_token,review_notes):
    root=Path(output_root).resolve();_,directory=_load(path,root)
    from pilot_eval.stage0_evaluation import _recover_owner
    return _recover_owner(directory,owner_token,review_notes)
