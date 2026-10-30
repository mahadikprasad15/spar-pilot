"""Saved batch agreement, exact review, measured freeze, existing collector handoff."""
import json
import math
import os
from pathlib import Path
import time
import uuid

from pilot_eval.stage0_batching_sources import read, same_session, sources
from pilot_eval.stage0_budget import reserve, reservations
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state, _software, budget_status
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

# ADR 0014: batched-vs-singleton FP32 logits are judged against the spec's
# pre-registered 1e-3 hard stop, not the merged-loader allowance 3E, which
# measures a different rounding source. The loader gate stays unchanged.
BATCH_GATE=1e-3


def write(path, value):
    """Publish JSON only after flush/readback; completion markers are written last."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    if path.read_bytes()!=data:raise OSError(f'Artifact readback failed: {path}')


def frozen(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'Frozen artifact differs: {path}')
    else:write(path,value)


def implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_batching.py','stage0_batching_sources.py',
        'stage0_batching_backend.py','stage0_evaluation_backend.py','stage0_qualification_math.py','stage0_budget.py']}


def prepare(qualification_path, profile_path, output_root, name, *, batch_size=16):
    root=Path(output_root).resolve();safe_name(name)
    if type(batch_size) is not int or not 2<=batch_size<=16:
        raise ValueError('This bounded agreement workflow supports batch sizes 2–16.')
    print('Verifying diagnostic receipts and actual adapter inputs; no SVD/random-basis scan.',flush=True)
    q,report,qdir,items,tokens=sources(root,qualification_path,profile_path,batch_size)
    sizes=list(dict.fromkeys([batch_size,*([len(items)%batch_size] if len(items)%batch_size else [])]))
    groups=[dict(size=size,ids=q['sequence_ids'][:size]) for size in sizes]
    units=[dict(key=f'numeric-b{g["size"]}-{v}',kind='numeric',variant=v,**g)
           for g in groups for v in q['variants']]
    units += [dict(key=f'greedy-b{g["size"]}-{v}',kind='greedy',variant=v,arm=arm,**g)
              for g in groups for arm,v in [('sft','V1'),('grpo','V5')]]
    config=dict(protocol='stage0-batch-agreement-v1',schema_version=1,name=name,
        run_path=f'runs/stage0/{name}/batch-agreement',implementation=implementation(),
        qualification_path=str(inside(root,qualification_path).relative_to(root)),
        qualification_sha256=file_hash(inside(root,qualification_path)),
        source_diagnostic_sha256=file_hash(qdir/'complete.json'),
        profile_path=str(inside(root,profile_path).relative_to(root)),
        profile_complete_sha256=file_hash(inside(root,profile_path)/'complete.json'),
        batch_size=batch_size,qualified_sizes=sizes,units=units,
        source_runtime=report['runtime'],absolute_gate=report['proposed_absolute_gate'],batch_gate=BATCH_GATE,code_git_commit=_software()['git_commit'],
        fixture=q['fixture'],scope='Fixed16, all10 numeric checks; V1/V5 greedy replay; full and tail batch shapes',
        production_authorized=False,budget_ceiling_units=20)
    path=root/'plans'/name/'stage0.batch.json';directory=inside(root,config['run_path'])
    with _writer(path.parent):
        frozen(path,config);frozen(directory/'meta/manifest.json',dict(config_sha256=_hash(config),software=_software()))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(units))
    return path


def load(path,root):
    path=inside(root,path);config=read(path);directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-batch-agreement-v1' or config['implementation']!=implementation():
        raise ValueError('Batch implementation changed; preserve this evidence.')
    if read(directory/'meta/manifest.json')['config_sha256']!=_hash(config):
        raise ValueError('Batch plan/manifest changed.')
    if (file_hash(inside(root,config['qualification_path']))!=config['qualification_sha256']
            or file_hash(inside(root,config['profile_path'])/'complete.json')!=config['profile_complete_sha256']):
        raise ValueError('Batch source selection changed.')
    q,report,qdir,items,tokens=sources(root,config['qualification_path'],config['profile_path'],config['batch_size'])
    if (file_hash(qdir/'complete.json')!=config['source_diagnostic_sha256']
            or config['absolute_gate']!=report['proposed_absolute_gate'] or config.get('batch_gate')!=BATCH_GATE):
        raise ValueError('Original numerical calibration changed.')
    return config,directory,q,report,qdir,items,tokens


def unit_value(config,directory,unit):
    folder=directory/'units'/unit['key']
    if not (folder/'complete.json').exists():return None
    seal=read(folder/'complete.json')
    if seal!=dict(config_sha256=_hash(config),unit=unit,sha256=file_hash(folder/'result.json')):
        raise ValueError(f'Changed batch unit: {unit["key"]}')
    return read(folder/'result.json')


def numeric_ok(value,gate):
    errors=[e for row in value['examples'] for e in row['errors']]
    if (not errors or any(type(e) not in [int,float] or not math.isfinite(e) or e<0 for e in errors)
            or value['max_absolute_error']!=max(errors)):
        raise ValueError('Malformed numerical error evidence.')
    return max(errors)<=gate


def assess(config,values,load_seconds):
    numeric=[v for u,v in zip(config['units'],values) if u['kind']=='numeric']
    greedy=[v for u,v in zip(config['units'],values) if u['kind']=='greedy']
    eligible=all(numeric_ok(v,config['batch_gate']) for v in numeric) and all(v['acceptable'] for v in greedy)
    step_seconds=max(v['seconds']/max(1,max(len(r['generated_token_ids']) for r in v['responses'])) for v in greedy)
    batch_calls=10*math.ceil(150/config['batch_size'])
    forecast=dict(batch_size=config['batch_size'],batch_sizes=config['qualified_sizes'],
        batch_calls=batch_calls,sequences=1500,variants=10,prompts=150,
        conservative_total_seconds=load_seconds+batch_calls*1024*step_seconds,
        model_load_seconds=load_seconds,worst_measured_seconds_per_batch_step=step_seconds,
        method='one load + 100 percent cap length at worst observed batch-step rate; generation only',
        forecast_is_not_completion_guarantee=True,
        limitations='V1/V5 timing only; late-token slowdown and other edited-condition length tails remain uncertain; KL/write budget separate')
    return dict(eligible_for_review=eligible,production_authorized=False,
        absolute_gate=config['absolute_gate'],batch_gate=config['batch_gate'],max_absolute_error=max(v['max_absolute_error'] for v in numeric),
        variant_count=10,qualified_sizes=config['qualified_sizes'],near_tie_count=sum(not c['exact'] and c['near_tie'] for v in greedy for c in v['checks']),forecast=forecast)


def run(path,output_root,approval_path,*,dependencies=None,max_new_units=None):
    from pilot_eval.stage0_qualification_backend import deadline_guard
    root=Path(output_root).resolve();config,directory,q,report,qdir,items,tokens=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):
        raise ValueError('Invalid maximum new units.')
    with _writer(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        values=[unit_value(config,directory,u) for u in config['units']]
        if max_new_units==0:return dict(state='incomplete',completed=sum(v is not None for v in values))
        from pilot_eval.stage0_batching_backend import BatchDependencies
        deps=dependencies or BatchDependencies();runtime=deps.runtime()
        if (runtime['execution_kind']=='synthetic-cpu-fixture')!=config['fixture'] or not same_session(runtime,config['source_runtime']):
            raise ValueError('Same qualified GPU/software/session is required; fixture evidence cannot qualify CUDA.')
        frozen(directory/'meta/runtime.json',runtime)
        approval=read(approval_path)
        if approval.get('session_id')!=runtime['session_id']:raise ValueError('Budget session differs.')
        active=[r for r in reservations(root).values() if r['state']=='active']
        if active:
            if len(active)!=1 or active[0]['job_sha256']!=_hash(config) or active[0]['approval']!=approval:
                raise ValueError('Settle the prior workload before this batch agreement run.')
            allocation=active[0]
        else:allocation=reserve(root,_hash(config),approval)
        used=0.
        for folder in (directory/'attempts').glob('*'):
            start=read(folder/'start.json')
            if start['reservation_id']==allocation['reservation_id']:
                if not (folder/'timing.json').exists():raise ValueError('Interrupted attempt has unknown time; settle reservation before retry.')
                used+=read(folder/'timing.json')['seconds']
        remaining=approval['max_seconds']-used
        if remaining<=0:raise ValueError('Reserved time exhausted; reconcile usage before another attempt.')
        attempt=directory/'attempts'/uuid.uuid4().hex
        write(attempt/'start.json',dict(reservation_id=allocation['reservation_id']))
        started=time.monotonic();engine=None;created=0
        def deadline():
            if time.monotonic()-started>=remaining:raise TimeoutError('Batch agreement reserved time exhausted.')
        try:
            with deadline_guard(remaining,not config['fixture']):
                engine_config=dict(variants=q['variants'],variant_order=list(q['variants']),
                    weights_dir=q['weights_dir'],model_identity=q['model_identity'],seed=42,
                    prompt_contract=q.get('evaluation_config',{}).get('prompt_template',{}),
                    decoding={**q['decoding'],'use_cache':True},execution_kind=runtime['execution_kind'],
                    batch_gate=config['batch_gate'])
                begin=time.monotonic();engine=deps.engine(engine_config,root)
                load_seconds=time.monotonic()-begin
                selected=[r for r in items if r['id'] in q['sequence_ids']];by_id={r['id']:r for r in selected}
                prepared=engine.prepare_inputs(selected)
                if not config['fixture']:
                    by_token={r['id']:r for r in prepared}
                    for row in tokens:
                        if by_token[row['id']]['input_ids']!=row['input_ids'][:row['prompt_tokens']]:
                            raise ValueError('Prompt tokens differ from the original diagnostic.')
                for i,u in enumerate(config['units']):
                    if values[i] is not None:continue
                    if max_new_units is not None and created>=max_new_units:break
                    deadline();rows=[by_id[key] for key in u['ids']]
                    print(f'Batch agreement {i+1}/{len(values)}: {u["key"]}',flush=True)
                    begin=time.monotonic()
                    if u['kind']=='numeric':
                        value=engine.compare(rows,u['variant'],tokens)
                        passed=numeric_ok(value,config['batch_gate'])
                        engine.measured_error=max(getattr(engine,'measured_error',0.),value['max_absolute_error'])
                    else:
                        responses=engine.generate(rows,u['variant'],42)
                        if len(responses)!=len(rows):raise ValueError('Wrong generation response count.')
                        checks=[]
                        for j,(row,response) in enumerate(zip(rows,responses)):
                            index=q['sequence_ids'].index(row['id'])
                            reference=read(qdir/'units'/f'{u["arm"]}-{index:02d}'/'result.json')['greedy']['reference_token_ids']
                            actual=response['generated_token_ids']
                            expected_stop='eos' if config['fixture'] or (reference and reference[-1]==q['decoding']['eos_token_id']) else 'cap'
                            stop_matches=response['stop_reason']==expected_stop
                            actual_stop='eos' if config['fixture'] or (actual and actual[-1]==q['decoding']['eos_token_id']) else 'cap'
                            stop_consistent=response['stop_reason']==actual_stop
                            exact=actual==reference and stop_matches
                            check=dict(id=row['id'],exact=exact,near_tie=False,stop_matches=stop_matches,
                                stop_consistent=stop_consistent,singleton_stop_reason=expected_stop,batch_stop_reason=response['stop_reason'])
                            if actual!=reference:
                                check.update(engine.divergence(rows,u['variant'],j,reference,responses))
                            if not stop_consistent:check.update(exact=False,near_tie=False)
                            checks.append(check)
                        value=dict(responses=[dict(**r,id=row['id'],prompt=row['prompt'],gold=row['gold']) for row,r in zip(rows,responses)],
                                   checks=checks,acceptable=all(c['exact'] or c['near_tie'] for c in checks))
                        passed=value['acceptable']
                    value['seconds']=time.monotonic()-begin
                    proof=engine.check_integrity()
                    if not proof['base_unchanged'] or (not config['fixture'] and not proof.get('adapters_unchanged')):
                        raise ValueError('Model integrity failed during batch agreement.')
                    folder=directory/'units'/u['key'];write(folder/'result.json',value)
                    write(folder/'complete.json',dict(config_sha256=_hash(config),unit=u,sha256=file_hash(folder/'result.json')))
                    values[i]=value;created+=1
                    _state(directory,'running',sum(v is not None for v in values),len(values),reservation_id=allocation['reservation_id'])
                    if not passed:raise ValueError(f'Batch agreement failed: {u["key"]}; unchanged gate; inspect saved evidence.')
                deadline();proof=engine.check_integrity(full=True);deadline()
                if not proof['base_unchanged'] or (not config['fixture'] and not proof.get('adapters_unchanged')):
                    raise ValueError('Final model integrity failed.')
                if any(v is None for v in values):return dict(state='incomplete',completed=sum(v is not None for v in values))
                result={**assess(config,values,load_seconds), 'runtime':runtime,'integrity':proof,
                        'execution_kind':runtime['execution_kind']}
                if not result['eligible_for_review']:raise ValueError('Saved batch evidence is ineligible.')
                write(directory/'results/agreement.json',result)
                write(directory/'complete.json',dict(config_sha256=_hash(config),files={
                    'results/agreement.json':file_hash(directory/'results/agreement.json'),
                    **{f'units/{u["key"]}/complete.json':file_hash(directory/'units'/u['key']/'complete.json') for u in config['units']}}))
                _state(directory,'awaiting-review',len(values),len(values),reservation_id=allocation['reservation_id'])
                return result
        except BaseException as exc:
            _state(directory,'failed',sum(v is not None for v in values),len(values),error=str(exc),reservation_id=allocation['reservation_id'])
            raise
        finally:
            write(attempt/'timing.json',dict(seconds=time.monotonic()-started))
            if engine is not None:engine.close()


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,*_=load(path,root)
    seal=read(directory/'complete.json')
    if seal['config_sha256']!=_hash(config):raise ValueError('Batch completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError(f'Batch evidence changed: {name}')
    result=read(directory/'results/agreement.json')
    values=[unit_value(config,directory,u) for u in config['units']]
    expected=assess(config,values,result['forecast']['model_load_seconds'])
    if any(result[k]!=v for k,v in expected.items()):raise ValueError('Batch summary reconstruction differs.')
    if result['integrity']['base_unchanged'] is not True or (not config['fixture'] and not result['integrity']['adapters_unchanged']):
        raise ValueError('Missing final integrity evidence.')
    return result


def status(path,output_root):
    root=Path(output_root).resolve();config,directory,*_=load(path,root)
    return dict(completed=sum(unit_value(config,directory,u) is not None for u in config['units']),
        total=len(config['units']),agreement_complete=(directory/'complete.json').exists(),
        production_frozen=(directory/'production-freeze.json').exists(),budget=budget_status(root))


def approve(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,*_=load(path,root);result=verify(path,root)
    review=read(review_path)
    if (set(review)!={'reviewer','notes','agreement_sha256','absolute_gate','batch_gate'}
            or not review['reviewer'].strip() or not review['notes'].strip()
            or review['agreement_sha256']!=file_hash(directory/'complete.json')
            or review['absolute_gate']!=config['absolute_gate'] or review['batch_gate']!=config['batch_gate']
            or not result['eligible_for_review']):
        raise ValueError('Review the exact saved batch evidence and unchanged numerical gate.')
    receipt=dict(config_sha256=_hash(config),review=review,qualified=True,
        real_gpu_qualified=not config['fixture'],runtime=result['runtime'],
        qualified_sizes=config['qualified_sizes'])
    target=directory/'reviews'/_hash(review)/'qualification.json'
    with _writer(directory):frozen(target,receipt)
    return target


def freeze(path,output_root,review_path):
    root=Path(output_root).resolve();config,directory,q,_,_,_,_=load(path,root);result=verify(path,root)
    review=read(review_path)
    commit=_software()['git_commit']
    if not config['fixture']:
        import subprocess
        changed=subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).parent.parent,text=True)
        if changed.strip() or not commit or commit!=config['code_git_commit'] or commit!=result['runtime']['git_commit']:
            raise ValueError('Freeze requires the same clean committed revision used for agreement preparation and execution.')
    if (set(review)!={'reviewer','notes','agreement_sha256','max_units_per_hour','max_seconds','session_id','source_reviewed'}
            or review['agreement_sha256']!=file_hash(directory/'complete.json')
            or review['session_id']!=result['runtime']['session_id']
            or review['max_seconds']<result['forecast']['conservative_total_seconds']):
        raise ValueError('Production review must cover this batch forecast in the measured session.')
    receipts=[]
    for candidate in sorted((directory/'reviews').glob('*/qualification.json')):
        receipt=read(candidate);r=receipt['review']
        if (candidate.parent.name==_hash(r) and receipt['config_sha256']==_hash(config)
                and r['agreement_sha256']==file_hash(directory/'complete.json')
                and r['absolute_gate']==config['absolute_gate'] and r.get('batch_gate')==config['batch_gate']
                and receipt['qualified'] is True):receipts.append(candidate)
    if not receipts:raise ValueError('Record the separate numerical/batch review before production freeze.')
    target=root/'plans'/config['name']/'stage0.execution.json'
    with _writer(directory):
        if target.exists():
            if read(target)['budget_review']!=review:raise ValueError('Frozen budget changed.')
            execution(target,root);return target
        allocation=reserve(root,_hash(dict(config_sha256=_hash(config),budget_review=review)),
            {k:v for k,v in review.items() if k!='agreement_sha256'})
        value=dict(schema_version=1,protocol='stage0-batched-execution-v1',
            batch_plan_path=str(inside(root,path).relative_to(root)),batch_plan_sha256=file_hash(inside(root,path)),
            qualification_path=config['qualification_path'],qualification_sha256=config['qualification_sha256'],
            qualification_receipt=str(receipts[0].relative_to(root)),qualification_receipt_sha256=file_hash(receipts[0]),
            budget_review=review,reservation=allocation,runtime=result['runtime'],variants=q['variants'],code_git_commit=commit,
            cohort_path=q['cohort_path'],cohort_sha256=q['cohort_sha256'],decoding=q['decoding'],
            scorer=q['scorer'],strict_alongside=True,seed=42,absolute_gate=config['absolute_gate'],batch_gate=config['batch_gate'],
            estimate=result['forecast'],qualified_sizes=config['qualified_sizes'],
            real_gpu_authorized=not config['fixture'],measurement_complete=False,
            run_path=f'runs/stage0/{config["name"]}/paired-evaluation')
        frozen(target,value);frozen(directory/'production-freeze.json',dict(path=str(target.relative_to(root)),sha256=file_hash(target)))
    return target


def execution(path,root):
    path=inside(root,path);value=read(path)
    if value.get('protocol')!='stage0-batched-execution-v1':raise ValueError('Unexpected batch execution protocol.')
    plan=inside(root,value['batch_plan_path'])
    if file_hash(plan)!=value['batch_plan_sha256']:raise ValueError('Batch plan changed.')
    config,directory,q,_,qdir,*_=load(plan,root);report=verify(plan,root)
    receipt_path=inside(root,value['qualification_receipt']);receipt=read(receipt_path)
    if (file_hash(receipt_path)!=value['qualification_receipt_sha256']
            or receipt['config_sha256']!=_hash(config) or receipt['qualified'] is not True
            or receipt['review']['agreement_sha256']!=file_hash(directory/'complete.json')
            or receipt['review']['absolute_gate']!=config['absolute_gate']
            or receipt['review'].get('batch_gate')!=config['batch_gate'] or value.get('batch_gate')!=config['batch_gate']
            or value['qualified_sizes']!=config['qualified_sizes']
            or value['qualification_path']!=config['qualification_path']
            or value['qualification_sha256']!=config['qualification_sha256']
            or value['variants']!=q['variants'] or value['runtime']!=report['runtime']
            or value['decoding']!=q['decoding'] or value['cohort_sha256']!=q['cohort_sha256']
            or value['real_gpu_authorized'] is config['fixture']
            or read(directory/'production-freeze.json')!=dict(path=str(path.relative_to(root)),sha256=file_hash(path))):
        raise ValueError('Frozen batch execution does not bind its reviewed evidence.')
    return value,q,qdir
