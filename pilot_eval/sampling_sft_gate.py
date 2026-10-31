"""Source-bound gate score collection and CPU-only recipe reporting.

Gate scores come from the independent HF FP32 oracle, not the sampler's cached
acceptance score. Scoring is paid/model work; analysis and decisions are not.
"""
import json
import os
import platform
import time
import uuid
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_budget as budget
from pilot_eval import sampling_sft_generation as generation
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval import sampling_sft_pilot as pilot
from pilot_eval import sampling_sft_review as review
from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sampler import scored,diagnostic
from pilot_eval.sampling_sft_sources import inside,verify_parents
from pilot_eval.sft import load_sft,safe_name
from pilot_eval.training import file_hash,run_lock
from pilot_eval.stage0_workflow import _writer
from pilot_eval.workflow import _hash


METHODS=['gold','k0','k2','k5']


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
        ['sampling_sft_gate.py','sampling_sft_gate_math.py','sampling_sft_gate_report.py','sampling_sft_engine.py',
         'sampling_sft_ieee_diagnostic.py']}


def prepare(review_path,output_root,word_policy_path,name):
    root,parent,review_folder=review.load(review_path,output_root);coverage=review.verify(review_path,root)
    _,gen,directory=generation.load(root/parent['generation_plan'],root);safe_name(name)
    word_policy=json.loads(Path(word_policy_path).read_text())
    if (word_policy.get('version')!='unicode-words-v1' or word_policy.get('approved') is not True
            or not all(isinstance(word_policy.get(k),str) and word_policy[k].strip() for k in ['reviewer','notes'])):
        raise ValueError('Explicit reviewed word-normalization contract required; no silent default.')
    audit=source.verify(root/gen['source_plan'],root)['audit']
    _,gold_rows,_=load_sft(root/audit['training']['config_path'],root)
    requests=[]
    for pid,row in enumerate(gold_rows):
        boundary=row['prompt_tokens']
        requests.append(dict(method='gold',problem_id=pid,source_id=row['id'],attempt_id=None,
            prompt_ids=row['input_ids'][:boundary],response_ids=row['input_ids'][boundary:],
            text=row['gold'],gold=row['gold'],eligibility_origin='gold',source='original-loss-bearing-SFT-token-IDs'))
    for key in ['0','2','5']:
        arm=coverage['arms'][key];attempts={r['attempt_id']:r for r in arm['attempts']}
        for disposition in arm['dispositions']:
            if disposition['selected_attempt'] is None:continue
            attempt=attempts[disposition['selected_attempt']];pid=disposition['problem_id'];problem=gen['problems'][pid]
            if attempt['end_state']!='finished':raise ValueError('Gate cannot score an unfinished selected response as eligible.')
            requests.append(dict(method='k'+key,problem_id=pid,source_id=problem['source_id'],attempt_id=attempt['attempt_id'],
                prompt_ids=problem['scorer_prompt_ids'],response_ids=attempt['response_ids'],text=attempt['text'],gold=problem['gold'],
                eligibility_origin=disposition['eligibility_origin'],source='retained-generated-IDs'))
    units=[]
    for pid in range(512):
        rows=sorted([r for r in requests if r['problem_id']==pid],key=lambda r:METHODS.index(r['method']))
        for r in rows:
            if (not r['response_ids'] or r['response_ids'][-1] not in gen['settings']['runtime']['actual_stop_ids']
                    or any(i in gen['settings']['runtime']['actual_stop_ids'] for i in r['response_ids'][:-1])):
                raise ValueError('Gate targets must include exactly one final actual stop token.')
        units.append(dict(problem_id=pid,requests=rows))
    folder=root/json.loads((root/gen['source_plan']).read_text())['run_path']/'gate'/name
    parents={str(inside(root,review_path).relative_to(root)):file_hash(inside(root,review_path)),
        str((review_folder/'complete.json').relative_to(root)):file_hash(review_folder/'complete.json'),
        str((review_folder/'results/coverage.json').relative_to(root)):file_hash(review_folder/'results/coverage.json')}
    config=dict(protocol='sampling-sft-gate-v1',name=name,review_plan=str(inside(root,review_path).relative_to(root)),
        generation_plan=parent['generation_plan'],source_plan=gen['source_plan'],settings={**gen['settings'],'mode':'oracle'},
        problems=gen['problems'],units=units,runtime_hash=gen['runtime_hash'],word_policy=word_policy,
        score_contract='HF-FP32-full-vocabulary-raw-logprobs; scorer prompt; actual stop included',
        bootstrap=dict(resamples=10000,seed=42,unit='whole-problem',interval='percentile95-linear'),
        parents=parents,implementation=implementation(),run_path=str(folder.relative_to(root)),training_authorized=False)
    path=root/'plans'/name/'sampling-sft.gate.json'
    source.frozen(path,config);source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(config)))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);config=json.loads(path.read_text());folder=inside(root,config['run_path'])
    review.verify(root/config['review_plan'],root)
    if (config.get('protocol')!='sampling-sft-gate-v1' or config['implementation']!=implementation()
            or config['runtime_hash']!=_hash(config['settings']['runtime']) or len(config['units'])!=512
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(config)}):
        raise ValueError('Gate plan/source/runtime/code changed; preserve this attempt.')
    verify_parents(root,config['parents'])
    return root,config,folder


def unit(folder,index,config,config_sha256=None):
    directory=folder/'scores/units'/f'{index:03d}'
    if not directory.exists():return None
    seal=json.loads((directory/'complete.json').read_text())
    if (seal['config_sha256']!=(config_sha256 or _hash(config)) or seal['index']!=index
            or seal['request_sha256']!=_hash(config['units'][index])
            or seal['runtime_hash']!=config['runtime_hash']
            or file_hash(inside(folder,seal['env_path']))!=seal['env_sha256']):
        raise ValueError('Gate score unit environment/identity changed.')
    if file_hash(directory/'scores.json')!=seal['scores_sha256']:raise ValueError('Gate score payload changed.')
    values=json.loads((directory/'scores.json').read_text())
    requests=config['units'][index]['requests']
    if len(values)!=len(requests):raise ValueError('Gate scores do not cover declared request cardinality.')
    for value,request in zip(values,requests):scored(value,request)
    return dict(index=index,requests=requests,scores=values,execution_kind=seal['execution_kind'])


def score(path,output_root,job_path=None,*,max_new_units=None,dependencies=None):
    root,config,folder=load(path,output_root);directory=folder/'scores'
    config_sha256=_hash(config)
    if (directory/'complete.json').exists():return verify_scores(config,folder)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<=0):raise ValueError('Bounded unit count must be positive.')
    if dependencies is None:
        _,gen,gen_folder=generation.load(root/config['generation_plan'],root)
        generation.qualification(root,gen)
        receipt=json.loads((gen_folder/'generation.review.json').read_text())
        generation.plan_approval(gen,receipt['review'],file_hash(root/gen['pilot_report']))
        if generation.verify(root/config['generation_plan'],root)['execution_kind']!='real-cuda':
            raise ValueError('Real gate scoring requires real collected arms, not synthetic fixtures.')
        instrument.authorize(root,path,job_path,expected_command='sampling-sft-gate-score')
    with _writer(directory):
        verified=[unit(folder,index,config,config_sha256) for index in range(512)]
        remaining=[i for i,value in enumerate(verified) if value is None]
        if not remaining:return seal_scores(config,folder)
        if dependencies is None and any(value and value['execution_kind']!='real-cuda' for value in verified):
            raise ValueError('Do not combine CPU fixture scores with real evidence.')
        run_id=uuid.uuid4().hex
        pilot.event(directory,'started',run_id=run_id,completed=512-len(remaining),total=512,budget_snapshot=generation.snapshot(root))
        _write_state(directory,'loading-HF-oracle',512-len(remaining),512)
        try:
            if dependencies is None:
                from pilot_eval.sampling_sft_engine import open_engine
                engine,_,runtime=open_engine(config,root)
            else:
                engine,_,runtime=dependencies.open(config,root);runtime={**runtime,'execution_kind':'cpu-fixture'}
            env=directory/f'env.{run_id}.json';source.save(env,runtime)
            completed=512-len(remaining)
            for index in remaining[:max_new_units]:
                request=config['units'][index]
                if any(len(r['prompt_ids'])+len(r['response_ids'])>config['settings']['runtime']['max_model_len'] for r in request['requests']):
                    raise ValueError('Gate context overflow; no truncation allowed.')
                started=time.monotonic();values=None
                for replay in range(2):
                    try:
                        values=engine.score(request['requests'])
                        if not isinstance(values,list) or len(values)!=len(request['requests']):raise ValueError('HF score cardinality changed.')
                        for value,row in zip(values,request['requests']):scored(value,row)
                        break
                    except (RuntimeError,ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
                        pilot.event(directory,'score-engine-error',run_id=run_id,index=index,replay=replay,
                            error=str(exc),raw_outputs=diagnostic(values),request=request)
                        if replay==1:raise ValueError('HF score request failed twice; gate report blocked.') from exc
                partial=directory/'partial'
                if partial.exists():
                    target=directory/'abandoned'/uuid.uuid4().hex;target.parent.mkdir(exist_ok=True);partial.rename(target)
                    pilot.event(directory,'partial-unit-abandoned',index=index)
                partial.mkdir()
                source.save(partial/'scores.json',values)
                source.save(partial/'complete.json',dict(config_sha256=config_sha256,index=index,
                    runtime_hash=config['runtime_hash'],request_sha256=_hash(request),
                    execution_kind=runtime['execution_kind'],env_path=str(env.relative_to(folder)),env_sha256=file_hash(env),
                    scores_sha256=file_hash(partial/'scores.json')))
                target=directory/'units'/f'{index:03d}';target.parent.mkdir(exist_ok=True)
                if target.exists():raise ValueError('Never overwrite a sealed gate score unit.')
                partial.rename(target);completed+=1
                pilot.event(directory,'unit-sealed',index=index,seconds=time.monotonic()-started,
                    answer_count=len(values),budget_snapshot=generation.snapshot(root))
                _write_state(directory,'running',completed,512)
                if completed%32==0 or completed==512:print(f'Gate HF score: {completed}/512 problems sealed.',flush=True)
                if dependencies is not None and hasattr(dependencies,'after_unit'):dependencies.after_unit(index)
            if completed<512:
                _write_state(directory,'partial',completed,512)
                return dict(state='partial',completed=completed,total=512,training_authorized=False)
            return seal_scores(config,folder)
        except BaseException as exc:
            pilot.event(directory,'interrupted',run_id=run_id,error=str(exc),type=type(exc).__name__)
            _write_state(directory,'failed',sum(unit(folder,i,config,config_sha256) is not None for i in range(512)),512,str(exc))
            raise


def seal_scores(config,folder):
    config_sha256=_hash(config)
    values=[unit(folder,i,config,config_sha256) for i in range(512)]
    if any(v is None for v in values):raise ValueError('HF score evidence incomplete.')
    result=dict(state='completed',problem_count=512,answer_count=sum(len(v['scores']) for v in values),
        execution_kind='real-cuda' if all(v['execution_kind']=='real-cuda' for v in values) else 'cpu-fixture',
        runtime_hash=config['runtime_hash'],training_authorized=False)
    source.frozen(folder/'scores/results.json',result)
    source.frozen(folder/'scores/complete.json',dict(config_sha256=config_sha256,
        units={str(i):file_hash(folder/'scores/units'/f'{i:03d}'/'complete.json') for i in range(512)},
        results_sha256=file_hash(folder/'scores/results.json')))
    _write_state(folder/'scores','completed',512,512)
    return result


def verify_scores(config,folder):
    config_sha256=_hash(config)
    seal=json.loads((folder/'scores/complete.json').read_text())
    if (seal['config_sha256']!=config_sha256 or set(seal['units'])!={str(i) for i in range(512)}
            or seal['results_sha256']!=file_hash(folder/'scores/results.json')):raise ValueError('HF score completion changed.')
    for i in range(512):
        if seal['units'][str(i)]!=file_hash(folder/'scores/units'/f'{i:03d}'/'complete.json'):raise ValueError('HF score unit seal changed.')
        unit(folder,i,config,config_sha256)
    return json.loads((folder/'scores/results.json').read_text())


def report(path,output_root):
    root,config,folder=load(path,output_root);scores=verify_scores(config,folder)
    if (folder/'complete.json').exists():return verify(path,root)
    from pilot_eval.sampling_sft_gate_report import build,render
    value,rows=build(root,config,folder,scores)
    source.frozen(folder/'results/report.json',value)
    generation.jsonl(folder/'results/answers.jsonl',rows)
    render(folder,value,rows)
    files={str(p.relative_to(folder)):file_hash(p) for p in (folder/'results').rglob('*') if p.is_file()}
    files.update({str(p.relative_to(folder)):file_hash(p) for p in (folder/'plots').rglob('*') if p.is_file()})
    files['report.html']=file_hash(folder/'report.html')
    source.frozen(folder/'complete.json',dict(config_sha256=_hash(config),scores_sha256=file_hash(folder/'scores/complete.json'),files=files))
    _write_state(folder,'report-completed',1,1)
    return value


def verify(path,output_root):
    root,config,folder=load(path,output_root);scores=verify_scores(config,folder)
    seal=json.loads((folder/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config) or seal['scores_sha256']!=file_hash(folder/'scores/complete.json'):
        raise ValueError('Gate report identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Gate report/row/plot payload changed.')
    from pilot_eval.sampling_sft_gate_report import build
    value,rows=build(root,config,folder,scores)
    if (json.loads((folder/'results/report.json').read_text())!=value
            or generation.read_lines(folder/'results/answers.jsonl')!=rows):raise ValueError('Gate report does not reconcile with raw HF scores.')
    return value


def status(path,output_root):
    _,config,folder=load(path,output_root)
    config_sha256=_hash(config)
    completed=sum(unit(folder,i,config,config_sha256) is not None for i in range(512))
    return dict(scored_problems=completed,total=512,report_complete=(folder/'complete.json').exists(),
        runtime_hash=config['runtime_hash'],training_authorized=False)


def recover_writer(path,output_root,owner_token,review_notes):
    _,_,parent=load(path,output_root);folder=parent/'scores'
    if not owner_token or not review_notes.strip():raise ValueError('Recorded owner token and reviewed notes required.')
    with run_lock(folder):
        marker=folder/'writer.json';owner=json.loads(marker.read_text())
        if owner.get('token')!=owner_token:raise ValueError('Gate score writer token mismatch.')
        if owner.get('host')==platform.node():
            pid=owner.get('pid')
            if type(pid) is not int or pid<=0:raise ValueError('Invalid writer PID.')
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise ValueError('Recorded score writer is still active.')
        pilot.event(folder,'ownership-released',owner=owner,review_notes=review_notes);marker.unlink()
    return dict(state='ownership-released')


def console(value):
    if 'own_cohort' not in value:return value
    return dict(execution_kind=value['execution_kind'],own_cohort=value['own_cohort'],
        common_problem_count=value['common_problem_count'],baseline='not measured',relative_sharpening='not measured',
        training_authorized=False)


def training_forecast(forecast,ledger):
    if not isinstance(forecast,dict) or set(forecast)!={'arms','per_arm','notes'}:
        raise ValueError('Training forecast requires explicit arms, per_arm cost estimates and notes.')
    arms=forecast['arms']
    if not isinstance(arms,list) or any(type(k) is not int for k in arms) or arms not in [[0],[5],[0,5]]:
        raise ValueError('This gate only selects k0/k5; k2 requires separate approval.')
    if set(forecast['per_arm'])!={str(k) for k in arms} or not isinstance(forecast['notes'],str) or not forecast['notes'].strip():
        raise ValueError('Describe the measured training budget and every selected arm.')
    from decimal import Decimal
    total=0;per_arm={}
    for k in arms:
        value=forecast['per_arm'][str(k)]
        if set(value)!={'rate_usd_per_hour','estimated_hours','setup_hours'}:raise ValueError('Unexpected training forecast fields.')
        cost=budget.micros(budget.number(value['rate_usd_per_hour'],True)*
            (budget.number(value['estimated_hours'],True)*Decimal('1.5')+budget.number(value['setup_hours'])))
        per_arm[str(k)]=budget.usd(cost);total+=cost
    existing=budget.micros(ledger['stage_totals']['training']['committed_usd'])
    committed=budget.micros(ledger['committed_usd'])
    return dict(arms=arms,per_arm_reservation_usd=per_arm,reservation_usd=budget.usd(total),forecast=forecast,
        blocked=existing+total>6_000_000 or committed+total>25_000_000,
        review_80_percent_required=ledger['review_required'] or (existing+total)*5>=6_000_000*4 or (committed+total)*5>=25_000_000*4)


def request(path,output_root,forecast_path):
    root,config,folder=load(path,output_root);report=verify(path,root)
    ledger=budget.status(root);forecast=training_forecast(json.loads(Path(forecast_path).read_text()),ledger)
    flags=[f'k{k}:omissions-require-T16' for k,value in report['coverage']['t16_required'].items() if value]
    flags.extend('uncovered-set-difference:'+pair for pair,value in report['coverage']['pairwise_flags'].items() if value['flagged'])
    value=dict(protocol='sampling-sft-training-request-v1',config_sha256=_hash(config),
        report_sha256=file_hash(folder/'results/report.json'),coverage_sha256=config['parents'][
            next(k for k in config['parents'] if k.endswith('/results/coverage.json'))],
        score_evidence_sha256=file_hash(folder/'scores/complete.json'),
        execution_kind=report['execution_kind'],selection=forecast,budget_snapshot=generation.snapshot(root),
        required_acknowledgments=sorted(flags),coverage_blocked=report['coverage']['blocked'],
        pending_readiness=['Ticket08 T15/T16 and exact training bundle','per-job paid approval and reservation'],
        training_seed=42,training_gpu='L4',training_authorized=False,
        deferred_arms=[2],scope='explicit recipe/arm selection request; no training launch, automatic merge or equivalence')
    target=folder/'training-requests'/(_hash(value)+'.json');source.frozen(target,value)
    return dict(request_path=str(target.relative_to(root)),request_sha256=file_hash(target),**value)


def approve(path,output_root,approval_path):
    root,config,folder=load(path,output_root);report=verify(path,root)
    approval=json.loads(Path(approval_path).read_text())
    required={'request_path','request_sha256','report_sha256','budget_sha256','reviewer','notes','gate_approved','acknowledged_coverage_flags','review_80_percent'}
    if not isinstance(approval,dict) or set(approval)!=required:raise ValueError('Approve the exact gate request/report/budget with reviewer and coverage acknowledgments.')
    if (not isinstance(approval['acknowledged_coverage_flags'],list)
            or any(not isinstance(s,str) for s in approval['acknowledged_coverage_flags'])
            or type(approval['review_80_percent']) is not bool):raise ValueError('Coverage flags must be a list; budget review must be boolean.')
    target=inside(root,approval['request_path'])
    if target.parent!=folder/'training-requests' or file_hash(target)!=approval['request_sha256']:
        raise ValueError('Stale or unrelated training request.')
    value=json.loads(target.read_text());ledger=budget.status(root)
    if (value['config_sha256']!=_hash(config) or value['report_sha256']!=file_hash(folder/'results/report.json')
            or approval['report_sha256']!=value['report_sha256'] or approval['budget_sha256']!=ledger['sha256']
            or value['budget_snapshot']['sha256']!=ledger['sha256'] or approval['gate_approved'] is not True
            or not all(isinstance(approval[k],str) and approval[k].strip() for k in ['reviewer','notes'])
            or sorted(approval['acknowledged_coverage_flags'])!=value['required_acknowledgments']):
        raise ValueError('Gate/request/report/budget approval is stale or required coverage flags are not acknowledged.')
    fresh=training_forecast(value['selection']['forecast'],ledger)
    if fresh!=value['selection'] or fresh['blocked'] or value['coverage_blocked'] or report['coverage']['blocked']:
        raise ValueError('Coverage or hard budget cap blocks the training selection.')
    if fresh['review_80_percent_required'] and approval['review_80_percent'] is not True:
        raise ValueError('Explicit80% budget review required before any subsequent paid job.')
    receipt=dict(request_path=str(target.relative_to(root)),request_sha256=file_hash(target),approval=approval,
        report_sha256=value['report_sha256'],arms=fresh['arms'],execution_kind=report['execution_kind'],
        selection_confirmed=True,training_authorized=False,pending_readiness=value['pending_readiness'],
        scope='post-gate selection recorded; Ticket08 readiness and Ticket09 paid execution remain separate')
    destination=folder/'training-selections'/(_hash(receipt)+'.json');source.frozen(destination,receipt)
    return dict(selection_path=str(destination.relative_to(root)),**receipt)
