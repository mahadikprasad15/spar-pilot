"""Untuned-model dataset scoring for a sealed offline pool and SFT exports."""
import json
import time
import uuid
from pathlib import Path

import numpy as np

from pilot_eval import offline_grpo_pool, sampling_sft_training_data
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_distance_math import check_arrays
from pilot_eval.stage0_evaluation import _reservation, _attempt_timing
from pilot_eval.stage0_qualification_backend import deadline_guard
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer, _state
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


def _answer(prompt, targets):
    if not prompt or not targets or any(type(x) is not int or x<0 for x in prompt+targets):
        raise ValueError('Invalid prompt or target token IDs')
    return dict(answer_id=_hash([prompt,targets]),input_ids=prompt+targets,prompt_tokens=len(prompt),
                target_ids=targets,positions=list(range(len(prompt)-1,len(prompt)+len(targets)-1)),answer_length=len(targets))


def pool_answers(pool,eos):
    answers={};members=[];seen=set()
    for row in pool:
        for draw in row['draws']:
            tokens=draw['generated_token_ids']
            if draw['draw_id'] in seen:raise ValueError('Duplicate pool draw identity')
            seen.add(draw['draw_id'])
            if draw['completion_mask']!=[1]*len(tokens):raise ValueError('Completion mask differs from generated tokens')
            if not tokens or (draw['stop_reason']=='eos' and (tokens[-1]!=eos or eos in tokens[:-1] or draw['capped'])) or (draw['stop_reason']=='cap' and (eos in tokens or not draw['capped'])) or draw['stop_reason'] not in ['eos','cap']:
                raise ValueError('EOS/cap token alignment invalid')
            answer=_answer(row['input_ids'],tokens);answers.setdefault(answer['answer_id'],answer)
            members.append(dict(problem_id=row['id'],answer_id=answer['answer_id'],draw_id=draw['draw_id'],
                                correct=bool(draw['reward_v3']),capped=draw['capped']))
    return list(answers.values()),members


def aggregate(members,scores):
    if not members:return dict(answers=0,questions=0,tokens=0,token_weighted=None,equal_answer=None,equal_question=None)
    per_question={};answer_means=[];total=np.zeros(5)
    for member in members:
        a=scores[member['answer_id']];n=len(a['surprise'])
        sums=np.array([sum(a['surprise']),sum(a['entropy']),sum(a['rank']),sum(a['rank']==1),n],dtype=float)
        total+=sums;answer_means.append(sums[:4]/n)
        per_question.setdefault(member['problem_id'],[]).append(sums[:4]/n)
    names=['nll','entropy','rank','top1']
    def out(values):
        result=dict(zip(names,map(float,values)));result['excess_surprise']=result['nll']-result['entropy'];return result
    return dict(answers=len(members),questions=len(per_question),tokens=int(total[4]),
                token_weighted=out(total[:4]/total[4]),equal_answer=out(np.mean(answer_means,axis=0)),
                equal_question=out(np.mean([np.mean(v,axis=0) for v in per_question.values()],axis=0)))


def _identity():
    from pilot_eval import stage0_distance
    return dict(workflow=file_hash(Path(__file__)),distance=stage0_distance.implementation())


def _legacy(path,root,identity,eos):
    """Audit saved v1 payloads; this does not run or certify historical code."""
    path=inside(root,path);config=json.loads(path.read_text())
    if config.get('protocol')!='sampling-stage0-dataset-v1':raise ValueError('Unsupported historical dataset protocol')
    for name in ['stage0_distance_backend.py','stage0_distance_math.py','stage0_rollout_backend.py']:
        if config.get('implementation',{}).get(name)!=file_hash(Path(__file__).with_name(name)):
            raise ValueError('Historical numerical implementation differs')
    for k in ['model','model_revision','tokenizer_revision','base_sha256']:
        if config['model_identity'].get(k)!=identity.get(k):raise ValueError('Historical scoring model identity differs')
    if (config.get('scored_eos_token_id')!=eos or config.get('mask')!='stored prompt masked; supervised answer tokens through the single final end-of-turn'
        or config.get('rank')!='1 + full-vocabulary logits strictly greater than the target; ties share rank'):
        raise ValueError('Historical token mask or scoring definition differs')
    directory=inside(root,config['run_path']);seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config) or json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Historical dataset configuration seal changed')
    required={'config.json','inputs/answers.jsonl','inputs/memberships.json','meta/final-integrity.json','meta/runtime.json','meta/vocabulary.json'}
    required.update(f"units/{unit['key']}/complete.json" for unit in config['units'])
    if not required.issubset(seal['files']):raise ValueError('Historical dataset completion seal incomplete')
    for name,digest in seal['files'].items():
        if file_hash(inside(directory,name))!=digest:raise ValueError('Historical dataset payload changed')
    for parents in ['parents','data_parents']:
        for name,digest in config.get(parents,{}).items():
            if file_hash(inside(root,name))!=digest:raise ValueError('Historical dataset source changed')
    if not json.loads((directory/'meta/final-integrity.json').read_text()).get('base_unchanged'):
        raise ValueError('Historical dataset base integrity failed')
    vocab=json.loads((directory/'meta/vocabulary.json').read_text())
    if (vocab.get('includes_padded_rows') is not True or vocab.get('rank_and_softmax_same_full_vector') is not True
        or vocab.get('rank_definition')!='1 + count of strictly greater entries in the full FP64-promoted FP32 logit vector used for softmax'):
        raise ValueError('Historical full-vocabulary definition differs')
    answers=[json.loads(line) for line in (directory/'inputs/answers.jsonl').read_text().splitlines()]
    memberships=json.loads((directory/'inputs/memberships.json').read_text())
    if _hash(answers)!=config['answers_sha256'] or _hash(memberships)!=config['memberships_sha256'] or set(memberships)!={'gold','k0','k2','k5'}:
        raise ValueError('Historical answer or membership identity changed')
    byid={}
    for a in answers:
        prompt=a['input_ids'][:a['prompt_tokens']];targets=a['target_ids']
        normalized=_answer(prompt,targets)
        if (a['input_ids']!=normalized['input_ids'] or a['positions']!=normalized['positions'] or a['answer_id']!=normalized['answer_id']
            or not targets or targets[-1]!=eos or eos in targets[:-1] or a['answer_id'] in byid):
            raise ValueError('Historical prompt, target or EOS identity differs')
        byid[a['answer_id']]=normalized
    score={};runtime_hash=file_hash(directory/'meta/runtime.json')
    for unit in config['units']:
        folder=directory/'units'/unit['key'];marker=json.loads((folder/'complete.json').read_text())
        if (marker['config_sha256']!=_hash(config) or marker['unit']!=unit or marker['runtime_sha256']!=runtime_hash
            or marker['sha256']!=file_hash(folder/'tokens.npz') or marker['integrity_sha256']!=file_hash(folder/'integrity.json')
            or not json.loads((folder/'integrity.json').read_text()).get('base_unchanged')):
            raise ValueError('Historical score unit seal changed')
        with np.load(folder/'tokens.npz',allow_pickle=False) as arr:
            if set(arr.files)!={f'{k}_{i}' for i in range(len(unit['ids'])) for k in ['surprise','entropy','rank']}:
                raise ValueError('Historical per-token score schema changed')
            for i,identifier in enumerate(unit['ids']):
                if identifier in score or identifier not in byid:raise ValueError('Historical score identity duplicated or missing')
                values={k:arr[f'{k}_{i}'].copy() for k in ['surprise','entropy','rank']}
                check_arrays(values,byid[identifier])
                if np.any(values['rank']>vocab['logit_vocabulary_size']):raise ValueError('Historical rank outside vocabulary')
                score[identifier]=values
    if set(score)!=set(byid):raise ValueError('Historical scores incomplete')
    for rows in memberships.values():
        if len({r['problem_id'] for r in rows})!=len(rows) or any(r['answer_id'] not in byid for r in rows):
            raise ValueError('Historical memberships duplicated or missing')
    return byid,memberships,score


def _inputs(pool_path,sft_path,root,comparison_root=None):
    comparison_root=Path(comparison_root or root).resolve()
    windows=offline_grpo_pool.load_pool(pool_path,root)
    pool_plan,pool_dir,training=offline_grpo_pool._plan(pool_path,root)
    receipt=json.loads((pool_dir/'handoff.json').read_text());eos=receipt['metadata']['eos_token_id']
    flat=[q for window in windows for q in window]
    answers,pool_members=pool_answers(flat,eos)
    if len(flat)!=512 or len(pool_members)!=4096:raise ValueError('Full 512 by 8 pool required')
    sft_value=json.loads(inside(comparison_root,sft_path).read_text())
    if sft_value.get('protocol')=='sampling-stage0-dataset-v1':
        identity=json.loads(inside(root,receipt['r0_path']).read_text())['model_identity']
        old_answers,old_members,_=_legacy(sft_path,comparison_root,identity,eos)
        prompts={q['id']:q['input_ids'] for q in flat}
        answers_by_id={a['answer_id']:a for a in answers}
        for rows in old_members.values():
            for row in rows:
                a=old_answers[row['answer_id']]
                if row['problem_id'] not in prompts or a['input_ids'][:a['prompt_tokens']]!=prompts[row['problem_id']]:
                    raise ValueError('Historical prompt or cohort differs')
                answers_by_id.setdefault(a['answer_id'],a)
        return list(answers_by_id.values()),dict(pool=pool_members,**old_members),receipt
    _,sft_config,_=sampling_sft_training_data.load(sft_path,comparison_root)
    sampling_sft_training_data.verify(sft_path,comparison_root)
    gold,arms=sampling_sft_training_data.expected(comparison_root,sft_config)
    if gold['model']!=pool_plan['source_identity']['model'] or gold['model_revision']!=pool_plan['source_identity']['model_revision'] or gold['tokenizer_revision']!=pool_plan['source_identity']['tokenizer_revision']:
        raise ValueError('SFT source scoring model identity differs')
    gold_rows={r['id']:r for r in training};answers_by_id={a['answer_id']:a for a in answers};members={'pool':pool_members}
    datasets={'gold':training,**{f'k{k}':arms[k]['rows'] for k in ['0','2','5']}}
    for name,rows in datasets.items():
        members[name]=[];seen=set()
        for row in rows:
            identifier=row['id'];n=row['prompt_tokens'];tokens=row['input_ids']
            if identifier in seen or identifier not in gold_rows:raise ValueError('Duplicate or foreign SFT question')
            seen.add(identifier)
            prompt=tokens[:n];targets=tokens[n:]
            if prompt!=gold_rows[identifier]['input_ids'][:gold_rows[identifier]['prompt_tokens']] or row['labels']!=[-100]*n+targets or not targets or targets[-1]!=eos or eos in targets[:-1]:
                raise ValueError('SFT prompt, mask or EOS differs')
            answer=_answer(prompt,targets);answers_by_id.setdefault(answer['answer_id'],answer)
            members[name].append(dict(problem_id=identifier,answer_id=answer['answer_id']))
    return list(answers_by_id.values()),members,receipt


def prepare(pool_path,sft_path,sampler_path,runtime_path,output_root,name,*,comparison_root=None,context_chunk=16,workspace_bytes=536870912):
    root=Path(output_root).resolve();safe_name(name)
    comparison_root=Path(comparison_root or root).resolve()
    pool_path,sampler_path,runtime_path=[inside(root,p) for p in [pool_path,sampler_path,runtime_path]]
    sft_path=inside(comparison_root,sft_path)
    answers,members,receipt=_inputs(pool_path,sft_path,root,comparison_root)
    sampler=json.loads(sampler_path.read_text());runtime=json.loads(runtime_path.read_text())
    if sampler.get('protocol') not in ['offline-grpo-fresh-r0-v1','stage0-rollouts-v1']:
        raise ValueError('Verified R0 scoring-model source required')
    offline_grpo_pool._verified_r0(sampler_path,root)
    if receipt['r0_sha256']!=file_hash(sampler_path):raise ValueError('Scoring source does not match the pool source')
    if type(context_chunk) is not int or context_chunk<1 or type(workspace_bytes) is not int or workspace_bytes<1:raise ValueError('Positive scoring workspace and chunk required')
    directory=root/'runs/offline-grpo'/name/'dataset';path=root/'plans'/name/'offline.dataset.json'
    config=dict(protocol='offline-grpo-dataset-v1',run_path=str(directory.relative_to(root)),
        pool_path=str(pool_path.relative_to(root)),pool_sha256=file_hash(pool_path),sft_root=str(comparison_root),sft_path=str(sft_path.relative_to(comparison_root)),sft_sha256=file_hash(sft_path),
        sampler_path=str(sampler_path.relative_to(root)),sampler_sha256=file_hash(sampler_path),runtime_path=str(runtime_path.relative_to(root)),runtime_sha256=file_hash(runtime_path),runtime=runtime,
        model_identity=sampler['model_identity'],weights_dir=sampler['weights_dir'],execution_kind=sampler['execution_kind'],
        context_chunk=context_chunk,workspace_bytes=workspace_bytes,answers_sha256=_hash(answers),memberships_sha256=_hash(members),
        mask='prompt excluded, observed EOS included, cap has no synthetic EOS',statistics='FP32 full vocabulary, FP64 log softmax; rank strictly greater',
        eos_token_id=receipt['metadata']['eos_token_id'],implementation=_identity())
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _save_frozen(directory/'inputs/answers.json',answers);_save_frozen(directory/'inputs/memberships.json',members)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),answers=len(answers),memberships={k:len(v) for k,v in members.items()}))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(answers))
    return path


def load(path,root):
    path=inside(root,path);config=json.loads(path.read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='offline-grpo-dataset-v1' or config['implementation']!=_identity() or json.loads((directory/'config.json').read_text())!=config:
        raise ValueError('Dataset configuration or implementation changed')
    comparison_root=Path(config['sft_root']).resolve()
    for key in ['pool','sft','sampler','runtime']:
        source_root=comparison_root if key=='sft' else root
        if file_hash(inside(source_root,config[key+'_path']))!=config[key+'_sha256']:raise ValueError('Dataset source metadata changed')
    offline_grpo_pool._verified_r0(root/config['sampler_path'],root)
    answers,members,receipt=_inputs(root/config['pool_path'],comparison_root/config['sft_path'],root,comparison_root)
    if _hash(answers)!=config['answers_sha256'] or _hash(members)!=config['memberships_sha256'] or answers!=json.loads((directory/'inputs/answers.json').read_text()) or members!=json.loads((directory/'inputs/memberships.json').read_text()):
        raise ValueError('Dataset token or membership identity changed')
    if json.loads((root/config['runtime_path']).read_text())!=config['runtime']:raise ValueError('Scoring runtime identity changed')
    return config,directory,answers,members


def _read_unit(config,directory,answer,attempt_cache=None):
    if attempt_cache is None:attempt_cache={}
    folder=directory/'units'/answer['answer_id'];seal_path=folder/'complete.json'
    if not seal_path.exists():return None
    seal=json.loads(seal_path.read_text())
    if seal.get('identity')!=dict(config_sha256=_hash(config),answer_sha256=_hash(answer)) or set(seal['files'])!={'tokens.npz','metadata.json'}:
        raise ValueError('Dataset score unit identity changed')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Dataset score payload changed')
    metadata=json.loads((folder/'metadata.json').read_text());attempt=inside(directory/'attempts',metadata['attempt_id'])
    if metadata['integrity'].get('base_unchanged') is not True:
        raise ValueError('Scoring attempt integrity changed')
    if metadata['attempt_id'] not in attempt_cache:
        final=attempt/'final-integrity.json'
        if not final.exists() or not json.loads(final.read_text()).get('base_unchanged'):
            raise ValueError('Scoring attempt lacks successful full base integrity proof')
        timing=_attempt_timing(attempt)
        if timing.get('final_integrity_sha256')!=file_hash(final):raise ValueError('Scoring attempt integrity timing changed')
        attempt_cache[metadata['attempt_id']]=file_hash(attempt/'start.json')
    if attempt_cache[metadata['attempt_id']]!=metadata['attempt_start_sha256']:
        raise ValueError('Scoring attempt identity changed')
    with np.load(folder/'tokens.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    arrays=check_arrays(arrays,answer)
    vocabulary=json.loads((directory/'meta/vocabulary.json').read_text())
    if metadata['vocabulary_sha256']!=file_hash(directory/'meta/vocabulary.json') or np.any(arrays['rank']>vocabulary['logit_vocabulary_size']):
        raise ValueError('Score vocabulary identity changed')
    return arrays


def run(path,output_root,*,dependencies=None,approval_path=None,max_new_units=None,progress=True):
    from pilot_eval.stage0_distance_backend import DistanceDependencies
    root=Path(output_root).resolve();config,directory,answers,_=load(path,root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units<0):raise ValueError('Invalid new-unit bound')
    with _writer(directory):
        if (directory/'measurement.complete.json').exists():return verify(path,root)
        cache={}
        saved=[_read_unit(config,directory,a,cache) for a in answers]
        if max_new_units==0:return dict(complete=False,completed=sum(v is not None for v in saved),total=len(answers))
        historical={}
        comparison_root=Path(config['sft_root']).resolve()
        source=json.loads(inside(comparison_root,config['sft_path']).read_text())
        if source.get('protocol')=='sampling-stage0-dataset-v1':
            _,_,historical=_legacy(comparison_root/config['sft_path'],comparison_root,config['model_identity'],config['eos_token_id'])
        deps=dependencies or DistanceDependencies();runtime=deps.runtime()
        if runtime!=config['runtime'] or runtime['execution_kind']!=config['execution_kind']:raise ValueError('Scoring runtime differs from approved session')
        execution={'reservation':{'reservation_id':None,'job_sha256':_hash(config)}}
        reservation,remaining=_reservation(config,execution,directory,root,runtime,approval_path)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],runtime_sha256=_hash(runtime)))
        started=time.monotonic();engine=None;new=0
        try:
            with deadline_guard(remaining,config['execution_kind']=='real-cuda'):
                if hasattr(deps,'preflight'):deps.preflight(config,root)
                _save_frozen(directory/'meta/runtime.json',runtime)
                engine=deps.engine(config,root)
                vocabulary=engine.vocabulary()
                if (type(vocabulary.get('logit_vocabulary_size')) is not int or vocabulary['logit_vocabulary_size']<1
                    or type(vocabulary.get('tokenizer_vocabulary_size')) is not int or vocabulary['tokenizer_vocabulary_size']<1
                    or vocabulary.get('padded_output_rows')!=max(0,vocabulary['logit_vocabulary_size']-vocabulary['tokenizer_vocabulary_size'])
                    or vocabulary.get('includes_padded_rows') is not True or vocabulary.get('rank_and_softmax_same_full_vector') is not True):
                    raise ValueError('Full scoring vocabulary identity required')
                if historical:
                    old_dir=inside(comparison_root,source['run_path'])
                    if json.loads((old_dir/'meta/vocabulary.json').read_text())!=vocabulary:
                        raise ValueError('Historical and current scoring vocabulary differ')
                _save_frozen(directory/'meta/vocabulary.json',vocabulary)
                for index,answer in enumerate(answers):
                    if saved[index] is not None:continue
                    if max_new_units is not None and new>=max_new_units:break
                    if time.monotonic()-started>=remaining:raise TimeoutError('Scoring allocation exhausted')
                    arrays=check_arrays(historical[answer['answer_id']] if answer['answer_id'] in historical else engine.score(answer),answer)
                    if np.any(arrays['rank']>vocabulary['logit_vocabulary_size']) or any(t>=vocabulary['logit_vocabulary_size'] for t in answer['target_ids']):
                        raise ValueError('Token or rank outside scoring vocabulary')
                    integrity=engine.check_integrity()
                    if not integrity.get('base_unchanged'):raise ValueError('Base integrity failed')
                    folder=directory/'units'/answer['answer_id'];folder.mkdir(parents=True,exist_ok=True)
                    tmp=folder/'tokens.tmp.npz';np.savez_compressed(tmp,**arrays);tmp.replace(folder/'tokens.npz')
                    _write_json(folder/'metadata.json',dict(attempt_id=attempt.name,attempt_start_sha256=file_hash(attempt/'start.json'),integrity=integrity,vocabulary_sha256=file_hash(directory/'meta/vocabulary.json'),origin='historical-verified' if answer['answer_id'] in historical else 'fresh'))
                    _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),answer_sha256=_hash(answer)),files={n:file_hash(folder/n) for n in ['tokens.npz','metadata.json']}))
                    saved[index]=arrays;new+=1
                    done=sum(v is not None for v in saved);_state(directory,'running',done,len(answers))
                    if progress:print(f'Offline dataset: {done}/{len(answers)} unique answers sealed',flush=True)
                integrity=engine.check_integrity(full=True);_write_json(attempt/'final-integrity.json',integrity)
                if not integrity.get('base_unchanged'):raise ValueError('Final base integrity failed')
                if any(v is None for v in saved):return _state(directory,'incomplete',sum(v is not None for v in saved),len(answers))
                _save_frozen(directory/'measurement.complete.json',dict(config_sha256=_hash(config),
                    units={a['answer_id']:file_hash(directory/'units'/a['answer_id']/'complete.json') for a in answers},
                    final_attempt=attempt.name,final_sha256=file_hash(attempt/'final-integrity.json')))
                _state(directory,'complete',len(answers),len(answers))
                return dict(complete=True,unique_answers=len(answers),execution_kind=config['execution_kind'])
        except (Exception,KeyboardInterrupt) as exc:
            if engine is not None and 'integrity' not in str(exc).lower() and time.monotonic()-started<remaining:
                try:
                    with deadline_guard(remaining-(time.monotonic()-started),config['execution_kind']=='real-cuda'):
                        _write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except Exception:pass
            _state(directory,'failed',sum(v is not None for v in saved),len(answers),error=str(exc));raise
        finally:
            if engine:engine.close()
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,newly_sealed_groups=new,reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None,usage_reconciled=False,reservation_stays_charged=True))
            _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def verify(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,members=load(path,root)
    seal=json.loads((directory/'measurement.complete.json').read_text())
    if seal['config_sha256']!=_hash(config) or set(seal['units'])!={a['answer_id'] for a in answers}:raise ValueError('Dataset measurement seal incomplete')
    final=inside(directory/'attempts',seal['final_attempt'])/'final-integrity.json'
    if file_hash(final)!=seal['final_sha256'] or not json.loads(final.read_text()).get('base_unchanged'):raise ValueError('Dataset final integrity changed')
    if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']:raise ValueError('Dataset runtime changed')
    cache={}
    comparison_root=Path(config['sft_root']).resolve()
    source=json.loads(inside(comparison_root,config['sft_path']).read_text())
    historical={}
    if source.get('protocol')=='sampling-stage0-dataset-v1':
        _,_,historical=_legacy(comparison_root/config['sft_path'],comparison_root,config['model_identity'],config['eos_token_id'])
    for answer in answers:
        folder=directory/'units'/answer['answer_id']
        if file_hash(folder/'complete.json')!=seal['units'][answer['answer_id']]:
            raise ValueError('Dataset score seal changed')
        arrays=_read_unit(config,directory,answer,cache)
        if arrays is None:raise ValueError('Dataset score missing')
        metadata=json.loads((folder/'metadata.json').read_text())
        if metadata['origin']=='historical-verified':
            old=historical.get(answer['answer_id'])
            if old is None or any(not np.array_equal(arrays[k],old[k]) for k in arrays):
                raise ValueError('Reused scores differ from verified historical payload')
        elif metadata['origin']!='fresh':raise ValueError('Unknown score origin')
    report_seal=directory/'report.complete.json'
    if report_seal.exists():
        marker=json.loads(report_seal.read_text())
        if marker['config_sha256']!=_hash(config) or marker['measurement_sha256']!=file_hash(directory/'measurement.complete.json'):
            raise ValueError('Dataset report identity changed')
        for name,digest in marker['files'].items():
            if file_hash(inside(directory,name))!=digest:raise ValueError('Dataset report payload changed')
    return dict(complete=True,unique_answers=len(answers),execution_kind=config['execution_kind'])


def report(path,output_root):
    root=Path(output_root).resolve();config,directory,answers,members=load(path,root);verify(path,root)
    cache={}
    scores={a['answer_id']:_read_unit(config,directory,a,cache) for a in answers}
    common=set.intersection(*[{m['problem_id'] for m in members[name]} for name in ['pool','gold','k0','k2','k5']])
    selections={**members,'pool_correct':[m for m in members['pool'] if m['correct']],
        'pool_incorrect':[m for m in members['pool'] if not m['correct']],
        'pool_capped':[m for m in members['pool'] if m['capped']],
        'pool_uncapped':[m for m in members['pool'] if not m['capped']]}
    origins={}
    for answer in answers:
        meta=json.loads((directory/'units'/answer['answer_id']/'metadata.json').read_text())
        origins[meta['origin']]=origins.get(meta['origin'],0)+1
    source=json.loads(inside(Path(config['sft_root']),config['sft_path']).read_text())
    old_runtime=None
    if source.get('protocol')=='sampling-stage0-dataset-v1':
        old_runtime=json.loads((inside(Path(config['sft_root']),source['run_path'])/'meta/runtime.json').read_text())
    result=dict(protocol='offline-grpo-dataset-report-v1',execution_kind=config['execution_kind'],
        scoring_runtime=config['runtime'],historical_scoring_runtime=old_runtime,unique_answer_origins=origins,
        full={name:aggregate(rows,scores) for name,rows in selections.items()},
        paired_common={name:aggregate([m for m in members[name] if m['problem_id'] in common],scores) for name in ['pool','gold','k0','k2','k5']},
        common_question_ids=sorted(common),note='Excess surprise is descriptive NLL minus entropy, not KL; no production acceptance gate.')
    with _writer(directory):
        _save_frozen(directory/'results/results.json',result)
        lines=['# Offline GRPO dataset scoring','','Excess surprise is descriptive NLL minus entropy, not KL.','',
               f'Common paired questions: {len(common)}. Unique score origins: {origins}.','','| Cohort | Dataset | Questions | Answers | Tokens | Token NLL | Equal-answer NLL | Equal-question NLL |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for cohort in ['full','paired_common']:
            for name,row in result[cohort].items():
                if row['answers']:
                    lines.append(f"| {cohort} | {name} | {row['questions']} | {row['answers']} | {row['tokens']} | {row['token_weighted']['nll']:.4f} | {row['equal_answer']['nll']:.4f} | {row['equal_question']['nll']:.4f} |")
        content='\n'.join(lines)+'\n';report_path=directory/'results/report.md'
        if report_path.exists() and report_path.read_text()!=content:raise ValueError('Dataset report text changed')
        report_path.write_text(content)
        files={str(p.relative_to(directory)):file_hash(p) for p in [directory/'results/results.json',report_path]}
        _save_frozen(directory/'report.complete.json',dict(config_sha256=_hash(config),measurement_sha256=file_hash(directory/'measurement.complete.json'),files=files))
    return result
