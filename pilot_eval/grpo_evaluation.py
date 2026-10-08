"""Held-out GRPO generation with frozen batch seeds and sealed inference units."""
import json
from pathlib import Path

from pilot_eval.grpo_training import verify_training, CHECKPOINTS
from pilot_eval.grpo_preflight import load_frozen, _finite
from pilot_eval.grpo_prepare import load_grpo_prepared, _within
from pilot_eval.grpo_baseline import _score
from pilot_eval.training import file_hash, verified_checkpoints, run_lock
from pilot_eval.workflow import _save_frozen, _hash
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name


def _implementation():
    from pilot_eval import grpo_eval_engine, scoring
    return {key:file_hash(Path(module.__file__)) for key,module in
            [('workflow',__import__(__name__,fromlist=[''])),('engine',grpo_eval_engine),('scorer',scoring)]}


def _source(path, root):
    verify_training(path,root)
    training=json.loads(path.read_text())
    frozen=load_frozen(root/training['frozen_path'],root)
    plan,_,items,_=load_grpo_prepared(root/frozen['source_path'],root)
    if len(items)!=150 or len({r['id'] for r in items})!=150: raise ValueError('expected 150 unique held-out items')
    checkpoints=verified_checkpoints(_within(root,training['run_path']))
    return training,frozen,plan,items,checkpoints


def _policies(settings):
    return [dict(step=step,mode=mode,draws=1 if mode=='greedy' else 8,
                 batch_size=settings['evaluation_batch_size'] if mode=='greedy' else settings['generation_groups'],
                 do_sample=mode=='sampled',temperature=None if mode=='greedy' else 1.,
                 top_p=settings['top_p'] if mode=='sampled' else None,
                 top_k=settings['top_k'] if mode=='sampled' else None,
                 completion_limit=settings['completion_limit'],num_beams=1,use_cache=True)
            for mode,steps in [('greedy',CHECKPOINTS),('sampled',[0,64])] for step in steps]


def _config(path, root, name):
    training,frozen,plan,items,checkpoints=_source(path,root)
    policies=_policies(frozen['settings']); units=[]
    ids=[r['id'] for r in items]
    for policy in policies:
        for start in range(0,150,policy['batch_size']):
            index=start//policy['batch_size']
            units.append(dict(key=f'{policy["mode"]}-{policy["step"]}-batch-{index:04d}',
                policy=policy,ids=ids[start:start+policy['batch_size']],seed=42+index))
    directory=root/'runs/pilot-4'/plan['model'].replace('/','--')/'behaviour'/safe_name(name)
    config=dict(protocol_version='pilot4-behaviour-v1',training_path=str(path.relative_to(root)),
        training_sha256=file_hash(path),training_seal_sha256=file_hash(root/training['run_path']/'complete.json'),
        model=plan['model'],model_revision=plan['model_revision'],tokenizer_revision=plan['tokenizer_revision'],dtype='float32',
        prompt_template=plan['source_prompt_contract'],items=items,seed=42,
        seed_rule='42 + ordered batch index; same schedule across checkpoints; reseed each sealed batch',
        scorer='gsm8k-flexible-v3',strict_scorer='gsm8k-strict',cap_reward=0,
        settings=frozen['settings'],runtime=frozen['runtime'],margin=frozen['review']['margin'],
        checkpoints={str(s):dict(path=str(p.relative_to(root)),seal_sha256=file_hash(p/'complete.json'),
            adapter_sha256=file_hash(p/'adapter_model.safetensors')) for s,p in checkpoints.items()},
        policies=policies,units=units,run_path=str(directory.relative_to(root)),implementation=_implementation())
    return config,plan


def _records(config,unit,outputs):
    policy=unit['policy'];draws=policy['draws'];rows={r['id']:r for r in config['items']}
    if len(outputs)!=len(unit['ids'])*draws: raise ValueError('missing or extra evaluation draws')
    result=[]
    for index,item_id in enumerate(unit['ids']):
        for draw in range(draws):
            output=outputs[index*draws+draw]
            record=_score(rows[item_id],output,draw)
            raw=output.get('generated_token_ids');tokens=output['token_ids'];cap=policy['completion_limit']
            if (not isinstance(raw,list) or any(type(t)is not int or t<0 for t in raw)
                or (record['capped'] and (len(tokens)!=cap or raw!=tokens))
                or (not record['capped'] and (len(tokens)>=cap or raw[:-1]!=tokens or len(raw)!=len(tokens)+1))):
                raise ValueError('raw token/end/cap metadata mismatch')
            result.append(dict(**record,item_id=item_id,step=policy['step'],mode=policy['mode'],
                record_id=f'{policy["mode"]}:{policy["step"]}:{item_id}:draw:{draw}',seed=unit['seed'],
                unit_key=unit['key'],scorer=config['scorer'],adapter_sha256=config['checkpoints'][str(policy['step'])]['adapter_sha256']))
    _finite(result)
    return result


def _read_unit(root, config, unit, config_digest=None):
    directory=_within(root,config['run_path']);shard=directory/'units'/f'{unit["key"]}.json'
    seal=json.loads(shard.with_suffix('.complete.json').read_text())
    if seal!=dict(sha256=file_hash(shard),config_sha256=config_digest or _hash(config),runtime_sha256=file_hash(directory/'meta/runtime.json')):
        raise ValueError('evaluation unit/config/runtime seal mismatch')
    records=json.loads(shard.read_text())
    outputs=[{k:r[k] for k in ['text','token_ids','generated_token_ids','stop_reason']} for r in records]
    if records!=_records(config,unit,outputs): raise ValueError('evaluation draw identity/scorer mismatch')
    return records


def verify_evaluation(path,root,*,marker=None):
    root=Path(root).resolve();path=_within(root,path);config=json.loads(path.read_text())
    expected,_=_config(root/config['training_path'],root,Path(config['run_path']).name)
    if config!=expected: raise ValueError('evaluation source/config/implementation changed')
    directory=_within(root,config['run_path'])
    if json.loads((directory/'config.json').read_text())!=config: raise ValueError('evaluation plan differs from run config')
    if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']: raise ValueError('evaluation runtime changed')
    if marker is None: marker=json.loads((directory/'complete.json').read_text())
    digest=_hash(config)
    required={'config.json','meta/runtime.json','results/responses.jsonl','results/results.json'}
    if not required.issubset(marker['files']) or marker['config_sha256']!=digest: raise ValueError('incomplete evaluation seal')
    for rel,file_digest in marker['files'].items():
        if file_hash(_within(root,directory/rel))!=file_digest: raise ValueError('evaluation result seal changed')
    records=[r for unit in config['units'] for r in _read_unit(root,config,unit,digest)]
    saved=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
    if saved!=records or len(records)!=3150 or len({r['record_id'] for r in records})!=3150:
        raise ValueError('missing/duplicate/inconsistent evaluation records')
    return config,records


def evaluate_grpo(path,root,name,*,dependencies=None):
    root=Path(root).resolve();path=_within(root,path);config,plan=_config(path,root,name)
    directory=root/config['run_path'];target=root/'plans'/safe_name(name)/'grpo.evaluation.json'
    engine=None
    with run_lock(directory):
        _save_frozen(directory/'config.json',config);_save_frozen(target,config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(target.relative_to(root)),
            source_path=config['training_path'],run_path=config['run_path'],responses=3150,
            batch_count=len(config['units']),policies=config['policies']))
        if (directory/'complete.json').exists():
            verify_evaluation(target,root)
            return json.loads((directory/'results/results.json').read_text())
        missing=[];digest=_hash(config)
        for unit in config['units']:
            seal=directory/'units'/f'{unit["key"]}.complete.json'
            if seal.exists(): _read_unit(root,config,unit,digest)
            else: missing.append(unit)
        done=len(config['units'])-len(missing)
        missing_keys={u['key'] for u in missing}
        response_count=sum(len(u['ids'])*u['policy']['draws'] for u in config['units'] if u['key'] not in missing_keys)
        try:
            _write_state(directory,'running',done,len(config['units']))
            print(f'GRPO evaluation resume: {response_count}/3150 responses; {done}/{len(config["units"])} sealed batches',flush=True)
            if missing:
                from pilot_eval.grpo_eval_engine import EvaluationDependencies
                engine=(dependencies or EvaluationDependencies()).load_evaluation(plan,config['settings'],directory)
                runtime=engine.runtime()
                if runtime!=config['runtime']: raise ValueError('evaluation runtime differs from frozen preflight')
                _save_frozen(directory/'meta/runtime.json',runtime)
            by_id={r['id']:r for r in config['items']};selected=None
            for unit in missing:
                step=unit['policy']['step']
                if selected!=step:
                    if not engine.check_integrity()['base_unchanged']: raise ValueError('evaluation changed frozen base')
                    engine.select_checkpoint(root/config['checkpoints'][str(step)]['path'],step);selected=step
                outputs=engine.generate([by_id[i] for i in unit['ids']],unit['policy'],unit['seed'])
                records=_records(config,unit,outputs)
                shard=directory/'units'/f'{unit["key"]}.json';_write_json(shard,records)
                _write_json(shard.with_suffix('.complete.json'),dict(sha256=file_hash(shard),config_sha256=digest,
                            runtime_sha256=file_hash(directory/'meta/runtime.json')))
                done+=1;response_count+=len(records);_write_state(directory,'running',done,len(config['units']))
                print(f'GRPO evaluation: {unit["key"]}; {response_count}/3150 responses; sealed {done}/{len(config["units"])} batches',flush=True)
            if engine is not None and not engine.check_integrity()['base_unchanged']: raise ValueError('evaluation changed frozen base')
            records=[r for unit in config['units'] for r in _read_unit(root,config,unit,digest)]
            destination=directory/'results/responses.jsonl';destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));temporary.replace(destination)
            result=dict(total=len(records),greedy=750,sampled=2400,scorer=config['scorer'],policies=config['policies'])
            _write_json(directory/'results/results.json',result)
            files={str(p.relative_to(directory)):file_hash(p) for p in [directory/'config.json',directory/'meta/runtime.json',
                    destination,directory/'results/results.json']}
            marker=dict(config_sha256=digest,files=files)
            verify_evaluation(target,root,marker=marker)
            _write_json(directory/'complete.json',marker)
            _write_state(directory,'completed',done,len(config['units']))
            return result
        except Exception as error:
            _write_state(directory,'failed',done,len(config['units']),error=str(error));raise
        finally:
            if engine is not None: engine.close()
