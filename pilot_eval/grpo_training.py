"""Checkpoint-boundary GRPO orchestration; accepted history is sealed with state."""
import json
import math
import time
from pathlib import Path

from pilot_eval.grpo_preflight import load_frozen, _finite
from pilot_eval.grpo_prepare import load_grpo_prepared, _within
from pilot_eval.grpo_baseline import _percentile
from pilot_eval.training import run_lock, file_hash, seal_checkpoint, verified_checkpoints
from pilot_eval.run import _write_json, _write_state
from pilot_eval.workflow import _save_frozen, _hash
from pilot_eval.sft import safe_name

CHECKPOINTS=[0,8,16,32,64]


def _implementation():
    from pilot_eval import grpo_train_engine, grpo_algorithm, scoring
    return {k:file_hash(Path(v.__file__)) for k,v in [('workflow',__import__(__name__,fromlist=[''])),
        ('engine',grpo_train_engine),('algorithm',grpo_algorithm),('scorer',scoring)]}


def _history(root, checkpoint, groups):
    history=json.loads((checkpoint/'history.json').read_text())
    if len(history)!=int(checkpoint.name.split('-')[-1]):raise ValueError('checkpoint accepted-history count mismatch')
    records=[];logs=[]
    for step,unit in enumerate(history,1):
        path=_within(root,unit['path'])
        if file_hash(path)!=unit['sha256']:raise ValueError('accepted training history changed')
        value=json.loads(path.read_text());_finite(value)
        if value['step']!=step or value['prompt_ids']!=groups[step-1]:raise ValueError('accepted group order mismatch')
        expected=[f'{q}:draw:{d}' for q in groups[step-1] for d in range(8)]
        if [r['draw_id'] for r in value['responses']]!=expected or any(r['step']!=step for r in value['responses']):raise ValueError('accepted rollout IDs mismatch')
        records.extend(value['responses']);logs.append(value['log'])
    order=json.loads((checkpoint/'data-order.json').read_text())
    if order!=dict(optimizer_groups=groups,next_step=len(history)+1):raise ValueError('checkpoint data-order state mismatch')
    return history,records,logs


def _monitor_events(logs, policies, definition):
    now=logs[-1];events=[]
    values={k:now[k] for k in ['cap_fraction','dead_group_fraction']}
    previous=None
    if definition=='step-lag-8' and len(logs)>8:previous=logs[-9]['mean_tokens'];current=now['mean_tokens']
    else:current=now['mean_tokens']
    values['length_ratio_change_8_steps']=None if previous is None or previous==0 else abs(current/previous-1)
    now['length_change']=dict(value=values['length_ratio_change_8_steps'],definition=definition,
                            defined=values['length_ratio_change_8_steps'] is not None,
                            reason=None if values['length_ratio_change_8_steps'] is not None else ('zero_previous_mean' if previous==0 else 'insufficient_history'))
    for key,value in values.items():
        if value is not None and value>policies[key]['threshold']:
            events.append(dict(measurement=key,value=value,**policies[key]))
    return events


def _seal(directory, step, engine, history, groups):
    destination=directory/'checkpoints'/f'checkpoint-{step}'
    engine.save_checkpoint(destination,step)
    _write_json(destination/'history.json',history)
    _write_json(destination/'data-order.json',dict(optimizer_groups=groups,next_step=step+1))
    seal_checkpoint(destination,step)
    return destination


def verify_training(path, root):
    root=Path(root).resolve();path=_within(root,path);config=json.loads(path.read_text())
    if config['implementation']!=_implementation():raise ValueError('training implementation changed')
    frozen=root/config['frozen_path']
    if file_hash(frozen)!=config['frozen_sha256']:raise ValueError('frozen training protocol changed')
    load_frozen(frozen,root)
    directory=_within(root,config['run_path']);marker=json.loads((directory/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config):raise ValueError('training config changed')
    for rel,digest in marker['files'].items():
        if file_hash(_within(root,directory/rel))!=digest:raise ValueError('training result seal changed')
    checkpoints=verified_checkpoints(directory)
    if sorted(checkpoints)!=CHECKPOINTS:raise ValueError('complete training requires all five checkpoints')
    _,records,logs=_history(root,checkpoints[64],config['optimizer_groups'])
    if len(records)!=4096 or len(logs)!=64 or len({r['draw_id'] for r in records})!=4096:raise ValueError('incomplete scientific training history')
    result=json.loads((directory/'results/results.json').read_text())
    if not result['base_integrity']['base_unchanged']:raise ValueError('frozen base integrity failed')
    return result


def run_grpo(config_path,root,name,*,length_change_definition,dependencies=None):
    if length_change_definition not in ['step-lag-8']:raise ValueError('explicit length-monitor definition required')
    root=Path(root).resolve();path=_within(root,config_path);frozen=load_frozen(path,root)
    plan,training,*_=load_grpo_prepared(root/frozen['source_path'],root)
    groups=frozen['optimizer_groups'];ids=[q for g in groups for q in g]
    if len(groups)!=64 or any(len(g)!=8 for g in groups) or len(ids)!=512 or len(set(ids))!=512:raise ValueError('exact 512-prompt order required')
    directory=root/'runs/pilot-4'/plan['model'].replace('/','--')/'gsm8k/train-512-seed-42/rank1-float32'/safe_name(name)
    config=dict(protocol_version='pilot4-training-v1',frozen_path=str(path.relative_to(root)),frozen_sha256=file_hash(path),
        model=plan['model'],model_revision=plan['model_revision'],tokenizer_revision=plan['tokenizer_revision'],
        adapter=plan['adapter'],seed=plan['seed'],dtype=plan['dtype'],prompt_template=plan['source_prompt_contract'],
        settings=frozen['settings'],optimizer=frozen['optimizer'],optimizer_groups=groups,
        monitor_policy=frozen['review']['monitors'],length_change_definition=length_change_definition,
        run_path=str(directory.relative_to(root)),implementation=_implementation(),runtime=frozen['runtime'])
    target=root/'plans'/safe_name(name)/'grpo.training.json'
    by_id={r['id']:r for r in training};engine=None
    with run_lock(directory):
        _save_frozen(directory/'config.json',config);_save_frozen(target,config)
        _save_frozen(directory/'meta/run_manifest.json',dict(config_path=str(target.relative_to(root)),run_path=config['run_path'],checkpoints=CHECKPOINTS))
        if (directory/'complete.json').exists():return verify_training(target,root)
        checkpoints=verified_checkpoints(directory)
        if any(n not in CHECKPOINTS for n in checkpoints):raise ValueError('unexpected scientific checkpoint boundary')
        last=max(checkpoints,default=0)
        history,records,logs=_history(root,checkpoints[last],groups) if checkpoints else ([],[],[])
        # Old attempts stay on disk; only sealed history above is accepted.
        attempt=directory/'attempts'/f'{len(list((directory/"attempts").glob("attempt-*"))):04d}'
        attempt=attempt.with_name('attempt-'+attempt.name);attempt.mkdir(parents=True,exist_ok=False)
        _write_json(attempt/'meta.json',dict(resumed_from=last,excluded_after_step=last))
        try:
            _write_state(directory,'running',last,64)
            from pilot_eval.grpo_train_engine import TrainingDependencies
            engine=(dependencies or TrainingDependencies()).load_training(plan,frozen['settings'],directory)
            if engine.runtime()!=frozen['runtime']:raise ValueError('training runtime differs from verified GPU preflight')
            if checkpoints:engine.restore_checkpoint(checkpoints[last])
            else:
                integrity=engine.check_integrity()
                if not integrity['base_unchanged']:raise ValueError('frozen base changed before training')
                _seal(directory,0,engine,[],groups)
            print(f'GRPO resume: sealed step {last}/64; prompts {last*8}/512; accepted draws {len(records)}/4096',flush=True)
            for step in range(last+1,65):
                rows=[by_id[q] for q in groups[step-1]]
                started=time.perf_counter();draws,diagnostics=engine.train_window(rows,step);_finite([draws,diagnostics])
                expected=[f'{r["id"]}:draw:{d}' for r in rows for d in range(8)]
                if [r['draw_id'] for r in draws]!=expected or any(r['step']!=step for r in draws):raise ValueError('training window group/draw identity mismatch')
                if any('advantage' not in r for r in draws):raise ValueError('training advantage record missing')
                lengths=[r['token_count'] for r in draws]
                log=dict(step=step,**diagnostics,reward=sum(r['reward'] for r in draws)/64,
                    strict_accuracy=sum(r['strict']['correct'] for r in draws)/64,
                    flexible_accuracy=sum(r['flexible']['correct'] for r in draws)/64,
                    cap_fraction=sum(r['capped'] for r in draws)/64,mean_tokens=sum(lengths)/64,p90_tokens=_percentile(lengths,90),
                    dead_group_fraction=sum(len({r['reward'] for r in draws[i:i+8]})==1 for i in range(0,64,8))/8)
                events=_monitor_events(logs+[log],config['monitor_policy'],length_change_definition);log['monitor_events']=events
                unit=attempt/f'step-{step:03d}.json'
                _write_json(unit,dict(step=step,prompt_ids=groups[step-1],responses=draws,log=log))
                history.append(dict(path=str(unit.relative_to(root)),sha256=file_hash(unit)));records.extend(draws);logs.append(log)
                action=next((e['action'] for e in events if e['action']=='stop'),None) or next((e['action'] for e in events if e['action']=='pause'),None)
                if action:
                    _write_json(attempt/'monitor-stop.json',dict(step=step,action=action,events=events,last_sealed_step=max(verified_checkpoints(directory))))
                    raise MonitorStop(action,f'monitor {action} at step {step}; inspection required; resume only from last sealed checkpoint')
                if step in CHECKPOINTS:
                    integrity=engine.check_integrity()
                    if not integrity['base_unchanged']:raise ValueError('frozen base changed at checkpoint boundary')
                    _seal(directory,step,engine,history,groups)
                _write_state(directory,'running',step,64)
                print(f'GRPO step {step}/64; prompts {step*8}/512; draws {step*64}/4096; sealed {max(verified_checkpoints(directory))}; seconds {time.perf_counter()-started:.2f}',flush=True)
            integrity=engine.check_integrity()
            if not integrity['base_unchanged']:raise ValueError('final frozen base integrity failed')
            # Reconstruct the authoritative record from checkpoint 64 before publishing.
            _,records,logs=_history(root,directory/'checkpoints/checkpoint-64',groups)
            result=dict(optimizer_steps=64,training_prompts=512,accepted_completions=len(records),checkpoints=CHECKPOINTS,
                base_integrity=integrity,attempt_count=len(list((directory/'attempts').glob('attempt-*'))),
                accepted_window_seconds=sum(x['seconds'] for x in logs),length_change_definition=length_change_definition)
            _write_json(directory/'results/results.json',result)
            for filename,values in [('responses.jsonl',records),('steps.jsonl',logs)]:
                destination=directory/'results'/filename;temporary=destination.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in values));temporary.replace(destination)
            files={str(p.relative_to(directory)):file_hash(p) for p in (directory/'results').glob('*') if p.is_file()}
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files=files))
            verify_training(target,root);_write_state(directory,'completed',64,64)
            return result
        except BaseException as exc:
            _write_state(directory,exc.state if isinstance(exc,MonitorStop) else 'failed',max(verified_checkpoints(directory),default=0),64,str(exc))
            raise
        finally:
            if engine is not None:engine.close()


class MonitorStop(RuntimeError):
    def __init__(self,state,message):self.state='paused' if state=='pause' else 'stopped';super().__init__(message)
