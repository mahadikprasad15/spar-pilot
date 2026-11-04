"""Versioned offline-GRPO behavior evaluation and paired reporting."""
import json
from pathlib import Path

import numpy as np

from pilot_eval import grpo_evaluation as online
from pilot_eval.grpo_behaviour_report import _interval, _summary
from pilot_eval.grpo_prepare import _within, load_grpo_prepared
from pilot_eval.grpo_training import CHECKPOINTS, _verified_checkpoints
from pilot_eval.offline_grpo_training import _load, _verify_complete
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen

summarize = _summary


def incompatible_fields(first, second):
    """Conservative full-contract comparison; caller verifies each source separately."""
    fields = ['items', 'prompt_template', 'scorer', 'strict_scorer', 'runtime',
              'policies', 'units', 'settings', 'seed', 'seed_rule', 'cap_reward',
              'model', 'model_revision', 'tokenizer_revision', 'dtype']
    return [k for k in fields if k not in first or k not in second or first[k] != second[k]]


def compatible(first, second):
    return not incompatible_fields(first, second)


def _identity():
    from pilot_eval import grpo_eval_engine, scoring, grpo_behaviour_report
    return {name: file_hash(Path(module.__file__)) for name, module in {
        'wrapper': __import__(__name__, fromlist=['']), 'evaluator': online,
        'engine': grpo_eval_engine, 'scorer': scoring, 'report': grpo_behaviour_report}.items()}


def _source(path, root):
    config, directory, frozen, plan, windows, zero = _load(path, root)
    _verify_complete(directory, root, config, config['optimizer_groups'])
    _, _, items, _ = load_grpo_prepared(root/frozen['source_path'], root)
    if len(items) != 150 or len({r['id'] for r in items}) != 150:
        raise ValueError('expected 150 unique held-out items')
    checkpoints = _verified_checkpoints(directory)
    if sorted(checkpoints) != CHECKPOINTS:
        raise ValueError('offline checkpoint source incomplete')
    return config, directory, frozen, plan, items, checkpoints


def _config(path, root, name):
    training, directory, frozen, plan, items, checkpoints = _source(path, root)
    policies = online._policies(frozen['settings'])
    units = []
    ids = [item['id'] for item in items]
    for policy in policies:
        for start in range(0, 150, policy['batch_size']):
            index = start // policy['batch_size']
            units.append(dict(key=f'{policy["mode"]}-{policy["step"]}-batch-{index:04d}',
                              policy=policy, ids=ids[start:start+policy['batch_size']], seed=42+index))
    run_path = root/'runs/offline-grpo'/safe_name(name)/'behaviour'
    config = dict(protocol='offline-grpo-behaviour-v1', training_path=str(path.relative_to(root)),
        training_sha256=file_hash(path), training_seal_sha256=file_hash(directory/'complete.json'),
        model=plan['model'], model_revision=plan['model_revision'], tokenizer_revision=plan['tokenizer_revision'],
        dtype=plan['dtype'], prompt_template=plan['source_prompt_contract'], items=items, seed=42,
        seed_rule='42 + ordered batch index; same schedule across checkpoints; reseed each sealed batch',cap_reward=0,
        scorer='gsm8k-flexible-v3', strict_scorer='gsm8k-strict', settings=frozen['settings'],
        runtime=training['runtime'], margin=frozen['review']['margin'], policies=policies, units=units,
        reference_zero=dict(path=plan['source_evidence']['checkpoints']['0']['path'],
            seal_sha256=plan['source_evidence']['checkpoints']['0']['complete_sha256'],
            adapter_sha256=plan['source_evidence']['checkpoints']['0']['adapter_sha256']),
        checkpoints={str(s):dict(path=str(p.relative_to(root)),seal_sha256=file_hash(p/'complete.json'),
                         adapter_sha256=file_hash(p/'adapter_model.safetensors')) for s,p in checkpoints.items()},
        run_path=str(run_path.relative_to(root)), implementation=_identity())
    return config, plan


def prepare(training_path, output_root, name):
    root=Path(output_root).resolve();path=_within(root,training_path)
    config,_=_config(path,root,name)
    target=root/'plans'/safe_name(name)/'offline.behaviour.json'
    directory=root/config['run_path']
    with run_lock(directory):
        _save_frozen(target,config)
        _save_frozen(directory/'config.json',config)
        _save_frozen(directory/'meta/run_manifest.json', dict(
            config_path=str(target.relative_to(root)), source_path=config['training_path'],
            run_path=config['run_path'], responses=3150, policies=config['policies']))
    return target


def _load_config(path,root):
    path=_within(root,path);config=json.loads(path.read_text())
    if config.get('protocol')!='offline-grpo-behaviour-v1':raise ValueError('unsupported behaviour source')
    expected,plan=_config(root/config['training_path'],root,Path(config['run_path']).parent.name)
    if config!=expected or json.loads((root/config['run_path']/'config.json').read_text())!=config:
        raise ValueError('offline evaluation source, plan or implementation changed')
    return config,plan


def _read(root,config,unit):
    return online._read_unit(root,config,unit,_hash(config))


def _verify_anchor(root, config):
    directory=root/config['run_path']; reference=config['reference_zero']
    ref_path=_within(root,reference['path'])
    if file_hash(ref_path/'complete.json')!=reference['seal_sha256'] or file_hash(ref_path/'adapter_model.safetensors')!=reference['adapter_sha256']:
        raise ValueError('reference checkpoint-zero identity changed')
    if reference['adapter_sha256']!=config['checkpoints']['0']['adapter_sha256']:
        raise ValueError('offline and historical checkpoint-zero adapters differ')
    receipt=json.loads((directory/'anchor/complete.json').read_text())
    evidence=directory/'anchor/reference.json'
    if receipt!=dict(config_sha256=_hash(config),sha256=file_hash(evidence)):
        raise ValueError('checkpoint-zero anchor seal changed')
    reference_records=json.loads(evidence.read_text())
    offline=[r for unit in config['units'] if unit['policy']['mode']=='greedy' and unit['policy']['step']==0 for r in _read(root,config,unit)]
    expected=[dict(item_id=r['item_id'],generated_token_ids=r['generated_token_ids']) for r in offline]
    if reference_records!=expected or len(expected)!=150:
        raise ValueError('same-runtime checkpoint-zero token anchor failed')
    return receipt


def _anchor(root,config,engine,rows):
    directory=root/config['run_path'];reference=config['reference_zero'];ref_path=_within(root,reference['path'])
    if file_hash(ref_path/'complete.json')!=reference['seal_sha256'] or file_hash(ref_path/'adapter_model.safetensors')!=reference['adapter_sha256']:
        raise ValueError('reference checkpoint-zero identity changed')
    if reference['adapter_sha256']!=config['checkpoints']['0']['adapter_sha256']:
        raise ValueError('offline and historical checkpoint-zero adapters differ')
    engine.select_checkpoint(ref_path,0)
    actual=[]
    for unit in config['units']:
        if unit['policy']['mode']!='greedy' or unit['policy']['step']!=0:continue
        outputs=engine.generate([rows[i] for i in unit['ids']],unit['policy'],unit['seed'])
        generated=online._records(config,unit,outputs)
        expected=_read(root,config,unit)
        if [r['generated_token_ids'] for r in generated]!=[r['generated_token_ids'] for r in expected]:
            raise ValueError('same-runtime checkpoint-zero token anchor failed')
        actual.extend(dict(item_id=r['item_id'],generated_token_ids=r['generated_token_ids']) for r in generated)
    evidence=directory/'anchor/reference.json';_save_frozen(evidence,actual)
    _save_frozen(directory/'anchor/complete.json',dict(config_sha256=_hash(config),sha256=file_hash(evidence)))
    _verify_anchor(root,config)


def verify(path,output_root,*,marker=None):
    root=Path(output_root).resolve();config,_=_load_config(path,root);directory=root/config['run_path']
    if json.loads((directory/'meta/runtime.json').read_text())!=config['runtime']:
        raise ValueError('evaluation runtime changed')
    if marker is None:
        marker=json.loads((directory/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config):raise ValueError('evaluation seal identity changed')
    required={'config.json','meta/runtime.json','results/responses.jsonl','results/results.json'}
    if not required.issubset(marker['files']):raise ValueError('evaluation seal incomplete')
    for name,digest in marker['files'].items():
        if file_hash(_within(directory,directory/name))!=digest:raise ValueError('evaluation payload changed')
    _verify_anchor(root,config)
    records=[record for unit in config['units'] for record in _read(root,config,unit)]
    saved=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
    if records!=saved or len(records)!=3150 or len({r['record_id'] for r in records})!=3150:
        raise ValueError('evaluation records incomplete or duplicated')
    return config,records


def run(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();config,plan=_load_config(path,root);directory=root/config['run_path']
    engine=None
    with run_lock(directory):
        if (directory/'complete.json').exists():
            verify(path,root)
            return json.loads((directory/'results/results.json').read_text())
        missing=[u for u in config['units'] if not (directory/'units'/f'{u["key"]}.complete.json').exists()]
        for unit in config['units']:
            if unit not in missing:_read(root,config,unit)
        try:
            if missing:
                if dependencies is None:
                    from pilot_eval.offline_grpo_training import _verify_preflight
                    training=json.loads((root/config['training_path']).read_text())
                    _verify_preflight(root/training['run_path'],training)
                from pilot_eval.grpo_eval_engine import EvaluationDependencies
                engine=(dependencies or EvaluationDependencies()).load_evaluation(plan,config['settings'],directory)
                if engine.runtime()!=config['runtime']:raise ValueError('evaluation runtime differs from approved runtime')
                if dependencies is None and engine.initial_base!=plan['source_evidence']['base_sha256']:
                    raise ValueError('evaluation base differs from verified training source')
                _save_frozen(directory/'meta/runtime.json',config['runtime'])
            rows={r['id']:r for r in config['items']};selected=None
            anchor_verified=False
            for number,unit in enumerate(missing,1):
                step=unit['policy']['step']
                if step != 0 and not anchor_verified:
                    if (directory/'anchor/complete.json').exists():_verify_anchor(root,config)
                    else:_anchor(root,config,engine,rows)
                    anchor_verified=True;selected=None
                if selected!=step:
                    if not engine.check_integrity()['base_unchanged']:raise ValueError('base integrity failure')
                    engine.select_checkpoint(root/config['checkpoints'][str(step)]['path'],step);selected=step
                outputs=engine.generate([rows[i] for i in unit['ids']],unit['policy'],unit['seed'])
                records=online._records(config,unit,outputs)
                shard=directory/'units'/f'{unit["key"]}.json';_write_json(shard,records)
                _write_json(shard.with_suffix('.complete.json'),dict(sha256=file_hash(shard),
                    config_sha256=_hash(config),runtime_sha256=file_hash(directory/'meta/runtime.json')))
                _write_state(directory,'running',len(config['units'])-len(missing)+number,len(config['units']))
                print(f'Offline behavior: {len(config["units"])-len(missing)+number}/{len(config["units"])} sealed batches; {unit["key"]}',flush=True)
            if engine and not engine.check_integrity()['base_unchanged']:raise ValueError('base integrity failure')
            records=[r for unit in config['units'] for r in _read(root,config,unit)]
            output=directory/'results/responses.jsonl';output.parent.mkdir(parents=True,exist_ok=True)
            temporary=output.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));temporary.replace(output)
            result=dict(total=len(records),greedy=750,sampled=2400,scorer=config['scorer'])
            _write_json(directory/'results/results.json',result)
            files={str(p.relative_to(directory)):file_hash(p) for p in [directory/'config.json',directory/'meta/runtime.json',output,directory/'results/results.json']}
            marker=dict(config_sha256=_hash(config),files=files)
            verify(path,root,marker=marker)
            _write_json(directory/'complete.json',marker)
            _write_state(directory,'completed',len(config['units']),len(config['units']))
            return result
        except Exception as exc:
            done=sum((directory/'units'/f'{u["key"]}.complete.json').exists() for u in config['units'])
            _write_state(directory,'failed',done,len(config['units']),error=str(exc));raise
        finally:
            if engine:engine.close()


def report(path, output_root, name, *, online_path=None):
    root=Path(output_root).resolve();path=_within(root,path)
    online_path=None if online_path is None else _within(root,online_path)
    source,records=verify(path,root)
    ids=[item['id'] for item in source['items']]
    grouped={}
    for row in records:grouped.setdefault((row['mode'],row['step'],row['item_id']),[]).append(row)
    indices=np.random.default_rng(42).integers(0,150,size=(10000,150))
    summaries={}; values={}; paired=[]
    for policy in source['policies']:
        mode,step,draws=policy['mode'],policy['step'],policy['draws']; ordered=[];item_values=[]
        for identifier in ids:
            rows=sorted(grouped.get((mode,step,identifier),[]),key=lambda row:row['draw_index'])
            if [row['draw_index'] for row in rows]!=list(range(draws)):raise ValueError('missing paired draws')
            ordered.extend(rows);item_values.append(sum(row['reward'] for row in rows)/draws)
            paired.append(dict(item_id=identifier,mode=mode,step=step,draws=rows))
        summaries[f'{mode}-{step}']=summarize(ordered)
        values[(mode,step)]=np.array(item_values)
    changes={}
    for policy in source['policies']:
        mode,step=policy['mode'],policy['step'];delta=values[(mode,step)]-values[(mode,0)]
        changes[f'{mode}-{step}']=dict(estimate=float(delta.mean()),interval_95=_interval(delta,indices))
    transitions=[]
    for step in CHECKPOINTS[1:]:
        for identifier in ids:
            before=grouped[('greedy',0,identifier)][0];after=grouped[('greedy',step,identifier)][0]
            transitions.append(dict(item_id=identifier,step=step,before=before['reward'],after=after['reward']))
    anchor=dict(state='passed',source=source['reference_zero']['path'],origin='fresh same-runtime historical checkpoint-zero inference')
    comparison=None
    cross_arm_transitions=[]
    comparison_status=dict(state='not-selected')
    if online_path is not None:
        other,other_records=online.verify_evaluation(online_path,root)
        differing=incompatible_fields(source,other)
        runtime_differs='runtime' in differing
        if not set(differing)-{'runtime'}:
            if source['checkpoints']['0']['adapter_sha256']!=other['checkpoints']['0']['adapter_sha256']:
                raise ValueError('checkpoint-zero adapter identities differ')
            offline_zero={(r['item_id'],r['draw_index']):r['generated_token_ids'] for r in records if r['mode']=='greedy' and r['step']==0}
            online_zero={(r['item_id'],r['draw_index']):r['generated_token_ids'] for r in other_records if r['mode']=='greedy' and r['step']==0}
            if runtime_differs:
                # Different hardware may change tokens without changing the policy; record it.
                changed=sum(online_zero.get(k)!=v for k,v in offline_zero.items())
                anchor['online_token_agreement']=dict(compared=len(offline_zero),differing_items=changed,exact=not changed)
                old,new=other['runtime'],source['runtime']
                comparison_status=dict(state='performed',runtime_differences=sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k)))
            else:
                if offline_zero!=online_zero:raise ValueError('same-runtime checkpoint-zero token anchor failed')
                anchor=dict(state='passed',source=str(online_path.relative_to(root)),source_sha256=file_hash(online_path))
                comparison_status=dict(state='performed')
            comparison={}
            other_groups={}
            for row in other_records:
                other_groups.setdefault((row['mode'],row['step'],row['item_id']),[]).append(row)
            for policy in source['policies']:
                mode,step,draws=policy['mode'],policy['step'],policy['draws']
                result_metrics={}
                for metric in ['reward','strict','length','invalid','cap']:
                    differences=[]
                    for identifier in ids:
                        a=grouped[(mode,step,identifier)]
                        b=other_groups.get((mode,step,identifier),[])
                        if sorted(r['draw_index'] for r in b)!=list(range(draws)):raise ValueError('online comparison cohort differs')
                        def value(r):
                            if metric=='strict':return int(r['strict']['correct'])
                            if metric=='length':return r['token_count']
                            if metric=='invalid':return int(r['flexible']['status']=='invalid')
                            if metric=='cap':return int(r['capped'])
                            return r['reward']
                        differences.append(sum(map(value,a))/draws-sum(map(value,b))/draws)
                    delta=np.array(differences)
                    result_metrics[metric]=dict(estimate=float(delta.mean()),interval_95=_interval(delta,indices))
                comparison[f'{mode}-{step}']=result_metrics
                if mode=='greedy':
                    for identifier in ids:
                        cross_arm_transitions.append(dict(item_id=identifier,step=step,
                            online=other_groups[(mode,step,identifier)][0]['reward'],
                            offline=grouped[(mode,step,identifier)][0]['reward']))
        else:
            anchor['historical_output_reuse']='incompatible runtime, cohort, decoder or batch contract; fresh anchor retained'
            comparison_status=dict(state='not-performed',differing_fields=differing)
    directory=root/'reports'/safe_name(name)
    config=dict(protocol='offline-grpo-behaviour-report-v1',source=str(_within(root,path).relative_to(root)),
                source_sha256=file_hash(path),source_seal_sha256=file_hash(root/source['run_path']/'complete.json'),
                online_path=None if online_path is None else str(online_path.relative_to(root)),
                online_sha256=None if online_path is None else file_hash(online_path),implementation=_identity())
    result=dict(summaries=summaries,changes=changes,online_comparison=comparison,online_comparison_status=comparison_status,anchor=anchor,
                note='Sampled accuracy is mean reward per draw, not pass@8. Intervals are pointwise paired item bootstraps; one training seed does not estimate run variability.')
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        _save_frozen(directory/'results/results.json',result)
        _save_frozen(directory/'results/bootstrap-indices.json',indices.tolist())
        paired_path=directory/'results/paired-items.jsonl'
        if paired_path.exists():
            if [json.loads(line) for line in paired_path.read_text().splitlines()]!=paired:raise ValueError('paired output changed')
        else:
            paired_path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in paired))
        _save_frozen(directory/'results/transitions.json',transitions)
        _save_frozen(directory/'results/cross-arm-transitions.json',cross_arm_transitions)
        report_path=directory/'results/report.md'
        content='# Offline GRPO behavior\n\n'+result['note']+'\n\nCheckpoint-zero anchor: '+anchor['state']+'\n\n'
        comparison_text={'performed':'performed','not-selected':'not selected'}.get(comparison_status['state'],'NOT performed; differing fields: '+', '.join(comparison_status.get('differing_fields',[])))
        if comparison_status.get('runtime_differences'):comparison_text+='; runtime differs in: '+', '.join(comparison_status['runtime_differences'])
        content+='Online comparison: '+comparison_text+'\n\n'
        content+='| Setting | Flexible accuracy | Strict accuracy | Mean tokens | Cap rate |\n|---|---:|---:|---:|---:|\n'
        for key,row in summaries.items():
            content+=f"| {key} | {row['flexible_accuracy']:.3f} | {row['strict_accuracy']:.3f} | {row['mean_tokens']:.2f} | {row['cap_rate']:.3f} |\n"
        if report_path.exists() and report_path.read_text()!=content:raise ValueError('report text changed')
        report_path.write_text(content)
        _save_frozen(directory/'meta/run_manifest.json',dict(source=config['source'],report_path=str(directory.relative_to(root)),anchor=anchor))
        files={str(p.relative_to(directory)):file_hash(p) for p in [directory/'config.json',*sorted((directory/'results').iterdir())]}
        _save_frozen(directory/'complete.json',dict(config_sha256=_hash(config),files=files))
    return result
