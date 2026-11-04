"""Versioned offline-pool handoff from independently verified R0 products.

Preparation and importing are CPU-only and do not authorize sampling.
"""
import json
import math
from pathlib import Path

from pilot_eval.grpo_prepare import load_grpo_prepared
from pilot_eval.sft import safe_name
from pilot_eval.run import _write_json
from pilot_eval.stage0_rollout_math import normalize
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer
from pilot_eval.grpo_preflight import load_frozen
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


def _prepared(path, root):
    config, training, evaluation, _ = load_grpo_prepared(path, root)
    return config, training, evaluation


def _verified_r0(path, root):
    if json.loads(Path(path).read_text()).get('protocol')=='offline-grpo-fresh-r0-v1':
        from pilot_eval.offline_grpo_fresh import verified_records
        return verified_records(path,root)
    from pilot_eval.stage0_rollout_plan import load
    from pilot_eval.stage0_rollout_results import verify
    summary = verify(path, root)
    config, directory, _, _ = load(path, root)
    records = [json.loads(line) for line in (directory/'results/rollouts.jsonl').read_text().splitlines()]
    specials = json.loads((directory/'collection/inputs/special-tokens.json').read_text())
    return records, dict(eos_token_id=specials['eos_token_id'], route=summary['route'],
                         execution_kind=config['execution_kind'], model_identity={k:config['model_identity'][k] for k in ['model','model_revision','tokenizer_revision','dtype']})


def _source(path, root):
    frozen = load_frozen(path, root)
    config, training, evaluation = _prepared(inside(root, frozen['source_path']), root)
    from pilot_eval.stage0_rollout_sources import POLICY
    expected = dict(completion_limit=POLICY['max_new_tokens'], top_p=POLICY['top_p'], top_k=POLICY['top_k'])
    if {k: frozen['settings'].get(k) for k in expected} != expected:
        raise ValueError('Frozen online sampling settings differ from the pool sampler policy')
    groups = frozen['optimizer_groups']
    if len(groups) != 64 or any(not isinstance(group, list) or len(group) != 8 or len(set(group)) != 8 for group in groups):
        raise ValueError('Frozen optimizer plan must contain 64 distinct eight-question windows')
    if [item for group in groups for item in group] != config['optimizer_prompt_order']:
        raise ValueError('Frozen optimizer windows differ from prepared source order')
    order = config['optimizer_prompt_order']
    ids = [r['id'] for r in training]
    if len(ids) != 512 or len(set(ids)) != 512 or len(order) != 512 or set(order) != set(ids):
        raise ValueError('Prepared training source must identify 512 unique ordered questions')
    if set(ids) & {r['id'] for r in evaluation}:
        raise ValueError('Held-out question overlaps training')
    for row in training:
        n = row.get('prompt_tokens'); tokens = row.get('input_ids')
        if type(n) is not int or n <= 0 or not isinstance(tokens,list) or n > len(tokens) or any(type(i) is not int or i < 0 for i in tokens[:n]):
            raise ValueError('Prepared prompt token identity invalid')
    return config, training, evaluation


def prepare(source, output_root, name):
    root=Path(output_root).resolve(); safe_name(name); source=inside(root, source)
    config, training, evaluation = _source(source, root)
    directory=root/'runs/offline-grpo'/name/'pool'; path=root/'plans'/name/'offline.pool.json'
    plan=dict(protocol='offline-grpo-pool-v1', name=name, route='unselected',
              source=str(source.relative_to(root)), source_sha256=file_hash(source),
              source_identity={k:config[k] for k in ['model','model_revision','tokenizer_revision','dtype']},
              implementation_sha256=file_hash(Path(__file__)),
              training_sha256=_hash(training), evaluation_sha256=_hash(evaluation),
              optimizer_order=config['optimizer_prompt_order'], run_path=str(directory.relative_to(root)),
              status='prepared', scientific_generation_performed=False)
    with _writer(directory):
        _save_frozen(path,plan)
        _save_frozen(directory/'config.json',plan)
        status=directory/'meta/status.json'
        if not status.exists():
            _write_json(status,dict(state='prepared',complete=False))
    return path


def _plan(path, root):
    path=inside(root,path); plan=json.loads(path.read_text())
    if plan.get('protocol')!='offline-grpo-pool-v1':raise ValueError('Unsupported pool plan')
    if plan.get('implementation_sha256') != file_hash(Path(__file__)):
        raise ValueError('Pool implementation changed; prepare a new plan')
    directory=inside(root,plan['run_path'])
    if json.loads((directory/'config.json').read_text())!=plan:raise ValueError('Pool plan changed')
    source=inside(root,plan['source'])
    if file_hash(source)!=plan['source_sha256']:raise ValueError('Prepared source changed')
    config, training, evaluation=_source(source,root)
    if _hash(training)!=plan['training_sha256'] or _hash(evaluation)!=plan['evaluation_sha256'] or config['optimizer_prompt_order']!=plan['optimizer_order']:
        raise ValueError('Prepared source cohort changed')
    return plan,directory,training


def _windows(plan, training, records, metadata):
    byid={r['id']:r for r in training}; groups={}; seen=set()
    if metadata.get('model_identity') != plan['source_identity']:
        raise ValueError('R0 model identity differs from prepared training source')
    eos=metadata['eos_token_id']
    if type(eos) is not int or eos<0:raise ValueError('Missing verified EOS identity')
    for record in records:
        key=(record.get('group_id'),record.get('draw_index'))
        if key in seen or key[0] not in byid:raise ValueError('Duplicate or foreign draw identity')
        seen.add(key); row=byid[key[0]]
        expected=normalize(row,record,key[1],origin=record['origin'],seed=record['seed'],limit=1024,eos=eos,
                           historical_reward=record['historical_training_reward'])
        if record!=expected:raise ValueError('Draw tokens, scores or provenance do not reconstruct')
        draw=dict(record, completion_mask=[1]*len(record['generated_token_ids']), batch_id=f"{record['origin']}:{record['group_id']}:{record['seed']}")
        groups.setdefault(key[0],[]).append(draw)
    if len(records)!=4096 or len(groups)!=512 or len({r['draw_id'] for r in records})!=4096:
        raise ValueError('Expected 512 questions and 4096 unique draws')
    windows=[]; summaries=[]
    for offset in range(0,512,8):
        window=[]; lengths=[]
        counts=dict(all_correct=0,all_wrong=0,mixed=0,tokens=0,correct=0,capped=0,scorer_invalid=0)
        for identifier in plan['optimizer_order'][offset:offset+8]:
            row=byid[identifier];draws=sorted(groups[identifier],key=lambda d:d['draw_index'])
            if [d['draw_index'] for d in draws]!=list(range(8)):raise ValueError('Expected eight ordered draws')
            successes=sum(d['reward_v3'] for d in draws)
            counts['all_correct' if successes==8 else 'all_wrong' if successes==0 else 'mixed']+=1
            counts['correct']+=successes;counts['tokens']+=sum(len(d['completion_mask']) for d in draws);counts['capped']+=sum(d['capped'] for d in draws)
            counts['scorer_invalid']+=sum(d['scores']['flexible_v3']['status']=='invalid' for d in draws)
            lengths.extend(len(d['generated_token_ids']) for d in draws)
            window.append(dict(id=identifier,prompt=row['prompt'],gold=row['gold'],input_ids=row['input_ids'][:row['prompt_tokens']],draws=draws))
        windows.append(window)
        summaries.append(dict(window=offset//8+1,**counts,mean_reward=counts['correct']/64,
                              cap_rate=counts['capped']/64,length_mean=sum(lengths)/64,
                              length_p90=sorted(lengths)[math.ceil(.9*len(lengths))-1]))
    return windows,summaries


def import_rollouts(path,rollouts,output_root):
    root=Path(output_root).resolve();plan,directory,training=_plan(path,root);rollouts=inside(root,rollouts)
    records,metadata=_verified_r0(rollouts,root)
    windows,summaries=_windows(plan,training,records,metadata)
    payload=dict(protocol='offline-grpo-pool-handoff-v1',plan_sha256=_hash(plan),r0_path=str(rollouts.relative_to(root)),
                 r0_sha256=file_hash(rollouts),metadata=metadata,
                 pool_order=list(dict.fromkeys(record['group_id'] for record in records)),
                 windows=windows,summaries=summaries)
    with _writer(directory):
        _plan(path, root)
        _save_frozen(directory/'windows.json',windows)
        _save_frozen(directory/'handoff.json',{k:v for k,v in payload.items() if k!='windows'} | dict(windows_sha256=file_hash(directory/'windows.json')))
        _write_json(directory/'meta/status.json',dict(state='complete',complete=True,execution_kind=metadata['execution_kind']))
    return dict(complete=True,execution_kind=metadata['execution_kind'],groups=512,draws=4096)


def load_pool(path,output_root):
    root=Path(output_root).resolve();plan,directory,training=_plan(path,root)
    receipt=json.loads((directory/'handoff.json').read_text());rollouts=inside(root,receipt['r0_path'])
    if receipt['protocol']!='offline-grpo-pool-handoff-v1' or receipt['plan_sha256']!=_hash(plan) or receipt['r0_sha256']!=file_hash(rollouts) or receipt['windows_sha256']!=file_hash(directory/'windows.json'):
        raise ValueError('Pool handoff identity changed')
    records,metadata=_verified_r0(rollouts,root);windows,summaries=_windows(plan,training,records,metadata)
    pool_order=list(dict.fromkeys(record['group_id'] for record in records))
    if metadata!=receipt['metadata'] or pool_order!=receipt['pool_order'] or summaries!=receipt['summaries'] or windows!=json.loads((directory/'windows.json').read_text()):
        raise ValueError('Pool handoff differs from verified rollout evidence')
    return windows


def verify(path,output_root):
    windows=load_pool(path,output_root)
    return dict(complete=True,windows=len(windows),questions=sum(map(len,windows)),draws=sum(len(q['draws']) for w in windows for q in w))


def inventory(audit, baseline, output_root, name):
    """Persist the existing R0 source audit, including missing raw evidence.

    This operation only inspects local evidence; it does not select a route or
    treat an incomplete audit as authorization to reuse historical draws.
    """
    from pilot_eval.stage0_rollout_sources import read_sources
    root=Path(output_root).resolve();safe_name(name)
    source=read_sources(audit,baseline,root)
    report={k:v for k,v in source.items() if k not in ['training','historical','source_audit']}
    report['selected_route']=None
    report['note']='Inventory only; sampling and reuse require their own verified route.'
    path=root/'runs/offline-grpo'/name/'pool'/'inputs'/'historical-inventory.json'
    with _writer(path.parent.parent):
        _save_frozen(path,report)
    return report
