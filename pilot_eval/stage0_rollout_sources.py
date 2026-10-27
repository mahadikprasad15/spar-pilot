"""Used-file audit of the original sampling groups and common training cohort."""
import json
from pathlib import Path

from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import status
from pilot_eval.stage0_rollout_math import normalize
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

POLICY=dict(groups_per_batch=1,max_new_tokens=1024,temperature=1.,top_p=1.,top_k=0,scorer='gsm8k-flexible-v3')


def read_sources(audit_path,baseline_path,root):
    audit_path,baseline_path=inside(root,audit_path),inside(root,baseline_path)
    audited=status(audit_path,root)['audit']
    if not audited or not audited['primary_sources_ready']:raise ValueError('Complete the Stage 0 source audit first.')
    parents={str(audit_path.relative_to(root)):file_hash(audit_path)}
    def read(path):
        path=inside(root,path);parents[str(path.relative_to(root))]=file_hash(path)
        return json.loads(path.read_text())
    audit_config=read(audit_path);audit_dir=inside(root,audit_config['run_path'])
    read(audit_dir/'complete.json');read(audit_dir/'results/source-audit.json')
    for source in audited['sources'].values():parents.update(source.get('parents',{}))
    legacy=read(baseline_path)
    if legacy.get('protocol_version')!='pilot4-sampling-baseline-v1' or legacy.get('group_size')!=8 or legacy.get('adapter') is not None:
        raise ValueError('Select the original untuned eight-draw sampling baseline.')
    source_path=inside(root,legacy['source_path']);plan=read(source_path)
    if file_hash(source_path)!=legacy['source_sha256'] or plan.get('protocol_version')!='pilot4-grpo-preparation-v1':
        raise ValueError('Historical sampling source plan changed.')
    marker=read(source_path.parent/'prepare-complete.json')
    if marker['config_sha256']!=_hash(plan) or str(source_path.relative_to(root)) not in marker['files']:
        raise ValueError('Historical preparation identity differs.')
    for name,digest in marker['files'].items():
        path=inside(root,name)
        if file_hash(path)!=digest:raise ValueError('Historical preparation payload changed.')
        parents[str(path.relative_to(root))]=digest
    rows=read(plan['training_items_path']);order=plan['optimizer_prompt_order']
    if (_hash(rows)!=plan['training_items_sha256'] or len(rows)!=512 or len(set(order))!=512
            or set(order)!={r['id'] for r in rows}
            or {r['id'] for r in rows}!={f"gsm8k:train:{i}" for i in plan['training_indices']}
            or any(not isinstance(r.get('prompt'),str) or not r['prompt'] or not isinstance(r.get('gold'),str) for r in rows)):
        raise ValueError('Expected the verified common 512 training inputs and recorded optimizer order.')
    identity=audited['sources']['sft']['identity']
    if any(plan.get(k)!=identity[k] or legacy.get(k)!=identity[k] for k in ['model','model_revision','tokenizer_revision','dtype']):
        raise ValueError('Sampling source model/tokenizer/FP32 identity differs from the audited experiment.')
    if legacy['prompt_template']!=plan['source_prompt_contract']:raise ValueError('Historical prompt contract differs from training.')
    directory=inside(root,legacy['run_path']);missing=[];records=[];verified=0;runtime=None
    runtime_path=directory/'meta/runtime.json'
    if runtime_path.exists():runtime=read(runtime_path)
    else:missing.append(str(runtime_path.relative_to(root)))
    if (directory/'config.json').exists():
        if read(directory/'config.json')!=legacy:raise ValueError('Historical run configuration differs from the selected config.')
    else:missing.append(str((directory/'config.json').relative_to(root)))
    settings=legacy['settings'];policy_matches=all(settings.get(k)==v for k,v in POLICY.items())
    batches=legacy.get('batches');seed_recovered=isinstance(batches,list) and len(batches)==128
    if seed_recovered:
        seed_recovered=all(isinstance(b,dict) and isinstance(b.get('ids'),list) and len(b['ids'])==1
            and type(b.get('seed')) is int and b['seed']>=0 for b in batches)
    recorded_ids=[b['ids'][0] for b in batches] if seed_recovered else None
    if seed_recovered and (len(set(recorded_ids))!=128 or recorded_ids!=settings.get('subset_ids')):seed_recovered=False
    if seed_recovered and [b['seed'] for b in batches]!=[settings.get('seed',-1)+n for n in range(128)]:
        seed_recovered=False
    by_id={r['id']:r for r in rows};seen=set()
    for index in range(128):
        shard=directory/'batches'/f'{index:04d}.json';seal_path=shard.with_suffix('.complete.json')
        if not shard.exists() or not seal_path.exists():
            missing.extend(str(p.relative_to(root)) for p in [shard,seal_path] if not p.exists());continue
        if runtime is None:continue
        seal=read(seal_path)
        if seal!=dict(sha256=file_hash(shard),config_sha256=_hash(legacy),runtime_sha256=file_hash(runtime_path)):
            raise ValueError('Historical sampling shard/config/runtime hash mismatch.')
        raw=read(shard)
        if len(raw)!=8 or [r.get('draw_index') for r in raw]!=list(range(8)) or len({r.get('group_id') for r in raw})!=1:
            raise ValueError('Historical group must contain exactly eight unique ordered draws.')
        identifier=raw[0]['group_id']
        if identifier not in by_id or identifier in seen or (seed_recovered and identifier!=recorded_ids[index]):
            raise ValueError('Historical group IDs are duplicated, reordered or outside the common 512.')
        seen.add(identifier);training=by_id[identifier]
        for d,value in enumerate(raw):
            if value['prompt']!=training['prompt'] or value['gold']!=training['gold'] or value['draw_id']!=f'{identifier}:draw:{d}':
                raise ValueError('Historical prompt/gold/draw identity changed.')
            item=normalize(training,value,d,origin='historical',seed=batches[index]['seed'] if seed_recovered else None,
                limit=settings['max_new_tokens'],eos=runtime.get('eos_token_id'),historical_reward=value['reward'])
            if (item['reward_v3']!=value['reward'] or bool(value['flexible']['correct'])!=bool(item['reward_v3'])):
                raise ValueError('Historical flexible-v3 reward disagrees with raw-response reconstruction; retain source and inspect.')
            records.append(item)
        verified+=1
    historical_order=recorded_ids if seed_recovered else [i for i in order if i in seen]
    missing=list(dict.fromkeys(missing))
    return dict(state='audited' if verified==128 and not missing else 'incomplete',verified_historical_groups=verified,
        missing=missing,parents=parents,identity=identity,source_audit=audited,training=rows,training_order=order,
        historical=records,historical_order=historical_order,legacy=legacy,legacy_runtime=runtime,
        seeds_recovered=seed_recovered,policy_matches=policy_matches,
        recovery_note='Explicit recorded batch IDs/seeds verified; never source-index seeds.' if seed_recovered else 'Historical seed/order unavailable; uniform fresh 512-group fallback required.')
