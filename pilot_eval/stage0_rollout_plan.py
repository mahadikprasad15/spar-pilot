"""Frozen R0 sources and exact-replay/fresh-uniform route selection."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_rollout_sources import read_sources,POLICY
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state,_verify_parents
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash,_save_frozen


def implementation():
    folder=Path(__file__).parent
    names=['stage0_rollout_plan.py','stage0_rollout_run.py','stage0_rollout_math.py','stage0_rollout_sources.py',
           'stage0_rollout_backend.py','stage0_rollout_results.py','stage0_rollouts.py','grpo_baseline.py','scoring.py']
    return {name:file_hash(folder/name) for name in names}


def prepare(audit_path,baseline_path,weights_dir,output_root,name,*,dependencies=None):
    import importlib.util
    if importlib.util.find_spec('pyarrow') is None:raise ValueError('Install the stage0-rollouts extra (pyarrow) on CPU before preparing sampling.')
    root=Path(output_root).resolve();safe_name(name);source=read_sources(audit_path,baseline_path,root)
    directory=root/f'runs/stage0/{name}/rollouts';path=root/'plans'/name/'stage0.rollouts.json'
    if source['state']!='audited':raise ValueError('Historical raw groups/runtime are incomplete; run the CPU source audit and restore its missing files.')
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    deps=dependencies or RolloutDependencies();weights=inside(root,weights_dir)
    inventory=deps.inventory(weights,source['source_audit']['sources']['sft'])
    fixture=inventory['execution_kind']=='synthetic-cpu-fixture';parents=dict(source['parents'])
    for value,digest in inventory['parents'].items():parents[str(inside(root,value).relative_to(root))]=digest
    if inventory['base_sha256']!=source['identity']['base_sha256']:raise ValueError('Selected weights differ from the audited untuned base.')
    if not fixture:
        for arm in ['sft','grpo']:
            metadata=source['source_audit']['sources'][arm]['training_metadata']
            if metadata['state']!='verified-recorded-config' or set(metadata.get('training_indices',[]))!={r['source_index'] for r in source['training']}:
                raise ValueError('Both audited training configurations must establish the common 512 IDs.')
        for name_ in ['tokenizer.json','tokenizer_config.json']:
            payload=weights/name_;parents[str(payload.relative_to(root))]=file_hash(payload)
    if source['seeds_recovered']:
        historical_order=source['historical_order'];remaining=[i for i in source['training_order'] if i not in historical_order]
        extension=[dict(key=f'group-{n:04d}',id=i,seed=source['legacy']['settings']['seed']+128+n,origin='new-extension') for n,i in enumerate(remaining)]
    else:historical_order=source['historical_order'];extension=[]
    seed=source['legacy']['settings'].get('seed',42)
    if type(seed) is not int or seed<0:seed=42
    uniform=[dict(key=f'group-{n:04d}',id=i,seed=seed+n,origin='new-uniform') for n,i in enumerate(source['training_order'])]
    config=dict(schema_version=1,protocol='stage0-rollouts-v1',name=name,run_path=str(directory.relative_to(root)),
        audit_path=str(inside(root,audit_path).relative_to(root)),baseline_path=str(inside(root,baseline_path).relative_to(root)),
        parents=parents,model_identity=source['identity'],weights_dir=str(weights.relative_to(root)),
        execution_kind='synthetic-cpu-fixture' if fixture else 'real-cuda',implementation=implementation(),numpy_version=np.__version__,
        legacy=source['legacy'],historical_runtime=source['legacy_runtime'],seeds_recovered=source['seeds_recovered'],policy_matches=source['policy_matches'],
        training_order=source['training_order'],uniform_order=source['training_order'],historical_order=historical_order,
        extension_units=extension,uniform_units=uniform,generation_settings=dict(POLICY,seed=seed),selection_seed=42,
        training_sha256=_hash(source['training']),historical_sha256=_hash(source['historical']),
        join_rule='two recorded questions x eight exact draws; matching source/settings hashes; otherwise new uniform 512x8',
        budget_ceiling_units=20,group_size=8,profile_questions=2,resources='CPU prepare/verify; one bounded FP32 CUDA runtime for profile/run')
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _jsonl(directory/'inputs/training.jsonl',source['training']);_jsonl(directory/'inputs/historical.jsonl',source['historical'])
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),total_training_questions=512,
            historical_questions=128,resources=config['resources'],budget_ceiling_units=20))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,512)
    return path


def load(path,root):
    path=inside(root,path);config=json.loads(path.read_text());directory=inside(root,config['run_path'])
    if config.get('protocol')!='stage0-rollouts-v1' or config.get('implementation')!=implementation() or config['numpy_version']!=np.__version__:
        raise ValueError('Rollout code/NumPy identity changed; retain evidence and select a new plan.')
    if config['budget_ceiling_units']!=20 or config['group_size']!=8 or config['profile_questions']!=2 or config['generation_settings']!={**POLICY,'seed':config['generation_settings']['seed']}:
        raise ValueError('Frozen rollout procedure changed.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):
        raise ValueError('Rollout plan/run identity changed.')
    _verify_parents(root,config['parents'])
    training=[json.loads(line) for line in (directory/'inputs/training.jsonl').read_text().splitlines()]
    historical=[json.loads(line) for line in (directory/'inputs/historical.jsonl').read_text().splitlines()]
    if _hash(training)!=config['training_sha256'] or _hash(historical)!=config['historical_sha256']:
        raise ValueError('Rollout source snapshots changed.')
    return config,directory,training,historical


def replay_policy(config,runtime):
    from pilot_eval import grpo_baseline
    reasons=[];old=config['historical_runtime']
    if not config['seeds_recovered']:reasons.append('historical seeds/order unavailable')
    if not config['policy_matches']:reasons.append('historical decoding/batching contract differs')
    if config['legacy']['implementation_sha256'].get('sampling')!=file_hash(Path(grpo_baseline.__file__)):
        reasons.append('recorded sampling implementation hash differs')
    if old.get('device') and old['device']!=runtime.get('device'):reasons.append('identified original GPU type differs')
    for key,value in old.get('versions',{}).items():
        if runtime.get('versions',{}).get(key)!=value:reasons.append('identified software version differs: '+key)
    if old.get('cuda') and old['cuda']!=runtime.get('cuda'):reasons.append('identified CUDA version differs')
    if not reasons:
        ids=config['historical_order'][:2];seeds=[b['seed'] for b in config['legacy']['batches'][:2]]
        units=[dict(key=f'profile-{n:04d}',id=i,seed=s,origin='profile') for n,(i,s) in enumerate(zip(ids,seeds))]
    else:units=[dict(u,key=f'profile-{n:04d}',origin='profile') for n,u in enumerate(config['uniform_units'][:2])]
    return dict(replay_eligible=not reasons,reasons=reasons,units=units,
        hardware_scope='original GPU/software enforced where recorded; unavailable metadata remains explicitly unknown')
