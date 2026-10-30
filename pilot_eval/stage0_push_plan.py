"""Source-bound initialization hashes and matched, token-normalized loss batches."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.grpo_algorithm import group_advantages
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_distance_math import prepare_answers
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_push_math import a0_hash
from pilot_eval.stage0_sources import factors, inside
from pilot_eval.stage0_spectral import _load, _verify_cache
from pilot_eval.stage0_cohort_source import load_cohort
from pilot_eval.stage0_numeric import dependency_fingerprint
from pilot_eval.stage0_workflow import _writer, _state, _verify_parents
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


def matching_execution_kinds(inventory_kind,run_kind):
    """A CPU-only weight audit feeds CUDA measurement; fixtures stay separate."""
    return {'cpu-local-pinned-weights':'real-cuda','synthetic-cpu-fixture':'synthetic-cpu-fixture'}.get(inventory_kind)==run_kind


def batches(training,draws,datasets,cases):
    """Identical CE problem batches; GRPO keeps all groups, including dead ones."""
    by_id={r['id']:r for r in training};draw_map={(r['group_id'],r['draw_index']):r for r in draws}
    selected=dict(datasets['own_correct_one'])
    if not selected:raise ValueError('No eligible own-correct questions; matched CE comparison undefined.')
    eligible=[r['id'] for r in training if r['id'] in selected]
    rng=np.random.default_rng(42);eligible=[eligible[i] for i in rng.permutation(len(eligible))]
    all_ids=[r['id'] for r in training];all_ids=[all_ids[i] for i in np.random.default_rng(42).permutation(len(all_ids))]
    result=[]
    for case in cases:
        for objective in ['gold','own_correct_one','grpo']:
            order=all_ids if objective=='grpo' else eligible
            for start in range(0,len(order),8):
                ids=order[start:start+8];sequences=[];dead=0
                for identifier in ids:
                    row=by_id[identifier];prompt=row['input_ids'][:row['prompt_tokens']]
                    if objective=='grpo':
                        rows=[draw_map[identifier,d] for d in range(8)]
                        rewards=[0 if r['capped'] else (r['historical_training_reward'] if r['historical_training_reward'] is not None else r['reward_v3']) for r in rows]
                        advantages=group_advantages(rewards);dead+=int(not any(advantages))
                        targets=[r['generated_token_ids'] for r in rows]
                    else:
                        targets=[row['input_ids'][row['prompt_tokens']:]] if objective=='gold' else [draw_map[identifier,selected[identifier]]['generated_token_ids']]
                        advantages=[1.]
                    for target,advantage in zip(targets,advantages):
                        sequences.append(dict(problem_id=identifier,input_ids=prompt+target,target_ids=target,
                            positions=list(range(len(prompt)-1,len(prompt)+len(target)-1)),advantage=advantage))
                result.append(dict(key=f'{case}-{objective}-{start//8:03d}',case=case,objective=objective,
                    problem_ids=ids,question_count=len(ids),partial=len(ids)<8,sequences=sequences,
                    token_count=sum(len(s['target_ids']) for s in sequences),dead_groups=dead))
    return result


def implementation():
    folder=Path(__file__).parent
    # Whole numerical dependency chain, including the model loader (ADR 0016 review).
    return dependency_fingerprint(['stage0_push_plan.py','stage0_push_math.py','stage0_push_backend.py','stage0_pushes.py','stage0_push_report.py',
        'grpo_algorithm.py','stage0_evaluation_backend.py','stage0_cohort_source.py','stage0_matched.py','stage0_numeric.py',
        'stage0_rollout_plan.py','stage0_rollout_results.py'],folder)


def prepare(spectral_path,rollouts_path,output_root,name,*,context_chunk=16):
    root=Path(output_root).resolve();safe_name(name)
    sp,rp=inside(root,spectral_path),inside(root,rollouts_path)
    spectral,audit=_load(sp,root);_verify_cache(spectral,inside(root,spectral['run_path']))
    cohort=load_cohort(rp,root);rollout,source,training=cohort['config'],cohort['directory'],cohort['training']
    if spectral['audit_path']!=rollout['audit_path'] or spectral['weights_dir']!=rollout['weights_dir'] or not matching_execution_kinds(spectral['inventory']['execution_kind'],rollout['execution_kind']):
        raise ValueError('Spectral and R0 sources must use the same audited base and artifact inventory.')
    if type(context_chunk) is not int or context_chunk<1:raise ValueError('Positive context chunk required.')
    draws,datasets,special=cohort['draws'],cohort['datasets'],cohort['special']
    prepare_answers(training,draws,datasets,special['eos_token_id'])  # independent existing mask/EOS validation
    initial={arm:factors(inside(root,audit['sources'][arm]['checkpoints']['0']['path'])) for arm in ['sft','grpo']}
    hashes={arm:a0_hash(values) for arm,values in initial.items()}
    if (hashes['sft']==hashes['grpo'])!=audit['initialization']['equal_a']:raise ValueError('A0 payload hashes contradict the source audit.')
    cases={}
    for arm,digest in hashes.items():
        cases.setdefault(digest,dict(initialization_arm=arm,paired_arms=[],a0_sha256=digest))['paired_arms'].append(arm)
    cases={f'a0-{n}':v for n,v in enumerate(cases.values())};units=batches(training,draws,datasets,cases)
    parents={str(p.relative_to(root)):file_hash(p) for p in [sp,rp,cohort['complete_path']]}
    directory=root/f'runs/stage0/{name}/e05_pushes';path=root/'plans'/name/'stage0.pushes.json'
    variants={v['initialization_arm']:dict(adapter_path=audit['sources'][v['initialization_arm']]['checkpoints']['0']['path']) for v in cases.values()}
    config=dict(protocol='stage0-pushes-v1',schema_version=1,name=name,run_path=str(directory.relative_to(root)),
        spectral_path=str(sp.relative_to(root)),rollouts_path=str(rp.relative_to(root)),parents=parents,implementation=implementation(),numpy_version=np.__version__,
        weights_dir=rollout['weights_dir'],model_identity=rollout['model_identity'],execution_kind=rollout['execution_kind'],
        sampled_runtime=cohort['sampled_runtime'],cohort=cohort['cohort'],
        gradient_runtime_rule='every runtime field except git_commit must match the frozen first gradient session; commit recorded per unit (ADR 0016)',
        reconstruction_rule='norm(GPU FP32 - FP64 hook reconstruction) <= max(1e-5 relative, a-priori FP32 accumulation bound) (ADR 0016)',
        gradient_runtime_policy='freeze first bounded gradient session; all paired objectives and continuations must match it; sampled tokens are immutable source data',
        variants=variants,variant_order=list(variants),seed=42,prompt_contract=rollout['legacy']['prompt_template'],
        decoding=dict(eos_token_id=special['eos_token_id'],pad_token_id=special['pad_token_id'],max_new_tokens=1024),
        cases=cases,initialization_audit=dict(a0_sha256=hashes,identical=hashes['sft']==hashes['grpo']),
        module_shapes={name:dict(a=list(initial['sft'][name][0].shape),b=list(initial['sft'][name][1].shape)) for name in spectral['inventory']['modules']},
        units_sha256=_hash(units),unit_keys=[u['key'] for u in units],modules=list(spectral['inventory']['modules']),
        context_chunk=context_chunk,microbatch_sequences=1,question_batch=8,
        objectives=['gold','own_correct_one','grpo'],reconstruction_gate=1e-5,directional_gate=1e-3,finite_difference_epsilon=1e-4,
        reductions='sum loss-bearing tokens / whole question-batch completion-token count; no extra accumulation division',
        grpo=f'initial-policy DAPO gradient; detached sample-SD(ddof1)+1e-4 advantages, cap reward zero; all {len(training)} groups of this cohort; no optimizer',
        reward='recorded historical training reward where present; otherwise frozen flexible-v3 binary reward',
        primary='batch mean push=-gradient; with and without final partial; consistency excludes partial',
        secondary='token-weighted pooled push',comparison='same A0 hash only; direction only across objectives',
        endpoint='orient paired a64,b64 so a64 dot A0>0; zero dot undefined; report flips',
        cohorts=dict(eligible_questions=len(datasets['own_correct_one']),grpo_questions=len(training),dropped=len(datasets['dropped'])),budget_ceiling_units=20)
    with _writer(directory):
        _save_frozen(path,config);_save_frozen(directory/'config.json',config)
        _jsonl(directory/'inputs/batches.jsonl',units);_save_frozen(directory/'inputs/initialization-audit.json',config['initialization_audit'])
        _save_frozen(directory/'meta/run_manifest.json',dict(config_sha256=_hash(config),resources='bounded GPU backward/profile; CPU summaries and explicit endpoint reveal',unit_count=len(units)))
        if not (directory/'meta/status.json').exists():_state(directory,'prepared',0,len(units))
    return path


def load(path,root):
    config=json.loads(inside(root,path).read_text());directory=inside(root,config['run_path'])
    # NumPy version is recorded provenance; CPU and GPU runtimes may differ (ADR 0016).
    if config.get('protocol')!='stage0-pushes-v1' or config['implementation']!=implementation():raise ValueError('Push implementation changed; preserve evidence and use a new plan.')
    if json.loads((directory/'config.json').read_text())!=config or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(config):raise ValueError('Push frozen identity changed.')
    _verify_parents(root,config['parents']);spectral,_audit=_load(inside(root,config['spectral_path']),root)
    _verify_cache(spectral,inside(root,spectral['run_path']));load_cohort(inside(root,config['rollouts_path']),root)
    units=[json.loads(x) for x in (directory/'inputs/batches.jsonl').read_text().splitlines()]
    if _hash(units)!=config['units_sha256'] or [u['key'] for u in units]!=config['unit_keys'] or json.loads((directory/'inputs/initialization-audit.json').read_text())!=config['initialization_audit']:raise ValueError('Push input/A0 audit changed.')
    return config,directory,units
