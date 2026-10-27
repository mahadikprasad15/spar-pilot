"""Saved Stage 0 R0 source audit, bounded sampling and derived answer cohorts."""
import json
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer,_state
from pilot_eval.stage0_rollout_sources import read_sources
from pilot_eval.workflow import _save_frozen


def audit(audit_path,baseline_path,output_root,name):
    root=Path(output_root).resolve();safe_name(name);directory=root/f'runs/stage0/{name}/rollouts'
    existing=root/'plans'/name/'stage0.rollouts.json'
    if existing.exists():return status(existing,root)
    with _writer(directory):
        selected=dict(audit_path=str(inside(root,audit_path).relative_to(root)),baseline_path=str(inside(root,baseline_path).relative_to(root)))
        _save_frozen(directory/'inputs/selection.json',selected)
        source=read_sources(audit_path,baseline_path,root)
        report={k:v for k,v in source.items() if k not in ['training','historical','source_audit']}
        _write_json(directory/'inputs/source-audit.json',report)
        _state(directory,source['state'],source['verified_historical_groups'],128)
        return report


def prepare(*args,**kwargs):
    from pilot_eval.stage0_rollout_plan import prepare as operation
    return operation(*args,**kwargs)


def profile(path,output_root,**kwargs):
    from pilot_eval.stage0_rollout_run import collect
    return collect(path,output_root,'profile',**kwargs)


def profile_verify(path,output_root):
    from pilot_eval.stage0_rollout_plan import load
    from pilot_eval.stage0_rollout_run import profile_verify as operation
    root=Path(output_root).resolve();config,base,training,historical=load(path,root)
    return operation(config,base,{r['id']:r for r in training},historical)


def freeze(*args,**kwargs):
    from pilot_eval.stage0_rollout_run import freeze as operation
    return operation(*args,**kwargs)


def run(path,output_root,**kwargs):
    from pilot_eval.stage0_rollout_run import collect
    return collect(path,output_root,'collection',**kwargs)


def verify(*args,**kwargs):
    from pilot_eval.stage0_rollout_results import verify as operation
    return operation(*args,**kwargs)


def status(path,output_root):
    from pilot_eval.stage0_rollout_plan import load
    from pilot_eval.stage0_workflow import budget_status
    root=Path(output_root).resolve();config,base,training,historical=load(path,root)
    profile_result=profile_verify(path,root) if (base/'profile/complete.json').exists() else None
    result=verify(path,root) if (base/'complete.json').exists() else None
    return dict(measurement_complete=result is not None,execution_kind=config['execution_kind'],
        profile=profile_result,summary=result,completed_groups={phase:len(list((base/phase/'units').glob('*/complete.json'))) for phase in ['profile','collection']},
        budget=budget_status(root),
        report=str(base/'report.html'))


def recover_writer(path,output_root,phase,owner_token,review_notes):
    from pilot_eval.stage0_rollout_plan import load
    from pilot_eval.stage0_evaluation import _recover_owner
    root=Path(output_root).resolve();_,base,_,_=load(path,root)
    return _recover_owner(base/phase,owner_token,review_notes)


def runtime(path,output_root,*,dependencies=None):
    from pilot_eval.stage0_rollout_plan import load,replay_policy
    from pilot_eval.stage0_rollout_backend import RolloutDependencies
    from pilot_eval.stage0_workflow import budget_status
    from pilot_eval.stage0_budget import reservations
    root=Path(output_root).resolve();config,base,_,_=load(path,root);deps=dependencies or RolloutDependencies()
    info=deps.runtime()
    if info['execution_kind']!=config['execution_kind']:raise ValueError('R0 preflight runtime/source kind differs.')
    storage=deps.preflight(config,root) if hasattr(deps,'preflight') else dict(passed=True,scope='synthetic fixture')
    return dict(runtime=info,storage=storage,policy=replay_policy(config,info),budget=budget_status(root),
        active_reservations=[r for r in reservations(root).values() if r['state']=='active'],model_loaded=False,
        note='Any active primary/supporting reservation must be reconciled; this branch cannot spend it.')
