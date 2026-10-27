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
    with _writer(directory):
        selected=dict(audit_path=str(inside(root,audit_path).relative_to(root)),baseline_path=str(inside(root,baseline_path).relative_to(root)))
        _save_frozen(directory/'inputs/selection.json',selected)
        source=read_sources(audit_path,baseline_path,root)
        report={k:v for k,v in source.items() if k not in ['training','historical','source_audit']}
        _write_json(directory/'inputs/source-audit.json',report)
        _state(directory,source['state'],source['verified_historical_groups'],128)
        return report
