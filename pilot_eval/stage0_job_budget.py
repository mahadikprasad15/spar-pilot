"""Matched-128 job reservations without a cumulative Stage 0 cap (ADR 0017).

The user removed the 20-unit ceiling on 2026-10-30 after the ledger recorded 21 units.
Everything else stays: an observed cumulative reading, a reviewed bounded approval, one
active GPU reservation at a time, per-job wall-time caps and settlement against a newer
reading. Completed Stage 0 products keep the capped `stage0_budget` path unchanged.
"""
import json
import math
import uuid
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.stage0_budget import _append, _directory, reservations
from pilot_eval.stage0_workflow import _now, _writer, budget_status
from pilot_eval.workflow import _hash


POLICY = 'no cumulative Stage 0 cap (ADR 0017); reviewed per-job wall-time caps and usage ledger remain'


def reserve(root,job_sha256,approval):
    required={'reviewer','notes','session_id','max_units_per_hour','max_seconds','source_reviewed'}
    if (set(approval)!=required or approval['source_reviewed'] is not True
            or any(not isinstance(approval[k],str) or not approval[k].strip() for k in ['reviewer','notes','session_id'])
            or any(type(approval[k]) not in [int,float] or not math.isfinite(approval[k]) or approval[k]<=0
                   for k in ['max_units_per_hour','max_seconds'])):
        raise ValueError('Approve source review, session, bounded wall time and conservative unit-rate ceiling.')
    with _writer(_directory(root)):
        usage=budget_status(root);states=reservations(root)
        if usage['state'] not in ['observed','ceiling-reached']:raise ValueError('Record a current cumulative usage observation first.')
        if any(r['state']=='active' for r in states.values()):raise ValueError('An active reservation needs reconciliation before another GPU job.')
        return _append(root,dict(reservation_id=uuid.uuid4().hex,state='active',job_sha256=job_sha256,
            approval=approval,created_at=_now(),reserved_units=approval['max_units_per_hour']*approval['max_seconds']/3600,
            starting_units=usage['spent_units'],starting_ledger_sha256=usage['last_sha256'],budget_policy=POLICY))


def reservation(config,execution,directory,root,runtime,approval_path,attempt_seconds):
    """Same reservation resolution as Stage 0 evaluation, without the cumulative cap."""
    if budget_status(root)['state'] not in ['observed','ceiling-reached']:
        raise ValueError('Record a current cumulative usage observation first.')
    if approval_path is not None:
        approval=json.loads(Path(approval_path).read_text())
        if approval.get('session_id')!=runtime['session_id']:
            raise ValueError('Resume budget approval belongs to another runtime session.')
        row=reserve(root,_hash(config),approval)
    else:
        selected=directory/'meta/active-reservation.json'
        rid=json.loads(selected.read_text())['reservation_id'] if selected.exists() else execution['reservation']['reservation_id']
        row=reservations(root).get(rid)
        if not row or row['state']!='active':
            raise ValueError('No active production reservation; reconcile usage and supply a new bounded --approval.')
        if row['job_sha256'] not in [execution['reservation']['job_sha256'],_hash(config)]:
            raise ValueError('Reservation belongs to another workload.')
    if row['approval']['session_id']!=runtime['session_id']:
        raise ValueError('Production reservation/runtime session differs.')
    _write_json(directory/'meta/active-reservation.json',dict(reservation_id=row['reservation_id']))
    used=0.
    for start in (directory/'attempts').glob('*/start.json'):
        if json.loads(start.read_text())['reservation_id']!=row['reservation_id']:continue
        used+=attempt_seconds(start.parent)['seconds']
    remaining=row['approval']['max_seconds']-used
    if remaining<=0:raise ValueError('Reserved wall time is exhausted; reconcile usage before another approval.')
    return row,remaining
