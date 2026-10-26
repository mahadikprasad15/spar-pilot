"""Global conservative GPU reservations, reconciled against cumulative observations.

A live or interrupted reservation stays charged until a newer observed total
is recorded. One GPU reservation at a time avoids overlapping notebook jobs.
"""
import json
import math
import os
import uuid

from pilot_eval.stage0_workflow import CEILING, _now, _writer, budget_status
from pilot_eval.run import _write_json
from pilot_eval.workflow import _hash


def _directory(root):return root/'runs/stage0/budget'


def reservations(root):
    folder=_directory(root);path=folder/'reservations.jsonl';head=folder/'reservations-head.json'
    records=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    previous=None;states={}
    for row in records:
        value={k:v for k,v in row.items() if k!='sha256'}
        if value['previous_sha256']!=previous or row['sha256']!=_hash(value):
            raise ValueError('Reservation ledger integrity failure.')
        previous=row['sha256'];states[row['reservation_id']]=row
    if head.exists() and json.loads(head.read_text()).get('sha256')!=previous:
        raise ValueError('Reservation ledger/head disagreement; inspect interrupted publication.')
    if records and not head.exists():raise ValueError('Reservation ledger head missing.')
    return states


def _append(root,value):
    folder=_directory(root);path=folder/'reservations.jsonl'
    previous=json.loads(path.read_text().splitlines()[-1])['sha256'] if path.exists() and path.stat().st_size else None
    row=dict(value,previous_sha256=previous);row['sha256']=_hash(row)
    with path.open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
    _write_json(folder/'reservations-head.json',dict(sha256=row['sha256']))
    return row


def reserve(root,job_sha256,approval):
    required={'reviewer','notes','session_id','max_units_per_hour','max_seconds','source_reviewed'}
    if (set(approval)!=required or not approval['source_reviewed'] or approval['source_reviewed'] is not True
            or any(not isinstance(approval[k],str) or not approval[k].strip() for k in ['reviewer','notes','session_id'])
            or any(type(approval[k]) not in [int,float] or not math.isfinite(approval[k]) or approval[k]<=0
                   for k in ['max_units_per_hour','max_seconds'])):
        raise ValueError('Approve source review, session, bounded wall time and conservative unit-rate ceiling.')
    with _writer(_directory(root)):
        usage=budget_status(root);states=reservations(root)
        if usage['state'] not in ['observed','ceiling-reached']:raise ValueError('Record a current cumulative usage observation first.')
        if any(r['state']=='active' for r in states.values()):raise ValueError('An active reservation needs reconciliation before another GPU job.')
        amount=approval['max_units_per_hour']*approval['max_seconds']/3600
        if usage['spent_units']+amount>CEILING:raise ValueError('Reservation exceeds cumulative 20-unit ceiling.')
        return _append(root,dict(reservation_id=uuid.uuid4().hex,state='active',job_sha256=job_sha256,
            approval=approval,created_at=_now(),reserved_units=amount,starting_units=usage['spent_units'],
            starting_ledger_sha256=usage['last_sha256']))


def settle(root,reservation_id):
    with _writer(_directory(root)):
        row=reservations(root).get(reservation_id)
        if not row:raise ValueError('Unknown reservation.')
        if row['state']=='settled':return row
        usage=budget_status(root)
        if usage['state'] not in ['observed','ceiling-reached'] or usage['last_sha256']==row['starting_ledger_sha256']:
            raise ValueError('A newer cumulative observation is required to release a reservation.')
        latest=json.loads((_directory(root)/'ledger.jsonl').read_text().splitlines()[-1])
        if latest['session_id']!=row['approval']['session_id']:
            raise ValueError('Reconcile the reservation session explicitly.')
        return _append(root,{k:v for k,v in dict(row,state='settled',settled_at=_now(),
            ending_ledger_sha256=usage['last_sha256'],observed_units=usage['spent_units']-row['starting_units']).items()
            if k not in ['sha256','previous_sha256']})
