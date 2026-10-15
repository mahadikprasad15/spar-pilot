"""Durable prereveal review history. No model imports or learned token outputs.

Exposure records describe what a reviewer saw, not the token lists themselves.
The review barrier covers supported operations, not direct access to raw files.
"""
import json
from pathlib import Path

from pilot_eval.sft import safe_name
from pilot_eval.training import run_lock
from pilot_eval.workflow import _hash, _save_frozen


def _directory(root, name):
    safe_name(name)
    return Path(root).resolve() / 'runs' / 'pilot-5' / name / 'review'


def record_exposure(output_root, name, event):
    """Append an immutable, explicitly identified reviewer exposure event.

    Reusing the same event ID is idempotent only when its complete content agrees.
    Unknown dates and unknown prior history must be stated, rather than inferred.
    """
    fields = {'schema_version', 'id', 'reviewer', 'experiment', 'output_kind',
              'description', 'observed_at', 'unknown_history'}
    if not isinstance(event, dict) or set(event) != fields or event['schema_version'] != 1:
        raise ValueError('Exposure requires the complete schema-version-1 record.')
    if not all(isinstance(event[key], str) and event[key].strip() for key in fields - {'schema_version'}):
        raise ValueError('Exposure fields must be nonempty; explicitly record unknown history.')
    safe_name(event['id'])
    if event['output_kind'] not in {'scalar-results', 'learned-lens-token-lists', 'other'}:
        raise ValueError('Distinguish scalar results from learned lens token lists.')
    directory = _directory(output_root, name)
    with run_lock(directory):
        events = directory / 'exposures'
        events.mkdir(exist_ok=True)
        payload = dict(event=event, event_sha256=_hash(event))
        path = events / (event['id'] + '.json')
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError('Exposure ID already has different content; append a new event instead.')
        _save_frozen(path, payload)
    return inspect_review(output_root, name)


def inspect_review(output_root, name):
    """Read and verify exposure records without creating files or granting reveal."""
    directory = _directory(output_root, name)
    events = []
    for path in sorted((directory / 'exposures').glob('*.json')):
        payload = json.loads(path.read_text())
        if payload['event_sha256'] != _hash(payload['event']):
            raise ValueError('Exposure record hash mismatch; inspect before continuing.')
        events.append(payload['event'])
    return dict(schema_version=1, state='awaiting-review', reveal_allowed=False,
                exposures=events,
                learned_lists_exposed=any(event['output_kind'] == 'learned-lens-token-lists'
                                         for event in events),
                scope='supported operations only; direct file access is outside this barrier')
