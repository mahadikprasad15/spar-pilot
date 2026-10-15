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


def validate_analysis_requests(requests):
    """Reject inference requiring data outside the approved retained-data family.

    These are availability checks, not approval of an estimator or primary claim.
    Capture freeze still owns the actual conditions, precision and primary identity.
    """
    if not isinstance(requests, list) or not requests:
        raise ValueError('Declare at least one analysis evidence request.')
    sites = {'embedding-output'} | {f'block-{index}-output' for index in range(28)}
    retained = {f'block-{index}-output' for index in [14, 17, 18, 19, 20, 21]}
    for request in requests:
        if not isinstance(request, dict) or set(request) != {'evidence', 'sites', 'positions'}:
            raise ValueError('Analysis request requires evidence, sites and positions.')
        selected = request['sites']
        positions = request['positions']
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(site, str) or site not in sites for site in selected)):
            raise ValueError('Analysis request has an unsupported named site.')
        if (not isinstance(positions, list) or not positions or
                any(type(position) is not int or not 0 <= position < 128 for position in positions)
                or len(set(positions)) != len(positions)):
            raise ValueError('Declare distinct content-relative positions between 0 and 127.')
        if request['evidence'] == 'example-primary-window':
            if positions != [0, 1, 2, 3, 4]:
                raise ValueError('Per-example all-site evidence has pooled positions 0 through 4 only.')
        elif request['evidence'] == 'example-position':
            if not set(selected).issubset(retained):
                raise ValueError('No retained per-position example evidence at the requested site.')
        elif request['evidence'] != 'cohort-position-mean':
            raise ValueError('Unsupported analysis evidence kind.')
    return requests


def prepare_review(corpus_path, output_root, name):
    """Create an immutable empty decision sheet bound to verified raw inputs.

    This is a draft, not a freeze operation: no statistical settings or approvals
    are inferred. Later reviewed proposals must use separately identified records.
    """
    from pilot_eval.trace_corpus import load_corpus
    from pilot_eval.training import file_hash
    root = Path(output_root).resolve()
    corpus_path = Path(corpus_path)
    corpus_path = (corpus_path if corpus_path.is_absolute() else root / corpus_path).resolve()
    corpus, _ = load_corpus(corpus_path, root)
    directory = _directory(root, name)
    sections = ['vocabulary', 'primary_testing', 'resampling', 'snr', 'robustness', 'grading']
    draft = dict(schema_version=1, state='awaiting-review', reveal_allowed=False,
                 corpus_path=str(corpus_path.relative_to(root)),
                 corpus_sha256=file_hash(corpus_path), corpus_counts=corpus['counts'],
                 choices={section: None for section in sections}, missing_choices=sections,
                 exposure_review='pending',
                 capture_reference='pending capture-contract freeze; no capture settings defined here')
    with run_lock(directory):
        _save_frozen(directory / 'draft.json', draft)
    return draft
