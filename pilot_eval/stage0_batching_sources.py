"""Validate the payloads used by batching; unrelated spectral caches are not inputs."""
import json
from pathlib import Path

from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_qualification import _verify as verify_diagnostic
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read JSON artifact {path}: {exc}') from exc


def same_session(a, b):
    # A measured, same-session batching check may follow an orchestration fix.
    # Forward instrumentation hashes are checked separately; retain both commits.
    return {k:v for k,v in a.items() if k!='git_commit'} == {k:v for k,v in b.items() if k!='git_commit'}


def sources(root, qualification_path, profile_path, batch_size):
    root=Path(root).resolve();qpath=inside(root,qualification_path);profile=inside(root,profile_path)
    q=read(qpath);qdir=inside(root,q['run_path'])
    if q.get('protocol')!='stage0-loader-qualification-v1' or q.get('qualification_kind'):
        raise ValueError('Batching applies to the primary ten-condition catalogue only.')
    for name in ['stage0_qualification_backend.py','stage0_qualification_math.py','stage0_budget.py']:
        if file_hash(Path(__file__).parent/name)!=q['implementation'].get(name):
            raise ValueError(f'Original qualification instrumentation changed: {name}')
    report=verify_diagnostic(q,qdir)
    if not report['eligible_for_review']:
        raise ValueError('An eligible completed numerical diagnostic is required.')
    if (q['hard_error_stop']!=1e-3 or q['allowance_multiplier']!=3
            or q['negative_error_multiplier']!=1000 or q['budget_ceiling_units']!=20):
        raise ValueError('Original numerical/budget policy differs from the approved protocol.')
    for key,digest in [('cohort_path','cohort_sha256'),('edits_path','edits_sha256')]:
        if file_hash(inside(root,q[key]))!=q[digest]:raise ValueError(f'Changed source: {q[key]}')
    for name,digest in q['tokenizer_files'].items():
        if file_hash(inside(root,q['weights_dir'])/name)!=digest:
            raise ValueError(f'Changed tokenizer: {name}')
    edits=read(inside(root,q['edits_path']));edir=inside(root,edits['run_path']);seal=read(edir/'complete.json')
    if seal.get('config_sha256')!=_hash(edits):raise ValueError('Edit catalogue identity changed.')
    catalogue=edir/'results/catalogue.json'
    if file_hash(catalogue)!=seal['files']['results/catalogue.json'] or read(catalogue)['variants']!=q['variants']:
        raise ValueError('Assembled adapter catalogue changed.')
    for variant,row in q['variants'].items():
        folder=inside(root,row['adapter_path']);receipt=folder/'complete.json'
        if file_hash(receipt)!=seal['files'][f'adapters/{variant}/complete.json']:
            raise ValueError(f'Adapter receipt changed: {variant}')
        adapter=read(receipt)
        if adapter['identity']!=dict(config_sha256=_hash(edits),variant=variant):
            raise ValueError(f'Adapter identity changed: {variant}')
        for name,digest in adapter['files'].items():
            if file_hash(inside(folder,name))!=digest:raise ValueError(f'Adapter payload changed: {variant}/{name}')
    pconfig=read(profile/'config.json');pseal=read(profile/'complete.json')
    if pseal['config_sha256']!=_hash(pconfig):raise ValueError('Capacity-profile config changed.')
    for name,digest in pseal['files'].items():
        if file_hash(inside(profile,name))!=digest:raise ValueError(f'Capacity-profile payload changed: {name}')
    measurements=read(profile/'summary.json')
    measured=next((r for r in measurements if r['batch_size']==batch_size),None)
    if (not measured or measured['memory']['fits'] is not True
            or measured['exact_singleton_token_agreement'] is not True
            or not same_session(pconfig['runtime'],report['runtime'])
            or pconfig['qualification_sha256']!=file_hash(qpath)
            or pconfig['variants']!=q['variants']
            or pconfig['decoding']!={**q['decoding'],'use_cache':True}):
        raise ValueError('Use the successful same-session capacity/speed profile for this batch size.')
    cohort=read(inside(root,q['cohort_path']));items=cohort['items'] if isinstance(cohort,dict) else cohort
    if len(items)!=150 or len({r['id'] for r in items})!=150:
        raise ValueError('Expected the original 150-question cohort.')
    ids=q['sequence_ids']
    if len(ids)!=16 or [r['id'] for r in items[:16]]!=ids:
        raise ValueError('Use the original fixed sixteen qualification questions.')
    if not q['fixture']:
        from pilot_eval.stage0_qualification import PILOT4_COHORT_SHA256
        if q['cohort_sha256']!=PILOT4_COHORT_SHA256:raise ValueError('Unexpected real evaluation cohort.')
    token_rows=read(qdir/'meta/tokenized-inputs.json')['sequences']
    return q,report,qdir,items,token_rows
