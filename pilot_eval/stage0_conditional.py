"""Verified conditional catalogue contracts shared by qualification and collectors."""
from pathlib import Path

from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash


def binding(path,root):
    from pilot_eval.stage0_intruder_edits import _load,verify
    from pilot_eval.stage0_merged_backend import runner_variants
    path=inside(root,path);config,_,_,_=_load(path,root);catalogue=verify(path,root)
    if catalogue['state']=='not-applicable':
        raise ValueError('Verified catalogue is not-applicable; no extension model execution is allowed.')
    arms=[arm for arm in ['sft','grpo'] if catalogue['arms'][arm]['status']=='applicable']
    variants={k:v for k,v in runner_variants(catalogue).items() if v.get('mode')!='unhalved'}
    diagnostics={arm:catalogue['variants'][f'V6-{arm}-unhalved'] for arm in arms}
    return dict(qualification_kind='conditional-intruders',mixed_catalogue_path=str(path.relative_to(root)),
        mixed_catalogue_sha256=file_hash(path),arms=arms,variants=variants,diagnostic_variants=diagnostics,
        edits_path=config['edits_path'],catalogue_scope=dict(arms=catalogue['arms'],matching=catalogue['matching'],
            limits=catalogue['limits'],variants=catalogue['variants']))


def validate(config,root):
    if config.get('qualification_kind')!='conditional-intruders':return
    path=inside(root,config['mixed_catalogue_path'])
    if file_hash(path)!=config['mixed_catalogue_sha256']:raise ValueError('Conditional catalogue plan changed.')
    expected=binding(path,root)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Conditional catalogue matrices, masks, applicability or references changed.')


def implementation():
    folder=Path(__file__).parent
    return {n:file_hash(folder/n) for n in ['stage0_conditional.py','stage0_merged_backend.py',
        'stage0_mixed_qualification_backend.py','stage0_mixed_drift_backend.py']}


def contrasts(variants):
    """Each edit versus its own original; matched edits versus matched controls."""
    result=[]
    for name,row in variants.items():
        if row.get('representation')!='mixed-full-matrices':continue
        ref=row['source_reference']
        result.append(dict(id=f'{name}-minus-{ref}',edited=name,reference=ref,
            family='intruder-'+row['mode']+'-vs-own-reference'))
    for arm in ['sft','grpo']:
        for suffix in ['matched','ratio']:
            edited,reference=f'V6-{arm}-{suffix}',f'V6c-{arm}-{suffix}'
            if edited in variants and reference in variants:
                if variants[edited]['adapter_disable_modules']!=variants[reference]['adapter_disable_modules']:
                    raise ValueError('Conditional matched module masks differ.')
                result.append(dict(id=f'{edited}-minus-{reference}',edited=edited,reference=reference,
                    family='intruder-'+suffix+'-vs-matched-control'))
    return result


def validate_observation(value,token_sequence=None):
    """Reconcile saved scalar diagnostics; never trust a contradictory gate flag."""
    import math
    teacher=value['teacher'];positions=teacher['positions']
    if token_sequence is not None and positions!=token_sequence['answer_positions']:
        raise ValueError('Diagnostic omits or changes teacher-forced answer positions.')
    if not positions or any(type(i) is not int or i<0 for i in positions) or len(set(positions))!=len(positions):
        raise ValueError('Invalid diagnostic answer position mapping.')
    for maximum,per_position in [('max_absolute_error','position_max_errors'),('negative_max_absolute_error','negative_position_max_errors')]:
        values=teacher[per_position]
        if len(values)!=len(positions) or any(not math.isfinite(v) or v<0 for v in values) or max(values)!=teacher[maximum]:
            raise ValueError('Diagnostic per-position errors contradict the recorded maximum.')
    greedy=value['greedy'];a,b=greedy['reference_token_ids'],greedy['merged_token_ids']
    if any(not isinstance(v,list) or not v or any(type(i) is not int or i<0 for i in v) for v in [a,b]):
        raise ValueError('Greedy token evidence is invalid.')
    exact=a==b
    if greedy['exact'] is not exact or (exact and greedy['near_tie']):
        raise ValueError('Greedy equality claim contradicts saved tokens.')
    if not exact:
        d=greedy.get('divergence');index=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        if (not d or index>=min(len(a),len(b)) or not d['prefix']
                or d['reference_top_id']!=a[index] or d['merged_top_id']!=b[index]
                or (index and d['prefix'][-index:]!=a[:index])):
            raise ValueError('Greedy mismatch lacks evidence at its common prefix.')
        margin,error=d['reference_top2_margin'],d['local_max_error']
        if not all(math.isfinite(v) and v>=0 for v in [margin,error]) or greedy['near_tie'] is not (margin<error):
            raise ValueError('Greedy near-tie claim contradicts measured shared-prefix error/margin.')
