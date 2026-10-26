"""Independent array-level merged-loader evidence; no model dependencies."""
import numpy as np


def compare_logits(reference, merged, negative, positions):
    arrays=[np.asarray(x,dtype=np.float64) for x in [reference,merged,negative]]
    if (any(x.shape!=arrays[0].shape or not np.isfinite(x).all() for x in arrays)
            or arrays[0].ndim!=2 or not arrays[0].size or len(positions)!=len(arrays[0])):
        raise ValueError('Logits must be finite, equally shaped answer-position matrices.')
    real=np.abs(arrays[1]-arrays[0]);bad=np.abs(arrays[2]-arrays[0])
    p,v=np.unravel_index(np.argmax(real),real.shape)
    return dict(max_absolute_error=float(real.max()),negative_max_absolute_error=float(bad.max()),
        position_max_errors=real.max(axis=1).tolist(),positions=list(positions),
        worst=dict(position=int(positions[p]),vocabulary_id=int(v)),
        worst_values=dict(reference=float(arrays[0][p,v]),merged=float(arrays[1][p,v])))


def divergence(reference_tokens,merged_tokens,reference_logits,merged_logits,*,prefix):
    """Logits must be evaluated on the shared prefix at the first divergence."""
    a,b=(np.asarray(x,dtype=np.float64) for x in [reference_logits,merged_logits])
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Common-prefix logits must be finite vocabulary vectors.')
    error=float(np.abs(a-b).max());top=np.sort(a)[-2:];margin=float(top[1]-top[0])
    return dict(exact=reference_tokens==merged_tokens,near_tie=reference_tokens!=merged_tokens and margin<error,
        prefix=list(prefix),local_max_error=error,reference_top2_margin=margin,
        reference_top_id=int(a.argmax()),merged_top_id=int(b.argmax()),
        candidates=[dict(token_id=int(i),reference=float(a[i]),merged=float(b[i]))
                    for i in sorted({int(a.argmax()),int(b.argmax()),int(a.argsort()[-2])})],
        scope='first divergence only; no comparison after contexts diverge')


def assess(rows):
    if not rows:raise ValueError('No diagnostic evidence.')
    error=max(r['teacher']['max_absolute_error'] for r in rows)
    hard=error>1e-3
    sensitivity={arm:dict(real=max(r['teacher']['max_absolute_error'] for r in rows if r['arm']==arm),
        negative=max(r['teacher']['negative_max_absolute_error'] for r in rows if r['arm']==arm))
        for arm in sorted({r['arm'] for r in rows})}
    sensitive=all(v['negative']>0 and v['negative']>=1000*v['real'] for v in sensitivity.values())
    greedy=all(r['greedy']['exact'] or r['greedy']['near_tie'] for r in rows)
    eligible=not hard and sensitive and greedy
    return dict(max_absolute_error=error,hard_stop=hard,negative_control=sensitivity,
        sensitive=sensitive,greedy_acceptable=greedy,eligible_for_review=eligible,qualified=False,
        proposed_absolute_gate=3*error if eligible else None,
        note='Calibration evidence only; separate recorded review required. No independent validation claim.')
