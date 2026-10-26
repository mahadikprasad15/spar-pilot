"""Evidence gates and public budget/freeze flow; CPU only, no downloads."""
import json

import numpy as np
import pytest


def test_logit_errors_use_every_answer_position_and_negative_control_is_sensitive():
    from pilot_eval.stage0_qualification_math import compare_logits, assess
    reference=np.array([[1.,2.,3.],[4.,5.,6.]])
    merged=reference.copy();merged[1,2]+=2e-6
    negative=reference+.02
    row=compare_logits(reference,merged,negative,[7,8])
    assert row['worst']=={'position':8,'vocabulary_id':2}
    assert row['position_max_errors'][0]==0
    assert row['max_absolute_error']==pytest.approx(2e-6)
    report=assess([dict(arm='sft',teacher=row,greedy={'exact':True,'near_tie':False})])
    assert report['eligible_for_review'] and not report['qualified']
    assert report['proposed_absolute_gate']==pytest.approx(6e-6)
    assert not assess([dict(arm='sft',teacher=compare_logits(reference,reference,reference,[7,8]),
                            greedy={'exact':True,'near_tie':False})])['eligible_for_review']
    with pytest.raises(ValueError,match='finite'):
        compare_logits(reference,merged*np.nan,negative,[7,8])


def test_hard_stop_and_common_prefix_near_tie_rules():
    from pilot_eval.stage0_qualification_math import compare_logits, assess, divergence
    ref=np.array([[2.,2.0000001,0.]])
    other=np.array([[2.0000002,2.,0.]])
    evidence=divergence([1,2],[0,2],ref[0],other[0],prefix=[9])
    assert evidence['near_tie'] and evidence['prefix']==[9]
    assert not divergence([1],[0],np.array([0.,1.]),np.array([1.,0.]),prefix=[9])['near_tie']
    assert not divergence([1],[0],np.array([1.,1.]),np.array([1.,1.]),prefix=[9])['near_tie']
    row=compare_logits(ref,ref+.0011,ref+1,[0])
    report=assess([dict(arm='sft',teacher=row,greedy={'exact':True,'near_tie':False})])
    assert report['hard_stop'] and report['proposed_absolute_gate'] is None


def test_global_reservations_count_interrupted_jobs_and_unknown_cost(tmp_path):
    from pilot_eval.stage0_budget import reserve, settle, reservations
    from pilot_eval.stage0_workflow import record_budget
    from test_stage0_workflow import prepared, write_json
    plan=prepared(tmp_path)
    approval=dict(reviewer='Prasad',notes='bounded diagnostics',session_id='session-a',
                  max_units_per_hour=10.,max_seconds=360.,source_reviewed=True)
    with pytest.raises(ValueError,match='observ'):
        reserve(tmp_path,'job-a',approval)
    obs=tmp_path/'usage.json'
    write_json(obs,dict(event_id='start',session_id='session-a',total_units=19.,
                       observed_at='2026-10-26T10:00:00+00:00',note='cumulative'))
    record_budget(plan,tmp_path,obs)
    receipt=reserve(tmp_path,'job-a',approval)
    assert receipt['reserved_units']==1
    assert reservations(tmp_path)[receipt['reservation_id']]['state']=='active'
    with pytest.raises(ValueError,match='active|ceiling'):
        reserve(tmp_path,'job-b',approval)
    with pytest.raises(ValueError,match='newer'):
        settle(tmp_path,receipt['reservation_id'])
    write_json(obs,dict(event_id='after',session_id='session-a',total_units=19.4,
                       observed_at='2026-10-26T11:00:00+00:00',note='job stopped; cumulative'))
    record_budget(plan,tmp_path,obs)
    settle(tmp_path,receipt['reservation_id'])
    assert reservations(tmp_path)[receipt['reservation_id']]['state']=='settled'
    with pytest.raises(ValueError,match='ceiling'):
        reserve(tmp_path,'job-c',approval)
