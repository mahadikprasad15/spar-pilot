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
    assert row['negative_worst']=={'position':7,'vocabulary_id':0}
    assert row['negative_position_max_errors']==pytest.approx([.02,.02])
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


class TinyDiagnostic:
    def __init__(self,error=2e-6):self.error=error;self.calls=[]
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id='session-a',device='tiny CPU oracle')
    def engine(self,config,root):return self
    def diagnose(self,row,arm,check_deadline):
        check_deadline();self.calls.append((row['id'],arm))
        base=np.array([[1.,2.,3.],[2.,3.,4.]])
        return dict(reference=base,merged=base+self.error,negative=base+.1,positions=[4,5],
            greedy=dict(exact=True,near_tie=False,reference_token_ids=[1,2],merged_token_ids=[1,2],
                        reference_text='1 2',merged_text='1 2'),integrity=True)
    def close(self):pass


def qualification(root):
    from test_stage0_edits import prepared,edits
    from test_stage0_workflow import write_json,command
    from pilot_eval.cli import main
    plan=prepared(root);assert edits(root,'run',plan)==0
    cohort=root/'cohort.json'
    write_json(cohort,dict(items=[dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 2') for i in range(150)]))
    assert main(['stage0-qualification-prepare','--edits',str(plan),'--cohort',str(cohort),
                 '--name','qualify','--output-root',str(root)])==0
    observation=root/'observation.json'
    write_json(observation,dict(event_id='initial',session_id='session-a',total_units=1.,
        observed_at='2026-10-26T10:00:00+00:00',note='Stage 0 cumulative'))
    audit_plan=root/'plans/audit/stage0.plan.json'
    assert command(root,'budget-record','--config',audit_plan,'--observation',observation)==0
    approval=root/'diagnostic-review.json'
    write_json(approval,dict(reviewer='Prasad',notes='bounded',session_id='session-a',
        max_units_per_hour=1.,max_seconds=300.,source_reviewed=True))
    return root/'plans/qualify/stage0.qualification.json',approval


def qcommand(root,op,plan,*args,dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-qualification-'+op,'--config',str(plan),'--output-root',str(root),*map(str,args)],
                dependencies=dependencies)


def test_public_qualification_resume_review_and_freeze_do_not_authorize_real_gpu_from_fixture(tmp_path):
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import reservations
    from pilot_eval.cli import main
    from test_stage0_workflow import write_json
    plan,approval=qualification(tmp_path);engine=TinyDiagnostic()
    preflight=qcommand(tmp_path,'runtime',plan,dependencies=engine)
    assert preflight==0
    assert json.loads((tmp_path/'runs/stage0/qualify/qualification/meta/preflight.json').read_text())['model_loaded'] is False
    assert qcommand(tmp_path,'run',plan,'--approval',approval,'--max-new-units',1,dependencies=engine)==0
    directory=tmp_path/'runs/stage0/qualify/qualification'
    marker=next((directory/'units').glob('*/complete.json'));sealed=marker.read_bytes()
    reservation=next(iter(reservations(tmp_path)))
    usage=tmp_path/'usage.json'
    write_json(usage,dict(event_id='after-interrupt',session_id='session-b',total_units=1.01,
        observed_at='2026-10-26T11:00:00+00:00',note='interrupted diagnostic'))
    record_budget(tmp_path/'plans/audit/stage0.plan.json',tmp_path,usage)
    assert main(['stage0-budget-settle','--output-root',str(tmp_path),'--reservation-id',reservation])==0
    assert qcommand(tmp_path,'run',plan,'--approval',approval,dependencies=engine)==0
    assert len(engine.calls)==32 and marker.read_bytes()==sealed
    assert qcommand(tmp_path,'verify',plan)==0
    report=json.loads((directory/'results/diagnostic.json').read_text())
    assert report['eligible_for_review'] and not report['qualified'] and report['completed_units']==32
    assert report['production_estimate']['sequences']==1500
    token_manifest=directory/'meta/tokenized-inputs.json';token_bytes=token_manifest.read_bytes()
    altered=json.loads(token_bytes);altered['sequences'][0]['prompt_sha256']='changed'
    token_manifest.write_text(json.dumps(altered))
    assert qcommand(tmp_path,'verify',plan)==1
    token_manifest.write_bytes(token_bytes)
    assert qcommand(tmp_path,'verify',plan)==0
    review=tmp_path/'review.json'
    write_json(review,dict(reviewer='Prasad',notes='approve measured evidence',
        diagnostic_sha256=__import__('hashlib').sha256((directory/'complete.json').read_bytes()).hexdigest(),
        absolute_gate=report['proposed_absolute_gate']))
    assert qcommand(tmp_path,'approve',plan,'--review',review)==0
    assert qcommand(tmp_path,'freeze',plan,'--review',review)==1  # budget approval is distinct
    # Interrupted jobs, including the completed second attempt, remain reserved until reconciled.
    reservation=[k for k,v in reservations(tmp_path).items() if v['state']=='active'][0]
    write_json(usage,dict(event_id='after-complete',session_id='session-c',total_units=1.03,
        observed_at='2026-10-26T12:00:00+00:00',note='complete diagnostic'))
    record_budget(tmp_path/'plans/audit/stage0.plan.json',tmp_path,usage)
    assert main(['stage0-budget-settle','--output-root',str(tmp_path),'--reservation-id',reservation])==0
    write_json(review,dict(reviewer='Prasad',notes='approve measured forecast; fixture only',
        diagnostic_sha256=__import__('hashlib').sha256((directory/'complete.json').read_bytes()).hexdigest(),
        max_units_per_hour=1.,max_seconds=report['production_estimate']['conservative_total_seconds'],
        session_id='session-a',source_reviewed=True))
    assert qcommand(tmp_path,'freeze',plan,'--review',review)==0
    frozen=json.loads((tmp_path/'plans/qualify/stage0.execution.json').read_text())
    assert not frozen['real_gpu_authorized'] and not frozen['measurement_complete']
    assert frozen['decoding']['max_new_tokens']==1024 and len(frozen['variants'])==10
    assert qcommand(tmp_path,'run',plan,'--approval',approval,dependencies=engine)==0
    assert len(engine.calls)==32


def test_failed_gate_saved_and_changed_inputs_refused(tmp_path):
    plan,approval=qualification(tmp_path)
    assert qcommand(tmp_path,'run',plan,'--approval',approval,dependencies=TinyDiagnostic(error=.002))==1
    directory=tmp_path/'runs/stage0/qualify/qualification'
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'
    assert len(list((directory/'units').glob('*/complete.json')))==1
    (tmp_path/'cohort.json').write_text('[]')
    assert qcommand(tmp_path,'verify',plan)==1
