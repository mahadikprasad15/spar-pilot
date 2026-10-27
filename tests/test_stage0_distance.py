"""Teacher-forced answer scoring; independent CPU oracles, public saved workflow."""
import json
from pathlib import Path

import numpy as np
import pytest
from pilot_eval.training import file_hash


def test_full_vocabulary_statistics_match_independent_probability_oracle():
    from pilot_eval.stage0_distance_math import token_statistics
    p=np.array([[.2,.3,.5],[.4,.4,.2]])
    result=token_statistics(np.log(p),np.array([0,1]))
    np.testing.assert_allclose(result['surprise'],-np.log([.2,.4]))
    np.testing.assert_allclose(result['entropy'],-(p*np.log(p)).sum(-1))
    np.testing.assert_array_equal(result['rank'],[3,1])
    with pytest.raises(ValueError):token_statistics(np.log(p),np.array([3,0]))


def test_identical_answers_share_one_forward_but_keep_membership_multiplicity():
    from pilot_eval.stage0_distance_math import prepare_answers
    training=[dict(id='q',prompt='question',gold='#### 2',input_ids=[10,20,99],prompt_tokens=1,labels=[-100,20,99])]
    draws=[dict(group_id='q',draw_index=i,generated_token_ids=[20,99],stop_reason='eos',capped=False,text='#### 2') for i in range(8)]
    memberships=dict(gold=['q'],own_all=[['q',i] for i in range(8)],own_correct=[['q',i] for i in range(8)],own_correct_one=[['q',2]],p_correct={'q':1},neg_log_p_correct={'q':0},dropped=[])
    answers,mapping=prepare_answers(training,draws,memberships,99)
    assert len(answers)==1 and len(mapping['own_all'])==8 and len(mapping['gold'])==1
    assert answers[0]['positions']==[0,1] and answers[0]['target_ids']==[20,99]
    assert len({r['answer_id'] for r in mapping['own_all']})==1
    bad=dict(training[0],labels=[20,20,99])
    with pytest.raises(ValueError,match='mask'):prepare_answers([bad],draws,memberships,99)


def cmd(root,op,*args,dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-distance-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


@pytest.fixture(scope='module')
def rollouts(tmp_path_factory):
    from test_stage0_rollouts import sources,prepare,Sampling,profile_and_freeze,cmd as rcmd,usage
    root=tmp_path_factory.mktemp('distance-r0');sources(root,gold_eos=99);plan=prepare(root);engine=Sampling()
    profile_and_freeze(root,plan,engine)
    assert rcmd(root,'run','--config',plan,'--no-progress',dependencies=engine)==0
    usage(root,'r0-done',4.02)
    return root


class Scores:
    def __init__(self,error=0.,session='session-a'):self.calls=[];self.loads=0;self.error=error;self.session=session
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU')
    def engine(self,config,root):self.loads+=1;return self
    def score(self,answer):
        self.calls.append(answer['answer_id']);n=len(answer['target_ids'])
        return dict(surprise=np.full(n,1.+self.error,dtype=np.float64),entropy=np.ones(n,dtype=np.float64),rank=np.ones(n,dtype=np.int64))
    def check_integrity(self,*,full=False):return dict(base_unchanged=True)
    def close(self):pass


@pytest.fixture
def ready(tmp_path,rollouts):
    import shutil
    shutil.copytree(rollouts,tmp_path,dirs_exist_ok=True)
    assert cmd(tmp_path,'prepare','--rollouts',tmp_path/'plans/r0/stage0.rollouts.json','--name','distance')==0
    return tmp_path


def approval(root):
    from test_stage0_workflow import write_json
    p=root/'distance-budget.json';write_json(p,dict(reviewer='test',notes='bounded scoring fixture',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.));return p



def production(root,plan):
    from test_stage0_rollouts import usage
    from test_stage0_workflow import write_json
    assert cmd(root,'forecast','--config',plan)==0
    directory=root/'runs/stage0/distance/e06a_distance';forecast=json.loads((directory/'results/forecast.json').read_text())
    usage(root,'distance-profile-finished',4.03)
    review=root/'production-review.json'
    write_json(review,dict(reviewer='test',notes='bounded measured scoring production',session_id='session-a',source_reviewed=True,
        max_units_per_hour=1.,max_seconds=max(600.,forecast['conservative_total_seconds']),forecast_sha256=file_hash(directory/'results/forecast.json')))
    assert cmd(root,'freeze','--config',plan,'--review',review)==0

def test_public_unique_scoring_resume_cpu_report_and_corruption_rejection(ready):
    root=ready;plan=root/'plans/distance/stage0.distance.json';directory=root/'runs/stage0/distance/e06a_distance'
    engine=Scores()
    assert cmd(root,'runtime','--config',plan,dependencies=engine)==0
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=engine)==0
    marker=next((directory/'units').glob('*/complete.json'));before=marker.read_bytes()
    production(root,plan)
    assert cmd(root,'run','--config',plan,'--no-progress',dependencies=engine)==0
    assert marker.read_bytes()==before
    assert len(engine.calls)==len(set(engine.calls))==len(json.loads(plan.read_text())['unique_answer_ids'])
    assert len(engine.calls)<4608
    assert cmd(root,'verify-measurement','--config',plan)==0
    assert cmd(root,'report','--config',plan)==0
    assert cmd(root,'verify','--config',plan)==0
    result=json.loads((directory/'results/results.json').read_text())
    assert result['own_all_gate']['passed'] and not result['real_gpu_measurement']
    own=next(r for r in result['summaries'] if r['dataset']=='own_all')
    assert own['n_answers']==4096 and own['n_tokens']==8192 and own['nll']==own['entropy']==1.
    assert own['ci_low']==own['ci_high']==1. and own['neg_log_p_correct'] is None
    assert (directory/'summary.parquet').exists()
    import subprocess,sys
    code='''
import sys
class Reject:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from pilot_eval.stage0_distance_report import verify
assert verify(sys.argv[1],sys.argv[2])['analysis_complete']
'''
    subprocess.run([sys.executable,'-c',code,str(plan),str(root)],check=True)
    next((directory/'units').glob('*/tokens.npz')).write_bytes(b'changed')
    assert cmd(root,'verify','--config',plan)==1


def test_own_all_gate_failure_is_retained_and_does_not_widen_threshold(ready):
    root=ready;plan=root/'plans/distance/stage0.distance.json'
    engine=Scores(error=.03)
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=engine)==0
    production(root,plan)
    assert cmd(root,'run','--config',plan,'--no-progress',dependencies=engine)==0
    assert cmd(root,'report','--config',plan)==1
    directory=root/'runs/stage0/distance/e06a_distance';result=json.loads((directory/'results/results.json').read_text())
    assert not result['own_all_gate']['passed'] and result['own_all_gate']['threshold']==.02
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed-gate'


def test_measured_scoring_freeze_required_before_unbounded_production(ready):
    root=ready;plan=root/'plans/distance/stage0.distance.json';engine=Scores()
    assert cmd(root,'run','--config',plan,'--approval',approval(root),dependencies=engine)==1
    assert engine.loads==0
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=engine)==0
    assert cmd(root,'forecast','--config',plan)==0
    from test_stage0_rollouts import usage
    from test_stage0_workflow import write_json
    usage(root,'scoring-profile-done',4.03)
    directory=root/'runs/stage0/distance/e06a_distance'
    forecast=json.loads((directory/'results/forecast.json').read_text())
    review=root/'scoring-production-review.json'
    write_json(review,dict(reviewer='test',notes='measured unique-answer production',session_id='session-a',source_reviewed=True,
        max_units_per_hour=1.,max_seconds=max(600.,forecast['conservative_total_seconds']),forecast_sha256=file_hash(directory/'results/forecast.json')))
    assert cmd(root,'freeze','--config',plan,'--review',review)==0
    assert cmd(root,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0


def test_sampling_session_and_budget_are_checked_before_scoring(ready):
    root=ready;plan=root/'plans/distance/stage0.distance.json';engine=Scores(session='new-session')
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=engine)==1 and engine.loads==0
    from test_stage0_rollouts import usage
    usage(root,'ceiling',20.)
    engine=Scores()
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=engine)==1 and engine.loads==0


def test_scoring_failed_full_integrity_invalidates_sealed_answer(ready):
    root=ready;plan=root/'plans/distance/stage0.distance.json'
    class Broken(Scores):
        def check_integrity(self,*,full=False):return dict(base_unchanged=not full)
    assert cmd(root,'run','--config',plan,'--approval',approval(root),'--max-new-units',1,dependencies=Broken())==1
    engine=Scores()
    assert cmd(root,'run','--config',plan,'--max-new-units',1,dependencies=engine)==1 and engine.loads==0


def test_notebooks_expose_cpu_report_and_separate_measured_scoring_budget():
    import ast
    for name,required in [('stage-0-cpu-colab.ipynb',['distance-prepare','distance-report','distance-verify']),
                          ('stage-0-gpu-colab.ipynb',['distance-runtime','distance-forecast','distance-freeze','distance-run','budget-settle'])]:
        n=json.loads((Path('notebooks')/name).read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
        for operation in required:assert operation in code
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None


def test_token_weighted_point_estimate_and_problem_bootstrap_have_distinct_units(tmp_path,monkeypatch):
    import pilot_eval.stage0_distance_report as report
    arrays={'a':dict(surprise=np.array([1.]),entropy=np.array([.5]),rank=np.array([1])),
            'b':dict(surprise=np.array([3.,3.,3.]),entropy=np.array([1.,1.,1.]),rank=np.array([1,1,1]))}
    answers=[dict(answer_id='a',target_ids=[2],positions=[0]),dict(answer_id='b',target_ids=[2,2,2],positions=[0,1,2])]
    mapping={k:[dict(problem_id='p1',answer_id='a'),dict(problem_id='p2',answer_id='b')] for k in ['gold','own_all','own_correct','own_correct_one']}
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/datasets.json').write_text(json.dumps(dict(neg_log_p_correct={'p1':.1,'p2':.9})))
    monkeypatch.setattr(report,'read_unit',lambda c,d,a:arrays[a['answer_id']])
    result=report.calculate(dict(statistics={},premise={},execution_kind='synthetic-cpu-fixture'),tmp_path,answers,mapping)
    own=next(r for r in result['summaries'] if r['dataset']=='own_correct')
    assert own['nll']==2.5 and own['entropy']==.875 and own['excess_surprise']==1.625
    assert (own['ci_low'],own['ci_high'])==(1.,3.) # resampling whole problems, not individual tokens
    assert own['neg_log_p_correct']==.5 # equal problem average, whole-answer units
    assert own['excess_surprise_ci_low']==.5 and own['excess_surprise_ci_high']==2.
