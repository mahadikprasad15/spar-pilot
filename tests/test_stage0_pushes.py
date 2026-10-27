"""Initial pushes: independent dense oracles and frozen public CPU/GPU seam."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_initial_rank_one_gradient_and_finite_difference_oracles():
    pytest.importorskip('torch',exc_type=ImportError)
    from pilot_eval.stage0_push_math import instrument_check
    result=instrument_check()
    assert result['passed'] and result['a_gradient_exact_zero']
    assert result['gradient_relative_error']<1e-5
    assert result['directional_relative_error']<1e-3 and result['epsilon']==1e-4
    assert result['microbatch_global_denominator_passed'] and result['dead_group_zero_passed']


def test_push_statistics_partial_batches_and_gauge_orientation():
    from pilot_eval.stage0_push_math import summarize_vectors,orient_endpoint
    gradients=np.array([[-1.,0.],[-1.,0.],[9.,0.]])
    result=summarize_vectors(gradients,[2,4,1],[8,8,2])
    np.testing.assert_allclose(result['mean_push'],[-7/3,0])
    np.testing.assert_allclose(result['mean_push_full_batches'],[1,0])
    np.testing.assert_allclose(result['token_weighted_push'],[-3/7,0])
    assert result['consistency']==1. and result['partial_batches']==1
    endpoint,proof=orient_endpoint(np.array([-2.,0.]),np.array([-3.,1.]),np.array([1.,0.]))
    np.testing.assert_array_equal(endpoint,[3.,-1.]);assert proof['flipped']
    assert proof['oriented_dot_a64_a0']>0 and proof['a64_a0_cosine']==1.
    assert orient_endpoint(np.array([0.,1.]),np.array([3.,1.]),np.array([1.,0.]))[0] is None
    zero=summarize_vectors(np.zeros((2,2)),[1,1],[8,8])
    assert zero['consistency'] is None


def test_matched_cohorts_global_denominators_and_dead_groups():
    from pilot_eval.stage0_push_plan import batches
    training=[dict(id=f'q{i}',input_ids=[10,20,99],prompt_tokens=1,labels=[-100,20,99]) for i in range(10)]
    draws=[dict(group_id=f'q{i}',draw_index=d,generated_token_ids=[30,99],capped=False,
                reward_v3=int(i!=0 and d<4),historical_training_reward=None) for i in range(10) for d in range(8)]
    datasets=dict(own_correct_one=[[f'q{i}',0] for i in range(1,10)])
    units=batches(training,draws,datasets,['a0'])
    gold=[u for u in units if u['objective']=='gold'];own=[u for u in units if u['objective']=='own_correct_one']
    assert [u['problem_ids'] for u in gold]==[u['problem_ids'] for u in own]
    assert [u['question_count'] for u in gold]==[8,1]
    grpo=[u for u in units if u['objective']=='grpo']
    assert sum(u['question_count'] for u in grpo)==10
    assert sum(u['dead_groups'] for u in grpo)==1
    assert all(u['token_count']==sum(len(s['target_ids']) for s in u['sequences']) for u in units)
    assert all(s['advantage']==0 for u in grpo for s in u['sequences'] if s['problem_id']=='q0')


@pytest.mark.parametrize('objective',['gold','grpo'])
def test_real_collector_matches_unchunked_tiny_autograd(objective):
    torch=pytest.importorskip('torch',exc_type=ImportError)
    from types import SimpleNamespace
    from pilot_eval.stage0_push_backend import collect
    torch.manual_seed(42)
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=torch.nn.Linear(3,4,bias=False)
            self.lora_A=torch.nn.Linear(3,1,bias=False);self.lora_B=torch.nn.Linear(1,4,bias=False)
            self.lora_B.weight.data.zero_()
        def forward(self,x):return self.base(x)+self.lora_B(self.lora_A(x))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embed=torch.nn.Embedding(10,3);self.block=Block();self.head=torch.nn.Linear(4,5,bias=False)
        def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.block(self.embed(input_ids)))
    model=Tiny()
    for n,p in model.named_parameters():p.requires_grad_('lora_' in n)
    sequences=[dict(input_ids=[1,2,3],positions=[0,1],target_ids=[2,3],advantage=1.),
               dict(input_ids=[2,3,4],positions=[0,1],target_ids=[1,2],advantage=-.5 if objective=='grpo' else 1.)]
    unit=dict(objective=objective,sequences=sequences,token_count=4)
    arrays,proof=collect(model,model,model.head,{'block':(model.block,model.block.lora_A.weight,model.block.lora_B.weight)},unit,context_chunk=1,device='cpu')
    actual=arrays['gradient_b_0'].copy();model.zero_grad(set_to_none=True)
    loss=0
    for s in sequences:
        hidden=model(torch.tensor([s['input_ids']])).last_hidden_state[0,s['positions']]
        lp=torch.log_softmax(model.head(hidden),-1).gather(1,torch.tensor(s['target_ids'])[:,None]).sum()
        loss-=lp*s['advantage']/4
    loss.backward()
    np.testing.assert_allclose(actual,model.block.lora_B.weight.grad.numpy(),rtol=1e-5,atol=1e-8)
    assert proof['a_gradient_exact_zero'] and proof['optimizer_steps']==0
    np.testing.assert_array_equal(arrays['push_0'],-actual)
    dead=dict(unit,sequences=[dict(s,advantage=0.) for s in sequences])
    arrays,proof=collect(model,model,model.head,{'block':(model.block,model.block.lora_A.weight,model.block.lora_B.weight)},dead,context_chunk=1,device='cpu')
    assert not np.any(arrays['gradient_b_0']) and proof['active_sequences']==0


def cmd(root,op,*args,dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-pushes-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


@pytest.fixture(scope='module')
def parents(tmp_path_factory):
    from test_stage0_rollouts import sources,prepare,Sampling,profile_and_freeze,cmd as rcmd,usage
    from test_stage0_spectral import TinyWeights,MODULES,spectral
    from pilot_eval.cli import main
    root=tmp_path_factory.mktemp('push-parents');sources(root,gold_eos=99,width=64,output=64)
    rp=prepare(root);engine=Sampling();profile_and_freeze(root,rp,engine)
    assert rcmd(root,'run','--config',rp,'--no-progress',dependencies=engine)==0
    usage(root,'r0-finished',4.02)
    np.savez(root/'weights/tiny.npz',**{m:np.eye(64,dtype=np.float32) for m in MODULES})
    assert main(['stage0-spectral-prepare','--audit',str(root/'plans/audit/stage0.plan.json'),'--weights-dir',str(root/'weights'),
                 '--name','basis','--output-root',str(root)],dependencies=TinyWeights())==0
    sp=root/'plans/basis/stage0.spectral.json';assert spectral(root,'run',sp,dependencies=TinyWeights())==0
    return root


class Gradients:
    def __init__(self,fail_after=None,session='session-a'):self.calls=[];self.config=None;self.fail_after=fail_after;self.session=session
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU')
    def engine(self,config,root):self.config=config;return self
    def measure(self,unit):
        if len(self.calls)==self.fail_after:raise RuntimeError('interrupted fixture')
        self.calls.append(unit['key']);arrays={};active=sum(s['advantage']!=0 for s in unit['sequences'])
        for i,_ in enumerate(self.config['modules']):
            g=np.ones((64,1),np.float32)*(-1 if active else 0)
            arrays.update({f'gradient_b_{i}':g,f'push_{i}':-g,f'gradient_a_{i}':np.zeros((1,64),np.float32),f'reconstruction_{i}':g.astype(np.float64)})
        return arrays,dict(a_gradient_exact_zero=True,reconstruction_relative_errors={m:0. for m in self.config['modules']},
            optimizer_steps=0,advantages_detached=True,global_token_denominator=unit['token_count'],active_sequences=active,zero_advantage_sequences=len(unit['sequences'])-active)
    def check_integrity(self,*,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
    def close(self):pass


@pytest.fixture
def ready(tmp_path,parents,monkeypatch):
    import shutil
    from pilot_eval import stage0_pushes
    shutil.copytree(parents,tmp_path,dirs_exist_ok=True)
    assert cmd(tmp_path,'prepare','--spectral',tmp_path/'plans/basis/stage0.spectral.json','--rollouts',tmp_path/'plans/r0/stage0.rollouts.json','--name','pushes')==0
    # The actual CPU Torch oracles are tested separately, without model downloads.
    monkeypatch.setattr(stage0_pushes,'instrument_check',lambda:dict(passed=True,scope='synthetic workflow fixture'))
    assert cmd(tmp_path,'check','--config',tmp_path/'plans/pushes/stage0.pushes.json')==0
    return tmp_path


def test_public_profile_resume_freeze_report_and_changed_payload_rejection(ready):
    from test_stage0_workflow import write_json
    from test_stage0_rollouts import usage
    from pilot_eval.training import file_hash
    root=ready;plan=root/'plans/pushes/stage0.pushes.json';directory=root/'runs/stage0/pushes/e05_pushes'
    assert len(json.loads(plan.read_text())['cases'])==1
    approval=root/'push-budget.json';write_json(approval,dict(reviewer='test',notes='bounded gradient fixture',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    engine=Gradients(fail_after=1)
    assert cmd(root,'run','--config',plan,'--approval',approval,dependencies=engine)==1
    first=next((directory/'units').glob('*/complete.json'));original=first.read_bytes()
    engine.fail_after=None
    assert cmd(root,'run','--config',plan,dependencies=engine)==0
    assert len(engine.calls)==3 and first.read_bytes()==original
    assert cmd(root,'forecast','--config',plan)==0
    forecast=json.loads((directory/'results/forecast.json').read_text());usage(root,'push-profile-completed',4.03)
    review=root/'push-production.json';write_json(review,dict(reviewer='test',notes='measured fixture',session_id='session-a',source_reviewed=True,
        max_units_per_hour=1.,max_seconds=max(600.,forecast['conservative_total_seconds']),forecast_sha256=file_hash(directory/'results/forecast.json')))
    assert cmd(root,'freeze','--config',plan,'--review',review)==0
    assert cmd(root,'run','--config',plan,'--no-progress',dependencies=engine)==0
    assert len(engine.calls)==len(set(engine.calls))
    assert cmd(root,'verify-measurement','--config',plan)==0
    assert cmd(root,'report','--config',plan)==0
    summary=json.loads((directory/'results/summary.json').read_text())
    assert not summary['real_gpu_measurement'] and summary['endpoint_revealed'] is False
    assert all(r['cross_objective_norms_comparable'] is False for r in summary['modules'])
    predictions=root/'predictions.json';write_json(predictions,dict(reviewer='test',notes='freeze after pushes',predictions='supporting fixture prediction',prior_exposure=False))
    assert cmd(root,'predict','--config',plan,'--review',predictions)==0
    assert cmd(root,'reveal','--config',plan)==0
    assert cmd(root,'verify','--config',plan)==0
    reveal=json.loads((directory/'results/endpoints.json').read_text())
    assert reveal['flipped_modules']==0 and reveal['prediction_status']=='frozen-before-reveal'
    assert cmd(root,'status','--config',plan)==0
    import os,shutil,subprocess,sys
    code='''
import sys
class Reject:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from pilot_eval.stage0_push_report import verify
assert verify(sys.argv[1],sys.argv[2])['analysis_complete']
'''
    subprocess.run([sys.executable,'-c',code,str(plan),str(root)],check=True)
    if os.environ.get('STAGE0_PUSH_VALIDATION_DEST'):
        destination=Path(os.environ['STAGE0_PUSH_VALIDATION_DEST'])
        if destination.exists():raise AssertionError('Retain earlier validation snapshots; select a new destination.')
        shutil.copytree(root,destination)
    next((directory/'units').glob('*/gradients.npz')).write_bytes(b'changed')
    assert cmd(root,'verify','--config',plan)==1


def test_initialization_hash_is_payload_based_and_rejects_nonzero_b():
    from pilot_eval.stage0_push_math import a0_hash
    initial={'a':(np.array([[1.,2.]],np.float32),np.zeros((3,1),np.float32))}
    assert a0_hash(initial)==a0_hash(dict(initial))
    other={'a':(np.array([[1.,3.]],np.float32),np.zeros((3,1),np.float32))}
    assert a0_hash(initial)!=a0_hash(other)
    with pytest.raises(ValueError,match='zero'):a0_hash({'a':(initial['a'][0],np.ones((3,1),np.float32))})


def test_new_comparison_can_start_after_r0_session_but_cannot_mix_gradient_sessions(ready):
    from test_stage0_workflow import write_json
    root=ready;plan=root/'plans/pushes/stage0.pushes.json'
    approval=root/'fresh-gradient-session.json'
    write_json(approval,dict(reviewer='test',notes='new comparison on saved data',session_id='fresh',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    engine=Gradients(session='fresh')
    assert cmd(root,'run','--config',plan,'--approval',approval,'--max-new-units',1,dependencies=engine)==0
    other=Gradients(session='other')
    assert cmd(root,'run','--config',plan,dependencies=other)==1
    assert not other.calls


def test_rejects_source_changes_missing_approval_and_failed_full_integrity(ready):
    from test_stage0_workflow import write_json
    root=ready;plan=root/'plans/pushes/stage0.pushes.json'
    engine=Gradients()
    assert cmd(root,'run','--config',plan,dependencies=engine)==1 and not engine.calls
    approval=root/'bounded.json';write_json(approval,dict(reviewer='test',notes='fail closed test',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    class Bad(Gradients):
        def check_integrity(self,*,full=False):return dict(base_unchanged=not full,adapters_unchanged=True)
    assert cmd(root,'run','--config',plan,'--approval',approval,'--max-new-units',1,dependencies=Bad())==1
    assert cmd(root,'run','--config',plan,dependencies=engine)==1 and not engine.calls
    (root/'weights/tiny.npz').write_bytes(b'changed parent')
    assert cmd(root,'status','--config',plan)==1


def test_notebooks_expose_push_profile_freeze_and_prediction_boundary():
    import ast
    for name,required in [('stage-0-cpu-colab.ipynb',['pushes-prepare','pushes-report','pushes-predict','pushes-reveal','pushes-verify']),
                          ('stage-0-gpu-colab.ipynb',['pushes-check','pushes-runtime','pushes-run','pushes-forecast','pushes-freeze','pushes-verify-measurement','budget-settle'])]:
        n=json.loads((Path('notebooks')/name).read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
        for operation in required:assert operation in code
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None


def test_cpu_source_selection_runs_in_a_fresh_kernel(tmp_path):
    notebook=json.loads(Path('notebooks/stage-0-cpu-colab.ipynb').read_text())
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    namespace={};exec(cells[0].replace("Path('/content').is_dir()",'False'),namespace)
    calls=[];namespace['ARTIFACT_ROOT']=tmp_path;namespace['run_stage0']=lambda *args:calls.append(args)
    exec(next(c for c in cells if 'SELECTION = ' in c),namespace)
    assert calls[0][0]=='prepare' and namespace['SELECTION'].is_file()


def test_cpu_weight_inventory_is_compatible_with_real_cuda_execution():
    from pilot_eval.stage0_push_plan import matching_execution_kinds
    assert matching_execution_kinds('cpu-local-pinned-weights','real-cuda')
    assert matching_execution_kinds('synthetic-cpu-fixture','synthetic-cpu-fixture')
    assert not matching_execution_kinds('synthetic-cpu-fixture','real-cuda')
    assert not matching_execution_kinds('cpu-local-pinned-weights','synthetic-cpu-fixture')
