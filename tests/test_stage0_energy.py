"""Base-input energy instrument: independent arithmetic and public saved workflow."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_energy_components_preserve_cancellation_and_distinct_denominators():
    from pilot_eval.stage0_energy_math import energy_sums,ratios
    x=np.array([[1.,-1.],[2.,-2.]])
    a=np.array([1.,1.]);parallel=np.array([1.,0.])
    result=energy_sums(x,a,parallel)
    np.testing.assert_array_equal(result,[5.,5.,-10.,0.])
    r=ratios(result)
    assert r['principal_fraction']==.5
    assert r['parallel_over_total'] is None and r['cross_over_total'] is None
    zero=ratios(np.zeros(4));assert zero['principal_fraction'] is None
    reinforce=energy_sums(np.array([[1.,1.],[2.,2.]]),a,parallel)
    np.testing.assert_array_equal(reinforce,[5.,5.,10.,20.])
    r=ratios(reinforce)
    assert r['principal_fraction']==.5 and r['cross_over_total']==.5
    assert r['parallel_over_total']+r['rest_over_total']+r['cross_over_total']==1.


def test_energy_projection_is_subspace_invariant_and_uses_unshifted_masks():
    from pilot_eval.stage0_energy_math import parallel_factors,view_sums
    q=np.array([[1.,0.],[0.,1.],[0.,0.]])
    rotation=np.array([[0.,-1.],[1.,0.]])
    a=np.array([[1.,2.,3.],[-1.,0.,1.]])
    np.testing.assert_allclose(parallel_factors(a,q),parallel_factors(a,q@rotation))
    x=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
    row=dict(id='q',input_ids=[1,2,3],attention_mask=[1]*3,masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3))
    result=view_sums(x,a,parallel_factors(a,q)[:,None,:],row)
    # factor × cutoff × view × component; the final solution token is counted.
    assert result.shape==(2,1,3,4)
    assert result[0,0,1,3]==(x[2]@a[0])**2
    assert result[0,0,0,3]==(x[1]@a[0])**2
    assert not np.any(result[:,:,2])


def test_projection_checks_are_normalized_and_reject_oblique_basis():
    from pilot_eval.stage0_energy_math import projection_check
    from pilot_eval.stage0_spectral_math import projection
    basis=np.eye(3)[:,:2];a=np.array([1.,2.,3.]);parallel=basis@basis.T@a
    reference=projection(a,basis)
    first=projection_check(a,parallel,basis,reference,dict(atol=1e-5,rtol=1e-5))
    scaled=projection_check(a*1e6,parallel*1e6,basis,reference,dict(atol=1e-5,rtol=1e-5))
    assert first['passed'] and scaled['passed'] and first['norm_partition']==scaled['norm_partition']
    with pytest.raises(ValueError,match='projection'):projection_check(a,parallel,basis*2,reference,dict(atol=1e-5,rtol=1e-5))


def test_actual_tiny_hooks_share_one_disabled_adapter_forward_per_sequence():
    torch=pytest.importorskip('torch',exc_type=ImportError)
    from contextlib import contextmanager
    from types import SimpleNamespace
    from pilot_eval.stage0_energy_backend import capture_body
    from pilot_eval.stage0_energy_math import view_sums
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__();self.first=torch.nn.Linear(3,3,bias=False);self.second=torch.nn.Linear(3,3,bias=False)
            self.calls=0;self.disabled=False;self.seen=[]
        @contextmanager
        def disable_adapter(self):
            self.disabled=True
            try:yield
            finally:self.disabled=False
        def forward(self,input_ids,**kwargs):
            assert self.disabled;self.calls+=1
            x=input_ids.float()[...,None]*torch.tensor([1.,2.,3.]);y=self.first(x)
            self.seen.append([x[0].numpy(),y[0].numpy()]);return SimpleNamespace(last_hidden_state=self.second(y))
    model=Body();a=np.array([[1.,2.,3.],[-1.,0.,1.]])
    p=np.stack([np.array([[1.,2.,0.],[-1.,0.,0.]]),a],axis=1)
    tensors=[(torch.tensor(a),torch.tensor(p))]*2
    row=dict(id='q',input_ids=[1,2,3],attention_mask=[1]*3,masks=dict(question=[True,True,False],solution=[False,False,True],user=[False]*3))
    arrays,proof=capture_body(model,model,[model.first,model.second],tensors,[row,row],device='cpu',context_limit=100)
    assert model.calls==2 and proof['sequence_forwards']==2 and not model.disabled
    for ei in range(2):
        for mi in range(2):np.testing.assert_allclose(arrays['sums'][ei,:,:,:,mi],view_sums(model.seen[ei][mi],a,p,row),atol=1e-10)


def command(root,op,*args,dependencies=None):
    from pilot_eval.cli import main
    return main(['stage0-energy-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


@pytest.fixture
def source(tmp_path,frozen_root):
    # Shared qualification fixture is declared by the existing evaluation tests.
    import shutil
    from test_stage0_workflow import write_json
    shutil.copytree(frozen_root,tmp_path,dirs_exist_ok=True)
    rows=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3) if corpus=='gsm8k' else dict(question=[False]*3,solution=[False]*3,user=[True]*3)
            rows.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    from pilot_eval.workflow import _hash
    write_json(tmp_path/'fixed/items.json',rows)
    write_json(tmp_path/'fixed/activation.prepared.json',dict(items_path='fixed/items.json',items_sha256=_hash(rows),model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32'))
    random=json.loads((tmp_path/'inputs/grpo/activation.prepared.json').read_text())
    random['source_contract']['kind']='random-rank1-control'
    path=tmp_path/'random/activation.prepared.json';write_json(path,random)
    from pilot_eval.training import file_hash
    write_json(path.parent/'prepare-complete.json',dict(config_sha256=_hash(random),files={str(path.relative_to(tmp_path)):file_hash(path)}))
    # The greedy allocation is independent and must be reconciled first.
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import settle
    observation=tmp_path/'usage.json';write_json(observation,dict(event_id='energy-before',session_id='session-a',total_units=1.03,observed_at='2026-10-27T12:00:00+00:00',note='synthetic cumulative'))
    record_budget(tmp_path/'plans/audit/stage0.plan.json',tmp_path,observation)
    execution=json.loads((tmp_path/'plans/qualify/stage0.execution.json').read_text());settle(tmp_path,execution['reservation']['reservation_id'])
    assert command(tmp_path,'prepare','--spectral',tmp_path/'plans/basis/stage0.spectral.json',
        '--qualification',tmp_path/'plans/qualify/stage0.qualification.json','--inputs',tmp_path/'fixed/activation.prepared.json',
        '--random',path,'--name','energy')==0
    return tmp_path


from test_stage0_evaluation import frozen_root


class TinyEnergy:
    def __init__(self,fail_after=None,session='session-a'):self.calls=[];self.loads=0;self.fail_after=fail_after;self.session=session
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU')
    def engine(self,config,root):self.config=config;self.root=root;self.loads+=1;return self
    def measure(self,rows):
        if len(self.calls)==self.fail_after:raise RuntimeError('interrupted fixture')
        from pilot_eval.stage0_energy_math import view_sums
        outputs=[]
        for row in rows:
            self.calls.append(row['id']);sums=[]
            for i,module in enumerate(self.config['modules']):
                with np.load(self.root/self.config['run_path']/self.config['factor_files'][str(i)]['path'],allow_pickle=False) as arrays:
                    a,p=arrays['a'],arrays['parallel']
                    x=np.array(row['input_ids'],float)[:,None]*np.arange(1,a.shape[1]+1)[None,:]
                    sums.append(view_sums(x,a,p,row))
            outputs.append(np.stack(sums,axis=-2))
        return dict(sums=np.stack(outputs)),dict(base_input_scope=True,sequence_forwards=len(rows),all_factors_shared_inputs=True,hook_calls_per_module=len(rows))
    def check_integrity(self,*,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
    def close(self):pass


def test_public_energy_profile_freeze_resume_report_and_corruption(source):
    from test_stage0_workflow import write_json
    from pilot_eval.training import file_hash
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import settle
    root=source;plan=root/'plans/energy/stage0.energy.json';directory=root/'runs/stage0/energy/input-energy'
    config=json.loads(plan.read_text());assert len(config['factor_labels'])==5 and len(config['units'])==38
    review=root/'predictions.json';write_json(review,dict(reviewer='test',notes='fixture only',predictions='fixture preregistration',prior_exposure=False))
    assert command(root,'predict','--config',plan,'--review',review)==0
    approval=root/'energy-budget.json';write_json(approval,dict(reviewer='test',notes='bounded profile fixture',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    engine=TinyEnergy()
    assert command(root,'run','--config',plan,'--approval',approval,'--max-new-units',1,dependencies=engine)==0
    first=next((directory/'units').glob('*/complete.json'));before=first.read_bytes()
    assert command(root,'run','--config',plan,dependencies=engine)==0
    assert len(engine.calls)==16 and first.read_bytes()==before
    assert command(root,'forecast','--config',plan)==0
    observation=root/'usage-profile.json';write_json(observation,dict(event_id='energy-profile-finished',session_id='session-a',total_units=1.04,observed_at='2026-10-27T13:00:00+00:00',note='synthetic cumulative'))
    record_budget(root/'plans/audit/stage0.plan.json',root,observation)
    reservation=json.loads((directory/'meta/active-reservation.json').read_text())['reservation_id'];settle(root,reservation)
    forecast=json.loads((directory/'results/forecast.json').read_text())
    production=root/'production.json';write_json(production,dict(reviewer='test',notes='measured production fixture',session_id='session-a',source_reviewed=True,
        max_units_per_hour=1.,max_seconds=max(600.,forecast['conservative_total_seconds']),forecast_sha256=file_hash(directory/'results/forecast.json'),
        numerical_reviewed=True,agreement=config['agreement']))
    assert command(root,'freeze','--config',plan,'--review',production)==0
    engine.fail_after=len(engine.calls)+8
    assert command(root,'run','--config',plan,dependencies=engine)==1
    engine.fail_after=None
    assert command(root,'run','--config',plan,dependencies=engine)==0
    assert len(engine.calls)==len(set(engine.calls))==300
    assert command(root,'verify-measurement','--config',plan)==0
    assert command(root,'report','--config',plan)==0
    assert command(root,'verify','--config',plan)==0
    result=json.loads((directory/'results/results.json').read_text())
    assert not result['real_gpu_measurement'] and len(result['cells'])==2*5*3*3
    assert all(0<=r['principal_fraction']<=1 for r in result['cells'])
    assert all(abs(r['parallel_over_total']+r['rest_over_total']+r['cross_over_total']-1)<1e-6 for r in result['cells'])
    assert (directory/'results/groups.csv').is_file() and (directory/'plots/principal-fractions.svg').is_file()
    import os,shutil,subprocess,sys
    subprocess.run([sys.executable,'-c',"import sys;from pilot_eval.stage0_energy_report import verify;assert verify(sys.argv[1],sys.argv[2])['analysis_complete'];assert not any(m in sys.modules for m in ['torch','transformers','peft'])",str(plan),str(root)],check=True)
    if os.environ.get('STAGE0_ENERGY_VALIDATION_DEST'):
        destination=Path(os.environ['STAGE0_ENERGY_VALIDATION_DEST'])
        if destination.exists():raise AssertionError('Retain old validation snapshots; select a new path.')
        shutil.copytree(root,destination)
    next((directory/'units').glob('*/arrays.npz')).write_bytes(b'changed')
    assert command(root,'verify','--config',plan)==1


def review_and_budget(root):
    from test_stage0_workflow import write_json
    plan=root/'plans/energy/stage0.energy.json'
    review=root/'prediction.json';write_json(review,dict(reviewer='test',notes='fixture',predictions='test',prior_exposure=False))
    assert command(root,'predict','--config',plan,'--review',review)==0
    approval=root/'energy-approval.json';write_json(approval,dict(reviewer='test',notes='bounded fixture',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    return plan,approval


def test_failed_energy_gate_preserves_arrays_and_stops_without_a_seal(source):
    plan,approval=review_and_budget(source)
    class Bad(TinyEnergy):
        def measure(self,rows):
            arrays,proof=super().measure(rows);arrays['sums'][...,3]+=100.
            return arrays,proof
    assert command(source,'run','--config',plan,'--approval',approval,dependencies=Bad())==1
    directory=source/'runs/stage0/energy/input-energy'
    assert list((directory/'diagnostics').glob('*/*/failed-arrays.npz'))
    assert not list((directory/'units').glob('*/complete.json'))
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'


def test_energy_requires_budget_rejects_runtime_and_source_changes(source):
    plan,approval=review_and_budget(source);engine=TinyEnergy()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0
    assert command(source,'run','--config',plan,'--approval',approval,'--max-new-units',1,dependencies=engine)==0
    other=TinyEnergy(session='different-session')
    assert command(source,'run','--config',plan,dependencies=other)==1 and other.loads==0
    factor=next((source/'runs/stage0/energy/input-energy/inputs/factors').glob('*.npz'));factor.write_bytes(b'changed')
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==1


def test_projection_failure_is_saved_before_gpu_loading(source,monkeypatch):
    from pilot_eval import stage0_energy_plan
    def fail(*args,**kwargs):raise ValueError('projection negative control')
    monkeypatch.setattr(stage0_energy_plan,'projection_check',fail)
    assert command(source,'prepare','--spectral',source/'plans/basis/stage0.spectral.json',
        '--qualification',source/'plans/qualify/stage0.qualification.json','--inputs',source/'fixed/activation.prepared.json',
        '--random',source/'random/activation.prepared.json','--name','failed-projection')==1
    failure=source/'runs/stage0/failed-projection/input-energy/diagnostics/projection-failure.json'
    assert failure.is_file() and not (source/'plans/failed-projection/stage0.energy.json').exists()


def test_energy_honors_cumulative_twenty_unit_ceiling(source):
    from test_stage0_workflow import write_json
    from pilot_eval.stage0_workflow import record_budget
    plan,approval=review_and_budget(source)
    observation=source/'at-ceiling.json';write_json(observation,dict(event_id='energy-at-ceiling',session_id='session-a',total_units=20.,observed_at='2026-10-27T18:00:00+00:00',note='synthetic cumulative'))
    record_budget(source/'plans/audit/stage0.plan.json',source,observation)
    engine=TinyEnergy()
    assert command(source,'run','--config',plan,'--approval',approval,dependencies=engine)==1 and engine.loads==0


def test_notebooks_expose_separate_energy_collection_and_cpu_reporting():
    import ast
    for name,required in [('stage-0-cpu-colab.ipynb',['energy-prepare','energy-predict','energy-report','energy-verify']),
                          ('stage-0-gpu-colab.ipynb',['energy-runtime','energy-run','energy-forecast','energy-freeze','energy-verify-measurement','budget-settle'])]:
        n=json.loads((Path('notebooks')/name).read_text());code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
        for op in required:assert op in code
        for cell in n['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']));assert cell['outputs']==[] and cell['execution_count'] is None
        # Both fresh kernels must resolve precisely the same canonical product.
        selections=[line for line in code.splitlines() if line.startswith('ENERGY_NAME =')]
        assert selections and all(line=="ENERGY_NAME = STAGE0_NAME + '-input-energy-v1'" for line in selections)
