"""Fixed-context collection through public commands; CPU fixtures, no downloads."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import _hash
from test_stage0_evaluation import frozen_root
from test_stage0_workflow import write_json


@pytest.fixture
def source(tmp_path,frozen_root):
    shutil.copytree(frozen_root,tmp_path,dirs_exist_ok=True)
    rows=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3) if corpus=='gsm8k' else dict(question=[False]*3,solution=[False]*3,user=[True]*3)
            rows.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    write_json(tmp_path/'fixed/items.json',rows)
    write_json(tmp_path/'fixed/activation.prepared.json',dict(items_path='fixed/items.json',items_sha256=_hash(rows),
        model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32'))
    return tmp_path


def command(root,op,*args,dependencies=None):
    return main(['stage0-drift-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def prepared(root):
    assert command(root,'prepare','--execution',root/'plans/qualify/stage0.execution.json',
        '--inputs',root/'fixed/activation.prepared.json','--batch-size',64,'--batches-per-unit',1,'--name','drift')==0
    return root/'plans/drift/stage0.drift.json'


def test_public_complete_resume_and_cpu_reconstruction(source):
    plan=prepared(source)
    class Tiny:
        loads=0;calls=[]
        def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id='session-a',device='tiny CPU oracle')
        def engine(self,config,root):self.loads+=1;self.config=config;return self
        def measure(self,rows,variant,*,diagnostic):
            self.calls.append((variant,[r['id'] for r in rows]))
            return fixture_measurement(rows,self.config,variant,diagnostic)
        def check_integrity(self,*,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
        def close(self):pass
    budget=source/'drift-budget.json'
    write_json(budget,dict(reviewer='test',notes='bounded synthetic measurement',session_id='session-a',max_units_per_hour=1.,max_seconds=60.,source_reviewed=True))
    # Greedy reservation cannot be spent on a different workload.
    engine=Tiny()
    assert command(source,'run','--config',plan,'--approval',budget,dependencies=engine)==1
    assert engine.loads==0
    from pilot_eval.stage0_budget import settle
    from pilot_eval.stage0_workflow import record_budget
    observation=source/'usage.json'
    write_json(observation,dict(event_id='release-greedy',session_id='session-a',total_units=1.03,observed_at='2026-10-26T12:00:00+00:00',note='cumulative'))
    record_budget(source/'plans/audit/stage0.plan.json',source,observation)
    execution=json.loads((source/'plans/qualify/stage0.execution.json').read_text())
    settle(source,execution['reservation']['reservation_id'])
    assert command(source,'run','--config',plan,'--approval',budget,'--max-new-units',1,dependencies=engine)==0
    directory=source/'runs/stage0/drift/fixed-context'
    seal=next((directory/'units').glob('*/complete.json'));before=seal.read_bytes()
    assert command(source,'status','--config',plan)==0
    assert command(source,'run','--config',plan,dependencies=engine)==0
    assert seal.read_bytes()==before and engine.loads==2
    assert command(source,'verify','--config',plan)==0
    result=json.loads((directory/'results/results.json').read_text())
    assert result['measurement_complete'] and not result['real_gpu_measurement']
    assert result['main_documents_per_variant']==300 and len(result['controls'])==3
    assert result['kl_direction']=='variant||untuned' and result['historical_values_used'] is False
    assert len(result['cells'])==30 # ten variants × three nonempty corpus/view cells
    cell=result['cells'][0]
    assert cell['kl']['documents']==150
    assert 'equal_example_mean' in cell['kl'] and 'token_mean' in cell['kl']
    # Saved NPZ has only scalar KL and write sufficient sums, no vocabulary arrays.
    with np.load(next((directory/'units').glob('*/arrays.npz')),allow_pickle=False) as arrays:
        assert set(arrays.files)=={'kl','write_sums'}
    assert command(source,'run','--config',plan,dependencies=engine)==0 and engine.loads==2
    payload=source/'fixed/items.json';payload.write_text('[]')
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==2


def test_independent_full_vocabulary_kl_mapping_and_dense_write_oracles():
    from pilot_eval.stage0_drift_math import kl_values,write_sums,positions
    p=np.array([[.2,.3,.5],[.1,.1,.8]])
    q=np.array([[.5,.25,.25],[.3,.4,.3]])
    expected=np.sum(p*np.log(p/q),axis=1)
    np.testing.assert_allclose(kl_values(np.log(p),np.log(q)),expected,atol=1e-15)
    assert not np.allclose(kl_values(np.log(q),np.log(p)),expected)
    np.testing.assert_array_equal(kl_values(np.log(p),np.log(p)),[0,0])
    x=np.array([[1.,2.],[-3.,4.]])
    w=np.array([[1.,0.],[0.,2.],[1.,1.]])
    delta=np.array([[.3,-.2],[1.,.5],[-.8,.1]]) # general merged delta, not rank one
    got=write_sums(x@w.T,x@delta.T)
    norm=np.linalg.norm(x@delta.T,axis=1);base=np.linalg.norm(x@w.T,axis=1)
    np.testing.assert_allclose(got,[np.sum(norm**2),np.sum(base**2),np.sum(norm/base)])
    row=dict(id='a',input_ids=[1,2,3],attention_mask=[1,1,1],masks=dict(question=[True,True,False],solution=[False,False,True],user=[False]*3))
    assert positions([row])['predictions']==[dict(example=0,context=0,target=1,view='question',token_id=2),dict(example=0,context=1,target=2,view='solution',token_id=3)]
    assert len(positions([row])['writes'])==3 # write at zero exists; no KL at target zero
    row['masks']['user'][1]=True
    with pytest.raises(ValueError,match='overlap'):positions([row])


def fixture_measurement(rows,config,variant,diagnostic):
    from pilot_eval.stage0_drift_math import positions,SUM_FIELDS
    mapping=positions(rows);zero=variant in config['controls']
    kl=np.zeros(len(mapping['predictions']),dtype=np.float64) if zero else np.full(len(mapping['predictions']),.125,dtype=np.float64)
    sums=np.zeros((len(rows),3,len(config['modules']),3),dtype=np.float64)
    for e,row in enumerate(rows):
        for v,view in enumerate(['question','solution','user']):
            count=sum(row['masks'][view]);sums[e,v,:,0]=0 if zero else count*.25
            sums[e,v,:,1]=count*4;sums[e,v,:,2]=0 if zero else count*.25
    return dict(arrays=dict(kl=kl,write_sums=sums),
        metadata=dict(example_ids=[r['id'] for r in rows],**mapping,modules=config['modules'],variant=variant,
            kl_direction='variant||untuned',write_inputs='adapted-module-input',write_fields=SUM_FIELDS),
        validation=dict(reference_invariant=True,rank1_passed=True,finite=True,mapping_exact=True,
            reconstruction_max_fraction=0.,diagnostic=diagnostic,repeat_passed=True if diagnostic else None,
            chunk_passed=True if diagnostic else None,context_chunk=config['context_chunk'],max_logit_contexts=2,
            workspace_bytes=config['workspace_bytes'],max_abs_log_sum=10.,kl_roundoff_bound=64*np.finfo(float).eps*11))


def release_greedy(root):
    from pilot_eval.stage0_budget import settle
    from pilot_eval.stage0_workflow import record_budget
    path=root/'usage.json'
    write_json(path,dict(event_id='release-greedy',session_id='session-a',total_units=1.03,
        observed_at='2026-10-26T12:00:00+00:00',note='cumulative'))
    record_budget(root/'plans/audit/stage0.plan.json',root,path)
    execution=json.loads((root/'plans/qualify/stage0.execution.json').read_text())
    settle(root,execution['reservation']['reservation_id'])
    review=root/'drift-budget.json'
    write_json(review,dict(reviewer='test',notes='bounded fixture',session_id='session-a',
        max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
    return review


class TinyDrift:
    def __init__(self,*,session='session-a',fail_after=None,corrupt=None):
        self.session=session;self.fail_after=fail_after;self.corrupt=corrupt;self.loads=0;self.calls=[]
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU oracle')
    def engine(self,config,root):self.loads+=1;self.config=config;return self
    def measure(self,rows,variant,*,diagnostic):
        if self.fail_after is not None and len(self.calls)>=self.fail_after:raise RuntimeError('injected interruption')
        self.calls.append((variant,[r['id'] for r in rows]))
        result=fixture_measurement(rows,self.config,variant,diagnostic)
        if self.corrupt:self.corrupt(result)
        return result
    def check_integrity(self,*,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
    def close(self):pass


def test_interrupted_units_resume_and_sealed_corruption_is_rejected(source):
    plan=prepared(source);review=release_greedy(source);engine=TinyDrift(fail_after=1)
    assert command(source,'run','--config',plan,'--approval',review,dependencies=engine)==1
    directory=source/'runs/stage0/drift/fixed-context'
    assert len(list((directory/'units').glob('*/complete.json')))==1
    engine=TinyDrift()
    assert command(source,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    assert engine.calls[0][0]=='sft-zero'
    first=next((directory/'units').glob('*/arrays.npz'));first.write_bytes(b'changed')
    assert command(source,'status','--config',plan)==1
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==1


@pytest.mark.parametrize('failure',['zero','mapping','repeat','chunk','nonfinite'])
def test_invalid_observations_never_seal(source,failure):
    plan=prepared(source);review=release_greedy(source)
    def corrupt(result):
        if failure=='zero':result['arrays']['kl'][0]=.1
        elif failure=='mapping':result['metadata']['predictions'][0]['target']=99
        elif failure in ['repeat','chunk']:result['validation'][failure+'_passed']=False
        else:result['arrays']['write_sums'][0,0,0,0]=float('nan')
    assert command(source,'run','--config',plan,'--approval',review,dependencies=TinyDrift(corrupt=corrupt))==1
    directory=source/'runs/stage0/drift/fixed-context'
    assert not list((directory/'units').glob('*/complete.json'))
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'
    assert (directory/'logs/errors.jsonl').exists()


def test_changed_runtime_unknown_time_and_budget_ceiling_refuse_loading(source):
    plan=prepared(source);review=release_greedy(source)
    engine=TinyDrift(session='other')
    assert command(source,'run','--config',plan,'--approval',review,dependencies=engine)==1 and engine.loads==0
    engine=TinyDrift()
    assert command(source,'run','--config',plan,'--approval',review,'--max-new-units',1,dependencies=engine)==0
    directory=source/'runs/stage0/drift/fixed-context'
    next((directory/'attempts').glob('*/timing.complete.json')).unlink()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==1
    from pilot_eval.stage0_workflow import record_budget
    observation=source/'ceiling.json'
    write_json(observation,dict(event_id='ceiling',session_id='session-a',total_units=20.,
        observed_at='2026-10-26T13:00:00+00:00',note='actual cumulative usage'))
    record_budget(source/'plans/audit/stage0.plan.json',source,observation)
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==1


def test_saved_summary_uses_document_weighting_not_token_independence():
    from pilot_eval.stage0_drift_math import summarize
    rows=[dict(id='a',corpus='fineweb',input_ids=[1,2],attention_mask=[1]*2,
               masks=dict(question=[False]*2,solution=[False]*2,user=[True]*2)),
          dict(id='b',corpus='fineweb',input_ids=[1,2,3,4],attention_mask=[1]*4,
               masks=dict(question=[False]*4,solution=[False]*4,user=[True]*4))]
    config=dict(modules=['linear'],controls=[],units=[dict(variant='V0a',ids=['a','b'])],
        execution_kind='synthetic-cpu-fixture',runtime={},execution_sha256='source',factor_norm_sums={'V0a':1.})
    value=fixture_measurement(rows,{**config,'context_chunk':32,'workspace_bytes':1000},'V0a',False)
    value['arrays']['kl']=np.array([1.,3.,3.,3.])
    result=summarize(config,[value],rows);cell=result['cells'][0]
    assert cell['kl']['equal_example_mean']==2.
    assert cell['kl']['token_mean']==2.5
    assert cell['writes']['modules'][0]['rms_ratio']==.25


def test_cpu_verification_never_imports_model_runtime(source):
    import subprocess,sys
    plan=prepared(source);review=release_greedy(source)
    assert command(source,'run','--config',plan,'--approval',review,dependencies=TinyDrift())==0
    code='''
import sys
class Reject:
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from pilot_eval.stage0_drift import verify
assert verify(sys.argv[1],sys.argv[2])['measurement_complete']
'''
    subprocess.run([sys.executable,'-c',code,str(plan),str(source)],check=True)


def test_failed_full_integrity_invalidates_earlier_sealed_units(source):
    plan=prepared(source);review=release_greedy(source)
    class Broken(TinyDrift):
        def check_integrity(self,*,full=False):return dict(base_unchanged=not full,adapters_unchanged=True)
    assert command(source,'run','--config',plan,'--approval',review,'--max-new-units',1,dependencies=Broken())==1
    engine=TinyDrift()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0


def test_real_engine_hook_and_chunk_path_against_dense_numpy_oracle():
    """Exercise the actual backend without Torch: tiny hook-capable dense model."""
    from contextlib import contextmanager,nullcontext
    from types import SimpleNamespace
    from pilot_eval.stage0_drift_backend import DriftEngine
    from pilot_eval.stage0_drift_math import kl_values,write_sums
    class Tensor(np.ndarray):
        def __new__(cls,value):return np.asarray(value).view(cls)
        def double(self):return Tensor(self.astype(np.float64))
        def abs(self):return Tensor(np.abs(self))
        def norm(self,dim):return Tensor(np.linalg.norm(self,axis=dim))
        def cpu(self):return self
        def numpy(self):return np.asarray(self)
        def sum(self,dim=None,**kwargs):return Tensor(np.asarray(self).sum(axis=dim,**kwargs))
        def max(self):return np.asarray(self).max()
        def exp(self):return Tensor(np.exp(self))
    class Layer:
        def __init__(self,weight):self.weight=Tensor(np.asarray(weight,dtype=np.float32));self.hooks=[]
        def __call__(self,x):
            if getattr(self,'is_output_head',False):
                raise AssertionError('KL must project hidden states and head weights in FP64, before rounding logits.')
            value=Tensor(x@self.weight.T)
            for hook in self.hooks:hook(self,(x,),value)
            return value
        def register_forward_hook(self,hook):
            self.hooks.append(hook);return SimpleNamespace(remove=lambda:self.hooks.remove(hook))
    w=np.array([[1.,.2],[-.3,1.]],dtype=np.float32)
    a=np.array([[.1,-.2]],dtype=np.float32);b=np.array([[.3],[.4]],dtype=np.float32)
    head=np.array([[.2,.3],[-.4,.8],[.7,-.5]],dtype=np.float32)
    class Module:
        def __init__(self):
            self.base_layer=Layer(w);self.lora_A={'V0a':Layer(a),'sft-zero':Layer(a)}
            self.lora_B={'V0a':Layer(b),'sft-zero':Layer(b*0)};self.scaling={'V0a':1.,'sft-zero':1.};self.hooks=[]
        def register_forward_hook(self,hook):
            self.hooks.append(hook);return SimpleNamespace(remove=lambda:self.hooks.remove(hook))
        def __call__(self,x):
            value=self.base_layer(x)
            if not model.disabled:
                name=model.active_adapters[0];value=value+self.lora_B[name](self.lora_A[name](x))*self.scaling[name]
            for hook in self.hooks:hook(self,(x,),value)
            return value
    class Model:
        disabled=False;active_adapters=['V0a']
        def set_adapter(self,name):self.active_adapters=[name]
        def eval(self):pass
        def get_output_embeddings(self):return output_head
        @contextmanager
        def disable_adapter(self):
            old=self.disabled;self.disabled=True
            try:yield
            finally:self.disabled=old
    model=Model();module=Module();output_head=Layer(head);output_head.is_output_head=True;output_head.bias=None
    projections=[]
    def linear64(x,weight,bias=None):
        assert x.dtype==weight.dtype==np.float64
        projections.append((x.dtype,weight.dtype))
        return Tensor(x@weight.T+(0 if bias is None else bias))
    def body(**inputs):
        ids=np.asarray(inputs['input_ids']);x=Tensor(np.stack([ids,ids+.5],axis=-1).astype(np.float32))
        return SimpleNamespace(last_hidden_state=module(x))
    def softmax(x,dim):
        x=np.asarray(x,dtype=np.float64);x=x-x.max(axis=-1,keepdims=True)
        return Tensor(x-np.log(np.exp(x).sum(axis=-1,keepdims=True)))
    torch=SimpleNamespace(tensor=lambda value,**kwargs:Tensor(value),long='long',float32=np.dtype('float32'),float64=np.dtype('float64'),
        ones_like=lambda x:Tensor(np.ones_like(x)),zeros_like=lambda x:Tensor(np.zeros_like(x)),isfinite=np.isfinite,
        equal=np.array_equal,inference_mode=nullcontext,autocast=lambda *args,**kwargs:nullcontext(),
        log_softmax=softmax,finfo=np.finfo,nn=SimpleNamespace(functional=SimpleNamespace(linear=linear64)))
    engine=DriftEngine.__new__(DriftEngine)
    engine.torch=torch;engine.model=model;engine.body=body;engine.modules=[module];engine.cache={};engine.cache_group=None
    engine.config=dict(batch_size=1,context_chunk=2,workspace_bytes=4096,modules=['linear'],controls=['untuned','sft-zero'])
    row=dict(id='a',corpus='fineweb',input_ids=[1,2,3],attention_mask=[1]*3,
        masks=dict(question=[False]*3,solution=[False]*3,user=[True]*3))
    result=engine.measure([row],'V0a',diagnostic=True)
    x=np.stack([np.array([1,2,3]),np.array([1,2,3])+.5],axis=-1).astype(np.float32)
    base=x@w.T;branch=(x@a.T)@b.T;tuned=base+branch
    expected=kl_values(tuned[:2].astype(np.float64)@head.astype(np.float64).T,base[:2].astype(np.float64)@head.astype(np.float64).T)
    np.testing.assert_allclose(result['arrays']['kl'],expected,atol=1e-15)
    np.testing.assert_allclose(result['arrays']['write_sums'][0,2,0],write_sums(base,branch),atol=1e-12)
    assert result['validation']['repeat_passed'] and result['validation']['chunk_passed']
    assert result['validation']['chunk_comparison']['max_absolute_error']<1e-14
    assert projections and output_head.weight.dtype==np.float32
    for control in ['untuned','sft-zero']:
        value=engine.measure([row],control,diagnostic=True)
        np.testing.assert_array_equal(value['arrays']['kl'],np.zeros(2))
    assert all(not layer.hooks for layer in [module,module.base_layer,*module.lora_A.values(),*module.lora_B.values()])


def test_factor_norms_read_actual_fp32_tensors_in_one_payload_pass(tmp_path):
    from test_stage0_workflow import tensor_file
    from pilot_eval.stage0_drift import _factor_norms
    arrays={'linear.lora_A.weight':np.array([[3.,4.]],dtype=np.float32),
            'linear.lora_B.weight':np.array([[0.],[2.]],dtype=np.float32)}
    tensor_file(tmp_path/'adapter_model.safetensors',arrays)
    assert _factor_norms(tmp_path)=={'linear':10.}


def test_explicit_writer_recovery_keeps_budget_and_sealed_work(source):
    import os,platform
    plan=prepared(source);directory=source/'runs/stage0/drift/fixed-context'
    owner=directory/'writer.json'
    write_json(owner,dict(token='owner',pid=os.getpid(),host=platform.node()))
    engine=TinyDrift()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0
    assert command(source,'recover-writer','--config',plan,'--owner-token','owner','--review-notes','inspect')==1
    write_json(owner,dict(token='owner',pid=1234,host='stopped-colab'))
    assert command(source,'recover-writer','--config',plan,'--owner-token','wrong','--review-notes','inspect')==1
    assert command(source,'recover-writer','--config',plan,'--owner-token','owner','--review-notes','old process stopped')==0
    assert not owner.exists() and (directory/'logs/ownership-recovery.jsonl').exists()


def test_notebooks_expose_fixed_context_stages_and_keep_code_parseable():
    import ast
    for file,names in [('stage-0-gpu-colab.ipynb',['drift-prepare','drift-run','drift-status']),
                       ('stage-0-cpu-colab.ipynb',['drift-status','drift-verify'])]:
        data=json.loads((Path('notebooks')/file).read_text())
        code='\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code')
        for name in names:assert name in code
        for cell in data['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                assert cell['outputs']==[] and cell['execution_count'] is None


def test_same_session_code_release_change_is_recorded_not_joined_silently(source):
    """ADR 0015: a later code release may measure in the identical frozen session."""
    plan=prepared(source);review=release_greedy(source)
    class LaterRelease(TinyDrift):
        def __init__(self,commit,**kw):super().__init__(**kw);self.commit=commit
        def runtime(self):return dict(super().runtime(),git_commit=self.commit)
    moved=LaterRelease('later-commit',session='other')
    assert command(source,'run','--config',plan,'--approval',review,dependencies=moved)==1 and moved.loads==0
    engine=LaterRelease('later-commit')
    assert command(source,'run','--config',plan,'--approval',review,'--max-new-units',1,dependencies=engine)==0
    directory=source/'runs/stage0/drift/fixed-context'
    config=json.loads(plan.read_text())
    assert json.loads((directory/'meta/runtime.json').read_text())==config['runtime']
    unit=next((directory/'units').glob('*/attempt.json'))
    assert json.loads(unit.read_text())['code_git_commit']=='later-commit'
    assert command(source,'run','--config',plan,dependencies=engine)==0
    assert command(source,'verify','--config',plan)==0
