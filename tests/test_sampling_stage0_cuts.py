import numpy as np
import pytest

from pilot_eval.sampling_stage0_cuts import module_edits, variant_catalogue


def test_dense_oracle_and_controls():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 40)).astype(np.float32)
    a = rng.normal(size=(1, 40)).astype(np.float32)
    b = rng.normal(size=(40, 1)).astype(np.float32)
    edits, evidence = module_edits(matrix, {'arm': (a, b)})
    u, s, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    k = np.searchsorted(np.cumsum(s*s), .5 * np.sum(s*s)) + 1
    original = b @ a
    np.testing.assert_allclose(edits['arm-output'][1] @ edits['arm-output'][0], (np.eye(40)-u[:,:k]@u[:,:k].T)@original, atol=2e-6)
    np.testing.assert_allclose(edits['arm-input'][1] @ edits['arm-input'][0], original@(np.eye(40)-vt[:k].T@vt[:k]), atol=2e-6)
    for control, cut in [('output-magnitude', 'output'), ('input-magnitude', 'input')]:
        ca, cb = edits['arm-'+control]; xa, xb = edits['arm-'+cut]
        np.testing.assert_allclose(np.linalg.norm(cb@ca), np.linalg.norm(xb@xa), rtol=1e-6)
    assert evidence['spectrum']['k50'] == k
    assert len(edits) == 7


def test_catalogue_is_arm_parameterized_and_rejects_unsafe_names():
    variants = variant_catalogue(['a', 'b'])
    assert len(variants) == 16
    assert variants['a-original']['operation'] == 'original'
    with pytest.raises(ValueError): variant_catalogue(['../escape'])
    with pytest.raises(ValueError): variant_catalogue(['a', 'a'])


def test_build_resumes_sealed_units_and_rejects_changed_payload(tmp_path):
    from pilot_eval.sampling_stage0_cuts import build_modules
    rng = np.random.default_rng(4)
    adapters = {'a': {m: (rng.normal(size=(1,40)).astype('f4'), rng.normal(size=(40,1)).astype('f4')) for m in ['one', 'two']}}
    matrices = {m: rng.normal(size=(40,40)).astype('f4') for m in ['one', 'two']}
    calls = []
    def load(m):
        calls.append(m)
        return matrices[m]
    assert build_modules(tmp_path, ['one','two'], adapters, load, identity='abc', limit=1)['state'] == 'incomplete'
    assert calls == ['one']
    assert build_modules(tmp_path, ['one','two'], adapters, load, identity='abc')['state'] == 'complete'
    assert calls == ['one', 'two']
    unit = next((tmp_path/'units').glob('*/factors.npz'))
    unit.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        build_modules(tmp_path, ['one','two'], adapters, load, identity='abc')


def test_assemble_preserves_reference_and_writes_all_edits(tmp_path):
    from pilot_eval.sampling_stage0_cuts import build_modules, assemble
    from pilot_eval.stage0_edits import _save_tensors
    from pilot_eval.stage0_sources import factors
    import json
    rng = np.random.default_rng(6)
    module='base_model.model.layer'
    a=rng.normal(size=(1,40)).astype('f4'); b=rng.normal(size=(40,1)).astype('f4')
    source=tmp_path/'source'; source.mkdir()
    (source/'adapter_config.json').write_text('{}')
    _save_tensors(source/'adapter_model.safetensors', {module+'.lora_A.weight':a,module+'.lora_B.weight':b})
    originals={'a':{module:(a,b)}}
    build_modules(tmp_path/'build',[module],originals,lambda _:rng.normal(size=(40,40)),identity='test')
    from pilot_eval.trace_readout_assets import read_weight
    def read(folder):
        path=folder/'adapter_model.safetensors'
        return {module:(read_weight(path,module+'.lora_A.weight'), read_weight(path,module+'.lora_B.weight'))}
    paths=assemble(tmp_path/'build',tmp_path,{'a':source},originals,[module],identity='test',load_factors=read)
    assert len(paths)==8
    assert (tmp_path/paths['a-original']['adapter_path']/'adapter_model.safetensors').read_bytes()==(source/'adapter_model.safetensors').read_bytes()
    edited=read(tmp_path/paths['a-output']['adapter_path'])
    assert not np.array_equal(edited[module][1],b)


def test_collect_resumes_and_rejects_runtime_change(tmp_path):
    from pilot_eval.sampling_stage0_cuts import collect
    config={'variants':{'a-original':{'source_arm':'a','operation':'original'}},'variant_order':['a-original'],
            'seed':42,'scorer':'gsm8k-flexible-v3','decoding':{'max_new_tokens':3,'eos_token_id':2},'adapter_files':{}}
    items=[{'id':'one','prompt':'one','gold':'1'},{'id':'two','prompt':'two','gold':'1'}]
    class Engine:
        def __init__(self): self.calls=[]
        def prepare_inputs(self, rows): return [{'id':r['id'],'input_ids':[1], 'attention_mask':[1]} for r in rows]
        def check_integrity(self,full=False): return {'base_unchanged':True,'adapters_unchanged':True}
        def generate(self,rows,variant,seed):
            self.calls.append(rows[0]['id'])
            return [dict(text='#### 1',token_ids=[1],generated_token_ids=[1,2],stop_reason='eos')]
    engine=Engine()
    assert collect(config,items,tmp_path,{'device':'fake','session_id':'first'},engine,limit=1)['state']=='incomplete'
    assert collect(config,items,tmp_path,{'device':'fake','session_id':'second'},engine)['state']=='complete'
    assert engine.calls==['one','two']
    assert (tmp_path/'meta/final-integrity.json').is_file()
    from pilot_eval.sampling_stage0_cut_workflow import verify
    assert verify(tmp_path)['total'] == 2
    from pilot_eval.sampling_stage0_cut_workflow import export
    class Hub:
        def __init__(self): self.store={}
        def info(self,repo): return type('Info',(),{'private':True})()
        def publish(self,repo,files):
            self.store={k:p.read_bytes() for k,p in files.items()}; return 'a'*40
        def fetch(self,repo,name,revision):
            p=tmp_path.parent/'remote';p.write_bytes(self.store[name]);return p
    assert export(tmp_path,tmp_path,repo='private/repo',client=Hub())['revision']=='a'*40
    with pytest.raises(ValueError,match='runtime'):
        collect(config,items,tmp_path,{'device':'changed'},engine)


def test_export_refuses_incomplete_and_verifies_remote_readback(tmp_path):
    from pilot_eval.sampling_stage0_cut_workflow import export
    class Hub:
        def __init__(self): self.store={}
        def info(self,repo): return type('Info',(),{'private':True})()
        def publish(self,repo,files):
            self.store={k:p.read_bytes() for k,p in files.items()}; return 'a'*40
        def fetch(self,repo,name,revision):
            p=tmp_path/'remote';p.write_bytes(self.store[name]);return p
    run=tmp_path/'runs'/'x';run.mkdir(parents=True)
    with pytest.raises(ValueError,match='complete'):
        export(run,tmp_path,repo='test/private',client=Hub())


def test_switch_adapter_verifies_before_and_after_swap(monkeypatch,tmp_path):
    from pilot_eval.sampling_stage0_cut_backend import CutEngine
    from pilot_eval import sampling_stage0_cut_backend as backend
    calls=[]
    class Model:
        def load_adapter(self,path,adapter_name,is_trainable): calls.append(('load',adapter_name))
        def set_adapter(self,name): calls.append(('set',name))
        def delete_adapter(self,name): calls.append(('delete',name))
    engine=CutEngine.__new__(CutEngine); engine.model=Model();engine.root=tmp_path
    engine.config={'variants':{'a':{'adapter_path':'a'},'b':{'adapter_path':'b'}}}; engine.current='a'
    engine._versions=lambda adapters=False: {'guard':1}
    engine.check_integrity=lambda full=False: {'base_unchanged':True,'adapters_unchanged':True}
    monkeypatch.setattr(backend,'verify_loaded_adapter',lambda model,name,path: calls.append(('verify',name)))
    (tmp_path/'b').mkdir()
    engine.select('b')
    assert calls==[('load','b'),('set','b'),('verify','b'),('delete','a')]
    assert engine.current=='b'


def test_cli_verifies_completed_fixture(tmp_path, capsys):
    from pilot_eval.sampling_stage0_cuts import collect
    from pilot_eval.sampling_stage0_cut_workflow import main
    class Engine:
        def prepare_inputs(self,rows): return [dict(id=r['id'],input_ids=[1],attention_mask=[1]) for r in rows]
        def check_integrity(self,full=False): return dict(base_unchanged=True,adapters_unchanged=True)
        def generate(self,rows,variant,seed): return [dict(text='#### 1',token_ids=[1],generated_token_ids=[1,2],stop_reason='eos')]
    config=dict(variants={'x-original':dict(source_arm='x',operation='original')},variant_order=['x-original'],
                seed=42,scorer='gsm8k-flexible-v3',decoding=dict(max_new_tokens=3,eos_token_id=2),adapter_files={})
    collect(config,[dict(id='x',prompt='p',gold='1')],tmp_path,{},Engine())
    main(['verify','--run-dir',str(tmp_path)])
    assert 'complete' in capsys.readouterr().out


def test_fixture_workflow_build_run_verify_export(tmp_path,monkeypatch):
    import json
    from pilot_eval import sampling_stage0_cut_workflow as workflow, sampling_stage0_geometry as geometry
    from pilot_eval import stage0_sources, stage0_weights, sampling_stage0_cut_backend, stage0_qualification_backend
    from pilot_eval.stage0_edits import _save_tensors
    from pilot_eval.trace_readout_assets import read_weight
    from pilot_eval.workflow import _hash
    rng=np.random.default_rng(7); module='base_model.model.layer'
    matrix=rng.normal(size=(40,40)).astype('f4'); a=rng.normal(size=(1,40)).astype('f4'); b=rng.normal(size=(40,1)).astype('f4')
    src=tmp_path/'source';src.mkdir();(src/'adapter_config.json').write_text('{}')
    _save_tensors(src/'adapter_model.safetensors',{module+'.lora_A.weight':a,module+'.lora_B.weight':b})
    def factors(path):
        p=path/'adapter_model.safetensors'
        return {module:(read_weight(p,module+'.lora_A.weight'),read_weight(p,module+'.lora_B.weight'))}
    monkeypatch.setattr(stage0_sources,'factors',factors)
    (tmp_path/'weights').mkdir(); (tmp_path/'weights'/'shard').write_text('fixture')
    items=[dict(id='x',prompt='p',gold='1')];(tmp_path/'items.json').write_text(json.dumps(items))
    behaviour=dict(model_identity={'base_sha256':'base'},weights_dir='weights',model_snapshot={},prompt_contract={},
                   decoding={'max_new_tokens':3,'eos_token_id':2},cohort_path='items.json',cohort_sha256=_hash(items))
    (tmp_path/'behaviour.json').write_text(json.dumps(behaviour)); (tmp_path/'geometry.json').write_text('{}')
    source=dict(execution_kind='cpu-local-pinned-weights',behaviour_plan='behaviour.json',weights_dir='weights',
        model_identity={'base_sha256':'base'},base_sha256='base',modules={module:{'path':'weights/shard','key':'m'}},
        variants={'x':{'adapter_path':'source'}})
    monkeypatch.setattr(geometry,'_load',lambda path,root:(root,source,None))
    class Weights:
        def inventory(self,directory,audit): return dict(base_sha256='base',modules={module:{'path':str(tmp_path/'weights'/'shard'),'key':'m'}})
        def matrix(self,entry):return matrix
    monkeypatch.setattr(stage0_weights,'LocalWeights',Weights)
    class Engine:
        def __init__(self,config,root): pass
        def prepare_inputs(self,rows):return [dict(id=r['id'],input_ids=[1],attention_mask=[1]) for r in rows]
        def check_integrity(self,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
        def generate(self,rows,variant,seed):return [dict(text='#### 1',token_ids=[1],generated_token_ids=[1,2],stop_reason='eos')]
        def close(self):pass
    monkeypatch.setattr(sampling_stage0_cut_backend,'CutEngine',Engine)
    monkeypatch.setattr(stage0_qualification_backend.QualificationDependencies,'runtime',lambda self:dict(execution_kind='real-cuda',device='A100'))
    monkeypatch.setattr(stage0_qualification_backend.QualificationDependencies,'preflight',lambda self,config,root:None)
    monkeypatch.setattr(workflow,'hf_preflight',lambda *args,**kwargs:None)
    plan=workflow.prepare(tmp_path/'geometry.json',tmp_path,'test')
    assert workflow.build(plan,tmp_path,limit=0)['state']=='incomplete'
    assert workflow.build(plan,tmp_path)['variants']==8
    assert workflow.run(plan,tmp_path,repo='private/repo',limit=2)['state']=='incomplete'
    assert workflow.run(plan,tmp_path,repo='private/repo')['total']==8
    directory=tmp_path/'runs/stage0/test/principal-cuts'
    assert workflow.verify(directory)['state']=='complete'
    adapter=directory/'build/adapters/x-output/adapter_model.safetensors'
    content=adapter.read_bytes();adapter.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='adapter'):workflow.verify(directory)
    adapter.write_bytes(content)


def test_verify_recomputes_summary(tmp_path):
    import json
    from pilot_eval.sampling_stage0_cuts import collect
    from pilot_eval.sampling_stage0_cut_workflow import verify
    from pilot_eval.training import file_hash
    class E:
        def prepare_inputs(self,rows):return [dict(id=r['id'],input_ids=[1],attention_mask=[1]) for r in rows]
        def check_integrity(self,full=False):return dict(base_unchanged=True,adapters_unchanged=True)
        def generate(self,rows,variant,seed):return [dict(text='#### 1',token_ids=[1],generated_token_ids=[1,2],stop_reason='eos')]
    c=dict(variants={'x-original':dict(source_arm='x',operation='original')},variant_order=['x-original'],seed=42,
           scorer='gsm8k-flexible-v3',decoding=dict(max_new_tokens=3,eos_token_id=2))
    collect(c,[dict(id='x',prompt='p',gold='1')],tmp_path,{},E())
    summary=tmp_path/'results/results.json'; data=json.loads(summary.read_text());data['variants']['x-original']['accuracy']=0
    summary.write_text(json.dumps(data));marker=tmp_path/'complete.json';receipt=json.loads(marker.read_text())
    receipt['results_sha256']=file_hash(summary);marker.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='summary'):verify(tmp_path)


def test_hf_preflight_proves_private_write_and_readback(tmp_path):
    from pilot_eval.sampling_stage0_cut_workflow import hf_preflight
    class Hub:
        def __init__(self):self.files={}
        def info(self,repo):return type('Info',(),{'private':True})()
        def publish(self,repo,files):self.files={k:p.read_bytes() for k,p in files.items()};return 'a'*40
        def fetch(self,repo,name,revision):
            p=tmp_path/'remote';p.write_bytes(self.files[name]);return p
    hub=Hub(); result=hf_preflight(tmp_path, 'private/repo', 'a'*64, client=hub)
    assert result['revision']=='a'*40
    assert hf_preflight(tmp_path, 'private/repo', 'a'*64, client=hub)==result


def test_integrity_exception_is_sticky(tmp_path):
    from pilot_eval.sampling_stage0_cuts import collect
    class E:
        def prepare_inputs(self,rows):return [dict(id=r['id'],input_ids=[1],attention_mask=[1]) for r in rows]
        def check_integrity(self,full=False):raise RuntimeError('broken guard')
    c=dict(variants={'x-original':dict(source_arm='x',operation='original')},variant_order=['x-original'],seed=42,
           scorer='gsm8k-flexible-v3',decoding=dict(max_new_tokens=3,eos_token_id=2))
    items=[dict(id='x',prompt='p',gold='1')]
    with pytest.raises(RuntimeError,match='broken guard'):collect(c,items,tmp_path,{},E())
    assert (tmp_path/'meta/integrity-failure.json').exists()
    with pytest.raises(ValueError,match='Prior integrity'):collect(c,items,tmp_path,{},E())
