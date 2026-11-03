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
    assert collect(config,items,tmp_path,{'device':'fake'},engine,limit=1)['state']=='incomplete'
    assert collect(config,items,tmp_path,{'device':'fake'},engine)['state']=='complete'
    assert engine.calls==['one','two']
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
