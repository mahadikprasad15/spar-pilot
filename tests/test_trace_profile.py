"""Bounded saved profile: exact workload, interrupted units and CPU reuse."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_trace_capture_workflow import sources


def prepared(tmp_path):
    from pilot_eval.trace_capture_workflow import prepare_diagnostic
    from pilot_eval.trace_profile import prepare_profile
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    from pilot_eval.grpo_random_control import construct_control
    from safetensors.numpy import save_file
    from pilot_eval.activation_prepare import PROJECTIONS
    corpus,paths=sources(tmp_path)
    diagnostic=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'zero')
    for arm in ['sft','grpo']:
        config=json.loads(paths[arm].read_text());cp=tmp_path/'checkpoints'/f'{arm}-64';cp.mkdir()
        (cp/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0)))
        arrays={}
        for i in range(28):
            for p in PROJECTIONS:
                stem=f'base_model.model.model.layers.{i}.{"self_attn" if p in PROJECTIONS[:4] else "mlp"}.{p}'
                arrays[stem+'.lora_A.weight']=np.array([[1.,0]],dtype=np.float32)
                arrays[stem+'.lora_B.weight']=np.array([[.1],[.2]],dtype=np.float32)
        save_file(arrays,str(cp/'adapter_model.safetensors'))
        files={n:file_hash(cp/n) for n in ['adapter_config.json','adapter_model.safetensors']}
        (cp/'complete.json').write_text(json.dumps(dict(step=64,files=files)))
        config['source_evidence']['checkpoints']['64']=dict(path=str(cp.relative_to(tmp_path)),
            complete_sha256=file_hash(cp/'complete.json'),adapter_sha256=files['adapter_model.safetensors'])
        paths[arm].write_text(json.dumps(config))
        seal=paths[arm].parent/'prepare-complete.json';marker=json.loads(seal.read_text())
        marker['config_sha256']=_hash(config);marker['files'][str(paths[arm].relative_to(tmp_path))]=file_hash(paths[arm])
        seal.write_text(json.dumps(marker))
    # Reprepare the diagnostic under its immutable updated parents.
    diagnostic=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'zero-new')
    control=tmp_path/'control';construct_control(tmp_path/'checkpoints/grpo-64',control)
    parent=json.loads(paths['grpo'].read_text());config={**parent,'source_contract':dict(kind='random-rank1-control',
        parent_prepared_path=str(paths['grpo'].relative_to(tmp_path)),control_path='control')}
    config['source_evidence']={**parent['source_evidence'],'checkpoints':{
        '64':dict(path='control',complete_sha256=file_hash(control/'complete.json'),adapter_sha256=file_hash(control/'adapter_model.safetensors'))}}
    random=tmp_path/'plans/random/activation.prepared.json';random.parent.mkdir();random.write_text(json.dumps(config))
    (random.parent/'prepare-complete.json').write_text(json.dumps(dict(config_sha256=_hash(config),
        files={str(random.relative_to(tmp_path)):file_hash(random)})))
    model=tmp_path/'model-config.json';model.write_text(json.dumps(dict(hidden_size=2,num_hidden_layers=28)))
    return prepare_profile(diagnostic,random,tmp_path,'bounded-profile',model_config=model)


class Engine:
    def __init__(self,stop=None):self.calls=[];self.closed=False;self.stop=stop
    def integrity(self):return 'fixture-base'
    def capture(self,rows,condition):
        self.calls.append((condition,len(rows)))
        if len(self.calls)==self.stop:raise RuntimeError('interrupted')
        from pilot_eval.trace_capture_workflow import SITES
        values=np.stack([np.array(r['input_ids'],dtype=np.float32) for r in rows])[:,:,None]
        return {site:np.repeat(values,2,axis=2)+(0 if condition=='base' else .25) for site in SITES}
    def memory(self):return dict(peak_allocated_bytes=128,peak_reserved_bytes=256)
    def close(self):self.closed=True


class Deps:
    def __init__(self,engine):self.instance=engine
    def runtime(self):return dict(execution_kind='synthetic-cpu',device='fixture')
    def engine(self,config,root):return self.instance


def test_profile_resumes_units_and_completed_run_needs_no_gpu(tmp_path):
    from pilot_eval.trace_profile import run_profile,verify_profile
    path=prepared(tmp_path);config=json.loads(path.read_text())
    assert config['workload']==dict(sequence_forwards=32,model_calls=20,units=8)
    assert config['batch_sizes']==[1,4] and len(config['examples'])==4
    first=Engine(stop=6)
    with pytest.raises(RuntimeError,match='interrupted'):run_profile(path,tmp_path,dependencies=Deps(first))
    assert first.closed
    second=Engine();result=run_profile(path,tmp_path,dependencies=Deps(second))
    assert len(second.calls)==16 # batch-1 base unit reused; interrupted unit repeated
    assert result['profile_complete'] and not result['production_ready']
    assert result['completed_units']==8 and result['sequence_forwards']==32
    assert verify_profile(path,tmp_path)==result
    class NoGPU:
        def runtime(self):raise AssertionError('completed profile must verify without GPU')
    assert run_profile(path,tmp_path,dependencies=NoGPU())==result
    unit=tmp_path/config['run_path']/'units/b1-base/evidence.json';unit.write_text('{}')
    with pytest.raises(ValueError,match='hash'):verify_profile(path,tmp_path)


def test_profile_checks_sources_before_runtime_and_rejects_gpu_change(tmp_path):
    from pilot_eval.trace_profile import run_profile
    path=prepared(tmp_path)
    first=Engine(stop=6)
    with pytest.raises(RuntimeError):run_profile(path,tmp_path,dependencies=Deps(first))
    class Other(Deps):
        def runtime(self):return dict(execution_kind='synthetic-cpu',device='other')
    second=Engine()
    with pytest.raises(ValueError,match='immutable|changed|frozen'):run_profile(path,tmp_path,dependencies=Other(second))
    assert not second.calls
    (tmp_path/'control/construction.json').write_text('{}')
    with pytest.raises(ValueError,match='parent|hash'):run_profile(path,tmp_path,dependencies=Deps(second))


def test_public_profile_cli_and_notebook_guard_reuse_on_cpu(tmp_path):
    import subprocess,sys
    from pilot_eval.trace_profile import run_profile
    from pilot_eval.notebook_session import NotebookSession
    path=prepared(tmp_path)
    runner=NotebookSession(tmp_path,Path.cwd(),environment_probe=lambda:dict(cuda_available=False,gpu_count=0,versions={}))
    args=[sys.executable,'-m','pilot_eval','traces-profile-run','--config',str(path),'--output-root',str(tmp_path)]
    with pytest.raises(RuntimeError,match='Select a GPU runtime'):runner.run(args)
    run_profile(path,tmp_path,dependencies=Deps(Engine()))
    result=subprocess.run([sys.executable,'-m','pilot_eval','traces-profile-verify','--config',str(path),
        '--output-root',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['profile_complete'] is True
    def no_probe():raise AssertionError('completed profile must reuse on CPU')
    runner.environment_probe=no_probe
    assert runner.run(args).returncode==0


def test_shared_notebook_exposes_bounded_profile_with_restored_paths():
    import ast
    notebook=json.loads(Path('notebooks/pilot-5-colab.ipynb').read_text())
    code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for command in ['traces-lens-inspect','traces-profile-prepare','traces-profile-run','traces-profile-verify']:
        cell=next((c for c in code if repr(command) in c),None)
        assert cell is not None,f'Missing public notebook command {command}'
        ast.parse(cell)
        if command!='traces-lens-inspect':assert 'saved_selection' in cell
    for cell in notebook['cells']:
        if cell['cell_type']=='code':assert cell['outputs']==[] and cell['execution_count'] is None


def test_profile_records_random_norm_evidence_without_copying_direction_payload(tmp_path):
    path=prepared(tmp_path);config=json.loads(path.read_text())
    assert config['random_construction']['seed']==42
    assert 'directions' not in config['random_construction']
    first=config['random_norms'][0]
    assert first['target_norm']==pytest.approx(.2236068,abs=1e-7)
    assert first['realized_norm']==pytest.approx(.2236068,abs=1e-7)
