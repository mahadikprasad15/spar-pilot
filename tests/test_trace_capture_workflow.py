"""Public saved capture diagnostics with a tiny independent CPU backend."""
import json
import numpy as np
import pytest
from test_trace_corpus import Sources,review,digest


def sources(root):
    from pilot_eval.trace_corpus import prepare_corpus
    from pilot_eval.training import file_hash
    deps=Sources(root);corpus=prepare_corpus(deps.path,root,'capture-corpus',review=review(),dependencies=deps)
    paths={}
    for arm in ['sft','grpo']:
        folder=root/'plans'/arm;folder.mkdir(parents=True)
        checkpoint=root/'checkpoints'/arm;checkpoint.mkdir(parents=True)
        (checkpoint/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0)))
        (checkpoint/'adapter_model.safetensors').write_bytes(b'fixture zero adapter')
        files={name:file_hash(checkpoint/name) for name in ['adapter_config.json','adapter_model.safetensors']}
        (checkpoint/'complete.json').write_text(json.dumps(dict(step=0,files=files)))
        payloads={}
        for name in ['items.json','audit.json','manifest.json']:
            p=folder/name;p.write_text('{}');payloads[str(p.relative_to(root))]=file_hash(p)
        config=dict(protocol_version='pilot3-write-v1' if arm=='sft' else 'pilot4-writes-v1',
            source_contract={} if arm=='sft' else dict(kind='grpo'),model=deps.config['model'],
            model_revision='model-pin',tokenizer_revision='tokenizer-pin',
            items_path=str((folder/'items.json').relative_to(root)),audit_path=str((folder/'audit.json').relative_to(root)),
            source_evidence=dict(base_sha256='fixture-base',checkpoints={'0':dict(path=str(checkpoint.relative_to(root)),
                complete_sha256=file_hash(checkpoint/'complete.json'),adapter_sha256=files['adapter_model.safetensors'])}))
        p=folder/'activation.prepared.json';p.write_text(json.dumps(config));payloads[str(p.relative_to(root))]=file_hash(p)
        (folder/'prepare-complete.json').write_text(json.dumps(dict(config_sha256=digest(config),files=payloads)))
        paths[arm]=p
    return corpus,paths


class Engine:
    def __init__(self,*,stop=None,nonzero=False):self.calls=[];self.stop=stop;self.nonzero=nonzero;self.closed=False
    def capture(self,rows,condition):
        self.calls.append(condition)
        if condition==self.stop:raise RuntimeError('simulated interruption')
        result=[]
        for row in rows:
            tokens=np.asarray(row['input_ids'],dtype=np.float32)
            values={s:np.stack([tokens+i,tokens+i+1],axis=-1)[None] for i,s in enumerate(['embedding-output']+[f'block-{j}-output' for j in range(28)])}
            if self.nonzero and condition=='sft-zero':values['block-14-output'][0,4,1]+=.25
            result.append(values)
        return result
    def integrity(self):return 'fixture-base'
    def close(self):self.closed=True


class Dependencies:
    def __init__(self,engine):self.instrument=engine
    def runtime(self):return dict(execution_kind='synthetic-cpu',device='independent-fixture')
    def engine(self,config,root):return self.instrument


def test_public_capture_diagnostic_freezes_four_inputs_and_resumes_sealed_units(tmp_path):
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic,verify_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'diagnostic-v1')
    config=json.loads(plan.read_text())
    assert len(config['examples'])==4
    assert [r['stratum'] for r in config['examples']]==['legacy-150','legacy-150','new-50','new-50']
    first=Engine(stop='grpo-zero')
    with pytest.raises(RuntimeError,match='interruption'):run_diagnostic(plan,tmp_path,dependencies=Dependencies(first))
    assert first.closed
    second=Engine();result=run_diagnostic(plan,tmp_path,dependencies=Dependencies(second))
    assert second.calls==['grpo-zero']
    assert result['newly_sealed_sequence_forwards']==4 and result['reused_sequence_forwards']==8
    assert result['qualified'] is False and result['state']=='awaiting-review'
    assert result['sequence_forwards']==12 and result['sites']==29
    assert result['base_moment_count']==508
    assert result['checks']['sft-zero']['exact_zero'] is True
    assert verify_diagnostic(plan,tmp_path)==result
    class NoGPU:
        def runtime(self):raise AssertionError('completed diagnostics must verify without CUDA')
    assert run_diagnostic(plan,tmp_path,dependencies=NoGPU())==result
    directory=tmp_path/config['run_path']
    assert (directory/'meta/status.json').is_file()
    assert (directory/'units/grpo-zero/evidence.json').is_file()
    (directory/'units/grpo-zero/evidence.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):verify_diagnostic(plan,tmp_path)


def test_failed_zero_check_retains_coordinates_and_never_seals_failed_condition(tmp_path):
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'failed-zero')
    engine=Engine(nonzero=True)
    with pytest.raises(ValueError,match='Step-zero'):
        run_diagnostic(plan,tmp_path,dependencies=Dependencies(engine))
    directory=tmp_path/json.loads(plan.read_text())['run_path']
    evidence=json.loads((directory/'units/sft-zero/evidence.json').read_text())
    failed=[r for r in evidence['comparisons'] if r['max_absolute_error']>0]
    assert failed[0]['site']=='block-14-output' and failed[0]['coordinate']==[0,4,1]
    assert failed[0]['max_absolute_error']==.25 and failed[0]['allowance']==0
    assert not (directory/'units/sft-zero/complete.json').exists()
    assert not (directory/'complete.json').exists()
    errors=[json.loads(line) for line in (directory/'logs/errors.jsonl').read_text().splitlines()]
    assert errors[-1]['diagnostics']['worst']['coordinate']==[0,4,1]
    assert engine.closed


def test_changed_plan_is_rejected_before_runtime_or_model_access(tmp_path):
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'changed-plan')
    changed=json.loads(plan.read_text());changed['examples'][0]['input_ids'][0]+=1
    plan.write_text(json.dumps(changed))
    class NoAccess:
        def runtime(self):raise AssertionError('Changed input must fail before GPU access')
    with pytest.raises(ValueError,match='manifest'):
        run_diagnostic(plan,tmp_path,dependencies=NoAccess())


def test_capture_cli_verifies_saved_fixture_in_fresh_cpu_process(tmp_path):
    import subprocess,sys
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'cli-capture')
    run_diagnostic(plan,tmp_path,dependencies=Dependencies(Engine()))
    result=subprocess.run([sys.executable,'-m','pilot_eval','traces-capture-verify',
        '--config',str(plan),'--output-root',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['diagnostic_complete'] is True
    assert json.loads(result.stdout)['qualified'] is False


def test_notebook_runner_guards_new_gpu_work_and_verifies_complete_run_on_cpu(tmp_path):
    import sys
    from pathlib import Path
    from pilot_eval.notebook_session import NotebookSession
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'runner-capture')
    runner=NotebookSession(tmp_path,Path.cwd(),environment_probe=lambda:dict(cuda_available=False,gpu_count=0,versions={}))
    args=[sys.executable,'-m','pilot_eval','traces-capture-diagnose','--config',str(plan),'--output-root',str(tmp_path)]
    with pytest.raises(RuntimeError,match='Select a GPU runtime'):
        runner.run(args)
    run_diagnostic(plan,tmp_path,dependencies=Dependencies(Engine()))
    def no_probe():raise AssertionError('Complete diagnostic must be verified on CPU')
    runner.environment_probe=no_probe
    assert runner.run(args).returncode==0


def test_notebook_capture_cells_prepare_restore_and_verify_without_previous_execution(tmp_path):
    import ast,sys
    from pathlib import Path
    from pilot_eval.notebook_session import NotebookSession
    from pilot_eval.trace_capture_workflow import run_diagnostic
    corpus,paths=sources(tmp_path)
    notebook=json.loads(Path('notebooks/pilot-5-colab.ipynb').read_text())
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    prepare=next((s for s in cells if "'traces-capture-prepare'" in s),None)
    verify=next((s for s in cells if "'traces-capture-verify'" in s),None)
    assert prepare is not None and verify is not None,'Shared notebook must expose capture workflow'
    class Session:
        def restore(self):return dict(corpus=dict(path=str(corpus)),sources={a:dict(path=str(p),state='audited-config') for a,p in paths.items()})
    namespace=dict(session=Session(),runner=NotebookSession(tmp_path,Path.cwd(),plan_name='notebook-capture'),
        ARTIFACT_ROOT=tmp_path,EXPERIMENT_NAME='notebook-capture',Path=Path,sys=sys,json=json)
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']));assert cell['outputs']==[]
    exec(compile(prepare,'notebook-capture-prepare','exec'),namespace)
    plan=namespace['CAPTURE_DIAGNOSTIC_PATH']
    assert NotebookSession.saved_selection(tmp_path,'notebook-capture')['CAPTURE_DIAGNOSTIC_NAME']=='notebook-capture-capture-diagnostic-v1'
    run_diagnostic(plan,tmp_path,dependencies=Dependencies(Engine()))
    exec(compile(verify,'notebook-capture-verify','exec'),namespace)


def test_real_runtime_requires_saved_numeric_lens_diagnostic(tmp_path):
    from pilot_eval.trace_capture_workflow import prepare_diagnostic,run_diagnostic
    corpus,paths=sources(tmp_path)
    plan=prepare_diagnostic(corpus,paths['sft'],paths['grpo'],tmp_path,'lens-evidence')
    class RealScope(Dependencies):
        def runtime(self):return dict(execution_kind='real-cuda',device='fixture-only-no-CUDA-used')
    with pytest.raises(ValueError,match='numeric lens evidence'):
        run_diagnostic(plan,tmp_path,dependencies=RealScope(Engine()))
