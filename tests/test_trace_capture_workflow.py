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
