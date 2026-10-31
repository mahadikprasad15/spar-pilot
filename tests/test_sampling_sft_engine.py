"""Versioned vLLM adapter contracts with independent tiny token/logprob objects."""
import sys
from types import SimpleNamespace

import pytest

from pilot_eval.sampling_sft_engine import VLLMEngine
from pilot_eval.sampling_sft_sampler import ProtocolError


class Params:
    def __init__(self,**kwargs):self.values=kwargs


class LLM:
    def generate(self,prompts,params,use_tqdm=False):
        self.params=params
        outputs=[]
        for prompt in prompts:
            ids=prompt['prompt_token_ids']
            outputs.append(SimpleNamespace(prompt_token_ids=ids,num_cached_tokens=0,
                outputs=[SimpleNamespace(token_ids=[3],finish_reason='stop',stop_reason=3)],
                prompt_logprobs=[None]+[{token:SimpleNamespace(logprob=-i)} for i,token in enumerate(ids[1:],1)]))
        return outputs


def engine(monkeypatch):
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=Params))
    llm=LLM()
    return VLLMEngine(llm,dict(top_k=-1,min_p=0.0,presence_penalty=0.0,
                             frequency_penalty=0.0,actual_stop_ids=[3,4],max_model_len=4096)),llm


def test_proposer_allows_stop_only_first_token_and_score_has_no_sampling_penalties(monkeypatch):
    value,llm=engine(monkeypatch)
    value.generate([dict(prompt_ids=[1,2],max_tokens=64,seed=0)])
    params=llm.params[0].values
    assert params['min_tokens']==0 # min_tokens=1 masks EOS at first token in vLLM
    assert params['temperature']==0.6 and params['repetition_penalty']==1.1
    scores=value.score([dict(prompt_ids=[1,2],response_ids=[7,3])])
    assert scores[0]['positions']==[2,3] and scores[0]['logprobs']==[-2,-3]
    assert llm.params.values['temperature']==1.0 and llm.params.values['repetition_penalty']==1.0


def test_adapter_rejects_prompt_and_target_alignment_errors(monkeypatch):
    value,llm=engine(monkeypatch)
    original=llm.generate
    def bad(*args,**kwargs):
        outputs=original(*args,**kwargs);outputs[0].prompt_logprobs[2]={}
        return outputs
    llm.generate=bad
    with pytest.raises(ProtocolError,match='target token'):
        value.score([dict(prompt_ids=[1,2],response_ids=[7,3])])


class NamedRPC:
    """Model the worker wire contract: method names and JSON-safe payloads only."""
    def __init__(self,worker):self.worker=worker;self.calls=[]

    def apply_model(self,func):
        raise TypeError('Object of type function is not serializable')

    def collective_rpc(self,method):
        import json
        json.dumps(method)
        assert isinstance(method,str)
        self.calls.append(method)
        result=getattr(self.worker,method)()
        json.dumps(result)
        return [result]


def test_memory_measurement_uses_named_worker_rpc():
    worker=SimpleNamespace(sampling_sft_reset_memory=lambda:None,
        sampling_sft_peak_memory=lambda:dict(allocated_bytes=123,reserved_bytes=456))
    llm=NamedRPC(worker)
    value=VLLMEngine(llm,{})
    value.measurement_start()
    assert value.measurement_end()==[dict(allocated_bytes=123,reserved_bytes=456)]
    assert llm.calls==['sampling_sft_reset_memory','sampling_sft_peak_memory']


def test_worker_extension_runs_helpers_on_actual_worker_model(monkeypatch):
    import pilot_eval.sampling_sft_engine as module
    extension=module.SamplingSFTWorkerExtension()
    model=object()
    extension.model_runner=SimpleNamespace(model=model)
    seen=[]
    for helper in ['worker_evidence','reset_memory','peak_memory']:
        def record(actual,name=helper):
            assert actual is model
            seen.append(name)
            return {'helper':name}
        monkeypatch.setattr(module,helper,record)
    llm=NamedRPC(extension)
    for method,helper in [('sampling_sft_worker_evidence','worker_evidence'),
                          ('sampling_sft_reset_memory','reset_memory'),
                          ('sampling_sft_peak_memory','peak_memory')]:
        assert llm.collective_rpc(method)==[{'helper':helper}]
    assert seen==['worker_evidence','reset_memory','peak_memory']


def test_open_engine_registers_extension_and_checks_worker_without_function_transport(monkeypatch,tmp_path):
    import json
    import pilot_eval.sampling_sft_engine as module
    import pilot_eval.sampling_sft as workflow
    runtime=dict(vllm_version='0.11.1',actual_stop_ids=[3,4],max_model_len=4096,
        gpu_memory_utilization=.85,prefix_caching=True,max_num_seqs=32,max_num_batched_tokens=8192)
    model=tmp_path/'revision';model.mkdir();(model/'config.json').write_text('{}')
    tokenizer=SimpleNamespace(chat_template='template',eos_token_id=3,
        convert_tokens_to_ids=lambda value:3 if value=='<|im_end|>' else 4)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True,device_count=lambda:1)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer),AutoModelForCausalLM=object))
    monkeypatch.setattr(module.importlib.metadata,'version',lambda name:
        {'transformers':'4.57.6','vllm':'0.11.1'}[name])
    monkeypatch.setattr(module,'numeric_flags',lambda:None)
    monkeypatch.setattr(module,'proposer',lambda *args:[1])
    monkeypatch.setattr(workflow,'verify',lambda *args:dict(audit=dict(training=dict(
        identity=dict(model_revision='revision',tokenizer_revision='revision'),
        prompt_mask_contract=dict(chat_template_sha256=module._hash('template'))))))
    evidence={'actual_worker':True};checked=[];created=[]
    def make_llm(**kwargs):
        assert kwargs['worker_extension_cls']=='pilot_eval.sampling_sft_engine.SamplingSFTWorkerExtension'
        llm=NamedRPC(SimpleNamespace(sampling_sft_worker_evidence=lambda:evidence))
        created.append(llm)
        return llm
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=make_llm))
    monkeypatch.setattr(module,'check_worker',lambda actual,expected:checked.append((actual,expected)))
    plan=dict(source_plan='source.json',runtime_hash='frozen',problems=[{}],settings=dict(
        runtime=runtime,local_model='revision',local_tokenizer='revision',mode='bounded'))
    engine,_,receipt=module.open_engine(plan,tmp_path)
    assert engine.llm is created[0]
    assert checked==[(evidence,runtime)]
    assert created[0].calls==['sampling_sft_worker_evidence']
    assert receipt['worker']==evidence


def test_forced_cap_is_explicit_and_default_stop_policy_stays_unchanged(monkeypatch):
    value,llm=engine(monkeypatch)
    value.generate([dict(prompt_ids=[1,2],prefix_ids=[],max_tokens=1024,seed=0,
                         qualification_min_tokens=1024)])
    assert llm.params[0].values['min_tokens']==1024
    value.generate([dict(prompt_ids=[1,2],max_tokens=64,seed=0)])
    assert llm.params[0].values['min_tokens']==0
    with pytest.raises(ProtocolError):
        value.generate([dict(prompt_ids=[1,2],prefix_ids=[],max_tokens=64,seed=0,
                             qualification_min_tokens=64)])
