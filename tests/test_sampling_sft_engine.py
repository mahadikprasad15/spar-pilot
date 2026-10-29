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
