"""Exact scientific replay and pilot alarms at the public saved workflow seam."""
import json

from pilot_eval.sampling_sft_sampler import sample
from test_sampling_sft_sampler import Engine, Tokenizer, generation, problem


class PilotEngine(Engine):
    def __init__(self):super().__init__([])
    def generate(self,requests):
        self.requests.extend(requests)
        return [generation([65]*r['max_tokens']) if r['block']<2 else generation([3],'stop',3)
                for r in requests]


def test_core_resumes_json_rng_state_and_identical_logical_requests_after_block_two():
    uninterrupted=sample([problem()],5,PilotEngine(),Tokenizer(),[3])
    blocks=[]
    first=sample([problem()],5,PilotEngine(),Tokenizer(),[3],stop_after_blocks=2,
        on_block=lambda b,states,rng:blocks.append(dict(next_block=b+1,states=states,rng=rng)))
    saved=json.loads(json.dumps(blocks[-1]))
    assert first['state']=='paused' and saved['next_block']==2
    second=sample([problem()],5,PilotEngine(),Tokenizer(),[3],resume=saved)
    assert second['traces']==uninterrupted['traces']
    assert first['steps']+second['steps']==uninterrupted['steps']
    assert first['requests']+second['requests']==uninterrupted['requests']
