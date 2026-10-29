"""Independent offline examples of the approved search, not model qualification."""
import math

import pytest

from pilot_eval.sampling_sft_sampler import clean_gold, sample, score_receipt


class Engine:
    def __init__(self, generations, scores=None):
        self.generations = iter(generations)
        self.scores = iter(scores) if scores is not None else None
        self.requests = []
        self.clears = 0

    def generate(self, requests):
        self.requests.extend(requests)
        return [next(self.generations) for _ in requests]

    def score(self, requests):
        return [dict(token_ids=r['response_ids'], positions=list(range(len(r['prompt_ids']),
                    len(r['prompt_ids'])+len(r['response_ids']))),
                    logprobs=next(self.scores) if self.scores else [-1.0]*len(r['response_ids']))
                for r in requests]

    def clear_cache(self): self.clears += 1


class Tokenizer:
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids if i != 3)
    def apply_chat_template(self, messages, **kwargs):
        return [ord(c) for c in messages[0]['content']] + [2]


def generation(ids, reason='length', stop=None):
    return dict(token_ids=ids, finish_reason=reason, stop_reason=stop)


def problem():
    return dict(problem_id=0, user_message='How many?', gold='2 <<1+1=2>>\n#### 2', scorer_prompt_ids=[1,2])


def test_gold_cleaning_and_k0_stop_only_is_legal():
    assert clean_gold(problem()['gold']) == '2 \nThe final answer is 2.'
    engine=Engine([generation([3], 'stop', 3)])
    result=sample([problem()], 0, engine, Tokenizer(), [3])
    assert result['traces'][0]['end_state']=='finished'
    assert result['traces'][0]['response_ids']==[3]
    assert result['steps']==[] and engine.clears==1
    assert engine.requests[0]['seed']==0


def test_strict_improvement_eos_proposal_can_be_overwritten_and_prefix_is_duplicated(monkeypatch):
    monkeypatch.setattr("random.Random.randint", lambda self,a,b: 0)
    engine=Engine([generation([65]*64), generation([3], 'stop', 3),
                   generation([66], 'length'), generation([3], 'stop',3)],
                  [[-2.0]*64, [-1.0], [-0.5], [-0.75,-0.75], [-0.75]])
    # k=2: EOS is accepted, then overwritten by a better non-EOS proposal.
    # Next block extends it. Equal-score proposal is rejected.
    # Supply two equal-score stop-only refinements in the final block.
    engine.generations=iter([generation([65]*64),generation([3],'stop',3),generation([66]),
                             generation([3],'stop',3),generation([3],'stop',3),generation([3],'stop',3)])
    engine.scores=iter([[-2.0]*64,[-1.0],[-0.5],[-0.75,-0.75],[-0.75],[-0.75]])
    result=sample([problem()],2,engine,Tokenizer(),[3])
    assert [s['accepted'] for s in result['steps']]==[True,True,False,False]
    assert result['traces'][0]['response_ids']==[66,3]
    extension=engine.requests[3]
    assert extension['prompt_ids'][-1]==66
    assert 'B' in Tokenizer().decode(extension['prompt_ids'][:-1])


def test_empty_generation_replays_once_with_identical_request_then_stops():
    engine=Engine([generation([]),generation([])])
    result=sample([problem()],0,engine,Tokenizer(),[3])
    assert result['state']=='engine-error'
    assert result['traces'][0]['end_state']=='engine_error'
    assert engine.requests[0]==engine.requests[1]
    assert len(result['errors'])==2


def test_cap_and_safety_bound_without_changing_production_numbers(monkeypatch):
    monkeypatch.setattr("random.Random.randint", lambda self,a,b: 0)
    engine=Engine([generation([65]*64)]*16)
    result=sample([problem()],0,engine,Tokenizer(),[3])
    assert result['traces'][0]['end_state']=='capped'
    assert len(result['traces'][0]['response_ids'])==1024
    # Every refinement shrinks the trace to one non-stop token.
    engine=Engine([g for _ in range(32) for g in [generation([65]*64),generation([66]),generation([66])]],
                  [s for _ in range(32) for s in [[-2.0]*64,[-1.0]]])
    # Scoring lengths depend on retained prefix; use length-aware synthetic scores.
    def scores(requests):
        return [dict(token_ids=r['response_ids'],positions=list(range(2,2+len(r['response_ids']))),
                     logprobs=[-1.0 if len(r['response_ids'])<64 else -2.0]*len(r['response_ids'])) for r in requests]
    engine.score=scores
    # cut can retain prefix, so a deterministic fixture RNG is tested separately;
    # this assertion checks every output remains bounded under full search.
    value=sample([problem()],2,engine,Tokenizer(),[3])
    assert value['traces'][0]['end_state'] in ['capped','block_budget_exhausted']


def test_score_gate_exact_alignment_and_both_thresholds():
    traces=[dict(trace_id=str(i),prompt_ids=[1],response_ids=[2,3],
                 categories=['short','long','finished','capped','empty-prefix']) for i in range(20)]
    scores=[dict(token_ids=[2,3],positions=[1,2],logprobs=[-1.0,-2.0]) for _ in traces]
    reference=[dict(token_ids=[2,3],positions=[1,2],logprobs=[-0.995,-2.005]) for _ in traces]
    result=score_receipt(traces,scores,reference)
    assert result['passed'] and len(result['traces'])==20
    reference[0]['logprobs']=[-0.98,-2.02] # means identical, per-token error fails
    assert not score_receipt(traces,scores,reference)['passed']
    reference[0]['positions']=[0,1]
    assert not score_receipt(traces,scores,reference)['passed']


@pytest.mark.parametrize('bad', [[], [float('nan')], [0.1]])
def test_malformed_scores_are_protocol_errors(bad):
    engine=Engine([generation([3],'stop',3)], [bad,bad])
    assert sample([problem()],0,engine,Tokenizer(),[3])['state']=='engine-error'
