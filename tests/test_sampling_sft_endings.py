"""Gold ending is a training requirement, independently of answer correctness."""
import json
from pathlib import Path
import pytest
from test_sampling_sft_instrument import source_template
from pilot_eval.sampling_sft_generation import attempt
from pilot_eval.sampling_sft_review import eligibility_override
from pilot_eval.sampling_sft_training_data import supervised_row

@pytest.mark.parametrize('ending,eligible', [(3,True),(4,False)])
def test_correct_attempt_requires_gold_ending(ending,eligible):
    problem=dict(problem_id=0,source_id='p0',gold='#### 42',gold_end_token_id=3,scorer_prompt_ids=[1,2])
    state=dict(response_ids=[7,ending],end_state='finished',text='#### 42',mean_logprob=-1.,blocks=1)
    row=attempt(0,0,state,problem)
    assert row['automatic_flexible_v3']['correct'] is True
    assert row['eligible'] is eligible
    assert row['response_ids']==[7,ending]
    if eligible: assert supervised_row(problem,row,3)['input_ids']==[1,2,7,3]
    else: assert row['rejection_reason']=='wrong-end-token'

def test_human_cannot_override_wrong_ending():
    problem=dict(gold='#### 42',gold_end_token_id=3)
    row=dict(end_state='finished',automatic_flexible_v3={'status':'invalid'},text='Answer: 42',response_ids=[7,4])
    label=dict(status='numeric',answer='42',evidence='42')
    assert eligibility_override(row,label,problem) is False

def test_notebook_selects_measured_gold_source():
    notebook=json.loads(Path('notebooks/sampling-sft.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert 'plans/pilot3-inputs-v1/activation.prepared.json' in code
    assert 'pilot2-sft-fp32-l4-batch2-v1' not in code

def test_other_ending_is_retried_and_only_gold_ending_is_covered(tmp_path,source_template):
    from test_sampling_sft_generation import prepare,cli,Dependencies,GenerationEngine,directory,read_lines
    from test_sampling_sft_sampler import Tokenizer,generation
    from pilot_eval.sampling_sft_sampler import request_seed
    class OtherEnding(Dependencies):
        def open(self,config,root):
            engine=GenerationEngine(config);original=engine.generate
            def generate(requests):
                outputs=original(requests)
                for i,r in enumerate(requests):
                    if r['problem_id']==1 and r['seed']==request_seed(0,1,r['block'],r['step']):
                        answer=engine.problems[1]['gold'].rsplit('####',1)[-1].strip()
                        outputs[i]=generation([ord(c) for c in '#### '+answer]+[4],'stop',4)
                return outputs
            engine.generate=generate
            return engine,Tokenizer(),dict(execution_kind='cpu-fixture')
    plan=prepare(tmp_path,source_template)
    assert cli(tmp_path,'run','--config',plan,'--k',0,deps=OtherEnding())==0
    assert cli(tmp_path,'verify','--config',plan,'--k',0)==0
    rows=[r for r in read_lines(directory(tmp_path)/'k0/attempts.jsonl') if r['problem_id']==1]
    assert len(rows)==2 and rows[0]['automatic_flexible_v3']['correct']
    assert rows[0]['rejection_reason']=='wrong-end-token' and not rows[0]['eligible']
    assert rows[1]['eligible'] and rows[1]['response_ids'][-1]==3
    dispositions=read_lines(directory(tmp_path)/'k0/dispositions.jsonl')
    assert next(r for r in dispositions if r['problem_id']==1)['selected_attempt']==rows[1]['attempt_id']

