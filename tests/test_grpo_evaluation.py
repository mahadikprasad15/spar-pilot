"""Public held-out evaluation and paired report, with controlled inference."""
import json
import pytest
from pilot_eval.cli import main
from test_grpo_training import frozen, train, Dependencies as TrainingDependencies


class Engine:
    def __init__(self, deps): self.deps=deps; self.step=None
    def runtime(self):
        from test_grpo_training import Engine as TrainingEngine
        return TrainingEngine(None).runtime()
    def select_checkpoint(self, path, step): self.step=step
    def generate(self, rows, policy, seed):
        self.deps.calls.append((self.step, policy['mode'], [r['id'] for r in rows], seed))
        if len(self.deps.calls)==self.deps.fail_at: raise RuntimeError('interrupted inference')
        return [dict(text=r['gold'] if self.step!=64 else '#### -999999',
                     token_ids=[7],generated_token_ids=[7,1],stop_reason='eos')
                for r in rows for _ in range(policy['draws'])]
    def check_integrity(self): return dict(base_unchanged=True)
    def close(self): pass


class Dependencies:
    def __init__(self): self.loads=0; self.calls=[]; self.fail_at=None
    def load_evaluation(self, plan, settings, directory): self.loads+=1; return Engine(self)


def source(root):
    path=frozen(root)
    assert train(root,path,TrainingDependencies())==0
    return root/'plans/arm/grpo.training.json'


def evaluate(root,path,deps):
    return main(['grpo-evaluate','--config',str(path),'--name','behaviour','--output-root',str(root)],dependencies=deps)


def report(root):
    return main(['grpo-behaviour-report','--config',str(root/'plans/behaviour/grpo.evaluation.json'),
                 '--name','paired','--output-root',str(root)])


def test_public_evaluation_and_literal_paired_drop_report(tmp_path):
    deps=Dependencies(); path=source(tmp_path)
    assert evaluate(tmp_path,path,deps)==0
    config=json.loads((tmp_path/'plans/behaviour/grpo.evaluation.json').read_text())
    directory=tmp_path/config['run_path']
    rows=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len(rows)==3150 and len({r['record_id'] for r in rows})==3150
    assert sum(r['mode']=='greedy' for r in rows)==750
    assert sum(r['mode']=='sampled' for r in rows)==2400
    assert evaluate(tmp_path,path,deps)==0 and deps.loads==1
    assert report(tmp_path)==0
    result=json.loads((tmp_path/'reports/paired/results/results.json').read_text())
    for mode in ['greedy','sampled']:
        assert result['drops'][mode]['drop']==1.0
        assert result['drops'][mode]['interval_95']==[1.0,1.0]
        assert result['drops'][mode]['outcome']=='demonstrated_harm_beyond_margin'
    assert len(json.loads((tmp_path/'reports/paired/results/bootstrap-indices.json').read_text()))==10000
    assert report(tmp_path)==0 and deps.loads==1


def test_interruption_reuses_sealed_batches_and_corruption_stops_before_load(tmp_path):
    deps=Dependencies(); path=source(tmp_path);deps.fail_at=3
    assert evaluate(tmp_path,path,deps)==1
    config=json.loads((tmp_path/'plans/behaviour/grpo.evaluation.json').read_text())
    directory=tmp_path/config['run_path']
    first=deps.calls[0];saved={p:p.read_bytes() for p in (directory/'units').glob('*')}
    deps.fail_at=None
    assert evaluate(tmp_path,path,deps)==0
    assert deps.calls.count(first)==1
    assert all(p.read_bytes()==value for p,value in saved.items())
    (directory/'results/responses.jsonl').write_text('corruption')
    loads=deps.loads
    assert evaluate(tmp_path,path,deps)==1 and deps.loads==loads
    assert report(tmp_path)==1


def test_sampled_report_clusters_all_eight_draws_instead_of_pass_at_eight(tmp_path,monkeypatch):
    class DrawEngine(Engine):
        def generate(self,rows,policy,seed):
            if policy['mode']=='greedy': return [dict(text=r['gold'],token_ids=[7],generated_token_ids=[7,1],stop_reason='eos') for r in rows]
            correct=6 if self.step==0 else 2
            return [dict(text=r['gold'] if d<correct else '#### -999999',token_ids=[7],generated_token_ids=[7,1],stop_reason='eos')
                    for r in rows for d in range(8)]
    class DrawDependencies(Dependencies):
        def load_evaluation(self,*args): self.loads+=1;return DrawEngine(self)
    path=source(tmp_path);deps=DrawDependencies()
    assert evaluate(tmp_path,path,deps)==0
    import sys
    monkeypatch.setitem(sys.modules,'torch',None)  # CPU report must not import/load Torch.
    assert report(tmp_path)==0
    result=json.loads((tmp_path/'reports/paired/results/results.json').read_text())
    assert result['summaries']['sampled-0']['flexible_accuracy']==.75
    assert result['summaries']['sampled-64']['flexible_accuracy']==.25
    assert result['drops']['sampled']['interval_95']==[.5,.5]
    assert result['drops']['greedy']['interval_95']==[0.,0.]
    assert result['drops']['greedy']['outcome']=='non_inferiority_passed'


@pytest.mark.parametrize('failure',['missing_draw','raw_tokens','runtime','capped'])
def test_bad_inference_cannot_publish_complete_evaluation(tmp_path,failure):
    class BadEngine(Engine):
        def runtime(self): return {'changed':True} if failure=='runtime' else super().runtime()
        def generate(self,rows,policy,seed):
            outputs=super().generate(rows,policy,seed)
            if failure=='missing_draw': return outputs[:-1]
            if failure=='raw_tokens': outputs[0]['generated_token_ids']=[9]
            if failure=='capped': outputs[0].update(stop_reason='cap',generated_token_ids=[7])
            return outputs
    class BadDependencies(Dependencies):
        def load_evaluation(self,*args): self.loads+=1;return BadEngine(self)
    assert evaluate(tmp_path,source(tmp_path),BadDependencies())==1
    config=json.loads((tmp_path/'plans/behaviour/grpo.evaluation.json').read_text())
    assert not (tmp_path/config['run_path']/'complete.json').exists()


def test_real_checkpoint_inference_is_batched_and_seed_reproducible_without_download(tmp_path):
    from test_grpo_algorithm import torch_stack
    torch=torch_stack();torch.set_num_threads(1);torch.manual_seed(5)
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from types import SimpleNamespace
    import hashlib
    from pilot_eval.grpo_eval_engine import EvaluationEngine
    from pilot_eval.sft import adapter_settings
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=3,hidden_size=16,intermediate_size=32,num_hidden_layers=28,
        num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,bos_token_id=0,eos_token_id=2,
        attention_dropout=0,tie_word_embeddings=False,attn_implementation='eager'))
    core=Tokenizer(WordLevel({'0':0,'1':1,'end':2},unk_token='0'));core.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=core,pad_token='0',unk_token='0',eos_token='end')
    tokenizer.padding_side='left';tokenizer.chat_template='tiny-test-template'
    plan=dict(seed=42,adapter=adapter_settings(),source_prompt_contract=dict(
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()))
    settings=dict(evaluation_batch_size=2,generation_groups=2,completion_limit=3,top_p=1.,top_k=0,gradient_checkpointing=False)
    sampler=SimpleNamespace(backend=SimpleNamespace(model=model,tokenizer=tokenizer),close=lambda:None)
    engine=EvaluationEngine(plan,settings,tmp_path,sampler=sampler)
    checkpoint=tmp_path/'zero';engine.model.save_pretrained(checkpoint);engine.select_checkpoint(checkpoint,0)
    rows=[dict(prompt='0',prompt_tokens=1),dict(prompt='0 0',prompt_tokens=2)]
    policy=dict(mode='sampled',draws=8,batch_size=2)
    first=engine.generate(rows,policy,42);second=engine.generate(rows,policy,42)
    assert len(first)==16 and first==second
    assert len(engine.generate(rows,dict(mode='greedy',draws=1,batch_size=2),42))==2
    assert engine.check_integrity()['base_unchanged']
    engine.close()


def test_failed_noninferiority_gate_is_not_automatically_harm_beyond_margin(tmp_path):
    class PartialEngine(Engine):
        def __init__(self,deps): super().__init__(deps);self.final_greedy_count=0
        def generate(self,rows,policy,seed):
            outputs=[]
            for row in rows:
                wrong=self.step==64 and policy['mode']=='greedy' and self.final_greedy_count<15
                if self.step==64 and policy['mode']=='greedy': self.final_greedy_count+=1
                for _ in range(policy['draws']):
                    outputs.append(dict(text='#### -999999' if wrong else row['gold'],token_ids=[7],
                                        generated_token_ids=[7,1],stop_reason='eos'))
            return outputs
    class PartialDependencies(Dependencies):
        def load_evaluation(self,*args): return PartialEngine(self)
    assert evaluate(tmp_path,source(tmp_path),PartialDependencies())==0
    assert report(tmp_path)==0
    result=json.loads((tmp_path/'reports/paired/results/results.json').read_text())
    greedy=result['drops']['greedy']
    assert greedy['drop']==pytest.approx(.1)
    assert greedy['outcome']=='inconclusive' and greedy['evidence_of_positive_decline']
    assert 0<greedy['interval_95'][0]<.1<greedy['interval_95'][1]
