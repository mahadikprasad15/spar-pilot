"""Scientific training/recovery through public workflow, without Hub downloads."""
import json
from pathlib import Path
import pytest
from pilot_eval.cli import main
from test_grpo_preflight import fixture, collect, Dependencies as PreflightDependencies


def frozen(root):
    path,baseline,options=fixture(root);deps=PreflightDependencies()
    assert collect(root,path,baseline,options,deps)==0
    review=root/'review.json'
    review.write_text(json.dumps(dict(reviewed=True,notes='Checked evidence',margin=.1,
        preregistration=dict(author='researcher',written_at='2026-10-08T00:00:00+00:00',prediction='No drop',falsifier='Drop exceeds margin'),
        monitors={k:dict(threshold=1.,action='flag') for k in ['cap_fraction','dead_group_fraction','length_ratio_change_8_steps']})))
    assert main(['grpo-freeze','--config',str(root/'plans/flight/grpo.preflight.json'),'--review',str(review),
        '--name','frozen','--output-root',str(root)])==0
    return root/'plans/frozen/grpo.frozen.json'


class Engine:
    def __init__(self, deps):self.step=0;self.deps=deps
    def runtime(self):return dict(device='controlled GPU',dtype='float32',adapter_dtype='float32',deterministic_algorithms=True,tf32=False,total_memory_bytes=8*2**30)
    def train_window(self, rows, step):
        self.deps.calls.append(step)
        if step==self.deps.fail_at:raise RuntimeError('injected interruption')
        self.step=step
        from pilot_eval.grpo_baseline import _score
        records=[dict(step=step,advantage=0.,**_score(r,dict(text='#### 1',token_ids=[7],generated_token_ids=[7,1],stop_reason='eos'),d)) for r in rows for d in range(8)]
        return records,dict(loss=0.,learning_rate=.0001,pre_clip_gradient_norm=0.,gradient_consistency=dict(defined=False,reason='zero_gradient',cosine=None),adapter_norm=float(step),parameter_movement=1.,seconds=.1)
    def save_checkpoint(self, directory, step):
        from pilot_eval.training import CHECKPOINT_FILES
        directory.mkdir(parents=True,exist_ok=True)
        for name in CHECKPOINT_FILES:(directory/name).write_text(json.dumps(dict(global_step=step)))
        (directory/'prior-gradient.pt').write_text(json.dumps(dict(step=step)))
    def restore_checkpoint(self, directory):self.step=json.loads((directory/'trainer_state.json').read_text())['global_step'];self.deps.restored.append(self.step)
    def check_integrity(self):return dict(base_before='base',base_after='base',base_unchanged=True)
    def close(self):pass


class Dependencies:
    def __init__(self):self.calls=[];self.restored=[];self.loads=0;self.fail_at=None
    def load_training(self, plan, settings, directory):self.loads+=1;return Engine(self)


def train(root,path,deps):
    return main(['grpo-train','--config',str(path),'--name','arm','--length-change-definition','step-lag-8','--output-root',str(root)],dependencies=deps)


def test_public_training_consumes_exact_groups_and_reuses_completed_run(tmp_path):
    path=frozen(tmp_path);deps=Dependencies()
    assert train(tmp_path,path,deps)==0
    plan=json.loads((tmp_path/'plans/arm/grpo.training.json').read_text());directory=tmp_path/plan['run_path']
    result=json.loads((directory/'results/results.json').read_text())
    assert result['optimizer_steps']==64 and result['training_prompts']==512 and result['accepted_completions']==4096
    assert result['checkpoints']==[0,8,16,32,64]
    assert result['gradient_consistency_coverage']==dict(defined=0,total=64,undefined_reasons={'zero_gradient':64})
    draws=[json.loads(s) for s in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len({r['draw_id'] for r in draws})==4096
    assert deps.calls==list(range(1,65))
    assert train(tmp_path,path,deps)==0 and deps.loads==1


def test_interruption_restores_latest_seal_and_excludes_later_attempt(tmp_path):
    path=frozen(tmp_path);deps=Dependencies();deps.fail_at=11
    assert train(tmp_path,path,deps)==1
    deps.fail_at=None
    assert train(tmp_path,path,deps)==0
    assert deps.restored==[8]
    assert deps.calls==list(range(1,12))+list(range(9,65))
    config=json.loads((tmp_path/'plans/arm/grpo.training.json').read_text());directory=tmp_path/config['run_path']
    records=[json.loads(s) for s in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len(records)==4096 and len({r['draw_id'] for r in records})==4096
    assert (directory/'attempts/attempt-0000/step-010.json').exists()
    accepted=json.loads((directory/'checkpoints/checkpoint-64/history.json').read_text())
    assert 'attempt-0000' in accepted[7]['path'] and 'attempt-0001' in accepted[8]['path']


def test_corruption_stops_recovery_without_loading_model(tmp_path):
    path=frozen(tmp_path);deps=Dependencies();deps.fail_at=11
    assert train(tmp_path,path,deps)==1
    directory=tmp_path/json.loads((tmp_path/'plans/arm/grpo.training.json').read_text())['run_path']
    (directory/'checkpoints/checkpoint-8/optimizer.pt').write_text('corrupt')
    loads=deps.loads
    assert train(tmp_path,path,deps)==1 and deps.loads==loads


def test_real_training_engine_restores_optimizer_rng_and_prior_gradient(tmp_path):
    from test_grpo_algorithm import torch_stack
    torch=torch_stack();torch.set_num_threads(1)
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from types import SimpleNamespace
    from pilot_eval.grpo_train_engine import GRPOTrainingEngine
    from pilot_eval.sft import optimizer_settings,adapter_settings
    import hashlib
    def build(directory):
        torch.manual_seed(5)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=3,hidden_size=16,intermediate_size=32,num_hidden_layers=28,
            num_attention_heads=2,num_key_value_heads=2,attention_dropout=0,tie_word_embeddings=False,
            pad_token_id=0,bos_token_id=0,eos_token_id=2,attn_implementation='eager'))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'0':0,'1':1,'end':2},unk_token='0')),
            pad_token='0',unk_token='0',eos_token='end')
        tokenizer.padding_side='left';tokenizer.chat_template='tiny-test-template'
        sampler=SimpleNamespace(backend=SimpleNamespace(model=model,tokenizer=tokenizer),close=lambda:None)
        plan=dict(seed=42,optimizer=optimizer_settings(),adapter=adapter_settings(),
            source_prompt_contract=dict(chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()))
        settings=dict(generation_groups=1,backward_groups=1,evaluation_batch_size=2,completion_limit=3,
            top_p=1.,top_k=0,gradient_checkpointing=True)
        return GRPOTrainingEngine(plan,settings,directory,sampler=sampler)
    rows=[dict(id=f'tiny:{i}',prompt='0',input_ids=[0],prompt_tokens=1,gold='#### 1') for i in range(8)]
    engine=build(tmp_path/'first')
    first,first_log=engine.train_window(rows,1)
    assert first_log['pre_clip_gradient_norm']>0;checkpoint=tmp_path/'checkpoint';engine.save_checkpoint(checkpoint,1)
    next_a,log_a=engine.train_window(rows,2)
    expected={n:p.detach().clone() for n,p in engine.model.named_parameters() if p.requires_grad}
    engine.close()
    restored=build(tmp_path/'second');restored.restore_checkpoint(checkpoint)
    next_b,log_b=restored.train_window(rows,2)
    assert next_a==next_b
    assert log_a['gradient_consistency']==log_b['gradient_consistency']
    assert all(torch.equal(p,expected[n]) for n,p in restored.model.named_parameters() if p.requires_grad)
    assert restored.check_integrity()['base_unchanged']
    restored.close()


@pytest.mark.parametrize('failure',['nonfinite','pause','stop','integrity'])
def test_failure_never_publishes_scientific_success(tmp_path,failure):
    path=frozen(tmp_path);deps=Dependencies()
    original=deps.load_training
    def load(*args):
        engine=original(*args)
        if failure=='integrity':engine.check_integrity=lambda:dict(base_before='base',base_after='changed',base_unchanged=False)
        if failure=='nonfinite':
            train_window=engine.train_window
            def bad(*args):
                draws,log=train_window(*args);log['loss']=float('nan');return draws,log
            engine.train_window=bad
        return engine
    deps.load_training=load
    if failure in ['pause','stop']:
        # This controlled fixture freezes its own policy before starting training.
        value=json.loads((tmp_path/'review.json').read_text());value['monitors']['dead_group_fraction']=dict(threshold=.5,action=failure)
        review=tmp_path/'review-monitor.json';review.write_text(json.dumps(value))
        assert main(['grpo-freeze','--config',str(tmp_path/'plans/flight/grpo.preflight.json'),'--review',str(review),
            '--name','monitor-frozen','--output-root',str(tmp_path)])==0
        path=tmp_path/'plans/monitor-frozen/grpo.frozen.json'
    assert train(tmp_path,path,deps)==1
    directory=tmp_path/json.loads((tmp_path/'plans/arm/grpo.training.json').read_text())['run_path']
    assert not (directory/'complete.json').exists()
    status=json.loads((directory/'meta/status.json').read_text())
    assert status['state']==({'pause':'paused','stop':'stopped'}.get(failure,'failed'))


def test_changed_settings_and_existing_lock_do_not_load_another_model(tmp_path):
    from pilot_eval.training import run_lock
    path=frozen(tmp_path);deps=Dependencies();deps.fail_at=2
    assert train(tmp_path,path,deps)==1
    config=json.loads((tmp_path/'plans/arm/grpo.training.json').read_text());directory=tmp_path/config['run_path']
    with run_lock(directory):assert train(tmp_path,path,deps)==1
    before=deps.loads
    value=json.loads(path.read_text());value['settings']['completion_limit']+=1;path.write_text(json.dumps(value))
    assert train(tmp_path,path,deps)==1 and deps.loads==before


def test_complete_verification_rejects_changed_run_config(tmp_path):
    path=frozen(tmp_path);deps=Dependencies();assert train(tmp_path,path,deps)==0
    training=tmp_path/'plans/arm/grpo.training.json';directory=tmp_path/json.loads(training.read_text())['run_path']
    (directory/'config.json').write_text('{}')
    assert main(['grpo-verify-training','--config',str(training),'--output-root',str(tmp_path)])==1
