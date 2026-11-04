"""Offline behavior uses the same scoring and paired item semantics."""
import pytest


def test_sampled_accuracy_is_draw_mean_not_pass_at_eight():
    from pilot_eval.offline_grpo_behaviour import summarize
    rows=[dict(mode='sampled',step=0,item_id='x',draw_index=i,reward=int(i==0),
               strict={'correct':i==0,'status':'valid'},flexible={'status':'valid'},
               capped=False,token_count=1) for i in range(8)]
    assert summarize(rows)['flexible_accuracy']==.125


def test_reuse_contract_requires_runtime_and_batch_identity():
    from pilot_eval.offline_grpo_behaviour import compatible
    base=dict(items=[{'id':'a'}],prompt_template={'name':'x'},scorer='v3',strict_scorer='strict',runtime={'gpu':'a'},policies=[{'batch_size':1}],units=[],settings={},seed=42,seed_rule='rule',cap_reward=0,model='m',model_revision='r',tokenizer_revision='r',dtype='float32')
    assert compatible(base,dict(base))
    assert not compatible(base,dict(base,runtime={'gpu':'b'}))
    assert not compatible(base,dict(base,policies=[{'batch_size':2}]))


def test_public_cpu_evaluation_and_source_tamper(tmp_path, monkeypatch):
    import json
    from pilot_eval import offline_grpo_behaviour as behavior
    from pilot_eval.cli import main
    source=tmp_path/'train.json';source.write_text('{}')
    train_dir=tmp_path/'train';train_dir.mkdir();(train_dir/'complete.json').write_text('{}')
    checkpoints={}
    for step in [0,8,16,32,64]:
        cp=tmp_path/f'checkpoint-{step}';cp.mkdir();(cp/'complete.json').write_text('{}');(cp/'adapter_model.safetensors').write_bytes(str(step).encode());checkpoints[step]=cp
    items=[dict(id=f'q{i}',prompt=f'question {i}',gold='#### 2',prompt_tokens=1) for i in range(150)]
    settings=dict(evaluation_batch_size=25,generation_groups=25,top_p=1.,top_k=0,completion_limit=4)
    frozen=dict(settings=settings,review=dict(margin=.1))
    from pilot_eval.training import file_hash
    reference=tmp_path/'historical-zero';reference.mkdir()
    (reference/'complete.json').write_text('{}');(reference/'adapter_model.safetensors').write_bytes(b'0')
    plan=dict(model='tiny',model_revision='r',tokenizer_revision='r',dtype='float32',source_prompt_contract={'template':'fixture'},source_evidence={'checkpoints':{'0':{'path':'historical-zero','complete_sha256':file_hash(reference/'complete.json'),'adapter_sha256':file_hash(reference/'adapter_model.safetensors')}}})
    training=dict(runtime={'kind':'fixture'})
    monkeypatch.setattr(behavior,'_source',lambda path,root:(training,train_dir,frozen,plan,items,checkpoints))
    control={'mismatch':True,'nonzero':False}
    class Engine:
        def runtime(self):return training['runtime']
        def check_integrity(self):return {'base_unchanged':True}
        def select_checkpoint(self,path,step):
            self.step=step;self.reference=path==reference
            if step:control['nonzero']=True
        def generate(self,items,policy,seed):
            token=2 if self.reference and control['mismatch'] else 1
            return [dict(text='#### 2',token_ids=[token],generated_token_ids=[token,99],stop_reason='eos') for _ in range(len(items)*policy['draws'])]
        def close(self):pass
    class Dependencies:
        def load_evaluation(self,plan,settings,directory):return Engine()
    args=['--output-root',str(tmp_path)]
    assert main(['offline-grpo-behaviour-prepare',*args,'--config',str(source),'--name','test'])==0
    config=tmp_path/'plans/test/offline.behaviour.json'
    assert main(['offline-grpo-behaviour-run',*args,'--config',str(config)],dependencies=Dependencies())==1
    assert not control['nonzero']
    control['mismatch']=False
    assert main(['offline-grpo-behaviour-run',*args,'--config',str(config)],dependencies=Dependencies())==0
    cfg,records=behavior.verify(config,tmp_path)
    assert len(records)==3150 and len([r for r in records if r['mode']=='sampled'])==2400
    assert main(['offline-grpo-behaviour-report',*args,'--config',str(config),'--name','report'])==0
    report=json.loads((tmp_path/'reports/report/results/results.json').read_text())
    assert report['anchor']['state']=='passed' and report['summaries']['sampled-64']['flexible_accuracy']==1
    assert report['online_comparison_status']['state']=='not-selected'
    other=dict(json.loads(config.read_text()),runtime={'kind':'other'})
    (tmp_path/'online.json').write_text('{}')
    monkeypatch.setattr(behavior.online,'verify_evaluation',lambda *args:(other,[]))
    monkeypatch.chdir(tmp_path.parent)
    assert main(['offline-grpo-behaviour-report',*args,'--config',str(config.relative_to(tmp_path)),'--name','report2','--online','online.json'])==0
    missing=json.loads((tmp_path/'reports/report2/results/results.json').read_text())
    assert missing['online_comparison'] is None
    assert missing['online_comparison_status']==dict(state='not-performed',differing_fields=['runtime'])
    text=(tmp_path/'reports/report2/results/report.md').read_text()
    assert 'Online comparison: NOT performed; differing fields: runtime' in text
    (checkpoints[8]/'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='source, plan or implementation changed'):
        behavior.verify(config,tmp_path)
