import json
import pytest
from pilot_eval.sampling_stage0_evaluation import collect


class Engine:
    def __init__(self): self.calls = []
    def prepare_inputs(self, rows):
        return [{'id': r['id'], 'input_ids': [7], 'attention_mask': [1]} for r in rows]
    def check_integrity(self, *, full=False): return {'base_unchanged': True, 'adapters_unchanged': True}
    def generate(self, rows, variant, seed):
        self.calls.append((variant, rows[0]['id']))
        return [{'text': '#### 1', 'token_ids': [1], 'generated_token_ids': [1, 2], 'stop_reason': 'eos'}]


def config():
    return {'variants': {k: {'adapter_path': k} for k in ('k0','k2','k5')},
            'variant_order': ['k0','k2','k5'], 'decoding': {'eos_token_id': 2, 'max_new_tokens': 10},
            'scorer': 'gsm8k-flexible-v3', 'seed': 42,
            'adapter_files': {k: {'adapter_model.safetensors': k} for k in ('k0','k2','k5')}}


def test_resume_reconstructs_seals_and_only_runs_missing_units(tmp_path):
    rows = [{'id': 'q1', 'prompt': 'a', 'gold': '#### 1', 'prompt_tokens': 1},
            {'id': 'q2', 'prompt': 'b', 'gold': '#### 1', 'prompt_tokens': 1}]
    engine = Engine()
    first = collect(config(), rows, tmp_path, {'device': 'fixture'}, engine, max_new_units=2)
    assert first['state'] == 'incomplete' and len(engine.calls) == 2
    second = collect(config(), rows, tmp_path, {'device': 'fixture'}, engine)
    assert second['state'] == 'complete' and len(engine.calls) == 6
    assert len((tmp_path/'results/responses.jsonl').read_text().splitlines()) == 6
    unit = next((tmp_path/'units').glob('*/responses.jsonl'))
    unit.write_text(unit.read_text().replace('#### 1', '#### 2'))
    with pytest.raises(ValueError, match='changed|mismatch'):
        collect(config(), rows, tmp_path, {'device': 'fixture'}, engine)


def test_resume_rejects_changed_runtime(tmp_path):
    rows = [{'id': 'q1', 'prompt': 'a', 'gold': '#### 1', 'prompt_tokens': 1}]
    collect(config(), rows, tmp_path, {'device': 'A'}, Engine(), max_new_units=1)
    with pytest.raises(ValueError, match='runtime'):
        collect(config(), rows, tmp_path, {'device': 'B'}, Engine())


def test_prepare_rejects_arm_identity_mismatch(tmp_path, monkeypatch):
    from pilot_eval import sampling_stage0_evaluation as module
    from pilot_eval import sampling_sft_training as training
    from pilot_eval.workflow import _hash
    gold={'model':'qwen','model_revision':'r','tokenizer_revision':'r', 'evaluation_items_path':'items.json',
          'evaluation_items_sha256':_hash([{'id':str(i),'prompt':'p','gold':'#### 1'} for i in range(150)]),
          'source_config':{'decoding':{'do_sample':False,'num_beams':1,'num_return_sequences':1,'max_new_tokens':1024,'eos_token_id':2,'pad_token_id':0}, 'chat_template_sha256':'x'}}
    (tmp_path/'gold.json').write_text(json.dumps(gold))
    (tmp_path/'items.json').write_text(json.dumps([{'id':str(i),'prompt':'p','gold':'#### 1'} for i in range(150)]))
    for arm in ('k0','k2','k5'): (tmp_path/f'{arm}.json').write_text('{}')
    manifest={'gold_config':'gold.json','weights_dir':'weights','training_plans':{a:f'{a}.json' for a in ('k0','k2','k5')}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(training, 'verify', lambda p,r: {'execution_kind':'real-cuda'})
    monkeypatch.setattr(training, 'load', lambda p,r: (r, {'k':99,'expected_base_sha256':'a'}, r))
    with pytest.raises(ValueError, match='arm'):
        module.prepare(tmp_path/'manifest.json', tmp_path, 'test')


def test_failed_final_integrity_prevents_reusing_attempt(tmp_path):
    class Bad(Engine):
        def check_integrity(self, *, full=False):
            return {'base_unchanged': not full, 'adapters_unchanged': True}
    rows=[{'id':'q1','prompt':'a','gold':'#### 1','prompt_tokens':1}]
    with pytest.raises(ValueError, match='integrity'):
        collect(config(), rows, tmp_path, {'device':'A'}, Bad(), max_new_units=1)
    with pytest.raises(ValueError, match='integrity'):
        collect(config(), rows, tmp_path, {'device':'A'}, Engine())


def test_completed_collection_is_read_only_on_repeat(tmp_path):
    rows=[{'id':'q1','prompt':'a','gold':'#### 1','prompt_tokens':1}]
    engine=Engine()
    collect(config(),rows,tmp_path,{'device':'A'},engine)
    path=tmp_path/'results/responses.jsonl'
    before=path.stat().st_mtime_ns
    collect(config(),rows,tmp_path,{'device':'A'},engine)
    assert path.stat().st_mtime_ns == before and len(engine.calls)==3



def test_run_rejects_non_a100_runtime_before_loading_model(tmp_path):
    from pilot_eval.sampling_stage0_evaluation import run
    from pilot_eval.workflow import _hash
    rows=[{'id':'q1'}]
    (tmp_path/'items.json').write_text(json.dumps(rows))
    cfg={'parents':{},'implementation':{},'cohort_path':'items.json','cohort_sha256':_hash(rows),
         'run_path':'runs/x','execution_kind':'real-cuda'}
    (tmp_path/'plan.json').write_text(json.dumps(cfg))
    class Dependencies:
        def runtime(self): return {'execution_kind':'real-cuda','device':'L4'}
        def engine(self,*args): raise AssertionError('must not load')
    with pytest.raises(ValueError,match='A100'):
        run(tmp_path/'plan.json',tmp_path,dependencies=Dependencies())


def test_run_smoke_precedes_generation_and_resumes(tmp_path):
    from pilot_eval.sampling_stage0_evaluation import run
    from pilot_eval.workflow import _hash
    rows=[{'id':f'q{i}','prompt':'a','gold':'#### 1','prompt_tokens':1} for i in range(2)]
    (tmp_path/'items.json').write_text(json.dumps(rows))
    cfg={**config(),'parents':{},'implementation':{},'cohort_path':'items.json','cohort_sha256':_hash(rows),
         'run_path':'runs/x','execution_kind':'real-cuda'}
    (tmp_path/'plan.json').write_text(json.dumps(cfg))
    class FakeEngine(Engine):
        loaded_adapter_checks=[{'variant':k,'tensor_count':392,'exact_payload_match':True} for k in ('k0','k2','k5')]
        def smoke(self, rows): return [{'variant':arm,'id':row['id'],'exact_repeat':True} for arm in ('k0','k2','k5') for row in rows]
        def close(self): pass
    class Dependencies:
        def __init__(self): self.model=FakeEngine()
        def runtime(self): return {'execution_kind':'real-cuda','device':'A100','session_id':'fixture'}
        def engine(self,*args): return self.model
        def preflight(self,*args): pass
    deps=Dependencies()
    assert run(tmp_path/'plan.json',tmp_path,dependencies=deps,max_new_units=2)['state']=='incomplete'
    assert run(tmp_path/'plan.json',tmp_path,dependencies=deps)['state']=='complete'
    assert len(deps.model.calls)==6
    assert (tmp_path/'runs/x/meta/direct-smoke.json').exists()


def test_verify_rejects_changed_result_without_loading_gpu(tmp_path):
    from pilot_eval.sampling_stage0_evaluation import verify
    from pilot_eval.workflow import _hash
    rows=[{'id':'q1','prompt':'a','gold':'#### 1','prompt_tokens':1}]
    (tmp_path/'items.json').write_text(json.dumps(rows))
    cfg={**config(),'parents':{},'implementation':{},'cohort_path':'items.json','cohort_sha256':_hash(rows),'run_path':'runs/x'}
    (tmp_path/'plan.json').write_text(json.dumps(cfg))
    collect(cfg,rows,tmp_path/'runs/x',{'device':'fixture'},Engine())
    assert verify(tmp_path/'plan.json',tmp_path)['state']=='complete'
    p=tmp_path/'runs/x/results/responses.jsonl'; p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        verify(tmp_path/'plan.json',tmp_path)


def test_grouped_units_keep_singleton_generation_and_resume(tmp_path):
    rows=[{'id':f'q{i}','prompt':'a','gold':'#### 1','prompt_tokens':1} for i in range(3)]
    cfg={**config(),'unit_size':2}
    engine=Engine()
    result=collect(cfg,rows,tmp_path,{'device':'fixture'},engine,max_new_units=1)
    assert result['completed']==2 and len(engine.calls)==2
    result=collect(cfg,rows,tmp_path,{'device':'fixture'},engine)
    assert result['state']=='complete' and len(engine.calls)==9
    assert len(list((tmp_path/'units').glob('*/complete.json')))==6


def test_interrupted_group_repeats_only_unsealed_calls(tmp_path):
    rows=[{'id':f'q{i}','prompt':'a','gold':'#### 1','prompt_tokens':1} for i in range(3)]
    cfg={**config(),'unit_size':2}
    engine=Engine()
    collect(cfg,rows,tmp_path,{'device':'fixture'},engine,max_new_units=1)
    class Fails(Engine):
        def generate(self, rows, variant, seed):
            if self.calls: raise RuntimeError('interrupted')
            return super().generate(rows,variant,seed)
    interrupted=Fails()
    with pytest.raises(RuntimeError,match='interrupted'):
        collect(cfg,rows,tmp_path,{'device':'fixture'},interrupted)
    resumed=Engine()
    assert collect(cfg,rows,tmp_path,{'device':'fixture'},resumed)['state']=='complete'
    assert len(resumed.calls)==7
    assert len(list((tmp_path/'units').glob('*/complete.json')))==6


def test_snapshot_checks_all_indexed_weight_shards(tmp_path):
    from pilot_eval.sampling_stage0_evaluation import snapshot_files
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'a.safetensors','y':'b.safetensors'}}))
    (tmp_path/'a.safetensors').write_bytes(b'a')
    with pytest.raises((ValueError,FileNotFoundError)):
        snapshot_files(tmp_path)
    (tmp_path/'b.safetensors').write_bytes(b'bc')
    assert snapshot_files(tmp_path)['weight_sizes']=={'a.safetensors':1,'b.safetensors':2}
