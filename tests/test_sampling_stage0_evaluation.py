import json
import pytest
from pilot_eval.sampling_stage0_evaluation import collect


class Engine:
    def __init__(self): self.calls = []
    def prepare_inputs(self, rows):
        return [{'id': r['id'], 'input_ids': [7], 'attention_mask': [1]} for r in rows]
    def check_integrity(self, *, full=False): return {'base_unchanged': True, 'adapters_unchanged': True}
    def generate(self, rows, variant, seed):
        self.calls.append((variant, [r['id'] for r in rows]))
        return [{'text': '#### 1', 'token_ids': [1], 'generated_token_ids': [1, 2], 'stop_reason': 'eos'} for _ in rows]


def config(**extra):
    return {'variants': {k: {'adapter_path': k} for k in ('k0','k2','k5')},
            'variant_order': ['k0','k2','k5'], 'decoding': {'eos_token_id': 2, 'max_new_tokens': 10},
            'scorer': 'gsm8k-flexible-v3', 'seed': 42,
            'adapter_files': {k: {'adapter_model.safetensors': k} for k in ('k0','k2','k5')}, **extra}


def items(n):
    return [{'id': f'q{i}', 'prompt': 'a', 'gold': '#### 1', 'prompt_tokens': 1} for i in range(n)]


def test_resume_reconstructs_seals_and_only_runs_missing_units(tmp_path):
    rows = items(2)
    engine = Engine()
    first = collect(config(), rows, tmp_path, {'device': 'fixture'}, engine, max_new_units=2)
    assert first['state'] == 'incomplete' and len(engine.calls) == 4 # two singleton calls per arm group
    second = collect(config(), rows, tmp_path, {'device': 'fixture'}, engine)
    assert second['state'] == 'complete' and len(engine.calls) == 6
    assert len((tmp_path/'results/responses.jsonl').read_text().splitlines()) == 6
    unit = next((tmp_path/'units').glob('*/responses.jsonl'))
    unit.write_text(unit.read_text().replace('#### 1', '#### 2'))
    with pytest.raises(ValueError, match='changed|mismatch'):
        collect(config(), rows, tmp_path, {'device': 'fixture'}, engine)


def test_groups_follow_stage0_membership_and_per_arm_batch_size(tmp_path):
    rows = items(22) # one full group of 16 and a tail of 6, as 150 = 9 x 16 + 6
    engine = Engine()
    result = collect(config(batch_sizes={'k0': 16, 'k2': 1, 'k5': 16}), rows, tmp_path, {'device': 'fixture'}, engine)
    assert result['state'] == 'complete'
    batched = [ids for arm, ids in engine.calls if arm in ('k0', 'k5')]
    assert batched == [[f'q{i}' for i in range(16)]]*2 + [[f'q{i}' for i in range(16, 22)]]*2
    assert [ids for arm, ids in engine.calls if arm == 'k2'] == [[f'q{i}'] for i in range(22)]
    assert result['variants']['k0']['inference_batch_size'] == 16 and result['variants']['k2']['inference_batch_size'] == 1
    assert sorted(p.name for p in (tmp_path/'units').iterdir()) == ['0000-k0','0000-k2','0000-k5','0016-k0','0016-k2','0016-k5']


def test_unqualified_batch_size_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='qualified'):
        collect(config(batch_sizes={'k0': 8, 'k2': 1, 'k5': 1}), items(2), tmp_path, {'device': 'fixture'}, Engine())


def test_resume_rejects_changed_runtime(tmp_path):
    collect(config(), items(1), tmp_path, {'device': 'A'}, Engine(), max_new_units=1)
    with pytest.raises(ValueError, match='runtime'):
        collect(config(), items(1), tmp_path, {'device': 'B'}, Engine())


def test_environment_ignores_session_and_commit_only():
    from pilot_eval.sampling_stage0_evaluation import environment
    a = {'device': 'NVIDIA L4', 'driver': '580', 'session_id': 'boot-1', 'git_commit': 'abc', 'versions': {'torch': '2.9.0'}}
    b = {**a, 'session_id': 'boot-2', 'git_commit': 'def'}
    assert environment(a) == environment(b)
    assert environment(a) != environment({**a, 'driver': '570'})


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
    with pytest.raises(ValueError, match='integrity'):
        collect(config(), items(1), tmp_path, {'device':'A'}, Bad(), max_new_units=1)
    with pytest.raises(ValueError, match='integrity'):
        collect(config(), items(1), tmp_path, {'device':'A'}, Engine())


def test_completed_collection_is_read_only_on_repeat(tmp_path):
    engine=Engine()
    collect(config(),items(1),tmp_path,{'device':'A'},engine)
    path=tmp_path/'results/responses.jsonl'
    before=path.stat().st_mtime_ns
    collect(config(),items(1),tmp_path,{'device':'A'},engine)
    assert path.stat().st_mtime_ns == before and len(engine.calls)==3


def _plan(tmp_path, monkeypatch, rows, **extra):
    from pilot_eval import sampling_stage0_evaluation as module
    monkeypatch.setattr(module, 'check_sources', lambda cfg, root: rows)
    cfg={**config(),'implementation':{},'run_path':'runs/x','execution_kind':'real-cuda','gpu':'L4',**extra}
    (tmp_path/'plan.json').write_text(json.dumps(cfg))
    return tmp_path/'plan.json'


class FakeEngine(Engine):
    loaded_adapter_checks=[{'variant':k,'tensor_count':392,'exact_payload_match':True} for k in ('k0','k2','k5')]
    def smoke(self, rows): return [{'variant':arm,'id':row['id'],'exact_repeat':True} for arm in ('k0','k2','k5') for row in rows]
    def close(self): pass


class Dependencies:
    def __init__(self, session='boot-1'): self.model=FakeEngine();self.session=session
    def runtime(self): return {'execution_kind':'real-cuda','device':'NVIDIA L4','session_id':self.session,'git_commit':'c'}
    def engine(self,*args): return self.model
    def preflight(self,*args): pass


def test_run_rejects_wrong_gpu_before_loading_model(tmp_path, monkeypatch):
    from pilot_eval.sampling_stage0_evaluation import run
    plan=_plan(tmp_path, monkeypatch, items(1), gpu='A100')
    class Wrong(Dependencies):
        def engine(self,*args): raise AssertionError('must not load')
    with pytest.raises(ValueError,match='A100'):
        run(plan,tmp_path,dependencies=Wrong())


def test_run_smoke_precedes_generation_and_resumes_after_pod_restart(tmp_path, monkeypatch):
    from pilot_eval.sampling_stage0_evaluation import run
    plan=_plan(tmp_path, monkeypatch, items(2))
    deps=Dependencies('boot-1')
    assert run(plan,tmp_path,dependencies=deps,max_new_units=2)['state']=='incomplete'
    restarted=Dependencies('boot-2') # new boot ID after a Pod restart: same hardware and software
    assert run(plan,tmp_path,dependencies=restarted)['state']=='complete'
    assert len(deps.model.calls)+len(restarted.model.calls)==6
    assert (tmp_path/'runs/x/meta/direct-smoke.json').exists()
    assert 'session_id' not in json.loads((tmp_path/'runs/x/meta/runtime.json').read_text())


def test_batched_plan_requires_agreement_environment(tmp_path, monkeypatch):
    from pilot_eval import sampling_stage0_evaluation as module
    plan=_plan(tmp_path, monkeypatch, items(2), batch_sizes={'k0':16,'k2':16,'k5':16},
               batch_agreement={'environment':{'execution_kind':'real-cuda','device':'NVIDIA L4','driver':'other'}})
    monkeypatch.setattr(module, '_agreement', lambda cfg, root: None)
    with pytest.raises(ValueError,match='different hardware'):
        module.run(plan,tmp_path,dependencies=Dependencies())


def test_verify_rejects_changed_result_without_loading_gpu(tmp_path, monkeypatch):
    from pilot_eval.sampling_stage0_evaluation import verify
    rows=items(1)
    plan=_plan(tmp_path, monkeypatch, rows, execution_kind='cpu-fixture')
    cfg=json.loads(plan.read_text())
    collect(cfg,rows,tmp_path/'runs/x',{'device':'fixture'},Engine())
    assert verify(plan,tmp_path)['state']=='complete'
    p=tmp_path/'runs/x/results/responses.jsonl'; p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        verify(plan,tmp_path)


def test_interrupted_group_repeats_only_unsealed_groups(tmp_path):
    rows=items(3)
    engine=Engine()
    collect(config(),rows,tmp_path,{'device':'fixture'},engine,max_new_units=1)
    class Fails(Engine):
        def generate(self, rows, variant, seed):
            if self.calls: raise RuntimeError('interrupted')
            return super().generate(rows,variant,seed)
    interrupted=Fails()
    with pytest.raises(RuntimeError,match='interrupted'):
        collect(config(),rows,tmp_path,{'device':'fixture'},interrupted)
    resumed=Engine()
    assert collect(config(),rows,tmp_path,{'device':'fixture'},resumed)['state']=='complete'
    assert len(resumed.calls)==6 # two remaining arm groups of three singleton calls
    assert len(list((tmp_path/'units').glob('*/complete.json')))==3


def test_snapshot_checks_all_indexed_weight_shards(tmp_path):
    from pilot_eval.sampling_stage0_evaluation import snapshot_files
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'a.safetensors','y':'b.safetensors'}}))
    (tmp_path/'a.safetensors').write_bytes(b'a')
    with pytest.raises((ValueError,FileNotFoundError)):
        snapshot_files(tmp_path)
    (tmp_path/'b.safetensors').write_bytes(b'bc')
    assert snapshot_files(tmp_path)['weight_sizes']=={'a.safetensors':1,'b.safetensors':2}


def test_sources_require_eos_and_pad_ids_before_any_gpu_work(tmp_path):
    from pilot_eval import sampling_stage0_evaluation as module
    from pilot_eval.workflow import _hash
    rows=[{'id':str(i),'prompt':'p','gold':'#### 1'} for i in range(150)]
    gold={'model':'qwen','model_revision':'r','tokenizer_revision':'r','evaluation_items_path':'items.json',
          'evaluation_items_sha256':_hash(rows),'source_config':{'chat_template_sha256':'x',
          'decoding':{'do_sample':False,'num_beams':1,'num_return_sequences':1,'max_new_tokens':1024,'eos_token_id':2}}}
    (tmp_path/'gold.json').write_text(json.dumps(gold)); (tmp_path/'items.json').write_text(json.dumps(rows))
    (tmp_path/'manifest.json').write_text(json.dumps({'gold_config':'gold.json','weights_dir':'w',
        'training_plans':{a:f'{a}.json' for a in ('k0','k2','k5')}}))
    with pytest.raises(ValueError, match='EOS/PAD'):
        module.sources(tmp_path/'manifest.json', tmp_path)
