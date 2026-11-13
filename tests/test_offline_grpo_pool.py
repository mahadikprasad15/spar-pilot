"""Public offline-pool preparation and verified R0 handoff (CPU fixture)."""
import json

from pilot_eval.cli import main
from pilot_eval.workflow import _hash


def test_pool_cli_prepares_and_revalidates_complete_handoff(tmp_path, monkeypatch):
    from pilot_eval import offline_grpo_pool as pool
    from pilot_eval.stage0_rollout_math import normalize
    rows = [dict(id=f'q{i}', prompt=f'question {i}', gold='#### 2', input_ids=[11, i+100, 5], prompt_tokens=2) for i in range(512)]
    plan = dict(optimizer_prompt_order=[r['id'] for r in rows], model='fixture', model_revision='fixture', tokenizer_revision='fixture', dtype='float32')
    monkeypatch.setattr(pool, '_prepared', lambda path, root: (plan, rows, [dict(id='heldout')]))
    monkeypatch.setattr(pool, 'load_frozen', lambda path, root: dict(source_path='prepared.json', settings=dict(completion_limit=1024,top_p=1.,top_k=0), optimizer_groups=[plan['optimizer_prompt_order'][i:i+8] for i in range(0,512,8)]))
    source=tmp_path/'prepared.json'; source.write_text('{}')
    args=['--output-root',str(tmp_path)]
    assert main(['offline-pool-prepare', *args, '--source', str(source), '--name', 'example']) == 0
    config=tmp_path/'plans/example/offline.pool.json'
    assert json.loads(config.read_text())['route']=='unselected'
    records=[]
    for row in rows:
        for d in range(8):
            output=dict(text='#### 2' if d<3 else '#### 9', token_ids=[7], generated_token_ids=[7,99], stop_reason='eos')
            records.append(normalize(row,output,d,origin='new-uniform',seed=42,limit=1024,eos=99))
    monkeypatch.setattr(pool,'_verified_r0',lambda path,root:(records,dict(eos_token_id=99,route='uniform-fresh-512',execution_kind='synthetic-cpu-fixture', model_identity={k:plan[k] for k in ['model','model_revision','tokenizer_revision','dtype']})))
    r0=tmp_path/'r0.json';r0.write_text('{}')
    assert main(['offline-pool-import', *args, '--config',str(config),'--rollouts',str(r0)]) == 0
    windows=pool.load_pool(config,tmp_path)
    assert len(windows)==64 and len(windows[0])==8
    assert windows[0][0]['input_ids']==[11,100]
    assert windows[0][0]['draws'][0]['completion_mask']==[1,1]
    assert windows[0][0]['draws'][0]['reward_v3']==1
    payload=tmp_path/'runs/offline-grpo/example/pool/windows.json'
    data=json.loads(payload.read_text());data[0][0]['draws'][0]['reward_v3']=0;payload.write_text(json.dumps(data))
    assert main(['offline-pool-verify', *args, '--config',str(config)]) == 1


def test_mapping_rejects_duplicate_draw_and_changed_scores(tmp_path):
    import pytest
    from pilot_eval import offline_grpo_pool as pool
    from pilot_eval.stage0_rollout_math import normalize
    rows=[dict(id=f'q{i}',prompt=f'q{i}',gold='#### 2',input_ids=[1,2],prompt_tokens=1) for i in range(512)]
    identity=dict(model='fixture',model_revision='revision',tokenizer_revision='revision',dtype='float32')
    plan=dict(optimizer_order=[r['id'] for r in rows],source_identity=identity)
    metadata=dict(eos_token_id=99,model_identity=identity)
    records=[normalize(row,dict(text='#### 2',token_ids=[7],generated_token_ids=[7,99],stop_reason='eos'),
                       d,origin='fixture',seed=42,limit=1024,eos=99) for row in rows for d in range(8)]
    windows,summary=pool._windows(plan,rows,records,metadata)
    assert summary[0]['all_correct']==8 and summary[0]['tokens']==128
    assert windows[0][0]['draws'][0]['generated_token_ids'][-1]==99
    with pytest.raises(ValueError,match='Duplicate'):
        pool._windows(plan,rows,records+[records[0]],metadata)
    records[0]['reward_v3']=0
    with pytest.raises(ValueError,match='reconstruct'):
        pool._windows(plan,rows,records,metadata)


def test_inventory_reports_missing_historical_raw_files(tmp_path):
    from test_stage0_rollouts import sources
    audit,baseline,_=sources(tmp_path)
    (tmp_path/'runs/historical/batches/0001.json').unlink()
    assert main(['offline-pool-inventory','--output-root',str(tmp_path),'--audit',str(audit),
                 '--baseline',str(baseline),'--name','example'])==0
    report=json.loads((tmp_path/'runs/offline-grpo/example/pool/inputs/historical-inventory.json').read_text())
    assert report['state']=='incomplete' and report['verified_historical_groups']==127
    assert 'runs/historical/batches/0001.json' in report['missing']
    assert report['selected_route'] is None


def test_source_rejects_frozen_window_reordering(tmp_path, monkeypatch):
    import pytest
    from pilot_eval import offline_grpo_pool as pool
    rows=[dict(id=f'q{i}',prompt='q',gold='#### 2',input_ids=[1],prompt_tokens=1) for i in range(512)]
    order=[row['id'] for row in rows]
    groups=[order[i:i+8] for i in range(0,512,8)]
    groups[0][0],groups[0][1]=groups[0][1],groups[0][0]
    monkeypatch.setattr(pool,'load_frozen',lambda path,root:dict(source_path='prepared.json',settings=dict(completion_limit=1024,top_p=1.,top_k=0),optimizer_groups=groups))
    monkeypatch.setattr(pool,'_prepared',lambda path,root:(dict(optimizer_prompt_order=order),rows,[dict(id='heldout')]))
    with pytest.raises(ValueError,match='windows differ'):
        pool._source(tmp_path/'frozen.json',tmp_path)


def test_source_rejects_heldout_overlap(tmp_path, monkeypatch):
    import pytest
    from pilot_eval import offline_grpo_pool as pool
    rows=[dict(id=f'q{i}',prompt='q',gold='#### 2',input_ids=[1],prompt_tokens=1) for i in range(512)]
    order=[row['id'] for row in rows]
    monkeypatch.setattr(pool,'load_frozen',lambda path,root:dict(source_path='prepared.json',settings=dict(completion_limit=1024,top_p=1.,top_k=0),optimizer_groups=[order[i:i+8] for i in range(0,512,8)]))
    monkeypatch.setattr(pool,'_prepared',lambda path,root:(dict(optimizer_prompt_order=order),rows,[dict(id='q3')]))
    with pytest.raises(ValueError,match='Held-out'):
        pool._source(tmp_path/'frozen.json',tmp_path)


def test_cap_forces_zero_reward_but_all_tokens_remain_in_mask():
    from pilot_eval.stage0_rollout_math import normalize
    row=dict(id='q',prompt='q',gold='#### 2')
    raw=dict(text='#### 2',token_ids=[7]*1024,generated_token_ids=[7]*1024,stop_reason='cap')
    draw=normalize(row,raw,0,origin='fixture',seed=42,limit=1024,eos=99)
    assert draw['reward_v3']==0 and draw['capped'] and len(draw['generated_token_ids'])==1024


def test_fresh_collector_public_cli_is_available(tmp_path):
    import pytest
    from pilot_eval.cli import main
    with pytest.raises(SystemExit) as error:
        main(['offline-pool-fresh-prepare','--help'])
    assert error.value.code == 0


def test_fresh_public_workflow_resumes_and_imports_cpu_fixture(tmp_path, monkeypatch):
    from pilot_eval import offline_grpo_pool as pool, offline_grpo_fresh as fresh
    rows=[dict(id=f'q{i}',prompt=f'q{i}',gold='#### 2',input_ids=[1,i+100,9],prompt_tokens=2) for i in range(512)]
    order=[r['id'] for r in rows]
    identity=dict(model='fixture',model_revision='revision',tokenizer_revision='revision',dtype='float32',base_sha256='base')
    prepared=dict(optimizer_prompt_order=order,**{k:identity[k] for k in ['model','model_revision','tokenizer_revision','dtype']})
    monkeypatch.setattr(pool,'load_frozen',lambda path,root:dict(source_path='prepared.json',settings=dict(completion_limit=1024,top_p=1.,top_k=0),optimizer_groups=[order[i:i+8] for i in range(0,512,8)]))
    monkeypatch.setattr(pool,'_prepared',lambda path,root:(prepared,rows,[dict(id='heldout')]))
    monkeypatch.setattr(fresh,'status',lambda path,root:dict(audit=dict(primary_sources_ready=True,sources=dict(sft=dict(identity=identity)))))
    monkeypatch.setattr(fresh,'_reservation',lambda *args:(dict(reservation_id='fixture'),1000))
    runtime=dict(execution_kind='synthetic-cpu-fixture',session_id='fixture')
    class Dependencies:
        def inventory(self,directory,source):return dict(execution_kind='synthetic-cpu-fixture',base_sha256='base',parents={})
        def runtime(self):return runtime
        def engine(self,config,root):return self
        def special_tokens(self):return dict(eos_token_id=99,pad_token_id=98)
        def prepare_inputs(self,items):return [dict(id=r['id'],input_ids=r['input_ids'][:r['prompt_tokens']]) for r in items]
        def sample(self,items,settings,seed):return [dict(text='#### 2',token_ids=[7],generated_token_ids=[7,99],stop_reason='eos') for _ in range(8)]
        def check_integrity(self,full=False):return dict(base_unchanged=True)
        def close(self):pass
    (tmp_path/'prepared.json').write_text('{}');(tmp_path/'frozen.json').write_text('{}');(tmp_path/'audit.json').write_text('{}')
    weights=tmp_path/'weights';weights.mkdir()
    (weights/'tokenizer.json').write_text('{}');(weights/'tokenizer_config.json').write_text('{}')
    (weights/'config.json').write_text(json.dumps(dict(eos_token_id=99)))
    runtime_path=tmp_path/'runtime.json';runtime_path.write_text(json.dumps(runtime))
    root=['--output-root',str(tmp_path)]
    assert main(['offline-pool-prepare',*root,'--source',str(tmp_path/'frozen.json'),'--name','fresh'])==0
    plan=tmp_path/'plans/fresh/offline.pool.json'
    deps=Dependencies()
    monkeypatch.chdir(tmp_path.parent)
    assert main(['offline-pool-fresh-prepare',*root,'--pool',str(plan.relative_to(tmp_path)),'--audit',str(tmp_path/'audit.json'),'--weights-dir',str(weights),'--runtime',str(runtime_path)],dependencies=deps)==0
    fresh_plan=tmp_path/'runs/offline-grpo/fresh/pool/fresh.plan.json'
    assert main(['offline-pool-fresh-run',*root,'--config',str(fresh_plan),'--max-new-units','1'],dependencies=deps)==0
    attempts=tmp_path/'runs/offline-grpo/fresh/pool/fresh/attempts'
    first_attempt=next(attempts.iterdir())
    final_integrity=first_attempt/'final-integrity.json'
    saved_integrity=final_integrity.read_bytes()
    final_integrity.unlink()
    assert main(['offline-pool-fresh-run',*root,'--config',str(fresh_plan)],dependencies=deps)==1
    final_integrity.write_bytes(saved_integrity)
    runtime['session_id']='other-session'
    assert main(['offline-pool-fresh-run',*root,'--config',str(fresh_plan)],dependencies=deps)==1
    runtime['session_id']='fixture'
    for name in ['final-integrity.json','timing.json','timing.complete.json']:(first_attempt/name).unlink()
    assert main(['offline-pool-fresh-run',*root,'--config',str(fresh_plan)],dependencies=deps)==0
    assert main(['offline-pool-fresh-verify',*root,'--config',str(fresh_plan)])==0
    assert main(['offline-pool-import',*root,'--config',str(plan),'--rollouts',str(fresh_plan)])==0
    assert len(pool.load_pool(plan,tmp_path))==64
    shard=tmp_path/'runs/offline-grpo/fresh/pool/fresh/units/group-0000/responses.jsonl'
    shard.write_text(shard.read_text()+'\n')
    assert main(['offline-pool-fresh-verify',*root,'--config',str(fresh_plan)])==1
    import pytest
    with pytest.raises(ValueError,match='payload changed'):
        pool.load_pool(plan,tmp_path)


def test_fresh_collection_reservation_requires_recorded_budget(tmp_path):
    import pytest
    from pilot_eval.stage0_evaluation import _reservation
    with pytest.raises(ValueError, match='usage'):
        _reservation({}, {'reservation': {'reservation_id': None, 'job_sha256': 'job'}},
                     tmp_path/'run',tmp_path,{'session_id':'fixture'},None)


def test_source_rejects_sampling_settings_that_differ_from_online_plan(tmp_path, monkeypatch):
    import pytest
    from pilot_eval import offline_grpo_pool as pool
    rows=[dict(id=f'q{i}',prompt='q',gold='#### 2',input_ids=[1],prompt_tokens=1) for i in range(512)]
    order=[row['id'] for row in rows]
    monkeypatch.setattr(pool,'_prepared',lambda path,root:(dict(optimizer_prompt_order=order),rows,[dict(id='heldout')]))
    for change in [dict(completion_limit=512),dict(top_p=.9),dict(top_k=50)]:
        settings=dict(dict(completion_limit=1024,top_p=1.,top_k=0),**change)
        monkeypatch.setattr(pool,'load_frozen',lambda path,root:dict(source_path='prepared.json',settings=settings,optimizer_groups=[order[i:i+8] for i in range(0,512,8)]))
        with pytest.raises(ValueError,match='sampling settings'):
            pool._source(tmp_path/'frozen.json',tmp_path)


def test_verified_records_validates_saved_groups_once(tmp_path, monkeypatch):
    from pilot_eval import offline_grpo_fresh as fresh
    config=dict(model_identity=dict(model='m',model_revision='r',tokenizer_revision='r',dtype='float32'),execution_kind='x',
                units=[],expected_eos_token_id=99,runtime={'r':1},run_path='run')
    calls=[]
    monkeypatch.setattr(fresh,'_verified',lambda path,root:calls.append(path) or (config,[[dict(id=1)]],dict(eos_token_id=99)))
    records,metadata=fresh.verified_records(tmp_path/'plan.json',tmp_path)
    assert records==[dict(id=1)] and metadata['eos_token_id']==99 and len(calls)==1
