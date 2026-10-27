"""Stage 0 untuned rollout source/import/cohort workflow; CPU fixtures only."""
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash
from test_stage0_workflow import write_json


def cmd(root,operation,*args,dependencies=None):
    return main(['stage0-rollouts-'+operation,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def output(row,draw):
    i=int(row['id'].rsplit(':',1)[-1]);correct=i%7!=0 and draw<3
    return dict(text='#### 2' if correct else '#### 9',token_ids=[7 if correct else 8],
                generated_token_ids=[7 if correct else 8,99],stop_reason='eos')


def sources(root,*,gold_eos=11):
    from test_stage0_workflow import sources as audit_sources,command
    from pilot_eval.grpo_baseline import _score
    selected=audit_sources(root)
    assert command(root,'prepare','--selection',selected,'--name','audit')==0
    audit=root/'plans/audit/stage0.plan.json'
    assert command(root,'audit','--config',audit)==0
    rows=[dict(id=f'gsm8k:train:{i}',source_index=i,source_split='train',prompt=f'question {i}',gold='#### 2',
               input_ids=[10,i+1000,gold_eos],prompt_tokens=2,labels=[-100,-100,gold_eos]) for i in range(512)]
    write_json(root/'training.json',rows)
    contract=dict(prompt_template='fixture',chat_template_sha256='template',attention_implementation='eager')
    plan=dict(protocol_version='pilot4-grpo-preparation-v1',model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',
        tokenizer_revision='pinned',dtype='float32',seed=42,training_items_path='training.json',training_items_sha256=_hash(rows),
        optimizer_prompt_order=[r['id'] for r in rows],training_indices=list(range(512)),source_prompt_contract=contract)
    source=root/'plans/grpo/grpo.prepared.json';write_json(source,plan)
    write_json(source.parent/'prepare-complete.json',dict(config_sha256=_hash(plan),files={str(source.relative_to(root)):file_hash(source)}))
    ids=[r['id'] for r in rows[:128]][::-1]
    settings=dict(subset_ids=ids,seed=42,groups_per_batch=1,max_new_tokens=1024,temperature=1.,top_p=1.,top_k=0,scorer='gsm8k-flexible-v3')
    from pilot_eval import grpo_baseline,scoring
    config=dict(protocol_version='pilot4-sampling-baseline-v1',model=plan['model'],model_revision='pinned',tokenizer_revision='pinned',dtype='float32',adapter=None,
        source_path=str(source.relative_to(root)),source_sha256=file_hash(source),settings=settings,group_size=8,
        decoding=dict(do_sample=True,num_beams=1,repetition_penalty=1.,no_repeat_ngram_size=0,temperature=1.,top_p=1.,top_k=0,max_new_tokens=1024),
        prompt_template=contract,implementation_sha256=dict(sampling=file_hash(Path(grpo_baseline.__file__)),scorer=file_hash(Path(scoring.__file__))),
        run_path='runs/historical',batches=[dict(ids=[i],seed=42+n) for n,i in enumerate(ids)])
    baseline=root/'plans/baseline/baseline.config.json';write_json(baseline,config)
    directory=root/config['run_path'];write_json(directory/'config.json',config)
    write_json(directory/'meta/runtime.json',dict(dtype='float32',device='tiny CPU',eos_token_id=99,pad_token_id=98,tf32=False))
    by_id={r['id']:r for r in rows}
    for index,batch in enumerate(config['batches']):
        row=by_id[batch['ids'][0]];path=directory/'batches'/f'{index:04d}.json'
        write_json(path,[_score(row,output(row,d),d) for d in range(8)])
        write_json(path.with_suffix('.complete.json'),dict(sha256=file_hash(path),config_sha256=_hash(config),runtime_sha256=file_hash(directory/'meta/runtime.json')))
    weights=root/'weights';weights.mkdir();np.save(weights/'fixture.npy',np.eye(2,dtype=np.float32))
    return audit,baseline,weights


class Tiny:
    def inventory(self,directory,audit):
        return dict(execution_kind='synthetic-cpu-fixture',base_sha256=audit['identity']['base_sha256'],
            parents={str(directory/'fixture.npy'):file_hash(directory/'fixture.npy')})
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id='session-a',device='tiny CPU')
    def engine(self,config,root):self.config=config;return self
    def prepare_inputs(self,rows):return [dict(id=r['id'],input_ids=r['input_ids'][:r['prompt_tokens']]) for r in rows]
    def sample(self,rows,settings,seed):return [output(row,d) for row in rows for d in range(8)]
    def check_integrity(self,*,full=False):return dict(base_unchanged=True)
    def close(self):pass


def test_source_audit_reports_missing_raw_payloads_before_any_gpu_work(tmp_path):
    audit,baseline,_=sources(tmp_path)
    missing=tmp_path/'runs/historical/batches/0001.json';missing.unlink()
    assert cmd(tmp_path,'audit','--audit',audit,'--baseline',baseline,'--name','r0')==0
    report=json.loads((tmp_path/'runs/stage0/r0/rollouts/inputs/source-audit.json').read_text())
    assert report['state']=='incomplete' and report['verified_historical_groups']==127
    assert 'runs/historical/batches/0001.json' in report['missing']
    assert not (tmp_path/'plans/r0/stage0.rollouts.json').exists()


def test_dataset_memberships_keep_zero_success_questions_and_one_draw_selection():
    from pilot_eval.stage0_rollout_math import cohorts,normalize
    order=['q2','q0','q1'];training=[dict(id=i,prompt=i,gold='#### 2') for i in order]
    records=[]
    for i in order:
        row=next(r for r in training if r['id']==i)
        for d in range(8):
            value=dict(text='#### 2' if i!='q0' and d<2 else '#### 9',token_ids=[7],generated_token_ids=[7,99],stop_reason='eos')
            records.append(normalize(row,value,d,origin='new',seed=42,limit=1024,eos=99))
    result=cohorts(records,order,selection_seed=42)
    assert len(result['own_all'])==24 and len(result['own_correct'])==4 and len(result['own_correct_one'])==2
    assert result['gold']==order and result['dropped']==['q0']
    assert result['p_correct']['q0']==0 and result['neg_log_p_correct']['q0'] is None
    assert result['own_correct_one']==cohorts(records,order,selection_seed=42)['own_correct_one']
    with pytest.raises(ValueError,match='eight|duplicate'):cohorts(records+[records[0]],order,selection_seed=42)


@pytest.fixture(scope='module')
def complete_sources(tmp_path_factory):
    return_root=tmp_path_factory.mktemp('r0-sources');sources(return_root);return return_root


@pytest.fixture
def ready(tmp_path,complete_sources):
    import shutil
    shutil.copytree(complete_sources,tmp_path,dirs_exist_ok=True)
    return tmp_path


def usage(root,event,total):
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import reservations,settle
    path=root/'usage.json';write_json(path,dict(event_id=event,session_id='session-a',total_units=total,
        observed_at='2026-10-27T12:00:00+00:00',note='synthetic cumulative Stage 0 usage'))
    record_budget(root/'plans/audit/stage0.plan.json',root,path)
    for key,row in reservations(root).items():
        if row['state']=='active':settle(root,key)


def prepare(root):
    assert cmd(root,'prepare','--audit',root/'plans/audit/stage0.plan.json','--baseline',root/'plans/baseline/baseline.config.json',
        '--weights-dir',root/'weights','--name','r0',dependencies=Tiny())==0
    return root/'plans/r0/stage0.rollouts.json'


class Sampling(Tiny):
    def __init__(self,*,mismatch=False,session='session-a',fail_after=None):
        self.calls=[];self.loads=0;self.mismatch=mismatch;self.session=session;self.fail_after=fail_after
    def runtime(self):return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU')
    def engine(self,config,root):self.loads+=1;return super().engine(config,root)
    def special_tokens(self):return dict(eos_token_id=99,pad_token_id=98)
    def sample(self,rows,settings,seed):
        if self.fail_after is not None and len(self.calls)>=self.fail_after:raise RuntimeError('injected interruption')
        self.calls.append((rows[0]['id'],seed));values=super().sample(rows,settings,seed)
        if self.mismatch:
            values[0]=dict(text='#### 3',token_ids=[6],generated_token_ids=[6,99],stop_reason='eos')
        return values


def profile_and_freeze(root,plan,engine):
    usage(root,'start',4.)
    review=root/'profile-review.json';write_json(review,dict(reviewer='test',notes='bounded eight-draw groups',
        session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    assert cmd(root,'profile','--config',plan,'--approval',review,'--max-new-units',1,dependencies=engine)==0
    profile=root/'runs/stage0/r0/rollouts/profile'
    marker=next((profile/'units').glob('*/complete.json'));before=marker.read_bytes()
    assert cmd(root,'profile','--config',plan,dependencies=engine)==0
    assert marker.read_bytes()==before and len(engine.calls)==2
    result=json.loads((profile/'results/profile.json').read_text())
    usage(root,'profile-complete',4.01)
    write_json(review,dict(reviewer='test',notes='approve evidence, route and forecast',session_id='session-a',
        source_reviewed=True,max_units_per_hour=1.,max_seconds=max(3600.,result['forecast']['conservative_total_seconds']),
        profile_sha256=file_hash(profile/'complete.json'),route=result['route']))
    assert cmd(root,'freeze','--config',plan,'--review',review)==0
    return result


@pytest.mark.parametrize('mismatch',[False,True])
def test_public_replay_route_generation_resume_and_verified_cohorts(ready,mismatch):
    root=ready;plan=prepare(root);engine=Sampling(mismatch=mismatch)
    config=json.loads(plan.read_text());assert config['uniform_order'][0]=='gsm8k:train:0'
    assert config['historical_order'][0]=='gsm8k:train:127'
    result=profile_and_freeze(root,plan,engine)
    assert result['route']==('uniform-fresh-512' if mismatch else 'reuse-historical-128')
    assert result['production_groups']==(512 if mismatch else 384)
    assert cmd(root,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    collection=root/'runs/stage0/r0/rollouts/collection'
    marker=next((collection/'units').glob('*/complete.json'));before=marker.read_bytes()
    assert cmd(root,'status','--config',plan)==0
    assert cmd(root,'run','--config',plan,dependencies=engine)==0
    assert marker.read_bytes()==before
    assert cmd(root,'verify','--config',plan)==0
    directory=root/'runs/stage0/r0/rollouts';summary=json.loads((directory/'results/results.json').read_text())
    rows=[json.loads(line) for line in (directory/'results/rollouts.jsonl').read_text().splitlines()]
    memberships=json.loads((directory/'results/datasets.json').read_text())
    assert len(rows)==4096 and len({r['draw_id'] for r in rows})==4096
    assert memberships['counts']['questions']==512 and len(memberships['own_all'])==4096
    assert summary['route']==result['route'] and not summary['real_gpu_measurement']
    origins={r['origin'] for r in rows}
    assert origins==({'new-uniform'} if mismatch else {'historical','new-extension'})
    assert (directory/'results/historical/rollouts.jsonl').exists()
    assert (directory/'results/rollouts.parquet').exists()
    assert json.loads((collection/'meta/status.json').read_text())['state']=='complete'
    first_new=engine.calls[2]
    assert first_new==('gsm8k:train:0',42) if mismatch else first_new==('gsm8k:train:128',170)
    calls=len(engine.calls);assert cmd(root,'run','--config',plan,dependencies=engine)==0 and len(engine.calls)==calls
    import pyarrow.parquet as pq
    assert pq.read_table(directory/'results/rollouts.parquet').num_rows==4096
    assert cmd(root,'audit','--audit',root/'plans/audit/stage0.plan.json','--baseline',root/'plans/baseline/baseline.config.json','--name','r0')==0
    assert json.loads((directory/'meta/status.json').read_text())['state']=='complete'
    payload=root/'runs/historical/batches/0000.json';payload.write_text('[]')
    assert cmd(root,'verify','--config',plan)==1


def test_missing_recorded_seeds_forces_uniform_route_and_reuses_timing_groups(ready):
    root=ready;path=root/'plans/baseline/baseline.config.json';config=json.loads(path.read_text());config.pop('batches')
    write_json(path,config);write_json(root/'runs/historical/config.json',config)
    for marker in (root/'runs/historical/batches').glob('*.complete.json'):
        value=json.loads(marker.read_text());value['config_sha256']=_hash(config);write_json(marker,value)
    plan=prepare(root);engine=Sampling();profile=profile_and_freeze(root,plan,engine)
    assert profile['route']=='uniform-fresh-512' and profile['profile_groups_reusable']==2
    assert not profile['policy']['replay_eligible']
    assert cmd(root,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    assert len(engine.calls)==2 # the first production group uses its sealed timing result


def test_unknown_budget_and_changed_runtime_cannot_load_sampler(ready):
    root=ready;plan=prepare(root);engine=Sampling()
    assert cmd(root,'runtime','--config',plan,dependencies=engine)==0
    review=root/'review.json';write_json(review,dict(reviewer='test',notes='bounded',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=60.))
    assert cmd(root,'profile','--config',plan,'--approval',review,dependencies=engine)==1
    assert engine.loads==0
    usage(root,'at-limit',19.99)
    assert cmd(root,'profile','--config',plan,'--approval',review,dependencies=engine)==1 and engine.loads==0


def test_full_integrity_failure_invalidates_already_sealed_groups(ready):
    root=ready;plan=prepare(root);usage(root,'initial',2.)
    review=root/'review.json';write_json(review,dict(reviewer='test',notes='bounded',session_id='session-a',source_reviewed=True,max_units_per_hour=1.,max_seconds=600.))
    class Broken(Sampling):
        def check_integrity(self,*,full=False):return dict(base_unchanged=not full)
    assert cmd(root,'profile','--config',plan,'--approval',review,dependencies=Broken())==1
    engine=Sampling()
    assert cmd(root,'profile','--config',plan,dependencies=engine)==1 and engine.loads==0
    assert not (root/'runs/stage0/r0/rollouts/profile/complete.json').exists()


def test_historical_environment_mismatch_requires_fresh_uniform_timing():
    from pilot_eval.stage0_rollout_plan import replay_policy
    import pilot_eval.grpo_baseline as old
    config=dict(seeds_recovered=True,policy_matches=True,legacy=dict(implementation_sha256=dict(sampling=file_hash(Path(old.__file__))),batches=[dict(seed=42),dict(seed=43)]),
        historical_runtime=dict(device='original GPU',versions=dict(torch='2.9.0')),historical_order=['old-a','old-b'],
        uniform_units=[dict(id='new-a',seed=42),dict(id='new-b',seed=43)])
    result=replay_policy(config,dict(device='new GPU',versions=dict(torch='2.9.0')))
    assert not result['replay_eligible'] and result['units'][0]['id']=='new-a'


def test_notebooks_expose_r0_without_requiring_primary_gpu_execution():
    import ast
    cpu=json.loads(Path('notebooks/stage-0-cpu-colab.ipynb').read_text())
    gpu=json.loads(Path('notebooks/stage-0-gpu-colab.ipynb').read_text())
    for notebook,names in [(cpu,['rollouts-audit','rollouts-prepare','rollouts-verify']),
                           (gpu,['rollouts-runtime','rollouts-profile','rollouts-freeze','rollouts-run','budget-settle'])]:
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for name in names:assert name in code
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None


def test_collection_rejects_new_session_before_model_load(ready):
    plan=prepare(ready);profile_and_freeze(ready,plan,Sampling())
    engine=Sampling(session='different-session')
    assert cmd(ready,'run','--config',plan,dependencies=engine)==1 and engine.loads==0


def test_recovery_is_explicit_and_cannot_release_a_live_writer(ready):
    import os,platform
    plan=prepare(ready);directory=ready/'runs/stage0/r0/rollouts/profile';directory.mkdir(parents=True)
    owner=directory/'writer.json';write_json(owner,dict(token='known',host=platform.node(),pid=os.getpid()))
    assert cmd(ready,'recover-writer','--config',plan,'--phase','profile','--owner-token','known','--review-notes','inspect')==1
    write_json(owner,dict(token='known',host='departed-colab',pid=123))
    assert cmd(ready,'recover-writer','--config',plan,'--phase','profile','--owner-token','wrong','--review-notes','stopped')==1
    assert cmd(ready,'recover-writer','--config',plan,'--phase','profile','--owner-token','known','--review-notes','stopped')==0
    assert not owner.exists() and (directory/'logs/ownership-recovery.jsonl').exists()


def test_untuned_hash_names_match_the_audited_peft_parameter_convention():
    from pilot_eval.stage0_rollout_backend import parameter_name
    assert parameter_name('model.layers.0.self_attn.q_proj.weight')=='base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight'
    assert parameter_name('model.layers.0.input_layernorm.weight')=='base_model.model.model.layers.0.input_layernorm.weight'
    assert parameter_name('lm_head.weight')=='base_model.model.lm_head.weight'


def test_real_sft_and_grpo_training_membership_use_their_actual_schemas(tmp_path):
    from pilot_eval.stage0_rollout_plan import validate_training_membership
    identity=dict(model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32')
    order=[f'gsm8k:train:{i}' for i in range(512)]
    sft=dict(identity,training_indices=list(range(512)))
    grpo=dict(identity,optimizer_groups=[order[i:i+8] for i in range(0,512,8)])
    write_json(tmp_path/'sft.json',sft);write_json(tmp_path/'grpo.json',grpo)
    source=dict(identity=identity,training=[dict(id=i,source_index=n) for n,i in enumerate(order)],training_order=order,
        source_audit=dict(sources={arm:dict(training_metadata=dict(state='verified-recorded-config',path=arm+'.json')) for arm in ['sft','grpo']}),
        parents={arm+'.json':file_hash(tmp_path/(arm+'.json')) for arm in ['sft','grpo']})
    validate_training_membership(source,tmp_path)
    grpo['optimizer_groups'][0][0]=order[1];write_json(tmp_path/'grpo.json',grpo)
    source['parents']['grpo.json']=file_hash(tmp_path/'grpo.json')
    with pytest.raises(ValueError,match='GRPO|optimizer'):validate_training_membership(source,tmp_path)
