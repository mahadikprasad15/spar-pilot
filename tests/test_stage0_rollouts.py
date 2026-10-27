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


def sources(root):
    from test_stage0_workflow import sources as audit_sources,command
    from pilot_eval.grpo_baseline import _score
    selected=audit_sources(root)
    assert command(root,'prepare','--selection',selected,'--name','audit')==0
    audit=root/'plans/audit/stage0.plan.json'
    assert command(root,'audit','--config',audit)==0
    rows=[dict(id=f'gsm8k:train:{i}',source_index=i,source_split='train',prompt=f'question {i}',gold='#### 2',
               input_ids=[10,i+1000,11],prompt_tokens=2,labels=[-100,-100,11]) for i in range(512)]
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
