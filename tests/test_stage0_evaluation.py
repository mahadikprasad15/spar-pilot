"""Public paired collection contract; synthetic CPU models, no downloads."""
import json
import shutil
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.training import file_hash
from test_stage0_workflow import write_json


@pytest.fixture(scope='module')
def frozen_root(tmp_path_factory):
    from test_stage0_edits import prepared, edits
    from test_stage0_qualification import TinyDiagnostic
    from pilot_eval.stage0_qualification import prepare, run, approve, freeze
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import settle
    root=tmp_path_factory.mktemp('paired-source')
    edit=prepared(root)
    assert edits(root,'run',edit)==0
    cohort=root/'cohort.json'
    write_json(cohort,dict(items=[dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 2')
                                 for i in range(150)],
        prompt_template=dict(decoding=dict(eos_token_id=99,pad_token_id=98))))
    plan=prepare(edit,cohort,root,'qualify')
    for event,total in [('before',1.),('after',1.02)]:
        observation=root/'usage.json'
        write_json(observation,dict(event_id=event,session_id='session-a',total_units=total,
            observed_at='2026-10-26T10:00:00+00:00',note='synthetic cumulative usage'))
        record_budget(root/'plans/audit/stage0.plan.json',root,observation)
        if event=='before':
            review=root/'diagnostic-approval.json'
            write_json(review,dict(reviewer='test',notes='fixture only',session_id='session-a',
                max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
            result=run(plan,root,review,dependencies=TinyDiagnostic())
        else:
            settle(root,result['reservation_id'])
    complete=root/'runs/stage0/qualify/qualification/complete.json'
    review=root/'review.json'
    write_json(review,dict(reviewer='test',notes='fixture only',diagnostic_sha256=file_hash(complete),
                          absolute_gate=result['proposed_absolute_gate']))
    approve(plan,root,review)
    write_json(review,dict(reviewer='test',notes='fixture budget',diagnostic_sha256=file_hash(complete),
        max_units_per_hour=1.,max_seconds=3600.,session_id='session-a',source_reviewed=True))
    freeze(plan,root,review)
    return root


@pytest.fixture
def source(tmp_path,frozen_root):
    shutil.copytree(frozen_root,tmp_path,dirs_exist_ok=True)
    return tmp_path


class TinyEvaluation:
    def __init__(self,session='session-a',fail_after=None):
        self.session=session;self.calls=[];self.loads=0;self.fail_after=fail_after
    def runtime(self):
        return dict(execution_kind='synthetic-cpu-fixture',session_id=self.session,device='tiny CPU oracle')
    def engine(self,config,root):
        self.loads+=1;return self
    def prepare_inputs(self,rows):
        return [dict(id=r['id'],input_ids=[10,int(r['id'].rsplit(':',1)[-1])+200],attention_mask=[1,1]) for r in rows]
    def generate(self,rows,variant,seed):
        if self.fail_after is not None and len(self.calls)>=self.fail_after:
            raise RuntimeError('injected interruption')
        self.calls.append((variant,[r['id'] for r in rows],seed))
        result=[]
        for row in rows:
            i=int(row['id'].rsplit(':',1)[-1])
            if i%4==3:
                result.append(dict(text='#### 2',token_ids=[7]*1024,
                    generated_token_ids=[7]*1024,stop_reason='cap'))
            else:
                text=['#### 2',r'\boxed{2}','#### 9'][i%4]
                result.append(dict(text=text,token_ids=[7,8],generated_token_ids=[7,8,99],stop_reason='eos'))
        return result
    def check_integrity(self,*,full=False):
        return dict(base_unchanged=True,kind='full-base-sha256' if full else 'parameter-identity-version-guard')
    def close(self):pass


def command(root,op,*args,dependencies=None):
    return main(['stage0-evaluation-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def prepared(root):
    execution=root/'plans/qualify/stage0.execution.json'
    assert command(root,'prepare','--execution',execution,'--batch-size',64,'--batches-per-unit',1)==0
    return root/'plans/qualify/stage0.evaluation.json'


def test_public_resume_complete_and_cpu_verify(source):
    plan=prepared(source);engine=TinyEvaluation()
    assert command(source,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    directory=source/'runs/stage0/qualify/paired-evaluation'
    first=next((directory/'units').glob('*/complete.json'));sealed=first.read_bytes()
    assert command(source,'status','--config',plan)==0
    assert command(source,'run','--config',plan,dependencies=engine)==0
    assert len(engine.calls)==30 and engine.loads==2
    assert first.read_bytes()==sealed
    assert command(source,'verify','--config',plan)==0
    records=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len(records)==1500 and len({r['record_id'] for r in records})==1500
    summary=json.loads((directory/'results/results.json').read_text())
    assert summary['measurement_complete'] and not summary['real_gpu_measurement']
    assert summary['human_review_status']=='not-applied'
    for row in summary['variants'].values():
        assert row['n']==150 and row['flexible_correct']==76 and row['strict_correct']==38
        assert row['cap_count']==37
    assert records[0]['prompt']=='question 0' and records[0]['gold']=='#### 2'
    assert records[0]['generated_token_ids']==[7,8,99]
    assert records[0]['token_count']==2 and records[0]['generated_token_count']==3
    assert records[1]['flexible']['correct'] and not records[1]['strict']['correct']
    assert records[3]['capped'] and records[3]['reward']==0 and records[3]['flexible']['status']=='invalid'
    assert command(source,'run','--config',plan,dependencies=TinyEvaluation(session='other'))==0
    assert engine.loads==2
    # A clean process verifies the durable result without importing model libraries.
    import subprocess,sys
    script='''
import sys
class Block:
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','transformers','peft'}:
            raise AssertionError('CPU verification imported optional model runtime: '+name)
sys.meta_path.insert(0,Block())
from pilot_eval.cli import main
raise SystemExit(main(['stage0-evaluation-verify','--config',sys.argv[1],'--output-root',sys.argv[2]]))
'''
    child=subprocess.run([sys.executable,'-c',script,str(plan),str(source)],text=True,capture_output=True)
    assert child.returncode==0,child.stderr
    # Verification reconstructs records and summaries, rather than trusting a DONE flag.
    shard=directory/'units'/json.loads(plan.read_text())['units'][0]['key']/'responses.jsonl'
    shard.write_text(shard.read_text().replace('#### 2','#### 3',1))
    assert command(source,'verify','--config',plan)==1


def test_same_runtime_budget_and_altered_source_rejected_before_inference(source):
    from pilot_eval.stage0_budget import settle
    from pilot_eval.stage0_workflow import record_budget
    plan=prepared(source);engine=TinyEvaluation(session='other')
    assert command(source,'run','--config',plan,dependencies=engine)==1
    assert engine.loads==0
    execution=json.loads((source/'plans/qualify/stage0.execution.json').read_text())
    observation=source/'after.json'
    write_json(observation,dict(event_id='stopped',session_id='session-a',total_units=1.1,
        observed_at='2026-10-26T12:00:00+00:00',note='cumulative'))
    record_budget(source/'plans/audit/stage0.plan.json',source,observation)
    settle(source,execution['reservation']['reservation_id'])
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,dependencies=engine)==1
    assert engine.loads==0
    review=source/'resume-budget.json'
    write_json(review,dict(reviewer='test',notes='new bounded attempt',session_id='session-a',
        max_units_per_hour=1.,max_seconds=60.,source_reviewed=True))
    assert command(source,'run','--config',plan,'--approval',review,'--max-new-units',1,dependencies=engine)==0
    assert engine.loads==1
    (source/'cohort.json').write_text('{}')
    assert command(source,'run','--config',plan,dependencies=engine)==1
    assert engine.loads==1


def test_interruption_unsealed_payload_and_wrong_metadata(source):
    plan=prepared(source);engine=TinyEvaluation(fail_after=1)
    assert command(source,'run','--config',plan,dependencies=engine)==1
    directory=source/'runs/stage0/qualify/paired-evaluation'
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'
    assert len(list((directory/'units').glob('*/complete.json')))==1
    # A partially written batch is never counted; it can be recomputed on resume.
    config=json.loads(plan.read_text());folder=directory/'units'/config['units'][1]['key']
    folder.mkdir(parents=True,exist_ok=True);(folder/'responses.jsonl').write_text('partial bytes')
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,dependencies=engine)==0
    assert len(engine.calls)==29
    assert command(source,'verify','--config',plan)==0


def test_eos_cap_and_output_count_contract():
    from pilot_eval.stage0_evaluation_records import records
    config=dict(decoding=dict(max_new_tokens=4,eos_token_id=99),scorer='gsm8k-flexible-v3')
    unit=dict(key='V0a-batch-0000',variant='V0a',ids=['q'],seed=42)
    rows=[dict(id='q',prompt='question',gold='#### 2')]
    tokens=[1,2,3,99]
    output=dict(text='#### 2',token_ids=tokens[:-1],generated_token_ids=tokens,stop_reason='eos')
    result=records(config,unit,rows,[output],runtime_sha256='runtime',input_sha256='inputs')
    assert result[0]['reward']==1 and not result[0]['capped']
    for bad in [{**output,'stop_reason':'cap'}, {**output,'generated_token_ids':[1,2,99,99]}]:
        with pytest.raises(ValueError,match='token|EOS|cap'):
            records(config,unit,rows,[bad],runtime_sha256='runtime',input_sha256='inputs')
    with pytest.raises(ValueError,match='count'):
        records(config,unit,rows,[],runtime_sha256='runtime',input_sha256='inputs')


def test_checkpoint_grouping_keeps_inference_batches_and_seed_schedule_fixed(source):
    execution=source/'plans/qualify/stage0.execution.json'
    assert command(source,'prepare','--execution',execution,'--batch-size',4,'--batches-per-unit',2)==0
    plan=source/'plans/qualify/stage0.evaluation.json';config=json.loads(plan.read_text())
    assert len(config['units'])==190
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    assert [len(ids) for _,ids,_ in engine.calls]==[4,4]
    assert [seed for _,_,seed in engine.calls]==[42,43]
    directory=source/'runs/stage0/qualify/paired-evaluation'
    shard=directory/'units'/config['units'][0]['key']/'responses.jsonl'
    rows=[json.loads(line) for line in shard.read_text().splitlines()]
    assert [r['seed'] for r in rows]==[42]*4+[43]*4
    assert json.loads((directory/'meta/status.json').read_text())['completed_responses']==8


def test_writer_recovery_and_cumulative_time_are_not_reset(source):
    import os,platform
    plan=prepared(source);directory=source/'runs/stage0/qualify/paired-evaluation'
    owner=directory/'writer.json'
    write_json(owner,dict(token='known',pid=os.getpid(),host=platform.node()))
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,dependencies=engine)==1
    assert engine.loads==0
    assert command(source,'recover-writer','--config',plan,'--owner-token','known','--review-notes','checked')==1
    write_json(owner,dict(token='known',pid=1234,host='disconnected-colab'))
    assert command(source,'recover-writer','--config',plan,'--owner-token','wrong','--review-notes','checked')==1
    assert command(source,'recover-writer','--config',plan,'--owner-token','known','--review-notes','old runtime stopped')==0
    assert command(source,'run','--config',plan,'--max-new-units',1,dependencies=engine)==0
    timing=next((directory/'attempts').glob('*/timing.json'))
    saved=json.loads(timing.read_text());saved['seconds']=3600.;write_json(timing,saved)
    assert command(source,'run','--config',plan,dependencies=engine)==1
    assert engine.loads==1


def test_unknown_attempt_time_and_ceiling_refuse_model_loading(source):
    from pilot_eval.stage0_workflow import record_budget
    plan=prepared(source);directory=source/'runs/stage0/qualify/paired-evaluation'
    execution=json.loads((source/'plans/qualify/stage0.execution.json').read_text())
    write_json(directory/'attempts/killed/start.json',dict(reservation_id=execution['reservation']['reservation_id']))
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0
    (directory/'attempts/killed/start.json').unlink()
    obs=source/'ceiling.json'
    write_json(obs,dict(event_id='ceiling',session_id='session-a',total_units=20.,
        observed_at='2026-10-26T12:00:00+00:00',note='actual cumulative usage'))
    record_budget(source/'plans/audit/stage0.plan.json',source,obs)
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0


def test_full_integrity_failure_cannot_be_resumed_as_good_evidence(source):
    class Corrupt(TinyEvaluation):
        def check_integrity(self,*,full=False):return dict(base_unchanged=not full,kind='full-base-sha256' if full else 'version-guard')
    plan=prepared(source)
    assert command(source,'run','--config',plan,'--max-new-units',1,dependencies=Corrupt())==1
    engine=TinyEvaluation()
    assert command(source,'run','--config',plan,dependencies=engine)==1 and engine.loads==0


def test_real_generation_boundary_selects_adapter_and_preserves_eos_cap(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from pilot_eval.stage0_evaluation_backend import EvaluationEngine
    seen={}
    class Model:
        def set_adapter(self,name):seen['variant']=name;self.active_adapters=[name]
        def eval(self):pass
        def generate(self,**kwargs):
            seen.update(kwargs)
            return np.concatenate([kwargs['input_ids'],np.array([[7,99,98,98],[7,7,7,7]])],axis=1)
    engine=EvaluationEngine.__new__(EvaluationEngine)
    engine.config=dict(variants={'V0a':{}},decoding=dict(pad_token_id=98,eos_token_id=99,max_new_tokens=4,use_cache=True))
    engine.inputs={'a':[1,2],'b':[3]};engine.model=Model()
    engine.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:' '.join(map(str,ids)))
    engine.torch=SimpleNamespace(tensor=lambda value,**kwargs:np.asarray(value),long='long',
        inference_mode=nullcontext,cuda=SimpleNamespace(synchronize=lambda:None))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(set_seed=lambda value:seen.update(seed=value)))
    result=engine.generate([{'id':'a'},{'id':'b'}],'V0a',42)
    assert seen['variant']=='V0a' and seen['do_sample'] is False and seen['use_cache'] is True
    assert seen['input_ids'].tolist()==[[1,2],[98,3]]
    assert seen['attention_mask'].tolist()==[[1,1],[0,1]]
    assert result[0]==dict(text='7',token_ids=[7],generated_token_ids=[7,99],stop_reason='eos')
    assert result[1]['stop_reason']=='cap' and len(result[1]['token_ids'])==4


def test_notebooks_expose_collection_and_cpu_inspection_without_hidden_gpu_setup():
    import ast
    for file,required in [('stage-0-gpu-colab.ipynb',['evaluation-prepare','evaluation-run','budget-settle']),
                          ('stage-0-cpu-colab.ipynb',['evaluation-status','evaluation-verify'])]:
        notebook=json.loads((Path('notebooks')/file).read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for command_name in required:assert command_name in code
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                assert cell['execution_count'] is None and cell['outputs']==[]
                ast.parse(''.join(cell['source']))


def test_generation_prompt_ids_must_match_the_qualified_inputs(tmp_path):
    from pilot_eval.stage0_evaluation import _inputs
    rows=[dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 2') for i in range(150)]
    write_json(tmp_path/'cohort.json',dict(items=rows))
    qualified=[dict(id=row['id'],input_ids=[10,i+200,7,8,99],prompt_tokens=2) for i,row in enumerate(rows[:16])]
    write_json(tmp_path/'qualified.json',dict(sequences=qualified))
    config=dict(cohort_path='cohort.json',item_ids=[r['id'] for r in rows],execution_kind='real-cuda',
                qualification_inputs_path='qualified.json')
    assert len(_inputs(TinyEvaluation(),config,tmp_path)['sequences'])==150
    qualified[0]['input_ids'][0]=11
    write_json(tmp_path/'qualified.json',dict(sequences=qualified))
    with pytest.raises(ValueError,match='prompt IDs'):
        _inputs(TinyEvaluation(),config,tmp_path)
