"""Conditional extension through the shared public workflow; no model downloads."""
import json
from pathlib import Path
import shutil

import pytest

from pilot_eval.cli import main
from pilot_eval.training import file_hash
from test_stage0_workflow import write_json


@pytest.fixture(scope='module')
def catalogue_root(tmp_path_factory):
    from test_stage0_intruder_edits import public_sources,command
    root=tmp_path_factory.mktemp('conditional-source')
    plan,deps=public_sources(root,True)
    assert command(root,'run','--config',plan,dependencies=deps)==0
    return root


@pytest.fixture
def source(tmp_path,catalogue_root):
    shutil.copytree(catalogue_root,tmp_path,dirs_exist_ok=True)
    return tmp_path


def cmd(root,op,*args,dependencies=None):
    return main(['stage0-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def usage(root,event,total):
    from pilot_eval.stage0_workflow import record_budget
    from pilot_eval.stage0_budget import reservations,settle
    path=root/'usage.json'
    write_json(path,dict(event_id=event,session_id='session-a',total_units=total,
        observed_at='2026-10-27T10:00:00+00:00',note='synthetic cumulative Stage 0 usage'))
    record_budget(root/'plans/audit/stage0.plan.json',root,path)
    for key,row in reservations(root).items():
        if row['state']=='active':settle(root,key)


def prepare(root):
    cohort=root/'cohort.json'
    write_json(cohort,dict(items=[dict(id=f'gsm8k:test:{i}',prompt=f'question {i}',gold='#### 2') for i in range(150)],
        prompt_template=dict(decoding=dict(eos_token_id=99,pad_token_id=98))))
    assert cmd(root,'qualification-prepare','--catalogue',root/'plans/conditional/stage0.intruder-edits.json',
        '--cohort',cohort,'--name','extension')==0
    return root/'plans/extension/stage0.qualification.json'


def qualify(root,plan):
    from test_stage0_qualification import TinyDiagnostic
    usage(root,'before',3.)
    approval=root/'diagnostic-review.json'
    write_json(approval,dict(reviewer='test',notes='bounded synthetic extension',session_id='session-a',
        max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
    engine=TinyDiagnostic()
    assert cmd(root,'qualification-run','--config',plan,'--approval',approval,'--max-new-units',1,dependencies=engine)==0
    directory=root/'runs/stage0/extension/qualification'
    marker=next((directory/'units').glob('*/complete.json'));before=marker.read_bytes()
    usage(root,'after-interrupt',3.01)
    assert cmd(root,'qualification-run','--config',plan,'--approval',approval,dependencies=engine)==0
    assert marker.read_bytes()==before and len(engine.calls)==16
    assert {arm for _,arm in engine.calls}=={'sft'}
    assert cmd(root,'qualification-verify','--config',plan)==0
    diagnostic=json.loads((directory/'results/diagnostic.json').read_text())
    assert diagnostic['completed_units']==16 and diagnostic['production_estimate']['sequences']==750
    usage(root,'after-diagnostic',3.02)
    review=root/'numerical-review.json'
    write_json(review,dict(reviewer='test',notes='measured fixture gate only',diagnostic_sha256=file_hash(directory/'complete.json'),absolute_gate=diagnostic['proposed_absolute_gate']))
    assert cmd(root,'qualification-approve','--config',plan,'--review',review)==0
    write_json(review,dict(reviewer='test',notes='separate measured production budget',diagnostic_sha256=file_hash(directory/'complete.json'),
        max_units_per_hour=1.,max_seconds=3600.,session_id='session-a',source_reviewed=True))
    assert cmd(root,'qualification-freeze','--config',plan,'--review',review)==0
    return root/'plans/extension/stage0.execution.json'


def test_conditional_catalogue_qualification_and_shared_collection(source):
    plan=prepare(source)
    config=json.loads(plan.read_text())
    assert config['qualification_kind']=='conditional-intruders'
    assert config['arms']==['sft'] and config['total_units']==16
    assert 'V6-sft-unhalved' not in config['variants']
    assert set(config['variants'])=={'V0a','V0b','V6-sft-full','V6-sft-matched','V6c-sft-matched'}
    execution=qualify(source,plan)
    from test_stage0_evaluation import TinyEvaluation
    assert cmd(source,'evaluation-prepare','--execution',execution,'--batch-size',64,'--batches-per-unit',1)==0
    ep=source/'plans/extension/stage0.evaluation.json';engine=TinyEvaluation()
    assert cmd(source,'evaluation-run','--config',ep,'--max-new-units',1,dependencies=engine)==0
    directory=source/'runs/stage0/extension/paired-evaluation'
    marker=next((directory/'units').glob('*/complete.json'));before=marker.read_bytes()
    assert cmd(source,'evaluation-run','--config',ep,dependencies=engine)==0
    assert marker.read_bytes()==before
    assert cmd(source,'evaluation-verify','--config',ep)==0
    result=json.loads((directory/'results/results.json').read_text())
    assert result['response_count']==750 and not result['real_gpu_measurement']
    assert set(result['variants'])==set(config['variants'])
    assert cmd(source,'evaluation-run','--config',ep,dependencies=TinyEvaluation(session='other'))==0 # complete CPU reuse
    # Same frozen execution, new bounded drift allocation, independent CPU report.
    from pilot_eval.workflow import _hash
    from test_stage0_drift import TinyDrift
    rows=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks={v:[False,True,True] if v==('solution' if corpus=='gsm8k' else 'user') else [False]*3
                   for v in ['question','solution','user']}
            rows.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    write_json(source/'fixed/items.json',rows)
    write_json(source/'fixed/activation.prepared.json',dict(items_path='fixed/items.json',items_sha256=_hash(rows),
        model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32'))
    assert cmd(source,'drift-prepare','--execution',execution,'--inputs',source/'fixed/activation.prepared.json',
        '--name','extension-drift','--batch-size',64,'--batches-per-unit',1)==0
    dp=source/'plans/extension-drift/stage0.drift.json'
    assert cmd(source,'findings-prepare','--evaluation',ep,'--drift',dp,'--name','extension-findings')==0
    fp=source/'plans/extension-findings/stage0.findings.json';analysis=json.loads(fp.read_text())
    assert analysis['analysis_kind']=='conditional-intruders'
    assert len(analysis['contrasts'])==4
    assert {c['family'] for c in analysis['contrasts']}=={'intruder-full-vs-own-reference','intruder-matched-vs-own-reference','intruder-control-vs-own-reference','intruder-matched-vs-matched-control'}
    usage(source,'after-greedy',3.04)
    review=source/'drift-review.json'
    write_json(review,dict(reviewer='test',notes='separate fixed-context allocation',session_id='session-a',
        max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
    def conditional_gate(result):result['validation']['operator_passed']=result['validation'].pop('rank1_passed')
    drift=TinyDrift(corrupt=conditional_gate)
    assert cmd(source,'drift-run','--config',dp,'--approval',review,'--max-new-units',1,dependencies=drift)==0
    assert cmd(source,'drift-run','--config',dp,dependencies=drift)==0
    assert cmd(source,'drift-verify','--config',dp)==0
    measured=json.loads((source/'runs/stage0/extension-drift/fixed-context/results/results.json').read_text())
    assert len(measured['cells'])==10 # five variants × solution/user, no question view in tiny inputs
    assert next(c for c in measured['cells'] if c['variant']=='V6-sft-full')['sum_factor_norm_products'] is None
    assert cmd(source,'findings-run','--config',fp,'--max-new-units',1)==0
    assert cmd(source,'findings-run','--config',fp)==0
    assert cmd(source,'findings-verify','--config',fp)==0
    findings=source/'runs/stage0/extension-findings/findings'
    report=json.loads((findings/'results/results.json').read_text())
    assert report['mode']=='stage0-conditional-intruder-findings'
    assert report['catalogue_scope']==config['catalogue_scope']
    assert cmd(source,'findings-audit-prepare','--config',fp)==0
    assert cmd(source,'findings-report','--config',fp)==0
    source_matrix=source/next(iter(config['diagnostic_variants']['sft']['merged_modules'].values()))['path']
    source_matrix.write_bytes(b'changed matrix')
    assert cmd(source,'evaluation-verify','--config',ep)==1


def test_conditional_budget_and_hard_gate_stop_before_freeze(source):
    from test_stage0_qualification import TinyDiagnostic
    from pilot_eval.stage0_budget import reservations
    plan=prepare(source);usage(source,'almost-exhausted',19.99)
    approval=source/'approval.json'
    write_json(approval,dict(reviewer='test',notes='cannot reset cumulative extension credits',session_id='session-a',
        max_units_per_hour=1.,max_seconds=600.,source_reviewed=True))
    engine=TinyDiagnostic()
    assert cmd(source,'qualification-run','--config',plan,'--approval',approval,dependencies=engine)==1
    assert engine.calls==[] and not reservations(source)


def test_not_applicable_catalogue_never_prepares_inference(tmp_path):
    from test_stage0_intruder_edits import public_sources,command
    plan,deps=public_sources(tmp_path,False)
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    assert cmd(tmp_path,'qualification-prepare','--catalogue',plan,'--name','forbidden')==1
    assert not (tmp_path/'plans/forbidden/stage0.qualification.json').exists()
    from pilot_eval.stage0_budget import reservations
    assert not reservations(tmp_path)


def test_full_merged_write_uses_direct_operator_not_difference_of_rounded_outputs():
    from types import SimpleNamespace
    import numpy as np
    from pilot_eval.stage0_mixed_drift_backend import MixedDriftEngine
    class Tensor(np.ndarray):
        def double(self):return self.astype(np.float64).view(Tensor)
    def tensor(x):return np.asarray(x).view(Tensor)
    def linear(x,w,bias):return tensor(x@w.T+(0 if bias is None else bias))
    torch=SimpleNamespace(equal=np.array_equal,nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)))
    original=tensor(np.array([[1.,0.],[0.,2.]],dtype=np.float32))
    edited=tensor(np.array([[3.,1.],[1.,5.]],dtype=np.float32)) # full rank delta
    x=tensor(np.array([[[2.,3.],[4.,5.]]],dtype=np.float32))
    class Module:pass
    module=Module();module._disable_adapters=True;module.base_layer=SimpleNamespace(weight=edited,bias=None)
    engine=object.__new__(MixedDriftEngine);engine.torch=torch;engine.original_weights={module:original}
    engine._finite=lambda *values: None
    output=linear(x,edited,None)
    old,branch,coefficient,fraction=engine._module_write(module,x,output,output,{},'V6',{},False)
    np.testing.assert_array_equal(old,x@original.T)
    np.testing.assert_array_equal(branch,x.astype(float)@(edited.astype(float)-original.astype(float)).T)
    assert coefficient is None and fraction==0
    module._disable_adapters=False
    with pytest.raises(ValueError,match='adapter disabled'):
        engine._module_write(module,x,output,output,{},'V6',{},False)


def test_conditional_evidence_rejects_contradictory_greedy_and_position_claims():
    import numpy as np
    from pilot_eval.stage0_conditional import validate_observation
    from pilot_eval.stage0_qualification_math import compare_logits
    teacher=compare_logits(np.ones((2,3)),np.ones((2,3))+1e-6,np.ones((2,3))+.1,[4,5])
    row=dict(teacher=teacher,greedy=dict(exact=True,near_tie=False,reference_token_ids=[1,2],merged_token_ids=[1,2]))
    validate_observation(row)
    bad=json.loads(json.dumps(row));bad['greedy']['merged_token_ids']=[1,3]
    with pytest.raises(ValueError,match='Greedy'):validate_observation(bad)
    bad=json.loads(json.dumps(row));bad['teacher']['position_max_errors'][0]=.1
    with pytest.raises(ValueError,match='position'):validate_observation(bad)
    with pytest.raises(ValueError,match='answer'):validate_observation(row,dict(answer_positions=[3,4,5]))


def test_notebooks_select_conditional_workflow_without_resetting_budget():
    import ast
    for name in ['stage-0-cpu-colab.ipynb','stage-0-gpu-colab.ipynb']:
        notebook=json.loads((Path('notebooks')/name).read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert 'CONDITIONAL_EXTENSION = False' in code
        assert '-intruder-qualification-v1' in code and '-intruder-findings-v1' in code
        if 'gpu' in name:assert "'--catalogue'" in code
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                assert not cell['outputs'] and cell['execution_count'] is None



def test_qualification_status_and_explicit_owner_recovery(source):
    import os,platform
    plan=prepare(source)
    assert cmd(source,'qualification-status','--config',plan)==0
    directory=source/'runs/stage0/extension/qualification';marker=directory/'writer.json'
    write_json(marker,dict(token='known',host=platform.node(),pid=os.getpid()))
    assert cmd(source,'qualification-recover-writer','--config',plan,'--owner-token','known','--review-notes','checked')==1
    write_json(marker,dict(token='known',host='departed-colab',pid=1234))
    assert cmd(source,'qualification-recover-writer','--config',plan,'--owner-token','wrong','--review-notes','checked')==1
    assert cmd(source,'qualification-recover-writer','--config',plan,'--owner-token','known','--review-notes','old runtime stopped')==0
    assert not marker.exists() and (directory/'logs/ownership-recovery.jsonl').exists()
