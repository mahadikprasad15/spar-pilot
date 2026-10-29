"""Stored-ID exports, exact initialization and honestly incomplete readiness."""
import json
import os
import shutil
import sys

import pytest

from pilot_eval.cli import main
from test_sampling_sft_gate import reviewed,collected,source_template


def cli(root,operation,*args):
    return main(['sampling-sft-training-data-'+operation,'--output-root',str(root),*map(str,args)])


def setup(root,reviewed):
    shutil.copytree(reviewed,root,dirs_exist_ok=True)
    review=root/'plans/review/sampling-sft.review.json'
    assert cli(root,'prepare','--review',review,'--name','training-bundle')==0
    return root/'plans/training-bundle/sampling-sft.training-data.json'


def folder(root):return root/'runs/sampling-sft/sample/training-data/training-bundle'


def test_export_uses_exact_saved_response_ids_masks_and_gold_step_slots(tmp_path,reviewed):
    plan=setup(tmp_path,reviewed)
    assert cli(tmp_path,'export','--config',plan)==0
    assert cli(tmp_path,'verify','--config',plan)==0
    config=json.loads(plan.read_text());root=folder(tmp_path)
    coverage=json.loads((tmp_path/'runs/sampling-sft/sample/eligibility/review/results/coverage.json').read_text())
    gen=json.loads((tmp_path/'plans/generation/sampling-sft.generation.json').read_text())
    for k in [0,2,5]:
        rows=[json.loads(s) for s in (root/f'arms/k{k}/training.jsonl').read_text().splitlines()]
        assert len(rows)==511
        batches=json.loads((root/f'arms/k{k}/batches.json').read_text())
        assert len(batches)==64 and sum(len(b['slots']) for b in batches)==512
        assert sum(s['missing'] for b in batches for s in b['slots'])==1
        attempts={a['attempt_id']:a for a in coverage['arms'][str(k)]['attempts']}
        for row in rows:
            expected=attempts[row['attempt_id']]['response_ids']
            prompt=gen['problems'][row['problem_id']]['scorer_prompt_ids']
            assert row['input_ids']==prompt+expected
            assert row['labels']==[-100]*len(prompt)+expected
            assert row['input_ids'][-1]==3 and row['supervised_end_turn_count']==1
        assert [r['id'] for r in rows]==[s['source_id'] for b in batches for s in b['slots'] if not s['missing']]
    receipt=json.loads((root/'results/export.json').read_text())
    assert receipt['t15']['passed'] and receipt['t15']['b_exact_zero']
    assert receipt['t15']['modules']==196 and not receipt['training_authorized']
    assert receipt['checkpoint_steps']==[0,8,16,32,64]
    assert receipt['optimizer']['learning_rate']==1e-4
    assert receipt['training_seed']==42 and receipt['hardware']=='L4'
    assert cli(tmp_path,'export','--config',plan)==0
    # The exported initialization is immutable and must agree byte-for-byte.
    weights=root/'initialization/checkpoint-0/adapter_model.safetensors'
    payload=bytearray(weights.read_bytes());payload[-1]^=1;weights.write_bytes(payload)
    assert cli(tmp_path,'verify','--config',plan)==1


def test_end_turn_and_prompt_contracts_reject_retokenization_or_extra_stops():
    from pilot_eval.sampling_sft_training_data import supervised_row
    problem=dict(problem_id=0,source_id='p0',scorer_prompt_ids=[1,2])
    attempt=dict(attempt_id='a',response_ids=[7,3],end_state='finished')
    row=supervised_row(problem,attempt,3)
    assert row['labels']==[-100,-100,7,3]
    for ids in [[],[3,7],[3,7,3],[7,4]]:
        with pytest.raises(ValueError):supervised_row(problem,{**attempt,'response_ids':ids},3)


def test_missing_full_trainer_check_never_becomes_readiness(tmp_path,reviewed):
    plan=setup(tmp_path,reviewed)
    assert cli(tmp_path,'export','--config',plan)==0
    code=cli(tmp_path,'check','--config',plan,'--python',sys.executable,'--mode','full')
    status=json.loads((folder(tmp_path)/'results/readiness.json').read_text())
    if code:
        assert status['state'] in ['missing-check','failed-check'] and not status['training_ready']
    else:
        assert status['t16_passed'] and status['t15_passed'] and status['training_ready']
    assert status['training_authorized'] is False


def test_real_torch_kernel_math_is_separate_from_full_t16(tmp_path,reviewed):
    python=os.environ.get('SAMPLING_SFT_CPU_PYTHON')
    if not python:pytest.skip('Set SAMPLING_SFT_CPU_PYTHON to an existing working Torch interpreter for the real CPU diagnostic.')
    plan=setup(tmp_path,reviewed)
    assert cli(tmp_path,'export','--config',plan)==0
    assert cli(tmp_path,'check','--config',plan,'--python',python,'--mode','kernel')==0
    result=json.loads((folder(tmp_path)/'results/readiness.json').read_text())
    assert result['kernel_passed'] and not result['t16_passed'] and not result['training_ready']
    checks=result['check']['sizes']
    assert [r['batch_size'] for r in checks]==[3,4,5,6,7]
    assert all(r['finite_gradients'] and r['optimizer_steps']==r['scheduler_steps']==1 for r in checks)
    assert all(r['absolute_error']<=1e-6+1e-6*abs(r['oracle_loss']) for r in checks)
    assert cli(tmp_path,'status','--config',plan)==0
    result['training_ready']=True
    (folder(tmp_path)/'results/readiness.json').write_text(json.dumps(result))
    assert cli(tmp_path,'status','--config',plan)==1


def test_incomplete_or_unbound_readiness_receipts_cannot_pass():
    from pilot_eval.sampling_sft_training_data import validate_check
    request=dict(mode='kernel',loss_gate=dict(atol=1e-6,rtol=1e-6))
    from pilot_eval.workflow import _hash
    value=dict(mode='kernel',loss_gate=request['loss_gate'],request_sha256=_hash(request),
        kernel_passed=True,t16_passed=False,sizes=[])
    with pytest.raises(ValueError,match='five short-batch'):validate_check(value,request,0)
    value['request_sha256']='altered'
    with pytest.raises(ValueError,match='identity'):validate_check(value,request,0)


def test_optional_training_notebook_cells_are_disabled_and_syntax_valid():
    import ast
    from pathlib import Path
    notebook=json.loads((Path(__file__).parents[1]/'notebooks/sampling-sft.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    export=next(s for s in code if 'TRAINING_REVIEW = ' in s)
    check=next(s for s in code if 'RUN_FULL_T16 = False' in s)
    assert 'TRAINING_REVIEW = ""' in export and 'training-data-export' in export
    assert 'RUN_FULL_T16 = False' in check and '--mode", "full"' in check


def test_readiness_recomputes_loss_and_step_checks_instead_of_trusting_flags():
    from pilot_eval.sampling_sft_training_data import validate_check
    from pilot_eval.workflow import _hash
    request=dict(mode='kernel',loss_gate=dict(atol=1e-6,rtol=1e-6))
    rows=[dict(batch_size=n,actual_loss=2.0,oracle_loss=2.0,absolute_error=0.0,
        finite_gradients=True,parameters_changed=True,optimizer_steps=1,scheduler_steps=1,passed=True)
        for n in [3,4,5,6,7]]
    value=dict(mode='kernel',loss_gate=request['loss_gate'],request_sha256=_hash(request),
        kernel_passed=True,t16_passed=False,sizes=rows)
    assert validate_check(value,request,0)==(False,True)
    rows[0]['scheduler_steps']=2
    with pytest.raises(ValueError,match='measurements failed'):validate_check(value,request,0)
    rows[0]['scheduler_steps']=1
    rows[0].update(actual_loss=2.1,absolute_error=abs(2.1-2.0))
    with pytest.raises(ValueError,match='measurements failed'):validate_check(value,request,0)
