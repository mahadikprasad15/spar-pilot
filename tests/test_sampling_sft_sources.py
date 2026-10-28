"""Sampling SFT through the notebook's public CPU workflow; no model downloads."""
import ast
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.sft import prepare_sft
from pilot_eval.training import file_hash, training_directory
from pilot_eval.workflow import _hash
from test_sft_prepare import TrainingData, source_plan
from test_stage0_workflow import sources, write_json


def fixture_sources(root):
    """Real saved schemas, tiny independent FP32 factors across all196 modules."""
    selected = json.loads(sources(root, steps=(0, 8, 16, 32, 64)).read_text())
    path = prepare_sft(source_plan(root), root, 'gold', dependencies=TrainingData(), hardware='L4')
    config = json.loads(path.read_text())
    directory = training_directory(root, config)
    directory.mkdir(parents=True, exist_ok=True)
    runtime = dict(device='NVIDIA L4', dtype='float32', tf32=False, autocast=False,
                   cuda='12.8', versions={'torch': '2.9.0'}, git_commit='a'*40)
    write_json(directory/'meta/runtime.json', runtime)
    write_json(directory/'meta/run_manifest.json', dict(config=config, runtime=runtime))
    write_json(directory/'results/preflight.json', dict(base_sha256='b'*64, zero_write=True,
                                                       finite_loss=True, finite_gradients=True, target_count=196))
    write_json(directory/'results/results.json', dict(successful_steps=64, example_exposures=512,
        base_sha256_before='b'*64, base_sha256_after='b'*64))
    rows = json.loads((root/config['training_items_path']).read_text())
    order = [r['id'] for r in rows]
    random.Random(42).shuffle(order)
    write_json(directory/'inputs/order.json', order)
    # Copy only the adapter/seal payloads. Auditing adapters does not need optimizer imports.
    for step in config['checkpoint_steps']:
        old=root/'inputs/sft'/str(step)
        new=directory/'checkpoints'/f'checkpoint-{step}'
        new.mkdir(parents=True)
        for name in ['adapter_config.json','adapter_model.safetensors','complete.json']:
            (new/name).write_bytes((old/name).read_bytes())
    logs=directory/'logs/steps.jsonl'; logs.parent.mkdir(parents=True)
    logs.write_text(''.join(json.dumps({'step':i,'loss':1.0,'learning_rate':1e-4})+'\n' for i in range(1,65)))
    # Prepared GRPO identity must agree with the actual gold source.
    grpo=root/selected['sources']['grpo']
    value=json.loads(grpo.read_text())
    value.update(model_revision=config['model_revision'],tokenizer_revision=config['tokenizer_revision'])
    value['source_evidence']['base_sha256']='b'*64
    write_json(grpo,value)
    write_json(grpo.parent/'prepare-complete.json',dict(config_sha256=_hash(value),
        files={str(grpo.relative_to(root)):file_hash(grpo)}))
    selection=root/'sampling-selection.json'
    write_json(selection,dict(sft=str(path.relative_to(root)),grpo=str(grpo.relative_to(root))))
    return selection,path,directory


def command(root, operation, *args):
    return main(['sampling-sft-'+operation,'--output-root',str(root),*map(str,args)])


def prepare(root,selection,name='sample'):
    assert command(root,'prepare','--selection',selection,'--name',name)==0
    return root/'plans'/name/'sampling-sft.plan.json'


def result(root,name='sample'):
    return json.loads((root/'runs/sampling-sft'/name/'results/source-audit.json').read_text())


def test_public_workflow_verifies_all_sources_and_restores_without_gpu(tmp_path):
    selection,gold,directory=fixture_sources(tmp_path)
    original={p:p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    audit=result(tmp_path)
    assert audit['source_ready'] is True
    assert audit['training']['original_runtime']['device']=='NVIDIA L4'
    assert audit['training']['checkpoint_steps']==[0,8,16,32,64]
    assert audit['training']['checkpoint0']['b_exact_zero'] is True
    assert len(audit['training']['batches'])==64
    assert all(len(b)==8 for b in audit['training']['batches'])
    assert audit['training']['prompt_mask_contract']['stop_included'] is True
    assert audit['recipe']['k_values']==[0,2,5]
    assert audit['recipe']['data_seed']==0
    assert audit['recipe']['training_seed']==42
    assert audit['recipe']['train_arms']==[0,5]
    assert audit['gpu_qualified'] is False and audit['executed'] is False
    assert audit['recipe']['baseline']=='deferred'
    assert audit['runtime_hash'] is None  # No fake runtime qualification on CPU.
    sealed=(tmp_path/'runs/sampling-sft/sample/complete.json').read_bytes()
    assert command(tmp_path,'audit','--config',plan)==0
    assert (tmp_path/'runs/sampling-sft/sample/complete.json').read_bytes()==sealed
    assert all(p.read_bytes()==content for p,content in original.items())
    script='''
import json,sys
from pilot_eval.sampling_sft import status
value=status(sys.argv[1],sys.argv[2])
assert value['audit']['source_ready']
assert not value['audit']['gpu_qualified']
assert not any(x in sys.modules for x in ('torch','transformers','peft','vllm','datasets'))
print('fresh CPU restore passed')
'''
    subprocess.run([sys.executable,'-c',script,str(plan),str(tmp_path)],check=True)
    assert command(tmp_path,'verify','--config',plan)==0


@pytest.mark.parametrize('missing',['runtime','order','checkpoint','manifest'])
def test_missing_training_evidence_is_blocked_not_inferred(tmp_path,missing):
    selection,gold,directory=fixture_sources(tmp_path)
    target={'runtime':'meta/runtime.json','order':'inputs/order.json',
            'checkpoint':'checkpoints/checkpoint-16/adapter_model.safetensors',
            'manifest':'meta/run_manifest.json'}[missing]
    (directory/target).unlink()
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    audit=result(tmp_path)
    assert audit['source_ready'] is False
    assert audit['state']=='blocked'
    assert audit['blockers']
    assert command(tmp_path,'verify','--config',plan)==0 # valid blocked evidence, not ready


def test_evaluation_runtime_cannot_replace_original_training_evidence(tmp_path):
    selection,gold,directory=fixture_sources(tmp_path)
    (directory/'meta/runtime.json').unlink()
    write_json(directory/'evaluation/64/config.json',dict(runtime={'device':'NVIDIA L4'}))
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    assert result(tmp_path)['source_ready'] is False


@pytest.mark.parametrize('change',['runtime','order','mask','pin','checkpoint','plan','audit'])
def test_changed_sources_or_products_reject_restore_and_preserve_seal(tmp_path,change):
    selection,gold,directory=fixture_sources(tmp_path)
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    seal=tmp_path/'runs/sampling-sft/sample/complete.json'; before=seal.read_bytes()
    config=json.loads(gold.read_text())
    targets={'runtime':directory/'meta/runtime.json','order':directory/'inputs/order.json',
       'mask':tmp_path/config['training_items_path'],'pin':gold,
       'checkpoint':directory/'checkpoints/checkpoint-0/adapter_model.safetensors',
       'plan':plan,'audit':tmp_path/'runs/sampling-sft/sample/results/source-audit.json'}
    path=targets[change];path.write_bytes(path.read_bytes()+b' ')
    assert command(tmp_path,'status','--config',plan)==1
    assert command(tmp_path,'audit','--config',plan)==1
    assert seal.read_bytes()==before


@pytest.mark.parametrize('change',['runtime','order','grpo-pin','zero','logs'])
def test_semantic_source_contradictions_are_blocked(tmp_path,change):
    selection,gold,directory=fixture_sources(tmp_path)
    if change=='runtime':
        write_json(directory/'meta/runtime.json',{'device':'Tesla T4','dtype':'float32','tf32':False})
    elif change=='order':
        p=directory/'inputs/order.json';ids=json.loads(p.read_text());ids[0]=ids[1];write_json(p,ids)
    elif change=='grpo-pin':
        p=tmp_path/json.loads(selection.read_text())['grpo'];value=json.loads(p.read_text());value['model_revision']='c'*40;write_json(p,value)
        write_json(p.parent/'prepare-complete.json',dict(config_sha256=_hash(value),files={str(p.relative_to(tmp_path)):file_hash(p)}))
    elif change=='logs':
        (directory/'logs/steps.jsonl').unlink()
    else:
        p=directory/'checkpoints/checkpoint-0/adapter_model.safetensors'
        # An independently sealed step64 payload is invalid as zero, despite its shape.
        p.write_bytes((directory/'checkpoints/checkpoint-64/adapter_model.safetensors').read_bytes())
        marker=p.parent/'complete.json';value=json.loads(marker.read_text());value['files'][p.name]=file_hash(p);write_json(marker,value)
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    assert result(tmp_path)['source_ready'] is False
    assert result(tmp_path)['blockers']


def test_source_missing_then_added_requires_new_plan_identity(tmp_path):
    selection=tmp_path/'selection.json';write_json(selection,dict(sft='missing.json',grpo='also-missing.json'))
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    before=(tmp_path/'runs/sampling-sft/sample/complete.json').read_bytes()
    write_json(tmp_path/'missing.json',{})
    assert command(tmp_path,'status','--config',plan)==1
    assert (tmp_path/'runs/sampling-sft/sample/complete.json').read_bytes()==before


def test_prepare_resume_after_plan_publication_and_unsafe_names(tmp_path):
    selection,_,_=fixture_sources(tmp_path)
    plan=prepare(tmp_path,selection)
    (tmp_path/'runs/sampling-sft/sample/meta/run_manifest.json').unlink()
    assert command(tmp_path,'prepare','--selection',selection,'--name','sample')==0
    assert command(tmp_path,'audit','--config',plan)==0
    assert command(tmp_path,'prepare','--selection',selection,'--name','../escape')==1


def test_notebook_uses_same_public_commands_and_clean_cpu_flow():
    path=Path(__file__).parents[1]/'notebooks/sampling-sft.ipynb'
    book=json.loads(path.read_text())
    codes=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    joined='\n'.join(codes)
    for operation in ['prepare','audit','status']:
        assert 'sampling-sft-'+operation in joined
    assert 'torch' not in joined and 'cuda' not in joined.lower()
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in book['cells'] if c['cell_type']=='code')


def test_reviewed_writer_recovery_preserves_source_and_audit_identity(tmp_path):
    import platform
    selection,_,directory=fixture_sources(tmp_path)
    plan=prepare(tmp_path,selection)
    run=tmp_path/'runs/sampling-sft/sample'
    marker=run/'writer.json'
    write_json(marker,dict(token='stale-owner',pid=99999999,host=platform.node()))
    original=(directory/'checkpoints/checkpoint-0/adapter_model.safetensors').read_bytes()
    assert command(tmp_path,'audit','--config',plan)==1
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','wrong','--review-notes','inspected')==1
    assert marker.exists()
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','stale-owner','--review-notes','inspected dead process')==0
    assert not marker.exists()
    assert command(tmp_path,'audit','--config',plan)==0
    assert (directory/'checkpoints/checkpoint-0/adapter_model.safetensors').read_bytes()==original
    events=[json.loads(line) for line in (run/'logs/ownership-recovery.jsonl').read_text().splitlines()]
    assert events[0]['owner']['token']=='stale-owner'
    assert events[0]['review_notes']=='inspected dead process'


def test_notebook_runs_all_cpu_cells_against_saved_fixture_sources(tmp_path):
    selection,gold,_=fixture_sources(tmp_path)
    picked=json.loads(selection.read_text())
    repo=Path(__file__).parents[1]
    book=json.loads((repo/'notebooks/sampling-sft.ipynb').read_text())
    codes=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    script='\n'.join(codes)
    # Simulate the user editing only the notebook's source/root parameters.
    script=script.replace('ARTIFACT_ROOT = REPO_DIR / "artifacts"',f'ARTIFACT_ROOT = Path({str(tmp_path)!r})')
    script=script.replace('plans/pilot2-sft-fp32-l4-batch2-v1/sft.config.json',picked['sft'])
    script=script.replace('plans/pilot4-grpo-fp32-v1-writes-v1/activation.prepared.json',picked['grpo'])
    completed=subprocess.run([sys.executable,'-c',script],cwd=repo,text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    assert 'Sources ready: True' in completed.stdout
    assert (tmp_path/'plans/sampling-sft-v1/sampling-sft.plan.json').exists()


@pytest.mark.parametrize('arm',['sft','grpo'])
def test_malformed_source_schema_is_recorded_as_a_blocker(tmp_path,arm):
    selection,gold,_=fixture_sources(tmp_path)
    picked=json.loads(selection.read_text())
    write_json(tmp_path/picked[arm],[])
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    assert result(tmp_path)['source_ready'] is False
    assert any(b['source']==arm for b in result(tmp_path)['blockers'])


def test_saved_plan_binds_proposer_text_and_gold_chat_template(tmp_path):
    selection,_,_=fixture_sources(tmp_path)
    plan=prepare(tmp_path,selection)
    assert command(tmp_path,'audit','--config',plan)==0
    audit=result(tmp_path)
    prompts=audit['recipe']['prompts']
    assert prompts['version']=='sampling-sft-proposer-v1'
    assert prompts['template_sha256']==_hash(prompts['proposer_template'])
    assert '{TRAINING_USER_MESSAGE}' in prompts['proposer_template']
    assert '{CLEANED_GOLD}' in prompts['proposer_template']
    assert '{PARTIAL_RESPONSE_TEXT}' in prompts['proposer_template']
    assert prompts['prefix_policy']=='decoded-in-user-and-token-ids-as-assistant-prefill'
    assert len(audit['training']['prompt_mask_contract']['chat_template_sha256'])==64
