"""Final inventory, partial status and CPU restoration through public commands."""
import json
from pathlib import Path
import pytest
from pilot_eval.cli import main


def test_fixture_completion_is_not_research_completion():
    from pilot_eval.sampling_sft_handoff import completion
    stages=dict(source_audited=True,real_qualified=False,generation_complete=True,
        generation_real=False,review_complete=True,gate_complete=True,gate_real=False,
        t15_t16_ready=False,training_complete=True,training_real=False,lifecycle_complete=True,budget_settled=True)
    assert completion(stages) is False
    stages.update(real_qualified=True,generation_real=True,gate_real=True,t15_t16_ready=True,training_real=True)
    assert completion(stages) is True
    stages['lifecycle_complete']=False
    assert completion(stages) is False


def test_unknown_selection_keys_are_not_silently_ignored(tmp_path):
    p=tmp_path/'selection.json';p.write_text(json.dumps(dict(source_plan='missing',training_done=True)))
    assert main(['sampling-sft-handoff-prepare','--selection',str(p),'--name','handoff','--output-root',str(tmp_path)])==1


@pytest.fixture(scope='module')
def source_fixture(tmp_path_factory):
    from test_sampling_sft_sources import fixture_sources,prepare,command
    root=tmp_path_factory.mktemp('handoff-source')
    selected,_,_=fixture_sources(root);plan=prepare(root,selected)
    assert command(root,'audit','--config',plan)==0
    return root


def cli(root,operation,*args):
    return main(['sampling-sft-handoff-'+operation,'--output-root',str(root),*map(str,args)])


def make(root,name,selection):
    p=root/(name+'.selection.json');p.write_text(json.dumps(selection))
    assert cli(root,'prepare','--selection',p,'--name',name)==0
    return root/'plans'/name/'sampling-sft.handoff.json'


def test_source_only_handoff_is_partial_immutable_and_detects_source_changes(tmp_path,source_fixture):
    import shutil
    shutil.copytree(source_fixture,tmp_path,dirs_exist_ok=True)
    plan=make(tmp_path,'source-handoff',dict(source_plan='plans/sample/sampling-sft.plan.json'))
    assert cli(tmp_path,'build','--config',plan)==0
    from pilot_eval.sampling_sft_handoff import verify
    value=verify(plan,tmp_path)
    assert value['state']=='partial' and value['stages']['source_audited']
    assert not value['stages']['generation_complete'] and not value['stages']['training_complete']
    assert value['deferred'] and not value['training_authorized']
    folder=tmp_path/'runs/sampling-sft/sample/handoffs/source-handoff'
    inv=json.loads((folder/'results/inventory.json').read_text())
    assert any(n.endswith('resolved_trainer.json') for n in inv['absent_paths'])
    missing=tmp_path/inv['absent_paths'][0];missing.write_text('{}')
    try:assert cli(tmp_path,'verify','--config',plan)==1
    finally:missing.unlink()
    before=(folder/'complete.json').read_bytes()
    assert cli(tmp_path,'build','--config',plan)==0
    assert (folder/'complete.json').read_bytes()==before
    audit=tmp_path/'runs/sampling-sft/sample/results/source-audit.json'
    audit.write_text('{}')
    assert cli(tmp_path,'verify','--config',plan)==1


@pytest.fixture(scope='module')
def full_fixture(tmp_path_factory):
    import os,shutil
    selected=os.environ.get('SAMPLING_SFT_HANDOFF_FIXTURE')
    if not selected:pytest.skip('Select a previously verified offline endpoint-workflow fixture; no model collection is run by a handoff.')
    root=tmp_path_factory.mktemp('handoff-complete-fixture');shutil.copytree(selected,root,dirs_exist_ok=True)
    durable=os.environ.get('SAMPLING_SFT_HANDOFF_DURABLE')
    if durable:
        target=Path(durable).resolve()
        if target.exists():raise ValueError('Choose a new durable validation target.')
        shutil.copytree(root,target);root=target
    training=next((root/'plans').glob('uninterrupted-*/sampling-sft.training.json'))
    selection=next((root/'runs/sampling-sft/sample/gate/gate/training-selections').glob('*.json'))
    return root,dict(source_plan='plans/sample/sampling-sft.plan.json',generation_plan='plans/generation/sampling-sft.generation.json',
        review_plan='plans/review/sampling-sft.review.json',gate_plan='plans/gate/sampling-sft.gate.json',
        training_data_plan='plans/training-data/sampling-sft.training-data.json',
        training_selection=str(selection.relative_to(root)),training_plans={'0':str(training.relative_to(root))})


def test_full_offline_inventory_restores_without_models_and_never_claims_real_completion(full_fixture,tmp_path):
    import subprocess,sys
    from pilot_eval.sampling_sft_handoff import verify
    root,selection=full_fixture;plan=make(root,'offline-handoff',selection)
    assert cli(root,'build','--config',plan)==0
    value=verify(plan,root)
    assert value['stages']['generation_complete'] and value['stages']['review_complete'] and value['stages']['gate_complete']
    assert value['details']['training']['0']['successful_steps']==64
    assert not value['stages']['real_qualified'] and not value['stages']['generation_real']
    assert value['state']=='partial' and not value['stages']['training_complete'] # selected k5 is absent
    folder=root/'runs/sampling-sft/sample/handoffs/offline-handoff'
    inventory=json.loads((folder/'results/inventory.json').read_text())['files']
    assert any(p.endswith('attempts.jsonl') for p in inventory)
    assert any('drafts/human/' in p for p in inventory)
    assert any(p.endswith('checkpoint-64/adapter_model.safetensors') for p in inventory)
    assert any(p.endswith('results/answers.jsonl') for p in inventory)
    code='''import sys,json
from pilot_eval.sampling_sft_handoff import verify
r=verify(sys.argv[1],sys.argv[2]);assert r['state']=='partial'
assert not any(p in sys.modules for p in ['torch','transformers','trl','peft','vllm'])
print('fresh CPU restoration; no model imports')
'''
    result=subprocess.run([sys.executable,'-c',code,str(plan),str(root)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'fresh CPU' in result.stdout
    import shutil
    portable=tmp_path/'portable-snapshot';portable.mkdir()
    files=list(inventory)+[str((folder/n).relative_to(root)) for n in ['complete.json','results/inventory.json','results/report.json','report.md']]
    for name in files:
        target=portable/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    assert cli(portable,'verify','--config',plan.relative_to(root))==0
    victim=next(root/p for p in inventory if p.endswith('checkpoint-8/adapter_model.safetensors') and '/training/' in p)
    original=victim.read_bytes()
    try:
        victim.unlink()
        assert cli(root,'verify','--config',plan)==1
    finally:victim.write_bytes(original)
    assert cli(root,'verify','--config',plan)==0
    selected=root/selection['training_selection'];original=selected.read_bytes()
    try:
        selected.write_text('{}')
        assert cli(root,'verify','--config',plan)==1
    finally:selected.write_bytes(original)
    before=(folder/'complete.json').read_bytes()
    assert cli(root,'build','--config',plan)==0 and (folder/'complete.json').read_bytes()==before


def test_copy_receipts_and_external_stop_proof_are_checked_without_provider_calls(tmp_path,source_fixture):
    from test_sampling_sft_jobs import paid_job,paid_receipts,cli as job_cli
    from pilot_eval.sampling_sft_handoff import job_evidence
    job,approval,worker=paid_job(tmp_path,source_fixture)
    assert job_cli(tmp_path,'run','--job',job,'--approval',approval,provider=worker)==0
    network,downloaded,client=paid_receipts(tmp_path,job,worker)
    override=dict(network_receipt=str(network.relative_to(tmp_path)),download_receipt=str(downloaded.relative_to(tmp_path)))
    value=job_evidence(job,tmp_path,override)
    assert value['export_receipts']==2 and not value['closed'] and not value['stop_confirmed']
    assert job_cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded,provider=worker)==0
    value=job_evidence(job,tmp_path,override)
    assert value['stop_requested'] and not value['stop_confirmed']
    client.stopped=True
    assert job_cli(tmp_path,'confirm','--job',job,provider=client)==0
    calls=list(worker.calls)
    value=job_evidence(job,tmp_path,override)
    assert value['closed'] and value['stop_confirmed'] and worker.calls==calls
    receipt=json.loads(downloaded.read_text());receipt['job_sha256']='changed';downloaded.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt'):job_evidence(job,tmp_path,override)


@pytest.mark.parametrize('value',[{},dict(source_plan=None),dict(source_plan=4),dict(source_plan='p',training_plans={'2':'x'}),dict(source_plan='p',jobs=[{}])])
def test_bad_selection_schema_is_rejected(tmp_path,value):
    from pilot_eval.sampling_sft_handoff import normalize
    with pytest.raises(ValueError):normalize(value,tmp_path)


def test_release_requires_exact_commit_and_rejects_missing_source():
    from pilot_eval.sampling_sft_handoff import check_release
    with pytest.raises(ValueError):check_release('main')
    with pytest.raises(ValueError):check_release('0'*40)


def test_notebook_handoff_cells_are_optional_cpu_clients():
    import ast
    book=json.loads((Path(__file__).parents[1]/'notebooks/sampling-sft.ipynb').read_text())
    code=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    for text in code:ast.parse(text)
    assert any('SOURCE_RELEASE = ' in c and 'sampling-sft-handoff-release' in c for c in code)
    assert any('HANDOFF_SELECTION = ""' in c and 'sampling-sft-handoff-prepare' in c for c in code)
    assert any('sampling-sft-handoff-verify' in c and 'globals().get' in c for c in code)
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in book['cells'] if c['cell_type']=='code')


def test_software_receipt_does_not_establish_gpu_qualification(tmp_path,source_fixture):
    import shutil
    from pilot_eval.sampling_sft_handoff import verify
    from pilot_eval.training import file_hash
    shutil.copytree(source_fixture,tmp_path,dirs_exist_ok=True)
    repo=Path(__file__).parents[1];path='pilot_eval/sampling_sft_handoff.py'
    evidence=tmp_path/'software.json';evidence.write_text(json.dumps(dict(state='software-validated',
        ticket='synthetic-test-receipt',source_files={path:file_hash(repo/path)},tests={'fixture':True})))
    plan=make(tmp_path,'software-handoff',dict(source_plan='plans/sample/sampling-sft.plan.json',software_evidence=['software.json']))
    assert cli(tmp_path,'build','--config',plan)==0
    value=verify(plan,tmp_path)
    assert value['stages']['software_tested'] and not value['stages']['real_qualified'] and value['state']=='partial'
    assert value['details']['software'][0]['ticket']=='synthetic-test-receipt'
    evidence.write_text('{}')
    assert cli(tmp_path,'verify','--config',plan)==1
