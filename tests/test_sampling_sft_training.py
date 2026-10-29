"""Paid training gates and durable checkpoint semantics at the public seam."""
import json
from pathlib import Path
import pytest


def test_weights_only_state_cannot_be_recovery(tmp_path):
    from pilot_eval.sampling_sft_training import checkpoint
    p=tmp_path/'checkpoints/checkpoint-8';p.mkdir(parents=True)
    (p/'adapter_config.json').write_text('{}')
    (p/'adapter_model.safetensors').write_bytes(b'adapter')
    from pilot_eval.sampling_sft_training import seal
    seal(p,8,False,'plan','runtime')
    assert checkpoint(p,'plan','runtime')['recoverable'] is False
    with pytest.raises(ValueError,match='optimizer'):
        checkpoint(p,'plan','runtime',require_recovery=True)


def test_public_training_prepare_rejects_k2_before_loading_sources(tmp_path):
    from pilot_eval.cli import main
    assert main(['sampling-sft-train-prepare','--data','missing','--selection','missing',
        '--name','k2','--k','2','--output-root',str(tmp_path)])==1


from test_sampling_sft_gate import reviewed,collected,source_template


@pytest.fixture(scope='module')
def approved(tmp_path_factory,request):
    import shutil,os
    if os.environ.get('SAMPLING_SFT_APPROVED_FIXTURE'):
        original=Path(os.environ['SAMPLING_SFT_APPROVED_FIXTURE'])
        root=tmp_path_factory.mktemp('restored-approved')
        shutil.copytree(original,root,dirs_exist_ok=True)
        return root,root/'plans/training-data/sampling-sft.training-data.json',next((root/'runs/sampling-sft/sample/gate/gate/training-selections').glob('*.json'))
    reviewed=Path(os.environ['SAMPLING_SFT_REVIEWED_FIXTURE']) if os.environ.get('SAMPLING_SFT_REVIEWED_FIXTURE') else request.getfixturevalue('reviewed')
    from test_sampling_sft_gate import prepare,cli,Dependencies,directory
    from pilot_eval.sampling_sft_gate import request,approve
    from pilot_eval import sampling_sft_training_data as data
    root=tmp_path_factory.mktemp('endpoint-training');shutil.copytree(reviewed,root,dirs_exist_ok=True)
    gate=prepare(root,reviewed)
    assert cli(root,'score','--config',gate,deps=Dependencies())==0
    assert cli(root,'report','--config',gate)==0
    forecast=root/'forecast.json';forecast.write_text(json.dumps(dict(arms=[0,5],per_arm={str(k):dict(
        rate_usd_per_hour='0.59',estimated_hours='1',setup_hours='0.1') for k in [0,5]},notes='CPU fixture forecast; no paid approval')))
    value=request(gate,root,forecast)
    approval=root/'gate-approval.json';approval.write_text(json.dumps(dict(request_path=value['request_path'],
        request_sha256=value['request_sha256'],report_sha256=value['report_sha256'],budget_sha256=value['budget_snapshot']['sha256'],
        reviewer='fixture',notes='Synthetic workflow test, not human approval.',gate_approved=True,
        acknowledged_coverage_flags=value['required_acknowledgments'],review_80_percent=False)))
    decision=approve(gate,root,approval)
    training=data.prepare(root/'plans/review/sampling-sft.review.json',root,'training-data')
    data.export(training,root)
    if os.environ.get('SAMPLING_SFT_DURABLE_FIXTURE'):
        target=Path(os.environ['SAMPLING_SFT_DURABLE_FIXTURE']).resolve()
        if target.exists():raise ValueError('Durable synthetic fixture already exists; choose a new validation name.')
        shutil.copytree(root,target)
        training=target/training.relative_to(root);root=target
    return root,training,root/decision['selection_path']


def test_public_training_refuses_unready_real_job_before_import_and_exports_inventory(approved):
    from pilot_eval.cli import main
    from pilot_eval import sampling_sft_training as train
    root,data,selection=approved
    import uuid
    name='real-blocked-'+uuid.uuid4().hex[:8]
    plan=train.prepare(data,selection,root,name,0)
    assert main(['sampling-sft-train-run','--config',str(plan),'--output-root',str(root)])==1
    assert not (root/f'runs/sampling-sft/sample/training/{name}/started.json').exists()
    template=root/'job-template.json';template.write_text(json.dumps(dict(job_id=name,kind='gpu',mode='runpod',
        stage='training',network_volume_id='fixture',rate_usd_per_hour='0.59',estimated_hours='1',setup_hours='0.1')))
    req=train.job_request(plan,root,template);request=json.loads(req.read_text())
    from pilot_eval import sampling_sft_jobs as jobs
    job_path=jobs.prepare(root/'plans/sample/sampling-sft.plan.json',root,req)
    _,job,_=jobs._load(job_path,root)
    with pytest.raises(ValueError,match='T15/T16'):train.authorize_job(root,job)
    assert request['command'][3]=='sampling-sft-train-run'
    assert any(n.endswith('checkpoint-64/optimizer.pt') for n in request['outputs'])
    assert any(n.endswith('checkpoint-0/adapter_model.safetensors') for n in request['outputs'])
    # A source-bound selection never implies paid execution or a CPU fixture grant.
    assert not (job_path.parent/'launch.json').exists()
    from pilot_eval.sampling_sft_budget import status as ledger
    before=ledger(root)['sha256']
    with pytest.raises(ValueError,match='T15/T16'):jobs.run(job_path,root,provider=object())
    assert ledger(root)['sha256']==before
    with pytest.raises(ValueError):train.prepare(data,selection,root,'forbidden',2)


def test_real_cpu_training_preserves_steps_and_recovery_matches_uninterrupted(approved):
    import os,subprocess
    python=os.environ.get('SAMPLING_SFT_CPU_PYTHON')
    if not python:pytest.skip('Use an existing working CPU Torch interpreter; no installation or GPU fallback.')
    root,data,selection=approved
    script=r'''
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/helpers'))
from sampling_sft_tiny_training import Dependencies
from pilot_eval.cli import main
from pilot_eval import sampling_sft_training as train
root=Path(sys.argv[1]);data=Path(sys.argv[2]);selection=Path(sys.argv[3])
def cli(operation,*args,deps=None):
    return main(['sampling-sft-train-'+operation,'--output-root',str(root),*map(str,args)],dependencies=deps)
import uuid
suffix='-'+uuid.uuid4().hex[:8]
def prepare(name,**kw):return train.prepare(data,selection,root,name+suffix,0,**kw)
reference=prepare('uninterrupted')
# Independent first-step loss calculation on the same stored, masked targets.
from pilot_eval.sampling_sft_readiness_backend import oracle_loss
from pilot_eval import sampling_sft_training_data as data_workflow
dc=json.loads(data.read_text());df=root/dc['run_path']
from pilot_eval.sampling_sft_generation import read_lines
stored=read_lines(df/'arms/k0/training.jsonl')
first=[r for r in stored if r['step']==1]
engine,_=Dependencies().open({},first,df/'initialization/checkpoint-0')
hand_loss,_=oracle_loss(engine.model,first)

assert cli('run','--config',reference,deps=Dependencies())==0
result=json.loads((root/json.loads(reference.read_text())['run_path']/'results/results.json').read_text())
assert result['successful_steps']==64 and result['example_exposures']==511
assert result['execution_kind']=='cpu-fixture'
ref_folder=root/json.loads(reference.read_text())['run_path']
first_loss=json.loads((ref_folder/'logs/steps.jsonl').read_text().splitlines()[0])['loss']
assert abs(first_loss-hand_loss)<=1e-6+1e-6*abs(hand_loss)
assert cli('run','--config',reference,deps=Dependencies(stop=1))==0
partial=prepare('interrupted')
assert cli('run','--config',partial,deps=Dependencies(stop=9))==1
assert max(train.status(partial,root)['recoverable_steps'])==8
resume=prepare('resumed',resume_from=partial)
assert cli('run','--config',resume,deps=Dependencies())==0
left=root/json.loads(reference.read_text())['run_path'];right=root/json.loads(resume.read_text())['run_path']
assert (left/'logs/steps.jsonl').read_bytes()==(right/'logs/steps.jsonl').read_bytes()
assert (left/'checkpoints/checkpoint-64/adapter_model.safetensors').read_bytes()==(right/'checkpoints/checkpoint-64/adapter_model.safetensors').read_bytes()
assert cli('verify','--config',resume)==0
# A durable export contains every file required by completed-result verification.
resume_config=json.loads(resume.read_text())
seal=json.loads((right/'complete.json').read_text())
exported={n.removeprefix(resume_config['run_path']+'/') for n in train.required_outputs(resume_config)}
assert set(seal['files']).issubset(exported)
assert not any(n.endswith('README.md') for n in seal['files'])
# Adapter-only interruptions require a fresh checkpoint0 restart, never a resume.
weights=prepare('weights-only',capture_recovery=False)
assert cli('run','--config',weights,deps=Dependencies(stop=8))==1
try:prepare('bad-resume',resume_from=weights)
except ValueError as e:assert 'optimizer' in str(e)
else:raise AssertionError('Adapter-only resume was accepted')
restart=prepare('restart',restart_of=weights)
assert cli('run','--config',restart,deps=Dependencies())==0
# A changed numerical runtime cannot continue a prior optimizer state.
mismatch=prepare('changed-runtime',resume_from=partial)
assert cli('run','--config',mismatch,deps=Dependencies(device='changed fixture CPU'))==1
assert (left/'checkpoints/checkpoint-0/adapter_model.safetensors').read_bytes()==(root/'runs/sampling-sft/sample/training-data/training-data/initialization/checkpoint-0/adapter_model.safetensors').read_bytes()
# A changed completed payload fails restoration rather than being overwritten.
(right/'logs/steps.jsonl').write_text('changed')
assert cli('verify','--config',resume)==1
'''
    log=root/'cpu-training.console.log'
    with log.open('w') as out:
        child=subprocess.run([python,'-c',script,str(root),str(data),str(selection)],stdout=out,stderr=subprocess.STDOUT,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},timeout=900)
    assert child.returncode==0,log.read_text()[-7000:]


@pytest.mark.parametrize('change',['runtime','payload','progress','manifest'])
def test_recovery_seals_reject_changed_state(tmp_path,change):
    from pilot_eval.sampling_sft_training import seal,checkpoint
    from pilot_eval.training import CHECKPOINT_FILES
    cp=tmp_path/'checkpoint-8';cp.mkdir()
    for n in CHECKPOINT_FILES:(cp/n).write_bytes(b'fixture state')
    (cp/'trainer_state.json').write_text(json.dumps(dict(global_step=8)))
    seal(cp,8,True,'plan','runtime')
    assert checkpoint(cp,'plan','runtime',require_recovery=True)['recoverable']
    if change=='runtime':
        with pytest.raises(ValueError):checkpoint(cp,'plan','different-runtime',True)
        return
    if change=='payload':(cp/'optimizer.pt').write_bytes(b'changed')
    elif change=='progress':(cp/'trainer_state.json').write_text('{"global_step": 9}')
    else:
        value=json.loads((cp/'complete.json').read_text());del value['files']['rng_state.pth']
        (cp/'complete.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):checkpoint(cp,'plan','runtime',True)


def test_optional_training_cells_are_safe_cpu_clients():
    import ast
    notebook=json.loads((Path(__file__).parents[1]/'notebooks/sampling-sft.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    text=''.join(''.join(c['source']) for c in notebook['cells']
        if c['cell_type']=='code' and 'sampling-sft-train-' in ''.join(c['source']))
    assert 'sampling-sft-train-prepare' in text and 'sampling-sft-train-job-request' in text
    assert 'sampling-sft-train-run' not in text
