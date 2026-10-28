"""Saved job lifecycle with fake provider/runner and verified local copy fixtures."""
import json
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import _hash
from test_sampling_sft_budget import setup,write,ledger


class Provider:
    host_id='fixture-pod-host'
    calls=[]
    stopped=False
    fail_stop=False

    def own_pod_id(self):return 'fixture-pod'
    def inspect(self,pod_id):
        return dict(id=pod_id,desiredStatus='EXITED' if self.stopped else 'RUNNING',networkVolumeId='volume-1')
    def stop(self,pod_id):
        self.calls.append(pod_id)
        if self.fail_stop:raise RuntimeError('provider stop failed')
        self.stopped=True
        return dict(id=pod_id,desiredStatus='EXITED')
    def run_job(self,request,root,log,timeout):
        self.calls.append('run')
        log.parent.mkdir(parents=True,exist_ok=True);log.write_text('fixture console\n')
        for path in request['outputs']:
            target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture result\n')
        return dict(returncode=0,elapsed_seconds=0.1)


def cli(root,operation,*args,provider=None):
    return main(['sampling-sft-job-'+operation,'--output-root',str(root),*map(str,args)],dependencies=provider)


def prepared(root,mode='noop'):
    plan=setup(root)
    request=dict(job_id='demo',kind='noop',stage='sampling',mode=mode,
                 command=[],outputs=['runs/sampling-sft/budget-test/demo/result.json'])
    path=write(root/'job-request.json',request)
    assert cli(root,'prepare','--config',plan,'--request',path)==0
    return root/'runs/sampling-sft/budget-test/jobs/demo/job.plan.json'


def manifest(root):
    return root/'runs/sampling-sft/budget-test/jobs/demo/export-manifest.json'


def copy_receipt(root,job,copy_root,role,provider=None):
    receipt=root/(role+'-receipt.json')
    assert cli(root,'export','--job',job,'--copy-root',copy_root,'--role',role,
               '--receipt',receipt,provider=provider)==0
    return receipt


def test_noop_saved_workflow_runs_exports_and_finishes_without_cloud(tmp_path):
    job=prepared(tmp_path)
    assert cli(tmp_path,'run','--job',job)==0
    assert cli(tmp_path,'run','--job',job)==0 # no repeat or duplicate reservation
    assert ledger(tmp_path)['committed_usd']=='0.000000'
    network=copy_receipt(tmp_path,job,tmp_path/'network-copy','network')
    downloaded=copy_receipt(tmp_path,job,tmp_path/'downloaded-copy','downloaded')
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded)==0
    assert cli(tmp_path,'status','--job',job)==0
    events=(job.parent/'events.jsonl').read_text()
    assert 'noop-completed' in events
    assert 'stop-confirmed' not in events # offline simulation does not claim a real stopped Pod


def test_no_export_or_same_copy_or_tampered_payload_cannot_finish(tmp_path):
    job=prepared(tmp_path)
    assert cli(tmp_path,'run','--job',job)==0
    missing=tmp_path/'missing-receipt.json'
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',missing,'--download-receipt',missing)==1
    network=copy_receipt(tmp_path,job,tmp_path/'copy','network')
    assert cli(tmp_path,'export','--job',job,'--copy-root',tmp_path/'copy','--role','downloaded',
               '--receipt',tmp_path/'downloaded.json')==1
    (tmp_path/'runs/sampling-sft/budget-test/demo/result.json').write_text('changed')
    assert cli(tmp_path,'status','--job',job)==1


def test_changed_job_or_unknown_interruption_never_relaunches(tmp_path):
    job=prepared(tmp_path)
    write(job.parent/'launch.json',dict(state='running',note='interrupted before completion'))
    assert cli(tmp_path,'run','--job',job)==1
    assert not manifest(tmp_path).exists()
    value=json.loads(job.read_text());value['request']['command']=['unexpected'];write(job,value)
    assert cli(tmp_path,'status','--job',job)==1


def test_real_mode_cannot_launch_without_source_preflight_and_explicit_approval(tmp_path):
    job=prepared(tmp_path)
    value=json.loads(job.read_text())['request']
    value.update(job_id='paid',kind='gpu',mode='runpod',rate_usd_per_hour='1',estimated_hours='0.1',
                 setup_hours='0.01',network_volume_id='volume-1',command=['python','-V'])
    path=write(tmp_path/'paid-request.json',value)
    plan=tmp_path/'plans/budget-test/sampling-sft.plan.json'
    assert cli(tmp_path,'prepare','--config',plan,'--request',path)==0
    paid=tmp_path/'runs/sampling-sft/budget-test/jobs/paid/job.plan.json'
    provider=Provider();provider.calls=[]
    assert cli(tmp_path,'run','--job',paid,provider=provider)==1
    assert provider.calls==[]


def test_receipt_command_checks_actual_downloaded_files_without_provider(tmp_path):
    job=prepared(tmp_path)
    assert cli(tmp_path,'run','--job',job)==0
    output=copy_receipt(tmp_path,job,tmp_path/'download','downloaded')
    receipt=json.loads(output.read_text())
    assert receipt['manifest_sha256']
    assert receipt['verified_files']>0
    assert receipt['simulation'] is True
    victim=next((tmp_path/'download').rglob('result.json'));victim.write_text('corrupt')
    assert cli(tmp_path,'receipt','--manifest',manifest(tmp_path),'--copy-root',tmp_path/'download',
               '--role','downloaded','--receipt',tmp_path/'new.json')==1


@pytest.fixture(scope='module')
def source_template(tmp_path_factory):
    from test_sampling_sft_sources import fixture_sources,prepare,command
    root=tmp_path_factory.mktemp('job-source-template')
    selection,_,_=fixture_sources(root)
    plan=prepare(root,selection)
    assert command(root,'audit','--config',plan)==0
    return root


def paid_job(root,template):
    import shutil
    from pilot_eval.training import file_hash
    shutil.copytree(template,root,dirs_exist_ok=True)
    plan=root/'plans/sample/sampling-sft.plan.json'
    request=dict(job_id='paid',kind='gpu',stage='sampling',mode='runpod',command=['reviewed-fixture-command'],
        outputs=['runs/sampling-sft/sample/payload/result.json'],network_volume_id='volume-1',
        rate_usd_per_hour='1',estimated_hours='0.1',setup_hours='0.01')
    request_path=write(root/'paid-request.json',request)
    assert cli(root,'prepare','--config',plan,'--request',request_path)==0
    job=root/'runs/sampling-sft/sample/jobs/paid/job.plan.json'
    approval=write(root/'paid-approval.json',dict(reviewer='Prasad',notes='Simulated test approval, not a real paid job',
        request_sha256=_hash(request),source_audit_sha256=file_hash(root/'runs/sampling-sft/sample/results/source-audit.json'),
        preflight_reviewed=True,paid_execution_approved=True))
    provider=Provider();provider.calls=[]
    return job,approval,provider


def paid_receipts(root,job,provider):
    network=copy_receipt(root,job,root/'network','network',provider)
    client=Provider();client.host_id='external-fixture-client'
    downloaded=copy_receipt(root,job,root/'downloaded','downloaded',client)
    return network,downloaded,client


def test_fake_paid_workflow_distinguishes_stop_request_and_external_confirmation(tmp_path,source_template):
    job,approval,provider=paid_job(tmp_path,source_template)
    assert cli(tmp_path,'run','--job',job,'--approval',approval,provider=provider)==0
    assert provider.calls==['run']
    assert ledger(tmp_path)['reserved_usd']=='0.160000'
    network,downloaded,client=paid_receipts(tmp_path,job,provider)
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded,provider=provider)==0
    assert provider.calls==['run','fixture-pod']
    events=(job.parent/'events.jsonl').read_text()
    assert 'stop-requested' in events and 'stop-confirmed' not in events
    assert cli(tmp_path,'confirm','--job',job,provider=client)==1 # provider still says RUNNING
    client.stopped=True
    assert cli(tmp_path,'confirm','--job',job,provider=client)==0
    assert 'stop-confirmed' in (job.parent/'events.jsonl').read_text()
    assert ledger(tmp_path)['reserved_usd']=='0.160000' # provider status is not a bill


def test_fake_stop_failure_and_wrong_owner_never_terminate_or_release_cost(tmp_path,source_template):
    job,approval,provider=paid_job(tmp_path,source_template)
    assert cli(tmp_path,'run','--job',job,'--approval',approval,provider=provider)==0
    network,downloaded,_=paid_receipts(tmp_path,job,provider)
    provider.fail_stop=True
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded,provider=provider)==1
    assert 'stop-request-failed' in (job.parent/'events.jsonl').read_text()
    provider.own_pod_id=lambda:'another-pod'
    before=list(provider.calls)
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded,provider=provider)==1
    assert provider.calls==before
    assert ledger(tmp_path)['reserved_usd']=='0.160000'


def test_paid_stop_requires_external_receipt_and_confirm_cannot_self_certify(tmp_path,source_template):
    job,approval,provider=paid_job(tmp_path,source_template)
    assert cli(tmp_path,'run','--job',job,'--approval',approval,provider=provider)==0
    network,downloaded,_=paid_receipts(tmp_path,job,provider)
    assert cli(tmp_path,'finish','--job',job,'--network-receipt',network,'--download-receipt',downloaded,provider=provider)==0
    assert cli(tmp_path,'confirm','--job',job,provider=provider)==1


def test_missing_download_blocks_execute_without_automatic_stop(tmp_path):
    job=prepared(tmp_path)
    assert cli(tmp_path,'execute','--job',job,'--network-root',tmp_path/'network',
               '--download-receipt',tmp_path/'not-yet-downloaded.json','--max-wait-seconds','0')==1
    assert 'stop-requested' not in (job.parent/'events.jsonl').read_text()
    assert 'export-blocked' in (job.parent/'events.jsonl').read_text()
