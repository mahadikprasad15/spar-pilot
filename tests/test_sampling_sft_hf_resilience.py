"""Bounded transient retries, disk failures, and separate acknowledgement clocks."""
from types import SimpleNamespace
import pytest
from test_sampling_sft_hf import bundle
from test_sampling_sft_jobs import source_template
from pilot_eval import sampling_sft_hf as hf

@pytest.mark.parametrize('error',[TimeoutError('timeout'),ConnectionError('reset')])
def test_temporary_ack_failure_is_not_a_failed_job(bundle,tmp_path,error):
    root,path,store=bundle;record=hf.upload(path,root,repo='owner/data',client=store)
    store.fetch=lambda *a: (_ for _ in ()).throw(error)
    events=[]
    assert hf.receive_ack(record,tmp_path/'receipt.json',client=store,on_retry=events.append) is False
    assert events and not (tmp_path/'receipt.json').exists()

@pytest.mark.parametrize('status,retry',[(429,True),(503,True),(401,False),(403,False)])
def test_only_transient_http_errors_are_retryable(status,retry):
    error=RuntimeError('HTTP failure');error.response=SimpleNamespace(status_code=status)
    assert hf.retryable(error) is retry

def test_insufficient_worker_space_blocks_before_upload(bundle,monkeypatch):
    root,path,store=bundle
    monkeypatch.setattr(hf.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='space'):hf.upload(path,root,repo='owner/data',client=store)
    assert store.commits==0

def test_insufficient_restore_space_never_acknowledges(bundle,tmp_path,monkeypatch):
    root,path,store=bundle;record=hf.upload(path,root,repo='owner/data',client=store)
    monkeypatch.setattr(hf.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='space'):hf.restore(record,tmp_path/'client',tmp_path/'ack.json',client=store)
    assert not (tmp_path/'ack.json').exists()

def test_ack_clock_starts_after_work_but_cannot_extend_total_allocation():
    from pilot_eval.sampling_sft_jobs import acknowledgement_remaining
    # Long work took 900s. A new 600s ack allowance starts at t=1000.
    assert acknowledgement_remaining(now=1000,ack_started=1000,wait=600,total_remaining=100)==100
    assert acknowledgement_remaining(now=1030,ack_started=1000,wait=600,total_remaining=1000)==570
    assert acknowledgement_remaining(now=1601,ack_started=1000,wait=600,total_remaining=1000)==0

def test_allowance_uses_compressed_and_unpacked_sizes():
    forecast=hf.acknowledgement_forecast(dict(archive_bytes=1000,unpacked_bytes=4000),100,200,10)
    assert forecast['recommended_ack_seconds']==55


def test_real_mode_runner_recovers_transient_ack_after_long_work(tmp_path,source_template,monkeypatch):
    import json
    from test_sampling_sft_hf import Store
    from test_sampling_sft_jobs import paid_job,Provider
    from pilot_eval import sampling_sft_jobs as jobs
    job,approval,provider=paid_job(tmp_path,source_template)
    clock=[0.];original_run=provider.run_job
    def work(*args,**kwargs):
        value=original_run(*args,**kwargs);clock[0]+=100;return value
    provider.run_job=work
    monkeypatch.setattr(jobs.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(jobs.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    store=Store(tmp_path/'remote');original_fetch=store.fetch;calls=[]
    def fetch(repo,name,revision):
        if name.startswith('acknowledgements/'):
            calls.append(name)
            if len(calls)<=2:raise ConnectionError('temporary reset')
            record=json.loads((job.parent/'hf-upload.json').read_text())
            receipt=tmp_path/'client/ack.json'
            hf.restore(record,tmp_path/'client/artifacts',receipt,client=store,provider=SimpleNamespace(host_id='external'))
            ack_revision=hf.acknowledge(record,receipt,client=store)
            return original_fetch(repo,name,ack_revision)
        return original_fetch(repo,name,revision)
    store.fetch=fetch
    result=jobs.execute(job,tmp_path,tmp_path/'network',tmp_path/'received.json',approval,max_wait_seconds=20,
        provider=provider,hf_repo='owner/data',hf_client=store)
    assert result['state']=='stop-requested' and provider.stopped and len(calls)==3
    assert clock[0]==106 # 100s of work did not consume the new20s acknowledgement allowance
    assert (job.parent/'events.jsonl').read_text().count('hf-ack-retry')==2



def test_allocation_expiry_is_fatal_and_cannot_be_retried():
    import signal,time
    if not hasattr(signal,'setitimer'):pytest.skip('Unix worker timer required')
    with pytest.raises(hf.TransferDeadlineExceeded):
        with hf.transfer_deadline(0.01,True):time.sleep(0.1)
    assert not hf.retryable(hf.TransferDeadlineExceeded('expired'))
