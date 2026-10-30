"""Independent archive transport: no cloud, model, or GPU dependencies."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import pytest
from pilot_eval import sampling_sft_hf as hf
from pilot_eval import sampling_sft_exports as exports
from pilot_eval.workflow import _hash

class Store:
    def __init__(self,root):self.root=root;self.private=True;self.commits=0
    def info(self,repo):return SimpleNamespace(private=self.private)
    def publish(self,repo,files):
        self.commits+=1;revision=f'{self.commits:040x}'
        for name,path in files.items():
            target=self.root/revision/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        return revision
    def fetch(self,repo,name,revision):return self.root/revision/name

@pytest.fixture
def bundle(tmp_path):
    root=tmp_path/'worker';p=root/'runs/sampling-sft/demo/results/samples.jsonl'
    p.parent.mkdir(parents=True);p.write_text('{"response_ids":[1,2,3]}\n')
    manifest=dict(files=exports.inventory(root,[str(p.relative_to(root))]),job_sha256='job',simulation=False,worker_host='worker')
    manifest['sha256']=_hash(manifest)
    path=p.parent.parent/'jobs/demo/export-manifest.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(manifest))
    return root,path,Store(tmp_path/'remote')

def test_private_upload_exact_revision_download_and_receipt(bundle,tmp_path):
    root,path,store=bundle
    transport=hf.upload(path,root,repo='owner/data',client=store)
    record=hf.restore(transport,tmp_path/'client',tmp_path/'receipt.json',client=store,provider=SimpleNamespace(host_id='client'))
    assert record['role']=='downloaded' and record['hf_revision']==transport['revision']
    assert record['manifest_sha256']==json.loads(path.read_text())['sha256']
    assert (tmp_path/'client/runs/sampling-sft/demo/results/samples.jsonl').read_bytes()==(root/'runs/sampling-sft/demo/results/samples.jsonl').read_bytes()
    network=tmp_path/'network.json';exports.receipt(path,root,'network',network,provider=SimpleNamespace(host_id='worker'))
    exports.check_receipts(json.loads(path.read_text()),network,tmp_path/'receipt.json',root)
    hf.restore(transport,tmp_path/'client',tmp_path/'receipt.json',client=store,provider=SimpleNamespace(host_id='client'))

def test_public_repo_and_corrupt_payload_block_upload(bundle):
    root,path,store=bundle;store.private=False
    with pytest.raises(ValueError,match='private'):hf.upload(path,root,repo='owner/data',client=store)
    assert store.commits==0
    store.private=True;(root/'runs/sampling-sft/demo/results/samples.jsonl').write_text('bad')
    with pytest.raises(ValueError):hf.upload(path,root,repo='owner/data',client=store)
    assert store.commits==0

def test_corrupted_download_and_same_host_cannot_ack(bundle,tmp_path):
    root,path,store=bundle;t=hf.upload(path,root,repo='owner/data',client=store)
    with pytest.raises(ValueError,match='external'):hf.restore(t,tmp_path/'same',tmp_path/'ack.json',client=store,provider=SimpleNamespace(host_id='worker'))
    store.fetch(t['repo'],t['archive'],t['revision']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):hf.restore(t,tmp_path/'client',tmp_path/'ack.json',client=store)
    assert not (tmp_path/'ack.json').exists()

def test_receipt_requires_pinned_revision(bundle,tmp_path):
    root,path,store=bundle;t=hf.upload(path,root,repo='owner/data',client=store);t['revision']='main'
    with pytest.raises(ValueError,match='revision'):hf.restore(t,tmp_path/'client',tmp_path/'ack.json',client=store)

def test_acknowledgement_roundtrip_and_worker_checks_job_binding(bundle,tmp_path):
    root,path,store=bundle;t=hf.upload(path,root,repo='owner/data',client=store)
    receipt=tmp_path/'client-receipt.json'
    hf.restore(t,tmp_path/'client',receipt,client=store,provider=SimpleNamespace(host_id='client'))
    revision=hf.acknowledge(t,receipt,client=store)
    assert hf.receive_ack(t,tmp_path/'worker-ack.json',client=store,revision=revision)
    assert json.loads((tmp_path/'worker-ack.json').read_text())['hf_revision']==t['revision']
    value=json.loads(receipt.read_text());value['job_sha256']='different';value['sha256']=_hash({k:v for k,v in value.items() if k!='sha256'});receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):hf.acknowledge(t,receipt,client=store)

def test_job_runner_uploads_before_accepting_external_ack(tmp_path,monkeypatch):
    from test_sampling_sft_jobs import prepared
    from pilot_eval import sampling_sft_jobs as jobs
    root=tmp_path/'worker';job=prepared(root);store=Store(tmp_path/'remote')
    original=hf.receive_ack
    def client_download(record,destination,**kwargs):
        # Simulates the separate controller, including independently hashed files.
        local=tmp_path/'client';receipt=tmp_path/'external.json'
        hf.restore(record,local,receipt,client=store,provider=SimpleNamespace(host_id='external'))
        revision=hf.acknowledge(record,receipt,client=store)
        return original(record,destination,client=store,revision=revision)
    monkeypatch.setattr(hf,'receive_ack',client_download)
    result=jobs.execute(job,root,tmp_path/'network',root/'downloaded.json',hf_repo='owner/data',hf_client=store)
    assert result['phase']=='noop-completed'
    assert json.loads((root/'downloaded.json').read_text())['hf_revision']
    assert (job.parent/'hf-upload.json').is_file()

def test_failed_hf_upload_never_calls_shutdown(tmp_path,monkeypatch):
    from test_sampling_sft_jobs import prepared
    from pilot_eval import sampling_sft_jobs as jobs
    root=tmp_path/'worker';job=prepared(root);store=Store(tmp_path/'remote');store.private=False
    stopped=[];monkeypatch.setattr(jobs,'finish',lambda *a,**k:stopped.append(True))
    with pytest.raises(ValueError,match='private'):
        jobs.execute(job,root,tmp_path/'network',root/'downloaded.json',hf_repo='owner/data',hf_client=store)
    assert not stopped and (job.parent/'export-manifest.json').exists()

def test_upload_resume_reuses_sealed_revision(bundle):
    root,path,store=bundle;t=hf.upload(path,root,repo='owner/data',client=store)
    assert hf.upload(path,root,repo='owner/data',client=store)==t
    assert store.commits==1

def test_restore_rejects_archive_path_escape(bundle,tmp_path):
    import io,tarfile
    from pilot_eval.training import file_hash
    root,path,store=bundle;t=hf.upload(path,root,repo='owner/data',client=store)
    archive=store.fetch(t['repo'],t['archive'],t['revision'])
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo('../escape');member.size=1;stream.addfile(member,io.BytesIO(b'x'))
    t['archive_sha256']=file_hash(archive)
    with pytest.raises(ValueError):hf.restore(t,tmp_path/'client',tmp_path/'ack.json',client=store)
    assert not (tmp_path/'escape').exists()
