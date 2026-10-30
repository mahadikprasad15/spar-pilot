"""Private HF transport of sealed job exports; restore preserves artifact paths.

Run with python -m pilot_eval.sampling_sft_hf. Credentials use HF_TOKEN or the
Hub credential store, never serialized configuration. This module rents no GPU.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import math
import shutil
import tarfile
import tempfile

from pilot_eval import sampling_sft_exports as exports
from pilot_eval.sampling_sft import frozen,save
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

DEFAULT_REPO='Prasadmahadik/spar-sampling-sft-artifacts'

class Hub:
    def __init__(self):
        from huggingface_hub import HfApi
        self.api=HfApi()
    @property
    def cache_root(self):
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    def info(self,repo):return self.api.repo_info(repo,repo_type='dataset')
    def publish(self,repo,files):
        from huggingface_hub import CommitOperationAdd
        return self.api.create_commit(repo_id=repo,repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(p)) for n,p in files.items()],
            commit_message='Archive sealed Sampling SFT job').oid
    def fetch(self,repo,name,revision):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision))


class TransferDeadlineExceeded(RuntimeError):
    """Distinct from transport timeouts, so retries cannot swallow allocation expiry."""


@contextmanager
def transfer_deadline(seconds,enabled):
    if not enabled:
        yield;return
    import signal
    if seconds<=0:raise TransferDeadlineExceeded('HF transfer allocation expired.')
    if not hasattr(signal,'setitimer'):raise ValueError('Bounded HF transfers require a supported worker signal timer.')
    def expired(*_):raise TransferDeadlineExceeded('HF transfer allocation expired; preserve data and reconcile budget.')
    previous=signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    import time
    began=time.monotonic()
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,old_timer[0]-(time.monotonic()-began)),old_timer[1])


def require_space(path,needed):
    directory=Path(path).resolve()
    while not directory.exists():directory=directory.parent
    if not directory.is_dir():directory=directory.parent
    free=shutil.disk_usage(directory).free
    if free<needed:raise ValueError(f'Insufficient disk space at {directory}: need {needed} bytes, have {free}.')


def retryable(error):
    """Transport failures only; auth, validation and disk failures remain fatal."""
    status=getattr(getattr(error,'response',None),'status_code',None)
    if status is not None:return status==429 or 500<=status<600
    if isinstance(error,(TimeoutError,ConnectionError)):return True
    module=type(error).__module__.split('.')[0]
    return (module in ['requests','httpx','httpx2','httpcore','httpcore2','urllib3','socket']
            and type(error).__name__ in ['Timeout','TimeoutError','ReadTimeout','ConnectTimeout',
                'WriteTimeout','PoolTimeout','ConnectionError','ConnectError','ReadError','WriteError','gaierror'])


def acknowledgement_forecast(record,download_bytes_per_second,verify_bytes_per_second,overhead_seconds=60):
    rates=[float(download_bytes_per_second),float(verify_bytes_per_second)]
    if any(not math.isfinite(v) or v<=0 for v in rates) or not math.isfinite(float(overhead_seconds)) or float(overhead_seconds)<0:
        raise ValueError('Measured positive transfer/hash rates and nonnegative overhead required.')
    if 'unpacked_bytes' not in record:raise ValueError('Re-export with unpacked size before forecasting acknowledgement time.')
    seconds=math.ceil(1.5*(record['archive_bytes']/rates[0]+record['unpacked_bytes']/rates[1])+float(overhead_seconds))
    return dict(archive_bytes=record['archive_bytes'],unpacked_bytes=record['unpacked_bytes'],
        download_bytes_per_second=rates[0],verify_bytes_per_second=rates[1],overhead_seconds=float(overhead_seconds),
        recommended_ack_seconds=seconds,scope='Transfer allowance only; reserve within total job budget before launch.')


def private(client,repo):
    if client.info(repo).private is not True:raise ValueError('HF destination must be private before any transfer.')


def upload(manifest_path,output_root,*,repo=DEFAULT_REPO,client=None):
    client=client or Hub();private(client,repo)
    root=Path(output_root).resolve();manifest_path=inside(root,manifest_path)
    manifest=exports.verify_manifest(manifest_path,root)
    relative=str(manifest_path.relative_to(root));prefix='exports/'+manifest['sha256']
    receipt_path=manifest_path.parent/'hf-upload.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text());validate(old)
        if old['repo']!=repo or old['manifest_sha256']!=manifest['sha256']:raise ValueError('Existing HF export identity differs.')
        if file_hash(client.fetch(repo,old['archive'],old['revision']))!=old['archive_sha256']:raise ValueError('Existing HF archive differs.')
        return old
    unpacked=sum(v['bytes'] for v in manifest['files'].values())+manifest_path.stat().st_size
    archive_bound=math.ceil(unpacked*1.01)+10240*(len(manifest['files'])+2)
    # Allow both construction and a read-back cache copy on the same filesystem.
    require_space(manifest_path.parent,2*archive_bound)
    if hasattr(client,'cache_root'):require_space(client.cache_root,archive_bound)
    # Two remote files per commit avoids a many-small-files upload transaction.
    with tempfile.TemporaryDirectory(prefix='hf-export-',dir=manifest_path.parent) as temp:
        archive=Path(temp)/'bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for name in sorted(set(manifest['files'])|{relative}):
                target=inside(root,exports.checked_relative(name))
                if not target.is_file() or target.is_symlink():raise ValueError('Only regular sealed files may be archived.')
                stream.add(target,arcname=name,recursive=False)
        exports.verify_manifest(manifest_path,root)
        metadata=dict(schema_version=1,repo=repo,archive=prefix+'/bundle.tar.gz',
            manifest_path=relative,manifest_sha256=manifest['sha256'],job_sha256=manifest['job_sha256'],
            archive_sha256=file_hash(archive),archive_bytes=archive.stat().st_size,unpacked_bytes=unpacked)
        index=Path(temp)/'index.json';index.write_text(json.dumps(metadata,indent=2)+'\n')
        revision=client.publish(repo,{metadata['archive']:archive,prefix+'/index.json':index})
        record={**metadata,'revision':revision};validate(record)
        # Read-back verification before accepting an upload receipt.
        if file_hash(client.fetch(repo,metadata['archive'],revision))!=metadata['archive_sha256']:
            raise ValueError('HF archive readback failed; preserve local export.')
        frozen(receipt_path,record)
        return record


def validate(record):
    if re.fullmatch('[0-9a-f]{40}',record.get('revision','')) is None:raise ValueError('Exact HF commit revision required.')
    for key in ['archive','manifest_path']:exports.checked_relative(record[key])
    if record['archive']!='exports/'+record['manifest_sha256']+'/bundle.tar.gz':raise ValueError('HF archive identity mismatch.')


def restore(record,copy_root,receipt_path,*,client=None,provider=None):
    client=client or Hub();validate(record);private(client,record['repo'])
    require_space(copy_root,record['archive_bytes']+record.get('unpacked_bytes',0))
    if hasattr(client,'cache_root'):require_space(client.cache_root,record['archive_bytes'])
    archive=client.fetch(record['repo'],record['archive'],record['revision'])
    if file_hash(archive)!=record['archive_sha256']:raise ValueError('HF download archive hash mismatch.')
    root=Path(copy_root).resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();by_name={m.name:m for m in members}
        if len(by_name)!=len(members):raise ValueError('Duplicate archive members.')
        for m in members:
            exports.checked_relative(m.name)
            if not m.isfile():raise ValueError('Only regular files accepted; archive links forbidden.')
        manifest=json.load(stream.extractfile(by_name[record['manifest_path']]))
        if manifest.get('sha256')!=record['manifest_sha256'] or manifest.get('job_sha256')!=record['job_sha256']:
            raise ValueError('Archived manifest identity mismatch.')
        if _hash({k:v for k,v in manifest.items() if k!='sha256'})!=manifest['sha256']:raise ValueError('Archived manifest hash differs.')
        if not manifest['simulation'] and exports.host(provider)==manifest['worker_host']:raise ValueError('Restore must run on an external download host.')
        expected=set(manifest['files'])|{record['manifest_path']}
        if set(by_name)!=expected:raise ValueError('Archive must contain exactly the sealed inventory and manifest.')
        require_space(root,sum(m.size for m in members))
        for name in sorted(expected):
            target=inside(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            digest=manifest['files'][name]['sha256'] if name in manifest['files'] else None
            # Copy through a temporary file; never overwrite a conflicting local artifact.
            with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as temp:
                temporary=Path(temp.name)
                try:
                    import shutil
                    shutil.copyfileobj(stream.extractfile(by_name[name]),temp);temp.flush()
                    import os
                    os.fsync(temp.fileno())
                    if digest and file_hash(temporary)!=digest:raise ValueError('Archived file hash differs: '+name)
                    if target.exists():
                        if file_hash(target)!=file_hash(temporary):raise ValueError('Restore conflicts with existing local artifact: '+name)
                    else:temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
    # Standard receipt remains consumable by the existing job-finish handshake.
    proof=exports.receipt(root/record['manifest_path'],root,'downloaded',(root/record['manifest_path']).parent/'hf-download-base-receipt.json',provider=provider)
    proof.update(hf_repo=record['repo'],hf_revision=record['revision'],hf_archive_sha256=record['archive_sha256'])
    proof['sha256']=_hash({k:v for k,v in proof.items() if k!='sha256'})
    frozen(Path(receipt_path),proof)
    return proof


def checked_ack(record,proof):
    if (proof.get('sha256')!=_hash({k:v for k,v in proof.items() if k!='sha256'})
            or proof.get('manifest_sha256')!=record['manifest_sha256']
            or proof.get('job_sha256')!=record['job_sha256']
            or proof.get('hf_revision')!=record['revision'] or proof.get('hf_repo')!=record['repo']
            or proof.get('role')!='downloaded'):
        raise ValueError('HF acknowledgement differs from the restored job/revision.')


def acknowledge(record,receipt_path,*,client=None):
    client=client or Hub();validate(record);private(client,record['repo'])
    proof=json.loads(Path(receipt_path).read_text());checked_ack(record,proof)
    # Receipt issuer must still possess and verify the actual downloaded copy.
    manifest=exports.verify_manifest(Path(proof['copy_root'])/record['manifest_path'],proof['copy_root'])
    if proof['verified_files']!=len(manifest['files']) or (not manifest['simulation'] and proof['host']==manifest['worker_host']):
        raise ValueError('External verified copy required before acknowledgement.')
    return client.publish(record['repo'],{'acknowledgements/'+record['manifest_sha256']+'.json':Path(receipt_path)})


def receive_ack(record,destination,*,client=None,revision='main',on_retry=None):
    client=client or Hub()
    try:
        path=client.fetch(record['repo'],'acknowledgements/'+record['manifest_sha256']+'.json',revision)
    except FileNotFoundError:return False
    except Exception as exc:
        if type(exc).__name__ in ['EntryNotFoundError','RemoteEntryNotFoundError']:return False
        if retryable(exc):
            event=dict(kind='hf-ack-retry',error_type=type(exc).__name__,status=getattr(getattr(exc,'response',None),'status_code',None))
            if on_retry:on_retry(event)
            else:print(json.dumps(event),flush=True)
            return False
        raise
    proof=json.loads(Path(path).read_text());checked_ack(record,proof)
    frozen(Path(destination),proof);return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    up=commands.add_parser('upload');up.add_argument('--manifest',type=Path,required=True);up.add_argument('--output-root',type=Path,required=True);up.add_argument('--repo',default=DEFAULT_REPO)
    down=commands.add_parser('restore');down.add_argument('--upload-receipt',type=Path,required=True);down.add_argument('--copy-root',type=Path,required=True);down.add_argument('--receipt',type=Path,required=True);down.add_argument('--acknowledge',action='store_true')
    forecast=commands.add_parser('forecast');forecast.add_argument('--upload-receipt',type=Path,required=True)
    forecast.add_argument('--download-bytes-per-second',type=float,required=True);forecast.add_argument('--verify-bytes-per-second',type=float,required=True);forecast.add_argument('--overhead-seconds',type=float,default=60)
    args=parser.parse_args()
    if args.command=='forecast':
        print(json.dumps(acknowledgement_forecast(json.loads(args.upload_receipt.read_text()),args.download_bytes_per_second,args.verify_bytes_per_second,args.overhead_seconds),indent=2));return
    result=upload(args.manifest,args.output_root,repo=args.repo) if args.command=='upload' else restore(json.loads(args.upload_receipt.read_text()),args.copy_root,args.receipt)
    if args.command=='restore' and args.acknowledge:
        acknowledge(json.loads(args.upload_receipt.read_text()),args.receipt)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
