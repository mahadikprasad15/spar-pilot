"""Hash-verified job bundles and copy receipts, usable from an offline client."""
import json
from pathlib import Path
import platform
import shutil

from pilot_eval.sampling_sft import save
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def checked_relative(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Export paths must be nonempty relative artifact paths.')
    return path


def host(provider=None):
    return getattr(provider,'host_id',platform.node())


def inventory(root, paths):
    files={}
    for name in paths:
        path=inside(root,checked_relative(name))
        if not path.is_file():raise ValueError('Missing required export payload: '+str(name))
        files[str(path.relative_to(root))]=dict(sha256=file_hash(path),bytes=path.stat().st_size)
    return files


def verify_manifest(path,root):
    value=json.loads(Path(path).read_text())
    payload={k:v for k,v in value.items() if k!='sha256'}
    if value.get('sha256')!=_hash(payload) or not value.get('files'):
        raise ValueError('Export manifest integrity/empty inventory failure.')
    if inventory(Path(root).resolve(),value['files'])!=value['files']:
        raise ValueError('Export copy payload/hash mismatch.')
    return value


def receipt(manifest_path,copy_root,role,receipt_path,*,provider=None):
    if role not in ['network','downloaded']:raise ValueError('Unknown export role.')
    copy_root=Path(copy_root).resolve()
    value=verify_manifest(manifest_path,copy_root)
    marker=Path(manifest_path).resolve().parent/'work-complete.json'
    expected=dict(manifest_sha256=value['sha256'],job_sha256=value['job_sha256'])
    if marker.exists() and json.loads(marker.read_text())!=expected:
        raise ValueError('Export work marker differs from the verified manifest.')
    if not marker.exists():save(marker,expected)
    actual_host=host(provider)
    if role=='downloaded' and not value['simulation'] and actual_host==value['worker_host']:
        raise ValueError('A live downloaded receipt must be verified on the external download host.')
    record=dict(schema_version=1,manifest_sha256=value['sha256'],job_sha256=value['job_sha256'],
        role=role,copy_root=str(copy_root),host=actual_host,verified_files=len(value['files']),
        simulation=value['simulation'],scope='client-verified hashes; receipt is an operator handoff, not authentication')
    record['sha256']=_hash(record)
    path=Path(receipt_path).resolve()
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError('Existing receipt differs; preserve it and select a new receipt path.')
    save(path,record)
    return record


def export(manifest_path,root,copy_root,role,receipt_path,*,provider=None):
    root=Path(root).resolve();copy_root=Path(copy_root).resolve()
    value=verify_manifest(manifest_path,root)
    if copy_root==root:
        if role!='network':raise ValueError('Downloaded copy must be independent of the worker artifact root.')
        return receipt(manifest_path,copy_root,role,receipt_path,provider=provider)
    if copy_root.is_relative_to(root/'runs/sampling-sft'):
        raise ValueError('Use a separate export root, not the active job tree.')
    # Never overwrite a different existing payload or follow a destination escape.
    for name,entry in value['files'].items():
        source=inside(root,checked_relative(name));target=inside(copy_root,name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and (not target.is_file() or file_hash(target)!=entry['sha256']):
            raise ValueError('Export destination already contains a different payload.')
        if not target.exists():
            temporary=target.with_suffix(target.suffix+'.copying')
            with source.open('rb') as left, temporary.open('wb') as right:
                shutil.copyfileobj(left,right)
                right.flush()
                import os
                os.fsync(right.fileno())
            if file_hash(temporary)!=entry['sha256']:raise ValueError('Export copy failed verification.')
            temporary.replace(target)
    relative_manifest=str(Path(manifest_path).resolve().relative_to(root))
    target_manifest=inside(copy_root,relative_manifest)
    if target_manifest.exists() and json.loads(target_manifest.read_text())!=value:
        raise ValueError('Destination manifest differs.')
    save(target_manifest,value)
    save(target_manifest.parent/'work-complete.json',dict(manifest_sha256=value['sha256'],job_sha256=value['job_sha256']))
    return receipt(manifest_path,copy_root,role,receipt_path,provider=provider)


def check_receipts(manifest,network_path,download_path,root):
    records=[]
    for role,path in [('network',network_path),('downloaded',download_path)]:
        value=json.loads(Path(path).read_text());payload={k:v for k,v in value.items() if k!='sha256'}
        if (value.get('sha256')!=_hash(payload) or value.get('role')!=role
                or value.get('manifest_sha256')!=manifest['sha256']
                or value.get('job_sha256')!=manifest['job_sha256']
                or value.get('verified_files')!=len(manifest['files'])
                or value.get('simulation')!=manifest['simulation']):
            raise ValueError('Export receipt is stale, altered or bound to another job.')
        if role=='network':
            if inventory(Path(value['copy_root']).resolve(),manifest['files'])!=manifest['files']:
                raise ValueError('Network copy no longer matches its receipt.')
        elif not manifest['simulation'] and value.get('host')==manifest['worker_host']:
            raise ValueError('Downloaded receipt must come from the external client.')
        records.append(value)
    if (records[0]['host'],records[0]['copy_root'])==(records[1]['host'],records[1]['copy_root']):
        raise ValueError('Network and downloaded receipts describe the same copy.')
    return records


def persist_control(directory,root,network_root):
    """Persist dynamic stop-intent evidence separately from immutable payloads."""
    import os
    target_dir=inside(Path(network_root).resolve(),directory.relative_to(root))
    for name in ['state.json','events.jsonl']:
        source=directory/name;target=target_dir/name
        if source.resolve()==target.resolve():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.pending')
        with temporary.open('wb') as stream:
            stream.write(source.read_bytes());stream.flush();os.fsync(stream.fileno())
        temporary.replace(target)
