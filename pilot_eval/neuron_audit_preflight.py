"""Ticket 03b worker preflight for the joint neuron/adapter audit (runs inside the actual worker).

Order matters: call ``prepare_environment`` before anything imports ``huggingface_hub`` or
``transformers`` (the Hub reads its offline flag at import time). Every check returns a record
and a list of problems; ``run`` raises ``PreflightFailure`` if any problem remains. Secrets are
never read into a record; only whether ``HF_TOKEN`` is set.
"""
import hashlib
import importlib.metadata
import os
import shutil
import sys
import time
from pathlib import Path


class PreflightFailure(RuntimeError):
    def __init__(self, problems, record):
        super().__init__('preflight failed: ' + '; '.join(problems))
        self.problems, self.record = problems, record


def prepare_environment(environ=None, modules=None):
    """Set offline/transfer/unbuffered flags before Hub imports; report anything set too late."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    problems = []
    late = sorted(m for m in ['huggingface_hub', 'transformers', 'peft'] if m in modules)
    if late and environ.get('HF_HUB_OFFLINE') != '1':
        problems.append(f'HF_HUB_OFFLINE set after importing {late}; restart the worker')
    environ['HF_HUB_OFFLINE'] = '1'
    environ['TRANSFORMERS_OFFLINE'] = '1'
    transfer = environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')
    if transfer == '1':
        try:
            __import__('hf_transfer')
        except ImportError:
            problems.append('HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not importable')
    else:
        environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    if environ.get('PYTHONUNBUFFERED') != '1':
        problems.append('PYTHONUNBUFFERED must be 1 for line-oriented logs')
    if environ.get('CUBLAS_WORKSPACE_CONFIG') is None:
        problems.append('CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes')
    record = dict(hf_hub_offline=environ['HF_HUB_OFFLINE'], hf_transfer=environ['HF_HUB_ENABLE_HF_TRANSFER'],
                  hf_home=environ.get('HF_HOME'), hf_hub_cache=environ.get('HF_HUB_CACHE'),
                  hf_token_set=bool(environ.get('HF_TOKEN')), cublas_workspace_config=environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  imported_before_offline=late)
    return record, problems


def check_interpreter(expected_executable, expected_prefix, packages, *, executable=None, prefix=None,
                      version=None):
    executable, prefix = executable or sys.executable, prefix or sys.prefix
    version = version or importlib.metadata.version
    problems = []
    if expected_executable is None or expected_prefix is None:
        problems.append('expected interpreter/prefix not recorded in the plan')
    if executable != expected_executable:
        problems.append(f'sys.executable {executable} != {expected_executable}')
    if prefix != expected_prefix:
        problems.append(f'sys.prefix {prefix} != {expected_prefix}')
    installed = {}
    for name, pinned in packages.items():
        try:
            installed[name] = version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != str(pinned):
            problems.append(f'{name} {installed[name]} != pinned {pinned}')
    return dict(sys_executable=executable, sys_prefix=prefix, packages=installed), problems


def storage_probe(directory, size_bytes=1 << 20, *, label):
    """The writer's exact operations: create, write, fsync, read back, rename, copyfile, delete."""
    directory = Path(directory)
    record, problems = dict(directory=str(directory), label=label, bytes=size_bytes, operations={}), []
    stem = directory / f'.preflight-{os.getpid()}-{time.time_ns()}'
    paths = [stem.with_suffix('.tmp'), stem.with_suffix('.bin'), stem.with_suffix('.copy')]
    payload = os.urandom(min(size_bytes, 1 << 20))

    def step(name, action):
        started = time.monotonic()
        try:
            action()
            record['operations'][name] = dict(ok=True, seconds=time.monotonic() - started)
            return True
        except OSError as exc:
            record['operations'][name] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
            problems.append(f'{label}: {name} failed ({type(exc).__name__})')
            return False

    digest = hashlib.sha256()

    def write():
        directory.mkdir(parents=True, exist_ok=True)
        with paths[0].open('wb') as stream:
            remaining = size_bytes
            while remaining > 0:
                chunk = payload[:remaining]
                stream.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())

    def read_back():
        if hashlib.sha256(paths[0].read_bytes()).hexdigest() != digest.hexdigest():
            raise OSError('read-back hash differs')

    try:
        for name, action in [('create_write_fsync', write), ('read_back_hash', read_back),
                             ('rename', lambda: os.replace(paths[0], paths[1])),
                             ('copyfile', lambda: shutil.copyfile(paths[1], paths[2]))]:
            if not step(name, action):
                break
    finally:
        step('delete', lambda: [p.unlink() for p in paths if p.exists()])
    return record, problems


def used_bytes(directory):
    total = 0
    for base, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total


def volume_record(volume_dir):
    """Global Volume: no provisioned capacity exists, so only usage is recorded (billing is per GB used)."""
    return dict(volume_dir=str(volume_dir), used_bytes=used_bytes(volume_dir), capacity_rule='none (global volume)')


def local_disk(local_dir):
    """Container disk free space; statvfs is trustworthy on the Pod's local overlay (unlike FUSE volumes)."""
    usage = shutil.disk_usage(local_dir)
    return dict(local_dir=str(local_dir), total_bytes=usage.total, free_bytes=usage.free)


def local_working_set(per_arm_bytes, model_cache_bytes=0):
    """One arm's units + its package + its verification download, plus the model cache."""
    return int(3 * per_arm_bytes + model_cache_bytes)


def confirm_storage(local, hf, *, per_arm_bytes, total_upload_bytes, model_cache_bytes=0,
                    local_fraction=0.60, hf_fraction=0.85, local_dir=None, probe_fraction=0.05):
    """Local working set <= 60% of free local disk; observed HF private + upload <= 85% of quota."""
    problems = []
    need_local = local_working_set(per_arm_bytes, model_cache_bytes)
    record = dict(local_need_bytes=need_local, local_allowed_bytes=int(local_fraction * local['free_bytes']),
                  hf_observed_private_bytes=hf.get('observed_private_bytes'), hf_quota_bytes=hf.get('quota_bytes'),
                  upload_bytes=int(total_upload_bytes))
    if need_local > local_fraction * local['free_bytes']:
        problems.append(f"local working set {need_local} > {local_fraction:.0%} of free local disk {local['free_bytes']}")
    if hf.get('observed_private_bytes') is None or not hf.get('quota_bytes'):
        problems.append('HF private usage/quota not measured')
    elif hf['observed_private_bytes'] + total_upload_bytes > hf_fraction * hf['quota_bytes']:
        problems.append(f"HF private usage {hf['observed_private_bytes']} + upload {total_upload_bytes} > "
                        f"{hf_fraction:.0%} of quota {hf['quota_bytes']}")
    record['hf_allowed_bytes'] = int(hf_fraction * hf['quota_bytes']) if hf.get('quota_bytes') else None
    if local_dir is not None and not problems:
        record['probe'], probe_problems = storage_probe(local_dir, max(1, int(per_arm_bytes * probe_fraction)),
                                                        label='local-capacity-probe')
        problems += probe_problems
    return record, problems


def private_usage(api):
    """Sum current file sizes over the account's private repos (history is not visible: a lower bound)."""
    me = api.whoami()['name']
    total, repos = 0, {}
    for kind, lister, info in [('dataset', api.list_datasets, api.dataset_info), ('model', api.list_models, api.model_info)]:
        for repo in lister(author=me):
            if getattr(repo, 'private', False):
                size = sum((s.size or 0) for s in info(repo.id, files_metadata=True).siblings or [])
                repos[f'{kind}:{repo.id}'] = size
                total += size
    return dict(observed_private_bytes=total, repos=repos, scope='current revisions only (lower bound)')


def hf_online_check(client, repo, temp_dir):
    """Separate online client: private repo, small upload, fresh pinned-revision download, hash compare."""
    record, problems = dict(repo=repo), []
    info = client.info(repo)
    record['private'] = info.private is True
    if not record['private']:
        return record, ['HF export repository is not private']
    payload = Path(temp_dir) / 'preflight-probe.txt'
    payload.write_bytes(f'neuron-audit preflight {time.time_ns()}\n'.encode())
    name = f'neuron-audit/preflight/{payload.stat().st_mtime_ns}.txt'
    revision = client.publish(repo, {name: payload})
    fresh = Path(temp_dir) / 'fresh'
    fresh.mkdir(exist_ok=True)
    fetched = client.fetch_to(repo, name, revision, fresh)
    same = hashlib.sha256(Path(fetched).read_bytes()).hexdigest() == hashlib.sha256(payload.read_bytes()).hexdigest()
    record.update(revision=revision, path=name, readback_matches=same)
    if hasattr(client, 'api'):
        record['usage'] = private_usage(client.api)
    if not same:
        problems.append('HF readback hash differs')
    return record, problems


def precision_and_device(torch_module=None):
    from pilot_eval.neuron_audit_collector import set_ieee_precision
    torch = torch_module or __import__('torch')
    torch.use_deterministic_algorithms(True)
    observed = set_ieee_precision()
    record = dict(precision=observed, deterministic=torch.are_deterministic_algorithms_enabled(),
                  torch=torch.__version__, cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record.update(device=props.name, total_memory=props.total_memory, cuda_version=torch.version.cuda)
    else:
        record.update(device='cpu', total_memory=None)
    return record, []


def hf_online_subprocess(repo, temp_dir, *, runner=None):
    """Run the online HF check in a separate process with the Hub online (the worker stays offline)."""
    import json
    import subprocess
    env = dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
    command = [sys.executable, '-m', 'pilot_eval.neuron_audit_preflight', 'hf-check', '--repo', repo, '--temp', str(temp_dir)]
    result = (runner or subprocess.run)(command, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return dict(repo=repo, returncode=result.returncode, stderr_tail=result.stderr[-2000:]), \
            ['online HF export check process failed']
    out = json.loads(result.stdout.strip().splitlines()[-1])
    return out['record'], out['problems']


def run(plan, *, volume_dir, local_temp, offline_load, hf_check=None, environ=None, interpreter=None, rehearsal=False):
    """Full preflight; ``offline_load`` performs a real offline model+tokenizer load and returns a record."""
    record, problems = {}, []
    record['environment'], p = prepare_environment(environ)
    problems += p
    pre = plan['preflight']
    record['interpreter'], p = (interpreter or check_interpreter)(
        pre['expected_sys_executable'], pre['expected_sys_prefix'], plan['runtime']['packages'])
    if rehearsal and pre['expected_sys_executable'] is None:
        # Colab rehearsal: no pinned interpreter paths; record them, keep the package pins.
        p = [x for x in p if not x.startswith(('sys.executable', 'sys.prefix', 'expected interpreter'))]
    problems += p
    record['precision_device'], p = precision_and_device()
    problems += p
    expected = plan['runtime'].get('expected_device')
    if expected and record['precision_device']['device'] != expected:
        problems.append(f"device {record['precision_device']['device']} != expected {expected}")
    for label, directory in [('volume', volume_dir), ('local-temp', local_temp)]:
        record[f'storage_{label}'], p = storage_probe(directory, label=label)
        problems += p
    record['volume'] = volume_record(volume_dir)
    record['local_disk'] = local_disk(local_temp)
    try:
        record['offline_load'] = offline_load()
    except Exception as exc:  # report, never mask the failure type
        problems.append(f'offline model/tokenizer load failed: {type(exc).__name__}: {exc}')
    if hf_check is not None:
        record['hf_online'], p = hf_check()
        problems += p
    else:
        problems.append('HF online export check was not run')
    record['problems'] = problems
    if problems:
        raise PreflightFailure(problems, record)
    return record


if __name__ == '__main__':
    import argparse
    import json
    import tempfile
    parser = argparse.ArgumentParser(description='Online HF export check (separate process).')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('hf-check')
    check.add_argument('--repo', required=True)
    check.add_argument('--temp', required=True)
    args = parser.parse_args()
    from pilot_eval.neuron_audit_export import AuditHub
    with tempfile.TemporaryDirectory(dir=args.temp) as temp:
        record, problems = hf_online_check(AuditHub(), args.repo, temp)
    print(json.dumps(dict(record=record, problems=problems)))
