"""Ticket 03b worker preflight for the joint neuron/adapter audit (runs inside the actual worker).

Order matters: call ``prepare_environment`` before anything imports ``huggingface_hub`` or
``transformers`` (the Hub reads its offline flag at import time). Every check returns a record
and a list of problems; ``run`` raises ``PreflightFailure`` if any problem remains. Secrets are
never read into a record; only whether ``HF_TOKEN`` is set.
"""
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VOLUME_SENTINEL = '.neuron-audit-volume.json'


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


def version_matches(installed, pinned, allowed_local=()):
    """Public version must equal the pin; a local build suffix (``+cu128``) is accepted only if the plan lists
    it. A pin that itself carries a suffix must match exactly. The full installed string is always recorded."""
    if installed is None:
        return False
    pinned = str(pinned)
    if '+' in pinned:
        return installed == pinned
    public, _, local = installed.partition('+')
    return public == pinned and (not local or local in set(allowed_local or ()))


def check_interpreter(expected_executable, expected_prefix, packages, *, executable=None, prefix=None,
                      version=None, allowed_local=None):
    executable, prefix = executable or sys.executable, prefix or sys.prefix
    allowed_local = allowed_local or {}
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
        if not version_matches(installed[name], pinned, allowed_local.get(name, ())):
            problems.append(f'{name} {installed[name]} != pinned {pinned} '
                            f'(allowed local builds: {list(allowed_local.get(name, ())) or "none"})')
    return dict(sys_executable=executable, sys_prefix=prefix, packages=installed,
                allowed_local_builds={k: list(v) for k, v in allowed_local.items()}), problems


def storage_probe(directory, size_bytes=1 << 20, *, label, create=True):
    """The writer's exact operations: create, write, fsync, read back, rename, copyfile, delete.

    ``create=False`` (the volume): a missing directory is a failure, never created on local disk."""
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
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise FileNotFoundError(f'{directory} does not exist (not created by the probe)')
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


def _mount_point(path):
    path = Path(path).resolve()
    while not os.path.ismount(path):
        path = path.parent
    return path


def _mount_entry(mount, mounts_text):
    found = None
    for line in (mounts_text or '').splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == str(mount):
            found = dict(source=parts[0], fstype=parts[2])
    return found


def volume_identity(volume_dir, *, expected_id, local_dir, mounts_text=None, require_sentinel=True,
                    mount_of=None, device_of=None):
    """Establish that ``volume_dir`` is the intended persistent volume, not the Pod's container disk.

    Requires: the directory already exists (never created here); it lives under a different mount point and
    device than local disk; and it holds the identity file written once by ``init-volume`` naming the volume.
    """
    volume, record = Path(volume_dir), dict(volume_dir=str(volume_dir), expected_id=expected_id)
    if not expected_id:
        return record, ['storage.global_volume.id is not recorded in the plan']
    if not volume.is_dir():
        return record, [f'volume directory {volume} does not exist; attach/mount {expected_id} first '
                        '(preflight never creates it)']
    mount_of = mount_of or _mount_point
    device_of = device_of or (lambda p: Path(p).stat().st_dev)
    mount, local_mount = Path(mount_of(volume)), Path(mount_of(local_dir))
    if mounts_text is None and Path('/proc/mounts').is_file():
        mounts_text = Path('/proc/mounts').read_text()
    record.update(mount_point=str(mount), local_mount_point=str(local_mount), device=device_of(volume),
                  local_device=device_of(local_dir), mount=_mount_entry(mount, mounts_text))
    problems = []
    if mount == local_mount or record['device'] == record['local_device'] or mount == Path('/'):
        problems.append(f'{volume} is on the same filesystem as local disk ({mount}); {expected_id} is not mounted there')
    sentinel = volume / VOLUME_SENTINEL
    if sentinel.is_file():
        try:
            record['identity'] = json.loads(sentinel.read_text())
        except json.JSONDecodeError:
            record['identity'] = None
        if (record['identity'] or {}).get('volume_id') != expected_id:
            problems.append(f'volume identity file names {(record["identity"] or {}).get("volume_id")}, not {expected_id}')
    elif require_sentinel:
        problems.append(f'volume identity file {VOLUME_SENTINEL} is missing; confirm the mount and run '
                        '`python -m pilot_eval.neuron_audit_preflight init-volume` once')
    return record, problems


def init_volume(volume_dir, *, volume_id, local_dir, mounts_text=None, **probes):
    """Write the identity file once, only after the mount checks pass (never over an existing one)."""
    record, problems = volume_identity(volume_dir, expected_id=volume_id, local_dir=local_dir,
                                       mounts_text=mounts_text, require_sentinel=False, **probes)
    if problems:
        raise PreflightFailure(problems, record)
    sentinel = Path(volume_dir) / VOLUME_SENTINEL
    if sentinel.exists():
        return record
    body = dict(volume_id=volume_id, created_unix=time.time(), mount=record.get('mount'),
                mount_point=record['mount_point'])
    with sentinel.open('w') as stream:
        stream.write(json.dumps(body, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return dict(record, identity=body)


def check_code_commit(expected, *, repo_dir=None, runner=subprocess.run):
    """The running checkout must be the plan's pushed commit with no tracked modifications."""
    repo_dir = Path(repo_dir or Path(__file__).resolve().parents[1])
    record, problems = dict(expected=expected, repo_dir=str(repo_dir)), []
    try:
        head = runner(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        dirty = runner(['git', '-C', str(repo_dir), 'status', '--porcelain', '--untracked-files=no'],
                       capture_output=True, text=True, check=True)
        record.update(head=head.stdout.strip(), tracked_changes=[l for l in dirty.stdout.splitlines() if l.strip()])
    except (OSError, subprocess.CalledProcessError) as exc:
        return record, [f'cannot read the code commit ({type(exc).__name__})']
    if not expected:
        problems.append('runtime.code_commit is not recorded in the plan')
    elif record['head'] != expected:
        problems.append(f"running commit {record['head']} != plan runtime.code_commit {expected}")
    if record['tracked_changes']:
        problems.append(f"tracked files differ from the commit: {record['tracked_changes'][:5]}")
    return record, problems


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


def runtime_fingerprint(device_info, torch_module=None):
    """What 03b's numerical evidence is about: device, memory, torch/CUDA/cuDNN builds, Python. A production run
    must match it exactly; a different Pod with the same hardware and builds matches (hostnames are not part)."""
    torch = torch_module or __import__('torch')
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return dict(device=device_info.get('name'), total_memory=device_info.get('total_memory'),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=cudnn, python=sys.version.split()[0])


def hf_online_subprocess(repo, temp_dir, *, runner=None):
    """Run the online HF check in a separate process with the Hub online (the worker stays offline)."""
    env = dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
    command = [sys.executable, '-m', 'pilot_eval.neuron_audit_preflight', 'hf-check', '--repo', repo, '--temp', str(temp_dir)]
    result = (runner or subprocess.run)(command, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return dict(repo=repo, returncode=result.returncode, stderr_tail=result.stderr[-2000:]), \
            ['online HF export check process failed']
    out = json.loads(result.stdout.strip().splitlines()[-1])
    return out['record'], out['problems']


def run(plan, *, volume_dir, local_temp, offline_load, hf_check=None, environ=None, interpreter=None, rehearsal=False,
        volume_check=None, commit_check=None):
    """Full preflight; ``offline_load`` performs a real offline model+tokenizer load and returns a record.

    Rehearsals (Colab: no sl-0, no pinned interpreter) record the volume and commit checks as warnings."""
    record, problems = {}, []
    warnings = record['warnings'] = []
    record['environment'], p = prepare_environment(environ)
    problems += p
    pre = plan['preflight']
    record['interpreter'], p = (interpreter or check_interpreter)(
        pre['expected_sys_executable'], pre['expected_sys_prefix'], plan['runtime']['packages'],
        allowed_local=plan['runtime'].get('allowed_local_builds') or {})
    if rehearsal and pre['expected_sys_executable'] is None:
        # Colab rehearsal: no pinned interpreter paths; record them, keep the package pins.
        p = [x for x in p if not x.startswith(('sys.executable', 'sys.prefix', 'expected interpreter'))]
    problems += p
    record['precision_device'], p = precision_and_device()
    problems += p
    expected = plan['runtime'].get('expected_device')
    if expected and record['precision_device']['device'] != expected:
        problems.append(f"device {record['precision_device']['device']} != expected {expected}")
    volume_id = ((plan.get('storage') or {}).get('global_volume') or {}).get('id')
    record['volume_identity'], p = (volume_check or volume_identity)(volume_dir, expected_id=volume_id,
                                                                     local_dir=local_temp)
    (warnings if rehearsal else problems).extend(p)
    record['code_commit'], p = (commit_check or check_code_commit)(plan['runtime'].get('code_commit'))
    (warnings if rehearsal else problems).extend(p)
    for label, directory, create in [('volume', volume_dir, rehearsal), ('local-temp', local_temp, True)]:
        record[f'storage_{label}'], p = storage_probe(directory, label=label, create=create)
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
    parser = argparse.ArgumentParser(description='Online HF export check (separate process) and volume init.')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('hf-check')
    check.add_argument('--repo', required=True)
    check.add_argument('--temp', required=True)
    init = sub.add_parser('init-volume', help='write the volume identity file once, after mount checks pass')
    init.add_argument('--volume_dir', required=True)
    init.add_argument('--volume_id', required=True)
    init.add_argument('--local_dir', required=True, help='a directory on the container disk')
    args = parser.parse_args()
    if args.command == 'init-volume':
        print(json.dumps(init_volume(args.volume_dir, volume_id=args.volume_id, local_dir=args.local_dir), indent=2))
        raise SystemExit(0)
    from pilot_eval.neuron_audit_export import AuditHub
    with tempfile.TemporaryDirectory(dir=args.temp) as temp:
        record, problems = hf_online_check(AuditHub(), args.repo, temp)
    print(json.dumps(dict(record=record, problems=problems)))
