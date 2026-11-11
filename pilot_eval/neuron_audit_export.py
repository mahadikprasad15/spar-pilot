"""Ticket 07: partial/complete export and clean-root restore for the joint neuron/adapter audit.

Snapshot -> private HF dataset upload -> independent pinned-revision download -> clean-root
restore, each with its own receipt. Callable after preparation, any stage or a failure; it
never waits for a report. Reuses the Sampling SFT transport (``sampling_sft_hf``): private
repo check, disk checks, retryable-error classification, bounded transfer deadlines and
archive+index commits.

Only stable sealed data is archived: sealed run directories (their ``complete.json`` seal
verifies), sealed block units (``seal.json`` verifies), plus status/log files of unsealed
runs as failure evidence. Unsealed or interrupted work is listed, not archived. A snapshot
is ``complete`` only when nothing expected is missing or unsealed; otherwise ``partial``.
Reconstructible inputs (model, datasets) are omitted with exact recovery requirements.

``may_stop_pod`` is True only after a verified restore of the same snapshot. Credentials
come from HF_TOKEN or the Hub credential store and are never written to any artifact.

Usage:
    python -m pilot_eval.neuron_audit_export snapshot --artifact_root <root> --include <rel> [...] \
        --label <label> [--expected_units <json>] [--omitted <json>]
    python -m pilot_eval.neuron_audit_export upload --snapshot <snapshot.json> --artifact_root <root> [--repo <repo>]
    python -m pilot_eval.neuron_audit_export restore --upload_receipt <receipt.json> --clean_root <dir>
"""
import argparse
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from pilot_eval.neuron_audit_blocks import BlockStore, CorruptUnit
from pilot_eval.sampling_sft_hf import Hub, private, require_space, retryable, transfer_deadline
from pilot_eval.training import file_hash

DEFAULT_REPO = 'Prasadmahadik/spar-sampling-sft-artifacts'
SNAPSHOT_SCHEMA = 'joint-neuron-adapter-audit/snapshot-v1'
EXPORT_PREFIX = 'neuron-audit'
SECRET = re.compile(r'hf_[A-Za-z0-9]{30,}|(?i:authorization:\s*bearer\s+\S+)')
TEXT_SUFFIXES = {'.json', '.jsonl', '.md', '.log', '.txt', '.yaml', '.yml', '.csv'}
EVIDENCE_NAMES = {'status.json', 'run.log', 'error.json', 'progress.json'}


class AuditHub(Hub):
    """Sampling SFT Hub client with an audit commit message and fetch into an independent dir.

    Never construct it in the offline worker (``HF_HUB_OFFLINE=1``); use ``OnlineHubProcess`` there."""

    def __init__(self):
        from huggingface_hub import constants
        if constants.HF_HUB_OFFLINE:
            raise RuntimeError('AuditHub created in an offline process (HF_HUB_OFFLINE=1); the worker must use '
                               'OnlineHubProcess, which runs every Hub call in a separate online process')
        super().__init__()

    def publish(self, repo, files, message='Archive joint neuron/adapter audit snapshot'):
        from huggingface_hub import CommitOperationAdd
        return self.api.create_commit(repo_id=repo, repo_type='dataset', commit_message=message,
                                      operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p))
                                                  for n, p in files.items()]).oid

    def fetch_to(self, repo, name, revision, directory):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=str(directory)))


class HubProcessError(RuntimeError):
    """A Hub call that failed in the online child process; carries its status for retry classification."""

    def __init__(self, message, *, status=None, transport=False):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status) if status is not None else None
        self.transport = transport


class HubTransportError(HubProcessError, ConnectionError):
    pass


class OnlineHubProcess:
    """Hub client for the offline worker: each call runs ``hub-op`` in a child process with the Hub online.

    The worker keeps ``HF_HUB_OFFLINE=1`` for model loading; uploads and pinned-revision readbacks happen in
    the same kind of online process that preflight's export check exercised. The token comes from the
    environment / credential store of the child and is never passed on the command line."""

    cache_root = None

    def __init__(self, *, runner=None, timeout=None):
        import subprocess
        self.runner, self.timeout = runner or subprocess.run, timeout

    def _call(self, op, **arguments):
        import sys
        env = dict(os.environ, HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0')
        command = [sys.executable, '-m', 'pilot_eval.neuron_audit_export', 'hub-op', '--request',
                   json.dumps(dict(op=op, **arguments))]
        result = self.runner(command, env=env, capture_output=True, text=True, timeout=self.timeout)
        lines = [l for l in (result.stdout or '').splitlines() if l.startswith('{')]
        reply = json.loads(lines[-1]) if lines else None
        if result.returncode != 0 or not reply or not reply.get('ok'):
            reply = reply or {}
            message = SECRET.sub('[redacted]', reply.get('message') or (result.stderr or '')[-1000:])
            kind = HubTransportError if reply.get('retryable') else HubProcessError
            raise kind(f"online Hub {op} failed: {reply.get('error_type')}: {message}", status=reply.get('status'),
                       transport=bool(reply.get('retryable')))
        return reply

    def info(self, repo):
        return SimpleNamespace(private=self._call('info', repo=repo)['private'])

    def publish(self, repo, files):
        return self._call('publish', repo=repo, files={n: str(p) for n, p in files.items()})['revision']

    def fetch_to(self, repo, name, revision, directory):
        return Path(self._call('fetch_to', repo=repo, name=name, revision=revision, directory=str(directory))['path'])


def hub_op(request):
    """Child side of ``OnlineHubProcess``; prints one JSON reply line."""
    try:
        client = AuditHub()
        if request['op'] == 'info':
            reply = dict(private=client.info(request['repo']).private is True)
        elif request['op'] == 'publish':
            reply = dict(revision=client.publish(request['repo'], {n: Path(p) for n, p in request['files'].items()}))
        elif request['op'] == 'fetch_to':
            reply = dict(path=str(client.fetch_to(request['repo'], request['name'], request['revision'],
                                                  request['directory'])))
        else:
            raise ValueError(f"unknown hub op {request['op']}")
        return dict(ok=True, **reply)
    except Exception as exc:  # reported to the parent, classified there
        return dict(ok=False, error_type=type(exc).__name__, message=SECRET.sub('[redacted]', str(exc))[-1000:],
                    status=getattr(getattr(exc, 'response', None), 'status_code', None), retryable=retryable(exc))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _relative(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'path escapes the artifact root: {path}')
    return str(path.relative_to(root))


def _checked_name(name):
    parts = Path(name).parts
    if not name or Path(name).is_absolute() or '..' in parts or name.startswith('/'):
        raise ValueError(f'unsafe archive path: {name}')
    return name


def _scan_secret(path):
    if path.suffix in TEXT_SUFFIXES and SECRET.search(path.read_text(errors='ignore')):
        raise ValueError(f'possible credential in {path.name}; refusing to export it')


# --- snapshot -----------------------------------------------------------------------------

def _run_seal_files(directory):
    """Files covered by a sealed run directory (inventory/sequences style ``complete.json``)."""
    seal = json.loads((directory / 'complete.json').read_text())
    if 'files' in seal:
        listed = dict(seal['files'])
    elif 'result_sha256' in seal:  # 03b stage seal
        listed = {'result.json': seal['result_sha256']}
    else:  # inventory seal: inventory.json, sources.json and key lists
        listed = {'inventory.json': seal['inventory_sha256'], 'sources.json': seal['sources_sha256'],
                  **seal.get('key_lists', {})}
    for name, digest in listed.items():
        path = directory / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'sealed run {directory.name}: {name} failed its seal')
    return sorted(listed) + ['complete.json']


def snapshot(artifact_root, include, label, *, expected_units=None, omitted=None, stage=None, pinned_inputs=None,
             include_unsealed_evidence=False):
    """Build and save a stable snapshot manifest; returns the manifest dict.

    ``pinned_inputs``: explicit input files (e.g. adapters, tokenizer) as {path, sha256}; each must match its
    recorded hash (from the sealed inventory) or the snapshot is refused.
    """
    root = Path(artifact_root).resolve()
    files, unsealed, corrupt, evidence = {}, [], [], []

    def add(path, kind):
        rel = _relative(root, path)
        _scan_secret(path)
        files[rel] = dict(sha256=file_hash(path), bytes=path.stat().st_size, kind=kind)

    sealed_units = set()
    for item in include:
        base = (root / item).resolve()
        if not base.exists():
            unsealed.append(dict(path=item, state='absent'))
            continue
        run_dirs = sorted({p.parent for p in base.rglob('complete.json')} | ({base} if (base / 'complete.json').exists() else set()))
        unit_roots = sorted({p.parent for p in base.rglob('units') if p.is_dir()})
        covered = set()
        for directory in run_dirs:
            if any(directory.is_relative_to(u) for u in unit_roots):
                continue
            for name in _run_seal_files(directory):
                add(directory / name, 'sealed-run')
                covered.add((directory / name).resolve())
            for extra in (directory / 'meta').glob('*.json') if (directory / 'meta').is_dir() else []:
                add(extra, 'status')
                covered.add(extra.resolve())
            review = directory / 'review' / 'review.json'
            if review.is_file():
                add(review, 'review-record')
                covered.add(review.resolve())
        for store_root in unit_roots:
            store = BlockStore(store_root)
            for key, state in store.status().items():
                unit_dir = store.unit_dir(key)
                rel_key = _relative(root, unit_dir)
                if state != 'sealed':
                    unsealed.append(dict(path=rel_key, state=state))
                    covered |= {p.resolve() for p in unit_dir.rglob('*')}
                    continue
                try:
                    store.verify_unit(key, force=True)
                except CorruptUnit as exc:
                    corrupt.append(dict(path=rel_key, error=str(exc)))
                    covered |= {p.resolve() for p in unit_dir.rglob('*')}
                    continue
                sealed_units.add(key)
                for path in sorted(unit_dir.iterdir()):
                    add(path, 'sealed-unit')
                    covered.add(path.resolve())
            interrupted = store_root / 'interrupted'
            if interrupted.is_dir():
                for path in interrupted.rglob('*'):
                    covered.add(path.resolve())
                evidence.append(dict(path=_relative(root, interrupted),
                                     entries=sorted(p.name for p in interrupted.iterdir()),
                                     note='interrupted attempts kept locally as evidence; not archived'))
        for path in sorted(p for p in base.rglob('*') if p.is_file()):
            if path.resolve() in covered or path.name.endswith('.tmp'):
                continue
            if path.name in EVIDENCE_NAMES or path.suffix == '.log':
                add(path, 'failure-evidence')
            else:
                unsealed.append(dict(path=_relative(root, path), state='unsealed-file'))
                if include_unsealed_evidence:  # diagnosis only: archived but labelled; snapshot stays partial
                    add(path, 'unsealed-evidence')
    for item in pinned_inputs or []:
        path = (root / item['path']).resolve()
        if not path.is_file() or file_hash(path) != item['sha256']:
            raise ValueError(f"pinned input {item['path']} is missing or differs from its recorded hash")
        add(path, 'pinned-input')
    missing_units = sorted(set(expected_units or []) - sealed_units)
    complete = not unsealed and not corrupt and not missing_units
    body = dict(schema=SNAPSHOT_SCHEMA, label=label, stage=stage, created_unix=time.time(),
                completeness='complete' if complete else 'partial',
                scope=dict(include=list(include), stage=stage, expected_units=sorted(expected_units or [])),
                completeness_meaning='complete = every included item and expected unit is sealed; it says '
                                     'nothing about stages or arms outside this scope',
                files=dict(sorted(files.items())),
                unsealed=unsealed, corrupt=corrupt, missing_expected_units=missing_units,
                interrupted_evidence=evidence, omitted_reconstructible=omitted or [],
                note='A verified partial export is not a complete audit.')
    body['sha256'] = _canonical_hash({k: v for k, v in body.items() if k != 'created_unix'})
    path = root / 'exports' / 'joint-neuron-adapter-audit' / label / f"snapshot-{body['sha256'][:16]}.json"
    _write_json(path, body)
    body['snapshot_path'] = _relative(root, path)
    return body


def _verify_snapshot_files(root, manifest):
    for rel, record in manifest['files'].items():
        path = root / _checked_name(rel)
        if not path.is_file() or path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError(f'snapshot file changed after snapshot: {rel}')


def _load_snapshot(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != SNAPSHOT_SCHEMA:
        raise ValueError('not an audit snapshot manifest')
    if _canonical_hash({k: v for k, v in manifest.items() if k not in ('sha256', 'created_unix')}) != manifest['sha256']:
        raise ValueError('snapshot manifest hash differs')
    return manifest


# --- upload / restore ----------------------------------------------------------------------

def _with_retries(action, *, attempts, on_event, sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:  # classify, never swallow non-transport failures
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403):
                raise PermissionError(f'HF authentication/authorization failed ({status}); local export preserved.') from exc
            if not (retryable(exc) or getattr(exc, 'transport', False)) or attempt == attempts:
                raise
            on_event(dict(kind='hf-retry', attempt=attempt, error_type=type(exc).__name__, status=status))
            sleep(min(30, 2 ** attempt))


def upload(snapshot_path, artifact_root, *, repo=DEFAULT_REPO, client=None, attempts=3, deadline_seconds=None,
           on_event=print, sleep=time.sleep, temp_dir=None):
    """``temp_dir`` defaults to the system temp dir: archives are built on local disk, never the volume."""
    root = Path(artifact_root).resolve()
    temp_dir = Path(temp_dir or tempfile.gettempdir())
    client = client or AuditHub()
    started = time.monotonic()
    private(client, repo)
    snapshot_path = Path(snapshot_path).resolve()
    manifest = _load_snapshot(snapshot_path)
    receipt_path = snapshot_path.with_name(snapshot_path.stem + '.hf-upload.json')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old['repo'] != repo or old['snapshot_sha256'] != manifest['sha256']:
            raise ValueError('existing upload receipt identity differs')
        return old
    _verify_snapshot_files(root, manifest)
    unpacked = sum(r['bytes'] for r in manifest['files'].values()) + snapshot_path.stat().st_size
    bound = math.ceil(unpacked * 1.01) + 10240 * (len(manifest['files']) + 2)
    require_space(temp_dir, 2 * bound)
    if getattr(client, 'cache_root', None) is not None:
        require_space(client.cache_root, bound)
    prefix = f"{EXPORT_PREFIX}/{manifest['label']}/{manifest['sha256']}"
    snapshot_rel = _relative(root, snapshot_path)
    with tempfile.TemporaryDirectory(prefix='audit-export-', dir=temp_dir) as temp:
        archive = Path(temp) / 'bundle.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            for name in sorted(set(manifest['files']) | {snapshot_rel}):
                target = root / _checked_name(name)
                if not target.is_file() or target.is_symlink():
                    raise ValueError('only regular files may be archived')
                stream.add(target, arcname=name, recursive=False)
        _verify_snapshot_files(root, manifest)  # nothing changed while archiving
        index = dict(schema_version=1, repo=repo, archive=prefix + '/bundle.tar.gz', snapshot_path=snapshot_rel,
                     snapshot_sha256=manifest['sha256'], completeness=manifest['completeness'],
                     archive_sha256=file_hash(archive), archive_bytes=archive.stat().st_size, unpacked_bytes=unpacked,
                     file_count=len(manifest['files']))
        index_path = Path(temp) / 'index.json'
        index_path.write_text(json.dumps(index, indent=2) + '\n')
        enabled = deadline_seconds is not None
        with transfer_deadline(deadline_seconds or 0, enabled):
            revision = _with_retries(lambda: client.publish(repo, {index['archive']: archive,
                                                                   prefix + '/index.json': index_path}),
                                     attempts=attempts, on_event=on_event, sleep=sleep)
        if not re.fullmatch('[0-9a-f]{40}', str(revision)):
            raise ValueError('HF commit did not return an exact revision')
        with tempfile.TemporaryDirectory(prefix='audit-readback-', dir=temp_dir) as fresh:
            fetched = _with_retries(lambda: client.fetch_to(repo, index['archive'], revision, fresh),
                                    attempts=attempts, on_event=on_event, sleep=sleep)
            if file_hash(fetched) != index['archive_sha256']:
                raise ValueError('HF readback hash differs; local export preserved, do not stop the Pod.')
    record = dict(index, revision=revision, upload_seconds=time.monotonic() - started)
    _write_json(receipt_path, record)
    return record


def restore(upload_receipt, clean_root, *, client=None, attempts=3, on_event=print, sleep=time.sleep):
    """Fresh pinned-revision download into an empty root; verify every item; write a receipt."""
    record = json.loads(Path(upload_receipt).read_text()) if not isinstance(upload_receipt, dict) else upload_receipt
    client = client or AuditHub()
    started = time.monotonic()
    private(client, record['repo'])
    root = Path(clean_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('restore target must be an empty, clean root')
    root.mkdir(parents=True, exist_ok=True)
    require_space(root, record['archive_bytes'] + record['unpacked_bytes'])
    with tempfile.TemporaryDirectory(prefix='audit-restore-') as fresh:
        archive = _with_retries(lambda: client.fetch_to(record['repo'], record['archive'], record['revision'], fresh),
                                attempts=attempts, on_event=on_event, sleep=sleep)
        if file_hash(archive) != record['archive_sha256']:
            raise ValueError('downloaded archive hash differs from the upload receipt')
        with tarfile.open(archive, 'r:gz') as stream:
            members = stream.getmembers()
            names = [m.name for m in members]
            if len(set(names)) != len(names) or any(not m.isfile() for m in members):
                raise ValueError('archive must contain unique regular files only')
            for name in names:
                _checked_name(name)
            manifest = json.load(stream.extractfile(record['snapshot_path']))
            if manifest['sha256'] != record['snapshot_sha256']:
                raise ValueError('archived snapshot identity differs')
            if set(names) != set(manifest['files']) | {record['snapshot_path']}:
                raise ValueError('archive contents differ from the snapshot inventory')
            for name in sorted(names):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as out:
                    out.write(stream.extractfile(name).read())
    _load_snapshot(root / record['snapshot_path'])
    _verify_snapshot_files(root, manifest)
    units = sorted({str(Path(n).parent) for n, r in manifest['files'].items() if r['kind'] == 'sealed-unit'})
    for unit in units:
        store = BlockStore((root / unit).parent.parent)
        store.verify_unit(Path(unit).name, force=True)
    proof = dict(role='restored', repo=record['repo'], revision=record['revision'],
                 snapshot_sha256=manifest['sha256'], archive_sha256=record['archive_sha256'],
                 completeness=manifest['completeness'], verified_files=len(manifest['files']),
                 verified_units=len(units), clean_root=str(root), restore_seconds=time.monotonic() - started,
                 missing=dict(unsealed=manifest['unsealed'], corrupt=manifest['corrupt'],
                              expected_units=manifest['missing_expected_units']),
                 omitted_reconstructible=manifest['omitted_reconstructible'])
    proof['sha256'] = _canonical_hash({k: v for k, v in proof.items() if k != 'restore_seconds'})
    _write_json(root.parent / f"{root.name}.restore-receipt.json", proof)
    return proof


def may_stop_pod(upload_record, restore_proof, *, require_complete=False):
    """Automatic Pod stop is allowed only after a verified restore of the same snapshot."""
    ok = (restore_proof.get('role') == 'restored' and restore_proof.get('revision') == upload_record.get('revision')
          and restore_proof.get('snapshot_sha256') == upload_record.get('snapshot_sha256')
          and restore_proof.get('archive_sha256') == upload_record.get('archive_sha256')
          and restore_proof.get('verified_files') == upload_record.get('file_count'))
    if require_complete:
        ok = ok and restore_proof.get('completeness') == 'complete'
    return bool(ok)


# --- per-arm packages and the reference-only final manifest -----------------------------------------

PACKAGE_SCHEMA = 'joint-neuron-adapter-audit/arm-package-v1'
FINAL_SCHEMA = 'joint-neuron-adapter-audit/final-manifest-v1'


def _deterministic_tar(path, members):
    """Uncompressed (FP32 data barely compresses), fixed metadata, sorted: identical inputs -> identical bytes."""
    with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as tar:
        for arcname, source in sorted(members):
            info = tar.gettarinfo(str(source), arcname=_checked_name(arcname))
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, '', '', 0o644
            info.pax_headers = {}
            with Path(source).open('rb') as stream:
                tar.addfile(info, stream)


def package_arm(store_root, arm, label, temp_dir, *, expected_keys=None):
    """Package one arm's sealed, verified units with its own manifest (package.json inside and beside).

    ``expected_keys`` (production): exactly these units, each sealed locally and belonging to ``arm``; a missing
    or foreign unit refuses the package, so a package can never be smaller than the arm's frozen batch list.
    """
    store = BlockStore(store_root)
    if expected_keys is None:
        keys = [k for k, s in store.status().items() if s == 'sealed']
    else:
        keys = sorted(expected_keys)
        missing = [k for k in keys if store.state(k) != 'sealed']
        if missing:
            raise ValueError(f'arm {arm}: {len(missing)} expected units are not sealed locally (e.g. {missing[:3]}); '
                             'restore them from the volume before packaging')
    units = {}
    for key in keys:
        store.verify_unit(key, force=True)
        if store.index(key)['identity'].get('arm') != arm:
            if expected_keys is not None:
                raise ValueError(f'unit {key} belongs to another arm')
            continue
        seal = json.loads((store.unit_dir(key) / 'seal.json').read_text())
        units[key] = dict(seal=seal['files'], seal_sha256=file_hash(store.unit_dir(key) / 'seal.json'))
    if not units:
        raise ValueError(f'no sealed units for arm {arm}')
    manifest = dict(schema=PACKAGE_SCHEMA, label=label, arm=arm, units=units, unit_count=len(units),
                    expected_units=sorted(expected_keys) if expected_keys is not None else None,
                    data_bytes=sum(r['size'] for u in units.values() for r in u['seal'].values()))
    manifest['sha256'] = _canonical_hash(manifest)
    work = Path(tempfile.mkdtemp(prefix=f'arm-{arm}-', dir=temp_dir))
    manifest_path = work / 'package.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    members = [('package.json', manifest_path)]
    for key in units:
        for name in sorted(units[key]['seal']) + ['seal.json']:
            members.append((f'units/{key}/{name}', store.unit_dir(key) / name))
    package = work / 'package.tar'
    _deterministic_tar(package, members)
    return package, manifest_path, manifest


def upload_package(package, manifest_path, manifest, *, repo, label, client=None, temp_dir=None, attempts=3,
                   on_event=print, sleep=time.sleep):
    """Upload one arm package and verify it by a fresh pinned-revision download into an empty directory."""
    client = client or AuditHub()
    private(client, repo)
    started = time.monotonic()
    digest = file_hash(package)
    prefix = f"{EXPORT_PREFIX}/{label}/arms/{manifest['arm']}/{digest}"
    revision = _with_retries(lambda: client.publish(repo, {prefix + '/package.tar': package,
                                                           prefix + '/package.json': manifest_path}),
                             attempts=attempts, on_event=on_event, sleep=sleep)
    if not re.fullmatch('[0-9a-f]{40}', str(revision)):
        raise ValueError('HF commit did not return an exact revision')
    with tempfile.TemporaryDirectory(prefix='arm-verify-', dir=temp_dir) as fresh:
        fetched = _with_retries(lambda: client.fetch_to(repo, prefix + '/package.tar', revision, fresh),
                                attempts=attempts, on_event=on_event, sleep=sleep)
        if file_hash(fetched) != digest:
            raise ValueError(f"arm {manifest['arm']} package readback differs; keep local units, do not stop the Pod")
    return dict(arm=manifest['arm'], repo=repo, revision=revision, path=prefix + '/package.tar',
                manifest_path=prefix + '/package.json', sha256=digest, bytes=Path(package).stat().st_size,
                package_manifest_sha256=manifest['sha256'], units=sorted(manifest['units']),
                verified='fresh pinned-revision download, sha256 equal', seconds=time.monotonic() - started)


def upload_directory_package(directory, *, repo, label, kind, client=None, temp_dir=None, attempts=3,
                             on_event=print, sleep=time.sleep):
    """Deterministic tar of every regular file under ``directory`` (e.g. the 03b qualification run), uploaded
    under ``neuron-audit/<label>/<kind>/<sha256>/`` and verified by a fresh pinned-revision download."""
    client = client or AuditHub()
    private(client, repo)
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    for path in files:
        _scan_secret(path)
    listing = {str(p.relative_to(directory)): dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in files}
    manifest = dict(schema=PACKAGE_SCHEMA, label=label, kind=kind, name=directory.name, files=listing)
    manifest['sha256'] = _canonical_hash(manifest)
    with tempfile.TemporaryDirectory(prefix=f'{kind}-', dir=temp_dir) as work:
        manifest_path = Path(work) / 'package.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        package = Path(work) / 'package.tar'
        _deterministic_tar(package, [('package.json', manifest_path)] +
                           [(f'files/{n}', directory / n) for n in listing])
        digest = file_hash(package)
        prefix = f'{EXPORT_PREFIX}/{label}/{kind}/{digest}'
        revision = _with_retries(lambda: client.publish(repo, {prefix + '/package.tar': package,
                                                               prefix + '/package.json': manifest_path}),
                                 attempts=attempts, on_event=on_event, sleep=sleep)
        with tempfile.TemporaryDirectory(prefix=f'{kind}-verify-', dir=temp_dir) as fresh:
            fetched = _with_retries(lambda: client.fetch_to(repo, prefix + '/package.tar', revision, fresh),
                                    attempts=attempts, on_event=on_event, sleep=sleep)
            if file_hash(fetched) != digest:
                raise ValueError(f'{kind} package readback differs; do not stop the Pod')
        size = package.stat().st_size
    return dict(kind=kind, name=directory.name, repo=repo, revision=revision, path=prefix + '/package.tar',
                sha256=digest, bytes=size, package_manifest_sha256=manifest['sha256'], files=len(listing),
                verified='fresh pinned-revision download, sha256 equal')


def _short_arms(arm_records, expected_units):
    """Arms whose verified package does not hold exactly the expected unit keys."""
    if expected_units is None:
        return {}
    by_arm = {r['arm']: set(r['units']) for r in arm_records}
    return {arm: sorted(set(keys) - by_arm[arm]) for arm, keys in expected_units.items()
            if arm in by_arm and by_arm[arm] != set(keys)}


def publish_final_manifest(arm_records, *, repo, label, required_arms, extras=None, client=None, temp_dir=None,
                           attempts=3, on_event=print, sleep=time.sleep, expected_units=None):
    """Reference-only final manifest: points at verified arm packages; never re-uploads their data.

    ``expected_units`` ({arm: [unit keys]}): completeness also requires every package to hold exactly those."""
    client = client or AuditHub()
    private(client, repo)
    missing = sorted(set(required_arms) - {r['arm'] for r in arm_records})
    short = _short_arms(arm_records, expected_units)
    body = dict(schema=FINAL_SCHEMA, label=label, completeness='complete' if not missing and not short else 'partial',
                required_arms=sorted(required_arms), missing_arms=missing, incomplete_arms=short,
                expected_unit_counts={a: len(k) for a, k in (expected_units or {}).items()},
                arms={r['arm']: {k: r[k] for k in ['repo', 'revision', 'path', 'manifest_path', 'sha256', 'bytes',
                                                   'package_manifest_sha256', 'units']} for r in arm_records},
                extras=extras or {}, note='References only; each arm package is stored once.')
    body['sha256'] = _canonical_hash(body)
    with tempfile.TemporaryDirectory(prefix='final-', dir=temp_dir) as work:
        path = Path(work) / 'final-manifest.json'
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')
        name = f"{EXPORT_PREFIX}/{label}/final/{body['sha256']}.json"
        revision = _with_retries(lambda: client.publish(repo, {name: path}), attempts=attempts, on_event=on_event,
                                 sleep=sleep)
        with tempfile.TemporaryDirectory(prefix='final-verify-', dir=temp_dir) as fresh:
            fetched = _with_retries(lambda: client.fetch_to(repo, name, revision, fresh), attempts=attempts,
                                    on_event=on_event, sleep=sleep)
            if file_hash(fetched) != file_hash(path):
                raise ValueError('final manifest readback differs; do not stop the Pod')
    return dict(repo=repo, revision=revision, path=name, sha256=body['sha256'], completeness=body['completeness'],
                missing_arms=missing, incomplete_arms=short, arms=sorted(body['arms']),
                expected_unit_counts=body['expected_unit_counts'])


def restore_final(final_record, clean_root, *, client=None, temp_dir=None, attempts=3, on_event=print,
                  sleep=time.sleep):
    """Clean-root restore from the final manifest: fetch each referenced package, verify, extract, re-verify units."""
    client = client or AuditHub()
    private(client, final_record['repo'])
    root = Path(clean_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('restore target must be an empty, clean root')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='final-restore-', dir=temp_dir) as fresh:
        manifest_file = _with_retries(lambda: client.fetch_to(final_record['repo'], final_record['path'],
                                                              final_record['revision'], fresh),
                                      attempts=attempts, on_event=on_event, sleep=sleep)
        body = json.loads(Path(manifest_file).read_text())
        if body.get('schema') != FINAL_SCHEMA or _canonical_hash({k: v for k, v in body.items() if k != 'sha256'}) != body['sha256'] \
                or body['sha256'] != final_record['sha256']:
            raise ValueError('final manifest identity differs')
        verified = {}
        for arm, ref in sorted(body['arms'].items()):
            package = _with_retries(lambda: client.fetch_to(ref['repo'], ref['path'], ref['revision'], fresh),
                                    attempts=attempts, on_event=on_event, sleep=sleep)
            if file_hash(package) != ref['sha256']:
                raise ValueError(f'arm {arm} package hash differs from the final manifest')
            with tarfile.open(package, 'r:') as tar:
                members = tar.getmembers()
                if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
                    raise ValueError(f'arm {arm} package must contain unique regular files only')
                for m in members:
                    target = root / 'arms' / arm / _checked_name(m.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('wb') as out:
                        out.write(tar.extractfile(m).read())
            store = BlockStore(root / 'arms' / arm)
            for key in ref['units']:
                store.verify_unit(key, force=True)
            verified[arm] = len(ref['units'])
            Path(package).unlink()  # keep local temp to one package at a time
        extra_packages = {}
        for kind in ['qualification_package']:
            ref = (body.get('extras') or {}).get(kind)
            if not ref:
                continue
            package = _with_retries(lambda: client.fetch_to(ref['repo'], ref['path'], ref['revision'], fresh),
                                    attempts=attempts, on_event=on_event, sleep=sleep)
            if file_hash(package) != ref['sha256']:
                raise ValueError(f'{kind} hash differs from the final manifest')
            target_root = root / 'qualification' / ref['name']
            with tarfile.open(package, 'r:') as tar:
                members = tar.getmembers()
                if any(not m.isfile() for m in members):
                    raise ValueError(f'{kind} must contain regular files only')
                listing = json.loads(tar.extractfile('package.json').read())['files']
                for m in members:
                    if m.name == 'package.json':
                        continue
                    rel = _checked_name(m.name)[len('files/'):]
                    target = target_root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = tar.extractfile(m).read()
                    if hashlib.sha256(data).hexdigest() != listing[rel]['sha256']:
                        raise ValueError(f'{kind} file {rel} failed its hash')
                    target.write_bytes(data)
            extra_packages[kind] = str(target_root)
            Path(package).unlink()
    proof = dict(role='restored-final', repo=final_record['repo'], revision=final_record['revision'],
                 final_sha256=body['sha256'], completeness=body['completeness'], missing_arms=body['missing_arms'],
                 verified_units=verified, clean_root=str(root), restored_extras=extra_packages)
    proof['sha256'] = _canonical_hash(proof)
    return proof


def restore_final_cli(progress_path, final_record_path, clean_root, *, temp_dir=None, require_complete=False,
                      client=None):
    """Restore every arm package of a final manifest into an empty root, verify every unit, write a receipt.

    Returns the proof plus ``store_roots`` (one per arm) to pass to 05 selection and 06 report as ``--stores``."""
    if progress_path:
        final = json.loads(Path(progress_path).read_text()).get('final_manifest')
        if not final:
            raise ValueError('progress record has no final manifest')
    else:
        final = json.loads(Path(final_record_path).read_text())
    if require_complete and final.get('completeness') != 'complete':
        raise ValueError(f"final manifest is {final.get('completeness')}; missing {final.get('missing_arms')}")
    proof = restore_final(final, clean_root, client=client, temp_dir=temp_dir)
    root = Path(proof['clean_root'])
    proof['store_roots'] = [str(root / 'arms' / arm) for arm in sorted(proof['verified_units'])]
    _write_json(root.parent / f'{root.name}.restore-final-receipt.json', proof)
    return proof


def may_stop_pod_final(final_record, arm_records, *, required_arms, restore_proof=None, expected_units=None):
    """Stop only when every required arm package is verified and the final manifest was read back.

    ``restore_proof`` (a full clean-root restore) is optional extra evidence: packages are already
    verified individually by fresh download when uploaded.
    """
    verified = {r['arm'] for r in arm_records if r.get('verified', '').startswith('fresh pinned-revision download')}
    ok = (final_record.get('completeness') == 'complete' and set(required_arms) <= verified
          and set(final_record.get('arms', [])) >= set(required_arms)
          and not final_record.get('incomplete_arms') and not _short_arms(arm_records, expected_units))
    if restore_proof is not None:
        ok = ok and restore_proof.get('final_sha256') == final_record['sha256'] and \
            set(restore_proof.get('verified_units', {})) >= set(required_arms)
    return bool(ok)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--artifact_root', required=True)
    s.add_argument('--include', nargs='+', required=True)
    s.add_argument('--label', required=True)
    s.add_argument('--stage')
    s.add_argument('--expected_units', help='JSON file with a list of expected unit keys')
    s.add_argument('--omitted', help='JSON file listing omitted reconstructible inputs')
    u = sub.add_parser('upload')
    u.add_argument('--snapshot', required=True)
    u.add_argument('--artifact_root', required=True)
    u.add_argument('--repo', default=DEFAULT_REPO)
    u.add_argument('--deadline_seconds', type=float)
    r = sub.add_parser('restore')
    r.add_argument('--upload_receipt', required=True)
    r.add_argument('--clean_root', required=True)
    f = sub.add_parser('restore-final', help='clean-root restore of a final manifest (for 05/06 off the Pod)')
    source = f.add_mutually_exclusive_group(required=True)
    source.add_argument('--progress', help="production meta/progress.json (uses its 'final_manifest')")
    source.add_argument('--final_record', help='JSON file with repo, revision, path and sha256 of the final manifest')
    f.add_argument('--clean_root', required=True, help='empty directory; arm stores land in <clean_root>/arms/<arm>')
    f.add_argument('--temp_dir', help='local temp for downloads (default: system temp)')
    f.add_argument('--require_complete', action='store_true', help='refuse a partial final manifest')
    h = sub.add_parser('hub-op', help='internal: one online Hub call for OnlineHubProcess')
    h.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    if args.command == 'hub-op':
        reply = hub_op(json.loads(args.request))
        print(json.dumps(reply))
        return 0 if reply['ok'] else 1
    if args.command == 'snapshot':
        load = (lambda p: json.loads(Path(p).read_text()) if p else None)
        result = snapshot(args.artifact_root, args.include, args.label, stage=args.stage,
                          expected_units=load(args.expected_units), omitted=load(args.omitted))
        result = {k: v for k, v in result.items() if k != 'files'} | dict(file_count=len(result['files']))
    elif args.command == 'restore-final':
        result = restore_final_cli(args.progress, args.final_record, args.clean_root, temp_dir=args.temp_dir,
                                   require_complete=args.require_complete)
    elif args.command == 'upload':
        result = upload(args.snapshot, args.artifact_root, repo=args.repo, deadline_seconds=args.deadline_seconds)
    else:
        result = restore(args.upload_receipt, args.clean_root)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
