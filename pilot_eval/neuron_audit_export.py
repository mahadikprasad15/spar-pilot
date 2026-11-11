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
    """Sampling SFT Hub client with an audit commit message and fetch into an independent dir."""

    def publish(self, repo, files, message='Archive joint neuron/adapter audit snapshot'):
        from huggingface_hub import CommitOperationAdd
        return self.api.create_commit(repo_id=repo, repo_type='dataset', commit_message=message,
                                      operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p))
                                                  for n, p in files.items()]).oid

    def fetch_to(self, repo, name, revision, directory):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=str(directory)))


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
    else:  # inventory seal: inventory.json, sources.json and key lists
        listed = {'inventory.json': seal['inventory_sha256'], 'sources.json': seal['sources_sha256'],
                  **seal.get('key_lists', {})}
    for name, digest in listed.items():
        path = directory / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'sealed run {directory.name}: {name} failed its seal')
    return sorted(listed) + ['complete.json']


def snapshot(artifact_root, include, label, *, expected_units=None, omitted=None, stage=None):
    """Build and save a stable snapshot manifest; returns the manifest dict."""
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
            if not retryable(exc) or attempt == attempts:
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
    args = parser.parse_args(argv)
    if args.command == 'snapshot':
        load = (lambda p: json.loads(Path(p).read_text()) if p else None)
        result = snapshot(args.artifact_root, args.include, args.label, stage=args.stage,
                          expected_units=load(args.expected_units), omitted=load(args.omitted))
        result = {k: v for k, v in result.items() if k != 'files'} | dict(file_count=len(result['files']))
    elif args.command == 'upload':
        result = upload(args.snapshot, args.artifact_root, repo=args.repo, deadline_seconds=args.deadline_seconds)
    else:
        result = restore(args.upload_receipt, args.clean_root)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
