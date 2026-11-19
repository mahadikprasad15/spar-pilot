"""C-attribution Pod input bundle (ticket 08).

A sealed, hash-listed snapshot of every input the run binds, so the Pod restores all of them (lesson 06a r1:
a prep snapshot is not all of a manifest's inputs). The export, upload and restore reuse
``neuron_audit_export`` (pinned inputs, private HF dataset, readback, clean-root restore); this module adds
the input list, the catalog revision, and a restore check against the **bundle seal** that says "missing" or
"differs" per file. The base model is not bundled: it comes from HF at its pinned revision and each file is
checked against the hashes recorded here.

    entries = default_entries()                      # roles and paths, relative to the artifact root
    bundle = build(artifact_root, entries, repo_root=REPO)   # hashes every file; refuses missing/differs
    receipt = export(artifact_root, bundle, label=..., client=hub)
    restored = restore(receipt, clean_root, client=hub)      # then verify(clean_root, bundle)
"""
import hashlib
import json
import subprocess
from pathlib import Path

from pilot_eval import neuron_audit_export as ex

SCHEMA = 'c-attribution-bundle-v1'
REPO = Path(__file__).resolve().parents[1]
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MODEL_DIR = f'inputs/models/{MODEL_REVISION}'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
MODEL_FILES = ('config.json', 'model.safetensors')
ONLINE = 'runs/pilot-4/Qwen--Qwen2.5-1.5B-Instruct/gsm8k/train-512-seed-42/rank1-float32/pilot4-grpo-fp32-v1-arm-v1/checkpoints'
OFFLINE = 'imports/offline-grpo-v1/runs/offline-grpo/offline-grpo-v1-training/training/checkpoints'


class BundleError(ValueError):
    pass


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def default_entries(*, offline_checkpoints=OFFLINE):
    """Every input the C-attribution run binds (relative to the artifact root), by role."""
    entries = [
        ('responses:online-grpo', 'imports/pilot4-behaviour-breakdown-2026-10-22/grpo/responses.jsonl'),
        ('responses:offline-grpo', 'imports/offline-grpo-v1/runs/offline-grpo/offline-grpo-v1-behaviour/behaviour/results/responses.jsonl'),
        ('run-config:online-grpo', 'imports/pilot4-behaviour-breakdown-2026-10-22/grpo/config.json'),
        ('run-config:offline-grpo', 'imports/offline-grpo-v1/runs/offline-grpo/offline-grpo-v1-behaviour/behaviour/config.json'),
        ('cohort-ids:gsm8k-test-cohort150', 'plans/pilot2-sft-fp32-l4-batch8-v1/evaluation.items.json'),
        ('hot-layer-source', 'analysis/results-hub-v1/data/writes_modules.json'),
        ('dataset:gsm8k-test', 'inputs/datasets/openai--gsm8k/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet'),
        ('generation-config', f'{MODEL_DIR}/generation_config.json'),
    ]
    for name in TOKENIZER_FILES:
        entries.append((f'tokenizer:{name}', f'{MODEL_DIR}/{name}'))
    for role, base in (('adapter:online-grpo:64', f'{ONLINE}/checkpoint-64'),
                       ('adapter:untuned-zero:0', f'{ONLINE}/checkpoint-0'),
                       ('adapter:offline-grpo:64', f'{offline_checkpoints}/checkpoint-64')):
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            entries.append((f'{role}:{name}', f'{base}/{name}'))
    return [{'role': r, 'path': p} for r, p in entries]


def catalog_revision(repo_root=REPO):
    """The commit and tree of ``catalog/`` plus each file's hash; refused if the catalog is absent or untracked."""
    repo_root = Path(repo_root)
    catalog = repo_root / 'catalog'
    if not catalog.is_dir():
        raise BundleError('missing catalog revision: no catalog/ in the code checkout')

    def git(*args):
        result = subprocess.run(['git', '-C', str(repo_root), *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise BundleError(f'missing catalog revision: git {" ".join(args)} failed: {result.stderr.strip()}')
        return result.stdout.strip()

    if git('status', '--porcelain', '--', 'catalog'):
        raise BundleError('catalog/ has uncommitted changes; commit it before building the bundle')
    files = {str(p.relative_to(repo_root)): sha256(p) for p in sorted(catalog.rglob('*')) if p.is_file()}
    return {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD:catalog'), 'files': files}


def _check(root, entry, expected=None):
    path = Path(root) / entry['path']
    if not path.is_file():
        raise BundleError(f"missing: {entry['role']} ({entry['path']})")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise BundleError(f"differs: {entry['role']} ({entry['path']}) sha256 {actual} != {expected}")
    return actual


def build(artifact_root, entries, *, repo_root=REPO, expected=None, model_dir=None, base_model='required'):
    """Hash every entry (refusing missing files and any mismatch against ``expected`` {role: sha256}).

    Every base-model file in ``MODEL_FILES`` must exist in ``model_dir`` and is hashed (fail closed). A CPU
    fixture without a model says ``base_model='not-bundled-fixture'``; such a bundle is refused by
    ``check_base_model``, so it can never pass on the Pod."""
    root = Path(artifact_root)
    roles = [e['role'] for e in entries]
    if len(set(roles)) != len(roles):
        raise BundleError('duplicate roles')
    files = []
    for entry in entries:
        files.append(dict(entry, sha256=_check(root, entry, (expected or {}).get(entry['role'])),
                          bytes=(root / entry['path']).stat().st_size))
    model = {}
    if base_model == 'required':
        model_dir = Path(model_dir) if model_dir else root / MODEL_DIR
        for name in MODEL_FILES:
            if not (model_dir / name).is_file():
                raise BundleError(f'missing: base model {name} ({model_dir})')
            model[name] = sha256(model_dir / name)
    elif base_model != 'not-bundled-fixture':
        raise BundleError(f"base_model must be 'required' or 'not-bundled-fixture', not {base_model!r}")
    body = {'schema': SCHEMA, 'files': files, 'catalog': catalog_revision(repo_root),
            'base_model': {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': MODEL_REVISION, 'file_sha256': model,
                           'mode': base_model,
                           'note': 'downloaded from HF at this revision on the Pod (HF_HOME on the container disk); '
                                   'each file is checked against these hashes'}}
    body['sha256'] = _canonical({k: v for k, v in body.items() if k != 'sha256'})
    return body


def verify(restored_root, bundle):
    """Check a restored root against the bundle seal (not by rebuilding from originals)."""
    if _canonical({k: v for k, v in bundle.items() if k != 'sha256'}) != bundle.get('sha256'):
        raise BundleError('bundle manifest hash differs from its seal')
    for entry in bundle['files']:
        _check(restored_root, entry, entry['sha256'])
    return {'verified_files': len(bundle['files']), 'bundle_sha256': bundle['sha256']}


def check_base_model(model_dir, bundle):
    """After the HF download on the Pod: every required base-model file must be recorded and match."""
    recorded = bundle['base_model'].get('file_sha256') or {}
    if bundle['base_model'].get('mode', 'required') != 'required' or sorted(recorded) != sorted(MODEL_FILES):
        raise BundleError(f'no base-model hashes for {sorted(set(MODEL_FILES) - set(recorded))}: this bundle '
                          'cannot verify a model')
    for name, expected in recorded.items():
        path = Path(model_dir) / name
        if not path.is_file():
            raise BundleError(f'missing: base model {name}')
        if sha256(path) != expected:
            raise BundleError(f'differs: base model {name}')
    return True


def pod_catalog_check(repo_root, bundle, restored_root, bound_entries):
    """On the Pod: the code revision holds the bundled catalog, and ``catalog verify`` passes the bound entries."""
    from pilot_eval import catalog as cat
    here = {str(p.relative_to(repo_root)): sha256(p) for p in sorted((Path(repo_root) / 'catalog').rglob('*')) if p.is_file()} \
        if (Path(repo_root) / 'catalog').is_dir() else None
    if here is None:
        raise BundleError('missing catalog revision: the Pod code checkout has no catalog/')
    if here != bundle['catalog']['files']:
        raise BundleError('the Pod catalog/ differs from the bundled catalog revision')
    result = cat.verify(cat.load(Path(repo_root) / 'catalog'), roots={'local': Path(restored_root)})
    by_entry = {}
    for check in result['checks']:
        by_entry.setdefault(check['entry'], []).append(check)
    report = {}
    for entry in bound_entries:
        checks = by_entry.get(entry)
        if not checks or any(c['status'] != 'pass' for c in checks):
            raise BundleError(f'catalog verify did not pass {entry}: {checks}')
        report[entry] = 'pass'
    return report


def export(artifact_root, bundle, *, label, client=None, repo=ex.DEFAULT_REPO, temp_dir=None, on_event=print):
    """Snapshot the bundle files as pinned inputs (plus the bundle manifest) and upload with readback."""
    root = Path(artifact_root)
    manifest_rel = f'bundles/c-attribution/{label}/bundle-{bundle["sha256"][:16]}.json'
    path = root / manifest_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != text:
        raise BundleError(f'{manifest_rel} exists with different content')
    path.write_text(text)
    pinned = [{'path': e['path'], 'sha256': e['sha256']} for e in bundle['files']] + \
             [{'path': manifest_rel, 'sha256': sha256(path)}]
    snap = ex.snapshot(root, [], label, pinned_inputs=pinned, stage='c-attribution-inputs')
    receipt = ex.upload(root / snap['snapshot_path'], root, repo=repo, client=client, temp_dir=temp_dir,
                        on_event=on_event)
    return dict(receipt, bundle_manifest=manifest_rel, bundle_sha256=bundle['sha256'])


def restore(receipt, clean_root, *, client=None, on_event=print):
    """Clean-root restore, then verification against the bundle seal it carries."""
    ex.restore(receipt, clean_root, client=client, on_event=on_event)
    bundle = json.loads((Path(clean_root) / receipt['bundle_manifest']).read_text())
    if bundle['sha256'] != receipt['bundle_sha256']:
        raise BundleError('restored bundle manifest differs from the receipt')
    return dict(verify(clean_root, bundle), bundle=bundle)
