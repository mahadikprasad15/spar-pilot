"""Sparse rewiring Pod input bundle (tickets 01 and 15).

The sparse arms bind a different input set from C attribution, so they get their own bundle, built and restored with
``c_attribution_bundle``'s machinery unchanged (hash every file, pin the base model to HF's own identities, bind the
catalog revision, export as pinned inputs to the private HF dataset with readback, clean-root restore checked
against the bundle seal).

Roles and paths are relative to the artifact root and come from ticket 01's ``source-records.md`` (all of them were
hashed there on the researcher's machine, under ``~/spar/pilot1/artifacts``):

- gold SFT's frozen plan: ``sft.config.json``, ``training.items.json`` (the 512 training rows),
  ``evaluation.items.json`` (cohort150, used by production only) and gold's saved ``inputs/order.json``;
- the gold-SFT step-64 adapter (the D20 reference line), as ``adapter:gold-sft:64:<file>``;
- ticket 04's frozen validation set (``validation.items.json`` and its exclusion record);
- the pinned tokenizer files and ``generation_config.json`` (``c_attribution_bundle.TOKENIZER_FILES``).

The base model is not bundled; the Pod downloads it at ``MODEL_REVISION`` and preflight checks each file.

    # on the machine that holds the artifact root (it needs HF write access):
    python -m pilot_eval.rewire_bundle export --artifact-root ~/spar/pilot1/artifacts --label rehearsal-pod-1 \\
        --receipt .scratch/sparse-rewiring-gsm8k/bundle-receipt-rehearsal-pod-1.json
    # on the Pod:
    python -m pilot_eval.rewire_bundle restore --receipt <receipt> --clean-root $WORK/inputs
"""
import argparse
import json
from pathlib import Path

from pilot_eval import c_attribution_bundle as cb
from pilot_eval import neuron_audit_export as ex

GOLD_PLAN = 'plans/pilot2-sft-fp32-l4-batch8-v1'
GOLD_RUN = ('runs/pilot-2/Qwen--Qwen2.5-1.5B-Instruct/gsm8k/train-512-seed-42/rank1-float32/'
            'pilot2-sft-fp32-l4-batch8-v1')
VALIDATION = 'validation/sparse-rewiring-ticket04-v1'
GOLD_ADAPTER_ROLE = 'adapter:gold-sft:64'   # = rewire_chain.GOLD_ADAPTER_ROLE

ROLES = {
    'gold:sft-config': f'{GOLD_PLAN}/sft.config.json',
    'gold:training-items': f'{GOLD_PLAN}/training.items.json',
    'gold:evaluation-items': f'{GOLD_PLAN}/evaluation.items.json',
    'gold:order': f'{GOLD_RUN}/inputs/order.json',
    'validation:items': f'{VALIDATION}/validation.items.json',
    'validation:exclusions': f'{VALIDATION}/validation.exclusions.json',
}
# Ticket 01's SHA-256 of the file bytes (source-records.md); ``export`` refuses a file that differs.
EXPECTED = {
    'gold:sft-config': 'e70679ab52600e5cca8c33daa1ee7c6206ed309daa5796346a9bc97c24da87e1',
    'gold:training-items': 'f27d0d75740438adafcdc3bf56104ac6fee06f80784366e5fc4e1a2c5850d260',
    'gold:evaluation-items': '76383395095c48a048b1477dc45ab1426bd0c3ca9d02f173c8a152f9e239874a',
    'gold:order': '3a92872e79ef88c2b7534e997370fd53c21ca95b98ab36b73ec5a4b9c942c0ab',
    f'{GOLD_ADAPTER_ROLE}:adapter_model.safetensors':
        'a20a344efabe576669b3c89701b161dd568efdf776bb512f7f128f9a8b75aeaa',
}
VALIDATION_IDS_SHA256 = '5ea47bc5954a15f4570af511bca8e623c0ca7ab27fab8f1b35ccf80fea082e61'  # ticket 04, frozen


def default_entries():
    entries = [{'role': role, 'path': path} for role, path in ROLES.items()]
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        entries.append({'role': f'{GOLD_ADAPTER_ROLE}:{name}', 'path': f'{GOLD_RUN}/checkpoints/checkpoint-64/{name}'})
    entries.append({'role': 'generation-config', 'path': f'{cb.MODEL_DIR}/generation_config.json'})
    for name in cb.TOKENIZER_FILES:
        entries.append({'role': f'tokenizer:{name}', 'path': f'{cb.MODEL_DIR}/{name}'})
    return entries


def path_of(bundle, artifact_root, role):
    entry = next((e for e in bundle['files'] if e['role'] == role), None)
    if entry is None:
        raise cb.BundleError(f'the input bundle has no {role}')
    return Path(artifact_root) / entry['path']


def check_validation(bundle, artifact_root, *, questions):
    """The bundled validation set is ticket 04's frozen set: its exclusion record's IDs hash and count."""
    record = json.loads(path_of(bundle, artifact_root, 'validation:exclusions').read_text())
    items = json.loads(path_of(bundle, artifact_root, 'validation:items').read_text())
    if record.get('ids_sha256') != VALIDATION_IDS_SHA256 or [i['id'] for i in items] != record.get('ids'):
        raise cb.BundleError('the bundled validation set is not ticket 04\'s frozen set')
    if len(items) != int(questions):
        raise cb.BundleError(f'{len(items)} validation questions; the launch record says {questions}')
    return items


COHORT150_SIZE = 150


def check_cohort150(bundle, artifact_root):
    """Gold's frozen cohort150 (``evaluation.items.json``): ticket 01's file hash and 150 ``gsm8k:test`` items, each
    with its rendered prompt and gold answer. Production only (D12: evaluated at the final step)."""
    path = path_of(bundle, artifact_root, 'gold:evaluation-items')
    if cb.sha256(path) != EXPECTED['gold:evaluation-items']:
        raise cb.BundleError("the bundled cohort150 is not ticket 01's evaluation.items.json")
    items = json.loads(path.read_text())
    ids = [i.get('id') for i in items]
    if (len(items) != COHORT150_SIZE or len(set(ids)) != COHORT150_SIZE
            or not all(str(q).startswith('gsm8k:test:') for q in ids)
            or not all(isinstance(i.get('prompt'), str) and i['prompt'] and i.get('gold') for i in items)):
        raise cb.BundleError('the bundled cohort150 is not 150 distinct gsm8k:test items with prompt and gold')
    return items


def main(argv=None, *, client=None):
    parser = argparse.ArgumentParser(description='Sparse rewiring input bundle')
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--artifact-root', required=True)
    e.add_argument('--label', required=True)
    e.add_argument('--receipt', required=True, help='receipt JSON to write (commit it: the Pod restores from it)')
    e.add_argument('--repo', default=ex.DEFAULT_REPO)
    e.add_argument('--model-dir', help='local snapshot of the pinned base model (default: <root>/' + cb.MODEL_DIR + ')')
    r = sub.add_parser('restore')
    r.add_argument('--receipt', required=True)
    r.add_argument('--clean-root', required=True)
    args = parser.parse_args(argv)
    if client is None:
        client = ex.OnlineHubProcess(timeout=3600)
    if args.command == 'export':
        receipt_path = Path(args.receipt)
        if receipt_path.exists():
            raise cb.BundleError(f'{receipt_path} exists; use a new label (receipts are never overwritten)')
        root = Path(args.artifact_root).expanduser()
        bundle = cb.build(root, default_entries(), expected=EXPECTED, model_dir=args.model_dir)
        check_validation(bundle, root, questions=128)
        temp = root / 'tmp' / 'bundle-export'
        temp.mkdir(parents=True, exist_ok=True)
        receipt = cb.export(root, bundle, label=f'sparse-rewiring-{args.label}', client=client, repo=args.repo,
                            temp_dir=temp)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n')
        print(f"exported bundle {bundle['sha256'][:16]} ({len(bundle['files'])} files); receipt {receipt_path}")
        return 0
    receipt = json.loads(Path(args.receipt).read_text())
    out = cb.restore(receipt, args.clean_root, client=client)
    print(json.dumps(dict(verified_files=out['verified_files'], bundle_sha256=out['bundle_sha256'],
                          bundle_manifest=str(Path(args.clean_root) / receipt['bundle_manifest']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
