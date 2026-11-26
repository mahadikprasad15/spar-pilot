"""Finish the partial export of an offline GRPO run that stopped at the export step.

The driver's export can fail on a leftover `tokens.tmp.npz` listing on the network
volume. This reuses the run's saved report, so no stage runs or is re-verified.

Without --upload it writes the manifest and prints what it holds; nothing leaves the
machine. With --upload --manifest <path> it publishes that manifest to the plan's
private repository, downloads it again to check every file, and records the result.

Run from the repository:
python3 scripts/offline_grpo_finish_export.py --root <artifact_root> --config <plan>
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pilot_eval import offline_grpo_overnight as workflow  # noqa: E402
from pilot_eval import sampling_sft_exports as exports  # noqa: E402
from pilot_eval.sampling_sft_hf import upload  # noqa: E402
from pilot_eval.stage0_sources import inside  # noqa: E402
from pilot_eval.workflow import _hash  # noqa: E402

TEMPORARY = '.tmp.npz'


def _snapshot(directory, plan, chosen):
    """Name the saved report this export belongs to."""
    found = [p.parent for p in sorted((directory/'results/snapshots').glob('*/final-report.json'))]
    if chosen:
        found = [p for p in found if p.name == chosen]
    if len(found) != 1:
        raise ValueError(f'Expected one saved report, found {len(found)}; name one with --snapshot')
    report = json.loads((found[0]/'final-report.json').read_text())
    if report['source_plan_sha256'] != _hash(plan):
        raise ValueError('The saved report belongs to a different plan')
    return found[0]


def _ready(path, root):
    """Refuse a run that is still going or has already exported."""
    state = workflow.status(path, root)
    if state.get('state') != 'incomplete':
        raise ValueError('Expected a stopped run in state incomplete, found '+str(state.get('state')))
    if state.get('export', {}).get('state') in ['verified', 'partial_verified']:
        raise ValueError('This run has already exported')
    return state


def build_manifest(path, output_root, *, snapshot=None):
    """Write the manifest, leaving out temporary files that are never sealed results."""
    root = Path(output_root).resolve()
    plan, directory = workflow._load(path, root)
    with workflow._writer(directory/'execution'):
        _ready(path, root)
        folder = _snapshot(directory, plan, snapshot)
        original, skipped = exports.inventory, []

        def without_temporary(base, names):
            kept = []
            for name in names:
                (skipped if name.endswith(TEMPORARY) else kept).append(name)
            return original(base, kept)
        exports.inventory = without_temporary
        try:
            manifest = workflow._artifact_manifest(plan, root, directory, folder, partial=True)
        finally:
            exports.inventory = original
    return manifest, skipped


def publish(path, output_root, manifest_path, *, snapshot=None, dependencies=None):
    """Upload the manifest, check it by downloading it again, then record the result."""
    root = Path(output_root).resolve()
    plan, directory = workflow._load(path, root)
    deps = dependencies or workflow.Dependencies()
    with workflow._writer(directory/'execution'):
        state = _ready(path, root)
        _snapshot(directory, plan, snapshot)
        manifest_path = inside(directory/'exports', manifest_path)
        record = upload(manifest_path, root, repo=plan['spec']['export']['repo'], client=deps.hub())
        value = json.loads(manifest_path.read_text())
        if record['manifest_sha256'] != value['sha256']:
            raise ValueError('HF receipt does not bind the export manifest')
        workflow._fresh_readback(record, value, deps)
        state['export'] = dict(state='partial_verified', revision=record['revision'],
                               manifest_sha256=record['manifest_sha256'],
                               manifest_path=str(manifest_path.relative_to(root)),
                               external_sources=len(value['external_sources']))
        workflow._write_state(directory, plan, state)
    return state['export']


def summary(manifest_path):
    """Describe a manifest by folder and file type."""
    value = json.loads(Path(manifest_path).read_text())
    folders, counts, kinds = Counter(), Counter(), Counter()
    for name, entry in value['files'].items():
        folder = '/'.join(name.split('/')[:4])
        folders[folder] += entry['bytes']
        counts[folder] += 1
        kinds[name.rsplit('.', 1)[-1]] += entry['bytes']
    lines = [f"files {len(value['files'])}, {sum(folders.values())/1e9:.2f} GB", 'missing:']
    lines += ['  '+item for item in value['missing']]
    lines += [f'{size/1e9:8.2f} GB {counts[name]:6d} {name}' for name, size in folders.most_common(30)]
    lines += [f'{size/1e9:8.2f} GB .{name}' for name, size in kinds.most_common(10)]
    return '\n'.join(lines)


def main(argv=None, *, dependencies=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args(argv)
    if args.upload:
        if args.manifest is None:
            parser.error('--upload needs --manifest from the earlier run')
        print(json.dumps(publish(args.config, args.root, args.manifest, snapshot=args.snapshot,
                                 dependencies=dependencies), indent=2))
        return 0
    manifest, skipped = build_manifest(args.config, args.root, snapshot=args.snapshot)
    print('manifest', manifest)
    print('temporary files left out:', len(skipped))
    print(summary(manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
