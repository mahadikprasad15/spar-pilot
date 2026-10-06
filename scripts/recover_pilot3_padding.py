"""Archive only the pre-measurement missing-padding failure before retrying."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_eval.training import run_lock, file_hash
from pilot_eval.workflow import _hash


def archive_failed_padding_validation(directory, prepared_sha256):
    directory = Path(directory)
    if not directory.exists():
        return None
    with run_lock(directory):
        status = json.loads((directory / 'meta/status.json').read_text())
        identity = json.loads((directory / 'config.json').read_text())
        errors = [json.loads(line) for line in (directory / 'logs/errors.jsonl').read_text().splitlines()]
        if ((directory / 'complete.json').exists() or status.get('state') != 'failed'
                or status.get('completed') != 0
                or status.get('error') != 'explicit padding token required'
                or identity.get('prepared_sha256') != prepared_sha256
                or not errors or errors[-1].get('stage') != 'capture-reference'
                or errors[-1].get('error') != 'explicit padding token required'
                or any(directory.rglob('*.npz'))):
            raise ValueError('recovery only accepts the matching zero-progress missing-padding failure')
        files = {str(path.relative_to(directory)): file_hash(path)
                 for path in directory.rglob('*') if path.is_file()}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = directory.parent / 'failed-attempts' / (directory.name + '-' + stamp)
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        if any(file_hash(archive / name) != digest for name, digest in files.items()):
            raise ValueError(f'archived evidence failed verification; inspect {archive}')
        (archive / 'recovery.json').write_text(json.dumps({
            'reason': 'missing-model-padding-id', 'original_path': str(directory),
            'archived_at': stamp, 'original_identity': identity, 'files': files,
        }, indent=2) + '\n')
        return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    from pilot_eval.activation_prepare import load_prepared
    root = args.output_root.resolve()
    prepared, _ = load_prepared(args.config, root)
    prepared_hash = _hash(prepared)
    run = (root / prepared['run_path']).resolve()
    if not run.is_relative_to(root):
        raise ValueError('prepared run is outside artifact root')
    if (run / 'profile/config.json').exists():
        raise ValueError('profiling has already started; inspect its identity before recovery')
    for path in (root / 'plans').glob('*/activation.execution.json'):
        if json.loads(path.read_text()).get('prepared_sha256') == prepared_hash:
            raise ValueError('an execution is already frozen for these inputs')
    found = False
    for directory in sorted((run / 'validation').glob('diagnostic-*')):
        archive = archive_failed_padding_validation(directory, prepared_hash)
        if archive:
            found = True
            print('Preserved failed validation:', archive)
    if not found:
        print('No old padding-failure attempt needs archiving.')
    print('Prepared inputs are unchanged. Retry notebook cell 7 under the new code pin.')


if __name__ == '__main__':
    main()
