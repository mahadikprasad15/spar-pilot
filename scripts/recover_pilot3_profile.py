"""Preserve a failed zero-candidate profile before a code/runtime update."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_eval.activation_prepare import load_prepared
from pilot_eval.training import run_lock, file_hash
from pilot_eval.workflow import _hash


def archive_failed_profile(config_path, output_root):
    root = Path(output_root).resolve()
    config, _ = load_prepared(config_path, root)
    run = (root / config['run_path']).resolve()
    if not run.is_relative_to(root):
        raise ValueError('prepared run is outside artifact root')
    profile = run / 'profile'
    if not profile.exists():
        return None
    for path in (root / 'plans').glob('*/activation.execution.json'):
        if json.loads(path.read_text()).get('prepared_sha256') == _hash(config):
            raise ValueError('execution already frozen; recovery refuses changing its evidence')
    with ExitStack() as locks:
        locks.enter_context(run_lock(profile))
        validation = run / 'validation'
        for diagnostic in sorted(validation.glob('diagnostic-*')):
            locks.enter_context(run_lock(diagnostic))
        status = json.loads((profile / 'meta/status.json').read_text())
        identity = json.loads((profile / 'config.json').read_text())
        if (status.get('state') != 'failed' or status.get('completed') != 0
                or identity.get('prepared_sha256') != _hash(config)
                or (profile / 'complete.json').exists()
                or list(profile.glob('batch-*/complete.json'))
                or list(profile.rglob('*.npz'))):
            raise ValueError('recovery requires matching failed profile with zero saved candidates')
        sources = [profile] + ([validation] if validation.exists() else [])
        files = {str(p.relative_to(run)): file_hash(p)
                 for directory in sources for p in directory.rglob('*') if p.is_file()}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = run / 'failed-attempts' / ('profile-' + stamp)
        archive.mkdir(parents=True)
        for directory in sources:
            directory.rename(archive / directory.name)
        if any(file_hash(archive / name) != digest for name, digest in files.items()):
            raise ValueError(f'preserved evidence hash mismatch; inspect {archive}')
        (archive / 'recovery.json').write_text(json.dumps({
            'reason': 'profiling-performance-code-update', 'prepared_sha256': _hash(config),
            'original_profile_identity': identity, 'files': files,
            'note': 'Previous diagnostics retained as evidence; rerun validation under new runtime.'
        }, indent=2) + '\n')
        return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    archive = archive_failed_profile(args.config, args.output_root)
    print('Preserved old attempt:', archive or 'already archived / no old profile')
    print('Prepared inputs/adapters unchanged. Rerun cell 7, then cell 8.')


if __name__ == '__main__':
    main()
