"""Copy a never-started misnamed qualification plan; retain all source evidence.

Run with the SAME checkout that prepared the source plan. This is a metadata
recovery, not integrity verification or permission to run a diagnostic. The
normal qualification-runtime/run commands still verify the payloads afterward.
"""
import argparse
import json
from pathlib import Path
import sys


def recover(output_root, name):
    from pilot_eval.sft import safe_name
    from pilot_eval.stage0_sources import inside
    from pilot_eval.stage0_budget import reservations
    from pilot_eval.stage0_qualification import _implementation
    from pilot_eval.stage0_workflow import _writer, _software, _state
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash, _save_frozen

    root = Path(output_root).resolve()
    safe_name(name)
    if name == 'tokenizer_config.json':
        raise ValueError('Select the intended experiment name.')
    source = root / 'plans/tokenizer_config.json/stage0.qualification.json'
    original = json.loads(source.read_text())
    if (original.get('protocol') != 'stage0-loader-qualification-v1'
            or original.get('name') != 'tokenizer_config.json'
            or original.get('run_path') != 'runs/stage0/tokenizer_config.json/qualification'):
        raise ValueError('Not the expected misnamed qualification plan.')
    if original['implementation'] != _implementation():
        raise ValueError('Use the same implementation that prepared the source plan.')
    old_dir = inside(root, original['run_path'])
    manifest = json.loads((old_dir / 'meta/manifest.json').read_text())
    state = json.loads((old_dir / 'meta/status.json').read_text())
    if manifest['config_sha256'] != _hash(original):
        raise ValueError('Source plan/manifest mismatch.')
    if state.get('state') != 'prepared' or state.get('completed') != 0:
        raise ValueError('Recovery applies only before diagnostic execution.')
    forbidden = ['writer.json', 'meta/runtime.json', 'meta/tokenized-inputs.json',
                 'complete.json', 'production-freeze.json']
    if (any((old_dir / item).exists() for item in forbidden)
            or any((old_dir / item).exists() for item in ['units', 'attempts', 'reviews'])
            or (source.parent / 'writer.json').exists()
            or any(row['state'] == 'active' for row in reservations(root).values())):
        raise ValueError('Execution/ownership/reservation evidence exists; do not rename.')

    # Only identity fields change. Do not rewrite source hashes, implementation
    # hashes, cohort, thresholds, variants, runtime policy or budget settings.
    corrected = {**original, 'name': name,
                 'run_path': f'runs/stage0/{name}/qualification'}
    target = root / 'plans' / name / 'stage0.qualification.json'
    directory = inside(root, corrected['run_path'])
    if target.exists() or directory.exists():
        raise ValueError('Destination already exists; inspect it rather than overwrite.')
    receipt = dict(source_path=str(source.relative_to(root)),
                   source_sha256=file_hash(source), source_manifest_sha256=file_hash(old_dir / 'meta/manifest.json'),
                   destination_config_sha256=_hash(corrected),
                   changed_fields=['name', 'run_path'], measurement_verified=False,
                   note='Metadata recovery only; normal payload verification remains required.')
    with _writer(target.parent):
        _save_frozen(target, corrected)
        _save_frozen(directory / 'meta/manifest.json',
                     dict(config_sha256=_hash(corrected), software=_software()))
        _save_frozen(directory / 'meta/name-recovery.json', receipt)
        _state(directory, 'prepared', 0, corrected['total_units'])
        if json.loads(target.read_text()) != corrected:
            raise OSError('Recovered plan readback failed.')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    print(recover(args.output_root, args.name))
