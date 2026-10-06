import json

import pytest


def failed_attempt(root, *, error='explicit padding token required', completed=0):
    directory = root / 'validation' / 'diagnostic-fixture'
    for name, value in {
        'config.json': {'prepared_sha256': 'pinned-inputs', 'runtime': {'git_commit': 'old'}},
        'meta/status.json': {'state': 'failed', 'completed': completed, 'error': error},
    }.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    log = directory / 'logs/errors.jsonl'
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({'stage': 'capture-reference', 'error': error}) + '\n')
    return directory


def test_padding_recovery_preserves_failed_evidence_and_does_not_touch_inputs(tmp_path):
    from scripts.recover_pilot3_padding import archive_failed_padding_validation
    directory = failed_attempt(tmp_path)
    original = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    inputs = tmp_path / 'inputs.json'
    inputs.write_text('frozen inputs')
    archive = archive_failed_padding_validation(directory, 'pinned-inputs')
    assert not directory.exists()
    assert all((archive / name).read_bytes() == content for name, content in original.items())
    assert inputs.read_text() == 'frozen inputs'
    assert json.loads((archive / 'recovery.json').read_text())['reason'] == 'missing-model-padding-id'
    assert archive_failed_padding_validation(directory, 'pinned-inputs') is None


@pytest.mark.parametrize('case', ['completed', 'partial', 'different-error', 'different-inputs', 'payload'])
def test_padding_recovery_refuses_other_or_successful_attempts(tmp_path, case):
    from scripts.recover_pilot3_padding import archive_failed_padding_validation
    directory = failed_attempt(tmp_path, error='rank-1 failed' if case == 'different-error' else
        'explicit padding token required', completed=1 if case == 'partial' else 0)
    if case == 'completed':
        (directory / 'complete.json').write_text('{}')
    if case == 'payload':
        (directory / 'summaries.npz').write_bytes(b'partial scientific payload')
    with pytest.raises(ValueError):
        archive_failed_padding_validation(directory, 'other-inputs' if case == 'different-inputs' else 'pinned-inputs')
    assert directory.exists()
    assert not (directory.parent / 'failed-attempts').exists()
