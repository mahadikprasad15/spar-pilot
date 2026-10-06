"""Recovery preserves a failed zero-candidate profile and its prior diagnostics."""
import json
from pathlib import Path
import pytest
from test_activation_validation import prepared
from pilot_eval.activation_profile import profile_activation
from test_activation_profile import ProfileDependencies


def test_profile_recovery_preserves_failed_attempt_and_prepared_inputs(tmp_path):
    from scripts.recover_pilot3_profile import archive_failed_profile
    config = prepared(tmp_path)
    before = config.read_bytes()
    with pytest.raises(ValueError, match='planted missing hook'):
        profile_activation(config, tmp_path, dependencies=ProfileDependencies(failure=1))
    run = tmp_path / json.loads(config.read_text())['run_path']
    old = (run / 'profile/config.json').read_bytes()
    diagnostic = next((run / 'validation').glob('diagnostic-*/complete.json')).read_bytes()
    archive = archive_failed_profile(config, tmp_path)
    assert (archive / 'profile/config.json').read_bytes() == old
    assert next((archive / 'validation').glob('diagnostic-*/complete.json')).read_bytes() == diagnostic
    assert config.read_bytes() == before
    assert not (run / 'profile').exists()
    assert not (run / 'validation').exists()
    assert archive_failed_profile(config, tmp_path) is None
    result = profile_activation(config, tmp_path, dependencies=ProfileDependencies())
    assert result['measurements'][0]['status'] == 'passed'


@pytest.mark.parametrize('protected', ['candidate', 'execution', 'running'])
def test_profile_recovery_refuses_work_with_results_or_active_execution(tmp_path, protected):
    from scripts.recover_pilot3_profile import archive_failed_profile
    config = prepared(tmp_path)
    with pytest.raises(ValueError):
        profile_activation(config, tmp_path, dependencies=ProfileDependencies(failure=1))
    run = tmp_path / json.loads(config.read_text())['run_path']
    if protected == 'candidate':
        path = run / 'profile/batch-1/complete.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    elif protected == 'execution':
        from pilot_eval.workflow import _hash
        path = tmp_path / 'plans/frozen/activation.execution.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'prepared_sha256': _hash(json.loads(config.read_text()))}))
    else:
        (run / 'profile/meta/status.json').write_text(json.dumps({'state': 'running', 'completed': 0}))
    with pytest.raises(ValueError):
        archive_failed_profile(config, tmp_path)
    assert (run / 'profile/config.json').exists()
