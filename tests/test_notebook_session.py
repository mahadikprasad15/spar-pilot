"""Resume and process orchestration at the public notebook session boundary."""
import json
import sys
from pathlib import Path

import pytest


def test_restore_fresh_kernel_reads_saved_paths_and_reviews_without_launching(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    plan = 'experiment'
    training = tmp_path / 'plans' / (plan + '-arm-v1')
    training.mkdir(parents=True)
    (training / 'grpo.training.json').write_text(json.dumps({
        'run_path': 'runs/training', 'frozen_path': 'plans/frozen-v2/grpo.frozen.json'}))
    execution = tmp_path / 'plans' / (plan + '-grpo-write-execution-v1')
    execution.mkdir()
    (execution / 'activation.execution.json').write_text(json.dumps({
        'run_path': 'runs/writes', 'batch_size': 8,
        'review': {'notes': 'Reviewed original evidence'}, 'runtime': {'device': 'A100 80GB'}}))
    session = NotebookSession(tmp_path, Path.cwd(), plan_name=plan)
    state = session.restore()
    assert state['TRAINING_CONFIG_PATH'] == training / 'grpo.training.json'
    assert state['FROZEN_PATH'] == tmp_path / 'plans/frozen-v2/grpo.frozen.json'
    assert state['WRITE_EXECUTIONS']['grpo'] == execution / 'activation.execution.json'
    assert state['WRITE_BATCH_SIZE'] == 8
    assert state['WRITE_PROFILE_NOTES'] == 'Reviewed original evidence'
    assert not (tmp_path / 'notebook').exists()  # Restore never writes/relaunches experiments.


def test_gpu_guard_reports_missing_gpu_and_hardware_difference_before_launch(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    execution = tmp_path / 'execution.json'
    execution.write_text(json.dumps({'runtime': {'device': 'A100 80GB', 'versions': {}, 'git_commit': 'old'}}))
    session = NotebookSession(tmp_path, Path.cwd(), environment_probe=lambda: {
        'cuda_available': False, 'gpu_count': 0, 'versions': {}})
    with pytest.raises(RuntimeError, match='GPU runtime'):
        session.run([sys.executable, '-m', 'pilot_eval', 'activation-measure', '--config', str(execution)], check=True)
    assert not (tmp_path / 'notebook').exists()
    session.environment_probe = lambda: {
        'cuda_available': True, 'gpu_count': 1, 'versions': {},
        'activation_runtime': {'device': 'A100 40GB', 'versions': {}, 'git_commit': 'new'}}
    with pytest.raises(RuntimeError, match='(?s)device.*80GB.*40GB'):
        session.run([sys.executable, '-m', 'pilot_eval', 'activation-measure', '--config', str(execution)], check=True)
    assert not (tmp_path / 'notebook').exists()


def test_runner_streams_output_and_preserves_error_and_log(tmp_path, capsys):
    from pilot_eval.notebook_session import NotebookSession
    session = NotebookSession(tmp_path, Path.cwd())
    with pytest.raises(RuntimeError, match='underlying failure'):
        session.run([sys.executable, '-u', '-c', "print('underlying failure', flush=True); raise SystemExit(3)"], check=True)
    assert 'underlying failure' in capsys.readouterr().out
    logs = list((tmp_path / 'notebook').rglob('*.console.log'))
    assert len(logs) == 1
    assert 'underlying failure' in logs[0].read_text()


def test_restore_does_not_guess_among_multiple_compatible_profiles(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    prepared = tmp_path / 'plans/experiment-writes-v1/activation.prepared.json'
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({'run_path': 'runs/prepared'}))
    for name in ['one', 'two']:
        folder = tmp_path / 'runs/prepared' / ('profile-v2-' + name)
        folder.mkdir(parents=True)
        (folder / 'results.json').write_text(json.dumps({
            'identity': {'calibration_path': 'plans/experiment-grpo-write-calibration-v1/activation.calibration.json'},
            'profile_path': str(folder.relative_to(tmp_path))}))
    with pytest.raises(RuntimeError, match='Multiple profiles'):
        NotebookSession(tmp_path, Path.cwd(), plan_name='experiment').restore()


def test_clean_notebook_has_one_restore_setup_and_no_patch_cells():
    import ast
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/pilot-4-colab.ipynb').read_text())
    code = {c['metadata']['stage']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    assert 'NotebookSession' in code['setup'] and 'session.restore()' in code['setup']
    assert 'install_environment' in code['sampling-install']
    assert 'baseline-review' in code
    assert 'session.recover_writes' in code['writes-measure']
    assert 'WRITE_PRODUCTS' in code['setup'] or 'extra_write_products' in code['setup']
    assert all('subprocess.run' not in v for k,v in code.items() if k not in ['setup'])
    for source in code.values():
        ast.parse(source)
    assert all(not c.get('outputs') for c in notebook['cells'] if c['cell_type'] == 'code')
    assert all(len(''.join(c['source']).split()) < 180 for c in notebook['cells'] if c['cell_type'] == 'markdown')


from test_activation_measurement import execution, frozen_source


def test_completed_measurement_resume_verifies_on_cpu_without_environment_probe(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation
    from pilot_eval.notebook_session import NotebookSession
    from test_activation_measurement import MeasurementDependencies
    measure_activation(execution, tmp_path, dependencies=MeasurementDependencies())
    def no_gpu_probe():
        raise AssertionError('Completed measurement reuse must not probe/load a GPU')
    session = NotebookSession(tmp_path, Path(__file__).resolve().parents[1], environment_probe=no_gpu_probe)
    command = [sys.executable, '-m', 'pilot_eval', 'activation-measure',
               '--config', str(execution), '--output-root', str(tmp_path)]
    assert session.run(command, check=True).returncode == 0
    directory = tmp_path / json.loads(execution.read_text())['run_path']
    (directory / 'results/results.json').write_text('{}')
    with pytest.raises(RuntimeError):
        session.run(command, check=True)


def test_explicit_operational_selection_survives_kernel_restart(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    session = NotebookSession(tmp_path, Path.cwd(), plan_name='experiment', write_variant='a10040-v1')
    selection = dict(WRITE_VARIANT='a10040-v1', INCLUDE_SFT_WRITES=True,
                     SFT_PLAN='saved-sft', MEASUREMENT_PLAN='saved-tokens')
    session.save_selection(selection)
    assert NotebookSession.saved_selection(tmp_path, 'experiment') == selection
    assert not (tmp_path / 'plans').exists()  # Preferences do not amend scientific plans.


def test_runner_blocks_second_gpu_stage_before_any_environment_or_model_probe(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    def must_not_probe():
        raise AssertionError('A second GPU stage must not start')
    session = NotebookSession(tmp_path, Path.cwd(), environment_probe=must_not_probe,
        process_probe=lambda: [{'pid': 123, 'command': 'python -m pilot_eval activation-measure'}])
    with pytest.raises(RuntimeError, match='Active experiment child'):
        session.run([sys.executable, '-m', 'pilot_eval', 'activation-profile'], check=True)


def test_readiness_reports_old_hardware_before_user_runs_calibration(tmp_path):
    from pilot_eval.notebook_session import NotebookSession
    path = tmp_path / 'plans/experiment-grpo-write-calibration-v1/activation.calibration.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'run_path': 'runs/calibration', 'runtime': {'device': 'A100 80GB'}}))
    session = NotebookSession(tmp_path, Path.cwd(), plan_name='experiment')
    rows = session.readiness({'activation_runtime': {'device': 'A100 40GB'}})
    row = next(r for r in rows if r['product'] == 'grpo' and r['kind'] == 'calibration')
    assert row['differences'] == {'device': {'saved': 'A100 80GB', 'current': 'A100 40GB'}}
    assert path.read_text() == json.dumps({'run_path': 'runs/calibration', 'runtime': {'device': 'A100 80GB'}})


def test_completed_calibration_and_profile_are_verified_without_gpu_probe(tmp_path):
    from test_activation_validation import prepared
    from test_activation_calibration import CalibrationDeps
    from pilot_eval.activation_calibration import prepare_calibration, collect_calibration, freeze_rule, validate_rule
    from pilot_eval.activation_profile import profile_activation
    from pilot_eval.notebook_session import NotebookSession
    config = prepared(tmp_path)
    deps = CalibrationDeps()
    calibration = prepare_calibration(config, tmp_path, name='calibration', dependencies=deps)
    collect_calibration(calibration, tmp_path, dependencies=deps)
    freeze_rule(calibration, tmp_path, review_notes='Reviewed fixed fixture evidence.')
    validate_rule(calibration, tmp_path, dependencies=deps)
    profile_activation(config, tmp_path, calibration=calibration, dependencies=deps)
    def must_not_probe():
        raise AssertionError('Reading completed evidence must not probe GPU')
    session = NotebookSession(tmp_path, Path(__file__).resolve().parents[1], environment_probe=must_not_probe)
    for command in ['activation-calibrate', 'activation-calibration-validate']:
        assert session.run([sys.executable, '-m', 'pilot_eval', command,
            '--config', str(calibration), '--output-root', str(tmp_path)], check=True).returncode == 0
    assert session.run([sys.executable, '-m', 'pilot_eval', 'activation-profile',
        '--config', str(config), '--calibration', str(calibration),
        '--output-root', str(tmp_path)], check=True).returncode == 0
