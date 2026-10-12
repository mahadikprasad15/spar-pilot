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
