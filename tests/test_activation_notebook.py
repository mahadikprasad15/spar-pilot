"""Guided notebook's runnable stage boundary, using controlled saved inputs."""
import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / 'notebooks/pilot-3-colab.ipynb'


def cells():
    notebook = json.loads(NOTEBOOK.read_text())
    result = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source)
            assert cell['outputs'] == [] and cell['execution_count'] is None
            result[cell['metadata']['stage']] = source
    return notebook, result


def test_notebook_is_runnable_and_uses_shared_workflow():
    notebook, code = cells()
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'SPAR/spar-pilot/artifacts' in text
    assert '/content/spar-pilot2' not in text and 'SPAR/pilot1' not in text
    assert list(code) == ['setup', 'install', 'environment', 'helpers', 'prepare', 'audit',
                          'validate', 'calibrate', 'calibration-review', 'calibration-validate', 'profile', 'review', 'freeze', 'measure', 'verify', 'report', 'read', 'recovery']
    import re
    assert re.search(r"HARNESS_COMMIT = ['\"][0-9a-f]{40}['\"]", code['setup'])
    assert 'activation-report' in code['report']
    assert 'activation-measure' in code['measure']
    assert 'BENCHMARK_REVIEWED' in code['freeze']
    assert 'CPU_REPORT_ONLY' in code['install']
    assert 'resume' in text.lower() and 'not causal' in text.lower()

from test_activation_measurement import frozen_source, MeasurementDependencies


def test_notebook_stages_freeze_measure_verify_report_and_restore(tmp_path, frozen_source):
    import shutil
    from IPython.display import display, Markdown
    from pilot_eval.cli import main
    _, code = cells()
    root, relative = frozen_source
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    prepared_path = tmp_path / 'plans/instrument/activation.prepared.json'
    scope = dict(Path=Path, json=json, os=__import__('os'), subprocess=__import__('subprocess'),
                 sys=__import__('sys'), time=__import__('time'), ARTIFACT_ROOT=tmp_path,
                 PREPARED_PATH=prepared_path, EXECUTION_PATH=tmp_path / 'plans/notebook-run/activation.execution.json',
                 RUN_NAME='notebook-run', REPORT_NAME='notebook-report',
                 REPORT_DIR=tmp_path / 'reports/notebook-report', SESSION_DIR=tmp_path / 'notebook/notebook-run',
                 REPO_DIR=NOTEBOOK.parents[1], CPU_REPORT_ONLY=False, display=display, Markdown=Markdown)
    scope['SESSION_DIR'].mkdir(parents=True)
    exec(code['helpers'], scope)
    calls = []
    deps = MeasurementDependencies()

    def run(arguments, label, run_dir=None):
        args = list(map(str, arguments)) + ['--output-root', str(tmp_path)]
        calls.append(args[0])
        assert main(args, dependencies=deps) == 0

    scope['PROFILE_DIR'] = tmp_path / json.loads(prepared_path.read_text())['run_path'] / 'profile'
    scope['run_command'] = run
    scope['live_children'] = lambda: []
    exec(code['audit'], scope)
    exec(code['review'], scope)
    # Default controls must stop production until evidence is reviewed.
    with pytest.raises(AssertionError, match='Input review'):
        exec(code['freeze'], scope)
    assert 'activation-freeze' not in calls
    scope.update(INPUTS_REVIEWED=True, INPUT_REVIEW_NOTES='Checked fixture masks and complete solutions.',
                 BENCHMARK_REVIEWED=True, BENCHMARK_NOTES='Checked saved passing candidate evidence.')
    exec(code['freeze'], scope)
    exec(code['measure'], scope)
    exec(code['verify'], scope)
    exec(code['report'], scope)
    result = json.loads((scope['REPORT_DIR'] / 'results/results.json').read_text())
    assert result['report_complete']
    assert len(result['measurements']) == 6720
    # Reconnect restores immutable inputs/plan, without relying on prior variables.
    restored, prepared, rows = scope['restore_execution']()
    assert len(rows) == 300 and restored['run_id'] == 'notebook-run'
    assert restored['review']['notes'].startswith('Input review:')
    assert prepared['source_evidence']['checkpoints']['64']['adapter_sha256']
    assert calls == ['activation-audit', 'activation-freeze', 'activation-measure', 'activation-verify', 'activation-report']
    # CPU-only route skips measurement and still reuses verified results.
    scope['CPU_REPORT_ONLY'] = True
    with pytest.raises(AssertionError):
        exec(code['measure'], scope)
    exec(code['verify'], scope)
    exec(code['report'], scope)


def test_notebook_command_checks_process_exit_without_parsing_partial_status(tmp_path):
    _, code = cells()
    import os, subprocess, sys, time
    scope = dict(Path=Path, json=json, os=os, subprocess=subprocess, sys=sys, time=time,
                 REPO_DIR=NOTEBOOK.parents[1], ARTIFACT_ROOT=tmp_path, SESSION_DIR=tmp_path / 'notebook',
                 PREPARED_PATH=tmp_path / 'unused', EXECUTION_PATH=tmp_path / 'unused')
    scope['SESSION_DIR'].mkdir()
    exec(code['helpers'], scope)
    scope['boot_identity'] = lambda: 'controlled-boot'
    status_dir = tmp_path / 'run'
    (status_dir / 'meta').mkdir(parents=True)
    (status_dir / 'meta/status.json').write_text('{')
    scope['run_command'](['--help'], 'help', status_dir)
    record = json.loads((scope['SESSION_DIR'] / 'help.process.json').read_text())
    assert record['returncode'] == 0 and not record['running']
    scope['run_command'](['--help'], 'help', status_dir)
    # A live child must block another invocation, not be overwritten or killed.
    active = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        record.update(pid=active.pid, running=True, boot_id='controlled-boot')
        (scope['SESSION_DIR'] / 'help.process.json').write_text(json.dumps(record))
        with pytest.raises(RuntimeError, match='earlier child'):
            scope['run_command'](['--help'], 'another', status_dir)
        assert active.poll() is None
    finally:
        active.terminate()
        active.wait()
    with pytest.raises(RuntimeError, match='Command failed'):
        scope['run_command'](['not-a-command'], 'bad', status_dir)
    assert 'invalid choice' in (scope['SESSION_DIR'] / 'bad.console.log').read_text()
