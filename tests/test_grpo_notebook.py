"""Guided notebook preparation runs the same public workflow as the CLI."""

import ast
import json
import sys
from pathlib import Path

from pilot_eval.cli import main
from test_grpo_prepare import sources


def test_guided_notebook_preparation_and_audit_use_public_commands(tmp_path):
    sft, measurement = sources(tmp_path)
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/pilot-4-colab.ipynb'
    notebook = json.loads(notebook_path.read_text())
    stages = {cell['metadata']['stage']: ''.join(cell['source']) for cell in notebook['cells']
              if cell['cell_type'] == 'code'}
    assert 'prepare' in stages and 'audit' in stages
    for source in stages.values():
        ast.parse(source)

    class Commands:
        @staticmethod
        def run(args, *, cwd, check):
            assert args[:3] == [sys.executable, '-m', 'pilot_eval']
            assert main(args[3:]) == 0

    scope = {'REPO_DIR': notebook_path.parents[1], 'ARTIFACT_ROOT': tmp_path,
             'SFT_CONFIG': sft, 'MEASUREMENT_CONFIG': measurement, 'PLAN_NAME': 'notebook',
             'subprocess': Commands, 'sys': sys, 'json': json, 'Path': Path}
    exec(stages['prepare'], scope)
    exec(stages['audit'], scope)
    path = tmp_path / 'plans/notebook/grpo.prepared.json'
    assert scope['PREPARED_PATH'] == path
    assert scope['PLAN']['state'] == 'prepared-with-pending-choices'
    assert scope['PLAN']['dtype'] == 'float32'
    assert scope['PLAN']['pending']['gate_margin']['value'] is None
    assert all(not cell.get('outputs') for cell in notebook['cells'] if cell['cell_type'] == 'code')


def test_guided_sampling_cell_runs_the_public_baseline(tmp_path):
    from test_grpo_baseline import prepared, Boundary
    path, settings = prepared(tmp_path)
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/pilot-4-colab.ipynb'
    notebook = json.loads(notebook_path.read_text())
    stages = {c['metadata']['stage']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    assert 'baseline' in stages
    deps = Boundary()
    class Commands:
        @staticmethod
        def run(args, *, cwd, check):
            assert main(args[3:], dependencies=deps) == 0
    scope = dict(SAMPLING_SETTINGS_PATH=settings, PREPARED_PATH=path, ARTIFACT_ROOT=tmp_path,
                 BASELINE_NAME='notebook-sampling', REPO_DIR=notebook_path.parents[1],
                 subprocess=Commands, sys=sys, json=json, Path=Path)
    exec(stages['baseline'], scope)
    assert scope['BASELINE_SUMMARY']['total'] == 1024
    assert scope['BASELINE_SUMMARY']['final_cap'] is None


def test_guided_controls_run_real_public_command(tmp_path):
    from test_grpo_algorithm import torch_stack
    torch = torch_stack()
    torch.set_num_threads(1)
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/pilot-4-colab.ipynb'
    notebook = json.loads(notebook_path.read_text())
    stages = {c['metadata']['stage']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    assert 'controls' in stages
    class Commands:
        @staticmethod
        def run(args, *, cwd, check):
            assert main(args[3:]) == 0
    scope = dict(ARTIFACT_ROOT=tmp_path, REPO_DIR=notebook_path.parents[1],
                 subprocess=Commands, sys=sys, json=json, Path=Path)
    exec(stages['controls'], scope)
    assert scope['CONTROL_RESULTS']['passed']
    assert scope['CONTROL_RESULTS']['algorithm_checks']['passed']
    assert scope['CONTROL_RESULTS']['scientific_training_steps'] == 0
    assert (scope['CONTROL_DIR'] / 'complete.json').exists()
    exec(stages['controls'], scope)
    (scope['CONTROL_DIR'] / 'algorithm/results.json').write_text('{}')
    assert main(['grpo-controls', '--name', scope['CONTROL_NAME'], '--output-root', str(tmp_path)]) == 1


def test_guided_preflight_and_freeze_run_public_commands(tmp_path):
    from test_grpo_preflight import fixture, Dependencies
    path, baseline, settings = fixture(tmp_path)
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/pilot-4-colab.ipynb'
    notebook = json.loads(notebook_path.read_text())
    stages = {c['metadata']['stage']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'}
    assert {'preflight', 'freeze'} <= stages.keys()
    deps=Dependencies()
    class Commands:
        @staticmethod
        def run(args, *, cwd, check):
            assert main(args[3:], dependencies=deps)==0
    review=tmp_path/'review.json'
    review.write_text(json.dumps(dict(reviewed=True, notes='Checked timings and memory', margin=.1,
        preregistration=dict(author='researcher',written_at='2026-10-08T00:00:00+00:00',prediction='No drop',falsifier='Drop exceeds margin'),
        monitors={k:dict(threshold=t,action='pause') for k,t in [('cap_fraction',.02),('dead_group_fraction',.8),('length_ratio_change_8_steps',.5)]})))
    scope=dict(PREPARED_PATH=path,BASELINE_CONFIG_PATH=baseline,PREFLIGHT_SETTINGS_PATH=settings,
        CONTROL_NAME='controls',PREFLIGHT_NAME='notebook-flight',FROZEN_NAME='notebook-frozen',REVIEW_PATH=review,
        ARTIFACT_ROOT=tmp_path,REPO_DIR=notebook_path.parents[1],subprocess=Commands,sys=sys,json=json,Path=Path)
    exec(stages['preflight'],scope)
    assert scope['PREFLIGHT_RESULTS']['passed']
    exec(stages['freeze'],scope)
    assert scope['FROZEN']['state']=='frozen'
