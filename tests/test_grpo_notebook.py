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


def test_guided_training_and_verification_use_public_workflow(tmp_path):
    from test_grpo_training import frozen,Dependencies
    path=frozen(tmp_path);deps=Dependencies()
    notebook_path=Path(__file__).resolve().parents[1]/'notebooks/pilot-4-colab.ipynb'
    notebook=json.loads(notebook_path.read_text())
    stages={c['metadata']['stage']:''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'}
    assert {'train','verify-training'}<=stages.keys()
    class Commands:
        @staticmethod
        def run(args,*,cwd,check):
            from types import SimpleNamespace
            result=main(args[4:],dependencies=deps);assert result==0
            return SimpleNamespace(returncode=result)
    scope=dict(FROZEN_PATH=path,ARTIFACT_ROOT=tmp_path,PLAN_NAME='notebook',REPO_DIR=notebook_path.parents[1],
        subprocess=Commands,sys=sys,json=json,Path=Path)
    exec(stages['train'],scope);exec(stages['verify-training'],scope)
    assert scope['TRAINING_RESULTS']['accepted_completions']==4096


def test_guided_training_records_dead_child_instead_of_stale_running(tmp_path):
    import pytest
    from types import SimpleNamespace
    notebook_path=Path(__file__).resolve().parents[1]/'notebooks/pilot-4-colab.ipynb'
    stages={c['metadata']['stage']:''.join(c['source']) for c in json.loads(notebook_path.read_text())['cells'] if c['cell_type']=='code'}
    directory=tmp_path/'runs/controlled';(directory/'meta').mkdir(parents=True)
    plan=tmp_path/'plans/notebook-arm-v1';plan.mkdir(parents=True)
    (plan/'grpo.training.json').write_text(json.dumps(dict(run_path='runs/controlled')))
    (directory/'meta/status.json').write_text(json.dumps(dict(state='running',completed=11,total=64)))
    class Commands:
        @staticmethod
        def run(*args,**kwargs):return SimpleNamespace(returncode=-9)
    scope=dict(FROZEN_PATH=tmp_path/'unused',ARTIFACT_ROOT=tmp_path,PLAN_NAME='notebook',REPO_DIR=notebook_path.parents[1],
        subprocess=Commands,sys=sys,json=json,Path=Path)
    with pytest.raises(RuntimeError,match='exit code -9'):exec(stages['train'],scope)
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'


def test_guided_behavioural_evaluation_and_cpu_report_use_public_workflow(tmp_path):
    from test_grpo_evaluation import source, Dependencies
    notebook_path=Path(__file__).resolve().parents[1]/'notebooks/pilot-4-colab.ipynb'
    stages={c['metadata']['stage']:''.join(c['source']) for c in json.loads(notebook_path.read_text())['cells'] if c['cell_type']=='code'}
    assert {'evaluate-behaviour','report-behaviour'}<=stages.keys()
    deps=Dependencies();path=source(tmp_path)
    class Commands:
        @staticmethod
        def run(args,*,cwd,check):
            assert main(args[4:],dependencies=deps)==0
    scope=dict(TRAINING_CONFIG_PATH=path,ARTIFACT_ROOT=tmp_path,PLAN_NAME='notebook',REPO_DIR=notebook_path.parents[1],
        subprocess=Commands,sys=sys,json=json,Path=Path,display=lambda value:None,Markdown=str)
    exec(stages['evaluate-behaviour'],scope)
    exec(stages['report-behaviour'],scope)
    assert scope['BEHAVIOUR_RESULTS']['drops']['greedy']['drop']==1.
    assert (scope['BEHAVIOUR_REPORT_DIR']/'complete.json').exists()


def test_ticket7_notebook_exposes_separate_forward_work_and_verified_gpu_release():
    notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/pilot-4-colab.ipynb').read_text())
    stages={c['metadata']['stage']:''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'}
    expected=['writes-prepare','writes-calibration-prepare','writes-calibrate','writes-calibration-freeze',
              'writes-calibration-validate','writes-profile','writes-freeze','writes-measure','writes-verify',
              'writes-report','writes-compare']
    assert all(s in stages for s in expected)
    assert [s for s in stages if s in expected]==expected
    assert 'grpo-writes-control' in stages['writes-prepare']
    assert 'activation-verify' in stages['writes-verify']
    assert 'grpo-writes-report' in stages['writes-compare']
    assert 'WRITE_PROFILE_REVIEWED' in stages['writes-freeze']
    assert 'SFT_WRITE_REPORT' in stages['writes-compare']
    for source in stages.values():ast.parse(source)
    text='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='markdown')
    assert 'forward-only' in text and 'one random' in text.lower()


def test_notebook_pin_contains_every_ticket7_entrypoint():
    import re,subprocess
    repository=Path(__file__).resolve().parents[1]
    notebook=json.loads((repository/'notebooks/pilot-4-colab.ipynb').read_text())
    setup=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and c['metadata']['stage']=='setup')
    commit=re.search(r"HARNESS_COMMIT = '([0-9a-f]{40})'",setup).group(1)
    cli=subprocess.check_output(['git','show',commit+':pilot_eval/cli.py'],cwd=repository,text=True)
    for command in ['grpo-writes-prepare','grpo-writes-control','grpo-writes-report',
                    'tokens-prepare','tokens-profile','tokens-freeze','tokens-measure','tokens-verify','tokens-report']:
        assert command in cli


def test_guided_ticket8_stages_have_profile_review_and_verified_release_boundary():
    notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/pilot-4-colab.ipynb').read_text())
    stages={c['metadata']['stage']:''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'}
    needed=['tokens-prepare','tokens-profile','tokens-freeze','tokens-measure','tokens-verify','tokens-report']
    assert all(stage in stages for stage in needed)
    ordered=list(stages)
    assert [ordered.index(s) for s in needed]==sorted(ordered.index(s) for s in needed)
    for stage in needed:
        ast.parse(stages[stage])
        assert "'"+stage+"'" in stages[stage]
        assert 'check=True' in stages[stage]
    assert 'TOKEN_PROFILE_REVIEWED' in stages['tokens-freeze']
    assert 'tokens.execution.json' in stages['tokens-freeze']
    assert 'GPU' in stages['tokens-verify']
    assert 'SFT_TOKEN_EXECUTION' in stages['tokens-prepare']
    assert '--context-chunk' in stages['tokens-prepare'] and '--workspace-mib' in stages['tokens-prepare']
