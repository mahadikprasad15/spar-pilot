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
