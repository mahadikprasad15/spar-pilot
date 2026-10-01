import ast
import json
from pathlib import Path

from test_sft_prepare import TrainingData, source_plan


def test_notebook_restores_frozen_state_after_reconnect(tmp_path):
    from pilot_eval.sft import prepare_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'sft', dependencies=TrainingData())
    notebook_path = Path(__file__).parents[1] / 'output/jupyter-notebook/pilot-2-colab.ipynb'
    notebook = json.loads(notebook_path.read_text())
    codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in codes:
        ast.parse(source)
    source = next(source for source in codes if 'def restore_state(' in source)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    scope = dict(json=json, Path=Path, ARTIFACT_ROOT=tmp_path, PLAN_NAME='sft')
    exec(compile(tree, 'notebook-restore', 'exec'), scope)
    saved_path, config, rows, items = scope['restore_state']()
    assert saved_path == path
    assert len(rows) == 512 and len(items) == 150
    assert config['optimizer']['learning_rate'] == 1e-4
    assert 'sft-train' in '\n'.join(codes)
    assert 'sft-evaluate' in '\n'.join(codes)
    assert 'sft-compare' in '\n'.join(codes)
