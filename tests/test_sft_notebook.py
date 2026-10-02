import ast
import json
from pathlib import Path

from test_sft_prepare import TrainingData, source_plan


def test_notebook_restores_frozen_state_after_reconnect(tmp_path):
    from pilot_eval.sft import prepare_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'sft', dependencies=TrainingData())
    notebook_path = Path(__file__).parents[1] / 'notebooks/pilot-2-colab.ipynb'
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


def test_notebook_selects_l4_and_benchmarks_before_baseline():
    notebook = json.loads((Path(__file__).parents[1] / 'notebooks/pilot-2-colab.ipynb').read_text())
    codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    setup = next(s for s in codes if 'REPO_URL =' in s)
    assert "HARDWARE = 'L4'" in setup
    assert "PLAN_NAME = 'pilot2-sft-fp32-l4-batch2-v1'" in setup
    assert 'EVALUATION_BATCH_SIZE = 2' in setup
    prepare = next(s for s in codes if "['sft-prepare'" in s)
    assert "'--hardware', HARDWARE" in prepare
    benchmark = next(i for i, s in enumerate(codes) if "'scripts/profile_inference.py'" in s)
    baseline = next(i for i, s in enumerate(codes) if "'--checkpoint', 'baseline'" in s)
    assert benchmark < baseline
    assert 'BENCHMARK_REVIEWED' in codes[baseline]
    for code in codes:
        ast.parse(code)
