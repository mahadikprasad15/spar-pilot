"""Pilot 4 preparation through public commands; saved sources, no network/model."""

import json
from pathlib import Path

from pilot_eval.cli import main
from test_activation_prepare import ActivationData, completed_source


def sources(root):
    sft = completed_source(root)
    assert main(['activation-prepare', '--source-config', str(sft), '--name', 'fixed-inputs',
                 '--output-root', str(root)], dependencies=ActivationData()) == 0
    return sft, root / 'plans/fixed-inputs/activation.prepared.json'


def test_public_grpo_preparation_reuses_sources_and_reports_pending_choices(tmp_path, capsys):
    sft, measurement = sources(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    args = ['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measurement),
            '--name', 'grpo', '--output-root', str(tmp_path)]
    assert main(args) == 0
    path = tmp_path / 'plans/grpo/grpo.prepared.json'
    plan = json.loads(path.read_text())
    source = json.loads(sft.read_text())
    assert plan['dtype'] == 'float32'
    assert plan['optimizer']['learning_rate'] == 1e-4
    assert plan['optimizer']['max_grad_norm'] == 1.0
    assert plan['adapter']['rank'] == plan['adapter']['alpha'] == 1
    assert plan['training_indices'] == source['training_indices']
    assert plan['training_items_sha256'] == source['training_items_sha256']
    assert plan['evaluation_items_sha256'] == source['evaluation_items_sha256']
    assert plan['measurement_items_sha256'] == json.loads(measurement.read_text())['items_sha256']
    assert plan['pending']['reward_scorer']['proposed'] == 'gsm8k-flexible-v3'
    assert plan['pending']['gate_margin']['value'] is None
    audit = (tmp_path / plan['audit_path']).read_text()
    assert '512' in audit and '150' in audit and 'FP32' in audit
    assert 'historical' in audit.lower() and 'BF16' in audit
    assert all(p.read_bytes() == content for p, content in before.items())
    prepared = {p: p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    assert main(args) == 0
    assert all(p.read_bytes() == content for p, content in prepared.items())
    assert main(['grpo-audit', '--config', str(path), '--output-root', str(tmp_path)]) == 0
    assert 'reward_scorer' in capsys.readouterr().out
    assert main(['grpo-check-ready', '--config', str(path), '--output-root', str(tmp_path)]) == 1
