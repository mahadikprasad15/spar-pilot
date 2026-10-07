"""Pilot 4 preparation through public commands; saved sources, no network/model."""

import json
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.training import training_directory
from test_activation_prepare import ActivationData, completed_source


def sources(root):
    sft = completed_source(root)
    config = json.loads(sft.read_text())
    order_path = training_directory(root, config) / 'inputs/order.json'
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text(json.dumps([f'gsm8k:train:{i}' for i in config['training_indices']]))
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


def test_preparation_failure_records_source_error_without_complete_marker(tmp_path, capsys):
    sft, measurement = sources(tmp_path)
    measured = json.loads(measurement.read_text())
    checkpoint = tmp_path / measured['source_evidence']['checkpoints']['64']['path']
    (checkpoint / 'adapter_model.safetensors').write_text('corrupt')
    args = ['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measurement),
            '--name', 'bad-source', '--output-root', str(tmp_path)]
    assert main(args) == 1
    directory = tmp_path / 'plans/bad-source'
    assert not (directory / 'prepare-complete.json').exists()
    status = json.loads((directory / 'meta/status.json').read_text())
    assert status['state'] == 'failed'
    assert 'checkpoint' in status['error']
    assert 'checkpoint' in capsys.readouterr().err


def test_grpo_matches_actual_saved_prompt_order_and_detects_changed_order(tmp_path):
    sft, measurement = sources(tmp_path)
    config = json.loads(sft.read_text())
    order_path = training_directory(tmp_path, config) / 'inputs/order.json'
    saved_order = list(reversed(json.loads(order_path.read_text())))
    order_path.write_text(json.dumps(saved_order))
    args = ['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measurement),
            '--name', 'order', '--output-root', str(tmp_path)]
    assert main(args) == 0
    path = tmp_path / 'plans/order/grpo.prepared.json'
    plan = json.loads(path.read_text())
    assert plan['optimizer_prompt_order'] == saved_order
    assert plan['optimizer_prompt_order'] != [f'gsm8k:train:{i}' for i in config['training_indices']]
    protected = {p: p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    order_path.write_text(json.dumps(saved_order[1:] + saved_order[:1]))
    assert main(args) == 1
    assert all(p.read_bytes() == content for p, content in protected.items())


def test_prepared_source_contract_is_not_a_full_execution_authorization(tmp_path, capsys):
    """A preparation marker cannot authorize training by filling placeholders."""
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    sft, measurement = sources(tmp_path)
    args = ['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measurement),
            '--name', 'not-frozen', '--output-root', str(tmp_path)]
    assert main(args) == 0
    path = tmp_path / 'plans/not-frozen/grpo.prepared.json'
    plan = json.loads(path.read_text())
    for entry in plan['pending'].values():
        entry['value'] = 'filled-but-not-reviewed'
    path.write_text(json.dumps(plan))
    run_manifest = tmp_path / plan['run_path'] / 'meta/run_manifest.json'
    run_manifest.write_text(json.dumps(plan))
    marker_path = path.parent / 'prepare-complete.json'
    marker = json.loads(marker_path.read_text())
    marker['config_sha256'] = _hash(plan)
    marker['files'][str(path.relative_to(tmp_path))] = file_hash(path)
    marker['files'][str(run_manifest.relative_to(tmp_path))] = file_hash(run_manifest)
    marker_path.write_text(json.dumps(marker))
    assert main(['grpo-check-ready', '--config', str(path), '--output-root', str(tmp_path)]) == 1
    assert 'frozen' in capsys.readouterr().err


def test_actual_sft_full_projection_names_are_accepted_without_loosening_targets(tmp_path):
    """The SFT engine passes full module names to PEFT, not seven suffixes."""
    from pilot_eval.training import seal_checkpoint
    sft = completed_source(tmp_path)
    config = json.loads(sft.read_text())
    directory = training_directory(tmp_path, config)
    targets = [f'model.layers.{i}.{group}.{p}' for i in range(28)
               for group, names in [('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                                    ('mlp', ['gate_proj', 'up_proj', 'down_proj'])]
               for p in names]
    for step in [0, 8, 16, 32, 64]:
        checkpoint = directory / 'checkpoints' / f'checkpoint-{step}'
        adapter_path = checkpoint / 'adapter_config.json'
        adapter = json.loads(adapter_path.read_text())
        adapter['target_modules'] = targets
        adapter_path.write_text(json.dumps(adapter))
        seal_checkpoint(checkpoint, step)
    (directory / 'inputs').mkdir(exist_ok=True)
    (directory / 'inputs/order.json').write_text(json.dumps([f'gsm8k:train:{i}' for i in config['training_indices']]))
    assert main(['activation-prepare', '--source-config', str(sft), '--name', 'full-targets',
                 '--output-root', str(tmp_path)], dependencies=ActivationData()) == 0
    measured = tmp_path / 'plans/full-targets/activation.prepared.json'
    assert main(['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measured),
                 '--name', 'grpo-full', '--output-root', str(tmp_path)]) == 0
    checkpoint = directory / 'checkpoints/checkpoint-64'
    adapter_path = checkpoint / 'adapter_config.json'
    adapter = json.loads(adapter_path.read_text())
    adapter['target_modules'] = targets[:-1] + ['lm_head']
    adapter_path.write_text(json.dumps(adapter))
    seal_checkpoint(checkpoint, 64)
    assert main(['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measured),
                 '--name', 'grpo-wrong', '--output-root', str(tmp_path)]) == 1
