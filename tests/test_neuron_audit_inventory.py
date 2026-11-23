"""Ticket 01 (joint neuron/adapter audit): read-only source inventory on CPU fixtures."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_eval import neuron_audit_inventory as inv

PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
DIMS = dict(q_proj=(8, 8), k_proj=(8, 4), v_proj=(8, 4), o_proj=(8, 8),
            gate_proj=(8, 12), up_proj=(8, 12), down_proj=(12, 8))  # (in, out)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for key, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype='<f4').tobytes()
        header[key] = dict(dtype='F32', shape=list(array.shape), data_offsets=[offset, offset + len(raw)])
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    Path(path).write_bytes(len(encoded).to_bytes(8, 'little') + encoded + b''.join(blobs))


def adapter_tensors(layers=2, seed=0, zero_b=False, skip=None, extra=None):
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(layers):
        for p in PROJ:
            group = 'self_attn' if p in PROJ[:4] else 'mlp'
            base = f'base_model.model.model.layers.{layer}.{group}.{p}'
            fan_in, fan_out = DIMS[p]
            out[base + '.lora_A.weight'] = rng.normal(size=(1, fan_in))
            out[base + '.lora_B.weight'] = np.zeros((fan_out, 1)) if zero_b else rng.normal(size=(fan_out, 1))
    if skip:
        out.pop(skip)
    out.update(extra or {})
    return out


def write_checkpoint(directory, *, targets=None, seed=0, zero_b=False, seal=True, config=True, **kw):
    directory.mkdir(parents=True)
    write_safetensors(directory / 'adapter_model.safetensors', adapter_tensors(seed=seed, zero_b=zero_b, **kw))
    if config:
        cfg = dict(r=1, lora_alpha=1, lora_dropout=0.0, bias='none', use_rslora=False, use_dora=False,
                   rank_pattern={}, alpha_pattern={}, peft_type='LORA', peft_version='0.18.0',
                   base_model_name_or_path='tiny/qwen', target_modules=targets or list(PROJ))
        (directory / 'adapter_config.json').write_text(json.dumps(cfg, indent=2))
    if seal:
        files = {n: sha(directory / n) for n in ['adapter_model.safetensors', 'adapter_config.json']
                 if (directory / n).exists()}
        files['optimizer.pt'] = '0' * 64  # Present in the seal but not copied locally.
        (directory / 'complete.json').write_text(json.dumps(dict(step=0, files=files)))
    return directory


@pytest.fixture
def root(tmp_path):
    r = tmp_path / 'artifacts'
    write_checkpoint(r / 'runs/gold/checkpoint-0', seed=0, zero_b=True)
    write_checkpoint(r / 'runs/gold/checkpoint-64', seed=1, targets=list(reversed(PROJ)))
    (r / 'plans').mkdir(parents=True)
    (r / 'plans/gold.training.json').write_text(json.dumps(dict(ids=['gsm8k:train:1'])))
    return r


def sources(root, arms=None, **extra):
    gold = dict(arm='gold-sft', label='Gold SFT', role='trained',
                checkpoints={'0': dict(path='runs/gold/checkpoint-0', role='initialization'),
                             '64': dict(path='runs/gold/checkpoint-64', role='audited')},
                lineage=[dict(role='training', path='plans/gold.training.json'),
                         dict(role='rollout', path='plans/missing.rollouts.jsonl')])
    value = dict(schema=inv.SOURCES_SCHEMA, audit_step=64, init_step=0,
                 base=dict(model='tiny/qwen', model_revision='a' * 40, tokenizer_revision='a' * 40,
                           snapshot_dir=None, expected_file_sha256={'model.safetensors': 'b' * 64}),
                 arms=arms if arms is not None else [gold], **extra)
    path = root / 'plans/inventory.sources.json'
    path.write_text(json.dumps(value))
    return path


def build(root, tmp_path, **kw):
    return inv.build(sources(root, **kw), root, tmp_path / 'out', 'fixture-v1')


def test_verified_arm_records_hashes_keys_scale_and_semantic_config(root, tmp_path):
    result = build(root, tmp_path)
    arm = result['arms']['gold-sft']
    audited = arm['checkpoints']['64']
    assert audited['status'] == 'verified', audited['reasons']
    assert arm['adapted_measurement'] == 'load-ready'
    files = audited['files']
    assert files['adapter_model.safetensors']['sha256'] == sha(root / 'runs/gold/checkpoint-64/adapter_model.safetensors')
    assert files['adapter_model.safetensors']['size'] > 0
    # File and parameter hashes are distinct, labelled algorithms.
    weights = audited['weights']
    assert weights['parameter_hash_algorithm'] == inv.PARAMETER_HASH_ALGORITHM
    assert weights['parameter_sha256'] != files['adapter_model.safetensors']['sha256']
    assert weights['layers'] == [0, 1] and weights['rank'] == 1 and weights['module_count'] == 14
    assert weights['dims']['down_proj'] == dict(in_features=12, out_features=8)
    assert audited['config']['scale'] == 1.0
    # Reordered target_modules: different file hash, same semantic identity.
    init = arm['checkpoints']['0']
    assert init['files']['adapter_config.json']['sha256'] != files['adapter_config.json']['sha256']
    assert init['config']['semantic_sha256'] == audited['config']['semantic_sha256']
    assert arm['consistency']['same_semantic_config'] is True
    assert arm['consistency']['same_key_list'] is True
    # Seal entries for files not copied locally are reported, not failures.
    assert audited['seal']['absent_locally'] == ['optimizer.pt']
    keys = json.loads((tmp_path / 'out' / audited['key_list_path']).read_text())
    assert len(keys['keys']) == 28 and keys['modules'][0]['projection'] == 'q_proj'


def test_lineage_presence_and_unknowns_are_explicit(root, tmp_path):
    lineage = build(root, tmp_path)['arms']['gold-sft']['lineage']
    assert lineage[0]['present'] is True and lineage[0]['sha256'] == sha(root / 'plans/gold.training.json')
    assert lineage[1] == dict(role='rollout', path='plans/missing.rollouts.jsonl', present=False,
                              sha256=None, size=None)


def test_zero_b_initialization_is_classified_not_hidden(root, tmp_path):
    arm = build(root, tmp_path)['arms']['gold-sft']
    assert arm['checkpoints']['0']['weights']['zero_b_modules'] == 14
    assert arm['checkpoints']['64']['weights']['zero_b_modules'] == 0


def test_zero_b_at_audit_step_is_unresolved(root, tmp_path):
    write_checkpoint(root / 'runs/zero/checkpoint-64', zero_b=True)
    arm = dict(arm='zero', label='Zero', role='trained',
               checkpoints={'64': dict(path='runs/zero/checkpoint-64', role='audited')}, lineage=[])
    result = build(root, tmp_path, arms=[arm])['arms']['zero']
    assert result['checkpoints']['64']['status'] == 'unresolved'
    assert any('zero lora_B' in r for r in result['checkpoints']['64']['reasons'])
    assert result['adapted_measurement'] == 'unresolved'


def test_missing_config_marks_arm_unresolved_with_restore_requirement(root, tmp_path):
    write_checkpoint(root / 'runs/off/checkpoint-64', config=False, seal=False)
    arm = dict(arm='offline-grpo', label='Offline GRPO', role='trained',
               checkpoints={'64': dict(path='runs/off/checkpoint-64', role='audited',
                                       expected=dict(config_sha256='c' * 64))},
               lineage=[], restore=dict(hf_repo='owner/repo', revision='d' * 40))
    result = build(root, tmp_path, arms=[arm])
    off = result['arms']['offline-grpo']
    assert off['adapted_measurement'] == 'unresolved'
    assert off['checkpoints']['64']['config'] is None
    assert any('adapter_config.json absent' in r for r in off['checkpoints']['64']['reasons'])
    need = result['restore_requirements']
    assert {'arm': 'offline-grpo', 'step': '64', 'file': 'adapter_config.json',
            'expected_sha256': 'c' * 64, 'restore': dict(hf_repo='owner/repo', revision='d' * 40)} in need
    assert result['seven_arm_complete'] is False


def test_changed_weights_against_seal_and_expected_hash_fail(root, tmp_path):
    path = root / 'runs/gold/checkpoint-64/adapter_model.safetensors'
    write_safetensors(path, adapter_tensors(seed=9))
    gold = dict(arm='gold-sft', label='Gold', role='trained',
                checkpoints={'64': dict(path='runs/gold/checkpoint-64', role='audited',
                                        expected=dict(adapter_sha256='e' * 64))}, lineage=[])
    audited = build(root, tmp_path, arms=[gold])['arms']['gold-sft']['checkpoints']['64']
    assert audited['status'] == 'unresolved'
    assert audited['seal']['mismatched'] == ['adapter_model.safetensors']
    assert audited['expected']['adapter_sha256'] == dict(expected='e' * 64, actual=sha(path), match=False)


def test_missing_and_unexpected_keys_and_unsupported_config(root, tmp_path):
    write_checkpoint(root / 'runs/bad/checkpoint-64',
                     skip='base_model.model.model.layers.1.mlp.down_proj.lora_B.weight',
                     extra={'base_model.model.lm_head.weight': np.ones((2, 2))})
    cfg_path = root / 'runs/bad/checkpoint-64/adapter_config.json'
    cfg = json.loads(cfg_path.read_text()); cfg['use_rslora'] = True
    cfg_path.write_text(json.dumps(cfg))
    arm = dict(arm='bad', label='Bad', role='trained',
               checkpoints={'64': dict(path='runs/bad/checkpoint-64', role='audited')}, lineage=[])
    audited = build(root, tmp_path, arms=[arm])['arms']['bad']['checkpoints']['64']
    assert audited['status'] == 'unresolved'
    assert audited['weights']['unexpected_keys'] == ['base_model.model.lm_head.weight']
    assert audited['weights']['incomplete_modules'] == ['base_model.model.model.layers.1.mlp.down_proj']
    assert audited['config']['scale'] is None
    assert 'use_rslora' in audited['config']['unsupported']


def test_targets_not_in_config_are_reported(root, tmp_path):
    write_checkpoint(root / 'runs/narrow/checkpoint-64', targets=['q_proj'])
    arm = dict(arm='narrow', label='Narrow', role='trained',
               checkpoints={'64': dict(path='runs/narrow/checkpoint-64', role='audited')}, lineage=[])
    audited = build(root, tmp_path, arms=[arm])['arms']['narrow']['checkpoints']['64']
    assert audited['status'] == 'unresolved'
    assert 'down_proj' in audited['coverage']['weight_projections_not_in_config']


def test_untuned_baseline_and_missing_arm_are_explicit(root, tmp_path):
    arms = [dict(arm='untuned', label='Untuned', role='baseline', checkpoints={}, lineage=[]),
            dict(arm='online-grpo', label='Online GRPO', role='trained',
                 checkpoints={'64': dict(path='runs/nowhere/checkpoint-64', role='audited')}, lineage=[])]
    result = build(root, tmp_path, arms=arms)
    assert result['arms']['untuned']['adapted_measurement'] == 'baseline-no-adapter'
    online = result['arms']['online-grpo']
    assert online['checkpoints']['64']['status'] == 'missing'
    assert online['adapted_measurement'] == 'missing'
    assert result['summary'] == dict(arms=2, load_ready=0, baseline=1, unresolved=0, missing=1)


def test_base_snapshot_not_local_is_recorded_not_assumed(root, tmp_path):
    base = build(root, tmp_path)['base']
    assert base['local'] is False and base['status'] == 'not-local'
    assert base['expected_file_sha256'] == {'model.safetensors': 'b' * 64}


def test_base_snapshot_hash_mismatch_detected(root, tmp_path):
    snap = root / 'snap'; snap.mkdir()
    (snap / 'model.safetensors').write_bytes(b'xyz')
    path = sources(root)
    value = json.loads(path.read_text()); value['base']['snapshot_dir'] = 'snap'
    path.write_text(json.dumps(value))
    base = inv.build(path, root, tmp_path / 'out', 'fixture-v1')['base']
    assert base['status'] == 'unresolved'
    assert base['files']['model.safetensors']['match'] is False


def test_sealed_output_reused_verified_and_changed_source_refused(root, tmp_path):
    path = sources(root)
    first = inv.build(path, root, tmp_path / 'out', 'fixture-v1')
    run = tmp_path / 'out' / inv.RUN_SUBDIR / 'fixture-v1'
    seal = json.loads((run / 'complete.json').read_text())
    assert seal['inventory_sha256'] == sha(run / 'inventory.json')
    assert json.loads((run / 'meta/status.json').read_text())['state'] == 'complete'
    assert inv.build(path, root, tmp_path / 'out', 'fixture-v1') == first  # resume reuses
    assert inv.verify(run, root)['ok'] is True
    write_safetensors(root / 'runs/gold/checkpoint-64/adapter_model.safetensors', adapter_tensors(seed=5))
    assert inv.verify(run, root)['ok'] is False
    with pytest.raises(ValueError, match='changed'):
        inv.build(path, root, tmp_path / 'out', 'fixture-v1')


def test_corrupt_sealed_inventory_is_detected(root, tmp_path):
    inv.build(sources(root), root, tmp_path / 'out', 'fixture-v1')
    run = tmp_path / 'out' / inv.RUN_SUBDIR / 'fixture-v1'
    (run / 'inventory.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        inv.build(sources(root), root, tmp_path / 'out', 'fixture-v1')


def test_paths_outside_root_are_rejected(root, tmp_path):
    arm = dict(arm='escape', label='Escape', role='trained',
               checkpoints={'64': dict(path='../elsewhere', role='audited')}, lineage=[])
    with pytest.raises(ValueError, match='outside'):
        build(root, tmp_path, arms=[arm])
