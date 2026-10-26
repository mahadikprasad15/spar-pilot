"""Stage 0 through the notebook's public command boundary; no model downloads."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import _hash


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tensor_file(path, arrays):
    """Independent small safetensors writer, with no optional model package."""
    header, chunks, offset = {}, [], 0
    for key, value in arrays.items():
        raw = value.astype('<f4').tobytes()
        header[key] = dict(dtype='F32', shape=list(value.shape), data_offsets=[offset, offset+len(raw)])
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, 'little') + encoded + b''.join(chunks))


def sources(root, *, changed_a=False):
    selected = {}
    projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    for arm in ['sft', 'grpo']:
        checkpoints = {}
        for step in [0, 64]:
            directory = root / 'inputs' / arm / str(step)
            directory.mkdir(parents=True)
            write_json(directory/'adapter_config.json', dict(r=1, lora_alpha=1, lora_dropout=0,
                                                            target_modules=projections))
            arrays = {}
            for layer in range(28):
                for projection in projections:
                    kind = 'self_attn' if projection in projections[:4] else 'mlp'
                    stem = f'base_model.model.model.layers.{layer}.{kind}.{projection}'
                    arrays[stem+'.lora_A.weight'] = np.array([[1., 2. + (changed_a and arm == 'grpo')]], dtype=np.float32)
                    arrays[stem+'.lora_B.weight'] = np.full((3, 1), step/64, dtype=np.float32)
            tensor_file(directory/'adapter_model.safetensors', arrays)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            write_json(directory/'complete.json', dict(step=step, files=hashes))
            checkpoints[str(step)] = dict(path=str(directory.relative_to(root)),
                complete_sha256=hashlib.sha256((directory/'complete.json').read_bytes()).hexdigest(),
                adapter_sha256=hashes['adapter_model.safetensors'])
        config = dict(protocol_version='pilot3-write-v1' if arm == 'sft' else 'pilot4-writes-v1',
            model='Qwen/Qwen2.5-1.5B-Instruct', model_revision='pinned', tokenizer_revision='pinned',
            dtype='float32', source_contract={'kind': 'grpo'} if arm == 'grpo' else {},
            source_evidence={'base_sha256': 'same-base', 'checkpoints': checkpoints},
            checkpoint_steps=[0,64])
        path = root/'inputs'/arm/'activation.prepared.json'
        write_json(path, config)
        write_json(path.parent/'prepare-complete.json', dict(config_sha256=_hash(config),
            files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}))
        selected[arm] = str(path.relative_to(root))
    selection = root/'selection.json'
    write_json(selection, dict(sources=selected, products={'kl': 'missing/complete.json'}))
    return selection


def command(root, operation, *args):
    return main(['stage0-'+operation, '--output-root', str(root), *map(str, args)])


def prepared(root, **kwargs):
    selection = sources(root, **kwargs)
    assert command(root, 'prepare', '--selection', selection, '--name', 'example') == 0
    return root/'plans/example/stage0.plan.json'


def test_public_audit_restores_real_factor_evidence_without_claiming_gpu_completion(tmp_path):
    plan = prepared(tmp_path)
    assert command(tmp_path, 'audit', '--config', plan) == 0
    run = tmp_path/'runs/stage0/example'
    result = json.loads((run/'results/source-audit.json').read_text())
    assert result['primary_sources_ready'] is True
    assert result['initialization']['equal_a'] is True
    assert result['initialization']['modules_compared'] == 196
    assert result['sources']['sft']['checkpoints']['0']['b_exact_zero'] is True
    assert result['sources']['grpo']['checkpoints']['64']['modules'] == 196
    assert result['products']['kl']['state'] == 'missing'
    assert result['gpu_qualified'] is False
    assert command(tmp_path, 'status', '--config', plan) == 0
    script = '''
import json,sys
from pilot_eval.stage0_workflow import status
s=status(sys.argv[1],sys.argv[2])
assert s['audit']['primary_sources_ready']
assert s['budget']['ceiling_units']==20
assert s['budget']['spent_units'] is None
assert not any(x in sys.modules for x in ('torch','transformers','peft'))
'''
    subprocess.run([sys.executable, '-c', script, str(plan), str(tmp_path)], check=True)
    receipt = (run/'complete.json').read_bytes()
    assert command(tmp_path, 'audit', '--config', plan) == 0
    assert (run/'complete.json').read_bytes() == receipt


def test_audit_compares_actual_initialization_not_seed(tmp_path):
    plan = prepared(tmp_path, changed_a=True)
    assert command(tmp_path, 'audit', '--config', plan) == 0
    result = json.loads((tmp_path/'runs/stage0/example/results/source-audit.json').read_text())
    assert result['initialization']['equal_a'] is False
    assert len(result['initialization']['different_modules']) == 196


def test_interrupted_audit_reuses_sealed_units_and_rejects_changed_payload(tmp_path):
    plan = prepared(tmp_path)
    assert command(tmp_path, 'audit', '--config', plan, '--max-new-units', 1) == 0
    run = tmp_path/'runs/stage0/example'
    assert not (run/'complete.json').exists()
    unit = run/'units/sft/complete.json'
    sealed = unit.read_bytes()
    assert command(tmp_path, 'audit', '--config', plan) == 0
    assert unit.read_bytes() == sealed
    tensor = tmp_path/'inputs/sft/64/adapter_model.safetensors'
    tensor.write_bytes(tensor.read_bytes()+b'changed')
    assert command(tmp_path, 'status', '--config', plan) == 1
    assert command(tmp_path, 'audit', '--config', plan) == 1


def test_missing_optional_zero_and_random_do_not_block_endpoint_readiness(tmp_path):
    selection = sources(tmp_path)
    sft = tmp_path/'inputs/sft/0/adapter_model.safetensors'
    sft.unlink()
    assert command(tmp_path, 'prepare', '--selection', selection, '--name', 'endpoints') == 0
    plan = tmp_path/'plans/endpoints/stage0.plan.json'
    assert command(tmp_path, 'audit', '--config', plan) == 0
    result = json.loads((tmp_path/'runs/stage0/endpoints/results/source-audit.json').read_text())
    assert result['primary_sources_ready'] is True
    assert result['initialization']['state'] == 'unavailable'
    assert result['sources']['sft']['checkpoints']['0']['state'] == 'missing'


def test_budget_is_unknown_until_observed_and_cumulative_across_plans(tmp_path, capsys):
    plan = prepared(tmp_path)
    observation = tmp_path/'observation.json'
    write_json(observation, dict(event_id='first', session_id='cpu-setup', total_units=3.5,
                                 observed_at='2026-10-26T12:00:00+00:00', note='Stage 0 cumulative meter reading'))
    assert command(tmp_path, 'budget-record', '--config', plan, '--observation', observation) == 0
    assert command(tmp_path, 'budget-record', '--config', plan, '--observation', observation) == 0
    assert command(tmp_path, 'prepare', '--selection', tmp_path/'selection.json', '--name', 'other') == 0
    other = tmp_path/'plans/other/stage0.plan.json'
    from pilot_eval.stage0_workflow import status
    assert status(other, tmp_path)['budget']['spent_units'] == 3.5
    assert status(other, tmp_path)['budget']['remaining_units'] == 16.5
    ledger = tmp_path/'runs/stage0/budget/ledger.jsonl'
    assert len(ledger.read_text().splitlines()) == 1
    data = json.loads(observation.read_text());data['total_units'] = 1
    write_json(observation, data)
    assert command(tmp_path, 'budget-record', '--config', plan, '--observation', observation) == 1


def test_tampered_plan_and_budget_records_fail_closed(tmp_path):
    plan = prepared(tmp_path)
    value = json.loads(plan.read_text());value['budget_ceiling_units'] = 200
    write_json(plan, value)
    assert command(tmp_path, 'status', '--config', plan) == 1


def test_missing_endpoint_is_audited_but_not_ready(tmp_path):
    selection = sources(tmp_path)
    (tmp_path/'inputs/grpo/64/adapter_model.safetensors').unlink()
    assert command(tmp_path, 'prepare', '--selection', selection, '--name', 'missing') == 0
    assert command(tmp_path, 'audit', '--config', tmp_path/'plans/missing/stage0.plan.json') == 0
    result = json.loads((tmp_path/'runs/stage0/missing/results/source-audit.json').read_text())
    assert result['primary_sources_ready'] is False


def test_budget_observations_cannot_be_rewritten_or_reset(tmp_path):
    plan = prepared(tmp_path)
    observation = tmp_path/'observation.json'
    def record(event, amount):
        write_json(observation, dict(event_id=event, session_id='colab', total_units=amount,
                                    observed_at='2026-10-26T12:00:00+00:00', note='Cumulative usage'))
        return command(tmp_path, 'budget-record', '--config', plan, '--observation', observation)
    assert record('start', 5) == 0
    assert record('decreasing', 4) == 1
    assert record('stop', 20) == 0
    from pilot_eval.stage0_workflow import status
    assert status(plan,tmp_path)['budget']['state'] == 'ceiling-reached'
    assert status(plan,tmp_path)['budget']['gpu_authorized'] is False
    ledger = tmp_path/'runs/stage0/budget/ledger.jsonl'
    saved = ledger.read_text()
    ledger.write_text(saved.replace('"total_units": 5', '"total_units": 0'))
    assert command(tmp_path, 'status', '--config', plan) == 1


def test_audit_rejects_ownership_and_explicit_recovery_is_recorded(tmp_path):
    plan = prepared(tmp_path)
    run = tmp_path/'runs/stage0/example'
    write_json(run/'writer.json', dict(token='abandoned', pid=99999999, host='old-colab'))
    assert command(tmp_path, 'audit', '--config', plan) == 1
    assert command(tmp_path, 'recover-writer', '--config', plan, '--owner-token', 'wrong',
                   '--review-notes', 'Previous Colab runtime was disconnected.') == 1
    assert command(tmp_path, 'recover-writer', '--config', plan, '--owner-token', 'abandoned',
                   '--review-notes', 'Previous Colab runtime was disconnected.') == 0
    assert (run/'logs/ownership-recovery.jsonl').is_file()
    assert command(tmp_path, 'audit', '--config', plan) == 0


def test_reserved_budget_name_cannot_collide_with_global_ledger(tmp_path):
    selection = sources(tmp_path)
    assert command(tmp_path, 'prepare', '--selection', selection, '--name', 'budget') == 1


def test_training_metadata_is_verified_and_preserved_without_replaying_optimizer_states(tmp_path):
    selection = sources(tmp_path)
    training = tmp_path/'inputs/sft/training.json'
    train_config = dict(optimizer={'learning_rate': 0.0001}, seed=42, dtype='float32',
                        training_indices=[10,20], prompt_template={'name':'fixture'})
    write_json(training, train_config)
    source = tmp_path/'inputs/sft/activation.prepared.json'
    config = json.loads(source.read_text())
    config['source_config_path'] = str(training.relative_to(tmp_path))
    config['source_evidence']['source_config_sha256'] = _hash(train_config)
    config['source_evidence']['files'] = {str(training.relative_to(tmp_path)): hashlib.sha256(training.read_bytes()).hexdigest()}
    write_json(source, config)
    write_json(source.parent/'prepare-complete.json', dict(config_sha256=_hash(config),
        files={str(source.relative_to(tmp_path)):hashlib.sha256(source.read_bytes()).hexdigest()}))
    assert command(tmp_path,'prepare','--selection',selection,'--name','metadata') == 0
    plan = tmp_path/'plans/metadata/stage0.plan.json'
    assert command(tmp_path,'audit','--config',plan) == 0
    result = json.loads((tmp_path/'runs/stage0/metadata/results/source-audit.json').read_text())
    assert result['sources']['sft']['training_metadata']['optimizer'] == {'learning_rate':0.0001}
    training.write_text('{}')
    assert command(tmp_path,'status','--config',plan) == 1


def test_explicit_recovery_refuses_a_live_same_host_writer(tmp_path):
    import os
    import platform
    plan = prepared(tmp_path)
    marker=tmp_path/'runs/stage0/example/writer.json'
    write_json(marker,dict(token='live',pid=os.getpid(),host=platform.node()))
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','live',
                   '--review-notes','Attempted while active.') == 1
    assert marker.exists()


def test_removing_last_budget_event_cannot_lower_known_consumption(tmp_path):
    plan=prepared(tmp_path)
    observation=tmp_path/'observation.json'
    for i,amount in enumerate([2,8]):
        write_json(observation,dict(event_id=str(i),session_id='session',total_units=amount,
                                   observed_at='2026-10-26T12:00:00+00:00',note='Observed total'))
        assert command(tmp_path,'budget-record','--config',plan,'--observation',observation)==0
    ledger=tmp_path/'runs/stage0/budget/ledger.jsonl'
    ledger.write_text(ledger.read_text().splitlines()[0]+'\n')
    assert command(tmp_path,'status','--config',plan)==1


def test_interrupted_budget_head_publication_recovers_with_same_observation(tmp_path):
    plan=prepared(tmp_path)
    observation=tmp_path/'observation.json'
    write_json(observation,dict(event_id='reading',session_id='session',total_units=2,
                               observed_at='2026-10-26T12:00:00+00:00',note='Observed total'))
    assert command(tmp_path,'budget-record','--config',plan,'--observation',observation)==0
    (tmp_path/'runs/stage0/budget/head.json').unlink()
    from pilot_eval.stage0_workflow import status
    assert status(plan,tmp_path)['budget']['state']=='unpublished-observation'
    assert command(tmp_path,'budget-record','--config',plan,'--observation',observation)==0
    assert status(plan,tmp_path)['budget']['state']=='observed'
    assert len((tmp_path/'runs/stage0/budget/ledger.jsonl').read_text().splitlines())==1


def test_notebook_cells_are_clean_public_commands_with_no_gpu_setup():
    import ast
    for name in ['stage-0-cpu-colab.ipynb', 'stage-0-gpu-colab.ipynb']:
        notebook = json.loads((Path('notebooks')/name).read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert "run_stage0('status', '--config', PLAN)" in code
        assert 'torch' not in code and 'pip install' not in code
        assert 'RELEASE_COMMIT' in code and 'checkout' in code
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                assert cell['outputs']==[] and cell['execution_count'] is None
