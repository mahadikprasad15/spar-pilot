"""Read-only Sampling SFT source checks; no model, tokenizer or GPU imports."""
import json
import random
from pathlib import Path

from pilot_eval.sft import load_sft
from pilot_eval.stage0_sources import audit_source, checkpoint
from pilot_eval.training import file_hash, training_directory
from pilot_eval.workflow import _hash

STEPS = [0, 8, 16, 32, 64]


def inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Sampling SFT source escapes artifact root.')
    return path


def observe(root, path, parents):
    path = inside(root, path)
    relative = str(path.relative_to(root))
    digest = file_hash(path) if path.is_file() else None
    if relative in parents and parents[relative] != digest:
        raise ValueError('Sampling SFT source changed during audit.')
    parents[relative] = digest
    return path


def read(root, path, parents):
    path = observe(root, path, parents)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required source: {path.relative_to(root)}')
    return json.loads(path.read_text())


def verify_parents(root, parents):
    for value, digest in parents.items():
        path = inside(root, value)
        actual = file_hash(path) if path.is_file() else None
        if actual != digest:
            raise ValueError(f'Sampling SFT source availability/hash changed: {value}; use a new plan name.')


def gold_source(root, value, parents):
    path = inside(root, value)
    source = read(root, path, parents)
    # Accept the historical prepared activation selector, but follow its explicit
    # source reference. Never infer a training directory from an evaluation file.
    if source.get('protocol_version') == 'pilot3-write-v1':
        prepared = audit_source(root, 'sft', path)
        parents.update(prepared['parents'])
        path = inside(root, source['source_config_path'])
        source = read(root, path, parents)
    for field in ['training_items_path', 'evaluation_items_path', 'analysis_plan_path']:
        read(root, source[field], parents)
    config, rows, evaluation = load_sft(path, root)
    if config.get('hardware') != 'L4':
        raise ValueError('Gold training configuration must explicitly select L4.')
    directory = inside(root, training_directory(root, config))
    manifest = read(root, directory/'meta/run_manifest.json', parents)
    runtime = read(root, directory/'meta/runtime.json', parents)
    result = read(root, directory/'results/results.json', parents)
    preflight = read(root, directory/'results/preflight.json', parents)
    order = read(root, directory/'inputs/order.json', parents)
    log = observe(root, directory/'logs/steps.jsonl', parents)
    if not log.is_file():
        raise FileNotFoundError('Missing original gold training step logs.')
    logs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if not logs or not any(row.get('step') == 64 for row in logs):
        raise ValueError('Gold training step logs lack completed step64.')
    if manifest.get('config') != config or manifest.get('runtime') != runtime:
        raise ValueError('Original gold training manifest/config/runtime disagree.')
    if 'L4' not in runtime.get('device', ''):
        raise ValueError('Original gold training runtime is not L4; evaluation metadata cannot substitute.')
    resolved_path = observe(root, directory/'meta/resolved_trainer.json', parents)
    resolved = read(root, resolved_path, parents) if resolved_path.is_file() else {}
    # HF's original runtime records may put numeric mode only in resolved trainer args.
    if (runtime.get('tf32', resolved.get('tf32')) is not False
            or runtime.get('autocast', False) is not False
            or resolved.get('fp16', False) is not False
            or resolved.get('bf16', False) is not False
            or runtime.get('dtype', config['dtype']) != 'float32'):
        raise ValueError('Original gold FP32/TF32-off evidence is missing or contradictory.')
    if runtime.get('tf32') is not None and resolved.get('tf32') is not None and runtime['tf32'] != resolved['tf32']:
        raise ValueError('Gold runtime/resolved trainer TF32 evidence disagrees.')
    base = result.get('base_sha256_before')
    if (not isinstance(base, str) or len(base) != 64
            or base != result.get('base_sha256_after') or base != preflight.get('base_sha256')
            or result.get('successful_steps') != 64 or result.get('example_exposures') != 512
            or preflight.get('zero_write') is not True):
        raise ValueError('Gold training completion/base/zero-write evidence disagrees.')
    expected_order = [row['id'] for row in rows]
    random.Random(42).shuffle(expected_order)
    if order != expected_order:
        raise ValueError('Gold saved step order is not the verified seed42 cohort permutation.')
    indices = config['training_indices']
    if (len(set(indices)) != 512 or indices != sorted(indices)
            or any(row['id'] != f'gsm8k:train:{row["source_index"]}'
                   or row.get('source_split') != 'train'
                   or row.get('dataset_revision') != config['source_config']['dataset_revision'] for row in rows)
            or len({row['id'] for row in evaluation}) != 150):
        raise ValueError('Gold cohort split/revision/identity mismatch.')
    # Inventory exactly the adapter payloads and seals consumed by this audit,
    # not expensive historical optimizer files that are irrelevant to source identity.
    evidence = {}
    for step in STEPS:
        folder = directory/'checkpoints'/f'checkpoint-{step}'
        for name in ['complete.json', 'adapter_config.json', 'adapter_model.safetensors']:
            observe(root, folder/name, parents)
        marker = read(root, folder/'complete.json', parents)
        evidence[str(step)] = dict(path=str(folder.relative_to(root)),
            complete_sha256=file_hash(folder/'complete.json'),
            adapter_sha256=marker.get('files', {}).get('adapter_model.safetensors'))
    checkpoints = {str(step): checkpoint(root, evidence[str(step)], str(step), parents) for step in STEPS}
    if any(c['state'] != 'verified-adapter-payload' for c in checkpoints.values()):
        raise ValueError('Gold requires all five complete adapter checkpoints.')
    return dict(state='source-audited', config_path=str(path.relative_to(root)),
        identity={key: config[key] for key in ['model', 'model_revision', 'tokenizer_revision', 'dtype']},
        base_sha256=base, original_runtime=runtime, config=config, resolved_trainer=resolved,
        checkpoint_steps=STEPS, checkpoints=checkpoints, checkpoint0=checkpoints['0'],
        training_ids=[row['id'] for row in rows], heldout_ids=[row['id'] for row in evaluation],
        batches=[order[i:i+8] for i in range(0, 512, 8)],
        prompt_mask_contract=dict(template=config['source_config']['prompt_template'],
            prompt_mask_value=-100, stop_included=True, eos_token_id=config['source_config']['decoding']['eos_token_id'],
            verification='saved token IDs and completion-only masks; retokenization qualification remains pending'),
        initialization_scope='saved original checkpoint0 payload verified; future trained-arm T15 still required')


def grpo_source(root, value, parents):
    path = inside(root, value)
    config = read(root, path, parents)
    # Observe metadata even on a blocked audit so availability changes invalidate it.
    observe(root, path.parent/'prepare-complete.json', parents)
    evidence = config.get('source_evidence', {})
    for item in evidence.get('files', {}):
        observe(root, item, parents)
    if config.get('source_config_path'):
        observe(root, config['source_config_path'], parents)
    for entry in evidence.get('checkpoints', {}).values():
        for name in ['complete.json', 'adapter_config.json', 'adapter_model.safetensors']:
            observe(root, inside(root, entry['path'])/name, parents)
    audit = audit_source(root, 'grpo', path)
    parents.update(audit['parents'])
    if audit['state'] != 'audited' or any(audit['checkpoints'][str(s)]['state'] != 'verified-adapter-payload' for s in [0,64]):
        raise ValueError('GRPO reference lacks verified endpoint adapters.')
    return audit
