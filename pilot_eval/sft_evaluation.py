"""Matched checkpoint evaluation through the existing frozen instrument."""

import json
from pathlib import Path

from pilot_eval.rescoring import rescore_gsm8k
from pilot_eval.sft import load_sft
from pilot_eval.training import TrainingDependencies, run_lock, training_directory, verified_checkpoints
from pilot_eval.workflow import _save_frozen, adapter_digest, execute_config


def evaluation_directory(root, config):
    return Path(root).resolve() / 'runs' / config['experiment'] / config['model'].replace('/', '--') / \
        config['dataset'] / config['cohort'] / config['variant'] / config['run_id']


def evaluation_config(config, root, checkpoint):
    source = config['source_config']
    training = training_directory(root, config)
    if not (training / 'results/preflight.json').exists():
        raise ValueError('successful training preflight is required before evaluation')
    adapter = None
    if checkpoint != 'baseline':
        if checkpoint not in ['0', '8', '16', '32', '64']:
            raise ValueError('unsupported checkpoint')
        checkpoints = verified_checkpoints(training)
        if int(checkpoint) not in checkpoints:
            raise ValueError('checkpoint is not complete and verified')
        adapter = str(checkpoints[int(checkpoint)].resolve())
    label = 'baseline' if checkpoint == 'baseline' else f'step-{checkpoint}'
    result = {**source, 'protocol_version': 'pilot2-eval-v1', 'experiment': 'pilot-2-eval',
        'run_id': config['run_id'] + '-' + label, 'variant': 'float32-' + label,
        'dtype': 'float32', 'batch_size': config.get('evaluation_batch_size', 1), 'adapter': adapter, 'adapter_revision': None,
        'adapter_sha256': adapter_digest(adapter) if adapter else None,
        'items_path': config['evaluation_items_path'], 'items_sha256': config['evaluation_items_sha256']}
    if config.get('hardware'):
        result['hardware'] = config['hardware']
    path = Path(root).resolve() / 'plans' / config['run_id'] / 'evaluations' / f'{label}.config.json'
    _save_frozen(path, result)
    return path, result


def verify_zero(root, config):
    pairs = []
    for checkpoint in ['baseline', '0']:
        _, variant = evaluation_config(config, root, checkpoint)
        directory = evaluation_directory(root, variant)
        if not (directory / 'results/results.json').exists():
            raise ValueError('baseline and checkpoint zero must be complete')
        pairs.append([json.loads(line) for line in (directory / 'results/responses.jsonl').read_text().splitlines()])
    fields = ['id', 'generated_text', 'token_count', 'stop_reason', 'score']
    if [[{k: r[k] for k in fields} for r in records] for records in pairs][0] != \
            [{k: r[k] for k in fields} for r in pairs[1]]:
        raise ValueError('checkpoint zero does not match untuned behavior')
    return True


def evaluate_sft(config_path, output_root, checkpoint, *, dependencies=None):
    root = Path(output_root).resolve()
    config, _, _ = load_sft(config_path, root)
    with run_lock(training_directory(root, config)):
        return _evaluate(config, root, checkpoint, dependencies)


def _evaluate(config, root, checkpoint, dependencies):
    checkpoint = str(checkpoint)
    if checkpoint not in ['baseline', '0']:
        verify_zero(root, config)
    path, variant = evaluation_config(config, root, checkpoint)
    deps = dependencies or TrainingDependencies()
    # Require the exact training environment, including commit, for all cells.
    recorded = json.loads((training_directory(root, config) / 'meta/runtime.json').read_text())
    if deps.runtime(variant) != recorded:
        raise ValueError('evaluation and training runtime mismatch')
    summary = execute_config(path, root, dependencies=deps)
    directory = evaluation_directory(root, variant)
    rescored = rescore_gsm8k(directory / 'results/responses.jsonl', directory / 'config.json',
        directory / 'results/results.json', root, variant['run_id'] + '-flexible-v2')
    result = dict(**summary, flexible_v2_accuracy=rescored['flexible_v2_accuracy'],
                  flexible_v2_correct=rescored['flexible_v2_correct'],
                  flexible_v2_wilson_95=rescored['flexible_v2_wilson_95'],
                  flexible_v2_invalid_count=rescored['flexible_v2_invalid_count'])
    if checkpoint == '0':
        result['zero_checkpoint_matches_baseline'] = verify_zero(root, config)
    return result
