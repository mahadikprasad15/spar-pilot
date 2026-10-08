"""CPU-only Pilot 4 source verification and immutable preparation.

Preparation records pending choices; it never chooses them or loads a model.
"""

import json
from pathlib import Path

from pilot_eval.activation_prepare import load_prepared
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import load_sft, safe_name
from pilot_eval.training import file_hash, run_lock, training_directory
from pilot_eval.workflow import _hash, _save_frozen


def _within(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('GRPO source/payload must be within the artifact root')
    return path


def _sources(root, sft_path, measurement_path):
    sft_path, measurement_path = _within(root, sft_path), _within(root, measurement_path)
    sft, training, evaluation = load_sft(sft_path, root)
    measured, sequences = load_prepared(measurement_path, root)
    if (_within(root, measured['source_config_path']) != sft_path
            or measured['source_evidence']['source_config_sha256'] != _hash(sft)):
        raise ValueError('measurement and SFT source identities differ')
    if any(measured[k] != sft[k] for k in ['model', 'model_revision', 'tokenizer_revision', 'dtype', 'seed']):
        raise ValueError('measurement and training pins/precision differ')
    gsm = [row for row in sequences if row['corpus'] == 'gsm8k']
    web = [row for row in sequences if row['corpus'] == 'fineweb']
    if ([row['id'] for row in gsm] != [row['id'] for row in evaluation]
            or [row['gold'] for row in gsm] != [row['gold'] for row in evaluation]
            or len(web) != 150 or len({row['id'] for row in sequences}) != 300):
        raise ValueError('measurement/evaluation cohort mismatch')
    if any(row['source_split'] != 'train' or row['id'] != f"gsm8k:train:{row['source_index']}"
           for row in training):
        raise ValueError('training cohort has wrong split or source IDs')
    if set(' '.join(row['question'].split()) for row in training) & set(
            ' '.join(row['question'].split()) for row in gsm):
        raise ValueError('training/evaluation question overlap')
    order = json.loads((training_directory(root, sft) / 'inputs/order.json').read_text())
    if (not isinstance(order, list) or len(order) != 512
            or set(order) != {row['id'] for row in training}):
        raise ValueError('saved SFT optimizer prompt order is incomplete or duplicated')
    return sft, training, evaluation, measured, order


def _pending():
    reasons = {
        'reward_scorer': ('gsm8k-flexible-v3', 'Confirm the corrected reward/evaluation scorer version.'),
        'loss_configuration': ('dapo', 'Verify full-window token normalization using the pinned trainer.'),
        'baseline_subset': (None, 'Freeze the chosen 128 of the existing training problems.'),
        'pilot_generation': (None, 'Freeze baseline sampling limit, filters and batch policy.'),
        'final_completion_limit': (None, 'Derive from uncensored baseline lengths above the 99th percentile.'),
        'monitor_policy': (None, 'Freeze stop/pause/validity actions for caps, dead groups and length changes.'),
        'training_memory_batching': (None, 'Profile microbatch/accumulation while preserving 8 prompts and 64 completions per update.'),
        'gate_margin': (0.10, 'Record the user-chosen non-inferiority margin in preregistration.'),
        'preregistration': (None, 'Dated predictions must precede scientific training.'),
    }
    return {key: {'value': None, 'proposed': proposed, 'reason': reason}
            for key, (proposed, reason) in reasons.items()}


def _audit(config):
    lines = ['# Pilot 4 source and matching audit', '',
             'Prepared sources only: no sampling, training or GPU validation has run.', '',
             f"Model: {config['model']} ({config['model_revision']})", '',
             '| Setting | SFT source | GRPO plan | State |',
             '|---|---|---|---|',
             '| Precision | FP32 | FP32, unquantized | approved; earlier BF16 text superseded |',
             '| Training cohort | 512 saved problems | same ordered IDs and contents | verified |',
             '| Optimizer prompt order | saved SFT execution order | same order, no reshuffle | verified |',
             '| Evaluation cohort | 150 held-out problems | same ordered IDs and contents | verified |',
             '| Fixed measurement inputs | 150 GSM8K + 150 FineWeb | same tokens and masks | verified |',
             '| Adapter | rank 1, alpha 1, 196 projections | same | source verified |',
             f"| Learning rate | {config['optimizer']['learning_rate']} | same | source verified |",
             '| Objective | completion-only SFT | group-normalized GRPO, beta 0 | differs |',
             '| Completions per step | 8 gold solutions | 64 samples for 8 prompts | differs |', '',
             'Historical 0.640 is a reference, not a measured baseline or an accuracy acceptance gate.',
             'Checkpoint zero must match a fresh untuned run under the same settings.', '',
             'Source dtype metadata is verified; live base/adapter tensor dtypes remain a GPU preflight check.', '',
             '## Remaining prerequisites', '']
    lines.extend(f"- **{key}**: {entry['reason']} Proposed: {entry['proposed']!r}; not yet frozen."
                 for key, entry in config['pending'].items())
    lines.extend(['', 'Next: resolve the baseline sampling settings and scorer, then collect the untuned sampling baseline.',
                  'Full scientific execution is blocked until every prerequisite is resolved and reviewed.'])
    return '\n'.join(lines) + '\n'


def prepare_grpo(sft_config, measurement_config, output_root, name):
    """Verify existing sources and publish an immutable plan/audit without HF access."""
    root = Path(output_root).resolve()
    sft_path, measurement_path = _within(root, sft_config), _within(root, measurement_config)
    plan_dir = root / 'plans' / safe_name(name)
    path = plan_dir / 'grpo.prepared.json'
    with run_lock(plan_dir):
        if (plan_dir / 'prepare-complete.json').exists():
            config, *_ = load_grpo_prepared(path, root)
            if (config['sources']['sft']['path'] != str(sft_path.relative_to(root))
                    or config['sources']['measurement']['path'] != str(measurement_path.relative_to(root))):
                raise ValueError('GRPO source options differ; use a new plan name')
            return path
        try:
            sft, training, evaluation, measured, order = _sources(root, sft_path, measurement_path)
        except (ValueError, OSError) as exc:
            _write_state(plan_dir, 'failed', 0, 1, str(exc))
            raise
        run_dir = root / 'runs/pilot-4' / sft['model'].replace('/', '--') / \
            'gsm8k/train-512-seed-42/rank1-float32' / name / 'preparation'
        audit = run_dir / 'results/matching-audit.md'
        config = {
            'schema_version': 1, 'protocol_version': 'pilot4-grpo-preparation-v1',
            'state': 'prepared-with-pending-choices', 'run_id': name,
            'model': sft['model'], 'model_revision': sft['model_revision'],
            'tokenizer_revision': sft['tokenizer_revision'], 'dtype': 'float32',
            'seed': sft['seed'], 'quantization': None, 'full_determinism': True,
            'adapter': sft['adapter'], 'source_optimizer': sft['optimizer'],
            'optimizer': {key: value for key, value in sft['optimizer'].items()
                          if key not in ['microbatch', 'gradient_accumulation_steps']},
            'checkpoint_steps': sft['checkpoint_steps'],
            'sources': {'sft': {'path': str(sft_path.relative_to(root)), 'sha256': file_hash(sft_path)},
                        'measurement': {'path': str(measurement_path.relative_to(root)),
                                        'sha256': file_hash(measurement_path)},
                        'optimizer_order': {
                            'path': str((training_directory(root, sft) / 'inputs/order.json').relative_to(root)),
                            'sha256': file_hash(training_directory(root, sft) / 'inputs/order.json')}},
            'training_indices': sft['training_indices'],
            'optimizer_prompt_order': order,
            'training_items_path': sft['training_items_path'],
            'training_items_sha256': sft['training_items_sha256'],
            'evaluation_items_path': sft['evaluation_items_path'],
            'evaluation_items_sha256': sft['evaluation_items_sha256'],
            'measurement_items_path': measured['items_path'],
            'measurement_items_sha256': measured['items_sha256'],
            'source_evidence': measured['source_evidence'],
            'source_prompt_contract': {key: sft['source_config'][key] for key in
                ['prompt_template', 'chat_template_sha256', 'dataset_path', 'dataset_config',
                 'dataset_revision', 'decoding', 'attention_implementation']},
            'approved': {'group_size': 8, 'prompts_per_optimizer_step': 8,
                         'completions_per_optimizer_step': 64, 'temperature': 1.0,
                         'beta': 0.0, 'num_iterations': 1, 'scale_rewards': 'group',
                         'generation_backend': 'hf', 'capped_reward': 0,
                         'bootstrap': {'method': 'paired-item-percentile', 'draws': 10000,
                                       'seed': 42, 'percentiles': [2.5, 97.5]},
                         'gradient_consistency': 'consecutive-preclip-flat-cosine',
                         'learning_controls': {'seeds': [42, 43], 'steps': 20, 'change': 0.2},
                         'resume': 'sealed-checkpoints-only'},
            'pending': _pending(), 'fineweb': measured['fineweb'], 'views': measured['views'],
            'run_path': str(run_dir.relative_to(root)), 'audit_path': str(audit.relative_to(root)),
        }
        try:
            _save_frozen(path, config)
            _save_frozen(run_dir / 'meta/run_manifest.json', config)
            text = _audit(config)
            if audit.exists() and audit.read_text() != text:
                raise ValueError('existing GRPO audit differs; preserve it and use a new plan')
            if not audit.exists():
                audit.parent.mkdir(parents=True, exist_ok=True)
                temporary = audit.with_suffix('.md.tmp')
                temporary.write_text(text)
                temporary.replace(audit)
            _write_state(run_dir, 'prepared', 1, 1)
            _write_state(plan_dir, 'prepared', 1, 1)
            payloads = [path, audit, run_dir / 'meta/run_manifest.json']
            _write_json(plan_dir / 'prepare-complete.json', {
                'config_sha256': _hash(config),
                'files': {str(p.relative_to(root)): file_hash(p) for p in payloads}})
        except BaseException as exc:
            _write_state(run_dir, 'failed', 0, 1, str(exc))
            raise
        return path


def load_grpo_prepared(config_path, output_root):
    """Revalidate the frozen plan and its source chain before any later stage."""
    root = Path(output_root).resolve()
    path = _within(root, config_path)
    config = json.loads(path.read_text())
    if config.get('protocol_version') != 'pilot4-grpo-preparation-v1':
        raise ValueError('unsupported GRPO preparation protocol')
    marker = json.loads((path.parent / 'prepare-complete.json').read_text())
    required = {str(path.relative_to(root)), config['audit_path'], config['run_path'] + '/meta/run_manifest.json'}
    if marker['config_sha256'] != _hash(config) or not required.issubset(marker['files']):
        raise ValueError('GRPO preparation marker/config mismatch')
    for relative, digest in marker['files'].items():
        if file_hash(_within(root, relative)) != digest:
            raise ValueError('GRPO prepared payload hash mismatch')
    for source in config['sources'].values():
        if file_hash(_within(root, source['path'])) != source['sha256']:
            raise ValueError('GRPO source config hash mismatch')
    sft, training, evaluation, measured, order = _sources(
        root, config['sources']['sft']['path'], config['sources']['measurement']['path'])
    if (config['optimizer_prompt_order'] != order
            or config['training_items_sha256'] != sft['training_items_sha256']
            or config['evaluation_items_sha256'] != sft['evaluation_items_sha256']
            or config['measurement_items_sha256'] != measured['items_sha256']):
        raise ValueError('GRPO cohort hashes differ from sources')
    return config, training, evaluation, measured


def require_grpo_ready(config_path, output_root):
    """Fail clearly on pending choices; no scientific backend is invoked."""
    if json.loads(Path(config_path).read_text()).get('protocol_version') == 'pilot4-frozen-v1':
        from pilot_eval.grpo_preflight import load_frozen
        return load_frozen(config_path, output_root)
    config, *_ = load_grpo_prepared(config_path, output_root)
    pending = [key for key, value in config['pending'].items() if value['value'] is None]
    if pending:
        raise ValueError('GRPO full execution is not ready; unresolved: ' + ', '.join(pending))
    raise ValueError('GRPO execution requires a frozen protocol and reviewed preregistration, '
                     'not a preparation marker')
