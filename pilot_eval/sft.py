"""Frozen Pilot 2 preparation; model weights are not needed at this boundary."""

import json
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval.config import validate_config
from pilot_eval.protocol import build_gsm8k_prompt, gsm8k_messages
from pilot_eval.workflow import HFDependencies, _hash, _save_frozen


def optimizer_settings():
    return dict(name='adamw_torch', learning_rate=1e-4, beta1=0.9, beta2=0.999,
                epsilon=1e-8, weight_decay=0.0, max_grad_norm=1.0,
                scheduler='constant', warmup_steps=0, microbatch=1,
                gradient_accumulation_steps=8, max_steps=64)


def adapter_settings():
    return dict(rank=1, alpha=1, dropout=0.0, layers=28,
                projections=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('unsafe plan/run name')
    return name


def read_artifact(root, relative):
    path = (Path(root).resolve() / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('artifact path escapes root')
    return json.loads(path.read_text())


def load_sft(config_path, root):
    config = json.loads(Path(config_path).read_text())
    safe_name(config.get('run_id', ''))
    if config['protocol_version'] != 'pilot2-sft-v1' or config['dtype'] != 'float32':
        raise ValueError('unsupported SFT protocol/precision')
    locked = dict(seed=42, experiment='pilot-2', optimizer=optimizer_settings(),
                  adapter=adapter_settings(), full_determinism=True,
                  gradient_checkpointing=True, use_cache=False, packing=False,
                  checkpoint_steps=[0, 8, 16, 32, 64])
    if any(config.get(key) != value for key, value in locked.items()):
        raise ValueError('configuration differs from locked Pilot 2 protocol; record a new protocol version')
    source = validate_config(config['source_config'])
    if (source['model'] != 'Qwen/Qwen2.5-1.5B-Instruct' or source['adapter'] is not None
            or source['scorer'] != 'gsm8k' or len(source['prompt_indices']) != 150
            or any(config.get(key) != source[key] for key in ['model', 'model_revision', 'tokenizer_revision'])):
        raise ValueError('configuration does not match pinned untuned source')
    rows = read_artifact(root, config['training_items_path'])
    items = read_artifact(root, config['evaluation_items_path'])
    for field, data in [('training', rows), ('evaluation', items)]:
        if _hash(data) != config[field + '_items_sha256']:
            raise ValueError(field + ' input hash mismatch')
    if (len(rows) != 512 or len({r['id'] for r in rows}) != 512
            or [r['source_index'] for r in rows] != config['training_indices']
            or config['max_length'] != max(len(r['input_ids']) for r in rows)
            or config['evaluation_items_sha256'] != source['items_sha256']):
        raise ValueError('configuration/cohort mismatch')
    for row in rows:
        boundary = row['prompt_tokens']
        if (not 0 < boundary < len(row['input_ids'])
                or row['labels'] != [-100] * boundary + row['input_ids'][boundary:]
                or row['labels'][-1] != source['decoding']['eos_token_id']):
            raise ValueError('invalid completion-only labels/end-turn')
    plan = read_artifact(root, config['analysis_plan_path'])
    if _hash(plan) != config['analysis_plan_sha256']:
        raise ValueError('analysis plan hash mismatch')
    return config, rows, items


def prepare_sft(source_config, output_root, name, *, dependencies=None):
    """Freeze a 512-item training plan derived from an existing untuned cell."""
    root = Path(output_root).resolve()
    directory = root / 'plans' / safe_name(name)
    path = directory / 'sft.config.json'
    source = validate_config(json.loads(Path(source_config).read_text()))
    if source['scorer'] != 'gsm8k' or source['adapter'] is not None:
        raise ValueError('SFT source must be an untuned GSM8K configuration')
    if len(source['prompt_indices']) != 150:
        raise ValueError('SFT requires the frozen 150-item evaluation cohort')
    # Runtime is measured afresh; all original protocol inputs remain pinned.
    source = {k: v for k, v in source.items() if k != 'runtime'}
    if path.exists():
        config, _, _ = load_sft(path, root)
        if config['source_config'] != source:
            raise ValueError('SFT source config mismatch; use a new plan')
        return path
    items = read_artifact(root, source['items_path'])
    if _hash(items) != source['items_sha256']:
        raise ValueError('source evaluation input hash mismatch')
    deps = dependencies or HFDependencies()
    tokenizer, context = deps.load_tokenizer(source['model'], source['tokenizer_revision'])
    if _hash(tokenizer.chat_template) != source['chat_template_sha256']:
        raise ValueError('chat template hash mismatch')
    data = deps.load_dataset(source['dataset_path'], source['dataset_config'], source['dataset_revision'])
    expected_ids = [f'gsm8k:test:{i}' for i in source['prompt_indices']]
    if [i['id'] for i in items] != expected_ids:
        raise ValueError('evaluation cohort IDs/order mismatch')
    test_questions = set()
    for item in items:
        row = data['test'][item['source_index']]
        if (item['prompt'] != build_gsm8k_prompt(row['question'], tokenizer)
                or item['gold'] != row['answer']):
            raise ValueError('source prompt/gold mismatch with pinned dataset')
        test_questions.add(' '.join(row['question'].split()))
    indices = sorted(random.Random(42).sample(range(len(data['train'])), 512))
    rows = []
    for index in indices:
        row = data['train'][index]
        if ' '.join(row['question'].split()) in test_questions:
            raise ValueError('training/evaluation question overlap')
        prompt = build_gsm8k_prompt(row['question'], tokenizer)
        full = tokenizer.apply_chat_template(
            gsm8k_messages(row['question']) + [{'role': 'assistant', 'content': row['answer']}],
            tokenize=False, add_generation_prompt=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = tokenizer.encode(full, add_special_tokens=False)
        if tokens[:len(prefix)] != prefix:
            raise ValueError('prompt/completion token boundary mismatch')
        ends = [i for i in range(len(prefix), len(tokens)) if tokens[i] == tokenizer.eos_token_id]
        if len(ends) != 1:
            raise ValueError('completion must contain exactly one supervised end-turn token')
        tokens = tokens[:ends[0] + 1]  # No supervision on template whitespace after end-turn.
        if len(tokens) > min(context, source['context_limit']):
            raise ValueError('full training sequence context overflow; truncation is forbidden')
        rows.append(dict(id=f'gsm8k:train:{index}', source_index=index, source_split='train',
                         dataset_revision=source['dataset_revision'], question=row['question'],
                         gold=row['answer'], prompt=prompt, rendered=full, input_ids=tokens,
                         prompt_tokens=len(prefix), labels=[-100] * len(prefix) + tokens[len(prefix):],
                         gold_tokens=len(tokenizer.encode(row['answer'], add_special_tokens=False))))
    plan_path = directory / 'analysis-plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = dict(recorded_at=datetime.now(timezone.utc).isoformat(), mode='exploratory',
                    directional_prediction=None, binary_collapse_threshold=None,
                    checkpoints=[0, 8, 16, 32, 64], scorers=['strict', 'gsm8k-flexible-v2'],
                    accuracy_interval='Wilson 95%',
                    paired_interval=dict(method='paired-percentile-bootstrap', draws=2000,
                                         seed=42, percentiles=[2.5, 97.5]),
                    measurements=['accuracy', 'mean/median tokens', 'paired accuracy change',
                                  'mean-length change', 'ratio of mean lengths', 'invalid/capped rates'],
                    diagnostics=['gold-target lengths', 'checkpoint trajectory', 'paired responses'],
                    limitations=['single training seed', 'sampled evaluation cohort',
                                 'historical recipe unavailable', 'no causal interpretation'])
    training_path = directory / 'training.items.json'
    eval_path = directory / 'evaluation.items.json'
    config = dict(protocol_version='pilot2-sft-v1', run_id=name, experiment='pilot-2',
                  source_config=source, model=source['model'], model_revision=source['model_revision'],
                  tokenizer_revision=source['tokenizer_revision'], seed=42, dtype='float32',
                  optimizer=optimizer_settings(), adapter=adapter_settings(),
                  full_determinism=True, gradient_checkpointing=True, use_cache=False,
                  packing=False, checkpoint_steps=[0, 8, 16, 32, 64],
                  max_length=max(len(r['input_ids']) for r in rows), training_indices=indices,
                  selection_method='python-random-sample-without-replacement-sorted',
                  training_items_path=str(training_path.relative_to(root)), training_items_sha256=_hash(rows),
                  evaluation_items_path=str(eval_path.relative_to(root)), evaluation_items_sha256=_hash(items),
                  analysis_plan_path=str(plan_path.relative_to(root)), analysis_plan_sha256=_hash(plan))
    for file, value in [(training_path, rows), (eval_path, items), (plan_path, plan),
                        (directory / 'lengths.json', dict(gold_mean=statistics.mean(r['gold_tokens'] for r in rows),
                            gold_median=statistics.median(r['gold_tokens'] for r in rows),
                            full_max=config['max_length'], gold_rule='gold text only; no chat tokens',
                            full_rule='prompt plus completion through supervised end-turn')), (path, config)]:
        _save_frozen(file, value)
    return path
