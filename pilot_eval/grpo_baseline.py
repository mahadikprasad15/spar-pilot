"""Frozen training-cohort sampling baseline; completed groups are never regenerated."""
import json
import math
from pathlib import Path

from pilot_eval.grpo_prepare import load_grpo_prepared, _within
from pilot_eval.run import _write_json, _write_state
from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v3
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen, _hash


def _settings(value, training):
    required = {'subset_ids', 'scorer', 'seed', 'groups_per_batch', 'max_new_tokens', 'temperature', 'top_p', 'top_k'}
    if set(value) != required:
        raise ValueError('baseline settings require exactly: ' + ', '.join(sorted(required)))
    ids = value['subset_ids']
    if (not isinstance(ids, list) or len(ids) != 128 or len(set(ids)) != 128
            or not set(ids).issubset({row['id'] for row in training})):
        raise ValueError('baseline subset must contain 128 unique training IDs in explicit order')
    if value['scorer'] != 'gsm8k-flexible-v3':
        raise ValueError('baseline reward scorer must be explicitly gsm8k-flexible-v3')
    for key in ['seed', 'groups_per_batch', 'max_new_tokens', 'top_k']:
        if type(value[key]) is not int or value[key] < (1 if key in ['groups_per_batch', 'max_new_tokens'] else 0):
            raise ValueError('invalid baseline integer setting: ' + key)
    if (value['groups_per_batch'] > 128
            or type(value['temperature']) not in [int, float] or value['temperature'] != 1.0
            or type(value['top_p']) not in [int, float] or not 0 < value['top_p'] <= 1):
        raise ValueError('invalid sampling temperature/filter/batching')
    return value


def _score(row, output, draw):
    tokens = output['token_ids']
    if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
            or output['stop_reason'] not in ['eos', 'cap'] or not isinstance(output['text'], str)):
        raise ValueError('invalid sampled continuation metadata')
    capped = output['stop_reason'] == 'cap'
    strict = score_gsm8k(output['text'], row['gold'], capped=capped)['strict']
    flexible = score_gsm8k_flexible_v3(output['text'], row['gold'], capped=capped)
    if strict['correct'] and not flexible['correct']:
        raise ValueError('strict-correct/flexible-incorrect scorer contract violated')
    return dict(group_id=row['id'], draw_id=f"{row['id']}:draw:{draw}", draw_index=draw,
                prompt=row['prompt'], gold=row['gold'], **output, token_count=len(tokens),
                capped=capped, strict=strict, flexible=flexible,
                reward=0 if capped else int(flexible['correct']))


def _percentile(values, q):
    ordered = sorted(values)
    position = (len(values) - 1) * q / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _summary(records, settings):
    groups = []
    for start in range(0, len(records), 8):
        rows = records[start:start + 8]
        groups.append(dict(group_id=rows[0]['group_id'], correct_count=sum(r['reward'] for r in rows)))
    lengths = [r['token_count'] for r in records]
    caps = sum(r['capped'] for r in records)
    p99 = _percentile(lengths, 99)
    # If censored draws could occupy the top one percent, p99 cannot justify a final cap.
    censored = caps > 0 and caps >= len(records) * .01
    return dict(total=len(records), groups=groups, cap_count=caps, cap_fraction=caps / len(records),
                mean_tokens=sum(lengths) / len(lengths),
                strict_invalid_count=sum(r['strict']['status'] == 'invalid' for r in records),
                flexible_invalid_count=sum(r['flexible']['status'] == 'invalid' for r in records),
                all_correct_groups=sum(g['correct_count'] == 8 for g in groups),
                all_incorrect_groups=sum(g['correct_count'] == 0 for g in groups),
                strict_accuracy=sum(r['strict']['correct'] for r in records) / len(records),
                flexible_accuracy=sum(r['reward'] for r in records) / len(records),
                dead_group_fraction=sum(g['correct_count'] in [0, 8] for g in groups) / len(groups),
                length_percentiles={str(q): _percentile(lengths, q) for q in [50, 90, 95, 99, 100]},
                p99_censored=censored, final_cap_evidence_eligible=not censored,
                final_cap=None, scorer=settings['scorer'],
                note='Capped responses receive reward zero, which can favor shorter answers. '
                     'Final cap requires explicit review; censored p99 cannot justify it. '
                     'Use a new sampling name for an extension; preserve this run.')


def sample_baseline(config_path, settings_path, output_root, name, dependencies=None):
    root = Path(output_root).resolve()
    source_path = _within(root, config_path)
    plan, training, *_ = load_grpo_prepared(source_path, root)
    settings = _settings(json.loads(Path(settings_path).read_text()), training)
    by_id = {row['id']: row for row in training}
    rows = [by_id[i] for i in settings['subset_ids']]
    from pilot_eval import scoring
    implementation = {'sampling': file_hash(Path(__file__)), 'scorer': file_hash(Path(scoring.__file__))}
    directory = root / 'runs/pilot-4' / plan['model'].replace('/', '--') / 'sampling-baseline' / safe_name(name)
    config = dict(protocol_version='pilot4-sampling-baseline-v1', model=plan['model'],
                  model_revision=plan['model_revision'], tokenizer_revision=plan['tokenizer_revision'],
                  dtype='float32', adapter=None, implementation_sha256=implementation, source_path=str(source_path.relative_to(root)),
                  source_sha256=file_hash(source_path), settings=settings, group_size=8,
                  decoding=dict(do_sample=True, num_beams=1, repetition_penalty=1.0,
                                no_repeat_ngram_size=0, **{k: settings[k] for k in ['temperature', 'top_p', 'top_k', 'max_new_tokens']}),
                  prompt_template=plan['source_prompt_contract'],
                  run_path=str(directory.relative_to(root)),
                  batches=[dict(ids=settings['subset_ids'][i:i + settings['groups_per_batch']], seed=settings['seed'] + n)
                           for n, i in enumerate(range(0, 128, settings['groups_per_batch']))])
    with run_lock(directory):
        _save_frozen(root / 'plans' / safe_name(name) / 'baseline.config.json', config)
        _save_frozen(directory / 'config.json', config)
        records, missing = [], []
        for index, batch in enumerate(config['batches']):
            shard = directory / 'batches' / f'{index:04d}.json'
            marker = shard.with_suffix('.complete.json')
            if not marker.exists():
                missing.append(index)
                continue
            seal = json.loads(marker.read_text())
            if seal != {'sha256': file_hash(shard), 'config_sha256': _hash(config)}:
                raise ValueError('sampling shard/config hash mismatch')
            saved = json.loads(shard.read_text())
            expected = [f'{i}:draw:{d}' for i in batch['ids'] for d in range(8)]
            if [r['draw_id'] for r in saved] != expected:
                raise ValueError('sampling shard draw identity mismatch')
        sampler = None
        try:
            _write_state(directory, 'running', len(config['batches']) - len(missing), len(config['batches']))
            if missing:
                dependencies = dependencies or HFSamplingDependencies()
                sampler = dependencies.load_sampler(plan)
                _save_frozen(directory / 'meta/runtime.json', sampler.runtime())
            for index in missing:
                batch = config['batches'][index]
                selected = [by_id[i] for i in batch['ids']]
                generated = sampler.sample(selected, settings, batch['seed'])
                if len(generated) != len(selected) * 8:
                    raise ValueError('sampling backend returned wrong completion count')
                saved = [_score(row, generated[n * 8 + d], d) for n, row in enumerate(selected) for d in range(8)]
                limit = settings['max_new_tokens']
                if any((r['capped'] and r['token_count'] != limit)
                       or (not r['capped'] and r['token_count'] >= limit) for r in saved):
                    raise ValueError('sample stop metadata disagrees with frozen cap')
                shard = directory / 'batches' / f'{index:04d}.json'
                _write_json(shard, saved)
                _write_json(shard.with_suffix('.complete.json'), {'sha256': file_hash(shard), 'config_sha256': _hash(config)})
                print(f'sampling: completed batch {index + 1}/{len(config["batches"])}', flush=True)
                _write_state(directory, 'running', len(config['batches']) - len(missing) + missing.index(index) + 1, len(config['batches']))
            for index in range(len(config['batches'])):
                records.extend(json.loads((directory / 'batches' / f'{index:04d}.json').read_text()))
            summary = _summary(records, settings)
            result = directory / 'results/responses.jsonl'
            result.parent.mkdir(parents=True, exist_ok=True)
            text = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records)
            if result.exists() and result.read_text() != text:
                raise ValueError('existing combined sampling responses differ')
            if not result.exists():
                temp = result.with_suffix('.tmp'); temp.write_text(text); temp.replace(result)
            _save_frozen(directory / 'results/results.json', summary)
            _write_state(directory, 'completed', len(config['batches']), len(config['batches']))
            return summary
        except BaseException as exc:
            _write_state(directory, 'failed', sum((directory / 'batches' / f'{i:04d}.complete.json').exists() for i in range(len(config['batches']))), len(config['batches']), str(exc))
            raise
        finally:
            if sampler is not None:
                sampler.close()


class HFSamplingDependencies:
    def load_sampler(self, plan):
        return HFSampler(plan)


class HFSampler:
    """One pinned FP32 untuned model; explicit full sampling policy and batch seeds."""
    def __init__(self, plan):
        import os
        import importlib.metadata
        from pilot_eval.sft_backend import PINS
        from pilot_eval.backend import load_hf_backend
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        versions = {k: importlib.metadata.version(k) for k in PINS}
        if any(versions[k].split('+')[0] != v for k, v in PINS.items()):
            raise ValueError('sampling requires the pinned Pilot 4 dependencies')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('sampling requires exactly one CUDA GPU')
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.backend = load_hf_backend(dict(model=plan['model'], model_revision=plan['model_revision'],
            tokenizer_revision=plan['tokenizer_revision'], dtype='float32', deterministic=True,
            seed=plan['seed'], attention_implementation='eager', adapter=None))
        if any(p.dtype != torch.float32 for p in self.backend.model.parameters()):
            raise ValueError('baseline model parameters must be FP32')
    def runtime(self):
        import importlib.metadata
        from pilot_eval.sft_backend import PINS
        import sys
        return dict(dtype='float32', python=sys.version, cuda=self.torch.version.cuda,
                    compute_capability=list(self.torch.cuda.get_device_capability(0)),
                    cublas_workspace_config=__import__('os').environ['CUBLAS_WORKSPACE_CONFIG'],
                    deterministic_algorithms=self.torch.are_deterministic_algorithms_enabled(),
                    device=self.torch.cuda.get_device_name(0),
                    versions={k: importlib.metadata.version(k) for k in PINS}, tf32=False,
                    attention_implementation='eager')
    def sample(self, rows, settings, seed):
        from transformers import set_seed
        set_seed(seed)
        tokenizer = self.backend.tokenizer
        for row in rows:
            if tokenizer.encode(row['prompt'], add_special_tokens=False) != row['input_ids'][:row['prompt_tokens']]:
                raise ValueError('sampling tokenizer/prompt IDs differ from saved training inputs')
        inputs = tokenizer([r['prompt'] for r in rows for _ in range(8)], return_tensors='pt', padding=True,
                           add_special_tokens=False).to(self.backend.model.device)
        with self.torch.inference_mode():
            output = self.backend.model.generate(**inputs, do_sample=True, temperature=settings['temperature'],
                top_p=settings['top_p'], top_k=settings['top_k'], max_new_tokens=settings['max_new_tokens'],
                num_beams=1, num_return_sequences=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        results = []
        eos = tokenizer.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        for row in output:
            raw = row[inputs['input_ids'].shape[1]:].tolist()
            end = next((i for i, t in enumerate(raw) if t in eos), None)
            tokens = raw if end is None else raw[:end]
            results.append(dict(text=tokenizer.decode(tokens, skip_special_tokens=True), token_ids=tokens, generated_token_ids=raw if end is None else raw[:end + 1],
                                stop_reason='cap' if end is None else 'eos'))
        return results
    def close(self):
        import gc
        del self.backend
        gc.collect()
        self.torch.cuda.empty_cache()
