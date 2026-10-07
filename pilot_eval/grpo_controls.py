"""Disposable CPU-only real Transformers/PEFT GRPO learning controls."""
import contextlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
from pilot_eval.grpo_algorithm import GRPOWindow, gradient_consistency
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen, _hash

TASK_PATH = Path(__file__).resolve().parents[1] / 'docs/fixtures/pilot4-control-task-v3.json'


@contextlib.contextmanager
def isolated_rng():
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def tiny_model(task, seed):
    """Random tiny Qwen architecture; no Hub, model download or scientific state."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=task['vocab_size'], hidden_size=task['hidden_size'],
        intermediate_size=task['intermediate_size'], num_hidden_layers=task['layers'],
        num_attention_heads=task['heads'], num_key_value_heads=task['heads'],
        max_position_embeddings=32, attention_dropout=0, tie_word_embeddings=False,
        pad_token_id=None, bos_token_id=0, eos_token_id=None, attn_implementation='eager'))
    # Versioned fixture construction, before any learning or base-weight hash.
    if task.get('readout'):
        model.eval()
        with torch.no_grad():
            h0 = model.model(input_ids=torch.tensor([task['prompt_tokens']])).last_hidden_state[0, -1].double()
            original = (model.lm_head.weight[1] - model.lm_head.weight[0]).double()
            direction = original - h0 * (original.dot(h0) / h0.dot(h0))
            if not torch.isfinite(direction).all() or direction.norm() <= 1e-12:
                raise ValueError('degenerate frozen readout construction')
            direction = (direction / direction.norm()).float()
            model.lm_head.weight[0].copy_(-direction / 2)
            model.lm_head.weight[1].copy_(direction / 2)
    model = get_peft_model(model, LoraConfig(r=task['rank'], lora_alpha=task['alpha'],
        lora_dropout=task['dropout'], target_modules=task['target_modules'], task_type='CAUSAL_LM'))
    backend = Tokenizer(WordLevel({'bad': 0, 'good': 1}, unk_token='bad'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='bad', unk_token='bad')
    tokenizer.padding_side = 'left'
    return model.cpu().float(), tokenizer


def _learning_run(directory, task, seed, sign):
    import torch
    from pilot_eval.sft_backend import frozen_weight_hash
    model, tokenizer = tiny_model(task, seed)
    window = GRPOWindow(model, tokenizer, output_dir=directory / 'trainer',
                        microbatch_groups=task['microbatch_groups'], seed=seed, completion_limit=task['horizon'])
    _save_frozen(directory / 'resolved-trainer-args.json', window.resolved_args)
    model.eval()
    with torch.no_grad():
        probability = float(model(input_ids=torch.tensor([task['prompt_tokens']])).logits[0, -1].softmax(-1)[1])
    low, high = task['initial_reward_probability_range']
    if not low <= probability <= high:
        raise ValueError('frozen toy initial policy has insufficient learning headroom')
    initial_hash = frozen_weight_hash(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=task['learning_rate'], betas=tuple(task['betas']),
                                 eps=task['epsilon'], weight_decay=task['weight_decay'])
    rows, draws, previous = [], [], None
    model.save_pretrained(directory / 'initial-adapter')
    prompts = [task['prompt_tokens']] * 8
    for step in range(1, task['steps'] + 1):
        model.eval()
        torch.manual_seed(seed + 1000 + step)  # paired controls share each fresh draw stream
        prompt_ids = torch.tensor([task['prompt_tokens']] * 64)
        with torch.no_grad():
            generated = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=task['horizon'], do_sample=True, temperature=task['temperature'],
                top_p=task['top_p'], top_k=task['top_k'], num_beams=1, num_return_sequences=1,
                repetition_penalty=1.0, no_repeat_ngram_size=0, eos_token_id=None, pad_token_id=0)
        completions = generated[:, prompt_ids.shape[1]:].tolist()
        rewards = [int(ids == [1]) for ids in completions]
        optimizer.zero_grad(set_to_none=True)
        loss = window.backward(prompts, completions, rewards, capped=[False] * 64, sign=sign)
        gradient = window.flat_gradient()
        before = torch.cat([p.detach().reshape(-1).clone() for p in parameters])
        consistency = gradient_consistency(previous, gradient)
        torch.nn.utils.clip_grad_norm_(parameters, task['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        movement = float((torch.cat([p.detach().reshape(-1) for p in parameters]) - before).norm())
        rows.append(dict(step=step, reward=float(np.mean(rewards)), loss=loss,
                         preclip_gradient_norm=float(gradient.norm()), parameter_movement=movement,
                         dead_group_fraction=window.dead_group_fraction, gradient_consistency=consistency))
        for index, (ids, reward) in enumerate(zip(completions, rewards)):
            draws.append(dict(step=step, group_id=f'toy:{index // 8}', draw=index % 8,
                              token_ids=ids, reward=reward, complete_at_task_horizon=True))
        previous = gradient.clone()
    final_hash = frozen_weight_hash(model)
    if initial_hash != final_hash:
        raise ValueError('learning control modified frozen base weights')
    early = float(np.mean([r['reward'] for r in rows[:5]]))
    late = float(np.mean([r['reward'] for r in rows[15:20]]))
    change = sign * (late - probability)
    summary = dict(seed=seed, sign=sign, initial_probability=probability, early_reward=early,
                   late_reward=late, baseline_reward=probability,
                   comparison_baseline=task['criterion']['baseline'], signed_change=change, passed=change >= task['criterion']['minimum_change'],
                   frozen_base_unchanged=True, base_sha256=initial_hash, parameter_names=window.parameter_names)
    model.save_pretrained(directory / 'final-adapter')
    _write_json(directory / 'steps.json', rows)
    for filename, records in [('responses.jsonl', draws)]:
        temp = directory / (filename + '.tmp')
        temp.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
        temp.replace(directory / filename)
    _write_json(directory / 'results.json', summary)
    return summary


def run_controls(output_root, name):
    """Freeze task/config before the four acceptance runs; reuse verified results."""
    from pilot_eval.sft_backend import PINS
    from pilot_eval import grpo_algorithm
    root = Path(output_root).resolve()
    directory = root / 'runs/diagnostics/pilot-4' / safe_name(name)
    task = json.loads(TASK_PATH.read_text())
    config = dict(scope='disposable-control-only', task=task, task_sha256=file_hash(TASK_PATH),
                  algorithm_sha256=file_hash(Path(grpo_algorithm.__file__)),
                  controls_sha256=file_hash(Path(__file__)), dependencies=PINS, device='cpu',
                  acceptance='both signs in both seeds; no settings/criterion retuning after failure')
    with run_lock(directory):
        _save_frozen(directory / 'config.json', config)
        marker = directory / 'complete.json'
        if marker.exists():
            seal = json.loads(marker.read_text())
            if seal['config_sha256'] != _hash(config):
                raise ValueError('control configuration changed')
            for relative, digest in seal['files'].items():
                if file_hash(directory / relative) != digest:
                    raise ValueError('control evidence hash mismatch')
            return json.loads((directory / 'results/results.json').read_text())
        try:
            versions = {k: importlib.metadata.version(k) for k in PINS}
            if any(versions[k].split('+')[0] != v for k, v in PINS.items()):
                raise ValueError('controls require the pinned environment; earlier environments are not modified')
            _save_frozen(directory / 'meta/runtime.json', dict(versions=versions, device='cpu'))
            _write_state(directory, 'running', 0, 4)
            runs = []
            with isolated_rng():
                for seed in task['seeds']:
                    for sign in [1, -1]:
                        print(f'controls: seed {seed}, advantage sign {sign}', flush=True)
                        unit = directory / f'seed-{seed}-sign-{sign}'
                        unit.mkdir(parents=True, exist_ok=True)
                        runs.append(_learning_run(unit, task, seed, sign))
                        _write_state(directory, 'running', len(runs), 4)
            result = dict(passed=all(r['passed'] for r in runs), runs=runs, scope=config['scope'],
                          scientific_training_steps=0,
                          compatibility='Pinned TRL DAPO via GRPOTrainer.compute_loss; direct summed backward, no second accumulation scaling.')
            _write_json(directory / 'results/results.json', result)
            if not result['passed']:
                raise ValueError('frozen learning-control criterion failed; diagnose without retuning')
            files = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob('*')
                     if p.is_file() and p.name not in ['complete.json', '.lock'] and '/logs/' not in str(p)
                     and p.parent.name != 'meta'}
            files['meta/runtime.json'] = file_hash(directory / 'meta/runtime.json')
            _write_json(marker, dict(config_sha256=_hash(config), files=files))
            _write_state(directory, 'completed', 4, 4)
            return result
        except BaseException as exc:
            _write_state(directory, 'failed', 0, 4, str(exc))
            raise
