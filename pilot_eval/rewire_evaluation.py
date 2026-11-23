"""Sparse rewiring ticket 09: evaluation wiring (validation CE, greedy validation, greedy cohort150).

Reuses:
- ``sft_backend.completion_loss`` for token-weighted validation CE with deterministic masks.
- ``c_attribution_generation.batches`` / ``greedy`` / ``score`` (``scoring.score_gsm8k_flexible_v3``).
- ``attribution_units.seal_run`` / ``verify_run`` for sealed evaluation units and prior-use records.
- ``rewire_module`` for installing/loading rewire checkpoints (``sft_evaluation`` only loads PEFT 0-64).

Contracts enforced:
- Spec test 22: pad and EOS token IDs come from the tokenizer files; a model config that supplies
  ``pad_token_id`` is rejected as unrepresentative of Qwen2.5-1.5B-Instruct.
- D12: cohort150 evaluation is only permitted at step 64 and the final step, labelled exploratory,
  and prior uses are recorded strictly from verified sealed units.
- Lesson 2026-11-19: gold-sft step 0 and 64 validation references run in a separate child process
  so no reference model stays resident alongside rank-0 training.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import rewire_module as rm
from pilot_eval import sft_backend as sb

CHECKPOINT_WEIGHTS_FILE: str = 'rewire_state.safetensors'
CHECKPOINT_META_FILE: str = 'checkpoint_meta.json'
CHECKPOINT_SCHEMA: str = 'sparse-rewiring-checkpoint-v1'
EVAL_CONFIG_SCHEMA: str = 'sparse-rewiring-eval-v1'
COHORT150_NAME: str = 'gsm8k-test-cohort150'
DEFAULT_FINAL_STEP: int = 1024


def resolve_token_ids(tokenizer: Any, *, model_config: Any = None) -> dict[str, int]:
    """Read pad and EOS token IDs from the tokenizer; reject model configs that define ``pad_token_id`` (test 22)."""
    if model_config is not None and getattr(model_config, 'pad_token_id', None) is not None:
        raise ValueError(
            'model_config defines pad_token_id (unrepresentative of Qwen2.5-1.5B-Instruct; '
            'pad and EOS IDs must come from the pinned tokenizer files)'
        )
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if pad_id is None:
        raise ValueError('tokenizer does not define pad_token_id')
    if eos_id is None:
        raise ValueError('tokenizer does not define eos_token_id')
    return {'pad_token_id': int(pad_id), 'eos_token_id': int(eos_id)}


# --- D24 rewire checkpoint save / load (factors + log_alpha + seal only) ---------------------------

def save_rewire_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    *,
    step: int,
    rank: int,
    plan_sha256: str,
    runtime_sha256: str,
    code_sha256: str,
    frozen: bool = False,
) -> str:
    """Save and seal a D24 checkpoint: ONLY ``rewire_`` factors + ``rewire_log_alpha`` + seal (no optimizer/RNG)."""
    from safetensors.torch import save as st_save

    params = rm.trainable_parameters(model)
    if not params:
        raise ValueError('model has no rewire parameters to save')
    tensors = {name: param.detach().cpu().contiguous() for name, param in params}
    meta = {
        'schema': CHECKPOINT_SCHEMA,
        'step': int(step),
        'rank': int(rank),
        'frozen': bool(frozen),
        'plan_sha256': str(plan_sha256),
        'runtime_sha256': str(runtime_sha256),
        'code_sha256': str(code_sha256),
        'parameter_names': sorted(tensors),
    }
    return au.seal_run(
        Path(directory),
        {
            CHECKPOINT_WEIGHTS_FILE: st_save(tensors),
            CHECKPOINT_META_FILE: meta,
        },
    )


def load_rewire_checkpoint(
    model: torch.nn.Module,
    directory: str | Path,
    *,
    expected_plan_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
    expected_code_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify seal, bind hashes, install rewire modules if needed, load factors/masks, and set deterministic gate."""
    from safetensors.torch import load_file as st_load_file

    ckpt_dir = Path(directory)
    au.verify_run(ckpt_dir)
    meta = json.loads((ckpt_dir / CHECKPOINT_META_FILE).read_text(encoding='utf-8'))
    if meta.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError(f'unexpected checkpoint schema {meta.get("schema")!r}')
    for key, expected in (
        ('plan_sha256', expected_plan_sha256),
        ('runtime_sha256', expected_runtime_sha256),
        ('code_sha256', expected_code_sha256),
    ):
        if expected is not None and meta.get(key) != expected:
            raise ValueError(f'checkpoint seal mismatch for {key}: got {meta.get(key)!r}, expected {expected!r}')

    rank = int(meta['rank'])
    try:
        mods = rm.modules(model)
    except ValueError:
        mods = rm.install(model, rank=rank)
    if any(m.rank != rank for m in mods):
        raise ValueError(f'installed rewire rank does not match checkpoint rank {rank}')

    state = st_load_file(str(ckpt_dir / CHECKPOINT_WEIGHTS_FILE), device='cpu')
    named = dict(rm.trainable_parameters(model))
    if set(state) != set(named):
        missing = sorted(set(named) - set(state))
        extra = sorted(set(state) - set(named))
        raise ValueError(f'checkpoint keys mismatch: missing={missing[:5]}, extra={extra[:5]}')

    with torch.no_grad():
        for name, param in named.items():
            src = state[name]
            if tuple(src.shape) != tuple(param.shape):
                raise ValueError(f'shape mismatch for {name}: {tuple(src.shape)} vs {tuple(param.shape)}')
            param.copy_(src.to(device=param.device, dtype=param.dtype))
            if meta.get('frozen'):
                if name.endswith('rewire_log_alpha'):
                    param.requires_grad_(False)

    # Always restore deterministic gate for evaluation
    rm.set_gate(model, None)
    return meta


# --- Validation CE (token-weighted, deterministic masks) -------------------------------------------

def evaluate_validation_ce(
    model: torch.nn.Module,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute exact token-weighted validation CE with deterministic masks using ``sft_backend.completion_loss``."""
    if not items:
        raise ValueError('validation items must not be empty')
    was_training = model.training
    prev_gates = None
    try:
        mods = rm.modules(model)
        prev_gates = [m.gate for m in mods]
        rm.set_gate(model, None)
    except ValueError:
        mods = None

    device = next(model.parameters()).device
    total_tokens = 0
    for item in items:
        labels = item['labels']
        total_tokens += sum(1 for tok in labels[1:] if int(tok) != -100)
    if total_tokens <= 0:
        raise ValueError('validation set has zero supervised completion tokens')

    model.eval()
    ce_sum = 0.0
    try:
        with torch.no_grad():
            for item in items:
                input_ids = torch.tensor([item['input_ids']], dtype=torch.long, device=device)
                labels = torch.tensor([item['labels']], dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids, use_cache=False)
                loss = sb.completion_loss(outputs, labels, num_items_in_batch=total_tokens)
                ce_sum += float(loss.item())
    finally:
        if mods is not None and prev_gates is not None:
            for m, g in zip(mods, prev_gates):
                m.gate = g
        model.train(was_training)

    return {
        'val_ce': float(ce_sum),
        'num_examples': len(items),
        'total_supervised_tokens': int(total_tokens),
    }


# --- Greedy evaluation (validation or cohort150) ---------------------------------------------------

def check_cohort_step_allowed(cohort: str, step: int, *, final_step: int = DEFAULT_FINAL_STEP) -> None:
    """Enforce D12: cohort150 may only be evaluated at step 64 and the final step."""
    if cohort == COHORT150_NAME and int(step) not in {64, int(final_step)}:
        raise ValueError(
            f'cohort150 evaluation is restricted to step 64 and final step ({final_step}) per D12; got step {step}'
        )


def evaluate_greedy(
    model: torch.nn.Module,
    questions: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    out_dir: str | Path,
    *,
    arm: str,
    step: int,
    cohort: str,
    decode: Callable[[Sequence[int]], str],
    pad_token_id: int,
    eos_token_id: int,
    cap: int = cg.CAP,
    batch_size: int = 16,
    final_step: int = DEFAULT_FINAL_STEP,
    checkpoint_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic-gate greedy evaluation, seal ``responses.jsonl``/``config.json``/``scores.json``,
    and return a verified summary + prior-use record."""
    check_cohort_step_allowed(cohort, step, final_step=final_step)
    ids = list(question_ids)
    plan = cg.batches(ids, int(batch_size))

    was_training = model.training
    prev_gates = None
    try:
        mods = rm.modules(model)
        prev_gates = [m.gate for m in mods]
        rm.set_gate(model, None)
    except ValueError:
        mods = None

    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for batch_qids in plan:
            prompts = [questions[q]['prompt_ids'] for q in batch_qids]
            gens = cg.greedy(
                model,
                prompts,
                pad_token_id=int(pad_token_id),
                eos_token_id=int(eos_token_id),
                cap=int(cap),
            )
            for qid, gen in zip(batch_qids, gens):
                q = questions[qid]
                text = decode(gen['content_token_ids'])
                capped = gen['stop_reason'] == 'cap'
                scored = cg.score(text, q['gold'], capped=capped)
                rows.append(
                    {
                        'question_id': qid,
                        'gold': q['gold'],
                        'text': text,
                        'generated_token_ids': gen['generated_token_ids'],
                        'stop_reason': gen['stop_reason'],
                        'capped': capped,
                        'n_tokens': len(gen['generated_token_ids']),
                        'reward': scored['reward'],
                        'invalid': scored['invalid'],
                        'flexible': scored['flexible'],
                    }
                )
    finally:
        if mods is not None and prev_gates is not None:
            for m, g in zip(mods, prev_gates):
                m.gate = g
        model.train(was_training)

    rewards = np.asarray([r['reward'] for r in rows], dtype=np.float64)
    tokens = np.asarray([r['n_tokens'] for r in rows], dtype=np.float64)
    scores = {
        'arm': arm,
        'step': int(step),
        'cohort': cohort,
        'exploratory': cohort == COHORT150_NAME,
        'questions': len(rows),
        'correct': int(rewards.sum()),
        'accuracy': float(rewards.mean()),
        'invalid': int(sum(1 for r in rows if r['invalid'])),
        'capped': int(sum(1 for r in rows if r['capped'])),
        'tokens_mean': float(tokens.mean()),
        'tokens_median': float(np.median(tokens)),
    }
    config = {
        'schema': EVAL_CONFIG_SCHEMA,
        'arm': arm,
        'step': int(step),
        'cohort': cohort,
        'exploratory': cohort == COHORT150_NAME,
        'question_ids': ids,
        'pad_token_id': int(pad_token_id),
        'eos_token_id': int(eos_token_id),
        'cap': int(cap),
        'batch_size': int(batch_size),
        'greedy_args': dict(cg.GREEDY_ARGS),
        'scorer': 'gsm8k_flexible_v3',
        'checkpoint_meta': dict(checkpoint_meta) if checkpoint_meta is not None else None,
    }
    jsonl_text = ''.join(json.dumps(r, sort_keys=True, default=au._json_default) + '\n' for r in rows)

    out_path = Path(out_dir)
    seal_status = au.seal_run(
        out_path,
        {
            'responses.jsonl': jsonl_text,
            'config.json': config,
            'scores.json': scores,
        },
    )
    prior_use = prior_use_from_sealed_eval(out_path)
    return {
        'seal_status': seal_status,
        'directory': str(out_path),
        'scores': scores,
        'prior_use': prior_use,
        'responses': {r['question_id']: r for r in rows},
    }


def prior_use_from_sealed_eval(eval_dir: str | Path) -> dict[str, Any]:
    """Derive a catalog prior-use record strictly from a hash-verified sealed evaluation unit (lesson 2026-11-22)."""
    directory = Path(eval_dir)
    au.verify_run(directory)
    cfg = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    seal = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
    return {
        'cohort': cfg['cohort'],
        'arm': cfg['arm'],
        'step': int(cfg['step']),
        'exploratory': bool(cfg['exploratory']),
        'questions': len(cfg['question_ids']),
        'unit_dir': str(directory),
        'responses_sha256': seal['files']['responses.jsonl'],
        'config_sha256': seal['files']['config.json'],
    }


# --- Separate child process for gold-sft references (lesson 2026-11-19) ----------------------------

def run_in_child_process(
    module_name: str,
    func_name: str,
    payload: Mapping[str, Any],
    *,
    python_executable: str = sys.executable,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``module_name.func_name(payload)`` in an isolated child process so GPU/CPU models are fully freed on exit."""
    code = (
        'import importlib, json, sys\n'
        'req = json.loads(sys.stdin.read())\n'
        'mod = importlib.import_module(req["module"])\n'
        'res = getattr(mod, req["func"])(req["payload"])\n'
        'sys.stdout.write(json.dumps(res, sort_keys=True))\n'
    )
    proc = subprocess.run(
        [python_executable, '-c', code],
        input=json.dumps({'module': module_name, 'func': func_name, 'payload': dict(payload)}),
        text=True,
        capture_output=True,
        env=dict(env) if env is not None else None,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'child process {module_name}.{func_name} failed (rc={proc.returncode}): {proc.stderr}')
    return json.loads(proc.stdout)
