"""Sparse rewiring ticket 09: evaluation wiring (validation CE, greedy validation, greedy cohort150).

Reuses:
- ``rewire_checkpoint.load_into`` (PR #16, ticket 08) for verified D24 checkpoint loading.
- ``sft_backend.completion_loss`` for token-weighted validation CE with deterministic masks.
- ``c_attribution_generation.batches`` / ``greedy`` / ``score`` (``scoring.score_gsm8k_flexible_v3``).
- ``attribution_units.seal_run`` / ``verify_run`` for sealed evaluation units and prior-use records.
- ``neuron_audit_collector.set_ieee_precision`` / ``TransformersBackend`` inside child processes.

Contracts enforced:
- Spec test 22: pad and EOS token IDs come from the tokenizer files; a model config that supplies
  ``pad_token_id`` is rejected as unrepresentative of Qwen2.5-1.5B-Instruct.
- D12: cohort150 evaluation is only permitted at step 64 and the final step, labelled exploratory,
  and prior uses are recorded strictly from verified sealed units.
- Lesson 2026-11-19 / 2026-11-20 / ADR 0021: gold-sft step 0 and 64 validation references run in
  a separate child process that sets and reads back IEEE FP32 precision (refusing TF32) and
  matches the parent's runtime fingerprint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_module as rm
from pilot_eval import sft_backend as sb
from pilot_eval.workflow import _hash

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


# --- Verified D24 checkpoint loading for evaluation (delegates to ticket 08 / rewire_checkpoint) ---

def load_checkpoint_for_eval(
    model: torch.nn.Module,
    folder: str | Path,
    plan: Mapping[str, Any],
    *,
    runtime_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    """Load a verified ticket-08 checkpoint into ``model`` and restore the deterministic eval gate.

    Requires both ``runtime_sha256`` and ``code_commit`` (never ``None``) so an unbound checkpoint
    can never be evaluated silently (lesson 2026-11-19).
    """
    if not isinstance(runtime_sha256, str) or not runtime_sha256:
        raise ValueError('runtime_sha256 is required when loading a checkpoint for evaluation')
    if not isinstance(code_commit, str) or not code_commit or code_commit == 'unknown':
        raise ValueError('a real code_commit is required when loading a checkpoint for evaluation')
    rank = int(plan['rank'])
    try:
        mods = rm.modules(model)
    except ValueError:
        mods = rm.install(model, rank=rank)
    if any(m.rank != rank for m in mods):
        raise ValueError(f'installed rewire rank does not match plan rank {rank}')
    seal = rckpt.load_into(model, folder, plan, runtime_sha256=runtime_sha256, code_commit=code_commit)
    rm.set_gate(model, None)
    return seal


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
    strict_eos: bool = False,
    checkpoint_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic-gate greedy evaluation, seal ``responses.jsonl``/``config.json``/``scores.json``,
    and return a verified summary + prior-use record."""
    if strict_eos and int(eos_token_id) != cg.EOS:
        raise ValueError(f'eos_token_id {eos_token_id} differs from pinned protocol EOS {cg.EOS}')
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


# --- Separate child process for gold-sft references (lessons 2026-11-19 / 2026-11-20) --------------

def _precision_problems(readback: Mapping[str, Any]) -> list[str]:
    return [
        f'child IEEE FP32 precision not in effect: {k}={v}'
        for k, v in sorted(readback.items())
        if v != ('highest' if k == 'matmul_precision' else 'ieee')
    ]


def run_in_child_process(
    module_name: str,
    func_name: str,
    payload: Mapping[str, Any],
    *,
    python_executable: str = sys.executable,
    env: Mapping[str, str] | None = None,
    expected_runtime_sha256: str | None = None,
    require_ieee: bool = True,
) -> dict[str, Any]:
    """Run ``module_name.func_name(payload)`` in an isolated child process with IEEE FP32 precision enforced.

    The child sets ``neuron_audit_collector.set_ieee_precision()`` before calling the function and
    returns its ``precision_readback`` alongside the function's output. The parent refuses unless
    every precision backend reads ``'ieee'`` / ``'highest'`` and (when supplied) the child's
    ``runtime_sha256`` equals the parent's.
    """
    child_env = dict(os.environ if env is None else env)
    child_env.setdefault('HF_HUB_OFFLINE', '1')
    child_env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    code = (
        'import importlib, json, sys, torch\n'
        'from pilot_eval import neuron_audit_collector as nac\n'
        'req = json.loads(sys.stdin.read())\n'
        'precision = None\n'
        'if req["require_ieee"]:\n'
        '    nac.set_ieee_precision()\n'
        '    precision = dict(nac.precision_backends(), matmul_precision=nac._legacy_matmul_precision())\n'
        'mod = importlib.import_module(req["module"])\n'
        'res = getattr(mod, req["func"])(req["payload"])\n'
        'out = {"result": res, "precision_readback": precision}\n'
        'sys.stdout.write(json.dumps(out, sort_keys=True))\n'
    )
    proc = subprocess.run(
        [python_executable, '-c', code],
        input=json.dumps(
            {'module': module_name, 'func': func_name, 'payload': dict(payload), 'require_ieee': bool(require_ieee)}
        ),
        text=True,
        capture_output=True,
        env=child_env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'child process {module_name}.{func_name} failed (rc={proc.returncode}): {proc.stderr}')
    envelope = json.loads(proc.stdout)
    if require_ieee:
        readback = envelope.get('precision_readback') or {}
        problems = _precision_problems(readback)
        if not readback or problems:
            raise RuntimeError('child process precision check failed: ' + '; '.join(problems or ['empty readback']))
    res = envelope['result']
    if expected_runtime_sha256 is not None:
        child_rt = res.get('runtime_sha256') if isinstance(res, Mapping) else None
        if child_rt != expected_runtime_sha256:
            raise RuntimeError(
                f'child runtime_sha256 {child_rt!r} differs from parent {expected_runtime_sha256!r}'
            )
    return res


def _gold_reference_child_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Child-process entrypoint: build a ``TransformersBackend`` (sets IEEE), evaluate untuned (step 0)
    and ``gold-sft`` (step 64) on validation CE + greedy validation, and seal outputs."""
    from pilot_eval import c_attribution_backend as cab

    record = payload['record']
    volume_root = payload['volume_root']
    device = payload.get('device', 'cuda')
    out_dir = Path(payload['out_dir'])
    val_items = payload['val_items']
    val_questions = payload['val_questions']
    val_question_ids = payload['val_question_ids']
    cap = int(payload['cap'])
    batch_size = int(payload['batch_size'])

    adapted = dict(record)
    if 'generation' not in adapted and 'evaluation' in adapted:
        adapted['generation'] = {'eos_token_id': int(adapted['evaluation']['eos_token_id'])}
    backend = cab.TransformersBackend(adapted, volume_root, device=device)
    probs = backend.precision_problems()
    if probs:
        raise RuntimeError('; '.join(probs))
    rt_info = backend.runtime()
    rt_sha = _hash(rt_info)

    results: dict[str, Any] = {}
    prior_uses: list[dict[str, Any]] = []
    for ref_arm, ref_step in (('untuned', 0), ('gold-sft', 64)):
        if ref_arm == 'untuned':
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                backend.model_dir, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager'
            ).to(backend.device)
        else:
            model = backend.load(ref_arm)
        try:
            ce = evaluate_validation_ce(model, val_items)
            greedy_dir = out_dir / f'{ref_arm}-step-{ref_step}' / 'greedy-val'
            greedy = evaluate_greedy(
                model,
                val_questions,
                val_question_ids,
                greedy_dir,
                arm=ref_arm,
                step=ref_step,
                cohort='gsm8k-train-val128',
                decode=backend.decode,
                pad_token_id=backend.pad_token_id,
                eos_token_id=backend.eos_token_id,
                cap=cap,
                batch_size=batch_size,
            )
            results[ref_arm] = {
                'step': ref_step,
                'val_ce': ce['val_ce'],
                'val_accuracy': greedy['scores']['accuracy'],
                'eval_dir': greedy['directory'],
            }
            prior_uses.append(greedy['prior_use'])
        finally:
            model = None
            backend.release()

    au.seal_run(
        out_dir / 'summary',
        {'references.json': results, 'prior_uses.json': prior_uses, 'runtime_sha256': rt_sha},
    )
    return {
        'references': results,
        'prior_uses': prior_uses,
        'runtime_sha256': rt_sha,
        'precision_readback': backend.precision_readback(),
    }


def evaluate_gold_references_in_child(
    *,
    record: Mapping[str, Any],
    volume_root: str | Path,
    out_dir: str | Path,
    val_items: Sequence[Mapping[str, Any]],
    val_questions: Mapping[str, Mapping[str, Any]],
    val_question_ids: Sequence[str],
    expected_runtime_sha256: str,
    device: str = 'cuda',
    cap: int = cg.CAP,
    batch_size: int = 16,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run untuned (step 0) and gold-sft (step 64) validation references in a fresh IEEE-verified child process."""
    return run_in_child_process(
        'pilot_eval.rewire_evaluation',
        '_gold_reference_child_worker',
        {
            'record': dict(record),
            'volume_root': str(volume_root),
            'out_dir': str(out_dir),
            'val_items': list(val_items),
            'val_questions': dict(val_questions),
            'val_question_ids': list(val_question_ids),
            'device': device,
            'cap': int(cap),
            'batch_size': int(batch_size),
        },
        env=env,
        expected_runtime_sha256=expected_runtime_sha256,
        require_ieee=True,
    )
