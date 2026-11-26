"""Sparse rewiring ticket 15: GPU qualification stage for ``c_attribution_chain.Chain``.

Runs on the real model before any scientific step and returns the per-check booleans the chain's automatic
sign-off reads (``Chain._signoff``: every check ``True``, no review flag, every arm present). Nothing here is
tuned at run time: the tolerances and sizes are the launch record's ``qualification`` block. An OOM or any other
exception stops the chain (ADR 0004: no automatic batch or precision fallback).

Per rank (``sparse-gold-sft-r1`` then ``sparse-gold-sft-r0``):

- ``zero_update``: the freshly installed rewired model (B = 0 / W = 0) gives logits equal, bit for bit, to the
  untuned model on the longest validation prompt (``rewire_module.zero_change``: modules on vs ``off``), and the
  frozen base-weight hash equals the expected base hash.
- ``memory``: one real training step on the **longest** gold training row (microbatch 1, gradient checkpointing,
  the D14 shared-mask gate, L0 term, read-only clipping, ``AdamW(foreach=False)`` with its state allocated):
  peak CUDA memory <= ``memory_fraction`` x device memory.
- ``recompute``: on that same model (now with nonzero reads), the gradients of one row with the cached ``u`` are
  the same with and without gradient-checkpoint recompute (max abs difference / max abs gradient <=
  ``recompute_rel_max``). This is the shared-mask cache check of spec section "FP32 and qualification".
- ``disposable_run``: the first ``disposable_steps`` steps of the arm's frozen plan under a separate run ID
  (``<run_id>-qualification``), checkpoint 0 and the last step, through ``rewire_checkpoint.run_arm`` and the real
  training loop; then the same in a fresh directory. ``restart_reproduces``: identical step logs (every logged
  number) and identical checkpoint tensors. ``checkpoint_reload``: the last checkpoint loads into a fresh base model
  through the evaluation loader and its tensors equal the saved ones. Throughput (seconds per step) is measured.
- ``cached_vs_forced``: on the reloaded, trained model, left-padded KV-cache decoding of ``cached_vs_forced_rows``
  validation rows against full forcing (``cached_vs_forced`` here: the C check's method, whose forced pass needs a
  PEFT model), max per-token |Δ log p| <=
  ``cached_vs_forced_abs_max``. Greedy evaluation decodes with the cache, so this binds it to training's forcing.
- one timed validation CE pass and one timed greedy pass on ``greedy_questions`` validation questions (measured,
  not gated): they size the production evaluation cost. Validation only; cohort150 is never touched here.

Once for the run: ``disk`` (free container disk >= the D24 planned bytes + ``min_free_gb``) and ``precision``
(IEEE readback still in effect at the end).

Not run, recorded with the reason: ``noop_check`` (it tests an attribution intervention context; this method has
none) and ``processor_check`` (it resolves the sampled-generation processors; greedy evaluation passes explicit
``GREEDY_ARGS``). The disposable runs live under ``inputs.work_dir`` (not the exported subtree) and are deleted
after their hashes are recorded, so rank 0's 3 GB checkpoints never reach the D24 budget.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au
from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_training as rt
from pilot_eval import sft_backend as sb
from pilot_eval import sparse_rewiring_plan as sp

NOT_RUN = {
    'noop_check': 'not applicable: it checks an attribution intervention context with an empty removal set; '
                  'sparse rewiring trains reads and has no intervention context',
    'processor_check': 'not applicable: it resolves the sampled-generation logits processors; greedy evaluation '
                       'passes explicit GREEDY_ARGS and no sampling runs',
}
# Step-log fields that are timings, not results; ``restart_reproduces`` compares every other field.
TIMING_KEYS = ('seconds', 'time', 'unix', 'elapsed', 'wall')


def disposable_plan(plan: Mapping[str, Any], steps: int) -> dict[str, Any]:
    """The arm's frozen plan cut to its first ``steps`` steps, under a separate run ID, with checkpoints at 0 and the
    last step. Every per-step entry (IDs, mask groups, LR multiplier, mask phase) is the frozen plan's own."""
    steps = int(steps)
    if not 0 < steps <= len(plan['steps']):
        raise ValueError(f'disposable run of {steps} steps does not fit the plan ({len(plan["steps"])} steps)')
    cut = json.loads(json.dumps(plan))
    cut.pop('plan_sha256', None)
    cut['run_id'] = f"{plan['run_id']}-qualification"
    cut['steps'] = cut['steps'][:steps]
    cut['checkpoints'] = [0, steps]
    cut['disposable_of'] = plan['plan_sha256']
    cut['plan_sha256'] = sp.plan_sha256(cut)
    return cut


def _without_timings(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not any(t in k for t in TIMING_KEYS)}


def _step_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / 'logs' / 'steps.jsonl'
    return [_without_timings(json.loads(line)) for line in path.read_text().splitlines() if line]


def _checkpoint_hash(run_dir: Path, step: int) -> str:
    seal = json.loads((rckpt.checkpoint_dir(run_dir, step) / rckpt.SEAL_FILE).read_text())
    return seal['files'][rckpt.STATE_FILE]


def _spread(items: Sequence[Mapping[str, Any]], n: int) -> list[Mapping[str, Any]]:
    """``n`` items spread over length, always including the longest (the pattern of ``Stages._qualification_rows``)."""
    ordered = sorted(items, key=lambda r: (len(r['input_ids']), r['id']))
    if n >= len(ordered):
        return ordered
    picks = {round(i * (len(ordered) - 1) / max(n - 1, 1)) for i in range(n)} | {len(ordered) - 1}
    return [ordered[i] for i in sorted(picks)][-n:]


def cached_vs_forced(model, items, *, pad_token_id):
    """Left-padded prefill plus incremental KV-cache decoding of each item's completion tokens, against one
    left-padded forced forward of the whole sequence (the comparison of ``c_attribution_checks.cached_vs_forced``,
    whose forced pass needs a PEFT model; this one runs on the plain rewired model). Deterministic gates."""
    import numpy as np
    device = next(model.parameters()).device
    prev = [m.gate for m in rm.modules(model)]
    rm.set_gate(model, None)
    was_training = model.training
    model.eval()
    try:
        prompts = [list(i['input_ids'][:int(i['prompt_tokens'])]) for i in items]
        generated = [list(i['input_ids'][int(i['prompt_tokens']):]) for i in items]
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[pad_token_id] * (width - len(p)) + p for p in prompts], device=device)
        attention = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
        cached = [[] for _ in items]
        steps = max(len(g) for g in generated)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attention,
                        position_ids=(attention.cumsum(1) - 1).clamp(min=0), use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
            for t in range(steps):
                logp = torch.log_softmax(logits.float(), -1)
                tokens = []
                for r, g in enumerate(generated):
                    if t < len(g):
                        cached[r].append(float(logp[r, g[t]]))
                    tokens.append(g[t] if t < len(g) else pad_token_id)
                if t == steps - 1:
                    break
                live = torch.tensor([[1 if t < len(g) else 0] for g in generated], device=device)
                attention = torch.cat([attention, live], 1)
                out = model(input_ids=torch.tensor(tokens, device=device)[:, None], attention_mask=attention,
                            position_ids=(attention.sum(1, keepdim=True) - 1), past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[:, -1]
            full = [list(i['input_ids']) for i in items]
            fwidth = max(len(f) for f in full)
            fids = torch.tensor([[pad_token_id] * (fwidth - len(f)) + f for f in full], device=device)
            fatt = torch.tensor([[0] * (fwidth - len(f)) + [1] * len(f) for f in full], device=device)
            flogits = model(input_ids=fids, attention_mask=fatt, position_ids=(fatt.cumsum(1) - 1).clamp(min=0),
                            use_cache=False).logits
        per_token, per_row = [], []
        for r, item in enumerate(items):
            offset = fwidth - len(full[r])
            p = int(item['prompt_tokens'])
            logp = torch.log_softmax(flogits[r].float(), -1)
            forced = [float(logp[offset + p + k - 1, tok]) for k, tok in enumerate(generated[r])]
            diffs = np.abs(np.array(forced) - np.array(cached[r]))
            per_token.extend(diffs.tolist())
            per_row.append(dict(id=item['id'], generated_length=len(generated[r]), max_token_abs=float(diffs.max()),
                                abs_delta_M=abs(float(np.sum(forced)) - float(np.sum(cached[r])))))
    finally:
        for m, g in zip(rm.modules(model), prev):
            m.gate = g
        model.train(was_training)
    tokens = np.array(per_token)
    return {'per_token_abs': {'max': float(tokens.max()), 'median': float(np.median(tokens)),
                              'p99': float(np.percentile(tokens, 99))},
            'per_row': per_row, 'rows': [i['id'] for i in items]}


def _forward_backward(model, row, *, collator, device, denominator):
    values = collator([row])
    values.pop('example_ids', None)
    labels = values.pop('labels').to(device)
    output = model(**{k: v.to(device) for k, v in values.items()}, use_cache=False)
    loss = sb.completion_loss(output, labels, denominator)
    loss.backward()
    return float(loss.detach().item())


def _supervised(row) -> int:
    return sum(1 for v in row['labels'][1:] if int(v) != -100)


class Qualification:
    """The ``qualification`` stage body. ``train_arm(model, plan, sink, out_dir)`` is the chain's own call of the real
    training loop (same settings), so the disposable run exercises exactly what the scientific run will."""

    def __init__(self, *, record: Mapping[str, Any], backend: Any, arm_plans: Mapping[str, Mapping[str, Any]],
                 training_rows: Sequence[Mapping[str, Any]], val_items: Sequence[Mapping[str, Any]],
                 val_questions: Mapping[str, Mapping[str, Any]], val_question_ids: Sequence[str],
                 arm_order: Sequence[str], train_arm: Callable[..., Any],
                 expected_base_sha256: Callable[[], str], runtime_sha256: Callable[[], str], code_commit: str,
                 out_rel: str, gradient_checkpointing: bool = True) -> None:
        self.record, self.backend, self.arm_plans = record, backend, arm_plans
        self.training_rows, self.val_items = list(training_rows), list(val_items)
        self.val_questions, self.val_question_ids = val_questions, list(val_question_ids)
        self.arm_order, self.train_arm = list(arm_order), train_arm
        self.expected_base_sha256, self.runtime_sha256 = expected_base_sha256, runtime_sha256
        self.code_commit, self.out_rel = code_commit, out_rel
        self.out_dir: Path | None = None  # local_root / out_rel, set when the stage runs
        self.gradient_checkpointing = gradient_checkpointing
        self.q = record['qualification']
        self.scratch = Path(record['inputs']['work_dir']) / 'qualification-scratch'

    # --- helpers --------------------------------------------------------------------------------

    def _rewired(self, plan):
        plan_seeds = plan.get('settings', {}).get('seeds', {})
        arm_seed = int(
            plan_seeds['read_init_base'] if 'read_init_base' in plan_seeds
            else self.record['seeds']['read_init_base']
        )
        return self.backend.load_rewired_model(rank=int(plan['rank']), seed=arm_seed,
                                               mask_init=float(self.record['hard_concrete']['log_alpha_init']))

    def _release(self):
        self.backend.release()

    def _peak(self, fn):
        self.backend.reset_peak()
        fn()
        return int(self.backend.peak_memory() or 0)

    # --- checks -----------------------------------------------------------------------------------

    def zero_update(self, plan, d):
        model = self._rewired(plan)
        try:
            longest = max(self.val_items, key=lambda r: (len(r['input_ids']), r['id']))
            device = next(model.parameters()).device
            prompt = torch.tensor([list(longest['input_ids'])], dtype=torch.long, device=device)
            equal = bool(rm.zero_change(model, prompt))
            base = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
            expected = self.expected_base_sha256()
            d['zero_update'] = dict(logits_equal_untuned=equal, base_sha256=base, expected_base_sha256=expected,
                                    row=longest['id'], tokens=len(longest['input_ids']))
            return equal and base == expected
        finally:
            model = None
            self._release()

    def memory_and_recompute(self, plan, d):
        """One full training step on the longest gold row, then the recompute comparison on the same model."""
        model = self._rewired(plan)
        try:
            device = next(model.parameters()).device
            mods = rm.modules(model)
            if self.gradient_checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
            reads, masks = rt.split_read_and_mask_parameters(model)
            opt = self.record['optimizer']
            optimizer = torch.optim.AdamW(
                [{'params': reads, 'lr': float(plan['optimizer']['reads']['lr'])},
                 {'params': masks, 'lr': float(opt['masks']['lr'])}],
                betas=tuple(float(b) for b in opt['reads']['betas']), eps=float(opt['reads']['eps']),
                weight_decay=0.0, foreach=False)
            plan_seeds = plan.get('settings', {}).get('seeds', {})
            arm_mask_seed = int(
                plan_seeds['mask_noise'] if 'mask_noise' in plan_seeds
                else self.record['seeds']['mask_noise']
            )
            cache = hc.SharedMaskCache(seed=arm_mask_seed,
                                       group_size=int(self.record['data']['examples_per_mask']), device=device)
            rm.set_gate(model, cache.gate)
            shapes = {m.layer: tuple(m.rewire_log_alpha.shape) for m in mods}
            collator = sb.CompletionCollator(int(self.backend.pad_token_id))
            longest = max(self.training_rows, key=lambda r: (len(r['input_ids']), r['id']))
            model.train()

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                cache.begin_step(1, 1)
                cache.draw_group(shapes, dtype=torch.float32)
                _forward_backward(model, longest, collator=collator, device=device, denominator=_supervised(longest))
                l0 = float(self.record['objective']['l0_lambda']) * sum(hc.expected_active(m.rewire_log_alpha)
                                                                         for m in mods)
                l0.backward()
                torch.nn.utils.clip_grad_norm_(reads, float(opt['reads']['clip_norm']), error_if_nonfinite=True)
                optimizer.step()
                cache.clear()

            peak = self._peak(one_step)
            total = int(self.backend.total_memory() or 0)
            d['memory'] = dict(peak_bytes=peak, total_bytes=total, fraction=peak / total if total else None,
                               limit_fraction=float(self.q['memory_fraction']), row=longest['id'],
                               tokens=len(longest['input_ids']),
                               trainable_parameters=sum(p.numel() for p in reads + masks))
            memory_ok = bool(total) and peak <= float(self.q['memory_fraction']) * total

            # recompute: same cached u, one short row, with and without checkpoint recompute
            short = min(self.training_rows, key=lambda r: (len(r['input_ids']), r['id']))
            cache.begin_step(2, 1)
            cache.draw_group(shapes, dtype=torch.float32)
            grads = []
            for checkpointing in (True, False):
                if checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                else:
                    model.gradient_checkpointing_disable()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                _forward_backward(model, short, collator=collator, device=device, denominator=_supervised(short))
                grads.append([p.grad.detach().clone() for p in reads + masks])
            cache.clear()
            scale = max(float(g.abs().max()) for g in grads[1]) or 1.0
            diff = max(float((a - b).abs().max()) for a, b in zip(*grads))
            d['recompute'] = dict(max_abs_diff=diff, max_abs_grad=scale, relative=diff / scale,
                                  limit=float(self.q['recompute_rel_max']), row=short['id'],
                                  nonzero_reads=bool(any(torch.count_nonzero(c).item()
                                                         for m in mods for c in m.coefficients().values())))
            recompute_ok = diff / scale <= float(self.q['recompute_rel_max'])
            optimizer = grads = None
            return memory_ok, recompute_ok
        finally:
            model = None
            self._release()

    def disposable_run(self, arm, plan, d):
        cut = disposable_plan(plan, int(self.q['disposable_steps']))
        rt_sha = self.runtime_sha256()
        runs, seconds = [], []
        for attempt in ('a', 'b'):  # b is the restart in a fresh directory
            run_dir = self.scratch / arm / attempt / cut['run_id']
            if run_dir.parent.exists():
                shutil.rmtree(run_dir.parent)
            t0 = time.monotonic()
            rckpt.run_arm(run_dir, cut, model_factory=lambda: self._rewired(cut),
                          train=lambda model, p, sink, out=run_dir: self.train_arm(model, p, sink, out),
                          runtime_sha256=rt_sha, code_commit=self.code_commit, files=('logs/steps.jsonl',))
            seconds.append(time.monotonic() - t0)
            self._release()
            runs.append(run_dir)
        last = cut['checkpoints'][-1]
        logs = [_step_rows(r) for r in runs]
        hashes = [{s: _checkpoint_hash(r, s) for s in cut['checkpoints']} for r in runs]
        reproduces = logs[0] == logs[1] and hashes[0] == hashes[1]

        # reload the last checkpoint through the evaluation loader into a fresh base model
        model = self.backend.load_base_model()
        try:
            seal = reval.load_checkpoint_for_eval(model, rckpt.checkpoint_dir(runs[0], last), cut,
                                                  runtime_sha256=rt_sha, code_commit=self.code_commit)
            _, saved = rckpt.load_tensors(rckpt.checkpoint_dir(runs[0], last), cut, runtime_sha256=rt_sha,
                                       code_commit=self.code_commit)
            live = rckpt.state_of(model, cut)
            reload_ok = sorted(saved) == sorted(live) and all(torch.equal(saved[k].cpu(), live[k].detach().cpu())
                                                              for k in saved)
            t0 = time.monotonic()
            ce = reval.evaluate_validation_ce(model, self.val_items)
            ce_seconds = time.monotonic() - t0
            ids = self.val_question_ids[:int(self.q['greedy_questions'])]
            t0 = time.monotonic()
            greedy = reval.evaluate_greedy(
                model, self.val_questions, ids, Path(self.out_dir) / 'greedy' / arm, arm=f'{arm}-qualification',
                step=last, cohort='gsm8k-train-val128', decode=self.backend.decode,
                pad_token_id=self.backend.pad_token_id, eos_token_id=self.backend.eos_token_id,
                cap=int(self.record['evaluation']['cap']), batch_size=int(self.record['evaluation']['batch_size']),
                final_step=int(self.record['training']['steps']), checkpoint_meta=seal)
            greedy_seconds = time.monotonic() - t0
            cvf = cached_vs_forced(model, _spread(self.val_items, int(self.q['cached_vs_forced_rows'])),
                                   pad_token_id=int(self.backend.pad_token_id))
        finally:
            model = None
            self._release()
        steps = len(cut['steps'])
        per_step = min(seconds) / steps
        full = int(self.record['training']['steps'])
        d['disposable_run'] = dict(
            run_id=cut['run_id'], steps=steps, plan_sha256=cut['plan_sha256'], of_plan=plan['plan_sha256'],
            seconds=seconds, seconds_per_step=per_step, step_logs_equal=logs[0] == logs[1],
            checkpoint_hashes=hashes, checkpoint_reload_equal=reload_ok,
            final_step_row=logs[0][-1] if logs[0] else None,
            note='includes model load and checkpoint writes; an upper bound on the training step time')
        d['validation_ce_pass'] = dict(seconds=ce_seconds, **ce)
        d['greedy_pass'] = dict(seconds=greedy_seconds, questions=len(ids), accuracy=greedy['scores']['accuracy'],
                                tokens_mean=greedy['scores']['tokens_mean'], capped=greedy['scores']['capped'],
                                directory=greedy['directory'])
        d['cached_vs_forced'] = dict(cvf, limit=float(self.q['cached_vs_forced_abs_max']))
        n_val = len(sp.validation_steps(self.record))
        n_ckpt = len(self.record['checkpoints'])
        d['estimates'] = dict(
            training_hours=per_step * full / 3600,
            validation_ce_hours=ce_seconds * n_val / 3600,
            greedy_validation_hours=greedy_seconds * (len(self.val_question_ids) / max(len(ids), 1)) * n_ckpt / 3600,
            basis=f'{steps}-step disposable run, one CE pass, one {len(ids)}-question greedy pass; '
                  f'{full} steps, {n_val} CE checks and {n_ckpt} checkpoints in this record')
        for r in runs:  # D24: the disposable payloads never stay on disk
            shutil.rmtree(r.parent, ignore_errors=True)
        cvf_ok = cvf['per_token_abs']['max'] <= float(self.q['cached_vs_forced_abs_max'])
        return reproduces, reload_ok, cvf_ok

    def disk(self, d):
        planned = sp.planned_disk_bytes(self.record)
        usage = self.backend.probes.get('disk_usage', shutil.disk_usage)(Path(self.record['inputs']['work_dir']))
        need = planned + float(self.record['disk']['min_free_gb']) * 1e9
        d['disk'] = dict(free_bytes=int(usage.free), planned_bytes=int(planned), required_free_bytes=int(need))
        return usage.free >= need

    # --- the stage --------------------------------------------------------------------------------

    def __call__(self, ctx) -> dict[str, Any]:
        self.out_dir = Path(ctx.chain.local_root) / self.out_rel
        checks: dict[str, bool] = {}
        details: dict[str, Any] = {'not_run': dict(NOT_RUN)}
        checks['disk'] = bool(self.disk(details))
        for arm in self.arm_order:
            plan = self.arm_plans[arm]
            ctx.log(f'qualification {arm} (rank {plan["rank"]})')
            d = details[arm] = {'rank': int(plan['rank']), 'plan_sha256': plan['plan_sha256']}
            checks[f'{arm}:zero_update'] = bool(self.zero_update(plan, d))
            memory_ok, recompute_ok = self.memory_and_recompute(plan, d)
            checks[f'{arm}:memory'] = bool(memory_ok)
            checks[f'{arm}:recompute'] = bool(recompute_ok)
            reproduces, reload_ok, cvf_ok = self.disposable_run(arm, plan, d)
            checks[f'{arm}:restart_reproduces'] = bool(reproduces)
            checks[f'{arm}:checkpoint_reload'] = bool(reload_ok)
            checks[f'{arm}:cached_vs_forced'] = bool(cvf_ok)
            ctx.boundary(arm=arm)
        problems = self.backend.precision_problems()
        details['precision'] = dict(readback=self.backend.precision_readback(), problems=problems)
        checks['precision'] = not problems
        result = dict(checks=checks, review_flags=[], arms=list(self.arm_order))
        out = Path(self.out_dir)
        au.seal_run(out / 'summary', {'qualification.json': dict(result, details=details,
                                                                  runtime_sha256=self.runtime_sha256(),
                                                                  record_qualification=dict(self.q))})
        rel = f'{self.out_rel}/summary'
        greedy_rel = [f'{self.out_rel}/greedy/{a}' for a in self.arm_order]
        return dict(result, outputs=[rel, *greedy_rel],
                    notes={a: details[a].get('estimates') for a in self.arm_order})
