"""Sparse rewiring ticket 14: Pod chain integration on ``c_attribution_chain.Chain`` and ``TransformersBackend``.

Runs both sparse-rewiring arms (rank 1 and rank 0) through the existing ``c_attribution_chain.Chain``
(no new runner) and ``c_attribution_backend.TransformersBackend``:
- Precision strictly via ``neuron_audit_collector.set_ieee_precision`` + ``precision_readback`` /
  ``precision_problems`` (never ``sampling_sft_training_backend.open_engine`` or legacy ``allow_tf32``).
- Preflight enforces IEEE readback, the D24-sized ``disk`` block, ``HF_HOME`` off the volume,
  ``HF_HUB_OFFLINE=1``, ``volume_identity``, private repo, and SHA-256 verification of every
  downloaded file in ``inputs.model_dir``.
- One training stage per arm: on a relaunch, verified completed arms are kept (all expected
  checkpoints re-verified), while an interrupted arm's partial directory is discarded and restarted
  cleanly from step 0 (D24).
- Teardown (mirror, volume copy, export, closing summary before stop, ``stop_commands``) runs
  unchanged through ``c_attribution_chain.Chain``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_backend as cab
from pilot_eval import c_attribution_chain as cc
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_training as rt
from pilot_eval.workflow import _hash

REWIRE_SUBTREE: str = f'{cc.SUBTREE}/sparse-rewiring'
PREFLIGHT_DIR: str = f'{REWIRE_SUBTREE}/preflight'


def hash_model_downloads(model_dir: str | Path, pinned: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Hash every file in ``model_dir`` by SHA-256 and check against ``pinned`` if provided."""
    root = Path(model_dir)
    if not root.is_dir():
        raise ValueError(f'model_dir does not exist: {root}')
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    problems: list[str] = []
    if pinned is not None:
        for rel, expected in sorted(pinned.items()):
            observed = hashes.get(rel)
            if observed is None:
                problems.append(f'missing model file: {rel}')
            elif observed != expected:
                problems.append(f'model file hash mismatch for {rel}: got {observed}, expected {expected}')
    return {'files': hashes, 'problems': problems}


class RewireBackend(cab.TransformersBackend):
    """Sparse-rewiring backend subclassing ``TransformersBackend`` (IEEE precision, preflight, release)."""

    def __init__(self, record: Mapping[str, Any], volume_root: str | Path, *, device: str = 'cuda',
                 probes: Mapping[str, Any] | None = None, repo_root: Path = cab.REPO) -> None:
        # Adapt sparse-rewiring record's ``evaluation.eos_token_id`` for TransformersBackend.__init__
        adapted = dict(record)
        if 'generation' not in adapted and 'evaluation' in adapted:
            adapted['generation'] = {'eos_token_id': int(adapted['evaluation']['eos_token_id'])}
        super().__init__(adapted, volume_root, device=device, probes=probes, repo_root=repo_root)

    def runtime(self) -> dict[str, Any]:
        import transformers
        try:
            import peft
            peft_ver: str | None = peft.__version__
        except ImportError:
            peft_ver = None
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        return dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            peft=peft_ver,
            python=sys.version.split()[0],
            dtype='float32',
            attention='eager',
            device=self.device,
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            cuda_runtime=torch.version.cuda,
            cudnn=cudnn,
            cuda_libraries=cab.cuda_libraries(),
            tokenizer=self.tokenizer_identity(),
            autocast=False,
            precision=self.precision_readback(),
        )

    def load_base_model(self) -> torch.nn.Module:
        """Load the pinned local FP32 eager Qwen2 model and enforce spec test 22 (no model-config pad_token_id)."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation='eager',
        )
        reval.resolve_token_ids(self._tokenizer, model_config=model.config)
        model.to(self.device)
        return model

    def load_rewired_model(self, *, rank: int, seed: int = 42, mask_init: float = 1.0) -> torch.nn.Module:
        """Load base model and install clean-room ``RewireMLP`` modules."""
        model = self.load_base_model()
        rm.install(model, rank=int(rank), seed=int(seed), mask_init=float(mask_init))
        return model


def _arm_root(arm: str, run_id: str) -> str:
    return f'{REWIRE_SUBTREE}/runs/{run_id}'


def _stage_order(arms: Sequence[str]) -> list[str]:
    order = ['preflight']
    for arm in arms:
        order.append(f'train:{arm}')
        order.append(f'eval:{arm}')
    return order


def _clear_partial_arm(arm_dir: Path) -> None:
    """D24: an interrupted arm's partial unsealed directory is discarded before restarting from step 0."""
    if arm_dir.exists():
        shutil.rmtree(arm_dir)


def build_rewire_stages(
    *,
    record: Mapping[str, Any],
    backend: RewireBackend,
    training_rows: Sequence[Mapping[str, Any]],
    arm_plans: Mapping[str, Mapping[str, Any]],
    val_items: Sequence[Mapping[str, Any]],
    val_questions: Mapping[str, Mapping[str, Any]],
    val_question_ids: Sequence[str],
    cohort150_questions: Mapping[str, Mapping[str, Any]] | None = None,
    cohort150_question_ids: Sequence[str] | None = None,
    arm_order: Sequence[str] | None = None,
    gradient_checkpointing: bool = True,
) -> tuple[list[str], dict[str, Callable[[Any], dict[str, Any]]]]:
    """Build stage list and handlers for ``c_attribution_chain.Chain``."""
    arms = list(arm_order) if arm_order is not None else list(record['arms'])
    stage_names = _stage_order(arms)
    code_sha = str(record.get('code_commit', 'unknown'))
    runtime_info: dict[str, Any] = {}

    def stage_preflight(ctx: Any) -> dict[str, Any]:
        problems = list(backend.preflight())
        dl = hash_model_downloads(backend.model_dir, pinned=backend.pinned_base_model)
        problems.extend(dl['problems'])
        if problems:
            raise RuntimeError('preflight failed: ' + '; '.join(problems))
        rt_info = backend.runtime()
        runtime_info.clear()
        runtime_info.update(rt_info)
        rt_sha = _hash(rt_info)
        ctx.chain.identity = {'runtime_sha256': rt_sha, 'code_commit': code_sha, 'model_files': dl['files']}
        receipt = {
            'runtime': rt_info,
            'runtime_sha256': rt_sha,
            'host': backend.host(),
            'model_downloads': dl['files'],
            'precision': backend.precision_readback(),
            'code_commit': code_sha,
        }
        au.seal_run(ctx.chain.local_root / PREFLIGHT_DIR, {'preflight.json': receipt})
        return {'outputs': [PREFLIGHT_DIR], 'runtime_sha256': rt_sha}

    def _make_train_stage(arm: str) -> Callable[[Any], dict[str, Any]]:
        def stage_train(ctx: Any) -> dict[str, Any]:
            arm_cfg = record['arms'][arm]
            plan = arm_plans[arm]
            plan_sha = _hash(plan)
            rt_sha = _hash(runtime_info or backend.runtime())
            arm_rel = _arm_root(arm, arm_cfg['run_id'])
            arm_dir = ctx.chain.local_root / arm_rel
            # D24: if we reached here, either first run or interrupted/unverified -> discard partial and restart from 0
            _clear_partial_arm(arm_dir)

            model = backend.load_rewired_model(
                rank=int(arm_cfg['rank']),
                seed=int(record['seeds']['read_init_base']),
                mask_init=float(record['hard_concrete']['log_alpha_init']),
            )
            try:
                summary = rt.train_rewire(
                    model,
                    training_rows,
                    plan['step_batches'],
                    arm_dir,
                    rank=int(arm_cfg['rank']),
                    read_lr=float(arm_cfg['read_lr']),
                    mask_lr=float(record['optimizer']['masks']['lr']),
                    l0_lambda=float(record['objective']['l0_lambda']),
                    betas=tuple(float(b) for b in record['optimizer']['reads']['betas']),
                    eps=float(record['optimizer']['reads']['eps']),
                    max_grad_norm=float(record['optimizer']['reads']['clip_norm']),
                    warmup_steps=int(record['training']['warmup_steps']),
                    freeze_step=int(record['training']['freeze_after_step']),
                    mask_seed=int(record['seeds']['mask_noise']),
                    mask_group_size=int(record['data']['examples_per_mask']),
                    checkpoint_steps=plan['checkpoint_steps'],
                    snapshot_interval=int(record['cadences']['gate_snapshot_every']),
                    settle_start=int(record['cadences']['settle']['first']),
                    settle_interval=int(record['cadences']['settle']['every']),
                    gradient_checkpointing=gradient_checkpointing,
                    pad_token_id=backend.pad_token_id,
                    plan_sha256=plan_sha,
                    runtime_sha256=rt_sha,
                    code_sha256=code_sha,
                    val_ce_fn=lambda m, _step: reval.evaluate_validation_ce(m, val_items),
                )
            finally:
                backend.release(model)

            summary_rel = f'{arm_rel}/train_summary'
            au.seal_run(ctx.chain.local_root / summary_rel, {'train_summary.json': summary, 'plan.json': plan})
            outputs = [summary_rel] + [f'{arm_rel}/checkpoints/checkpoint-{s}' for s in summary['checkpoint_steps']]
            outputs += [f'{arm_rel}/masks/step-{s}' for s in summary['checkpoint_steps']]
            return {'outputs': outputs, 'arm': arm, 'checkpoint_steps': summary['checkpoint_steps']}

        return stage_train

    def _make_eval_stage(arm: str) -> Callable[[Any], dict[str, Any]]:
        def stage_eval(ctx: Any) -> dict[str, Any]:
            arm_cfg = record['arms'][arm]
            plan = arm_plans[arm]
            plan_sha = _hash(plan)
            rt_sha = _hash(runtime_info or backend.runtime())
            arm_rel = _arm_root(arm, arm_cfg['run_id'])
            arm_dir = ctx.chain.local_root / arm_rel
            final_step = int(record['training']['steps'])
            eval_cfg = record['evaluation']
            outputs: list[str] = []
            prior_uses: list[dict[str, Any]] = []

            model = backend.load_base_model()
            try:
                for step in plan['checkpoint_steps']:
                    ckpt_dir = arm_dir / 'checkpoints' / f'checkpoint-{step}'
                    meta = reval.load_rewire_checkpoint(
                        model,
                        ckpt_dir,
                        expected_plan_sha256=plan_sha,
                        expected_runtime_sha256=rt_sha,
                        expected_code_sha256=code_sha,
                    )
                    val_rel = f'{arm_rel}/eval/{step}/greedy-val'
                    val_res = reval.evaluate_greedy(
                        model,
                        val_questions,
                        val_question_ids,
                        ctx.chain.local_root / val_rel,
                        arm=arm,
                        step=int(step),
                        cohort='gsm8k-train-val128',
                        decode=backend.decode,
                        pad_token_id=backend.pad_token_id,
                        eos_token_id=backend.eos_token_id,
                        cap=int(eval_cfg['cap']),
                        batch_size=int(eval_cfg['batch_size']),
                        final_step=final_step,
                        checkpoint_meta=meta,
                    )
                    outputs.append(val_rel)
                    prior_uses.append(val_res['prior_use'])

                    # D12: cohort150 only at configured steps (64 and final)
                    c150_steps = set(int(s) for s in eval_cfg['cohort150']['steps'])
                    if cohort150_questions is not None and cohort150_question_ids is not None and int(step) in c150_steps:
                        c150_rel = f'{arm_rel}/eval/{step}/greedy-cohort150'
                        c150_res = reval.evaluate_greedy(
                            model,
                            cohort150_questions,
                            cohort150_question_ids,
                            ctx.chain.local_root / c150_rel,
                            arm=arm,
                            step=int(step),
                            cohort=reval.COHORT150_NAME,
                            decode=backend.decode,
                            pad_token_id=backend.pad_token_id,
                            eos_token_id=backend.eos_token_id,
                            cap=int(eval_cfg['cap']),
                            batch_size=int(eval_cfg['batch_size']),
                            final_step=final_step,
                            checkpoint_meta=meta,
                        )
                        outputs.append(c150_rel)
                        prior_uses.append(c150_res['prior_use'])
            finally:
                backend.release(model)

            eval_summary_rel = f'{arm_rel}/eval_summary'
            au.seal_run(ctx.chain.local_root / eval_summary_rel, {'prior_uses.json': prior_uses})
            outputs.append(eval_summary_rel)
            return {'outputs': outputs, 'arm': arm, 'prior_uses': len(prior_uses)}

        return stage_eval

    handlers: dict[str, Callable[[Any], dict[str, Any]]] = {'preflight': stage_preflight}
    for arm in arms:
        handlers[f'train:{arm}'] = _make_train_stage(arm)
        handlers[f'eval:{arm}'] = _make_eval_stage(arm)
    return stage_names, handlers


def build_rewire_chain(
    *,
    record: Mapping[str, Any],
    backend: RewireBackend,
    local_root: str | Path,
    volume: str | Path,
    training_rows: Sequence[Mapping[str, Any]],
    arm_plans: Mapping[str, Mapping[str, Any]],
    val_items: Sequence[Mapping[str, Any]],
    val_questions: Mapping[str, Mapping[str, Any]],
    val_question_ids: Sequence[str],
    cohort150_questions: Mapping[str, Mapping[str, Any]] | None = None,
    cohort150_question_ids: Sequence[str] | None = None,
    arm_order: Sequence[str] | None = None,
    gradient_checkpointing: bool = True,
    exporter: Any = None,
    runner: Any = None,
    volume_identity: str | None = None,
    current_pod: str | None = None,
) -> cc.Chain:
    """Create a ``c_attribution_chain.Chain`` configured for the sparse-rewiring two-arm run."""
    stage_order, stages = build_rewire_stages(
        record=record,
        backend=backend,
        training_rows=training_rows,
        arm_plans=arm_plans,
        val_items=val_items,
        val_questions=val_questions,
        val_question_ids=val_question_ids,
        cohort150_questions=cohort150_questions,
        cohort150_question_ids=cohort150_question_ids,
        arm_order=arm_order,
        gradient_checkpointing=gradient_checkpointing,
    )
    chain_record = dict(record)
    chain_record.setdefault('run_id', str(record['label']))
    kwargs: dict[str, Any] = dict(
        local_root=local_root,
        volume=volume,
        stages=stages,
        volume_identity=volume_identity or record['volume_identity'],
        current_pod=current_pod if current_pod is not None else record['pod_id'],
        stage_order=stage_order,
    )
    if exporter is not None:
        kwargs['exporter'] = exporter
    if runner is not None:
        kwargs['runner'] = runner
    return cc.Chain(chain_record, **kwargs)
