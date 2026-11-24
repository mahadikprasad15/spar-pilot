"""Sparse rewiring ticket 14: Pod chain integration on ``c_attribution_chain.Chain`` and ``TransformersBackend``.

Runs both sparse-rewiring arms (rank 1 and rank 0) through the existing ``c_attribution_chain.Chain``
(no new runner), ``c_attribution_backend.TransformersBackend``, ``sparse_rewiring_plan`` (PR #6),
and ``rewire_checkpoint.run_arm`` (PR #16):
- Stage order: ``preflight -> qualification -> references -> train:<r1> -> eval:<r1> -> train:<r0> -> eval:<r0>``.
  Qualification sign-off in ``Chain._check_signoff`` stays enforced for every stage after ``qualification``.
- Precision strictly via ``TransformersBackend`` (``set_ieee_precision``, ``precision_readback``,
  ``precision_problems``); never ``sampling_sft_training_backend.open_engine`` or legacy TF32 flags.
- Base model files verified via ``c_attribution_bundle.check_base_model`` (streamed hashing).
- Training and D24 restart-from-step-0 delegate to ``rewire_checkpoint.run_arm`` (verifies & keeps
  completed arms, refuses damaged seals, moves interrupted attempts aside as ``.interrupted-<ns>``
  and mirrors checkpoints via ``on_checkpoint``).
- Model references are dropped (``model = None``) before ``backend.release()`` (lesson 2026-11-19),
  and ``reset_peak()`` / ``peak_memory()`` are recorded per stage.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_backend as cab
from pilot_eval import c_attribution_bundle as cabundle
from pilot_eval import c_attribution_chain as cc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_training as rt
from pilot_eval import sft_backend as sb
from pilot_eval.workflow import _hash

REWIRE_SUBTREE: str = f'{cc.SUBTREE}/sparse-rewiring'
PREFLIGHT_DIR: str = f'{REWIRE_SUBTREE}/preflight'
REFERENCES_DIR: str = f'{REWIRE_SUBTREE}/references'
# The gold-SFT step-64 adapter in the input bundle (``rewire_bundle``): ``TransformersBackend.load('gold-sft')``
# resolves it through ``inputs.sources``, which the sparse launch record does not carry.
GOLD_ADAPTER_ROLE: str = 'adapter:gold-sft:64'


def chain_run_id(record: Mapping[str, Any]) -> str:
    """The chain's volume run ID (``$VOL/c-attribution/<run id>``): the label, namespaced so a sparse rehearsal never
    shares a run directory with a C-attribution one."""
    return f"sparse-rewiring-{record['label']}"


class RewireBackend(cab.TransformersBackend):
    """Sparse-rewiring backend subclassing ``TransformersBackend`` (IEEE precision, preflight, release)."""

    def __init__(self, record: Mapping[str, Any], volume_root: str | Path, *, device: str = 'cuda',
                 probes: Mapping[str, Any] | None = None, repo_root: Path = cab.REPO) -> None:
        adapted = dict(record)
        if 'generation' not in adapted and 'evaluation' in adapted:
            adapted['generation'] = {'eos_token_id': int(adapted['evaluation']['eos_token_id'])}
        inputs = dict(adapted['inputs'])
        inputs.setdefault('sources', {'gold-sft': {'adapter_role': GOLD_ADAPTER_ROLE}})
        adapted['inputs'] = inputs
        super().__init__(adapted, volume_root, device=device, probes=probes, repo_root=repo_root)

    def runtime(self) -> dict[str, Any]:
        try:
            return super().runtime()
        except ImportError:
            # CPU-only test environments without peft installed; on GPU pods peft is pinned and super() runs.
            import sys
            import transformers

            cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            return dict(
                torch=torch.__version__,
                transformers=transformers.__version__,
                peft=None,
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


def _arm_root(run_id: str) -> str:
    return f'{REWIRE_SUBTREE}/runs/{run_id}'


def _stage_order(arms: Sequence[str]) -> list[str]:
    order = ['preflight', 'qualification', 'references']
    for arm in arms:
        order.append(f'train:{arm}')
        order.append(f'eval:{arm}')
    return order


QUALIFICATION_DIR: str = f'{REWIRE_SUBTREE}/qualification'


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
    expected_base_sha256: str | Callable[[], str] | None = None,
    execution_kind: str = 'real-cuda',
    model_context_limit: int = 32768,
    gradient_checkpointing: bool = True,
    qualification_fn: Callable[[Any], dict[str, Any]] | None = None,
    references_fn: Callable[[Any], dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Callable[[Any], dict[str, Any]]]]:
    """Build stage list and handlers for ``c_attribution_chain.Chain``."""
    code_sha = record.get('code_commit')
    if not isinstance(code_sha, str) or not code_sha or code_sha == 'unknown':
        raise ValueError(f'a real code_commit is required in the launch record, got {code_sha!r}')

    arms = list(arm_order) if arm_order is not None else list(record['arms'])
    stage_names = _stage_order(arms)
    runtime_info: dict[str, Any] = {}

    def stage_preflight(ctx: Any) -> dict[str, Any]:
        problems = list(backend.preflight())
        if backend.device != 'cpu' and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            problems.append('CUBLAS_WORKSPACE_CONFIG must be set for deterministic CUDA training (D8)')
        # Streamed per-file SHA-256 verification of the local base model via c_attribution_bundle
        try:
            cabundle.check_base_model(backend.model_dir, backend.bundle, pinned=backend.pinned_base_model)
            model_files = dict(backend.bundle.get('base_model', {}).get('file_sha256', {}))
        except Exception as exc:
            model_files = {}
            problems.append(f'base model check: {exc}')
        if problems:
            raise RuntimeError('preflight failed: ' + '; '.join(problems))
        torch.use_deterministic_algorithms(True)
        rt_info = backend.runtime()
        runtime_info.clear()
        runtime_info.update(rt_info)
        rt_sha = _hash(rt_info)
        identity = {
            'runtime_sha256': rt_sha,
            'code_commit': code_sha,
            'model_files': model_files,
            'bundle_sha256': backend.bundle.get('sha256'),
        }
        ctx.chain.identity = identity
        receipt = {
            'runtime': rt_info,
            'runtime_sha256': rt_sha,
            'host': backend.host(),
            'model_downloads': model_files,
            'precision': backend.precision_readback(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'code_commit': code_sha,
        }
        au.seal_run(ctx.chain.local_root / PREFLIGHT_DIR, {'preflight.json': receipt})
        # ``identity`` is what qualification is valid for: the chain refuses a resume under another one.
        return {'outputs': [PREFLIGHT_DIR], 'runtime_sha256': rt_sha, 'identity': identity}

    def stage_references(ctx: Any) -> dict[str, Any]:
        if references_fn is not None:
            return references_fn(ctx)
        rt_sha = _hash(runtime_info or backend.runtime())
        res = reval.evaluate_gold_references_in_child(
            record=record,
            volume_root=backend.volume_root,
            out_dir=ctx.chain.local_root / REFERENCES_DIR,
            val_items=val_items,
            val_questions=val_questions,
            val_question_ids=val_question_ids,
            expected_runtime_sha256=rt_sha,
            device=backend.device,
            cap=int(record['evaluation']['cap']),
            batch_size=int(record['evaluation']['batch_size']),
        )
        return {'outputs': [f'{REFERENCES_DIR}/summary'], 'references': res['references']}

    def _expected_base() -> str | None:
        return expected_base_sha256() if callable(expected_base_sha256) else expected_base_sha256

    def run_training(model: torch.nn.Module, p: Mapping[str, Any], sink: Any, out_dir: Path) -> dict[str, Any]:
        """The real loop with this record's settings: the scientific arms and qualification's disposable run."""
        return rt.train_rewire(
            model,
            training_rows,
            out_dir,
            plan=p,
            sink=sink,
            mask_lr=float(record['optimizer']['masks']['lr']),
            betas=tuple(float(b) for b in record['optimizer']['reads']['betas']),
            eps=float(record['optimizer']['reads']['eps']),
            max_grad_norm=float(record['optimizer']['reads']['clip_norm']),
            snapshot_interval=int(record['cadences']['gate_snapshot_every']),
            settle_start=int(record['cadences']['settle']['first']),
            settle_interval=int(record['cadences']['settle']['every']),
            gradient_checkpointing=gradient_checkpointing,
            model_context_limit=int(model_context_limit),
            pad_token_id=backend.pad_token_id,
            expected_base_sha256=_expected_base(),
            execution_kind=execution_kind,
            val_ce_fn=lambda m, _step: reval.evaluate_validation_ce(m, val_items),
        )

    def _make_train_stage(arm: str) -> Callable[[Any], dict[str, Any]]:
        def stage_train(ctx: Any) -> dict[str, Any]:
            plan = arm_plans[arm]
            arm_cfg = record['arms'][arm]
            rt_sha = _hash(runtime_info or backend.runtime())
            arm_rel = _arm_root(plan['run_id'])
            arm_dir = ctx.chain.local_root / arm_rel
            train_holder: dict[str, Any] = {}

            def _model_factory() -> torch.nn.Module:
                return backend.load_rewired_model(
                    rank=int(plan['rank']),
                    seed=int(record['seeds']['read_init_base']),
                    mask_init=float(record['hard_concrete']['log_alpha_init']),
                )

            def _train_fn(model: torch.nn.Module, p: Mapping[str, Any], sink: Any) -> None:
                try:
                    backend.reset_peak()
                    summary = run_training(model, p, sink, arm_dir)
                    summary['peak_memory_bytes'] = backend.peak_memory()
                    train_holder['summary'] = summary
                finally:
                    # Lesson 2026-11-19: drop local reference before calling backend.release()
                    model = None
                    backend.release()

            # PR #16 run_arm handles prepare_arm (keep verified completed arm, refuse damaged seal,
            # move interrupted attempt aside as .interrupted-<ns>), CheckpointSink, and seal_arm.
            arm_res = rckpt.run_arm(
                arm_dir,
                plan,
                model_factory=_model_factory,
                train=_train_fn,
                runtime_sha256=rt_sha,
                code_commit=code_sha,
                files=('logs/steps.jsonl',),
                on_checkpoint=lambda _step, _folder: ctx.chain.mirror(),
            )
            outputs = [f'{arm_rel}/checkpoints/checkpoint-{s}' for s in plan['checkpoints']]
            return {
                'outputs': outputs,
                'arm': arm,
                'action': arm_res['action'],
                'plan_sha256': plan['plan_sha256'],
                'checkpoint_steps': list(plan['checkpoints']),
                'peak_memory_bytes': train_holder.get('summary', {}).get('peak_memory_bytes'),
            }

        return stage_train

    def _make_eval_stage(arm: str) -> Callable[[Any], dict[str, Any]]:
        def stage_eval(ctx: Any) -> dict[str, Any]:
            plan = arm_plans[arm]
            rt_sha = _hash(runtime_info or backend.runtime())
            arm_rel = _arm_root(plan['run_id'])
            arm_dir = ctx.chain.local_root / arm_rel
            final_step = int(record['training']['steps'])
            eval_cfg = record['evaluation']
            outputs: list[str] = []
            prior_uses: list[dict[str, Any]] = []
            peak_bytes = 0

            backend.reset_peak()
            model: torch.nn.Module | None = backend.load_base_model()
            try:
                for step in plan['checkpoints']:
                    ckpt_dir = rckpt.checkpoint_dir(arm_dir, int(step))
                    seal = reval.load_checkpoint_for_eval(
                        model,
                        ckpt_dir,
                        plan,
                        runtime_sha256=rt_sha,
                        code_commit=code_sha,
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
                        checkpoint_meta=seal,
                    )
                    outputs.append(val_rel)
                    prior_uses.append(val_res['prior_use'])

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
                            checkpoint_meta=seal,
                        )
                        outputs.append(c150_rel)
                        prior_uses.append(c150_res['prior_use'])
                peak_bytes = backend.peak_memory()
            finally:
                model = None
                backend.release()

            eval_summary_rel = f'{arm_rel}/eval_summary'
            au.seal_run(ctx.chain.local_root / eval_summary_rel, {'prior_uses.json': prior_uses})
            outputs.append(eval_summary_rel)
            return {'outputs': outputs, 'arm': arm, 'prior_uses': len(prior_uses), 'peak_memory_bytes': peak_bytes}

        return stage_eval

    if qualification_fn is None:
        from pilot_eval.rewire_qualification import Qualification

        qualification_fn = Qualification(
            record=record, backend=backend, arm_plans=arm_plans, training_rows=training_rows, val_items=val_items,
            val_questions=val_questions, val_question_ids=val_question_ids, arm_order=arms, train_arm=run_training,
            expected_base_sha256=_expected_base, runtime_sha256=lambda: _hash(runtime_info or backend.runtime()),
            code_commit=code_sha, out_rel=QUALIFICATION_DIR, gradient_checkpointing=gradient_checkpointing)
    handlers: dict[str, Callable[[Any], dict[str, Any]]] = {
        'preflight': stage_preflight,
        'qualification': qualification_fn,
        'references': stage_references,
    }
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
    expected_base_sha256: str | Callable[[], str] | None = None,
    execution_kind: str = 'real-cuda',
    model_context_limit: int = 32768,
    gradient_checkpointing: bool = True,
    qualification_fn: Callable[[Any], dict[str, Any]] | None = None,
    references_fn: Callable[[Any], dict[str, Any]] | None = None,
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
        expected_base_sha256=expected_base_sha256,
        execution_kind=execution_kind,
        model_context_limit=model_context_limit,
        gradient_checkpointing=gradient_checkpointing,
        qualification_fn=qualification_fn,
        references_fn=references_fn,
    )
    chain_record = dict(record)
    chain_record.setdefault('run_id', chain_run_id(record))
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
