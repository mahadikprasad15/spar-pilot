"""Sparse rewiring ticket 14: Pod chain integration on ``c_attribution_chain.Chain`` and ``TransformersBackend``.

Runs both sparse-rewiring arms (rank 1 and rank 0) through the existing ``c_attribution_chain.Chain``
(no new runner), ``c_attribution_backend.TransformersBackend``, ``sparse_rewiring_plan`` (PR #6),
and ``rewire_checkpoint.run_arm`` (PR #16):
- Stage order: ``preflight -> qualification -> references -> train:<r1> -> eval:<r1> -> train:<r0> -> eval:<r0>``.
  Qualification sign-off in ``Chain._check_signoff`` stays enforced for every stage after ``qualification``.
- Pod identity: ``current_pod`` defaults to ``None`` so ``Chain`` reads ``RUNPOD_POD_ID`` directly (P1.1).
- Preflight returns a full ``identity`` dict (code, runtime, bundle, plans, frozen rows, validation,
  cohort) so ``Chain`` binds ``qualified_identity``, and seals receipts per attempt (P1.3).
- Training seals ``arm_evidence`` (``training-complete.json``, ``plan.json``, logs, ``hardening.json``,
  ``masks/trace/*.npz``) as a ``complete.json`` unit so ``copy_sealed`` backs up and restores every
  artifact needed for dynamics figures, hardening reports, and ``verify_arm`` (P1.4).
- Production requires 150 unique cohort150 questions at steps 64 and final; rehearsal/qualification
  refuses cohort150 (P1.7).
"""
from __future__ import annotations

import json
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
from pilot_eval.workflow import _hash

REWIRE_SUBTREE: str = f'{cc.SUBTREE}/sparse-rewiring'
PREFLIGHT_DIR: str = f'{REWIRE_SUBTREE}/preflight'
REFERENCES_DIR: str = f'{REWIRE_SUBTREE}/references'
ARM_EVIDENCE_DIR: str = 'arm_evidence'
# The gold-SFT step-64 adapter in the input bundle (``rewire_bundle``): ``TransformersBackend.load('gold-sft')``
# resolves it through ``inputs.sources``.
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


def seal_arm_evidence(arm_dir: str | Path, plan: Mapping[str, Any]) -> str:
    """Seal the arm's analysis/resume evidence (``training-complete.json``, ``plan.json``, logs,
    ``hardening.json``, and ``masks/trace/*.npz``) into ``arm_dir/arm_evidence/`` as a ``complete.json``
    unit so ``copy_sealed`` backs up and restores everything needed to verify and report (P1.4)."""
    root = Path(arm_dir)
    files: dict[str, Any] = {'plan.json': dict(plan)}
    arm_seal = root / rckpt.ARM_SEAL
    if not arm_seal.is_file():
        raise ValueError(f'{arm_seal} is missing; cannot seal arm evidence before training-complete.json')
    files[rckpt.ARM_SEAL] = arm_seal.read_bytes()
    for rel in ('logs/steps.jsonl', 'logs/validation.jsonl', 'logs/settle.jsonl', 'logs/events.jsonl', 'hardening.json'):
        p = root / rel
        if p.is_file():
            files[rel] = p.read_bytes()
    trace_dir = root / 'masks' / 'trace'
    if trace_dir.is_dir():
        for npz in sorted(trace_dir.glob('*.npz')):
            files[f'masks/trace/{npz.name}'] = npz.read_bytes()
    return au.seal_run(root / ARM_EVIDENCE_DIR, files)


def restore_arm_evidence(arm_dir: str | Path) -> set[str]:
    """Verify ``arm_dir/arm_evidence`` and materialize its files into ``arm_dir`` after a clean-directory restore."""
    root = Path(arm_dir)
    ev_dir = root / ARM_EVIDENCE_DIR
    verified_files = au.verify_run(ev_dir)
    for rel in sorted(verified_files):
        src = ev_dir / rel
        dst = root / rel
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    return verified_files


def _validate_cohort_inputs(
    record: Mapping[str, Any],
    cohort150_questions: Mapping[str, Mapping[str, Any]] | None,
    cohort150_question_ids: Sequence[str] | None,
) -> None:
    """Enforce P1.7: production requires 150 unique cohort150 questions; rehearsal refuses cohort150."""
    mode = record['mode']
    if mode == 'production':
        if cohort150_questions is None or cohort150_question_ids is None:
            raise ValueError('production mode requires frozen cohort150_questions and cohort150_question_ids (D12)')
        ids = list(cohort150_question_ids)
        if len(ids) != 150 or len(set(ids)) != 150:
            raise ValueError(f'production cohort150 requires 150 unique question IDs, got {len(ids)} ({len(set(ids))} unique)')
        missing = [q for q in ids if q not in cohort150_questions]
        if missing:
            raise ValueError(f'production cohort150 missing question payloads: {missing[:5]}')
    else:
        if cohort150_questions is not None or cohort150_question_ids is not None:
            raise ValueError(f'cohort150 inputs are forbidden in {mode!r} mode (D12 / lesson 2026-11-13: validation only)')


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
    _validate_cohort_inputs(record, cohort150_questions, cohort150_question_ids)

    arms = list(arm_order) if arm_order is not None else list(record['arms'])
    stage_names = _stage_order(arms)
    runtime_info: dict[str, Any] = {}
    rows_sha = rt.rows_content_sha256(training_rows)
    val_items_sha = _hash([dict(v) for v in val_items])
    val_q_sha = _hash({'ids': list(val_question_ids), 'questions': {q: val_questions[q] for q in val_question_ids}})
    c150_sha = (
        _hash({'ids': list(cohort150_question_ids), 'questions': {q: cohort150_questions[q] for q in cohort150_question_ids}})
        if cohort150_questions is not None and cohort150_question_ids is not None
        else None
    )
    plans_sha = {arm: arm_plans[arm]['plan_sha256'] for arm in arms}

    def stage_preflight(ctx: Any) -> dict[str, Any]:
        problems = list(backend.preflight())
        if backend.device != 'cpu' and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            problems.append('CUBLAS_WORKSPACE_CONFIG must be set for deterministic CUDA training (D8)')
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
        # P1.3: return full identity so Chain.run binds state['qualified_identity'] and checks it on resume
        identity = {
            'code_commit': code_sha,
            'runtime_sha256': rt_sha,
            'bundle_sha256': _hash(backend.bundle),
            'model_files': model_files,
            'plans_sha256': plans_sha,
            'rows_sha256': rows_sha,
            'val_items_sha256': val_items_sha,
            'val_questions_sha256': val_q_sha,
            'cohort150_sha256': c150_sha,
        }
        ctx.chain.identity = identity
        receipt = {
            'attempt': ctx.chain.attempt,
            'runtime': rt_info,
            'runtime_sha256': rt_sha,
            'host': backend.host(),
            'identity': identity,
            'model_downloads': model_files,
            'precision': backend.precision_readback(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'code_commit': code_sha,
        }
        # P1.3: attempt-scoped preflight receipt so a host-driver change on a replacement Pod never collides
        pf_rel = f'{PREFLIGHT_DIR}/{ctx.chain.attempt}'
        au.seal_run(ctx.chain.local_root / pf_rel, {'preflight.json': receipt})
        return {'outputs': [pf_rel], 'runtime_sha256': rt_sha, 'identity': identity}

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
            snapshot_interval=int(record['cadences']['gate_snapshot_every']),
            settle_start=int(record['cadences']['settle']['first']),
            settle_interval=int(record['cadences']['settle']['every']),
            gradient_checkpointing=gradient_checkpointing,
            model_context_limit=int(model_context_limit),
            pad_token_id=backend.pad_token_id,
            expected_base_sha256=_expected_base(),
            expected_rows_sha256=rows_sha,
            execution_kind=execution_kind,
            val_ce_fn=lambda m, _step: reval.evaluate_validation_ce(m, val_items),
        )

    def _make_train_stage(arm: str) -> Callable[[Any], dict[str, Any]]:
        def stage_train(ctx: Any) -> dict[str, Any]:
            plan = arm_plans[arm]
            rt_sha = _hash(runtime_info or backend.runtime())
            arm_rel = _arm_root(plan['run_id'])
            arm_dir = ctx.chain.local_root / arm_rel
            train_holder: dict[str, Any] = {}

            # If arm_evidence was restored from volume, materialize training-complete.json & logs before prepare_arm
            if (arm_dir / ARM_EVIDENCE_DIR / 'complete.json').is_file():
                restore_arm_evidence(arm_dir)

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
                    model = None
                    backend.release()

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
            # P1.4: seal arm_evidence (training-complete.json, plan.json, logs, hardening.json, masks/trace/*.npz)
            seal_arm_evidence(arm_dir, plan)
            outputs = [f'{arm_rel}/{ARM_EVIDENCE_DIR}']
            outputs += [f'{arm_rel}/checkpoints/checkpoint-{s}' for s in plan['checkpoints']]
            outputs += [f'{arm_rel}/masks/step-{s}' for s in plan['checkpoints'] if (arm_dir / 'masks' / f'step-{s}' / 'complete.json').is_file()]
            if arm_res.get('evidence'):
                ev_rel = Path(arm_res['evidence']).relative_to(ctx.chain.local_root).as_posix()
                outputs.append(ev_rel)
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
            mode = record['mode']
            outputs: list[str] = []
            prior_uses: list[dict[str, Any]] = []
            peak_bytes = 0

            # P1.4: restore & verify full arm seal under current plan, runtime, and code commit before eval
            if (arm_dir / ARM_EVIDENCE_DIR / 'complete.json').is_file():
                restore_arm_evidence(arm_dir)
            rckpt.verify_arm(arm_dir, plan, runtime_sha256=rt_sha, code_commit=code_sha)

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
                        mode=mode,
                        checkpoint_meta=seal,
                    )
                    outputs.append(val_rel)
                    prior_uses.append(val_res['prior_use'])

                    # P1.7: cohort150 only in production, required at both 64 and final_step
                    c150_steps = set(int(s) for s in eval_cfg['cohort150']['steps'])
                    if mode == 'production' and int(step) in c150_steps:
                        assert cohort150_questions is not None and cohort150_question_ids is not None
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
                            mode=mode,
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
    """Create a ``c_attribution_chain.Chain`` configured for the sparse-rewiring two-arm run.

    ``current_pod`` defaults to ``None`` so ``Chain`` reads ``RUNPOD_POD_ID`` from the environment (P1.1).
    """
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
        current_pod=current_pod,
        stage_order=stage_order,
    )
    if exporter is not None:
        kwargs['exporter'] = exporter
    if runner is not None:
        kwargs['runner'] = runner
    return cc.Chain(chain_record, **kwargs)
