"""Sparse rewiring on Sampling SFT (k = 2): typed launch record and frozen per-arm plan.

Design & spec: ``.scratch/sparse-rewiring-sampling-k2/{design.md,spec.md}``.
This module stays completely separate from ``pilot_eval.sparse_rewiring_plan`` so Gold SFT's
sealed ``launch-record-v1`` schema and historical seals remain untouched.

Key contracts (signed 2026-11-26):
- K2-D1: Run A trains ``seed 42`` first at Gold's exact ``l0_lambda = 6.0e-4``. Conditional Run B
  triggers if and only if verified completed Run A ``seed 42`` ends at step 1,024 with ``< 4``
  binary survivors, using ``lambda_k2 = 6e-4 * (delta_ce_k2 / 0.1777)`` from sealed ``k2_headroom.json``.
- K2-D2: Rank 1 (``read_lr = 3.0e-3``) across 2 seeds (``seed 42`` and ``seed 43``); ``seed 43``
  runs on Run A only if Run A ``seed 42`` has ``>= 4`` survivors, otherwise both seeds run on Run B.
- K2-D3: 64 steps per epoch for 477 covered ``k = 2`` items. Epoch 1 (steps 1..64) preserves
  Gold's exact 64 batches with the 35 uncovered slots omitted (batch sizes 3..8; never empty).
  Epochs 2..16 (steps 65..1024) reshuffle the 477 items with ``random.Random(data_order_base + e - 1)``
  and partition each epoch into 64 batches of 7–8 items (29×8 + 35×7 = 477), keeping
  1,024 steps = 16 exact epochs aligned with Gold's checkpoint schedule.

Every generated plan is directly compatible with ``rewire_training.train_rewire``,
``rewire_checkpoint.CheckpointSink``, ``rewire_evaluation``, and ``rewire_chain.build_rewire_stages``.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pilot_eval import c_attribution_chain as cc
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.attribution_chain import LaunchRefused
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

REPO = Path(__file__).resolve().parents[1]

LAUNCH_SCHEMA = 'sparse-rewiring-k2/launch-record-v1'
PLAN_SCHEMA = 'sparse-rewiring-k2/plan-v1'

GOLD_VAL_CE_HEADROOM = 0.5728 - 0.3951   # 0.1777 nats/token (Gold production report, 2026-11-26)
GOLD_MATCHED_LAMBDA = 6.0e-4
RUN_B_SURVIVOR_THRESHOLD = 4             # K2-D1: trigger Run B if Run A has < 4 survivors

# Pinned Sampling SFT (k = 2) artifact SHA-256 digests (from l4-training-data-v1 & l4-night-v1-train-k2)
K2_EXPORT_COMPLETE_SHA256 = '50b226b09128310f1c37fc6137a627870066f4a0d689724cb06336cdbcff5067'
K2_ADAPTER_WEIGHTS_SHA256 = 'a28f52a0b61d0ee6140ece1a8fc4983fb1dcf4bd75db84c8594e5a9c73c27f73'
K2_ADAPTER_CONFIG_SHA256 = '78954871dc6682bc2300828a51a65542951d56c6b868657430dfdc250a90712e'
K2_EXPORT_FILES_REQUIRED = (
    'arms/k2/training.jsonl',
    'arms/k2/batches.json',
    'arms/k2/coverage.json',
)

# Directories containing all executable Python code bound into every K2 launch seal (P1 #2)
EXECUTABLE_ROOTS = ('pilot_eval', 'scripts')

# Kept for introspection/docs; actual hashing & dirty-tree checks cover ALL .py files under EXECUTABLE_ROOTS
EXECUTABLE_SOURCES = (
    'pilot_eval/sparse_rewiring_k2_plan.py',
    'pilot_eval/rewire_k2_launch.py',
    'pilot_eval/rewire_chain.py',
    'pilot_eval/rewire_qualification.py',
    'pilot_eval/rewire_training.py',
    'pilot_eval/rewire_checkpoint.py',
    'pilot_eval/rewire_evaluation.py',
    'pilot_eval/rewire_module.py',
    'pilot_eval/hard_concrete.py',
    'pilot_eval/sft_backend.py',
    'pilot_eval/rewire_dynamics.py',
    'pilot_eval/c_attribution_chain.py',
    'pilot_eval/sparse_rewiring_plan.py',
    'pilot_eval/rewire_bundle.py',
    'scripts/generate_val128_k2_targets.py',
)

NUM, INT, TEXT = cc.NUM, cc.INT, cc.TEXT
SCHEMA = {
    'schema': TEXT, 'mode': TEXT, 'label': TEXT, 'code_commit': TEXT, 'executable_sha256': TEXT,
    'pod_id': TEXT, 'pod_stop_commands': [TEXT], 'volume_identity': TEXT, 'export_repo': TEXT,
    'hourly_cost_usd': cc.COST,
    'timeouts_seconds': {'copy': NUM, 'export_attempt': NUM, 'stop_attempt': NUM, 'writer_join': NUM},
    'marker_stale_seconds': NUM,
    'disk': {'min_free_gb': NUM, 'container_total_gb': NUM},
    'hardware': {'gpu': TEXT, 'count': INT},
    'arms': {'*': {'run_id': TEXT, 'rank': INT, 'read_lr': NUM, 'seed': INT}},
    'model': {'layers': INT, 'hidden': INT, 'intermediate': INT, 'parts': [TEXT], 'positions': TEXT},
    'data': {
        'recipe': TEXT, 'k': INT, 'total_slots': INT, 'training_examples': INT, 'uncovered_slots': INT,
        'steps_per_epoch': INT, 'examples_per_step_nominal': INT, 'microbatch': INT, 'examples_per_mask': INT,
        'gold_order_steps': INT, 'later_epochs': TEXT, 'loss': TEXT, 'targets': TEXT,
    },
    'training': {'steps': INT, 'warmup_steps': INT, 'schedule': TEXT, 'freeze_after_step': INT, 'freeze_rule': TEXT},
    'objective': {'lambda_regime': TEXT, 'l0_lambda': NUM, 'l0_term': TEXT, 'run_b_trigger_below': INT},
    'optimizer': {
        'name': TEXT, 'implementation': TEXT,
        'reads': {'betas': [NUM], 'eps': NUM, 'weight_decay': NUM, 'clip_norm': NUM},
        'masks': {'lr': NUM, 'betas': [NUM], 'eps': NUM, 'weight_decay': NUM, 'clip': TEXT},
    },
    'hard_concrete': {'beta': NUM, 'gamma': NUM, 'zeta': NUM, 'log_alpha_init': NUM, 'read_init': TEXT},
    'seeds': {'data_order_base': INT, 'mask_noise_offset': INT},
    'runtime': {
        'dtype': TEXT, 'precision': TEXT, 'attention': TEXT, 'kv_cache': TEXT,
        'gradient_checkpointing': TEXT, 'determinism': TEXT, 'cublas_workspace': TEXT,
    },
    'checkpoints': [INT],
    'cadences': {
        'step_metrics_every': INT, 'validation_ce_every': INT, 'gate_snapshot_every': INT,
        'settle': {'first': INT, 'every': INT, 'jaccard_min': NUM, 'expected_change_max': NUM},
    },
    'storage': {'checkpoint_contents': [TEXT], 'optimizer_state': TEXT, 'interruption': TEXT, 'gate_snapshot_dtype': TEXT},
    'evaluation': {
        'validation': {'questions': INT, 'targets': TEXT, 'ce_masks': TEXT, 'greedy_at': TEXT},
        'cohort150': {'steps': [INT], 'role': TEXT},
        'cap': INT, 'eos_token_id': INT, 'repetition_penalty': NUM, 'scorer': TEXT, 'batch_size': INT,
    },
    'qualification': {
        'memory_fraction': NUM, 'disposable_steps': INT, 'greedy_questions': INT,
        'cached_vs_forced_rows': INT, 'cached_vs_forced_abs_max': NUM, 'recompute_rel_max': NUM,
    },
    'routing': {
        'gate_receipt': TEXT,
        'headroom_summary': TEXT,
    },
    'decisions': {'*': TEXT},
    'inputs': {
        'artifact_root': TEXT, 'bundle': TEXT, 'model_dir': TEXT, 'work_dir': TEXT,
        'k2_training_data': TEXT, 'k2_lora_adapter': TEXT, 'gold_order': TEXT, 'val128_k2_targets': TEXT,
    },
}

USER_FILLED = (
    'code_commit', 'executable_sha256', 'pod_id', 'objective.l0_lambda',
    'routing.gate_receipt', 'routing.headroom_summary',
    'inputs.artifact_root', 'inputs.bundle', 'inputs.model_dir', 'inputs.work_dir', 'inputs.val128_k2_targets',
)
REQUIRED_DECISIONS = ('K2-D1', 'K2-D2', 'K2-D3', 'K2-D4', 'K2-D5')

RULES = {
    'model.layers': sp.LAYERS, 'model.hidden': sp.HIDDEN, 'model.intermediate': sp.INTERMEDIATE,
    'model.parts': ['gate', 'up'], 'model.positions': 'all-unpadded',
    'data.recipe': 'sampling-sft-k2', 'data.k': 2,
    'data.microbatch': 1, 'data.later_epochs': 'reshuffle-64-batches-per-epoch',
    'data.loss': 'gold-completion-loss-step-token-denominator',
    'data.targets': 'sampling-sft-k2-stored-ids',
    'training.schedule': 'linear-warmup-cosine-to-zero', 'training.freeze_rule': 'z_det>0',
    'objective.l0_term': 'once-per-step-expected-active',
    'optimizer.name': 'adamw', 'optimizer.implementation': 'foreach-off',
    'optimizer.masks.clip': 'none',
    'hard_concrete.read_init': 'a~N(0,1)/sqrt(d);b=0',
    'runtime.dtype': 'float32', 'runtime.precision': 'ieee', 'runtime.attention': 'eager',
    'runtime.kv_cache': 'off-in-training', 'runtime.gradient_checkpointing': 'non-reentrant',
    'runtime.determinism': 'full', 'runtime.cublas_workspace': ':4096:8',
    'storage.checkpoint_contents': ['read_factors', 'log_alpha', 'seal'],
    'storage.optimizer_state': 'none', 'storage.interruption': 'restart-from-step-0',
    'storage.gate_snapshot_dtype': 'float16',
    'evaluation.validation.targets': 'val128-k2-targets',
    'evaluation.validation.ce_masks': 'deterministic', 'evaluation.validation.greedy_at': 'checkpoints',
    'evaluation.repetition_penalty': 1.0, 'evaluation.scorer': 'flexible-v3',
}

PRODUCTION = {
    'hardware': {'gpu': 'A100-80GB', 'count': 1},
    'data.total_slots': 512, 'data.training_examples': 477, 'data.uncovered_slots': 35,
    'data.steps_per_epoch': 64, 'data.examples_per_step_nominal': 8, 'data.examples_per_mask': 4,
    'data.gold_order_steps': 64,
    'training.steps': 1024, 'training.warmup_steps': 20, 'training.freeze_after_step': 960,
    'objective.run_b_trigger_below': RUN_B_SURVIVOR_THRESHOLD,
    'optimizer.reads.betas': [0.9, 0.99], 'optimizer.reads.eps': 1.0e-8,
    'optimizer.reads.weight_decay': 0.0, 'optimizer.reads.clip_norm': 1.0,
    'optimizer.masks.lr': 0.1, 'optimizer.masks.betas': [0.9, 0.99],
    'optimizer.masks.eps': 1.0e-8, 'optimizer.masks.weight_decay': 0.0,
    'hard_concrete.beta': 2.0 / 3.0, 'hard_concrete.gamma': -0.1,
    'hard_concrete.zeta': 1.1, 'hard_concrete.log_alpha_init': 1.0,
    'seeds.data_order_base': 42, 'seeds.mask_noise_offset': 1000,
    'checkpoints': [0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024],
    'cadences.step_metrics_every': 1, 'cadences.validation_ce_every': 16, 'cadences.gate_snapshot_every': 16,
    'cadences.settle.first': 256, 'cadences.settle.every': 64,
    'cadences.settle.jaccard_min': 0.99, 'cadences.settle.expected_change_max': 0.02,
    'evaluation.validation.questions': 128,
    'evaluation.cohort150.steps': [64, 1024], 'evaluation.cohort150.role': 'evaluation-only',
    'evaluation.cap': 1024, 'evaluation.eos_token_id': 151645, 'evaluation.batch_size': 16,
}


# --- executable code identity (P1 #2) ----------------------------------------------------------------

def _discover_executable_files(repo_root: Path) -> list[str]:
    """Return sorted relative paths of every Python source file under ``pilot_eval/`` and ``scripts/``."""
    found: list[str] = []
    for sub in EXECUTABLE_ROOTS:
        d = repo_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob('*.py')):
            if '__pycache__' in p.parts:
                continue
            found.append(p.relative_to(repo_root).as_posix())
    if not found:
        raise LaunchRefused(f'no executable Python sources found under {EXECUTABLE_ROOTS} in {repo_root}')
    return sorted(set(found))


def executable_source_hashes(repo_root: str | Path = REPO) -> dict[str, str]:
    """Return SHA-256 of EVERY Python source file under ``pilot_eval/`` and ``scripts/``."""
    root = Path(repo_root)
    files = _discover_executable_files(root)
    return {rel: file_hash(root / rel) for rel in files}


def executable_code_sha256(repo_root: str | Path = REPO) -> str:
    """Single canonical digest over all executable Python sources under ``pilot_eval/`` and ``scripts/``."""
    return _hash(executable_source_hashes(repo_root))


def verified_code_identity(repo_root: str | Path = REPO, *, require_clean: bool = True) -> dict[str, Any]:
    """Return committed git HEAD and full executable-tree digest; refuse dirty/untracked sources under
    ``pilot_eval/`` and ``scripts/`` while permitting unrelated local caches (e.g. ``notebooks/cache/``)."""
    root = Path(repo_root)
    head_proc = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=root, check=False
    )
    if head_proc.returncode != 0 or not head_proc.stdout.strip():
        raise LaunchRefused(f'cannot determine git HEAD in {root}: {head_proc.stderr.strip()}')
    commit = head_proc.stdout.strip()
    sources = executable_source_hashes(root)

    if require_clean:
        dirty: list[str] = []
        st_proc = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=all', '--', *EXECUTABLE_ROOTS],
            capture_output=True, text=True, cwd=root, check=False,
        )
        if st_proc.returncode != 0:
            raise LaunchRefused(f'git status failed in {root}: {st_proc.stderr.strip()}')
        for line in st_proc.stdout.splitlines():
            if line.strip():
                dirty.append(line.strip())
        # Verify every discovered executable source exists in HEAD with zero uncommitted drift
        for rel in sources:
            in_head = subprocess.run(
                ['git', 'cat-file', '-e', f'HEAD:{rel}'], capture_output=True, text=True, cwd=root, check=False
            )
            if in_head.returncode != 0:
                dirty.append(f'not-in-HEAD:{rel}')
        if dirty:
            raise LaunchRefused(
                'refusing launch/seal on uncommitted or modified executable sources under pilot_eval/ or scripts/; '
                'commit them cleanly first: ' + ', '.join(sorted(set(dirty))[:10])
            )
    return {
        'code_commit': commit,
        'executable_sha256': _hash(sources),
        'sources_sha256': sources,
    }


# --- calibration & trigger helpers (K2-D1) -----------------------------------------------------------

def compute_scaled_lambda(
    ce_untuned_k2: float,
    ce_lora_k2: float,
    *,
    base_lambda: float = GOLD_MATCHED_LAMBDA,
    gold_headroom: float = GOLD_VAL_CE_HEADROOM,
) -> dict[str, float]:
    """Compute headroom-scaled lambda_k2 from Stage 1 val128-k2 reference CE measurements."""
    delta_ce = float(ce_untuned_k2) - float(ce_lora_k2)
    if delta_ce <= 0.0:
        raise ValueError(f'non-positive k=2 CE headroom: untuned={ce_untuned_k2}, lora={ce_lora_k2}')
    ratio = delta_ce / float(gold_headroom)
    return {
        'ce_untuned_k2': float(ce_untuned_k2),
        'ce_lora_k2': float(ce_lora_k2),
        'delta_ce_k2': delta_ce,
        'gold_headroom': float(gold_headroom),
        'headroom_ratio': ratio,
        'lambda_matched': float(base_lambda),
        'lambda_scaled': float(base_lambda) * ratio,
    }


def should_trigger_run_b(survivor_count: int, *, threshold: int = RUN_B_SURVIVOR_THRESHOLD) -> bool:
    """K2-D1 decision rule: trigger Run B if Run A ends with fewer than ``threshold`` binary survivors."""
    if survivor_count < 0:
        raise ValueError(f'survivor_count must be >= 0, got {survivor_count}')
    return int(survivor_count) < int(threshold)


# --- validation --------------------------------------------------------------------------------------

def validate(
    record: dict[str, Any],
    *,
    code_commit: str,
    executable_sha256: str | None = None,
) -> dict[str, Any]:
    problems = cc.typecheck(record, SCHEMA)
    if problems:
        raise LaunchRefused(f'k2 launch record problems: {problems[:10]}')
    if record['schema'] != LAUNCH_SCHEMA:
        raise LaunchRefused('not a sparse-rewiring-k2 launch record')
    if record['code_commit'] != code_commit:
        raise LaunchRefused(f"code commit {code_commit} differs from record's {record['code_commit']}")
    expected_exec_sha = executable_sha256 if executable_sha256 is not None else executable_code_sha256()
    if record['executable_sha256'] != expected_exec_sha:
        raise LaunchRefused(
            f"executable_sha256 {expected_exec_sha} differs from record's {record['executable_sha256']}; "
            'executable sources changed since sealing'
        )
    mode = record['mode']
    if mode not in ('rehearsal', 'production'):
        raise LaunchRefused('mode must be production or rehearsal')
    if mode == 'rehearsal' and not record['label'].startswith('rehearsal'):
        raise LaunchRefused('a rehearsal needs a label starting with "rehearsal"')
    if mode == 'production' and not record['label'].startswith('production'):
        raise LaunchRefused('a production record needs a label starting with "production"')
    for dotted, expected in RULES.items():
        if not sp._same(sp._get(record, dotted), expected):
            raise LaunchRefused(f'{dotted} must be {expected!r} (spec; fixed in every mode)')
    if mode == 'production':
        for dotted, expected in PRODUCTION.items():
            actual = sp._get(record, dotted)
            if not sp._same(actual, expected):
                raise LaunchRefused(f'production {dotted} must be {expected!r}, got {actual!r}')

    obj = record['objective']
    if obj['lambda_regime'] not in ('matched', 'scaled'):
        raise LaunchRefused('objective.lambda_regime must be "matched" or "scaled"')
    if obj['l0_lambda'] <= 0:
        raise LaunchRefused('objective.l0_lambda must be positive')
    if mode == 'production' and obj['lambda_regime'] == 'matched' and not sp._same(obj['l0_lambda'], GOLD_MATCHED_LAMBDA):
        raise LaunchRefused(f'matched regime requires l0_lambda == {GOLD_MATCHED_LAMBDA}')
    routing = record['routing']
    if obj['lambda_regime'] == 'scaled':
        if not routing['gate_receipt'] or routing['gate_receipt'] == 'none':
            raise LaunchRefused('scaled regime (Run B) requires routing.gate_receipt from verified Run A seed 42')
        if not routing['headroom_summary'] or routing['headroom_summary'] == 'none':
            raise LaunchRefused('scaled regime (Run B) requires routing.headroom_summary from sealed Stage 1 references')

    missing = [d for d in REQUIRED_DECISIONS if d not in record['decisions']]
    if missing:
        raise LaunchRefused(f'decisions missing: {missing}')

    _check_arms(record)
    _check_shape(record)

    planned = sp.planned_disk_bytes(record) / 1e9
    if record['disk']['container_total_gb'] < planned + record['disk']['min_free_gb']:
        raise LaunchRefused(
            f'disk.container_total_gb {record["disk"]["container_total_gb"]} is below planned '
            f'{planned:.2f} GB plus min_free_gb {record["disk"]["min_free_gb"]}'
        )
    cc.stop_commands(record)
    return record


def _check_arms(record: dict[str, Any]) -> None:
    arms = record['arms']
    if not arms:
        raise LaunchRefused('arms must be non-empty')
    seen_run_ids, seen_seeds = set(), set()
    for name, arm in arms.items():
        if arm['rank'] != 1:
            raise LaunchRefused(f'{name}: K2-D2 fixes rank == 1, got {arm["rank"]}')
        if not arm['read_lr'] > 0:
            raise LaunchRefused(f'{name}.read_lr must be positive')
        if record['mode'] == 'production' and not sp._same(arm['read_lr'], 3.0e-3):
            raise LaunchRefused(f'{name}: production Rank 1 read_lr must be 3.0e-3 (K2-D2)')
        if not str(arm['run_id']).replace('-', '').isalnum():
            raise LaunchRefused(f"run_id {arm['run_id']!r} must be alphanumeric/hyphen")
        if arm['run_id'] in seen_run_ids:
            raise LaunchRefused('each arm needs its own run_id')
        seen_run_ids.add(arm['run_id'])
        if arm['seed'] in seen_seeds:
            raise LaunchRefused('each arm needs its own distinct seed')
        seen_seeds.add(arm['seed'])
    if record['mode'] == 'production' and sorted(seen_seeds) != [42, 43]:
        raise LaunchRefused(
            f'production K2-D2 requires exact two-seed set [42, 43], got {sorted(seen_seeds)}'
        )


def _check_shape(record: dict[str, Any]) -> None:
    data, training, cad = record['data'], record['training'], record['cadences']
    steps, warmup, freeze = training['steps'], training['warmup_steps'], training['freeze_after_step']
    if not 0 < warmup < freeze < steps:
        raise LaunchRefused('need 0 < warmup_steps < freeze_after_step < steps')
    if data['training_examples'] + data['uncovered_slots'] != data['total_slots']:
        raise LaunchRefused('training_examples + uncovered_slots must equal total_slots')
    if data['total_slots'] % data['steps_per_epoch'] != 0:
        raise LaunchRefused('total_slots must divide evenly by steps_per_epoch')
    if data['training_examples'] < data['steps_per_epoch']:
        raise LaunchRefused('training_examples must be >= steps_per_epoch so every batch is non-empty')
    if data['examples_per_mask'] < 1:
        raise LaunchRefused('examples_per_mask must be >= 1')
    if data['gold_order_steps'] != min(data['steps_per_epoch'], steps):
        raise LaunchRefused('gold_order_steps must equal min(steps_per_epoch, steps)')
    points = record['checkpoints']
    if points != sorted(set(points)) or points[0] != 0 or points[-1] != steps or freeze not in points:
        raise LaunchRefused('checkpoints must be increasing, start at 0, end at steps and include freeze')
    for step in record['evaluation']['cohort150']['steps']:
        if step not in points:
            raise LaunchRefused(f'cohort150 step {step} is not a checkpoint')
    for key in ('step_metrics_every', 'validation_ce_every', 'gate_snapshot_every'):
        if not 0 < cad[key] <= steps:
            raise LaunchRefused(f'cadences.{key} must be in 1..steps')
    settle = cad['settle']
    if not (0 < settle['every'] and 0 < settle['first'] <= freeze and 0 < settle['jaccard_min'] <= 1
            and settle['expected_change_max'] > 0):
        raise LaunchRefused('cadences.settle is inconsistent with freeze or out of range')
    groups = record['optimizer']
    for part in ('reads', 'masks'):
        if len(groups[part]['betas']) != 2 or not all(0 <= b < 1 for b in groups[part]['betas']):
            raise LaunchRefused(f'optimizer.{part}.betas must be two numbers in [0, 1)')
    if groups['reads']['clip_norm'] <= 0:
        raise LaunchRefused('optimizer.reads.clip_norm must be positive')
    hc = record['hard_concrete']
    if not (hc['gamma'] < 0 < 1 < hc['zeta'] and hc['beta'] > 0):
        raise LaunchRefused('hard concrete needs gamma < 0, zeta > 1, beta > 0')
    q = record['qualification']
    if not 0 < q['memory_fraction'] <= 1:
        raise LaunchRefused('qualification.memory_fraction must be in (0, 1]')
    if not 0 < q['disposable_steps'] <= min(steps, data['gold_order_steps']):
        raise LaunchRefused('qualification.disposable_steps must be in 1..gold_order_steps')
    if not 1 <= q['greedy_questions'] <= record['evaluation']['validation']['questions']:
        raise LaunchRefused('qualification.greedy_questions must be in 1..evaluation.validation.questions')
    if q['cached_vs_forced_rows'] < 1 or q['cached_vs_forced_abs_max'] <= 0 or q['recompute_rel_max'] < 0:
        raise LaunchRefused('qualification tolerances out of range')


def load_launch_record(
    path: str | Path,
    *,
    code_commit: str,
    executable_sha256: str | None = None,
) -> dict[str, Any]:
    return cc.load_launch_record(
        path,
        code_commit=code_commit,
        validator=lambda r, *, code_commit: validate(r, code_commit=code_commit, executable_sha256=executable_sha256),
    )


def finalize_draft(
    draft_path: str | Path,
    out_path: str | Path,
    *,
    code_commit: str,
    executable_sha256: str | None = None,
    pod_id: str | None = None,
    fill: dict[str, Any] | None = None,
) -> Path:
    import tempfile
    import yaml

    exec_sha = executable_sha256 if executable_sha256 is not None else executable_code_sha256()
    merged_fill = dict(fill or {})
    merged_fill.setdefault('executable_sha256', exec_sha)

    # cc.finalize_draft casts `fill` values to str; inject numeric `objective.l0_lambda` directly as float
    numeric_lambda = merged_fill.pop('objective.l0_lambda', None)
    if numeric_lambda is not None:
        raw = yaml.safe_load(Path(draft_path).read_text(encoding='utf-8'))
        obj = raw.get('objective') if isinstance(raw, dict) else None
        if not isinstance(obj, dict):
            raise LaunchRefused('draft missing objective mapping')
        if obj.get('l0_lambda') is not None and not sp._same(obj['l0_lambda'], float(numeric_lambda)):
            raise LaunchRefused(
                f"draft already sets objective.l0_lambda={obj['l0_lambda']!r}; refusing overwrite with {numeric_lambda!r}"
            )
        obj['l0_lambda'] = float(numeric_lambda)
        with tempfile.TemporaryDirectory() as td:
            tmp_draft = Path(td) / Path(draft_path).name
            tmp_draft.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
            return cc.finalize_draft(
                tmp_draft,
                out_path,
                code_commit=code_commit,
                pod_id=pod_id,
                fill=merged_fill,
                validator=lambda r, *, code_commit: validate(r, code_commit=code_commit, executable_sha256=exec_sha),
                fillable=USER_FILLED,
            )

    return cc.finalize_draft(
        draft_path,
        out_path,
        code_commit=code_commit,
        pod_id=pod_id,
        fill=merged_fill,
        validator=lambda r, *, code_commit: validate(r, code_commit=code_commit, executable_sha256=exec_sha),
        fillable=USER_FILLED,
    )


def draft_problems(draft_path: str | Path) -> list[str]:
    import yaml
    record = yaml.safe_load(Path(draft_path).read_text())
    filled = set(USER_FILLED)
    return [p for p in cc.typecheck(record, SCHEMA)
            if not (p.endswith('(unresolved: null)') and p.split(' ')[0] in filled)]


# --- K2-D3 batching & sealed training export verification --------------------------------------------

def _check_stored_row_contract(row: Mapping[str, Any], *, label: str, eos_token_id: int | None = None) -> None:
    inp = row.get('input_ids')
    lbl = row.get('labels')
    ptok = row.get('prompt_tokens')
    if (not isinstance(inp, list) or not isinstance(lbl, list) or len(inp) != len(lbl)
            or type(ptok) is not int or not 0 < ptok < len(inp)):
        raise LaunchRefused(f'{label}: malformed input_ids/labels/prompt_tokens')
    if any(type(t) is not int or t < 0 for t in inp):
        raise LaunchRefused(f'{label}: input_ids must be non-negative ints')
    if lbl[:ptok] != [-100] * ptok:
        raise LaunchRefused(f'{label}: prompt labels must all be -100')
    comp_inp = inp[ptok:]
    comp_lbl = lbl[ptok:]
    if comp_lbl != comp_inp:
        raise LaunchRefused(f'{label}: completion labels must exactly match stored completion input_ids')
    if eos_token_id is not None:
        eos = int(eos_token_id)
        if comp_inp[-1] != eos or comp_inp.count(eos) != 1:
            raise LaunchRefused(f'{label}: completion must end with exactly one EOS ({eos})')


def load_k2_training_rows(
    path: str | Path,
    *,
    expected_export_seal_sha256: str | None = None,
    gold_order: Sequence[str] | None = None,
    eos_token_id: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify a sealed ``sampling_sft_training_data`` k = 2 export directory.

    Refuses direct raw JSON/JSONL files and refuses any directory missing ``export.complete.json``
    or any required sealed file (``arms/k2/training.jsonl``, ``arms/k2/batches.json``,
    ``arms/k2/coverage.json``). Reconciles covered/uncovered IDs and step slots against
    ``batches.json``, ``coverage.json``, and (when supplied) ``gold_order``.
    """
    export_dir = Path(path)
    if not export_dir.is_dir():
        raise LaunchRefused(
            f'k2 training data must be a sealed export directory (e.g. l4-training-data-v1), got: {export_dir}'
        )
    seal_path = export_dir / 'export.complete.json'
    if not seal_path.is_file():
        raise LaunchRefused(f'{export_dir}: missing export.complete.json seal')
    seal_sha = file_hash(seal_path)
    if expected_export_seal_sha256 is not None and seal_sha != expected_export_seal_sha256:
        raise LaunchRefused(
            f'{seal_path}: SHA-256 {seal_sha} differs from pinned {expected_export_seal_sha256}'
        )
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    sealed_files = seal.get('files')
    if not isinstance(sealed_files, dict) or not sealed_files:
        raise LaunchRefused(f'{seal_path}: malformed or empty files manifest')

    observed_hashes: dict[str, str] = {}
    for rel in K2_EXPORT_FILES_REQUIRED:
        if rel not in sealed_files:
            raise LaunchRefused(f'{seal_path}: missing required sealed entry {rel!r}')
        fpath = export_dir / rel
        if not fpath.is_file():
            raise LaunchRefused(f'{export_dir}: sealed file missing on disk: {rel}')
        actual = file_hash(fpath)
        if actual != sealed_files[rel]:
            raise LaunchRefused(f'{fpath}: SHA-256 {actual} differs from export.complete.json ({sealed_files[rel]})')
        observed_hashes[rel] = actual

    rows_file = export_dir / 'arms/k2/training.jsonl'
    batches_file = export_dir / 'arms/k2/batches.json'
    coverage_file = export_dir / 'arms/k2/coverage.json'

    rows = [json.loads(line) for line in rows_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    batches = json.loads(batches_file.read_text(encoding='utf-8'))
    coverage = json.loads(coverage_file.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise LaunchRefused(f'{rows_file}: expected non-empty JSONL rows')
    if not isinstance(batches, list) or not batches:
        raise LaunchRefused(f'{batches_file}: expected non-empty batch schedule list')
    if not isinstance(coverage, dict) or not isinstance(coverage.get('uncovered_ids'), list):
        raise LaunchRefused(f'{coverage_file}: missing uncovered_ids list')

    covered_ids: list[str] = []
    for idx, r in enumerate(rows):
        if not isinstance(r, dict) or not isinstance(r.get('id'), str) or not r['id']:
            raise LaunchRefused(f'{rows_file}: row {idx} missing valid id')
        _check_stored_row_contract(r, label=f'{rows_file}[{idx}:{r["id"]}]', eos_token_id=eos_token_id)
        covered_ids.append(r['id'])
    if len(set(covered_ids)) != len(covered_ids):
        raise LaunchRefused(f'{rows_file}: duplicate row IDs')

    uncovered_ids = list(coverage['uncovered_ids'])
    if len(set(uncovered_ids)) != len(uncovered_ids):
        raise LaunchRefused(f'{coverage_file}: duplicate uncovered_ids')
    if set(covered_ids) & set(uncovered_ids):
        raise LaunchRefused(f'{coverage_file}: covered_ids and uncovered_ids overlap')

    # Reconcile step slots in batches.json against rows and uncovered_ids
    batch_covered: list[str] = []
    batch_uncovered: list[str] = []
    batch_all_slots: list[str] = []
    for b_idx, batch in enumerate(batches, 1):
        slots = batch.get('slots')
        if not isinstance(slots, list) or not slots:
            raise LaunchRefused(f'{batches_file}: batch {b_idx} missing slots')
        nonmissing = [s for s in slots if not s.get('missing')]
        if not nonmissing:
            raise LaunchRefused(f'{batches_file}: batch {b_idx} has zero non-missing slots')
        if int(batch.get('nonmissing_count', -1)) != len(nonmissing):
            raise LaunchRefused(f'{batches_file}: batch {b_idx} nonmissing_count mismatch')
        for s_idx, slot in enumerate(slots):
            sid = slot.get('source_id')
            if not isinstance(sid, str) or not sid:
                raise LaunchRefused(f'{batches_file}: batch {b_idx} slot {s_idx} missing source_id')
            batch_all_slots.append(sid)
            if slot.get('missing'):
                batch_uncovered.append(sid)
            else:
                batch_covered.append(sid)

    if batch_covered != covered_ids:
        raise LaunchRefused(f'{rows_file}: covered row IDs/order do not reconcile with arms/k2/batches.json')
    if set(batch_uncovered) != set(uncovered_ids) or len(batch_uncovered) != len(uncovered_ids):
        raise LaunchRefused(f'{coverage_file}: uncovered_ids do not reconcile with arms/k2/batches.json')
    if gold_order is not None and batch_all_slots != list(gold_order):
        raise LaunchRefused(f'{batches_file}: batch slot sequence does not match Gold saved order.json')

    meta = {
        'export_dir': str(export_dir),
        'k2_export_seal_sha256': seal_sha,
        'k2_training_rows_sha256': observed_hashes['arms/k2/training.jsonl'],
        'k2_batches_sha256': observed_hashes['arms/k2/batches.json'],
        'k2_coverage_sha256': observed_hashes['arms/k2/coverage.json'],
        'covered_count': len(covered_ids),
        'uncovered_count': len(uncovered_ids),
    }
    return rows, meta


def partition_epoch_batches(ids: list[str], num_batches: int) -> list[list[str]]:
    """Partition ``ids`` into exactly ``num_batches`` non-empty batches whose sizes differ by at most 1.

    For 477 items and 64 batches: 29 batches of size 8 + 35 batches of size 7 = 477 (K2-D3).
    """
    n = len(ids)
    if num_batches <= 0 or n < num_batches:
        raise ValueError(f'cannot partition {n} items into {num_batches} non-empty batches')
    q, r = divmod(n, num_batches)
    batches: list[list[str]] = []
    pos = 0
    for b in range(num_batches):
        size = q + 1 if b < r else q
        batches.append(ids[pos:pos + size])
        pos += size
    return batches


def full_order_k2(
    covered_ids: list[str],
    gold_order: list[str],
    *,
    steps: int,
    steps_per_epoch: int,
    nominal_batch_size: int,
    base_seed: int,
) -> list[list[str]]:
    """Build the 1..steps batch schedule for Sampling SFT k = 2 (K2-D3).

    - Epoch 1 (steps 1..steps_per_epoch): Gold's exact saved 64 batches of ``nominal_batch_size``,
      filtering out uncovered slots while preserving order and refusing any empty batch.
    - Epoch e >= 2: ``covered_ids`` (in canonical input order) shuffled with
      ``random.Random(base_seed + e - 1)`` and partitioned into ``steps_per_epoch`` balanced
      batches (for 477 items / 64 steps: 29 batches of 8 + 35 batches of 7).
    """
    covered_ids = list(covered_ids)
    if len(set(covered_ids)) != len(covered_ids):
        raise ValueError('covered_ids are not unique')
    covered_set = set(covered_ids)
    gold_order = list(gold_order)
    if len(gold_order) != steps_per_epoch * nominal_batch_size:
        raise ValueError(
            f'gold_order length {len(gold_order)} != steps_per_epoch * nominal_batch_size '
            f'({steps_per_epoch * nominal_batch_size})'
        )
    if not covered_set.issubset(set(gold_order)):
        raise ValueError('covered_ids contains IDs not present in gold_order')

    epoch1: list[list[str]] = []
    for s in range(steps_per_epoch):
        slot_ids = gold_order[s * nominal_batch_size:(s + 1) * nominal_batch_size]
        kept = [eid for eid in slot_ids if eid in covered_set]
        if not kept:
            raise ValueError(f'epoch 1 batch {s + 1} is empty after omitting uncovered slots')
        epoch1.append(kept)

    all_batches: list[list[str]] = list(epoch1)
    epochs = -(-steps // steps_per_epoch)
    for epoch in range(2, epochs + 1):
        shuffled = list(covered_ids)
        random.Random(base_seed + epoch - 1).shuffle(shuffled)
        all_batches.extend(partition_epoch_batches(shuffled, steps_per_epoch))
    return all_batches[:steps]


def build_plan(
    record: dict[str, Any],
    arm: str,
    covered_ids: list[str],
    gold_order: list[str],
    *,
    inputs_sha256: dict[str, Any],
) -> dict[str, Any]:
    """Build a frozen per-arm plan compatible with ``rewire_training.train_rewire`` and ``rewire_checkpoint``."""
    if arm not in record['arms']:
        raise ValueError(f'unknown arm {arm}')
    data, training = record['data'], record['training']
    if len(covered_ids) != data['training_examples']:
        raise ValueError(f'{len(covered_ids)} covered IDs; record says {data["training_examples"]}')
    uncovered_ids = [eid for eid in gold_order if eid not in set(covered_ids)]
    if len(uncovered_ids) != data['uncovered_slots']:
        raise ValueError(f'{len(uncovered_ids)} uncovered IDs; record says {data["uncovered_slots"]}')

    arm_record = record['arms'][arm]
    arm_seed = int(arm_record['seed'])
    total, warmup, freeze = training['steps'], training['warmup_steps'], training['freeze_after_step']
    epoch_steps = data['steps_per_epoch']

    order = full_order_k2(
        covered_ids,
        gold_order,
        steps=total,
        steps_per_epoch=epoch_steps,
        nominal_batch_size=data['examples_per_step_nominal'],
        base_seed=record['seeds']['data_order_base'],
    )
    steps = [
        dict(
            step=s,
            epoch=(s - 1) // epoch_steps + 1,
            ids=ids,
            batch_size=len(ids),
            mask_groups=sp._groups(len(ids), data['examples_per_mask']),
            lr_multiplier=sp.lr_multiplier(s, warmup=warmup, total=total),
            masks='trainable' if s <= freeze else 'frozen-binary',
        )
        for s, ids in enumerate(order, 1)
    ]

    arm_seeds = dict(
        data_order_base=int(record['seeds']['data_order_base']),
        read_init_base=arm_seed,
        mask_noise=arm_seed + int(record['seeds']['mask_noise_offset']),
    )
    settings = {
        k: record[k]
        for k in ('model', 'data', 'training', 'objective', 'hard_concrete', 'runtime', 'storage', 'evaluation', 'decisions')
    }
    settings['seeds'] = arm_seeds

    plan = dict(
        schema=PLAN_SCHEMA,
        arm=arm,
        run_id=arm_record['run_id'],
        rank=arm_record['rank'],
        read_lr=arm_record['read_lr'],
        seed=arm_seed,
        launch_run_sha256=cc.run_sha256(record),
        executable_sha256=record['executable_sha256'],
        settings=settings,
        optimizer=dict(record['optimizer'], reads=dict(record['optimizer']['reads'], lr=arm_record['read_lr'])),
        read_init_seeds=[arm_seed + layer for layer in range(record['model']['layers'])],
        trainable_parameters=dict(
            reads=sp.read_parameters(
                arm_record['rank'],
                layers=record['model']['layers'],
                hidden=record['model']['hidden'],
                intermediate=record['model']['intermediate'],
            ),
            log_alpha=record['model']['layers'] * record['model']['intermediate'],
        ),
        checkpoints=list(record['checkpoints']),
        checkpoint_bytes=sp.checkpoint_bytes(
            arm_record['rank'],
            layers=record['model']['layers'],
            hidden=record['model']['hidden'],
            intermediate=record['model']['intermediate'],
        ),
        validation_steps=sp.validation_steps(record),
        gate_snapshot_steps=sp.gate_snapshot_steps(record),
        settle_steps=sp.settle_steps(record),
        cohort150_steps=list(record['evaluation']['cohort150']['steps']),
        inputs=dict(
            inputs_sha256,
            covered_ids=list(covered_ids),
            uncovered_ids=uncovered_ids,
            gold_order=list(gold_order),
        ),
        steps=steps,
    )
    plan['plan_sha256'] = sp.plan_sha256(plan)
    return plan


def plan_from_files(
    record: dict[str, Any],
    arm: str,
    k2_export_dir: str | Path,
    gold_order_path: str | Path,
    *,
    expected_export_seal_sha256: str | None = None,
    extra_inputs_sha256: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a frozen k = 2 per-arm plan from a verified sealed k = 2 training export and Gold's order.json."""
    order_file = Path(gold_order_path)
    gold_order = json.loads(order_file.read_text(encoding='utf-8'))
    if not isinstance(gold_order, list) or not all(isinstance(x, str) for x in gold_order):
        raise LaunchRefused(f'{order_file}: expected a JSON list of question IDs')
    eos = int(record['evaluation']['eos_token_id'])
    rows, export_meta = load_k2_training_rows(
        k2_export_dir,
        expected_export_seal_sha256=expected_export_seal_sha256,
        gold_order=gold_order,
        eos_token_id=eos,
    )
    covered_ids = [r['id'] for r in rows]
    digests: dict[str, Any] = {
        'k2_export_seal_sha256': export_meta['k2_export_seal_sha256'],
        'k2_training_rows_sha256': export_meta['k2_training_rows_sha256'],
        'k2_batches_sha256': export_meta['k2_batches_sha256'],
        'k2_coverage_sha256': export_meta['k2_coverage_sha256'],
        'gold_order_sha256': file_hash(order_file),
    }
    if extra_inputs_sha256:
        digests.update(extra_inputs_sha256)
    return build_plan(record, arm, covered_ids, gold_order, inputs_sha256=digests)


def load_plan(path: str | Path, record: dict[str, Any], arm: str) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != PLAN_SCHEMA or plan.get('plan_sha256') != sp.plan_sha256(plan):
        raise LaunchRefused(f'{path}: plan does not match its hash')
    if plan['arm'] != arm or plan['launch_run_sha256'] != cc.run_sha256(record):
        raise LaunchRefused(f'{path}: plan was built for another arm or launch record')
    extra_inputs = {k: v for k, v in plan['inputs'].items() if k not in ('covered_ids', 'uncovered_ids', 'gold_order')}
    rebuilt = build_plan(record, arm, plan['inputs']['covered_ids'], plan['inputs']['gold_order'],
                         inputs_sha256=extra_inputs)['plan_sha256']
    if rebuilt != plan['plan_sha256']:
        raise LaunchRefused(f'{path}: plan differs from what the launch record builds')
    return plan


# --- CLI ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m pilot_eval.sparse_rewiring_k2_plan')
    sub = parser.add_subparsers(dest='command', required=True)

    check = sub.add_parser('check', help='list unresolved or mistyped leaves of a YAML draft')
    check.add_argument('draft')

    seal = sub.add_parser('seal', help='approved YAML draft -> sealed, typed JSON record')
    seal.add_argument('--draft', required=True)
    seal.add_argument('--out', required=True)
    seal.add_argument('--code-commit', default=None)
    seal.add_argument('--pod-id')
    for name in ('artifact-root', 'bundle', 'model-dir', 'work-dir', 'val128-k2-targets'):
        seal.add_argument(f'--{name}')
    seal.add_argument('--l0-lambda', type=float, default=None)
    seal.add_argument('--gate-receipt', default=None)
    seal.add_argument('--headroom-summary', default=None)

    cal = sub.add_parser('calibrate-lambda', help='compute headroom-scaled lambda_k2 from Stage 1 val128-k2 CE')
    cal.add_argument('--ce-untuned', type=float, required=True)
    cal.add_argument('--ce-lora', type=float, required=True)

    args = parser.parse_args(argv)
    if args.command == 'check':
        problems = draft_problems(args.draft)
        print('\n'.join(problems) if problems else 'no problems')
        return 1 if problems else 0
    if args.command == 'calibrate-lambda':
        print(json.dumps(compute_scaled_lambda(args.ce_untuned, args.ce_lora), indent=2))
        return 0
    ident = verified_code_identity(require_clean=True)
    commit = args.code_commit or ident['code_commit']
    fill: dict[str, Any] = {
        f'inputs.{k}': getattr(args, k)
        for k in ('artifact_root', 'bundle', 'model_dir', 'work_dir', 'val128_k2_targets')
        if getattr(args, k) is not None
    }
    if args.l0_lambda is not None:
        fill['objective.l0_lambda'] = float(args.l0_lambda)
    if args.gate_receipt is not None:
        fill['routing.gate_receipt'] = args.gate_receipt
    if args.headroom_summary is not None:
        fill['routing.headroom_summary'] = args.headroom_summary
    print(finalize_draft(
        args.draft, args.out, code_commit=commit, executable_sha256=ident['executable_sha256'],
        pod_id=args.pod_id, fill=fill,
    ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
