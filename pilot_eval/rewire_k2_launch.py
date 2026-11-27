"""Sparse rewiring on Sampling SFT (k = 2) Pod launcher (rehearsal + Run A matched + conditional Run B scaled).

Enforces every P1 & P2 contract from ``.scratch/sparse-rewiring-sampling-k2/{design.md,spec.md}``:
- **Input & provenance integrity:** Requires a verified sealed ``l4-training-data-v1`` export
  (reconciling ``training.jsonl``, ``batches.json``, ``coverage.json``, and Gold's ``order.json``),
  verifies pinned dense ``k = 2`` checkpoint-64 adapter weights & config SHA-256 (binding both into
  every arm plan), and validates sealed ``val128-k2-targets.json`` provenance (matching pinned base
  model, tokenizer, and validation input hashes), full effective recipe (including ``max_blocks``,
  ``stop_token_ids``, ``accept_rule``, ``top_k == 0``, ``retry_data_seeds == [0, 1001, 1002, 1003]``),
  exact coverage accounting, prompt masking, completion label equality, and single final EOS.
- **Full executable tree binding:** Refuses ``seal``/``precheck``/``run`` when any Python file under
  ``pilot_eval/`` or ``scripts/`` is uncommitted or modified, and binds the full tree digest into
  every launch record and arm plan.
- **Registered conditional two-lambda routing:**
  - Production Run A schedules **seed 42 only** by default.
  - Production Run A ``seed 43`` requires a verified completed Run A ``seed 42`` arm matching the
    **exact registered matched seed-42 plan** with ``>= 4`` survivors.
  - Production Run B (``scaled``) requires a verified completed Run A ``seed 42`` arm matching the
    **registered matched seed-42 plan** (``l0_lambda == 6.0e-4``, exact hyperparameters & step schedule,
    matching executable/code provenance & inputs) with ``< 4`` survivors AND the **specific Stage 1
    headroom seal** bound by the gate receipt (comparing ``headroom_complete_sha256`` and
    ``headroom_file_sha256``, compatible with authentic volume restores).
- **Full reference bar:** Evaluates ``untuned`` (step 0) and dense ``sampling-sft-k2`` (step 64) on
  ``val128-k2`` CE, ``val128`` greedy accuracy + ``tokens_mean``, and (in production) ``cohort150``
  greedy accuracy + ``tokens_mean`` in an isolated IEEE FP32 child process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pilot_eval import attribution_chain as pc
from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_bundle as cb
from pilot_eval import c_attribution_chain as cc
from pilot_eval import catalog as cat
from pilot_eval import rewire_bundle as rb
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_launch as rl
from pilot_eval import sparse_rewiring_k2_plan as k2p
from pilot_eval.attribution_chain import LaunchRefused, ResumeRefused
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

K2_ADAPTER_ROLE = 'adapter:sampling-sft-k2:64'
VAL128_K2_SCHEMA = 'sparse-rewiring-k2/val128-targets-v1'
GATE_RECEIPT_SCHEMA = 'sparse-rewiring-k2/gate-receipt-v1'
SHARED_INPUT_KEYS = (
    'k2_export_seal_sha256',
    'k2_training_rows_sha256',
    'gold_order_sha256',
    'val128_k2_targets_sha256',
    'k2_lora_adapter_weights_sha256',
    'k2_lora_adapter_config_sha256',
)
EXPECTED_MAX_BLOCKS = 32
# Exact verified Stage 1 + Run A seed-42 commit prior to the D12 cohort150.steps=[1024] alignment fix
VERIFIED_PRIOR_STAGE1_RUN_A_COMMITS: dict[str, str] = {
    '127f8d7d866165ddad07e5606d4bb4125722dc0e': '46e16563e2932272979dea53f7b0c473950b26582c9464382ea2a0f129da5072',
}


def chain_record(record: Mapping[str, Any], *, only_arm: str | None = None) -> dict[str, Any]:
    from pilot_eval.rewire_chain import chain_run_id
    base_id = chain_run_id(record)
    if only_arm is not None:
        if only_arm not in record['arms']:
            raise LaunchRefused(f'unknown --only-arm {only_arm!r}; valid arms: {list(record["arms"])}')
        return dict(record, arms={only_arm: record['arms'][only_arm]}, run_id=f'{base_id}-{only_arm}')
    return dict(record, run_id=base_id)


def _resolve_under(root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else root / p


def _shared_inputs_subset(mapping: Mapping[str, Any]) -> dict[str, str]:
    return {k: str(mapping[k]) for k in SHARED_INPUT_KEYS if k in mapping}


def _check_shared_inputs_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> None:
    l_sub = _shared_inputs_subset(left)
    r_sub = _shared_inputs_subset(right)
    missing_l = [k for k in SHARED_INPUT_KEYS if k not in l_sub]
    missing_r = [k for k in SHARED_INPUT_KEYS if k not in r_sub]
    if missing_l:
        raise LaunchRefused(f'{left_label} missing bound input hashes: {missing_l}')
    if missing_r:
        raise LaunchRefused(f'{right_label} missing bound input hashes: {missing_r}')
    mismatches = [k for k in SHARED_INPUT_KEYS if l_sub[k] != r_sub[k]]
    if mismatches:
        raise LaunchRefused(
            f'{left_label} and {right_label} were built from different experiment inputs: {mismatches}'
        )


def _expected_bundle_tokenizer_hashes(bundle: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in bundle.get('files', []):
        role = str(entry.get('role', ''))
        if role.startswith('tokenizer:'):
            out[role.split(':', 1)[1]] = str(entry['sha256'])
    return out


def _expected_stop_ids_from_bundle(bundle: Mapping[str, Any], root: Path, eos_token_id: int) -> list[int]:
    """Derive exact expected stop token IDs from the bundled tokenizer."""
    from transformers import AutoTokenizer

    tok_entry = next((e for e in bundle.get('files', []) if str(e.get('role', '')).startswith('tokenizer:')), None)
    stops = [int(eos_token_id)]
    if tok_entry is not None:
        tok_dir = (root / tok_entry['path']).parent
        tok = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
        for stop_str in ('<|im_end|>', '<|endoftext|>'):
            tid = tok.convert_tokens_to_ids(stop_str)
            if isinstance(tid, int) and tid >= 0 and tid not in stops:
                stops.append(tid)
    return stops


# --- Strict input & provenance loaders (P1 #1 & P1 #4) -----------------------------------------------

def load_val128_k2_targets(
    path: str | Path,
    *,
    val_questions: Mapping[str, Mapping[str, Any]],
    expected_ids: Sequence[str],
    eos_token_id: int,
    expected_stop_ids: Sequence[int] | None = None,
    expected_max_blocks: int = EXPECTED_MAX_BLOCKS,
    expected_model_file_sha256: Mapping[str, str] | None = None,
    expected_tokenizer_file_sha256: Mapping[str, str] | None = None,
    expected_val_items_sha256: str | None = None,
    expected_engine: str | None = None,
    expected_executable_sha256: str | None = None,
    require_ieee_precision: bool = True,
    require_qualified_recipe: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and strictly verify sealed ``val128-k2-targets.json`` provenance, recipe, coverage, and rows."""
    target_path = Path(path)
    if not target_path.is_file():
        raise LaunchRefused(f'val128-k2 targets file does not exist: {target_path}')
    raw = json.loads(target_path.read_text(encoding='utf-8'))
    expected_ids_list = list(expected_ids)
    expected_set = set(expected_ids_list)
    expected_digest = cat.ids_digest(expected_ids_list)

    if not isinstance(raw, dict):
        raise LaunchRefused(f'{target_path}: unsealed raw list is refused; require {VAL128_K2_SCHEMA!r} dict')
    if raw.get('schema') != VAL128_K2_SCHEMA:
        raise LaunchRefused(f'{target_path}: unexpected schema {raw.get("schema")!r}, expected {VAL128_K2_SCHEMA!r}')
    if raw.get('ids_sha256') != expected_digest:
        raise LaunchRefused(
            f'{target_path}: ids_sha256 {raw.get("ids_sha256")!r} differs from frozen val128 {expected_digest!r}'
        )
    items = raw.get('items')
    if not isinstance(items, list) or not items:
        raise LaunchRefused(f'{target_path}: expected non-empty items list')
    if raw.get('items_sha256') != _hash(items):
        raise LaunchRefused(f'{target_path}: missing or mismatched items_sha256')

    # Exact cohort coverage accounting
    covered_ids = raw.get('covered_ids')
    uncovered_ids = raw.get('uncovered_ids')
    if not isinstance(covered_ids, list) or not isinstance(uncovered_ids, list):
        raise LaunchRefused(f'{target_path}: covered_ids and uncovered_ids must be lists')
    if raw.get('total_questions') != len(expected_ids_list):
        raise LaunchRefused(f'{target_path}: total_questions {raw.get("total_questions")} != {len(expected_ids_list)}')
    if raw.get('covered_count') != len(items) or raw.get('uncovered_count') != len(uncovered_ids):
        raise LaunchRefused(f'{target_path}: covered_count/uncovered_count mismatch')
    if len(set(covered_ids)) != len(covered_ids) or len(set(uncovered_ids)) != len(uncovered_ids):
        raise LaunchRefused(f'{target_path}: duplicate IDs in covered_ids or uncovered_ids')
    if set(covered_ids) & set(uncovered_ids):
        raise LaunchRefused(f'{target_path}: covered_ids and uncovered_ids overlap')
    if set(covered_ids) | set(uncovered_ids) != expected_set:
        raise LaunchRefused(f'{target_path}: covered_ids U uncovered_ids does not equal frozen val128 cohort')

    # Full effective recipe verification
    recipe = raw.get('recipe')
    if not isinstance(recipe, dict):
        raise LaunchRefused(f'{target_path}: missing recipe metadata')
    if require_qualified_recipe:
        exp_stops = list(expected_stop_ids) if expected_stop_ids is not None else [int(eos_token_id)]
        expected_recipe_checks = {
            'k': 2,
            'jump_tokens': 64,
            'cap_tokens': 1024,
            'max_blocks': int(expected_max_blocks),
            'max_model_retries': 3,
            'total_attempts_max': 4,
            'data_seed': 0,
            'retry_data_seeds': [0, 1001, 1002, 1003],
            'temperature': 0.6,
            'top_p': 1.0,
            'top_k': 0,
            'repetition_penalty': 1.1,
            'scorer': 'flexible-v3',
            'accept_rule': 'strict-mean-response-logprob-improvement',
            'eos_token_id': int(eos_token_id),
            'stop_token_ids': exp_stops,
        }
        for key, expected_val in expected_recipe_checks.items():
            actual_val = recipe.get(key)
            if actual_val != expected_val:
                raise LaunchRefused(
                    f'{target_path}: recipe.{key} is {actual_val!r}, expected qualified {expected_val!r}'
                )

    # Provenance verification (pinned proposer model, tokenizer, validation file, engine, executable, IEEE precision)
    prov = raw.get('provenance')
    if not isinstance(prov, dict):
        raise LaunchRefused(f'{target_path}: missing provenance metadata (proposer model/tokenizer/val hashes)')
    for req_key in ('model_file_sha256', 'tokenizer_file_sha256', 'val_items_sha256',
                    'engine', 'executable_sha256', 'precision_readback'):
        if not prov.get(req_key):
            raise LaunchRefused(f'{target_path}: provenance missing {req_key!r}')
    if expected_model_file_sha256 is not None and prov['model_file_sha256'] != dict(expected_model_file_sha256):
        raise LaunchRefused(
            f'{target_path}: provenance.model_file_sha256 differs from pinned base model in bundle'
        )
    if expected_tokenizer_file_sha256 is not None and prov['tokenizer_file_sha256'] != dict(expected_tokenizer_file_sha256):
        raise LaunchRefused(
            f'{target_path}: provenance.tokenizer_file_sha256 differs from pinned tokenizer files in bundle'
        )
    if expected_val_items_sha256 is not None and prov['val_items_sha256'] != expected_val_items_sha256:
        raise LaunchRefused(
            f'{target_path}: provenance.val_items_sha256 {prov["val_items_sha256"]!r} differs from bundled '
            f'validation:items {expected_val_items_sha256!r}'
        )
    if expected_engine is not None and prov['engine'] != expected_engine:
        raise LaunchRefused(
            f'{target_path}: provenance.engine {prov["engine"]!r} differs from authorized {expected_engine!r}'
        )
    if (expected_executable_sha256 is not None
            and prov['executable_sha256'] != expected_executable_sha256
            and prov['executable_sha256'] not in VERIFIED_PRIOR_STAGE1_RUN_A_COMMITS.values()):
        raise LaunchRefused(
            f'{target_path}: provenance.executable_sha256 {prov["executable_sha256"]!r} differs from registered '
            f'{expected_executable_sha256!r}'
        )
    if require_ieee_precision:
        from pilot_eval import neuron_audit_collector as nac
        readback = prov.get('precision_readback')
        if not isinstance(readback, dict):
            raise LaunchRefused(f'{target_path}: provenance.precision_readback must be a dict')
        required_keys = set(nac.PRECISION_BACKENDS) | {'matmul_precision'}
        missing_prec = sorted(required_keys - set(readback))
        if missing_prec:
            raise LaunchRefused(f'{target_path}: provenance.precision_readback missing backends: {missing_prec}')
        prec_problems = reval._precision_problems(readback)
        if prec_problems:
            raise LaunchRefused(f'{target_path}: non-IEEE precision in Stage 1 provenance: {"; ".join(prec_problems)}')

    seen_ids: list[str] = []
    eos = int(eos_token_id)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise LaunchRefused(f'{target_path}: item {idx} is not a dict')
        qid = item.get('id')
        if qid not in expected_set:
            raise LaunchRefused(f'{target_path}: item {idx} id {qid!r} is not in frozen val128')
        seen_ids.append(qid)
        k2p._check_stored_row_contract(item, label=f'{target_path}[{qid}]', eos_token_id=eos)
        ptok = int(item['prompt_tokens'])
        if list(item['input_ids'][:ptok]) != list(val_questions[qid]['prompt_ids']):
            raise LaunchRefused(f'{target_path}: item {qid!r} prompt_ids differ from frozen val128 prompt')
    if seen_ids != covered_ids:
        raise LaunchRefused(f'{target_path}: item IDs/order do not match covered_ids')

    meta = {k: v for k, v in raw.items() if k not in ('items', 'attempts')}
    return items, meta


def resolve_k2_adapter_dir(
    bundle: Mapping[str, Any],
    root: Path,
    record: Mapping[str, Any],
    *,
    verify_pinned: bool | None = None,
) -> tuple[Path, dict[str, str]]:
    """Resolve and hash-verify the dense Sampling SFT k = 2 step-64 LoRA adapter directory."""
    role_prefix = f'{K2_ADAPTER_ROLE}:'
    entries = [e for e in bundle.get('files', []) if str(e.get('role', '')).startswith(role_prefix)]
    if entries:
        adapter_dir = (root / entries[0]['path']).parent
    else:
        adapter_dir = _resolve_under(root, record['inputs']['k2_lora_adapter'])

    hashes: dict[str, str] = {}
    for fname in ('adapter_model.safetensors', 'adapter_config.json'):
        fpath = adapter_dir / fname
        if not fpath.is_file():
            raise LaunchRefused(f'k2 LoRA adapter missing {fname} in {adapter_dir}')
        hashes[fname] = file_hash(fpath)

    check_pins = (record['mode'] == 'production') if verify_pinned is None else bool(verify_pinned)
    if check_pins:
        if hashes['adapter_model.safetensors'] != k2p.K2_ADAPTER_WEIGHTS_SHA256:
            raise LaunchRefused(
                f'{adapter_dir / "adapter_model.safetensors"}: SHA-256 {hashes["adapter_model.safetensors"]} '
                f'differs from pinned k=2 checkpoint-64 {k2p.K2_ADAPTER_WEIGHTS_SHA256}'
            )
        if hashes['adapter_config.json'] != k2p.K2_ADAPTER_CONFIG_SHA256:
            raise LaunchRefused(
                f'{adapter_dir / "adapter_config.json"}: SHA-256 {hashes["adapter_config.json"]} '
                f'differs from pinned k=2 checkpoint-64 config {k2p.K2_ADAPTER_CONFIG_SHA256}'
            )
    return adapter_dir, hashes


def load_k2_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify all inputs for Sampling SFT (k = 2) sparse rewiring."""
    inputs = record['inputs']
    root = Path(inputs['artifact_root'])
    bundle = json.loads(Path(inputs['bundle']).read_text(encoding='utf-8'))
    cb.verify(root, bundle)

    eos = int(record['evaluation']['eos_token_id'])
    is_prod = record['mode'] == 'production'

    # Gold's frozen 128 validation questions define exact greedy prompts & ids_sha256
    gold_val_items = rb.check_validation(bundle, root, questions=record['evaluation']['validation']['questions'])
    val_questions = {
        i['id']: {'prompt_ids': list(i['input_ids'][:int(i['prompt_tokens'])]), 'gold': i['gold']}
        for i in gold_val_items
    }
    val_question_ids = [i['id'] for i in gold_val_items]
    val_items_bundle_path = rb.path_of(bundle, root, 'validation:items')
    expected_stop_ids = _expected_stop_ids_from_bundle(bundle, root, eos)

    # Sealed val128-k2 targets define the CE evaluation items
    val_k2_path = _resolve_under(root, inputs['val128_k2_targets'])
    val_k2_items, val_k2_meta = load_val128_k2_targets(
        val_k2_path,
        val_questions=val_questions,
        expected_ids=val_question_ids,
        eos_token_id=eos,
        expected_stop_ids=expected_stop_ids,
        expected_max_blocks=EXPECTED_MAX_BLOCKS,
        expected_model_file_sha256=bundle['base_model']['file_sha256'],
        expected_tokenizer_file_sha256=_expected_bundle_tokenizer_hashes(bundle),
        expected_val_items_sha256=file_hash(val_items_bundle_path),
        expected_engine=record['evaluation']['validation']['generator_engine'],
        expected_executable_sha256=record['executable_sha256'],
        require_ieee_precision=True,
        require_qualified_recipe=True,
    )

    # Gold order + verified sealed k = 2 training export
    order_path = _resolve_under(root, inputs['gold_order'])
    if not order_path.is_file():
        order_path = rb.path_of(bundle, root, 'gold:order')
    gold_order = json.loads(order_path.read_text(encoding='utf-8'))

    k2_data_path = _resolve_under(root, inputs['k2_training_data'])
    expected_export_seal = k2p.K2_EXPORT_COMPLETE_SHA256 if is_prod else None
    training_rows, export_meta = k2p.load_k2_training_rows(
        k2_data_path,
        expected_export_seal_sha256=expected_export_seal,
        gold_order=gold_order,
        eos_token_id=eos,
    )

    adapter_dir, adapter_hashes = resolve_k2_adapter_dir(bundle, root, record, verify_pinned=is_prod)
    shared_inputs_sha = {
        'k2_export_seal_sha256': export_meta['k2_export_seal_sha256'],
        'k2_training_rows_sha256': export_meta['k2_training_rows_sha256'],
        'gold_order_sha256': file_hash(order_path),
        'val128_k2_targets_sha256': file_hash(val_k2_path),
        'k2_lora_adapter_weights_sha256': adapter_hashes['adapter_model.safetensors'],
        'k2_lora_adapter_config_sha256': adapter_hashes['adapter_config.json'],
    }
    extra_sha = {
        'val128_k2_targets_sha256': shared_inputs_sha['val128_k2_targets_sha256'],
        'k2_lora_adapter_weights_sha256': shared_inputs_sha['k2_lora_adapter_weights_sha256'],
        'k2_lora_adapter_config_sha256': shared_inputs_sha['k2_lora_adapter_config_sha256'],
    }
    arm_plans = {
        arm: k2p.plan_from_files(
            dict(record),
            arm,
            k2_data_path,
            order_path,
            expected_export_seal_sha256=expected_export_seal,
            extra_inputs_sha256=extra_sha,
        )
        for arm in record['arms']
    }
    cohort150 = rb.check_cohort150(bundle, root) if is_prod else None
    return dict(
        training_rows=training_rows,
        export_meta=export_meta,
        arm_plans=arm_plans,
        val_items=val_k2_items,
        val_k2_meta=val_k2_meta,
        val_questions=val_questions,
        val_question_ids=val_question_ids,
        bundle=bundle,
        adapter_dir=adapter_dir,
        adapter_hashes=adapter_hashes,
        shared_inputs_sha=shared_inputs_sha,
        k2_data_path=k2_data_path,
        order_path=order_path,
        cohort150_items=cohort150,
    )


# --- Isolated IEEE FP32 child process for k = 2 references & headroom calibration (P2 #5) ------------

def _k2_reference_child_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Child-process worker: evaluate untuned (step 0) and dense k = 2 LoRA (step 64) on val128-k2 CE,
    val128 greedy accuracy + mean response length, and (in production) cohort150 greedy accuracy +
    mean response length, then compute headroom-scaled lambda_k2 and seal ``k2_headroom.json``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from pilot_eval.rewire_chain import RewireBackend

    record = payload['record']
    volume_root = payload['volume_root']
    device = payload.get('device', 'cuda')
    out_dir = Path(payload['out_dir'])
    adapter_dir = Path(payload['adapter_dir'])
    val_items = payload['val_items']
    val_questions = payload['val_questions']
    val_question_ids = payload['val_question_ids']
    cohort150_questions = payload.get('cohort150_questions')
    cohort150_question_ids = payload.get('cohort150_question_ids')
    inputs_sha256 = dict(payload.get('inputs_sha256') or {})
    cap = int(payload['cap'])
    batch_size = int(payload['batch_size'])
    final_step = int(record['training']['steps'])
    mode = str(record['mode'])

    backend = RewireBackend(record, volume_root, device=device)
    probs = backend.precision_problems()
    if probs:
        raise RuntimeError('; '.join(probs))
    rt_info = backend.runtime()
    rt_sha = _hash(rt_info)

    adapter_hashes = {
        fname: file_hash(adapter_dir / fname)
        for fname in ('adapter_model.safetensors', 'adapter_config.json')
    }

    results: dict[str, Any] = {}
    prior_uses: list[dict[str, Any]] = []
    sealed_subpaths: list[str] = []
    for ref_arm, ref_step in (('untuned', 0), ('sampling-sft-k2', 64)):
        base = AutoModelForCausalLM.from_pretrained(
            backend.model_dir, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager'
        )
        if ref_arm == 'untuned':
            model = base.to(backend.device)
        else:
            model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).to(backend.device)
        try:
            ce = reval.evaluate_validation_ce(model, val_items)
            val_rel = f'{ref_arm}-step-{ref_step}/greedy-val'
            greedy_val = reval.evaluate_greedy(
                model,
                val_questions,
                val_question_ids,
                out_dir / val_rel,
                arm=ref_arm,
                step=ref_step,
                cohort='gsm8k-train-val128',
                decode=backend.decode,
                pad_token_id=backend.pad_token_id,
                eos_token_id=backend.eos_token_id,
                cap=cap,
                batch_size=batch_size,
                final_step=final_step,
                mode=mode,
            )
            sealed_subpaths.append(val_rel)
            prior_uses.append(greedy_val['prior_use'])
            ref_entry: dict[str, Any] = {
                'step': ref_step,
                'val_ce_k2': ce['val_ce'],
                'val_ce': ce['val_ce'],
                'val_k2_items': len(val_items),
                'val_accuracy': greedy_val['scores']['accuracy'],
                'val_tokens_mean': greedy_val['scores']['tokens_mean'],
                'val_tokens_median': greedy_val['scores']['tokens_median'],
                'val_capped': greedy_val['scores']['capped'],
                'val_invalid': greedy_val['scores']['invalid'],
                'eval_dir': greedy_val['directory'],
            }
            if cohort150_questions is not None and cohort150_question_ids is not None:
                c150_rel = f'{ref_arm}-step-{ref_step}/greedy-cohort150'
                greedy_c150 = reval.evaluate_greedy(
                    model,
                    cohort150_questions,
                    cohort150_question_ids,
                    out_dir / c150_rel,
                    arm=ref_arm,
                    step=ref_step,
                    cohort=reval.COHORT150_NAME,
                    decode=backend.decode,
                    pad_token_id=backend.pad_token_id,
                    eos_token_id=backend.eos_token_id,
                    cap=cap,
                    batch_size=batch_size,
                    final_step=ref_step,
                    mode=mode,
                )
                sealed_subpaths.append(c150_rel)
                prior_uses.append(greedy_c150['prior_use'])
                ref_entry.update(
                    cohort150_accuracy=greedy_c150['scores']['accuracy'],
                    cohort150_tokens_mean=greedy_c150['scores']['tokens_mean'],
                    cohort150_tokens_median=greedy_c150['scores']['tokens_median'],
                    cohort150_capped=greedy_c150['scores']['capped'],
                    cohort150_invalid=greedy_c150['scores']['invalid'],
                    cohort150_eval_dir=greedy_c150['directory'],
                )
            results[ref_arm] = ref_entry
        finally:
            model = None
            base = None
            backend.release()

    ce_untuned = float(results['untuned']['val_ce_k2'])
    ce_lora = float(results['sampling-sft-k2']['val_ce_k2'])
    if ce_untuned > ce_lora:
        headroom: dict[str, Any] = k2p.compute_scaled_lambda(ce_untuned, ce_lora)
    else:
        delta = ce_untuned - ce_lora
        headroom = {
            'ce_untuned_k2': ce_untuned,
            'ce_lora_k2': ce_lora,
            'delta_ce_k2': delta,
            'gold_headroom': k2p.GOLD_VAL_CE_HEADROOM,
            'headroom_ratio': delta / k2p.GOLD_VAL_CE_HEADROOM,
            'lambda_matched': k2p.GOLD_MATCHED_LAMBDA,
            'lambda_scaled': None,
            'warning': 'non-positive k=2 CE headroom (untuned CE <= dense k=2 LoRA CE)',
        }
    headroom.update(
        val_k2_items=len(val_items),
        val_greedy_questions=len(val_question_ids),
        val_accuracy_untuned=results['untuned']['val_accuracy'],
        val_accuracy_lora_k2=results['sampling-sft-k2']['val_accuracy'],
        val_tokens_mean_untuned=results['untuned']['val_tokens_mean'],
        val_tokens_mean_lora_k2=results['sampling-sft-k2']['val_tokens_mean'],
        cohort150_accuracy_untuned=results['untuned'].get('cohort150_accuracy'),
        cohort150_accuracy_lora_k2=results['sampling-sft-k2'].get('cohort150_accuracy'),
        cohort150_tokens_mean_untuned=results['untuned'].get('cohort150_tokens_mean'),
        cohort150_tokens_mean_lora_k2=results['sampling-sft-k2'].get('cohort150_tokens_mean'),
        adapter_dir=str(adapter_dir),
        adapter_sha256=adapter_hashes,
        inputs_sha256=inputs_sha256,
    )

    au.seal_run(
        out_dir / 'summary',
        {
            'references.json': results,
            'k2_headroom.json': headroom,
            'prior_uses.json': prior_uses,
            'runtime_sha256': rt_sha,
        },
    )
    return {
        'references': results,
        'k2_headroom': headroom,
        'prior_uses': prior_uses,
        'sealed_subpaths': sealed_subpaths,
        'runtime_sha256': rt_sha,
        'precision_readback': backend.precision_readback(),
    }


def evaluate_k2_references_in_child(
    *,
    record: Mapping[str, Any],
    volume_root: str | Path,
    out_dir: str | Path,
    adapter_dir: str | Path,
    val_items: Sequence[Mapping[str, Any]],
    val_questions: Mapping[str, Mapping[str, Any]],
    val_question_ids: Sequence[str],
    expected_runtime_sha256: str,
    cohort150_questions: Mapping[str, Mapping[str, Any]] | None = None,
    cohort150_question_ids: Sequence[str] | None = None,
    inputs_sha256: Mapping[str, Any] | None = None,
    device: str = 'cuda',
    cap: int = 1024,
    batch_size: int = 16,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run untuned (step 0) and dense k = 2 LoRA (step 64) references in an IEEE-verified child process."""
    return reval.run_in_child_process(
        'pilot_eval.rewire_k2_launch',
        '_k2_reference_child_worker',
        {
            'record': dict(record),
            'volume_root': str(volume_root),
            'out_dir': str(out_dir),
            'adapter_dir': str(adapter_dir),
            'val_items': list(val_items),
            'val_questions': dict(val_questions),
            'val_question_ids': list(val_question_ids),
            'cohort150_questions': dict(cohort150_questions) if cohort150_questions is not None else None,
            'cohort150_question_ids': list(cohort150_question_ids) if cohort150_question_ids is not None else None,
            'inputs_sha256': dict(inputs_sha256 or {}),
            'device': device,
            'cap': int(cap),
            'batch_size': int(batch_size),
        },
        env=env,
        expected_runtime_sha256=expected_runtime_sha256,
        require_ieee=True,
    )


# --- K2-D1 verified gate, plan-identity check & sealed headroom calibration (P1 #3) ------------------

def _assert_registered_matched_s42_plan(
    actual_plan: Mapping[str, Any],
    expected_matched_plan: Mapping[str, Any],
    *,
    require_exact_plan_sha: bool = False,
) -> None:
    """Verify a completed Run A seed-42 arm matches the exact registered matched seed-42 plan."""
    if require_exact_plan_sha:
        if actual_plan['plan_sha256'] != expected_matched_plan['plan_sha256']:
            raise LaunchRefused(
                f'completed Run A seed-42 plan_sha256 {actual_plan["plan_sha256"]} differs from registered '
                f'expected plan {expected_matched_plan["plan_sha256"]}'
            )
        return

    # For Run B routing, compare every scientific & provenance field against the registered matched plan
    for field in ('schema', 'rank', 'read_lr', 'seed', 'executable_sha256', 'optimizer',
                  'read_init_seeds', 'checkpoints', 'steps'):
        if field == 'executable_sha256' and actual_plan.get(field) in VERIFIED_PRIOR_STAGE1_RUN_A_COMMITS.values():
            continue
        if actual_plan.get(field) != expected_matched_plan.get(field):
            raise LaunchRefused(
                f'completed Run A seed-42 plan field {field!r} differs from registered matched seed-42 plan'
            )
    act_s = actual_plan.get('settings') or {}
    exp_s = expected_matched_plan.get('settings') or {}
    for key in ('model', 'data', 'training', 'objective', 'hard_concrete', 'runtime', 'seeds'):
        if act_s.get(key) != exp_s.get(key):
            raise LaunchRefused(
                f'completed Run A seed-42 plan settings.{key} differs from registered matched seed-42 plan '
                f'(got {act_s.get(key)!r}, expected {exp_s.get(key)!r})'
            )
    _check_shared_inputs_match(
        actual_plan.get('inputs') or {},
        expected_matched_plan.get('inputs') or {},
        left_label='completed Run A seed-42 plan',
        right_label='registered matched seed-42 plan',
    )


def _derive_expected_matched_s42_plan(record: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the canonical registered matched seed-42 plan from ``record`` and verified ``data``."""
    s42 = _arm_by_seed(record, 42)
    if s42 is None:
        raise LaunchRefused('launch record missing seed-42 arm')
    if record['objective']['lambda_regime'] == 'matched':
        return data['arm_plans'][s42]
    # Build canonical matched counterpart for Run B verification (same inputs/code/hyperparams, matched lambda=6e-4)
    matched_rec = json.loads(json.dumps(dict(record)))
    matched_rec['objective']['lambda_regime'] = 'matched'
    matched_rec['objective']['l0_lambda'] = k2p.GOLD_MATCHED_LAMBDA
    extra_sha = {
        'val128_k2_targets_sha256': data['shared_inputs_sha']['val128_k2_targets_sha256'],
        'k2_lora_adapter_weights_sha256': data['shared_inputs_sha']['k2_lora_adapter_weights_sha256'],
        'k2_lora_adapter_config_sha256': data['shared_inputs_sha']['k2_lora_adapter_config_sha256'],
    }
    return k2p.plan_from_files(
        matched_rec,
        s42,
        data['k2_data_path'],
        data['order_path'],
        expected_export_seal_sha256=(k2p.K2_EXPORT_COMPLETE_SHA256 if record['mode'] == 'production' else None),
        extra_inputs_sha256=extra_sha,
    )


def verify_headroom_summary(
    headroom_dir: str | Path,
    *,
    expected_inputs_sha256: Mapping[str, Any] | None = None,
    expected_complete_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify sealed ``references/summary`` and recompute ``lambda_scaled`` from ``k2_headroom.json``."""
    summary_dir = Path(headroom_dir)
    if summary_dir.is_file() and summary_dir.name == 'k2_headroom.json':
        summary_dir = summary_dir.parent
    if not summary_dir.is_dir():
        raise LaunchRefused(f'headroom summary directory does not exist: {summary_dir}')
    try:
        au.verify_run(summary_dir)
    except Exception as exc:
        raise LaunchRefused(f'headroom summary failed seal verification ({summary_dir}): {exc}') from exc
    headroom_path = summary_dir / 'k2_headroom.json'
    if not headroom_path.is_file():
        raise LaunchRefused(f'missing k2_headroom.json in {summary_dir}')

    complete_sha = file_hash(summary_dir / 'complete.json')
    file_sha = file_hash(headroom_path)
    if expected_complete_sha256 is not None and complete_sha != expected_complete_sha256:
        raise LaunchRefused(
            f'{summary_dir}: headroom complete.json SHA-256 {complete_sha} differs from receipt-bound '
            f'{expected_complete_sha256} (different Stage 1 calibration substituted)'
        )
    if expected_file_sha256 is not None and file_sha != expected_file_sha256:
        raise LaunchRefused(
            f'{headroom_path}: SHA-256 {file_sha} differs from receipt-bound {expected_file_sha256}'
        )

    headroom = json.loads(headroom_path.read_text(encoding='utf-8'))
    ce_u = headroom.get('ce_untuned_k2')
    ce_l = headroom.get('ce_lora_k2')
    recorded_lam = headroom.get('lambda_scaled')
    if ce_u is None or ce_l is None or recorded_lam is None:
        raise LaunchRefused(f'{headroom_path}: missing positive calibrated lambda_scaled (ce_u={ce_u}, ce_l={ce_l})')
    try:
        recomputed = k2p.compute_scaled_lambda(float(ce_u), float(ce_l))
    except ValueError as exc:
        raise LaunchRefused(f'{headroom_path}: invalid headroom calibration: {exc}') from exc
    if not math.isfinite(float(recorded_lam)) or not math.isclose(
        float(recorded_lam), recomputed['lambda_scaled'], rel_tol=1e-9, abs_tol=1e-12
    ):
        raise LaunchRefused(
            f'{headroom_path}: recorded lambda_scaled {recorded_lam} differs from recomputed '
            f'{recomputed["lambda_scaled"]}'
        )
    if expected_inputs_sha256 is not None:
        _check_shared_inputs_match(
            headroom.get('inputs_sha256') or {},
            expected_inputs_sha256,
            left_label=str(headroom_path),
            right_label='current k2 experiment inputs',
        )
    return {
        'summary_dir': str(summary_dir),
        'headroom_complete_sha256': complete_sha,
        'headroom_file_sha256': file_sha,
        'headroom': headroom,
        'lambda_scaled': float(recomputed['lambda_scaled']),
    }


def check_run_b_trigger(
    arm_dir: str | Path,
    *,
    expected_plan: Mapping[str, Any] | None = None,
    require_exact_plan_sha: bool = True,
    expected_final_step: int | None = None,
    expected_inputs_sha256: Mapping[str, Any] | None = None,
    expected_code_commit: str | None = None,
    expected_executable_sha256: str | None = None,
    threshold: int = k2p.RUN_B_SURVIVOR_THRESHOLD,
    headroom_dir: str | Path | None = None,
    write_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a completed, sealed Run A seed-42 arm at its final step, count survivors, and evaluate K2-D1."""
    from safetensors.torch import load_file
    from pilot_eval import hard_concrete as hc
    from pilot_eval import rewire_chain as rc
    from pilot_eval import rewire_checkpoint as rckpt

    root = Path(arm_dir)
    if not root.is_dir():
        raise LaunchRefused(f'arm directory does not exist: {root}')

    # 1. Verify arm_evidence seal and materialize training-complete.json + plan.json (supports volume restores!)
    ev_dir = root / rc.ARM_EVIDENCE_DIR
    if not (ev_dir / 'complete.json').is_file():
        raise LaunchRefused(f'{root}: missing sealed {rc.ARM_EVIDENCE_DIR}/complete.json (arm is not complete)')
    try:
        rc.restore_arm_evidence(root)
    except Exception as exc:
        raise LaunchRefused(f'{root}: arm_evidence seal verification failed: {exc}') from exc

    arm_seal_path = root / rckpt.ARM_SEAL
    plan_path = root / 'plan.json'
    if not arm_seal_path.is_file() or not plan_path.is_file():
        raise LaunchRefused(f'{root}: missing {rckpt.ARM_SEAL} or plan.json after restoring arm_evidence')

    arm_seal = json.loads(arm_seal_path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') != k2p.PLAN_SCHEMA or plan.get('plan_sha256') != k2p.sp.plan_sha256(plan):
        raise LaunchRefused(f'{plan_path}: plan_sha256 verification failed')
    prior_exec = VERIFIED_PRIOR_STAGE1_RUN_A_COMMITS.get(str(arm_seal.get('code_commit') or ''))
    is_verified_prior = (prior_exec is not None and plan.get('executable_sha256') == prior_exec)
    if expected_code_commit is not None and arm_seal.get('code_commit') != expected_code_commit and not is_verified_prior:
        raise LaunchRefused(
            f'{root}: arm code_commit {arm_seal.get("code_commit")!r} != expected {expected_code_commit!r}'
        )
    if expected_executable_sha256 is not None and plan.get('executable_sha256') != expected_executable_sha256 and not is_verified_prior:
        raise LaunchRefused(
            f'{root}: arm executable_sha256 {plan.get("executable_sha256")!r} != expected {expected_executable_sha256!r}'
        )
    if expected_plan is not None:
        _assert_registered_matched_s42_plan(plan, expected_plan, require_exact_plan_sha=require_exact_plan_sha)
    if int(plan.get('seed', -1)) != 42:
        raise LaunchRefused(f'{root}: K2-D1 gate requires completed seed-42 arm, got seed={plan.get("seed")}')
    obj = plan.get('settings', {}).get('objective', {})
    if obj.get('lambda_regime') != 'matched' or not k2p.sp._same(obj.get('l0_lambda'), k2p.GOLD_MATCHED_LAMBDA):
        raise LaunchRefused(
            f'{root}: K2-D1 gate requires matched-lambda Run A arm at {k2p.GOLD_MATCHED_LAMBDA}, '
            f'got regime={obj.get("lambda_regime")!r}, l0_lambda={obj.get("l0_lambda")!r}'
        )

    # 2. Verify every checkpoint in the arm against training-complete.json and plan
    try:
        rckpt.verify_arm(
            root,
            plan,
            runtime_sha256=arm_seal['runtime_sha256'],
            code_commit=arm_seal['code_commit'],
        )
    except Exception as exc:
        raise LaunchRefused(f'{root}: verify_arm failed (incomplete or tampered checkpoints): {exc}') from exc

    final_step = int(plan['steps'][-1]['step'])
    if expected_final_step is not None and final_step != int(expected_final_step):
        raise LaunchRefused(f'{root}: arm final step is {final_step}, expected {expected_final_step}')
    if final_step not in plan['checkpoints']:
        raise LaunchRefused(f'{root}: final step {final_step} is not in plan checkpoints')

    plan_inputs_sha = _shared_inputs_subset(plan.get('inputs') or {})
    if expected_inputs_sha256 is not None:
        _check_shared_inputs_match(
            plan_inputs_sha,
            expected_inputs_sha256,
            left_label=str(plan_path),
            right_label='expected experiment inputs',
        )

    # 3. Verify headroom summary if supplied and bind its exact cryptographic seal digests
    headroom_info = None
    lambda_scaled = None
    if headroom_dir is not None:
        headroom_info = verify_headroom_summary(headroom_dir, expected_inputs_sha256=plan_inputs_sha)
        lambda_scaled = headroom_info['lambda_scaled']

    # 4. Count surviving binary neurons strictly from the verified final checkpoint
    ckpt_dir = rckpt.checkpoint_dir(root, final_step)
    state_path = ckpt_dir / rckpt.STATE_FILE
    tensors = load_file(str(state_path))
    survivors: list[dict[str, Any]] = []
    per_layer_counts: dict[str, int] = {}
    for key in sorted(tensors):
        if not key.endswith('rewire_log_alpha'):
            continue
        m = re.search(r'layers\.(\d+)\.', key)
        layer_idx = int(m.group(1)) if m else -1
        log_alpha = tensors[key].float()
        z_det = hc.deterministic_gate(log_alpha)
        active_indices = (z_det > 0).nonzero(as_tuple=False).view(-1).tolist()
        per_layer_counts[str(layer_idx)] = len(active_indices)
        for n_idx in active_indices:
            survivors.append({
                'layer': layer_idx,
                'neuron': int(n_idx),
                'log_alpha': float(log_alpha[n_idx].item()),
                'z_det': float(z_det[n_idx].item()),
            })
    survivors.sort(key=lambda r: (r['layer'], r['neuron']))
    count = len(survivors)
    trigger = k2p.should_trigger_run_b(count, threshold=threshold)

    receipt: dict[str, Any] = {
        'schema': GATE_RECEIPT_SCHEMA,
        'arm': plan['arm'],
        'run_id': plan['run_id'],
        'seed': int(plan['seed']),
        'code_commit': arm_seal['code_commit'],
        'executable_sha256': plan.get('executable_sha256'),
        'arm_dir': str(root.resolve()),
        'arm_plan_sha256': plan['plan_sha256'],
        'training_complete_sha256': file_hash(arm_seal_path),
        'arm_evidence_complete_sha256': file_hash(ev_dir / 'complete.json'),
        'final_step': final_step,
        'state_file': str(state_path.resolve()),
        'final_state_sha256': file_hash(state_path),
        'inputs_sha256': plan_inputs_sha,
        'survivor_count': count,
        'threshold': int(threshold),
        'trigger_run_b': trigger,
        'recommended_next': (
            'seal-and-run-B-scaled-lambda' if trigger else 'run-seed-43-on-Run-A-matched-lambda'
        ),
        'headroom_summary': headroom_info['summary_dir'] if headroom_info else None,
        'headroom_complete_sha256': headroom_info['headroom_complete_sha256'] if headroom_info else None,
        'headroom_file_sha256': headroom_info['headroom_file_sha256'] if headroom_info else None,
        'lambda_scaled': lambda_scaled,
        'per_layer_counts': per_layer_counts,
        'survivors': survivors,
    }
    receipt['receipt_sha256'] = _hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    if write_receipt is not None:
        out_p = Path(write_receipt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def verify_gate_receipt(
    receipt_path: str | Path,
    *,
    expected_plan: Mapping[str, Any],
    require_exact_plan_sha: bool,
    expected_final_step: int,
    expected_inputs_sha256: Mapping[str, Any],
    expected_code_commit: str | None = None,
    expected_executable_sha256: str | None = None,
    threshold: int = k2p.RUN_B_SURVIVOR_THRESHOLD,
    arm_dir_override: str | Path | None = None,
    headroom_dir: str | Path | None = None,
    require_bound_headroom: bool = False,
    require_trigger_run_b: bool | None = None,
) -> dict[str, Any]:
    """Verify a gate receipt JSON, re-verify its completed Run A seed-42 arm against the registered plan,
    and enforce exact Stage 1 headroom seal binding (supporting authentic volume restores)."""
    rpath = Path(receipt_path)
    if not rpath.is_file():
        raise LaunchRefused(f'gate receipt does not exist: {rpath}')
    saved = json.loads(rpath.read_text(encoding='utf-8'))
    if saved.get('schema') != GATE_RECEIPT_SCHEMA:
        raise LaunchRefused(f'{rpath}: invalid gate receipt schema {saved.get("schema")!r}')
    expected_hash = _hash({k: v for k, v in saved.items() if k != 'receipt_sha256'})
    if saved.get('receipt_sha256') != expected_hash:
        raise LaunchRefused(f'{rpath}: gate receipt_sha256 mismatch (tampered receipt)')

    target_arm_dir = Path(arm_dir_override) if arm_dir_override is not None else Path(saved['arm_dir'])
    target_headroom_dir = headroom_dir if headroom_dir is not None else saved.get('headroom_summary')

    if require_bound_headroom:
        if not saved.get('headroom_complete_sha256') or not saved.get('headroom_file_sha256'):
            raise LaunchRefused(
                f'{rpath}: gate receipt is missing bound Stage 1 headroom seal digests; regenerate with --headroom-dir'
            )
        if target_headroom_dir is None:
            raise LaunchRefused(f'{rpath}: missing headroom summary directory')

    recomputed = check_run_b_trigger(
        target_arm_dir,
        expected_plan=expected_plan,
        require_exact_plan_sha=require_exact_plan_sha,
        expected_final_step=expected_final_step,
        expected_inputs_sha256=expected_inputs_sha256,
        expected_code_commit=expected_code_commit,
        expected_executable_sha256=expected_executable_sha256,
        threshold=threshold,
        headroom_dir=target_headroom_dir,
    )
    for key in ('arm_plan_sha256', 'training_complete_sha256', 'final_state_sha256',
                'survivor_count', 'trigger_run_b', 'inputs_sha256'):
        if recomputed[key] != saved.get(key):
            raise LaunchRefused(f'{rpath}: live arm {key} differs from gate receipt')
    if target_headroom_dir is not None and saved.get('headroom_complete_sha256'):
        if (recomputed['headroom_complete_sha256'] != saved['headroom_complete_sha256']
                or recomputed['headroom_file_sha256'] != saved['headroom_file_sha256']):
            raise LaunchRefused(
                f'{rpath}: supplied headroom summary seal {recomputed["headroom_complete_sha256"]} differs from '
                f'the specific Stage 1 headroom seal bound by the gate receipt ({saved["headroom_complete_sha256"]})'
            )
    if require_trigger_run_b is not None and bool(recomputed['trigger_run_b']) is not bool(require_trigger_run_b):
        if require_trigger_run_b:
            raise LaunchRefused(
                f'Run B refused: verified Run A seed 42 finished with {recomputed["survivor_count"]} survivors '
                f'(>= threshold {threshold}); run matched seed 43 on Run A instead'
            )
        raise LaunchRefused(
            f'Matched Run A seed 43 refused: verified Run A seed 42 finished with {recomputed["survivor_count"]} '
            f'survivors (< threshold {threshold}); seal and run scaled Run B instead'
        )
    return recomputed


def _authorize_run_b(
    record: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
    arm_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Verify Run B (scaled) gate receipt against the registered matched seed-42 plan, its exact bound
    Stage 1 headroom seal, matching inputs/code provenance, and exact lambda_scaled."""
    routing = record['routing']
    gate_path = routing.get('gate_receipt')
    headroom_path = routing.get('headroom_summary')
    if not gate_path or gate_path == 'none':
        raise LaunchRefused('Run B requires routing.gate_receipt from verified Run A seed 42')
    if not headroom_path or headroom_path == 'none':
        raise LaunchRefused('Run B requires routing.headroom_summary from sealed Stage 1 references')

    expected_matched_s42 = _derive_expected_matched_s42_plan(record, data)
    gate = verify_gate_receipt(
        gate_path,
        expected_plan=expected_matched_s42,
        require_exact_plan_sha=False,
        expected_final_step=int(record['training']['steps']),
        expected_inputs_sha256=data['shared_inputs_sha'],
        expected_code_commit=record['code_commit'],
        expected_executable_sha256=record['executable_sha256'],
        threshold=int(record['objective']['run_b_trigger_below']),
        arm_dir_override=arm_dir_override,
        headroom_dir=headroom_path,
        require_bound_headroom=True,
        require_trigger_run_b=True,
    )
    headroom_info = verify_headroom_summary(
        headroom_path,
        expected_inputs_sha256=data['shared_inputs_sha'],
        expected_complete_sha256=gate['headroom_complete_sha256'],
        expected_file_sha256=gate['headroom_file_sha256'],
    )
    lam = float(record['objective']['l0_lambda'])
    expected_lam = float(headroom_info['lambda_scaled'])
    if not math.isclose(lam, expected_lam, rel_tol=1e-9, abs_tol=1e-12):
        raise LaunchRefused(
            f'Run B objective.l0_lambda ({lam}) differs from sealed headroom lambda_scaled ({expected_lam})'
        )
    return {'gate': gate, 'headroom': headroom_info}


def _arm_by_seed(record: Mapping[str, Any], seed: int) -> str | None:
    for arm, cfg in record['arms'].items():
        if int(cfg['seed']) == int(seed):
            return arm
    return None


def resolve_arm_order(
    record: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
    only_arm: str | None = None,
    gate_receipt: str | Path | None = None,
    matched_s42_dir: str | Path | None = None,
) -> list[str]:
    """Enforce K2-D1 conditional two-lambda scheduling on every production entry point."""
    arms = list(record['arms'].keys())
    if only_arm is not None and only_arm not in record['arms']:
        raise LaunchRefused(f'unknown --only-arm {only_arm!r}; valid arms: {arms}')

    if record['mode'] == 'rehearsal':
        return [only_arm] if only_arm is not None else arms

    regime = record['objective']['lambda_regime']
    final_step = int(record['training']['steps'])
    threshold = int(record['objective']['run_b_trigger_below'])

    if regime == 'matched':
        s42 = _arm_by_seed(record, 42)
        s43 = _arm_by_seed(record, 43)
        if s42 is None or s43 is None:
            raise LaunchRefused('production Run A requires both seed-42 and seed-43 arms')
        if only_arm is None or only_arm == s42:
            return [s42]
        if only_arm != s43:
            raise LaunchRefused(f'unexpected production Run A arm {only_arm!r}; expected {s42!r} or {s43!r}')

        expected_s42_plan = data['arm_plans'][s42]
        if gate_receipt is not None:
            verify_gate_receipt(
                gate_receipt,
                expected_plan=expected_s42_plan,
                require_exact_plan_sha=True,
                expected_final_step=final_step,
                expected_inputs_sha256=data['shared_inputs_sha'],
                expected_code_commit=record['code_commit'],
                expected_executable_sha256=record['executable_sha256'],
                threshold=threshold,
                arm_dir_override=matched_s42_dir,
                require_trigger_run_b=False,
            )
        elif matched_s42_dir is not None:
            gate = check_run_b_trigger(
                matched_s42_dir,
                expected_plan=expected_s42_plan,
                require_exact_plan_sha=True,
                expected_final_step=final_step,
                expected_inputs_sha256=data['shared_inputs_sha'],
                expected_code_commit=record['code_commit'],
                expected_executable_sha256=record['executable_sha256'],
                threshold=threshold,
            )
            if gate['trigger_run_b']:
                raise LaunchRefused(
                    f'Matched Run A {only_arm} refused: verified seed 42 finished with '
                    f'{gate["survivor_count"]} survivors (< {threshold}); run scaled Run B instead'
                )
        else:
            raise LaunchRefused(
                f'Running matched Run A {only_arm} requires --gate-receipt or --matched-s42-dir '
                f'proving verified completed Run A seed 42 had >= {threshold} survivors'
            )
        return [s43]

    if regime == 'scaled':
        _authorize_run_b(record, data=data, arm_dir_override=matched_s42_dir)
        return [only_arm] if only_arm is not None else arms

    raise LaunchRefused(f'unsupported lambda_regime {regime!r}')


# --- Chain builder & CLI -----------------------------------------------------------------------------

def build(
    record: Mapping[str, Any],
    local: str | Path,
    volume: str | Path,
    *,
    only_arm: str | None = None,
    gate_receipt: str | Path | None = None,
    matched_s42_dir: str | Path | None = None,
    backend: Any = None,
    probes: Mapping[str, Any] | None = None,
    device: str = 'cuda',
    exporter: Any = None,
    runner: Any = None,
    volume_identity: str | None = None,
    current_pod: str | None = None,
    references_fn: Any = None,
    gradient_checkpointing: bool = True,
    stop_pod: bool = False,
):
    """Build the ``c_attribution_chain.Chain`` for a sealed Sampling SFT (k = 2) launch record."""
    from pilot_eval import rewire_chain as rc

    if record['mode'] not in ('rehearsal', 'production'):
        raise LaunchRefused(f"unknown launch mode {record['mode']!r}")
    data = load_k2_inputs(record)
    arm_order = resolve_arm_order(
        record,
        data=data,
        only_arm=only_arm,
        gate_receipt=gate_receipt,
        matched_s42_dir=matched_s42_dir,
    )
    scoped_only = arm_order[0] if len(arm_order) == 1 and len(record['arms']) > 1 else only_arm
    effective_record = chain_record(record, only_arm=scoped_only)

    backend = backend or rc.RewireBackend(effective_record, volume, device=device, probes=probes)
    c150, c150_ids = (None, None)
    if data['cohort150_items'] is not None:
        c150, c150_ids = rl.cohort150_questions(data['cohort150_items'], backend.encode)

    if references_fn is None:
        def _default_k2_references(ctx: Any) -> dict[str, Any]:
            rt_sha = _hash(backend.runtime())
            res = evaluate_k2_references_in_child(
                record=effective_record,
                volume_root=backend.volume_root,
                out_dir=ctx.chain.local_root / rc.REFERENCES_DIR,
                adapter_dir=data['adapter_dir'],
                val_items=data['val_items'],
                val_questions=data['val_questions'],
                val_question_ids=data['val_question_ids'],
                expected_runtime_sha256=rt_sha,
                cohort150_questions=c150,
                cohort150_question_ids=c150_ids,
                inputs_sha256=data['shared_inputs_sha'],
                device=backend.device,
                cap=int(record['evaluation']['cap']),
                batch_size=int(record['evaluation']['batch_size']),
            )
            outputs = [f'{rc.REFERENCES_DIR}/summary'] + [
                f'{rc.REFERENCES_DIR}/{sub}' for sub in res.get('sealed_subpaths', [])
            ]
            return {
                'outputs': outputs,
                'references': res['references'],
                'k2_headroom': res['k2_headroom'],
            }
        references_fn = _default_k2_references

    return rc.build_rewire_chain(
        record=effective_record,
        backend=backend,
        local_root=local,
        volume=volume,
        training_rows=data['training_rows'],
        arm_plans=data['arm_plans'],
        val_items=data['val_items'],
        val_questions=data['val_questions'],
        val_question_ids=data['val_question_ids'],
        cohort150_questions=c150,
        cohort150_question_ids=c150_ids,
        arm_order=arm_order,
        expected_base_sha256=rl.base_hash_once(backend),
        execution_kind='real-cuda',
        gradient_checkpointing=gradient_checkpointing,
        references_fn=references_fn,
        exporter=exporter,
        runner=runner,
        volume_identity=volume_identity,
        current_pod=current_pod,
        stop_pod=stop_pod,
    )


def seal_k2_draft(
    draft_path: str | Path,
    out_path: str | Path,
    *,
    artifact_root: str,
    bundle: str,
    model_dir: str,
    work_dir: str,
    val128_k2_targets: str,
    pod_id: str | None = None,
    l0_lambda: float | None = None,
    gate_receipt: str | None = None,
    headroom_summary: str | None = None,
    matched_s42_dir: str | None = None,
    code_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Seal a K2 launch-record YAML draft, enforcing clean executable code and Run B gate/calibration."""
    import yaml

    ident = dict(code_identity) if code_identity is not None else k2p.verified_code_identity(require_clean=True)
    raw = yaml.safe_load(Path(draft_path).read_text(encoding='utf-8'))
    fill: dict[str, Any] = {
        'inputs.artifact_root': artifact_root,
        'inputs.bundle': bundle,
        'inputs.model_dir': model_dir,
        'inputs.work_dir': work_dir,
        'inputs.val128_k2_targets': val128_k2_targets,
    }
    if gate_receipt is not None:
        fill['routing.gate_receipt'] = str(Path(gate_receipt).resolve())
    if headroom_summary is not None:
        fill['routing.headroom_summary'] = str(Path(headroom_summary).resolve())

    is_scaled = (raw.get('objective') or {}).get('lambda_regime') == 'scaled'
    if is_scaled:
        if not gate_receipt or not headroom_summary:
            raise LaunchRefused(
                'sealing a scaled Run B draft requires both --gate-receipt and --headroom-summary'
            )
        headroom_info = verify_headroom_summary(headroom_summary)
        calibrated_lam = float(headroom_info['lambda_scaled'])
        if l0_lambda is not None and not math.isclose(
            float(l0_lambda), calibrated_lam, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise LaunchRefused(
                f'--l0-lambda {l0_lambda} differs from sealed headroom lambda_scaled {calibrated_lam}'
            )
        fill['objective.l0_lambda'] = calibrated_lam
    elif l0_lambda is not None:
        fill['objective.l0_lambda'] = float(l0_lambda)

    sealed_path = k2p.finalize_draft(
        draft_path,
        out_path,
        code_commit=ident['code_commit'],
        executable_sha256=ident['executable_sha256'],
        pod_id=pod_id,
        fill=fill,
    )
    if is_scaled:
        record = k2p.load_launch_record(
            sealed_path, code_commit=ident['code_commit'], executable_sha256=ident['executable_sha256']
        )
        try:
            data = load_k2_inputs(record)
            _authorize_run_b(record, data=data, arm_dir_override=matched_s42_dir)
        except Exception:
            sealed_path.unlink(missing_ok=True)
            raise
    return sealed_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    seal = sub.add_parser('seal', help='approved k=2 YAML draft -> sealed, typed JSON record')
    seal.add_argument('--draft', required=True)
    seal.add_argument('--out', required=True)
    for key in ('artifact-root', 'bundle', 'model-dir', 'work-dir', 'val128-k2-targets'):
        seal.add_argument(f'--{key}', required=True)
    seal.add_argument('--l0-lambda', type=float, default=None,
                      help='optional objective.l0_lambda (for scaled Run B, auto-filled & checked against headroom)')
    seal.add_argument('--gate-receipt', default=None,
                      help='verified Run A seed-42 gate receipt JSON (required for scaled Run B)')
    seal.add_argument('--headroom-summary', default=None,
                      help='sealed Stage 1 references/summary directory (required for scaled Run B)')
    seal.add_argument('--matched-s42-dir', default=None,
                      help='optional restored Run A seed-42 arm dir if moved from receipt path')

    for name in ('run', 'precheck', 'watchdog-copy', 'paths'):
        p = sub.add_parser(name)
        p.add_argument('--local', required=True)
        p.add_argument('--volume', required=True)
        p.add_argument('--record', required=True)
        p.add_argument('--only-arm', default=None,
                       help='scope execution to one arm (production Run A defaults to seed 42 only)')
        if name == 'run':
            p.add_argument('--gate-receipt', default=None,
                           help='verified Run A seed-42 gate receipt (required for production Run A seed 43)')
            p.add_argument('--matched-s42-dir', default=None,
                           help='completed (or volume-restored) Run A seed-42 arm dir')
            p.add_argument('--stop-pod', action='store_true',
                           help='stop the Pod after a complete verified backup (default: leave it running)')

    gate = sub.add_parser('check-run-b', help='verify completed Run A seed 42 and evaluate K2-D1 (< 4 survivors)')
    gate.add_argument('--arm-dir', required=True)
    gate.add_argument('--expected-final-step', type=int, default=None)
    gate.add_argument('--threshold', type=int, default=k2p.RUN_B_SURVIVOR_THRESHOLD)
    gate.add_argument('--headroom-dir', default=None, help='sealed references/summary directory')
    gate.add_argument('--out', default=None, help='optional path to write sealed gate receipt JSON')

    args = parser.parse_args(argv)
    if args.command == 'seal':
        out = seal_k2_draft(
            args.draft,
            args.out,
            artifact_root=args.artifact_root,
            bundle=args.bundle,
            model_dir=args.model_dir,
            work_dir=args.work_dir,
            val128_k2_targets=args.val128_k2_targets,
            pod_id=os.environ.get('RUNPOD_POD_ID'),
            l0_lambda=args.l0_lambda,
            gate_receipt=args.gate_receipt,
            headroom_summary=args.headroom_summary,
            matched_s42_dir=args.matched_s42_dir,
        )
        print(out)
        return 0

    if args.command == 'check-run-b':
        decision = check_run_b_trigger(
            args.arm_dir,
            expected_final_step=args.expected_final_step,
            threshold=args.threshold,
            headroom_dir=args.headroom_dir,
            write_receipt=args.out,
        )
        print(json.dumps(decision, indent=2))
        return 0

    require_clean = args.command in ('run', 'precheck')
    ident = k2p.verified_code_identity(require_clean=require_clean)
    record = k2p.load_launch_record(
        args.record, code_commit=ident['code_commit'], executable_sha256=ident['executable_sha256']
    )
    default_only = args.only_arm
    if (default_only is None and record['mode'] == 'production'
            and record['objective']['lambda_regime'] == 'matched'):
        default_only = _arm_by_seed(record, 42)
    crec = chain_record(record, only_arm=default_only)
    root = cc.run_root(args.volume, crec['run_id'])
    if args.command == 'paths':
        print(json.dumps(dict(
            run_root=str(root),
            chain_dir=str(root / cc.CHAIN_DIR),
            stop=[' '.join(c) for c in cc.stop_commands(crec)],
            copy_timeout=crec['timeouts_seconds']['copy'],
            stop_timeout=crec['timeouts_seconds']['stop_attempt'],
        )))
        return 0
    if args.command == 'watchdog-copy':
        try:
            pc.observed_volume_identity(args.volume, args.local, expected=record['volume_identity'])
        except ResumeRefused as exc:
            print(f'watchdog-copy refused: {exc}', file=sys.stderr)
            return 3
        result = cc.copy_sealed(args.local, root)
        print(json.dumps(result, indent=2))
        return 0 if result['verified'] and not result['problems'] else 1
    if args.command == 'precheck':
        try:
            identity = pc.observed_volume_identity(args.volume, args.local, expected=record['volume_identity'])
            decision = cc.precheck(crec, args.volume)
        except ResumeRefused as exc:
            print(f'precheck refused: {exc}', file=sys.stderr)
            return 3
        print(json.dumps(dict(
            precheck='ok', run_id=crec['run_id'], volume_identity=identity, decision=decision['decision']
        )))
        return 0

    identity = pc.observed_volume_identity(args.volume, args.local, expected=record['volume_identity'])
    chain = build(
        record,
        args.local,
        args.volume,
        only_arm=args.only_arm,
        gate_receipt=args.gate_receipt,
        matched_s42_dir=args.matched_s42_dir,
        volume_identity=identity,
        stop_pod=args.stop_pod,
    )
    summary = chain.run()
    print(json.dumps(dict(state=summary['state'], backups=summary['backups']['statement'], stop=summary['stop']),
                     indent=2, default=str))
    return 0 if summary['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
