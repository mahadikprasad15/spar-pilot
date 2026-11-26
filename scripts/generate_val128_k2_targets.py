"""Stage 1 generator: create sealed Sampling SFT (k = 2) targets on Gold's frozen 128 validation questions.

Reuses the exact qualified Sampling SFT ``k = 2`` recipe:
- ``pilot_eval.sampling_sft_sampler.sample`` with ``k = 2``, ``J = 64``, ``cap = 1024``, ``max_blocks = 32``
- ``pilot_eval.sampling_sft_generation.retry_seed``: 1 initial attempt + up to 3 bounded retries
  (4 total attempts, ``retry_data_seeds = [0, 1001, 1002, 1003]``)
- Proposal settings matching ``docs/operations/sampling-sft-engine.md``: ``temperature = 0.6``,
  ``top_p = 1.0``, ``top_k = 0`` (disabled, matching vLLM ``top_k = -1``), ``repetition_penalty = 1.1``
- Scored with ``pilot_eval.scoring.score_gsm8k_flexible_v3`` and packed via
  ``pilot_eval.sampling_sft_training_data.supervised_row`` (original stored integer token IDs,
  prompt masked ``-100``, completion labels equal to completion IDs, exactly one final EOS).

Provenance & durability:
- Verifies pinned base model files, tokenizer files, and frozen ``validation.items.json`` before
  generation and binds their SHA-256 digests, precision readback, engine identity, and executable
  source digest into ``provenance`` inside ``val128-k2-targets.json``.
- Uses ``sampling_sft_sampler.sample``'s durable ``on_block`` and ``resume`` hooks to checkpoint
  progress after every block under a manifest hash, resuming cleanly after interruption with
  bit-for-bit identical final targets and evidence (decoded text, full attempts log, and per-ID
  uncovered terminal dispositions).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pilot_eval import c_attribution_bundle as cb
from pilot_eval import catalog as cat
from pilot_eval import protocol as proto
from pilot_eval import rewire_bundle as rb
from pilot_eval import sampling_sft_generation as sgen
from pilot_eval import sampling_sft_sampler as sampler
from pilot_eval import sampling_sft_training_data as td
from pilot_eval import scoring
from pilot_eval import sparse_rewiring_k2_plan as k2p
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

VAL128_K2_SCHEMA = 'sparse-rewiring-k2/val128-targets-v1'
PROGRESS_SCHEMA = 'sparse-rewiring-k2/val128-progress-v1'
QUALIFIED_RETRY_INDICES = (0, 1, 2, 3)
QUALIFIED_RETRY_DATA_SEEDS = [sgen.retry_seed(i) for i in QUALIFIED_RETRY_INDICES]

DEFAULT_RECIPE: dict[str, Any] = {
    'k': 2,
    'jump_tokens': 64,
    'cap_tokens': 1024,
    'max_blocks': 32,
    'max_model_retries': 3,
    'total_attempts_max': 4,
    'data_seed': 0,
    'retry_indices': list(QUALIFIED_RETRY_INDICES),
    'retry_data_seeds': list(QUALIFIED_RETRY_DATA_SEEDS),
    'temperature': 0.6,
    'top_p': 1.0,
    'top_k': 0,                   # disabled in HF Transformers (equivalent to vLLM top_k=-1)
    'repetition_penalty': 1.1,
    'scorer': 'flexible-v3',
    'accept_rule': 'strict-mean-response-logprob-improvement',
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _json_safe_rng(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_json_safe_rng(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe_rng(v) for v in obj]
    return obj


def collect_provenance(
    *,
    model_dir: str | Path,
    tokenizer_dir: str | Path,
    val_items_path: str | Path,
    bundle_path: str | Path | None = None,
    pinned_base_model: Mapping[str, Any] | None = None,
    engine_name: str = 'hf-fp32-ieee',
    precision_readback: Mapping[str, Any] | None = None,
    executable_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify pinned model, tokenizer, and frozen validation inputs and return bound provenance dict."""
    mdir = Path(model_dir)
    tdir = Path(tokenizer_dir)
    vpath = Path(val_items_path)
    if not mdir.is_dir():
        raise ValueError(f'model_dir does not exist: {mdir}')
    if not tdir.is_dir():
        raise ValueError(f'tokenizer_dir does not exist: {tdir}')
    if not vpath.is_file():
        raise ValueError(f'val_items_path does not exist: {vpath}')

    val_file_sha = file_hash(vpath)
    bundle_sha = None

    if bundle_path is not None:
        bpath = Path(bundle_path)
        bundle = json.loads(bpath.read_text(encoding='utf-8'))
        bundle_sha = _hash(bundle)
        cb.check_base_model(mdir, bundle, pinned=pinned_base_model)
        model_hashes = dict(bundle['base_model']['file_sha256'])
        # Verify every tokenizer file in bundle against tokenizer_dir
        tok_hashes: dict[str, str] = {}
        for entry in bundle.get('files', []):
            role = str(entry.get('role', ''))
            if role.startswith('tokenizer:'):
                fname = role.split(':', 1)[1]
                fp = tdir / fname
                if not fp.is_file():
                    raise ValueError(f'tokenizer_dir {tdir} missing bundled tokenizer file {fname}')
                actual = file_hash(fp)
                if actual != entry['sha256']:
                    raise ValueError(f'tokenizer file {fp} SHA-256 {actual} != bundle {entry["sha256"]}')
                tok_hashes[fname] = actual
        # Verify val_items_path matches bundled validation:items SHA-256
        val_entry = next((e for e in bundle.get('files', []) if e.get('role') == 'validation:items'), None)
        if val_entry is None:
            raise ValueError('bundle missing validation:items entry')
        if val_file_sha != val_entry['sha256']:
            raise ValueError(f'val_items {vpath} SHA-256 {val_file_sha} != bundle {val_entry["sha256"]}')
    else:
        pinned = dict(pinned_base_model) if pinned_base_model is not None else cb.pinned_identities(mdir)
        model_hashes = {}
        for fname, expected_sha in sorted(pinned['file_sha256'].items()):
            fp = mdir / fname
            if not fp.is_file():
                raise ValueError(f'model_dir {mdir} missing pinned file {fname}')
            actual = file_hash(fp)
            if actual != expected_sha:
                raise ValueError(f'model file {fp} SHA-256 {actual} != pinned {expected_sha}')
            model_hashes[fname] = actual
        tok_hashes = {}
        for fname in cb.TOKENIZER_FILES:
            fp = tdir / fname
            if not fp.is_file():
                raise ValueError(f'tokenizer_dir {tdir} missing required tokenizer file {fname}')
            tok_hashes[fname] = file_hash(fp)

    return {
        'model_file_sha256': model_hashes,
        'tokenizer_file_sha256': tok_hashes,
        'val_items_sha256': val_file_sha,
        'bundle_sha256': bundle_sha,
        'engine': str(engine_name),
        'precision_readback': dict(precision_readback) if precision_readback is not None else None,
        'executable_sha256': executable_sha256 if executable_sha256 is not None else k2p.executable_code_sha256(),
    }


class HFK2Engine:
    """IEEE FP32 HuggingFace ``generate`` / ``score`` engine compatible with ``sampling_sft_sampler.sample``.

    Explicitly disables ``top_k`` (``top_k=0``) so Qwen's ``generation_config.json`` default
    (``top_k=20``) is never inherited, matching the qualified vLLM sampler's ``top_k=-1``.
    """

    def __init__(
        self,
        model: Any,
        *,
        stops: Sequence[int],
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.1,
        pad_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.stops = set(int(s) for s in stops)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.pad_token_id = pad_token_id
        self.device = next(model.parameters()).device

    def clear_cache(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def resolved_generation_kwargs(self, *, max_new_tokens: int) -> dict[str, Any]:
        """Return explicit HF generation kwargs with zero inherited top-k/min-p filtering."""
        kwargs: dict[str, Any] = {
            'max_new_tokens': int(max_new_tokens),
            'do_sample': self.temperature > 0,
            'temperature': self.temperature if self.temperature > 0 else 1.0,
            'top_p': self.top_p,
            'top_k': self.top_k,   # 0 disables top-k filtering in HF Transformers
            'min_p': None,
            'repetition_penalty': self.repetition_penalty,
            'eos_token_id': sorted(self.stops),
            'use_cache': True,
        }
        if self.pad_token_id is not None:
            kwargs['pad_token_id'] = int(self.pad_token_id)
        return kwargs

    def generate(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        import torch

        outputs: list[dict[str, Any]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for req in requests:
                    prompt_ids = [int(t) for t in req['prompt_ids']]
                    max_new = int(req['max_tokens'])
                    seed = int(req['seed'])
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                    attn = torch.ones_like(inp)
                    gen_kwargs = self.resolved_generation_kwargs(max_new_tokens=max_new)
                    seq = self.model.generate(input_ids=inp, attention_mask=attn, **gen_kwargs)
                    gen_ids = [int(t) for t in seq[0, len(prompt_ids):].tolist()]
                    stop_pos = next((i for i, tok in enumerate(gen_ids) if tok in self.stops), None)
                    if stop_pos is not None:
                        gen_ids = gen_ids[:stop_pos + 1]
                        finish_reason = 'stop'
                        stop_reason = gen_ids[-1]
                    else:
                        gen_ids = gen_ids[:max_new]
                        finish_reason = 'length'
                        stop_reason = None
                    outputs.append({
                        'token_ids': gen_ids,
                        'finish_reason': finish_reason,
                        'stop_reason': stop_reason,
                    })
        finally:
            self.model.train(was_training)
        return outputs

    def score(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        outputs: list[dict[str, Any]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for req in requests:
                    prompt_ids = [int(t) for t in req['prompt_ids']]
                    resp_ids = [int(t) for t in req['response_ids']]
                    full = prompt_ids + resp_ids
                    inp = torch.tensor([full], dtype=torch.long, device=self.device)
                    logits = self.model(input_ids=inp, use_cache=False).logits[0].float()
                    start = len(prompt_ids)
                    logprobs: list[float] = []
                    positions: list[int] = []
                    for offset, tok in enumerate(resp_ids):
                        pos = start + offset
                        lp = F.log_softmax(logits[pos - 1], dim=-1)[tok].item()
                        logprobs.append(float(lp))
                        positions.append(pos)
                    outputs.append({
                        'token_ids': list(resp_ids),
                        'positions': positions,
                        'logprobs': logprobs,
                    })
        finally:
            self.model.train(was_training)
        return outputs


def _build_sampler_problems(val_items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for idx, item in enumerate(val_items):
        prompt_tokens = int(item['prompt_tokens'])
        scorer_prompt_ids = [int(t) for t in item['input_ids'][:prompt_tokens]]
        if 'question' in item and item['question']:
            user_message = proto.gsm8k_messages(item['question'])[0]['content']
        else:
            user_message = str(item.get('prompt') or f'Problem {item["id"]}')
        problems.append({
            'problem_id': idx,
            'source_id': str(item['id']),
            'user_message': user_message,
            'gold': str(item['gold']),
            'scorer_prompt_ids': scorer_prompt_ids,
        })
    return problems


def generate_val128_k2_targets(
    val_items: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
    tokenizer: Any,
    eos_token_id: int,
    stop_token_ids: Sequence[int] | None = None,
    k: int = 2,
    max_retries: int = 3,
    data_seed: int = 0,
    block_limit: int = 32,
    provenance: Mapping[str, Any] | None = None,
    progress_path: str | Path | None = None,
    stop_after_blocks: int | None = None,
) -> dict[str, Any]:
    """Run Sampling SFT ``k = 2`` refinement using the established retry policy (1 initial + up to 3 retries),
    persisting durable per-block checkpoints when ``progress_path`` is set and preserving full evidence."""
    if not val_items:
        raise ValueError('val_items must be non-empty')
    if int(data_seed) != 0:
        raise ValueError(f'established Sampling SFT recipe fixes data_seed == 0, got {data_seed}')
    if not 0 <= int(max_retries) <= 3:
        raise ValueError(f'max_retries must be in 0..3 (established policy), got {max_retries}')

    stops = list(stop_token_ids) if stop_token_ids is not None else [int(eos_token_id)]
    if int(eos_token_id) not in stops:
        stops = [int(eos_token_id), *stops]

    retry_indices = list(range(int(max_retries) + 1))
    retry_data_seeds = [sgen.retry_seed(i) for i in retry_indices]
    ordered_ids = [str(item['id']) for item in val_items]
    ids_digest = cat.ids_digest(ordered_ids)

    recipe = dict(
        DEFAULT_RECIPE,
        k=int(k),
        max_blocks=int(block_limit),
        max_model_retries=int(max_retries),
        total_attempts_max=len(retry_indices),
        data_seed=int(data_seed),
        retry_indices=retry_indices,
        retry_data_seeds=retry_data_seeds,
        eos_token_id=int(eos_token_id),
        stop_token_ids=list(stops),
    )
    prov_dict = dict(provenance) if provenance is not None else None
    manifest_sha = _hash({'ids_sha256': ids_digest, 'recipe': recipe, 'provenance': prov_dict})

    problems = _build_sampler_problems(val_items)
    by_pid = {p['problem_id']: p for p in problems}
    selected_by_id: dict[str, dict[str, Any]] = {}
    attempts_log: list[dict[str, Any]] = []
    last_attempt_by_id: dict[str, dict[str, Any]] = {}
    completed_retries: set[int] = set()
    active_resume_retry: int | None = None
    active_resume_state: dict[str, Any] | None = None

    prog_p = Path(progress_path) if progress_path is not None else None
    if prog_p is not None and prog_p.is_file():
        saved_prog = json.loads(prog_p.read_text(encoding='utf-8'))
        if saved_prog.get('schema') != PROGRESS_SCHEMA or saved_prog.get('manifest_sha256') != manifest_sha:
            raise ValueError(f'{prog_p}: progress manifest mismatch; refuse resuming under changed inputs/recipe')
        selected_by_id = dict(saved_prog.get('selected_by_id') or {})
        attempts_log = list(saved_prog.get('attempts_log') or [])
        last_attempt_by_id = dict(saved_prog.get('last_attempt_by_id') or {})
        completed_retries = set(int(x) for x in (saved_prog.get('completed_retries') or []))
        active_resume_retry = saved_prog.get('active_retry')
        active_resume_state = saved_prog.get('active_resume')

    def _save_progress(*, active_retry: int | None, active_resume: dict[str, Any] | None) -> None:
        if prog_p is None:
            return
        _atomic_write_json(prog_p, {
            'schema': PROGRESS_SCHEMA,
            'manifest_sha256': manifest_sha,
            'ids_sha256': ids_digest,
            'completed_retries': sorted(completed_retries),
            'selected_by_id': selected_by_id,
            'attempts_log': attempts_log,
            'last_attempt_by_id': last_attempt_by_id,
            'active_retry': active_retry,
            'active_resume': active_resume,
        })

    for retry_idx, attempt_seed in zip(retry_indices, retry_data_seeds):
        if retry_idx in completed_retries:
            continue
        remaining = [p for p in problems if p['source_id'] not in selected_by_id]
        if not remaining:
            completed_retries.add(retry_idx)
            _save_progress(active_retry=None, active_resume=None)
            continue

        resume_arg = active_resume_state if active_resume_retry == retry_idx else None

        def _on_block(block_idx: int, states: dict[int, Any], rng_states: dict[int, Any], r_idx=retry_idx) -> None:
            _save_progress(
                active_retry=r_idx,
                active_resume={
                    'next_block': int(block_idx) + 1,
                    'states': {str(pid): st for pid, st in states.items()},
                    'rng': {str(pid): _json_safe_rng(st) for pid, st in rng_states.items()},
                },
            )

        result = sampler.sample(
            remaining,
            k=int(k),
            engine=engine,
            tokenizer=tokenizer,
            stops=stops,
            data_seed=attempt_seed,
            on_block=_on_block if prog_p is not None else None,
            block_limit=int(block_limit),
            resume=resume_arg,
            stop_after_blocks=stop_after_blocks,
        )
        if result['state'] == 'paused':
            return {
                'schema': PROGRESS_SCHEMA,
                'state': 'paused',
                'manifest_sha256': manifest_sha,
                'progress_path': str(prog_p) if prog_p is not None else None,
            }
        if result['state'] == 'engine-error':
            raise RuntimeError(f'sampling_sft_sampler stopped with engine-error on retry {retry_idx}')

        for trace in result['traces']:
            pid = int(trace['problem_id'])
            prob = by_pid[pid]
            resp_ids = list(trace['response_ids'])
            end_state = trace['end_state']
            single_eos = (
                end_state == 'finished'
                and bool(resp_ids)
                and resp_ids[-1] == int(eos_token_id)
                and sum(t == int(eos_token_id) for t in resp_ids) == 1
            )
            text = tokenizer.decode(resp_ids[:-1] if single_eos else resp_ids, skip_special_tokens=True)
            verdict = scoring.score_gsm8k_flexible_v3(text, prob['gold'], capped=(end_state != 'finished'))
            accepted = bool(single_eos and verdict['correct'])
            attempt_id = f"{prob['source_id']}:k{k}:r{retry_idx}"
            attempt_record = {
                'attempt_id': attempt_id,
                'id': prob['source_id'],
                'problem_id': pid,
                'retry': retry_idx,
                'attempt_seed': attempt_seed,
                'end_state': end_state,
                'single_eos': single_eos,
                'mean_logprob': trace['mean_logprob'],
                'response_tokens': len(resp_ids),
                'text': text,
                'verdict': verdict,
                'accepted': accepted,
            }
            attempts_log.append(attempt_record)
            last_attempt_by_id[prob['source_id']] = attempt_record
            if accepted:
                attempt = {
                    'attempt_id': attempt_id,
                    'end_state': end_state,
                    'response_ids': resp_ids,
                }
                row = td.supervised_row(prob, attempt, int(eos_token_id))
                row['gold'] = prob['gold']
                row['text'] = text
                row['mean_logprob'] = trace['mean_logprob']
                row['retry'] = retry_idx
                row['attempt_seed'] = attempt_seed
                selected_by_id[prob['source_id']] = row

        completed_retries.add(retry_idx)
        active_resume_retry = None
        active_resume_state = None
        _save_progress(active_retry=None, active_resume=None)

    covered_items = [selected_by_id[qid] for qid in ordered_ids if qid in selected_by_id]
    covered_ids = [row['id'] for row in covered_items]
    uncovered_ids = [qid for qid in ordered_ids if qid not in selected_by_id]
    uncovered_dispositions: dict[str, dict[str, Any]] = {}
    for qid in uncovered_ids:
        last = last_attempt_by_id.get(qid) or {}
        uncovered_dispositions[qid] = {
            'id': qid,
            'attempts_count': sum(1 for a in attempts_log if a['id'] == qid),
            'last_end_state': last.get('end_state'),
            'last_single_eos': last.get('single_eos'),
            'last_response_tokens': last.get('response_tokens'),
            'last_mean_logprob': last.get('mean_logprob'),
            'last_text': last.get('text'),
            'last_verdict': last.get('verdict'),
        }

    return {
        'schema': VAL128_K2_SCHEMA,
        'state': 'completed',
        'ids_sha256': ids_digest,
        'manifest_sha256': manifest_sha,
        'recipe': recipe,
        'provenance': prov_dict,
        'total_questions': len(ordered_ids),
        'covered_count': len(covered_items),
        'uncovered_count': len(uncovered_ids),
        'covered_ids': covered_ids,
        'uncovered_ids': uncovered_ids,
        'uncovered_dispositions': uncovered_dispositions,
        'items_sha256': _hash(covered_items),
        'items': covered_items,
        'attempts': attempts_log,
        'attempts_summary': {
            'total_attempts': len(attempts_log),
            'accepted': len(covered_items),
            'uncovered': len(uncovered_ids),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate sealed val128-k2 targets on Gold frozen val128')
    parser.add_argument('--val-items', required=True, help='path to frozen validation.items.json')
    parser.add_argument('--model-dir', required=True, help='local pinned Qwen2.5-1.5B-Instruct directory')
    parser.add_argument('--tokenizer-dir', default=None, help='tokenizer directory (default: --model-dir)')
    parser.add_argument('--bundle', default=None, help='sealed input bundle.json to verify pinned model/tokenizer/val')
    parser.add_argument('--out', required=True, help='output path for sealed val128-k2-targets.json')
    parser.add_argument('--progress', default=None, help='durable progress checkpoint path (default: <out>.progress.json)')
    parser.add_argument('--device', default='cuda', help='torch device (default: cuda)')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--block-limit', type=int, default=32)
    parser.add_argument('--skip-id-hash-check', action='store_true',
                        help='allow non-standard validation subset (for testing only)')
    args = parser.parse_args(argv)

    val_items = json.loads(Path(args.val_items).read_text(encoding='utf-8'))
    ids_digest = cat.ids_digest(i['id'] for i in val_items)
    if not args.skip_id_hash_check and ids_digest != rb.VALIDATION_IDS_SHA256:
        raise ValueError(
            f'{args.val_items}: ids_sha256 {ids_digest} != frozen ticket-04 {rb.VALIDATION_IDS_SHA256}'
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pilot_eval import neuron_audit_collector as nac
    from pilot_eval import rewire_evaluation as reval

    nac.set_ieee_precision()
    precision = dict(nac.precision_backends(), matmul_precision=nac._legacy_matmul_precision())
    tok_dir = args.tokenizer_dir or args.model_dir
    provenance = collect_provenance(
        model_dir=args.model_dir,
        tokenizer_dir=tok_dir,
        val_items_path=args.val_items,
        bundle_path=args.bundle,
        engine_name='hf-fp32-ieee',
        precision_readback=precision,
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager'
    )
    tok_ids = reval.resolve_token_ids(tokenizer, model_config=model.config)
    model.to(args.device)

    stop_ids = [tok_ids['eos_token_id']]
    for stop_str in ('<|im_end|>', '<|endoftext|>'):
        tid = tokenizer.convert_tokens_to_ids(stop_str)
        if isinstance(tid, int) and tid >= 0 and tid not in stop_ids:
            stop_ids.append(tid)

    engine = HFK2Engine(
        model,
        stops=stop_ids,
        temperature=DEFAULT_RECIPE['temperature'],
        top_p=DEFAULT_RECIPE['top_p'],
        top_k=DEFAULT_RECIPE['top_k'],
        repetition_penalty=DEFAULT_RECIPE['repetition_penalty'],
        pad_token_id=tok_ids['pad_token_id'],
    )
    out_path = Path(args.out)
    prog_path = Path(args.progress) if args.progress else out_path.with_suffix(out_path.suffix + '.progress.json')
    payload = generate_val128_k2_targets(
        val_items,
        engine=engine,
        tokenizer=tokenizer,
        eos_token_id=tok_ids['eos_token_id'],
        stop_token_ids=stop_ids,
        k=args.k,
        max_retries=args.max_retries,
        data_seed=args.data_seed,
        block_limit=args.block_limit,
        provenance=provenance,
        progress_path=prog_path,
    )
    _atomic_write_json(out_path, payload)
    if prog_path.exists():
        prog_path.unlink()
    print(json.dumps({
        'out': str(out_path),
        'ids_sha256': payload['ids_sha256'],
        'covered_count': payload['covered_count'],
        'uncovered_count': payload['uncovered_count'],
        'items_sha256': payload['items_sha256'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
