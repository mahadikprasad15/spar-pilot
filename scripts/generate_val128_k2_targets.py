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

The output ``val128-k2-targets.json`` binds ``ids_sha256`` to Gold's frozen
``artifacts/validation/sparse-rewiring-ticket04-v1/validation.items.json`` so held-out ``val128-k2``
CE is computed strictly on the exact same 128 questions used in Gold SFT sparse rewiring.

Usage (on A100 Pod, HF FP32 IEEE engine):
    python scripts/generate_val128_k2_targets.py \\
        --val-items artifacts/validation/sparse-rewiring-ticket04-v1/validation.items.json \\
        --model-dir $WORK/model \\
        --out $WORK/val128-k2-targets.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pilot_eval import catalog as cat
from pilot_eval import protocol as proto
from pilot_eval import rewire_bundle as rb
from pilot_eval import sampling_sft_generation as sgen
from pilot_eval import sampling_sft_sampler as sampler
from pilot_eval import sampling_sft_training_data as td
from pilot_eval import scoring
from pilot_eval.workflow import _hash

VAL128_K2_SCHEMA = 'sparse-rewiring-k2/val128-targets-v1'
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
) -> dict[str, Any]:
    """Run Sampling SFT ``k = 2`` refinement using the established retry policy (1 initial + up to 3 retries).

    Uses ``pilot_eval.sampling_sft_generation.retry_seed`` (seeds ``0, 1001, 1002, 1003`` when
    ``data_seed == 0`` and ``max_retries == 3``). Retains only attempts that finish cleanly with a
    single final ``eos_token_id`` and whose decoded response scores ``correct == True`` under
    ``scoring.score_gsm8k_flexible_v3``.
    """
    if not val_items:
        raise ValueError('val_items must be non-empty')
    if int(data_seed) != 0:
        raise ValueError(f'established Sampling SFT recipe fixes data_seed == 0, got {data_seed}')
    if not 0 <= int(max_retries) <= 3:
        raise ValueError(f'max_retries must be in 0..3 (established policy), got {max_retries}')

    stops = list(stop_token_ids) if stop_token_ids is not None else [int(eos_token_id)]
    if int(eos_token_id) not in stops:
        stops = [int(eos_token_id), *stops]

    retry_indices = list(range(int(max_retries) + 1))   # 0 (initial) + 1..max_retries -> 4 total attempts
    retry_data_seeds = [sgen.retry_seed(i) for i in retry_indices]

    problems = _build_sampler_problems(val_items)
    by_pid = {p['problem_id']: p for p in problems}
    selected_by_id: dict[str, dict[str, Any]] = {}
    attempts_log: list[dict[str, Any]] = []

    remaining = list(problems)
    for retry_idx, attempt_seed in zip(retry_indices, retry_data_seeds):
        if not remaining:
            break
        result = sampler.sample(
            remaining,
            k=int(k),
            engine=engine,
            tokenizer=tokenizer,
            stops=stops,
            data_seed=attempt_seed,
            block_limit=int(block_limit),
        )
        next_remaining: list[dict[str, Any]] = []
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
            attempts_log.append({
                'attempt_id': attempt_id,
                'id': prob['source_id'],
                'problem_id': pid,
                'retry': retry_idx,
                'attempt_seed': attempt_seed,
                'end_state': end_state,
                'single_eos': single_eos,
                'mean_logprob': trace['mean_logprob'],
                'response_tokens': len(resp_ids),
                'verdict': verdict,
                'accepted': accepted,
            })
            if accepted:
                attempt = {
                    'attempt_id': attempt_id,
                    'end_state': end_state,
                    'response_ids': resp_ids,
                }
                row = td.supervised_row(prob, attempt, int(eos_token_id))
                row['gold'] = prob['gold']
                row['mean_logprob'] = trace['mean_logprob']
                row['retry'] = retry_idx
                row['attempt_seed'] = attempt_seed
                selected_by_id[prob['source_id']] = row
            else:
                next_remaining.append(prob)
        remaining = next_remaining

    ordered_ids = [str(item['id']) for item in val_items]
    covered_items = [selected_by_id[qid] for qid in ordered_ids if qid in selected_by_id]
    covered_ids = [row['id'] for row in covered_items]
    uncovered_ids = [qid for qid in ordered_ids if qid not in selected_by_id]

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
    return {
        'schema': VAL128_K2_SCHEMA,
        'ids_sha256': cat.ids_digest(ordered_ids),
        'recipe': recipe,
        'total_questions': len(ordered_ids),
        'covered_count': len(covered_items),
        'uncovered_count': len(uncovered_ids),
        'covered_ids': covered_ids,
        'uncovered_ids': uncovered_ids,
        'items_sha256': _hash(covered_items),
        'items': covered_items,
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
    parser.add_argument('--out', required=True, help='output path for sealed val128-k2-targets.json')
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
    tok_dir = args.tokenizer_dir or args.model_dir
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
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
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
