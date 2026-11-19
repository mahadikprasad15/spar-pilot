# 02: Rows from saved behaviour records

**Status:** resolved
**Type:** task
**Blocked by:** 01

**What to build:** `pilot_eval/c_attribution_rows.py`. It reads a behaviour `responses.jsonl` (online or offline GRPO) and selects the rows with `step == 64, mode == 'sampled'`. From them it builds instrument rows and a sealed manifest.

- [x] **Bound inputs.** Each input is named by catalog id or path, together with:
  - the file's sha256 and the select;
  - the expected arm adapter sha256, from the launch record;
  - the **source run config** (sha256).

  Any mismatch is refused.
- [x] **Provenance block** in the manifest:
  - **recorded:** the base, tokenizer and dataset revisions; the chat template hash; the sampled policy (`do_sample`, temperature, top_k, top_p, `num_beams`, cap, draws, `batch_size`); the seed rule; the runtime versions and device; and `implementation.engine`, checked to equal sha256(`pilot_eval/grpo_eval_engine.py`).
  - **inferred:** `repetition_penalty` and `no_repeat_ngram_size` (from `grpo_gpu._generate`, which is unhashed), with the git evidence.
  - **unverified:** the resolved logits processors, until ticket 05's check passes.
- [x] **No filtering.** Wrong, invalid, capped, all-correct and all-wrong rows are all kept. The recorded counts are 1,200 rows (150 × 8), plus the group types and the capped and invalid counts.
- [x] **Row construction.**
  - Prompt IDs: the pinned tokenizer applied to the stored templated `prompt`, with `add_special_tokens=False`.
  - `input_ids`: the prompt IDs followed by `generated_token_ids`.
  - `masks.seq_targets`: every generated token's position in `input_ids`, in order: from
    `len(prompt_ids)` through `len(input_ids) - 1` inclusive. This includes the sampled EOS,
    and on capped rows all 1,024 generated tokens. Refuse an empty prompt or generated sequence.
    Supply `masks.length = len(input_ids)`; these are target positions, not shifted logit positions.
  - Row id: `draw_id`.
  - Rows are grouped by question, so a unit always holds all 8 draws of a question.
- [x] **Refusals.** Refuse any of the following:
  - a duplicate `draw_id`;
  - a per-row `adapter_sha256` that differs from the checkpoint;
  - an EOS row whose last token isn't 151645, or that has 151645 earlier;
  - a capped row without exactly 1,024 tokens, or containing 151645;
  - a stored `reward` that disagrees with `score_gsm8k_flexible_v3(text, gold, capped=capped)`;
  - a scorer source hash other than `c958d2de…`;
  - a question with fewer than 2 draws.
- [x] **Per-row manifest fields:** reward, group id, draw index, seed and source unit or batch identity (for the replay in ticket 05).
- [x] **Tests:**
  - real-format fixtures, including copies of 2 real groups per arm, checked against the committed counts;
  - exact target masks for a multi-token prompt with an EOS response, an EOS-only response,
    and a capped response, including their final target in each case;
  - every refusal above;
  - the provenance block, with a mutated engine hash and a mutated policy field each refused;
  - a CPU check against the real local files (skipped if absent) that reproduces 1,200 rows and the Data-table group counts for both arms.

## Comments

- 2026-11-19 (Claude): implemented in `pilot_eval/c_attribution_rows.py` as `load` (hashes, provenance, refusals, counts; no tokenizer) then `instrument_rows` (tokenize, masks) then `seal`/`verify` (idempotent, never overwrites a different seal). Additional checks beyond the list: each row's prompt and gold equal the run config item; each row's seed and question match its recorded unit; each question has exactly draws 0..7; the encoded prompt length equals the run's recorded `prompt_tokens`; an optional tokenizer identity (revision, chat template hash) must match the run. Fixtures are copies of 2 real groups per arm (online: one all-wrong, one mixed with an invalid row; offline: the capped row's group and a mixed group), plus one greedy row to exercise the select.

- 2026-11-18 (round 1): source-config binding, the engine-hash check and the recorded/inferred/unverified provenance block were added for F4.

## Answer

`pilot_eval/c_attribution_rows.py`; tests in `tests/test_c_attribution_rows.py` (36 pass). On the real local files both arms reproduce the Data table: 1,200 rows of 3,150; groups 49/12/89 and 57/13/80; capped 0 and 1; invalid 46 and 39; mean reward 0.675 and 0.6933. Every stored reward agrees with the re-score. Each of the 22 refusal fixtures was checked to fail for its intended reason. A cached Qwen2.5-0.5B tokenizer (same chat template hash `2e2d2512…`) reproduces the recorded prompt token counts for all 150 prompts; the pinned 1.5B revision `989aa798…` is not cached locally, so the production tokenizer is not yet exercised. Tests were run with `~/.codex/worktrees/sampling-hf-publish/pilot1/.venv` (torch 2.9.0, transformers 4.57.6, PEFT 0.18.0); the base miniforge torch install is missing `libtorch_cpu.dylib`.
