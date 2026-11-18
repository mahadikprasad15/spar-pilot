# 02: Rows from saved behaviour records

**Status:** needs-triage
**Type:** task
**Blocked by:** 01

**What to build:** `pilot_eval/c_attribution_rows.py`. It reads a behaviour `responses.jsonl` (online or offline GRPO) and selects the rows with `step == 64, mode == 'sampled'`. From them it builds instrument rows and a sealed manifest.

- [ ] **Bound inputs.** Each input is named by catalog id or path, together with:
  - the file's sha256 and the select;
  - the expected arm adapter sha256, from the launch record;
  - the **source run config** (sha256).

  Any mismatch is refused.
- [ ] **Provenance block** in the manifest:
  - **recorded:** the base, tokenizer and dataset revisions; the chat template hash; the sampled policy (`do_sample`, temperature, top_k, top_p, `num_beams`, cap, draws, `batch_size`); the seed rule; the runtime versions and device; and `implementation.engine`, checked to equal sha256(`pilot_eval/grpo_eval_engine.py`).
  - **inferred:** `repetition_penalty` and `no_repeat_ngram_size` (from `grpo_gpu._generate`, which is unhashed), with the git evidence.
  - **unverified:** the resolved logits processors, until ticket 05's check passes.
- [ ] **No filtering.** Wrong, invalid, capped, all-correct and all-wrong rows are all kept. The recorded counts are 1,200 rows (150 × 8), plus the group types and the capped and invalid counts.
- [ ] **Row construction.**
  - Prompt IDs: the pinned tokenizer applied to the stored templated `prompt`, with `add_special_tokens=False`.
  - `input_ids`: the prompt IDs followed by `generated_token_ids`.
  - Row id: `draw_id`.
  - Rows are grouped by question, so a unit always holds all 8 draws of a question.
- [ ] **Refusals.** Refuse any of the following:
  - a duplicate `draw_id`;
  - a per-row `adapter_sha256` that differs from the checkpoint;
  - an EOS row whose last token isn't 151645, or that has 151645 earlier;
  - a capped row without exactly 1,024 tokens, or containing 151645;
  - a stored `reward` that disagrees with `score_gsm8k_flexible_v3(text, gold, capped=capped)`;
  - a scorer source hash other than `c958d2de…`;
  - a question with fewer than 2 draws.
- [ ] **Per-row manifest fields:** reward, group id, draw index, seed and source unit or batch identity (for the replay in ticket 05).
- [ ] **Tests:**
  - real-format fixtures, including copies of 2 real groups per arm, checked against the committed counts;
  - every refusal above;
  - the provenance block, with a mutated engine hash and a mutated policy field each refused;
  - a CPU check against the real local files (skipped if absent) that reproduces 1,200 rows and the Data-table group counts for both arms.

## Comments

- 2026-11-18 (round 1): source-config binding, the engine-hash check and the recorded/inferred/unverified provenance block were added for F4.

## Answer
