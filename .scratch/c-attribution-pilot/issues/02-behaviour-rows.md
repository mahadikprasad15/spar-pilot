# 02: Rows from saved behaviour records

**Status:** needs-triage
**Type:** task
**Blocked by:** 01

**What to build:** `pilot_eval/c_attribution_rows.py`. It reads a behaviour `responses.jsonl` (online or offline GRPO), selects `step == 64, mode == 'sampled'`, and builds instrument rows plus a sealed manifest.

- [ ] Inputs named by catalog id or path, with file sha256, select, and the expected arm adapter sha256 (from the launch record). Refuse any mismatch.
- [ ] **No filtering.** Keep wrong, invalid, capped, all-correct and all-wrong rows. Counts are recorded: 1,200 rows, 150 × 8, group types, capped and invalid.
- [ ] Prompt IDs = pinned tokenizer applied to the stored templated `prompt` with `add_special_tokens=False`. `input_ids` = prompt IDs + `generated_token_ids`. Row id = `draw_id`.
- [ ] Refuse: duplicate `draw_id`; per-row `adapter_sha256` ≠ checkpoint; an EOS row whose last token isn't 151645 or that has 151645 earlier; a capped row without exactly 1,024 tokens or containing 151645; a stored `reward` that disagrees with `score_gsm8k_flexible_v3(text, gold, capped=capped)`; a scorer source hash ≠ `c958d2de…`.
- [ ] Manifest records reward, group id, draw index, seed and unit_key per row (for the replay check in 05).
- [ ] Tests: real-format fixtures, including copies of 2 real groups per arm checked against the committed counts; every refusal above; a CPU check against the real local files (skipped if absent) that reproduces 1,200 rows and the Data-table group counts for both arms.

## Comments

## Answer
