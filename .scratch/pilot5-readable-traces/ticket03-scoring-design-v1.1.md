# Ticket 03 design: scoring lens token lists (automatic overlap + blind-grading package)

> **Version 1.1 (15 Oct 2026).** Section A at the end amends this design. Where A differs from §1–§11, **A wins**.

Audience: Codex, implementing Ticket 03 in the SPAR pilot repository.
Author of decisions: Prasad (approved 15 Oct 2026). Questions go back to Prasad, not to guesswork.

---

## 0. How to work on this ticket (read first)

Work in three phases. **Do not skip phase 1, and do not start phase 2 until Prasad approves the preflight.**

### Phase 1 — Preflight (no implementation code)

Write `TICKET_03_PREFLIGHT.md` at the repo root with these sections, in this order:

1. **Ticket 03 verbatim.** Paste the full text of Ticket 03 exactly as it stands in the repo (path + commit). Do not summarize it.
2. **Mapping table.** One row per numbered section of this design (§2–§10). Columns: `design section` · `matching ticket text (quote)` · `status` (`same` / `ticket adds` / `design adds` / `conflict`) · `note`.
3. **Conflicts.** Every point where the ticket and this design disagree, with both texts quoted. Do not resolve a conflict yourself.
4. **Reuse plan.** Existing modules you will reuse (lens code, tokenizer loading, corpus loaders, run-directory and manifest helpers), with file paths and function names. Name any module you must create.
5. **Inputs found.** For every input in §2: the exact path you found, its size, and its SHA-256. Mark anything missing.
6. **Open decisions.** Every value this design leaves to Prasad (marked **[DECIDE]**), with the default you propose.

Then **stop** and print: `PREFLIGHT READY — waiting for approval`.

### Phase 2 — Implementation

Implement §3–§9 with the tests in §10. Follow the freeze rule in §4.4 strictly.

### Phase 3 — Report

Write `TICKET_03_REPORT.md` as specified in §11. Prasad will share it for review.

---

## 1. Purpose

Each activation-difference vector δ̄ is passed through a lens (logit lens, Jacobian lens), which gives a ranked token list. This ticket decides, with a fixed and pre-registered procedure, **how much each list points at the training task**, and whether that is more than chance.

There are two scores:

- **Automatic overlap score (primary).** The fraction of the list's top 20 tokens that fall in a frozen vocabulary of task-enriched tokens, compared against a null distribution from random vectors.
- **Blind LLM grading (supplement).** A package of shuffled, unlabeled lists that a fresh Claude session and Codex label token by token, Minder-style. This ticket builds the package and the agreement analysis. It does not call any API.

There are three separate claims, and each must be tested on its own:

- C1. The SFT trace is task-readable.
- C2. The GRPO trace is task-readable.
- C3. SFT is more readable than GRPO.

C1 or C2 does not imply C3.

---

## 2. Inputs

| ID | Input | Notes |
|---|---|---|
| I1 | Tokenizer `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Same revision as all pilots |
| I2 | The 512 GSM8K **training questions** used by both arms | Question field only, raw text, no chat template |
| I3 | The same 512 **gold solutions** (GSM8K `answer` field, as used for SFT) | Keep the text as is, including `<<…>>` calculator notes and `#### n` if present |
| I4 | GRPO training completions from the GRPO run `results/responses.jsonl` (4,096 accepted completions) | Use all completions, correct or not: they are what the policy produced |
| I5 | Background corpus: FineWeb `sample-10BT` @ `9bb295ddab0e05d785b879661af7260fed5140fc`, **4,000 documents**, disjoint from the 200 measurement documents | Truncate each to its first 1,024 tokens. Target ≥ 2M tokens |
| I6 | δ̄ vectors from the activation run: per arm, checkpoint, layer and position, plus the pooled vector | Primary cell defined in §6.1 |
| I7 | Lens implementations: logit lens (final-norm gain γ + `lm_head`) and the community Jacobian lens (`qbz506/prabodha-lenses`, `qwen2.5-1.5b-instruct.pt`) | If another ticket already implements them, reuse that code |
| I8 | Per-document mean δ at layer 14, positions 0–4 (from the stored per-token δ) | Needed for bootstrap and N-curves (§7) |
| I9 | The random-adapter δ̄ (random-64) | Reference condition, not the null |

Record every input path and SHA-256 in the run manifest.

---

## 3. Token normalization (used everywhere)

Lens rankings are over token IDs. Qwen gives different IDs to ` total`, `total`, ` Total` and `Total`. Matching on IDs would score ` Total` as a miss when the reference holds ` total`. So all matching is on **normalized strings**.

```
norm(id) = unicodedata.normalize("NFKC", tok.decode([id])).strip().lower()
```

Exclude a token ID (it is never counted and never scored) if any of these is true:

- it is in `tok.all_special_ids` or is an added token (for example `<|im_start|>`, `<|endoftext|>`);
- `norm(id)` is the empty string;
- `norm(id)` contains U+FFFD (a partial byte sequence).

**Lens list rule.** Walk the lens ranking from the top. Skip excluded IDs and any ID whose normalized string is already in the list. Stop at **20 unique normalized strings**. The list is these 20 strings, in order. Record the raw IDs and the decoded strings with `repr()` beside them.

Counting rule: **occurrences**, not document frequency. Tokenize each text with `add_special_tokens=False`, map every ID through `norm`, and count each normalized string.

---

## 4. Reference vocabularies

### 4.1 Three task corpora

Build one reference vocabulary for each task corpus A, always against the same background B (I5):

| Name | Corpus A | Role |
|---|---|---|
| `ref_questions` | I2 | **Primary.** Both arms trained on exactly these questions, so it favours neither arm |
| `ref_gold` | I3 | SFT's training text (Minder's convention). It may favour SFT |
| `ref_grpo` | I4 | GRPO's own outputs. It may favour GRPO |

### 4.2 Enrichment score: weighted log-odds with an informative Dirichlet prior

Reference: Monroe, Colaresi & Quinn (2008), "Fightin' Words", *Political Analysis* 16(4).

Do **not** use add-one smoothing. It ranks tokens seen 2–3 times in A and never in B above common task words.

For each normalized string w:

- y_A, y_B = counts of w in A and B; n_A, n_B = total counts.
- Prior: π_w = (y_A + y_B) / (n_A + n_B); α_w = α₀ · π_w, with **α₀ = 1,000** (sensitivity analysis: 100 and 10,000).

```
δ_w = log( (y_A + α_w) / (n_A + α₀ − y_A − α_w) )
    − log( (y_B + α_w) / (n_B + α₀ − y_B − α_w) )
σ²_w = 1/(y_A + α_w) + 1/(y_B + α_w)
z_w  = δ_w / sqrt(σ²_w)
```

### 4.3 Selection

1. Keep strings with **y_A ≥ 5**.
2. Rank by z_w (descending). Break ties by y_A (descending), then by the string (ascending).
3. **Reference set R** = the top **200** strings. **Grader list** = the top **100** strings of `ref_questions` (§8).
4. Split R into two subsets:
   - `format_number`: the normalized string contains no letter (regex `[^\W\d_]` finds nothing). This covers digits, `####`, `$`, `=`, `%`, `<<`.
   - `word`: everything else.

### 4.4 Freeze rule (pre-registration)

- Building references and the null **must not** read any learned δ̄ or any lens list from a learned vector.
- After building, write `references/freeze.json` with the full lists, the z-scores, the counts, every parameter, the input hashes, and one SHA-256 over all of it.
- The scorer **must refuse** to score any learned vector (SFT, GRPO, random adapter, baselines) unless `freeze.json` exists and its hash matches. Unit tests may use synthetic vectors without a freeze.
- Also write `references/top30.md`: for each reference, the top 30 strings with y_A, y_B and z, for Prasad to read before anything is scored.

---

## 5. Scores for one token list

Let L be the 20 normalized strings of a list (§3), and R a frozen reference set.

| Score | Definition | Role |
|---|---|---|
| `overlap` | \|L ∩ R\| / 20 | **Primary** |
| `overlap_word` | \|L ∩ R_word\| / 20 | Content |
| `overlap_format` | \|L ∩ R_format_number\| / 20 | Answer format and numbers |
| `soft_rank` | Spearman ρ between the lens logit and z_w, over all strings w with y_A + y_B ≥ 5. If several IDs share a normalized string, take the maximum logit | Secondary. It needs no cutoff, and it does not depend on the scale of the vector |

Compute every score for each of the three references.

---

## 6. Conditions, null and baselines

### 6.1 Primary cell

Layer 14, checkpoint 64, raw-text input form. **[DECIDE]** which vector form is primary:

- **(a) Minder-faithful (proposed default):** five per-position vectors δ̄₀…δ̄₄. The score of a condition is the **maximum over the 5 positions**. The null uses the same maximum (§6.2).
- **(b) Pooled:** one vector, the mean over positions 0–4.

Implement both. The one that is not primary is reported as secondary. Record the choice in the config before any learned vector is scored.

Secondary cells (same scores, reported as secondary and not tested for C1–C3): all other layers in the stored sums, checkpoints 8/16/32, the user-turn and assistant-turn input forms, and the pooled positions 5–127.

### 6.2 Null distribution (random vectors)

- **Primary null: isotropic.** Draw g ~ N(0, I₁₅₃₆) and scale it to ‖δ̄‖ of the vector being tested. For form (a), draw 5 independent vectors, one per position, each scaled to that position's norm, and take the maximum score over the five, exactly as for the learned vector.
- **Secondary null: activation-shaped.** g ~ N(0, Σ), where Σ is the covariance of the untuned layer-14 activations computed from the stored per-token data. It is more realistic, because real differences live in an anisotropic space.
- N_null = **1,000** draws per lens and per reference. Seed `20261015`. Store every null score.
- One-sided p-value: p = (1 + #{null ≥ observed}) / (1 + N_null). Also report the observed score's percentile in the null.

### 6.3 Reference conditions (not nulls; report beside the results)

- The random-adapter δ̄ (I9), through the same lenses.
- Minder's baselines: the lens readout of the **mean untuned activation** and of the **mean adapted activation**, per position, at layer 14. Use GSM8K question text if per-position activations on it exist; otherwise use the FineWeb measurement documents. State which in the report.
- The step-0 adapter: its δ̄ must be exactly zero. Assert this, then skip it.

### 6.4 Multiple testing

- **C1 / C2 family:** 4 tests = {SFT, GRPO} × {logit lens, Jacobian lens}, using `overlap` on `ref_questions`, at the primary cell. Apply a Holm correction across these 4 tests at α = 0.05.
- **C3:** For each lens, d = overlap_SFT − overlap_GRPO on `ref_questions`. Compute a 95% interval from the paired whole-document bootstrap in §7. The same document draw is used for both arms. With two lenses, use a Holm-corrected bootstrap p-value. Report d on `ref_gold` and `ref_grpo` as secondary.
- Everything else is descriptive. Label it "secondary" in every table.

---

## 7. Bootstrap, N-curves and SNR

- **Unit of resampling:** the document. Draw document IDs with replacement, keep each document's records together, recompute δ̄ (per position) from the per-document means (I8), re-apply the lens, and re-score.
- B = **1,000** resamples, seed `20261016`. Report the percentile 95% interval of each primary score.
- **N-curves:** for N ∈ {25, 50, 100, 200}, draw **200** subsets of size N without replacement (seed `20261017`) and score each. Report the median and the 5–95% range. Keep a record of the legacy-150 / new-50 membership of each draw. Do not interpret differences between the two groups as an effect of exposure.
- **SNR** (one fixed estimator): for each δ̄ₚ, with per-document means m_d (d = 1…N),

  ```
  SE  = sqrt( (1/N) · (1/(N−1)) · Σ_d ‖m_d − δ̄‖² )
  SNR = ‖δ̄‖ / SE
  ```

  Report the SNR beside every score. Matched SNR is a descriptive aid. It is not proof that the two adapters hold the same information.

---

## 8. Blind-grading package (no API calls)

Build `grading/` with three files:

- `rubric.md` contains:
  - the task description, written without gold-solution details: *"Grade-school math word problems. The model reads a word problem, reasons about it, and gives a final numeric answer."*;
  - the **top 100 of `ref_questions`** (strings only);
  - the rule: *"For each token, answer 1 if it relates to this task or to writing its answer, else 0. Judge each token on its own. Output JSON only."*;
  - the exact output format below.
- `lists.jsonl` contains one row per list: `{"list_id": "L-<8 hex>", "tokens": ["' total'", "'####'", …]}`, with tokens as `repr()` strings so that spaces are visible. IDs come from a salted hash, and rows are in random order (seed `20261018`). The lists are:
  - SFT, GRPO and random adapter × positions 0–4 and pooled × 2 lenses, at the primary cell;
  - Minder's two baselines × positions × lenses;
  - 20 isotropic-null lists per lens.
- `key.json` maps list_id to its condition. **It must never be in the same folder or message as the lists.** Write it to `grading_key/`.

Expected grader output: `{"list_id": "L-…", "labels": [0/1 × 20]}` per row, in `grader_claude.jsonl` and `grader_codex.jsonl`.

Implement `analyze_grades.py`, which:
- computes the per-list relevance fraction for each grader;
- computes Cohen's κ between the two graders over all token labels, plus the per-list agreement;
- writes a disagreement file for Prasad to settle;
- only after settlement, joins `key.json` and reports relevance by condition next to the automatic scores.

If κ < 0.4, the report says that the semantic score is unstable, and the conclusions rest on the automatic score.

---

## 9. Outputs

Write all files under `runs/pilot-5-readable-traces/analysis/ticket03/`:

```
config.json                 # every parameter in this design + the [DECIDE] choices
references/freeze.json      # §4.4
references/top30.md
null/<lens>__<ref>.npz      # all null scores
scores/scores.parquet       # one row per (condition, layer, position|pooled|max, lens, ref)
                            # columns: overlap, overlap_word, overlap_format, soft_rank,
                            #          p_value, null_percentile, snr, norm, list_strings, list_ids
bootstrap/primary.parquet
ncurves/ncurves.parquet
grading/  grading_key/
manifest.json               # input hashes, code commit, seeds, timings
```

Commit the code, `config.json`, `references/` and the report. Do not commit large arrays.

---

## 10. Tests (must pass before any learned vector is scored)

1. **Normalization:** ` total`, `total`, ` Total` → `total`; special IDs and U+FFFD strings excluded; the list rule returns exactly 20 unique strings.
2. **Log-odds:** on a hand-built toy corpus, z values match a hand computation to 1e-9. The rare-token case is ranked below the frequent enriched token: y_A=2, y_B=0 must rank below y_A=300, y_B=150 with n_A=61,000 and n_B=500,000.
3. **Add-one contrast (documentation test):** on the same toy data, add-one ranks the rare token higher. This shows why it is not used.
4. **Freeze:** the scorer refuses learned vectors without a valid `freeze.json`; editing one byte of the frozen list changes the hash and causes a refusal.
5. **Determinism:** two runs with the same seeds give identical null, bootstrap and package files.
6. **Null p-value:** an observed value above every null score gives p = 1/(N_null+1).
7. **Max-over-positions:** the null applies the same maximum over 5 draws as the learned score.
8. **Package blindness:** `lists.jsonl` contains no arm, lens, position or checkpoint name, and no field other than `list_id` and `tokens`.
9. **κ:** returns 1.0 for identical synthetic labels and about 0 for independent random labels.
10. **Zero adapter:** step-0 δ̄ is exactly zero.

---

## 11. Final report: `TICKET_03_REPORT.md`

Sections, in order:

1. **Ticket 03 verbatim** again (in case it changed), plus the approved preflight decisions.
2. **What was implemented:** files, functions, and the commit hash.
3. **Deviations** from this design and from the ticket, each with its reason.
4. **Test results:** each test in §10 with pass/fail and its output.
5. **Frozen references:** the freeze hash and the `top30.md` content for all three references. Note anything that looks wrong; for example, names or calculator notes dominating.
6. **Inputs:** paths, sizes and SHA-256 values; token counts of A and B for each reference.
7. **Results, primary:** one table for C1/C2 (4 tests: observed overlap, null percentile, raw p, Holm p, bootstrap interval, SNR) and one for C3 (d per lens with interval and Holm p).
8. **Results, secondary:** the subsets, the other references, soft_rank, the other layers, checkpoints and forms, and the reference conditions. Clearly marked secondary.
9. **The token lists themselves** for the primary cell (all conditions, both lenses), with raw IDs and `repr` strings.
10. **Grading package:** the number of lists and the file hashes. Confirm that the key was kept separate.
11. **Open questions and limits.**

Do not write conclusions beyond what the tables show. Use "is consistent with" for anything that the primary tests do not establish.

---

## A. Amendments v1.1 (approved by Prasad, 15 Oct 2026). These override §1–§11.

### A1. Scope and ticket boundaries
- **Ticket 03 freezes the procedure.** It implements and tests the reference builder, the scorer, the null generator, the bootstrap and N-curve procedures, the grading-package builder and the agreement analysis, using synthetic vectors only. It does not compute learned readouts or scientific results.
- Learned readouts belong to Ticket 08, statistical results to Ticket 09, and grading to Ticket 10. §6–§8 and §11 describe what those consumers will run.
- Input forms: **raw is primary**, the assistant-turn form is optional, and the user-turn form is **deferred**. Remove the user-turn form from §6.1.

### A2. Primary vector (replaces the [DECIDE] in §6.1)
- **Primary: one pooled vector**, the mean δ over positions 0–4, at layer 14, checkpoint 64, raw form, as recorded in ADR 0011.
- Max-over-five per-position vectors are **secondary**. When they are reported, the null and the bootstrap use the same maximum.

### A3. Background corpus (replaces I5)
1. Source: FineWeb `sample-10BT` @ `9bb295ddab0e05d785b879661af7260fed5140fc`.
2. Exclude the 200 measurement documents.
3. Remove exact duplicates: SHA-256 of whitespace-normalized text; keep the first occurrence in source order.
4. Drop documents shorter than 64 tokens (`add_special_tokens=False`).
5. Shuffle the remaining document indices with seed `20261019`.
6. Take documents in that order, each truncated to its first 1,024 tokens, until the total reaches **exactly 2,000,000 tokens**. Truncate the last document as needed.
7. Record the selected indices, their order and a SHA-256 over them.

2,000,000 is a requirement, not a target. CPU only; the measurement corpus is not changed.

### A4. Null vectors are scale-free and fully pre-computable (replaces the scaling in §6.2)
- The logit lens (W_U(γ ⊙ v)) and the Jacobian lens are linear in v. Every score in §5 depends only on the token ranking, so multiplying v by a positive number changes no score.
- Therefore **null draws are unit vectors**, with no reference to any learned norm. The full null distributions (isotropic, and activation-shaped if available) are computed and frozen in Ticket 03, before any learned vector exists.
- Add a test: the scores of v and of 7.3·v are identical for both lenses.
- General rule for any later scale-dependent readout (for example Patchscope): freeze the procedure and seeds in advance, and compute after capture.

### A5. C3 test (replaces "bootstrap p-value" in §6.4)
- Paired whole-document bootstrap, **B = 2,000**, seed `20261016`. Each resample draws document IDs once; both arms' pooled vectors are recomputed from that same draw. d*_b = overlap_SFT*_b − overlap_GRPO*_b.
- **Two-sided** test of H₀: d = 0:
  `p = min(1, 2 · min((1 + #{d* ≤ 0}) / (B + 1), (1 + #{d* ≥ 0}) / (B + 1)))`
  Resamples with d* = 0 count in both tails (conservative).
- Holm correction across the two lenses. Also report the 95% percentile interval of d*.

### A6. Activation-shaped null (defines §6.2's secondary null)
- **Requirement on the capture tickets (04–07):** during the base pass, raw form, layer 14, accumulate Σh and Σhhᵀ in float64 over every base token at positions **1–127** of the 200 documents (position 0 excluded because of its extreme norm).
- Covariance: Σ = (1/(n−1)) · Σ (h − μ)(h − μ)ᵀ, with μ the pooled mean, all positions 1–127 pooled.
- Draws: g = Σ^½ z, z ~ N(0, I), normalized to unit length (A4).
- If capture does not provide these sums, this null is **not run**, and the report says so. It is secondary.

### A7. Data requirement for bootstrap (clarifies §7)
- **Requirement on the capture tickets:** for every condition, store per-document δ at layer 14 for each of positions 0–4 separately. The primary bootstrap uses the per-document mean over positions 0–4. The secondary max-over-five bootstrap uses the per-position values.

### A8. Legacy-150 / new-50 (replaces the last N-curve bullet in §7)
- Compute the primary scores on **legacy-150 alone** and on **new-50 alone**, each with a within-group bootstrap (B = 2,000) and a 95% interval.
- **N-matched check:** draw 1,000 random 50-document subsets of legacy-150 (seed `20261020`), score each, and report the percentile of the new-50 score in that distribution. Do not compare the 150 and 50 results directly.
- Report as robustness only; no claim of an exposure effect.

### A9. Grading: two separate modes (extends §8)
- **Relevance mode** (as in §8). It measures relevance *to a disclosed task*, the same quantity as Minder's token-relevance grader. Label it this way in every report.
- **Naming mode (new).** It tests whether the task can be *inferred*. A grader receives the lists with **no task description** and, for each list, picks the most likely training task from 8 fixed candidates in this order: (1) GSM8K-style math word problems; (2) Python programming; (3) English–French translation; (4) medical question answering; (5) legal documents; (6) poetry; (7) product-review sentiment; (8) chemistry. The output is `{"list_id": …, "choice": 1–8}`. Score: top-1 accuracy by condition, against chance (1/8) and against the null lists.
- Package: `grading/naming_rubric.md` + the same `lists.jsonl`.
- **Order and isolation:** each grader does naming in a fresh session **before** any session that has seen the relevance rubric. Use different sessions for the two modes.

### A10. Short and degenerate cases (new)
| Case | Rule |
|---|---|
| Fewer than 200 eligible reference strings | R = all strings with y_A ≥ 5 **and z > 0**, in z order (this also applies when there are more than 200: never include z ≤ 0). Record \|R\|. Do not relax the count rule. Flag if \|R\| < 100. |
| A list cannot supply 20 valid strings | Walk the ranking up to rank 1,000. If still fewer than 20, the score = matches ÷ the number found. Flag it. Nulls use the same rule. |
| Tie in logits at the rank-20 boundary | Break the tie by token ID, ascending. |
| δ̄ exactly zero | Assert it, mark the condition `undefined`, and do not score it. |
| soft_rank undefined (constant logits) | NaN; excluded and reported. |
| SNR undefined (N < 2 or SE = 0) | NaN; reported. |
| κ undefined (a grader gives a constant label) | Report percent agreement only, with the reason. |

### A11. Changes to tests (§10)
Add: the scale-invariance test (A4); the C3 p-value formula on a synthetic d* distribution (A5); the degenerate-case rules (A10); the background selection is deterministic and reaches exactly 2,000,000 tokens (A3).
