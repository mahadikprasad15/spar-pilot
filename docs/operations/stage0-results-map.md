# Stage 0 results map (for report writing)

Where every Stage 0 measurement lives, what each file holds and how far each
experiment got. Written 2026-10-30, after the matched-128 E0.6a/E0.5 GPU run.

Read the sealed files below; do not recompute from notebooks. Every product has a
`complete.json` (or named `*.complete.json`) seal listing file SHA-256s; check it
before quoting a number. A `verify` command exists for each product and changes
nothing.

## Conventions

- Artifact root (Colab): `/content/drive/MyDrive/SPAR/spar-pilot/artifacts`. All
  paths below are relative to it. `S0 = stage0-geometry-v1` (the `STAGE0_NAME`).
- `plans/<name>/stage0.<kind>.json` is the frozen plan; its `run_path` field names
  the run folder. Results are under `<run_path>/results/` unless stated.
- Model: Qwen2.5-1.5B-Instruct, FP32, TF32 off, GSM8K. Adapters are rank 1 on all
  196 modules (28 layers × q,k,v,o,gate,up,down). SFT = gold SFT run
  `pilot2-sft-fp32-l4-batch8-v1` (via `plans/pilot3-inputs-v1/activation.prepared.json`);
  GRPO = `pilot4-grpo-fp32-v1-writes-v1`; a random-direction adapter is the control.
- Labels: E0.5 endpoint alignment and E0.2 trained overlaps are **exploratory**
  (no registered predictions; the reveal receipts say so). Report them as such.
- Superseded products are kept as evidence, never deleted. Use only the names
  listed here; older `-v1`/`-v2` siblings are failed or superseded attempts.

## Shared sources

| Product | Location | Contents |
|---|---|---|
| Source audit | `plans/S0/stage0.plan.json` → `runs/stage0/S0/results/source-audit.json` | Verified SFT/GRPO/random identities, checkpoints 0–64, A₀ equality, model revision |
| S0 spectral cache (base SVD) | `plans/S0-spectral-v1/stage0.spectral.json` → `runs/stage0/S0-spectral-v1/spectral/results/cache.json` | Per-module SVD of base weights; `k50` rank and basis arrays used by π_out/π_in |
| Matched-128 cohort | `plans/S0-matched128-cohort-v2/stage0.cohort.json` → `runs/stage0/S0-matched128-cohort-v2/cohort/results/` | `rollouts.jsonl` (128 questions × 8 historical Pilot 4 untuned draws), `gold.jsonl`, `datasets.json` (gold / own_all / own_correct / own_correct_one memberships, p_correct) |
| Usage ledger | `runs/stage0/budget/ledger.jsonl`, `reservations.jsonl` | Cumulative Colab units (21.9+ after E0.5); ADR 0017 removed the cap for E0.6a/E0.5 |

## E0.4: edits and controls, behaviour and internals (primary, complete)

Question: what do the trained SFT/GRPO adapters do when their principal parts are
edited (V-variants) versus controls?

| Part | Plan / run folder | Key files |
|---|---|---|
| Edit catalogue | `plans/S0-principal-edits-v1/stage0.edits.json` → `runs/stage0/S0-principal-edits-v1/edits/` | `results/catalogue.json` (variant definitions; the `projection_report_revealed: false` field there is static) |
| Batch agreement gate | `runs/stage0/S0-behavior-b16-v2/batch-agreement/results/agreement.json` | Batch-16 vs batch-1 agreement under the 1e-3 gate (ADR 0014) |
| Behaviour (paired greedy) | `plans/S0-behavior-b16-v2/stage0.evaluation.json` → `runs/stage0/S0-behavior-b16-v2/paired-evaluation/` | `results/responses.jsonl` (every response), `results/results.json`, `results/summary.csv` |
| Internals (fixed-context KL, writes) | `plans/S0-batch16-fp64-fixed-context-v3/stage0.drift.json` → `runs/stage0/S0-batch16-fp64-fixed-context-v3/fixed-context/` | `results/results.json`, `results/per-document.jsonl` (FP64 head KL, effective writes; ADR 0013/0015) |
| Paired findings | `plans/S0-batch16-primary-findings-v3/stage0.findings.json` → `runs/stage0/S0-batch16-primary-findings-v3/findings/` | `report.html`, `results/results.json`, `behaviour.csv`, `kl.csv`, `length.csv`, `writes.csv`, `paired-items.jsonl`, `review-evidence.jsonl`, `plots/*.svg` |

Headline accuracy (from the behaviour run): V1 69.3%, V2 72.7%, V0a 66.0%,
V0b 76.0%, V5 74.0%, controls 66.0–67.3%; no length restoration. Take exact values
and intervals from `findings/results/results.json`, not from this line.
Open item: the intended-answer review freeze (CPU sections 21–22) may not have run;
check `findings` status before calling review-based numbers final.

## E0.6a: dataset distance (complete)

Question: how far is each training dataset from the untuned model's own
distribution, per token?

Plan `plans/S0-matched128-distance-v3/stage0.distance.json`;
run `runs/stage0/S0-matched128-distance-v3/e06a_distance/`.

| File | Contents |
|---|---|
| `units/<answer_id>/` | Per-token surprise, entropy, rank for each of 1,152 unique answers (sealed) |
| `results/results.json` | Summaries per dataset, position windows, own_all gate, top gold tokens, limitations |
| `results/summary.csv`, `summary.parquet` | Per-dataset NLL, entropy, excess surprise, CIs |
| `results/per-problem.csv`, `position-windows.csv`, `top-gold-tokens.csv` | Per-problem values; first32/middle/last32; 50 most surprising gold tokens |
| `bootstrap/*.npz` | Saved whole-problem bootstrap indices (10,000, seed 42) |
| `report.html` | Readable report |

Headline (token-weighted excess surprise = NLL − entropy, nats/token, 95% CI):
gold +0.173 (0.153–0.195); own_all +0.001 (−0.003–0.004, gate passed at 0.02);
own_correct −0.007; own_correct_one −0.015. Gold's excess sits in the first 32
tokens (+0.48) and last 32 (+0.11); middle +0.05. Limitation: gold covers 128
questions, own_correct sets only the 120 with a correct sample.
`distance-verify` may not have been run yet; run it before final publication.

## E0.5: initial (step-0) pushes (complete; endpoint part exploratory)

Question: at B = 0, which direction does each objective push the rank-1 adapter,
and does that predict the trained endpoint?

Plan `plans/S0-matched128-pushes-v3/stage0.pushes.json`;
run `runs/stage0/S0-matched128-pushes-v3/e05_pushes/`.

| File | Contents |
|---|---|
| `units/<batch>/` | 46 sealed batches (15 gold, 15 own_correct_one, 16 GRPO): `gradient_b_<i>` per module, FP64 reconstruction and bound arrays, `metadata.json` proof |
| `results/summary.json`, `modules.jsonl`, `summary.csv` | Per objective × module: consistency (+ 1/√N noise reference), π_out vs chance band, norms, partial/full-batch variants |
| `results/directions.jsonl` | Cosine between objectives' mean pushes per module |
| `results/mean-pushes.npz` | Mean push vectors (mean, full-batch, token-weighted) |
| `report.html` | Pre-reveal summary |
| `predictions.json` | Declaration; `prediction_status: exploratory` |
| `results/endpoints.json`, `endpoint-alignments.jsonl`, `endpoints.csv`, `endpoints.html` | Cosine(push, trained B at step 64) per arm × objective × module; sign cosine; 100 Gaussian controls; orientation/flip |
| `exposure.json`, `complete.json` | Reveal receipt and final seal |

Numerical check: FP32 vs FP64 gradient relative error ≤ 1.9e-6 in every module;
worst error/allowance 2.7e-5.

Headline, step 0: median π_out gold 0.085, own_correct_one 0.086, GRPO 0.076
(chance ≈ 0.32). Consistency gold 0.87, own 0.26, GRPO 0.28 (noise ≈ 0.25–0.26).
Cross-objective median cosine: gold vs own −0.05 to −0.21; gold vs GRPO −0.03 to
−0.20; own vs GRPO +0.14 to +0.38 (partly shared samples).

Headline, endpoint (exploratory; median over 196 modules): SFT ← gold +0.62
(196/196 beyond controls); GRPO ← GRPO +0.20 (169/196); GRPO ← gold −0.10;
SFT ← GRPO −0.05; SFT ← own −0.04; GRPO ← own +0.03. No flipped modules. v_proj and
o_proj align most. Caveat: own and GRPO mean pushes are noisy (consistency at noise
level), which caps their cosines; one training seed per arm.

## E0.2: trained-adapter projections π_out/π_in (revealed, exploratory)

Review `notebook/overlap-review-exploratory.json`; report folder
`reports/S0-principal-overlap-v1/`: `results/summary.json` plus per-module
tables (3,528 rows, 196 modules, arms SFT/GRPO/random, 100 Gaussian draws),
`plots/principal-overlap.png|svg`, `exposure.json`, `complete.json`.
Limits stated in the summary: one training seed per arm; the random adapter is one
reference, not a null distribution.

## E0.3: rotation and intruders (measured; inspect via CPU section 25)

Plan `plans/S0-intruder-geometry-v1/stage0.intruders.json`;
run `runs/stage0/S0-intruder-geometry-v1/intruders/`: `report.html`,
`results/results.json`, `geometry.csv`, `candidates.csv`,
`plots/rotation.*`, `plots/overlap.*`, `units/<module,arm>/geometry.npz`.
"Undecidable" differs from "no intruders". Confirm the seal before quoting.

## Not run

- R0 (512-question untuned rollouts): dropped as too expensive (ADR 0016).
  Folders named `S0-untuned-rollouts-v1`, `S0-answer-distance-v1` and
  `S0-initial-pushes-v1` belong to that unrun R0 route.
- E0.1 checkpoint trajectory (CPU 40–42) and base-input energy (CPU 37–39): not run.
- Matched `-v1`/`-v2` distance and push plans: superseded before measurement.

## Decision records

ADR 0013 (FP64 head KL), 0014 (batch gate), 0015 (coefficient bound), 0016
(matched-128 design), 0017 on this branch (no cumulative unit cap for
E0.6a/E0.5), spec `.scratch/stage0-geometry/spec.md`, run guide
`docs/operations/stage0-matched128.md`.
