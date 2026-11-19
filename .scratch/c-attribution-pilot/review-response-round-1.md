# Response to review round 1

This responds to [review-round-1.md](review-round-1.md) (reviewed commit `1c57259`). The spec is now draft v2 and the tickets are revised. No historical study artifact was changed. No GPU run was started.

Each item is marked as one of:

- **Accepted**: a user decision carried into the spec.
- **Fixed**: the spec or a ticket now says what the review asked.
- **Residual risk**: known and reported, not removed.
- **Open**: needs the user's decision; listed in spec v2 under Open decisions.

## Changes made in this round

| Change | Commit |
|---|---|
| Fast-forwarded to `5b43af1` | — |
| Merged local `main` (catalog commits `6444345`, `f3ebb9d`, `271acf3`, `6f6e2f0`) into this branch, so the catalog is no longer only on the Mac's `main`. The only conflict was in `pyproject.toml`; both extras are kept. `pytest tests/test_catalog.py`: 24 passed. `python -m pilot_eval.catalog verify --root local=<artifacts>`: 51 pass, 4 not-local, OK. `main` itself is not pushed. | `a43dca0` |
| Rewrote `spec.md` as v2 and revised tickets 01–14 | next commit |

## New facts found while responding

These were checked on 2026-11-18 against the local imports.

### Source-run provenance

Both behaviour configs record the following:

- base and tokenizer revision `989aa798…`;
- chat template sha256 `2e2d2512…`;
- step-64 adapter and seal hashes;
- runtime: FP32, A100-SXM4-80GB, torch 2.9.0, transformers 4.57.6, PEFT 0.18.0;
- the sampled policy: temperature 1.0, top_k 0, top_p 1.0, `num_beams` 1, cap 1,024, 8 draws, batch 4;
- seed rule: 42 + batch index;
- `implementation.engine` = `e33ea870…`, which is byte-identical to `pilot_eval/grpo_eval_engine.py` today.

**Not recorded:**

- `repetition_penalty` and `no_repeat_ngram_size`. They are set explicitly to 1.0 and 0 in `grpo_gpu._generate`. That file is not hashed by the runs, but git shows no change since 2026-10-09, before both runs, so the values are **inferred**.
- Per-token generation log-probs.

The resolved logits-processor list was not verified. Ticket 05 adds a CPU check against the pinned `generation_config.json`.

### Hot layers

No documented hot-layer list exists in the repo or the local artifacts. A search for "hot layer" finds only the review itself.

The per-layer, per-module effective writes are in `artifacts/analysis/results-hub-v1/data/writes_modules.json` (sha256 `35747fda…`). Their `down_proj` ranking depends on the context:

- top 5 on `gsm8k_solution`: online 15, 16, 11, 10, 24; offline 16, 15, 24, 12, 13;
- on `fineweb`, online's top layer is 11 and offline's is 24.

Per review decision 5, this is surfaced as open decision 1. No list was chosen.

### Production-1 numerics

`qualification.json` records:

- finite differences passed on gold-sft and online-grpo;
- batch invariance passed on all six arms;
- measured batch-invariance discrepancy 2.2e-6;
- resolution floor rule: 10× that.

Production-1's identity tolerance already uses the gross denominator Σ_j |attr_j|, which is robust to cancellation. Spec v2 keeps it.

## User decisions 1–11

| # | Disposition | Where |
|---|---|---|
| 1. Reuse step-64 sampled responses; all draws and outcomes kept; conditional on provenance | **Accepted.** The conditions are now concrete: the provenance table with recorded, inferred and unverified items, the processor check, and the rule that a mismatch is material. | spec: Data and provenance; tickets 02, 05, 08 |
| 2. Greedy only, full matched 150, exploratory; greedy ≠ sampled J; follow-up separate | **Accepted.** Sampled mode is removed from this launch and from the rehearsal. Three standing labels are added. | spec: How to read the results, Generation; tickets 07, 11, 12, 14 |
| 3. Top-10 adapters + matched control; no top-10 neuron ablations | **Accepted.** | spec: Selection; ticket 06 |
| 4. Three top-64 read groups, each with its own size- and layer-matched control; LoRA read terms only | **Accepted.** `global64`, `hot64` (64 × \|H\|) and `layer64` (up to 1,792). Matching is same-layer, so the layer distribution matches by construction. | spec: Selection; ticket 06 |
| 5. One hot-layer list for both arms, from the prior effective-write study, bound before the new rankings | **Accepted as a rule. Open on content**: no documented list was found, so a candidate source is presented (open decision 1). | spec: Selection, Open decisions 1 |
| 6. Controls: adapters same module ±20% write; reads same layer ±20% \|a_j\|·mean\|z_j\|; no scaling; measurement specified; group-wise matching | **Accepted.** Measurement: the arm's own 1,200 rows, all unpadded positions, token-weighted, accumulated in the attribution pass. Exclusion: every same-kind target of the arm. Matching: seeded maximum-cardinality bipartite matching per stratum. | spec: Selection; ticket 06 |
| 7. Seeded 75/75 question split; draws kept together; adapter, global, hot and per-layer stability; no gate; full 150 for selection | **Accepted.** One split, shared by both arms. The v1 draw split is dropped (open decision 7). | spec: Uncertainty; ticket 04 |
| 8. Adapter bootstrap by question groups; benchmark the chunked neuron bootstrap; prioritize; exploratory intervals; no threshold | **Accepted.** Per-question pre-aggregation S_q makes this exact and cheap: a question resample keeps groups intact, so recomputing the leave-one-out weights is unnecessary. | spec: Estimator, Uncertainty; ticket 04 |
| 9. Trace FP32/FP64 lessons; summed-metric scale and cancellation; no silent loosening | **Accepted.** The numerics table distinguishes reproduced, measured and open items. Tolerances are set from rehearsal measurements; the historical ones are unchanged. | spec: Numerical precision; tickets 05, 11 |
| 10. Per-condition report list, untuned baseline, group predicted effect | **Accepted.** Untuned is regenerated in the qualified runtime (proposed). | spec: Report per condition; ticket 13 |
| 11. Helpful/harmful language is local; never label non-positive as helpful; exact-64 shortfall surfaced and settled before freeze | **Accepted.** Selection is positive only. The shortfall handling is open decision 2. | spec: How to read the results, Selection |

## Findings F1–F7

### F1: Different attribution and behavioural targets — Fixed

The Questions section now separates the sampled-J ranking (questions 1–3) from the exploratory greedy check (question 4). A "How to read the results" section states that:

- greedy neither proves nor disproves the gradient;
- removal is finite and its effects can interact;
- the summed local prediction is only a reference.

Every ablation table carries the three labels. The group's and the control's predicted removal effects are reported under their own label.

Sampled interventions and new cohorts are deferred, and the direct sampled-J test is named as what they would provide.

**Residual risk:** the pilot cannot validate the sampled-J ranking behaviourally. That is by design.

### F2: Neuron ranking stability and 64-read scope — Fixed

Stability is reported for adapters, global-64, hot-layer and per-layer reads, with matched question IDs across arms and the layer distributions and signs of the selected reads.

Each scope has its own matched control, and the report compares each group only with its own control.

The concentration denominator is defined:

- shares use gross Σ|dJdλ|;
- net |Σ dJdλ| is reported separately and never used as a denominator.

Neuron intervals are labelled exploratory and pointwise.

### F3: Matcher false-unavailable — Fixed in the design; not implemented

Ticket 06 now specifies seeded maximum-cardinality bipartite matching per stratum. A control is declared unavailable only when the maximum matching is smaller than the group, and the report then names the deficient targets.

The review's counterexample is a required regression test. It must fail against the existing `matched_controls` (red) before the new matcher is written.

Layer-by-layer feasibility for `layer64` is specified. One infeasible layer does not block the global or hot conditions. What happens to the `layer64` control in that case is open decision 3.

Production-1's frozen controls are not changed. The lesson entry is kept as written: no prior real-study failure is claimed.

**Residual risk:** none beyond implementation.

### F4: Sampler provenance and score fidelity — Fixed, with one residual risk

The provenance table binds:

- base, adapter and tokenizer;
- template;
- engine hash;
- the recorded policy;
- seeds and batch identity;
- EOS and cap semantics;
- runtime.

Two items stay inferred or unverified: the repetition penalty (inferred from an unchanged, unhashed file) and the resolved processors (to be checked by ticket 05 on CPU).

Qualification (a) now chooses short, long and capped rows, and reports per-token and per-sequence errors. The 1e-3 per-token figure is withdrawn as a tolerance. Spec v2 records that it allows up to 1.024 nats per sequence, as a bound, not an observation.

Replay stays report-only unless it points to a policy or checkpoint mismatch; a confirmed mismatch is material. A no-op check (f) and a fixed batch membership across conditions are added.

Tolerances are not treated as approved. Production-1's carry over unchanged, and the new cached-against-forced rule is set by the user after the rehearsal measures it (open decision 4).

**Residual risk:** with no generation-time log-probs, exact historical policy equivalence cannot be proven, only bound by the checks above.

### F5: Bootstrap, disk, CPU and total budget — Fixed

Storage is redesigned:

- units hold whole questions;
- neuron reads are pre-aggregated per question as S_q [150 × 250,880] FP64, about 300 MB per arm;
- no per-response neuron arrays are stored;
- the bootstrap streams over column chunks and keeps percentiles, not the 10,000 resamples.

The rehearsal benchmarks wall time and peak memory.

The volume is restated: 2,700 greedy answers, + 150 untuned, + 300 optional intact repeat. The v1 time estimate and caps are withdrawn. Caps are proposed only from rehearsal measurements. There is no automatic handover.

### F6: Comparative reporting and uncertainty — Fixed

The spec now requires:

- paired question-level intervals on ablated − intact and on target − control;
- the percentage remaining, with the denominator stated;
- gains and losses, invalid and capped counts, and lengths;
- group predicted effects;
- question-paired bootstrap across arms;
- cross-arm overlaps and layer and module distributions;
- the one-seed caveat.

### F7: Catalog, cohort and rehearsal provenance — Fixed

The catalog commits are merged into this branch (`a43dca0`) without pushing `main`. A clean checkout of the pushed branch will contain them. Ticket 08 seals the catalog revision in the input bundle and verifies on the Pod that it is present.

The rehearsal draws only from cohort150 and records its exact question IDs in the catalog's `prior_uses`.

Ticket 14 builds its exclusion list from every recorded prior use, not only cohort150 and "20 rehearsal items". It freezes the follow-up design before seeing any of its outcomes. It is not labelled confirmatory until prior exposure and the frozen plan have been checked.

## Ticket dispositions

| Ticket | Change |
|---|---|
| 01 | Summed metric with FP64 accumulation; per-question S_q output; old metrics bit-identical |
| 02 | Provenance status (recorded / inferred / unverified) in the row manifest; source config and engine hash bound |
| 03 | Per-question pre-aggregation; gross and net totals; gross-denominator identity |
| 04 | 75/75 question split across all four rankings; paired adapter bootstrap; chunked neuron bootstrap benchmark; no gate |
| 05 | Processor-resolution check; cached-against-forced per token and per sequence; numerics floors; no-op; replay report-only |
| 06 | Group-wise matcher with the counterexample regression; three read scopes; hot-layer provenance binding; shortfall surfaced; sizes measured in the attribution pass |
| 07 | Nine greedy conditions per arm; fixed batch membership; paired reporting; sampled mode removed |
| 08 | Bundle includes the catalog revision, the source configs and the hot-layer source file |
| 09 | Caps from rehearsal measurements; restore-hash safety; fixed batching |
| 10 | Real dependencies must execute (skips counted as failures for the real-path test); old-metric behaviour checked |
| 11 | Greedy only; short, long and capped rows; records rehearsal IDs; measures throughput, bootstrap and floors |
| 12 | 2,700 (+150, +300) greedy answers; measured budget; no handover; no sampled ablations |
| 13 | Predicted effects, paired deltas, control feasibility, cross-arm comparison, stability, labels |
| 14 | Exclusion from every recorded prior use; design frozen first; no automatic launch |

## Verification

The following was done in this round:

- the catalog tests and the catalog verify, both on this branch (results above);
- reading the source configs and hashing the engine file;
- reading the effective-write data.

The following was not done:

- the full project suite was not run in this round;
- no ticket code was written;
- no GPU work was done.

## Open decisions for round 2

These are listed in full in spec v2. In short:

1. Hot-layer source, context and k.
2. Positive-shortfall handling.
3. Infeasible `layer64` control layers.
4. The cached-against-forced tolerance, set after the rehearsal.
5. The intact repeat.
6. Caps.
7. Whether to drop the draw split.
8. Seeds.

## Update 2026-11-19 (user answers)

- 1 (hot layers): {11, 15, 16, 24} confirmed; derived by code with the union rule, with a k-sensitivity table.
- 3 (infeasible `layer64` control): run the target, report the control unavailable.
- 7 (draw split): kept in addition, with within-half leave-one-out weights.
- 8 (seeds): defaults confirmed.
- 2 (positive shortfall): freeze the smaller positive group with a size-matched control; never pad or drop.
- 5 (intact repeat): yes, +300 greedy answers.
- No round-1 decision remains open.
