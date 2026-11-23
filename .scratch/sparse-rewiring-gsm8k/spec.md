# Sparse gold-SFT rewiring pilot — draft implementation spec

**Status:** needs-triage. Draft v1 (2026-11-22) from `design.md` and `decision-log.md`. Design and drafting only: no code, GPU run, threshold or budget is approved by this file. **[user]** = researcher decision; **[proposed]** = awaiting decision; **BLOCKED(Dn)** = cannot be implemented or frozen until decision Dn in `decision-log.md` is made.

Four stages are kept distinct throughout: **CPU implementation** (tests on tiny fixtures), **GPU qualification** (real model, no scientific run), **training/evaluation execution** (a launch record approved by the researcher), and **scientific interpretation** (report). Passing one says nothing about the next.

## Problem

Every existing arm adapts dense rank-1 LoRA modules, and the neuron-level evidence (C attribution, production-1, joint audit) describes how a `down_proj` adapter reads neuron activations. No arm selects which MLP neurons need their **inputs** changed. Max Loeffler's sparse rewiring does this with a per-neuron hard-concrete gate and an L0 penalty, leaving every write frozen. A gold-SFT version on the existing 512 examples gives (a) a new baseline arm and (b) a sparse, interpretable neuron set to compare with the GRPO attribution rankings.

## Solution

Two new arms, `sparse-gold-sft-r1` and `sparse-gold-sft-r0` (catalog names **[proposed]**), identical except for rank and read LR **[user 2026-11-23]**: the untuned model with a `MaskedReRead`-style module on every MLP, trained on gold's frozen training items with gold's loss contract and runtime plus Max's L0 objective, then hardened to a binary survivor set. Saved factors, masks and full recovery state. Greedy evaluation, a rank-1 vs rank-0 comparison, and a CPU comparison of each with `c-production-1`.

## User stories

1. As a researcher, I want the new arm trained on gold's exact frozen token IDs and labels so data and format cannot explain differences from gold SFT.
2. As a researcher, I want the first 64 steps to consume gold's exact batches so the early trajectory has the same exposures.
3. As a researcher, I want only the new read factors and mask logits to train, and every base weight hash unchanged.
4. As a researcher, I want the zero-coefficient model to equal the untuned model so checkpoint 0 means "no change".
5. As a researcher, I want expected-active count, nonzero deterministic count and binary survivor count recorded separately, with the hardening rule and the loss just before and after it.
6. As a researcher, I want selected neurons exported as zero-based `(layer, neuron)` with their coefficients so they join exactly with attribution files.
7. As a researcher, I want hyperparameter or run decisions made on a validation set that is disjoint from every training and protected set.
8. As a researcher, I want behaviour measured under the established cohort150 protocol, labelled exploratory.
9. As a researcher, I want overlap with C attribution reported on the measured universe with a layer-matched null.
10. As an operator, I want resumable checkpoints with optimizer, scheduler, RNG and data cursor so an interruption does not repeat or skip updates.
11. As a maintainer, I want CPU tests with no model download, and the method reimplemented under this repository's license.

## Implementation decisions

### Data

- Read gold's frozen plan (`plans/pilot2-sft-fp32-l4-batch8-v1/sft.config.json`, `training.items.json`, `evaluation.items.json`) through `sft.load_sft` and its hash checks; never re-tokenize. Read `runs/…/pilot2-sft-fp32-l4-batch8-v1/inputs/order.json` for steps 1–64. **BLOCKED(ticket 01)** until those files are located and hashed.
- Order for steps 65+: **[proposed]** D7. The full step → example-ID list is frozen in the plan before training.
- Validation set: **[proposed]** D11, **BLOCKED(D11, ticket 01)**. Built by a seeded, exclusion-list sampler; exclusion sources and their hashes recorded; an unreadable source fails, never "no overlap".

### Module (clean-room, D16)

- `pilot_eval/rewire_module.py` **[proposed name]**: wraps the pinned Qwen2 MLP and reproduces its forward exactly (`down(act(gate(x)) * up(x))`) plus `Δgate = z ⊙ ((x·a_g) b_g)` and `Δup = z ⊙ ((x·a_u) b_u)` for rank 1; `(x Aᵀ) Bᵀ` for rank k; `x Wᵀ` for rank 0. Read changes at every unpadded position (prompt included). Parts gate+up **[user]**.
- Init: `a ~ N(0,1)/√d` from a CPU generator seeded per layer (D10); `b = 0`; `logα = 1.0`.
- Hard concrete with β 2/3, γ −0.1, ζ 1.1; training noise: one uniform `u` per layer per group of 4 consecutive examples, drawn from a dedicated generator, cached, and reused by each single-example pass and by checkpoint recompute; `z` is recomputed from the cached `u` and the current `logα` (D14, [proposed]); deterministic gate `clip(σ(logα)(ζ−γ)+γ, 0, 1)`; expected-active `Σ σ(logα − β ln(−γ/ζ))`; freeze rule `z_det > 0` → binary. Ranks: 1 and 0 are both run **[user 2026-11-23]**; the module also supports k > 1. Rank 0 adds 770.7 M FP32 read parameters, so the memory test in ticket 07 covers it explicitly.
- An `off` switch returning the exact base MLP, and a per-neuron "remove new reads" intervention for later causal checks.

### Training

- Explicit loop adapted from `sampling_sft_training_backend.train` (not TRL), reusing `sft_backend.completion_loss`, `CompletionCollator` and `frozen_weight_hash`. Objective per step: Σ over microbatches of `completion_loss(…, denominator = step's supervised tokens)` + `λ · expected_active` (once per step).
- Optimizer: one AdamW, `foreach=False`, two groups. Values **BLOCKED(D1, D2, D3)**; proposed reads lr 3e-3 (rank 1) / 3e-4 (rank 0), masks lr 0.1, betas (0.9, 0.99), eps 1e-8, wd 0. Clip 1.0 on read factors only (D4). `LambdaLR` shared by both groups; schedule **BLOCKED(D5)**, proposed 20 warmup + cosine to 512, fixed masks for the last 64.
- λ **BLOCKED(D6)**, proposed 6e-4.
- Runtime (D8): FP32, `set_ieee_precision()`, eager attention, `use_cache=False`, gradient checkpointing (D15), full determinism, pinned packages from `sft_backend.PINS`.
- Logged every step (`logs/steps.jsonl`): CE, L0 term, total, LR per group, read-factor grad norm before clipping, mask grad norm, examples seen, and the gate and read-change summaries listed under §Training dynamics.
- Every 8 steps **[proposed, D20]** (`logs/validation.jsonl`): validation CE with deterministic masks, plus a per-layer gate snapshot (`masks/trace/<step>.npz`: `logα` as FP16 for all 250,880 neurons, about 0.5 MB each).

### Training dynamics (figure requested by the researcher, 2026-11-23)

One figure with step on a shared x axis, built on CPU from the saved logs (ticket 12). Its purpose is descriptive: it shows how gate shutting, read-change size and performance move together within one run. Co-movement is not evidence that one causes the other.

| Panel | Quantities | Source and cadence |
|---|---|---|
| 1. Gates | expected-active `Σ P(z>0)`, nonzero deterministic count (`z_det > 0`), and binary survivors after the freeze, on a log y axis; a per-layer heatmap (layer × step) of nonzero-`z_det` counts | every step for the totals, every 8 steps for the per-layer counts |
| 2. Read-change size | the gauge-invariant per-neuron row change `‖Δw_i‖ = z_i ‖b_i‖ ‖a‖` (rank 1; `‖b_i A‖` for rank k; `‖w_i‖` for rank 0), summed and as a median, for gate and up separately; once over all neurons and once over the currently nonzero (survivor) set | every step (totals), every 8 steps (survivor subsets) |
| 3. Performance | validation CE (deterministic masks, dense); greedy flexible-v3 accuracy on validation at the checkpoints; cohort150 greedy only at 64 and 512 (D12); reference lines for untuned (step 0) and gold-sft step 64 evaluated on the same validation set | CE every 8 steps; accuracy at checkpoints |
| 4. Events | vertical line at the freeze step (449); mask churn between consecutive checkpoints as the Jaccard overlap of their nonzero sets, plus counts that switched on and off | checkpoints |

- Raw `‖a‖` or `‖b‖` alone is not reported as a size: rescaling `a` by k and `b` by 1/k leaves the change the same.
- The gold reference lines need gold-sft checkpoints 0 and 64 evaluated on the new validation set (CE plus one greedy pass). That is a new, cheap use of existing adapters and is registered in the catalog.
- Every panel shows the real evaluation points as markers; nothing is interpolated between checkpoints.
- Checkpoints **[proposed]** 0/8/16/32/64/128/256/448/512, each with unmerged factors, all `logα`, optimizer, scheduler, both RNG streams, data cursor and seal; 448 and 512 bracket the hardening. Resume only from a complete sealed checkpoint with matching plan and runtime hashes.

### Outputs

```text
artifacts/runs/sparse-rewiring/<run-id>/        one run ID per arm (r1, r0); both share one plan except rank and read LR
  config.json  runtime.json  status.json  logs/steps.jsonl  logs/validation.jsonl
  inputs/order.json  inputs/validation.items.json (+ exclusion record)
  checkpoints/checkpoint-<k>/  rewire_state.safetensors  optimizer.pt  scheduler.pt  rng.pt  complete.json
  masks/<k>.json             expected / nonzero / binary counts per layer; survivor list
  masks/trace/<step>.npz     logα snapshot every 8 steps (FP16), for the per-layer heatmap and churn
  dynamics/                  figure (PNG + SVG), its input table (CSV) and a provenance JSON
  hardening.json             rule, pre/post counts, validation CE before and after
  eval/<k>/greedy/           responses.jsonl  config.json  scores.json
  compare/                   overlap tables, null draws, figures (CPU)
```

### Evaluation

- Validation: gold CE (deterministic masks) and greedy flexible-v3 at checkpoints, on the D11 set.
- Cohort150 greedy at 64 and 512 **BLOCKED(D12)**: existing harness, cap 1,024, EOS 151645, explicit repetition penalty 1.0, flexible-v3; paired per-question comparison with the stored untuned, gold-sft, online- and offline-grpo runs, labelled with runtime differences. Every run writes its config and all responses (AGENTS.md).

### Comparison

- D21 (rank 1 vs rank 0, CPU): survivor counts; Jaccard overlap of the final binary sets and overlap per layer, with a layer-matched null; neurons unique to each arm; each arm's and the intersection's overlap with the C sets; validation CE and accuracy per checkpoint; cohort150 greedy at 64 and 512, paired by question; CE change across hardening; the dynamics figure side by side on shared axes.

- D19. Inputs: survivor set at 512 (and 448 continuous gates as a secondary), `c-production-1` `aggregate/<arm>/neurons.npz` and `freeze/<arm>.json` read in place (never modified). Joins on zero-based `(layer, neuron)`; the loader asserts the C arrays are [28, 8960] and the index convention from the C freeze.

### Registration

- New arm, checkpoints, response sets and cohort prior uses in `catalog/` in the same change as the run records; `catalog verify` must pass on local entries.

## Testing decisions (CPU, tiny Qwen2, no download)

1. Off-switch and zero-coefficient forward equal the base MLP bit-for-bit in FP32; logits equal across a whole tiny model.
2. Shapes and zero-based indexing: perturbing `b[j]` in layer l changes only neuron j's gate pre-activation in layer l.
3. Only `a`, `b`, `logα` require grad; base hash unchanged after a step (mutation: unfreezing a base weight fails).
4. Gradients: at init `a` gets zero gradient (b = 0) and `b` gets nonzero; `logα` gets only the L0 gradient; after one step all three are nonzero (mutation: detaching `z` fails).
5. Hard-concrete equations against hand-computed values; expected-active vs Monte-Carlo frequency within a declared sampling tolerance; freeze maps `logα > −ln 11` to 1.
6. Three counts and the export agree with an independent recomputation from saved `logα`.
7. Step loss equals an independent per-token CE sum over the step's supervised tokens with variable lengths (mutation: per-example mean fails).
8. First 64 steps consume gold's order; later orders match the frozen plan.
9. Clipping touches reads only; mask LR follows the shared schedule.
10. Checkpointed vs non-checkpointed backward identical for a fixed noise seed.
10a. Shared-mask equivalence (D14): the summed read and `logα` gradients of 4 single-example passes with one cached `u` equal one 4-row forward with that `u`, within FP32 round-off; a mutation that re-draws `u` per pass fails; the logged group boundaries match the step order.
10b. Rank 0 and rank 1 share one code path apart from the parameterisation: a rank-0 fixture with its rows set to `b_i a` reproduces a rank-1 fixture's forward and gradients with respect to the mask exactly.
11. Uninterrupted vs resumed (at 8 and at a fixed-mask step) give identical factors, masks and logs.
12. Validation sampler refuses an unreadable exclusion source and never returns an excluded ID.
13. Comparison: layer-matched null preserves per-layer sizes; overlap on a planted fixture; missing attribution values excluded, not zero.
14. Dynamics: the logged expected-active, nonzero and survivor counts equal a recomputation from the saved `logα` snapshots; the read-change norm is unchanged under `(a, b) → (k a, b / k)`; churn on a planted pair of mask sets; the figure builder refuses a log with missing steps instead of interpolating.

GPU qualification (ticket 07) adds: real longest example forward/backward and peak memory, zero-update logits identity on the real model, a disposable few-step run (not the scientific run), throughput, save/restore.

## Out of scope

λ sweeps, rank 4 runs, second seeds, per-example mask ablation, objective comparisons, sampled evaluation, causal removal experiments, any change to existing arms, the C-attribution records or ticket 14's held-out pool.

## Open decisions blocking a launch record

D1, D5, D6 (criterion), D11 (needs `sequences-v2` IDs), D12, D18, D20, D21 (D13 decided: both ranks). See `decision-log.md`.
