# Sparse rewiring on GSM8K: decision log

**Status:** needs-triage. Every recommendation here is **[proposed]** unless marked **[user]**. Evidence tags as in `design.md` §0. Predictions about training dynamics are derived from code, not measured; they are labelled "expected".

## The minimal pilot in one paragraph

Train two sparse-rewiring arms, rank 1 and rank 0 (full per-neuron rows) **[user 2026-11-23]**, from the untuned Qwen2.5-1.5B-Instruct on the 512 gold-SFT examples, with gold's token/label/loss contract, FP32 IEEE runtime and microbatching, and Max's hard-concrete L0 objective, mask optimizer and fixed-mask finish. λ 6e-4 for both, read LR 3e-3 (rank 1) and 3e-4 (rank 0), mask LR 0.1, one mask draw per 4 examples, 512-step cap whose first 64 steps are gold's exact batches, last 64 steps fixed-mask. Monitor a new train-split validation set only. Evaluate the step-64 and final checkpoints greedily on cohort150 as exploratory. Compare each arm's binary survivor set with the other's and with `c-production-1`'s per-read J scores on CPU, and plot both arms' training dynamics side by side. One seed per arm, no sweep.

## Why not just reuse every gold setting

Gold trains 1.15 M LoRA parameters (r1 on 196 modules) with no selection mechanism. Rewiring at rank 1 trains 0.59 M read parameters plus 250,880 mask logits, and its outcome (which neurons survive) is produced by a race between the L0 pressure and the CE gradient on each mask. Settings that shape that race have no gold counterpart. The table marks which settings are carried over and why.

| Setting | Gold SFT | Max (published case) | Recommended | Reason |
|---|---|---|---|---|
| Training examples, prompt, labels | §2.2 | own chat data, trailing newline supervised, 640 truncation | **gold** | Comparability; Max's newline label and truncation are incidental, not method-defining |
| Loss reduction | token-weighted over the step | token-weighted over the step | **same** | Already equal |
| Precision | FP32, TF32 off | BF16 frozen, FP32 reads | **gold FP32 + IEEE** | Repository rule; FP32 makes Max's casts no-ops |
| Microbatch / mask noise | 1 × 8 | 4 × 2 (one mask per 4 rows) | **gold 1 × 8, one mask per 4 examples** | Gold loss path, Max's noise level (D14) |
| Read LR | 1e-4 (LoRA) | 3e-3 (rank 4) | **3e-3** | D1 |
| Mask optimizer | — | AdamW lr 0.1, same schedule | **Max** | D2 |
| Betas | (0.9, 0.999) | (0.9, 0.99) | **Max** | D3 |
| Schedule | constant, 64 | 20 warmup + cosine, 1,400, last 100 fixed | **Max shape, 512 cap, last 64 fixed** | D5 |
| λ | — | 6e-4 | **6e-4 reference** | D6 |
| Checkpoints | 0/8/16/32/64 | every 50 | **gold's + 128/256/448/512** | D5 |

## Decisions

### D1. Read-update learning rate

| Option | For | Against |
|---|---|---|
| A. 1e-4 (gold) | Same number as every existing arm | Not the same parameters, so not the same optimisation. Expected to be degenerate with Max's mask LR: at init `b = 0`, so masks receive only the L0 gradient and Adam moves each `logα` by about the mask LR per step; from `logα = 1` the `z_det > 0` boundary (`logα = −ln 11`) is about 34 full-rate steps away. In that window `b` moves at most about 1e-4 per step, so the CE pull on masks stays small and near-total pruning is expected. Consistent with Max's log (expected-on 404,637 at step 10 → 3,900 at step 150 at lr 3e-3), but untested at 1e-4 |
| **B. 3e-3 (Max, rank 4)** | The value the mask LR and λ were tuned with; the only published operating point | Rank-1 and rank-4 optima can differ; no rank-1 result exists |
| C. 1e-4 and scale the mask LR down by 30 | Keeps gold's number | Two departures from Max instead of one; untested |

**Recommend B for rank 1**, with A as the single sensitivity run if budget allows.

**Rank 0: recommend 3e-4 [proposed].** `RANK0.md` reports that at rank 4's 3e-3, rank 0 overfit (over 1,000 neurons, run `v2_reread_r0_l06e-4_lr3e-3_s*`). At 3e-4 to 5e-4 with λ 6e-4 to 8e-4 it gave 82 to 122 neurons, mostly the same neurons as rank 4. Max's explanation: Adam moves every element of a full row at the LR, so full rows need a lower rate. The rank-1 and rank-0 arms therefore differ in LR as well as rank; the comparison (D21) states this.

### D2. Mask optimizer and LR

Recommend Max's exactly: a second AdamW parameter group, lr 0.1, weight decay 0, same `LambdaLR` multiplier as the reads (warmup + cosine apply to masks too), no gradient clipping on masks, `logα` init 1.0. Gold has no counterpart, so nothing is inherited from it. Record the resolved parameter groups in the run config.

### D3. Adam betas

Max uses (0.9, 0.99) for both groups (`--beta2`); gold uses (0.9, 0.999). Recommend Max's for both groups, because the second-moment window sets how fast mask steps shrink once the CE gradient appears. Record eps 1e-8 and `foreach=False` (gold) explicitly. Alternative: 0.999 for reads only (closer to gold, mixed recipe). Not recommended.

### D4. Gradient clipping

Both clip at 1.0. Max clips **read factors only**; masks are unclipped. Recommend the same, and log the read-factor norm before clipping and the unclipped mask-gradient norm separately.

### D5. Steps, schedule, freeze and checkpoints

| Option | Steps (epochs) | Fixed-mask phase | Comment |
|---|---|---|---|
| A. Gold-matched | 64 (1) | last 8 | Expected mid-pruning at step 64 (D1 arithmetic), so the survivor set would describe an unconverged race |
| **B. Proposed** | 512 (8) | last 64 | Max's case was 5.4 epochs (11,200 rows / 2,081); 8 epochs here is ~4,096 presentations |
| C. Max-matched steps | 1,400 (22) | last 100 | Heavy repetition of 512 examples; overfitting risk; 2.7× the GPU time |

Recommend B with 20-step warmup and cosine to 0 at step 512 over both groups, fixed-mask from step 449 to 512. Checkpoints: 0, 8, 16, 32, 64 (gold's), 128, 256, 448 (last continuous-mask step) and 512 (final binary). Step 64 is a reference point under a different schedule, not gold's trajectory and not equal token exposure. If validation shows the run is still moving at 448, report it; do not extend after the fact under the same name.

### D6. λ

6e-4 as the reference point only. Our model has 250,880 candidates vs 442,368 in Qwen3-8B; λ is per neuron, so equal λ is not equal total pressure. If the first run is degenerate (for example zero survivors, or validation CE no better than untuned), stop and propose one bounded validation-only adjustment (3e-4 or 1.2e-3), never selected on cohort150. **OPEN:** what counts as "acceptable" (validation CE gain relative to gold-step-64 on the same validation set? a survivor-count range?). Needs a researcher criterion before any calibration run.

### D7. Data order beyond one epoch

Recommend: steps 1–64 consume gold's saved `inputs/order.json` exactly; epoch e ≥ 2 uses the 512 IDs reshuffled with `random.Random(42 + e − 1)`; record the full 512-step order before training. Alternative: repeat gold's order every epoch (identical batches, but a periodic pattern). Max reshuffles each epoch.

### D8. Precision and runtime

FP32 weights, FP32 read factors and masks, no autocast, eager attention, `set_ieee_precision()` (IEEE for CUDA matmul and cuDNN, `highest`), CUBLAS workspace config, full determinism. A BF16 variant would be a separately named, separately qualified run and is not proposed.

### D9. Target and truncation contract

Gold's: labels from the frozen `training.items.json`, `<|im_end|>` supervised, trailing newline not supervised, no truncation. Max's `"<|im_end|>\n"` target and `MAX_LEN = 640` are recorded as differences not adopted.

### D10. Seeds

Separate and record: data order (gold's saved order, then D7), read-direction init (`seed + layer`, base 42, CPU `torch.Generator`, as in Max), mask noise (CUDA generator, seed 1042 proposed). A shared integer is not matched randomness with any other arm. A second seed (all three changed) is the first follow-up.

### D11. Validation set

No suitable set exists. Proposed: 128 GSM8K **train** questions, `random.Random(20261122)` sample from train indices excluding (a) `gsm8k-train-512`, (b) every `sequences-v2` calibration, discovery and held-out question, (c) any other train ID in a run record or the catalog. Train-split, because C ticket 14 plans to draw a held-out set from the **test** split minus prior uses, and this pilot should not consume it. Measures: gold-answer CE with deterministic masks (Max's selection quantity) every 32 steps, and greedy flexible-v3 accuracy at checkpoints. **Blocked** until the `sequences-v2` ID lists are read from the artifact root (not in Git). Size 128 is a proposal.

### D12. Cohort150 use

The cohort is not held out (C production selected on it today). A new greedy evaluation of the sparse arm there is comparable with every other arm and is a further prior use; label results exploratory and never tune on them. Needs the researcher's OK; register in `catalog/cohorts.yaml` when run.

### D13. Rank — decided

**[user 2026-11-23]: run both rank 1 and rank 0 (full rows), with everything else identical** (gold data and loss contract, FP32, shared mask per 4 examples, schedule, checkpoints, evaluations, dynamics figure). Rank 4, the published but self-described oversight, is not run.

| Arm | Status in Max's work | Trainable read parameters | Connection here |
|---|---|---|---|
| Rank 1 (`sparse-gold-sft-r1`) | Configured, **no released result** | 587,776 (28 layers × 2 parts × (1,536 + 8,960)) | Closest in shape to the r1 LoRA arms, but not equivalent |
| Rank 0 (`sparse-gold-sft-r0`) | **Intended method** (`RANK0.md`) | 770,703,360 (28 × 2 × 8,960 × 1,536) | Each neuron learns its own new gate and up rows |

Both arms have 250,880 mask logits. The arm names are proposed.

### D14. Microbatch and mask noise

**Revised 2026-11-23 after the researcher's question. Still [proposed].**

The principle is to split the recipe by what each part controls:
- The **data and loss path** (which tokens, which labels, how the loss is reduced, one example per forward) defines comparability with gold SFT. It stays gold's.
- The **stochastic selection mechanism** (mask noise level, λ, mask LR) defines the method. It stays Max's, because λ 6e-4 and mask LR 0.1 were tuned under his noise level.

In Max's trainer, one `z` per layer is drawn per forward call and shared by every row and position of that micro-batch (`train.py:301-314,366-376`; micro 4, batch 8). That gives **2 draws per step**.

| Option | Draws per step | Matches | Comment |
|---|---|---|---|
| A. Fresh mask per single-example pass | 8 | gold's microbatch path | Usually a less variable gate-gradient estimate, so the selection race is run at a noise level λ was not tuned for |
| **B. Microbatch 1, one mask per group of 4 consecutive examples** | 2 | gold's loss path and Max's noise level | Requires explicit caching (below) |
| C. Microbatch 4 with padding | 2 | Max literally | Changes gold's padding-free loss path; padded eager attention adds small numerical differences |

**Recommend B.** Both A and B estimate the same expected objective, because each example's mask has the same marginal distribution. They differ in gradient variability and so in trajectory. B is closer to Max's procedure. It is not inherently more principled or less noisy, and the choice can change which neurons survive. Option A is a later ablation, not part of the minimal pilot.

Implementation for B:
- At the start of each group of 4 (examples 1–4 and 5–8 of the step's gold order), sample one uniform `u` per layer from the mask-noise generator and cache it.
- In each of the 4 single-example passes, recompute `z = clip(σ((log u − log(1−u) + logα)/β)(ζ−γ)+γ, 0, 1)` from the cached `u` and the current `logα`, so gradients reach `logα` in every pass. `logα` does not change within a step, so all 4 passes see the same `z`.
- Gradient checkpointing recomputes with the same cached `u`. It does not re-draw.
- If a short step ever occurs, groups are `ceil(n/4)`, and the grouping is recorded. Gold steps always have 8 examples.
- Log the `u` seed and the group boundaries per step.

Test: on a tiny fixture, the summed gradients (reads and `logα`) of 4 single-example passes with a shared cached `u` equal those of one 4-row forward with the same `u`, within FP32 round-off. This is the same token-weighted loss, with a padding-free reference built row by row where needed.

### D15. Gradient checkpointing

Gold enables it and trains in train mode. Max calls it but keeps the model in eval mode, where HF decoder layers may skip checkpointing (code-inspected, not verified). If enabled, the recompute must replay the same mask noise (`torch.utils.checkpoint` default `preserve_rng_state=True`). Recommend on, with a CPU test that a checkpointed and an unchecked backward give the same gradients for a fixed noise seed.

### D16. Code reuse and license

Max's repository has no license; only the upstream Sauers trainer is Apache-2.0, and the rewiring arm may be Max's own addition. Recommend a clean-room implementation in `pilot_eval/`, tested against the published equations, citing `max@a6dc314`. Optionally ask Max for a license.

### D17. Code base for implementation

The catalog, IEEE helper, C outputs reader and neuron-audit helpers are on `c-attribution-pilot` only. Recommend implementation branches from `c-attribution-pilot@7667739` (or from `main` after it merges); this design PR is docs-only against `main`.

### D18. Hardware

Recommend **one A100-80GB for both arms** [proposed], so the rank comparison does not mix runtimes. Rank 0 needs about 12 GB more than gold (FP32 rows, gradients and Adam state), which is tight on an L4 (24 GB) next to the 6 GB FP32 model and activations. Ticket 07 measures both. Before this decision: L4 gives gold-hardware parity (gold and Sampling SFT trained on L4); A100-80GB Pod has a proven runbook and more memory (needed for rank 0). Different GPUs do not invalidate comparison of neuron lists, but behaviour comparisons record the runtime.

### D19. Comparison family

Primary (exploratory, predeclared): the final binary survivor set S against `c-production-1` per-read `dJdλ_j` for each GRPO arm, on the full 250,880 universe (both measured everywhere):
- per-layer counts of S;
- |S ∩ T| for C's frozen `global64-reads` and `layer64-reads` (positive selections), and for top-K by `dJdλ` with K ∈ {64, 256, 1,024, 4,096} (layer-free);
- null: 10,000 layer-matched random sets of |S|'s per-layer sizes, seed 20261123;
- stratified by the sign of `dJdλ` (positive = locally helpful), never treating it as a survivor probability.

Secondary (descriptive): S vs production-1's single-layer read rankings for gold-sft and the Sampling SFT arms (coverage one layer each, so the universe is that layer only), and vs joint-audit B/D lists per layer.

Not available: a full-universe attribution for any SFT arm. Missing means unmeasured, not zero.

### D20. Training-dynamics logging cadence

The researcher asked for a figure of gate shutting, read-change size and performance against step (`spec.md` §Training dynamics). Proposed cadence: gate totals and norms every step (free); validation CE and a `logα` snapshot every 8 steps (64 checks over 512 steps, about 10–15 min extra on an A100 by the same arithmetic as §Compute); greedy validation accuracy at the 9 checkpoints (about 9 × 128 answers, roughly 30–40 min); cohort150 only at 64 and 512. Alternatives: every 4 steps through step 128, where pruning is expected to be fastest (D1), then every 16; or greedy validation at fewer checkpoints to save time. Its value is descriptive, and one run cannot separate cause from co-movement.

### D21. Rank-1 vs rank-0 comparison

Requested by the researcher on 2026-11-23. It is descriptive: one seed per arm, and the arms differ in LR as well as rank (D1).

- **Survivors:** counts (expected-active, nonzero, binary) per arm; Jaccard overlap of the two final binary sets, plus overlap per layer; a layer-matched random reference for the overlap (10,000 draws, seed 20261124 **[proposed]**); and the neurons unique to each arm.
- **Against C attribution:** each arm's overlap with the `c-production-1` frozen sets and top-K lists (D19), plus the three-way intersection, all on the same universe and null.
- **Performance:** validation CE and greedy validation accuracy at each checkpoint, and cohort150 greedy at 64 and 512. Arm vs arm is compared paired by question (paired bootstrap on question IDs, 10,000 draws), with gold-sft and untuned as references.
- **Hardening:** the validation CE change across the freeze, per arm.
- **Dynamics:** ticket 12's four panels for both arms side by side, on shared y axes.
- **Read-change size:** the same gauge-invariant per-neuron norm (D20), so rank-0 `‖w_i‖` and rank-1 `‖b_i‖‖a‖` measure the same quantity: the norm of the change to neuron i's read row.

## Not equivalent to the rank-1 LoRA arms

- LoRA: `ΔW = B A` on 7 projection types (attention included) and on `down_proj` (writes). Rewiring: changes only gate/up reads, writes frozen.
- LoRA: one adapter per module, no per-neuron selection. Rewiring: per-neuron gate on a per-layer shared direction.
- LoRA init: A Kaiming-uniform, B zero, both trained. Rewiring init: shared direction `randn/√d`, coefficients zero, masks `logα = 1`, all trained.
- Same seed integer does not give the same random draws.

## Compute (arithmetic estimate, not a measurement)

Assumptions: about 250 tokens per training example (C spec records 115-token mean prompts for test questions; gold answer length to be read from `lengths.json`), 2,000 tokens per step, about 8 × 1.54 × 10⁹ FLOPs per token with checkpointing, A100 FP32 IEEE peak 19.5 TFLOP/s at 30–50% utilisation.

| Item | Rank 1 | Rank 0 |
|---|---|---|
| One step | 2.5–4 s | 3.5–6 s (two extra 1,536 × 8,960 matmuls per layer: roughly +40% FLOPs) |
| 512 steps | 20–35 min | 30–50 min |
| Validation, greedy validation at checkpoints, gold references (D20) | about 45 min | about 45 min (gold references shared, run once) |
| Greedy cohort150, 2 checkpoints | about 10 min | about 10 min |
| Setup, preflight, export | about 20 min, shared | — |
| **Total, one A100, run in sequence** | | **about 3–4 h** for both arms; L4 likely 2–3×, and rank 0 may not fit |
| Memory over gold | +0.84 M parameters, negligible | +770 M parameters, about +12 GB |

Ticket 07 replaces these with measured values before any launch record.

## Follow-ups, not in the minimal pilot

1. Second seed at the same settings (stability of S).
2. Rank 0 and rank 1 at matched survivor count, or at a shared LR; rank 4 only if the published model needs reproducing.
3. Read LR 1e-4 sensitivity (D1-A).
4. Causal check: remove the new reads of a selected subset vs matched controls on validation inputs.
5. Same-parameterisation objective comparison on a fixed offline pool (CE on correct draws vs offline GRPO).
6. Gold vs Sampling-SFT k0/k2/k5 data under the same rewiring objective.
