# Sparse rewiring on GSM8K: decision log

**Status:** needs-triage. Every recommendation here is **[proposed]** unless marked **[user]**. Evidence tags as in `design.md` §0. Predictions about training dynamics are derived from code, not measured; they are labelled "expected".

## The minimal pilot in one paragraph

Train two sparse-rewiring arms, rank 1 and rank 0 (full per-neuron rows) **[user 2026-11-23]**, from the untuned Qwen2.5-1.5B-Instruct on the 512 gold-SFT examples, with gold's token/label/loss contract, FP32 IEEE runtime and microbatching, and Max's hard-concrete L0 objective, mask optimizer and fixed-mask finish. λ 6e-4 for both, read LR 3e-3 (rank 1) and 3e-4 (rank 0), mask LR 0.1, one mask draw per 4 examples, 1,024 steps **[user 2026-11-23]** whose first 64 steps are gold's exact batches, last 64 steps fixed-mask (freeze at 960), on one A100-80GB **[user]**. Monitor a new train-split validation set only. Evaluate the step-64 and final checkpoints greedily on cohort150, approved for evaluation only **[user]**. Compare each arm's binary survivor set with the other's, with `c-production-1`'s per-read J scores and with production-1 adapter attribution, GRPO arms first **[user]**, on CPU; analyse each arm on its own (D22); and plot both arms' training dynamics side by side. One seed per arm, λ fixed at 6e-4, no sweep **[user]**.

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
| Schedule | constant, 64 | 20 warmup + cosine, 1,400, last 100 fixed | **Max shape, 1,024 steps, last 64 fixed [user]** | D5 |
| λ | — | 6e-4 | **6e-4 reference** | D6 |
| Checkpoints | 0/8/16/32/64 | every 50 | **gold's + 128/256/512/768/960/1,024** | D5 |

## Decisions

### D1. Read-update learning rate

| Option | For | Against |
|---|---|---|
| A. 1e-4 (gold) | Same number as every existing arm | Not the same parameters, so not the same optimisation. Expected to be degenerate with Max's mask LR: at init `b = 0`, so masks receive only the L0 gradient and Adam moves each `logα` by about the mask LR per step; from `logα = 1` the `z_det > 0` boundary (`logα = −ln 11`) is about 34 full-rate steps away. In that window `b` moves at most about 1e-4 per step, so the CE pull on masks stays small and near-total pruning is expected. Consistent with Max's log (expected-on 404,637 at step 10 → 3,900 at step 150 at lr 3e-3), but untested at 1e-4 |
| **B. 3e-3 (Max, rank 4)** | The value the mask LR and λ were tuned with; the only published operating point | Rank-1 and rank-4 optima can differ; no rank-1 result exists |
| C. 1e-4 and scale the mask LR down by 30 | Keeps gold's number | Two departures from Max instead of one; untested |

**Recommend B for rank 1**, with A as the single sensitivity run if budget allows.

**Rank 0: recommend 3e-4 [proposed].** `RANK0.md` reports that at rank 4's 3e-3, rank 0 overfit (over 1,000 neurons, run `v2_reread_r0_l06e-4_lr3e-3_s*`). At 3e-4 to 5e-4 with λ 6e-4 to 8e-4 it gave 82 to 122 neurons, mostly the same neurons as rank 4. Max's explanation: Adam moves every element of a full row at the LR, so full rows need a lower rate. The rank-1 and rank-0 arms therefore differ in LR as well as rank; the comparison (D21) states this.

**Rank 0: changed to 3e-5 [user 2026-11-25], superseding 3e-4.** The A100 rehearsal (`rehearsal-pod-1`, 16 steps, gold's order, masks frozen at 12) took rank 0 at 3e-4 from 103/128 validation questions correct at step 0 to 36/128 at step 12 and 12/128 at step 16; its 4-step qualification run reached validation CE 0.88 (base 0.573) with all 16 greedy answers at the 1,024-token cap. Rank 1 at 3e-3 reached 82/128. The rank-0 LR probe (`pilot_eval.rewire_lr_probe`, the same 16-step plan with only the read LR changed, `sparse-rewiring-lr-probe-1` on sl-0) gave at step 16:

| Rank-0 read LR | Correct /128 | Val CE step 12 → 16 |
|---|---|---|
| 3e-4 | 12 | — |
| 1e-4 | 29 | 0.432 → 0.513 |
| **3e-5** | **79** | 0.395 → 0.416 |
| 1e-5 | 73 | 0.394 → 0.408 |

References: base 103/128 (CE 0.573), gold SFT step 64 92/128 (CE 0.395). 3e-5 and 1e-5 are within noise of each other (one standard error is about ±5 questions); the researcher chose 3e-5 without a longer check. Open risk carried into production: 16 steps is short of gold's 64 and production's 1,024, and option A's concern above (a low read LR lets the L0 term prune nearly every mask before the reads matter) was not measured at this LR; the production run's checkpoints at 8, 16, 32 and 64 show it early.

### D2. Mask optimizer and LR

Recommend Max's exactly: a second AdamW parameter group, lr 0.1, weight decay 0, same `LambdaLR` multiplier as the reads (warmup + cosine apply to masks too), no gradient clipping on masks, `logα` init 1.0. Gold has no counterpart, so nothing is inherited from it. Record the resolved parameter groups in the run config.

### D3. Adam betas

Max uses (0.9, 0.99) for both groups (`--beta2`); gold uses (0.9, 0.999). Recommend Max's for both groups, because the second-moment window sets how fast mask steps shrink once the CE gradient appears. Record eps 1e-8 and `foreach=False` (gold) explicitly. Alternative: 0.999 for reads only (closer to gold, mixed recipe). Not recommended.

### D4. Gradient clipping

Both clip at 1.0. Max clips **read factors only**; masks are unclipped. Recommend the same, and log the read-factor norm before clipping and the unclipped mask-gradient norm separately.

### D5. Steps, schedule, freeze and checkpoints — decided

**[user 2026-11-23]: a fixed 1,024 steps (16 epochs) for both ranks.** 20-step warmup, then cosine to 0 at step 1,024 over both groups; masks frozen at step 960 and the final 64 steps trained with fixed masks.

Why fixed rather than an adaptive stop (considered and dropped): in Max's published run (`results/train/v2_reread_l06e-4_lr3e-3_s2/train_log.json@max`, rechecked here), expected-on was 476 at step 500 and 385 at 560, still falling fast; 158 at 1,000 and 150 at 1,060 (about −5% per 60 steps); and only flat (136 → 135) between 1,250 and 1,300, as the cosine LR approached zero. Freezing then took it from 135 to 89. The gates settle mainly because the LR decays, so a stop rule under a long cosine would rarely fire early, and an early stop would end at a high LR. Validation loss was already flat from step 1,000 (1.2182 → 1.2174 at 1,400).

| Option considered | Steps (epochs) | Fixed-mask phase | Comment |
|---|---|---|---|
| A. Gold-matched | 64 (1) | last 8 | Mid-pruning at step 64 |
| B. 512 | 512 (8) | last 64 | Max's run at 500 steps had 3.5× its final expected-on |
| **C. Chosen** | **1,024 (16)** | **last 64 (from 960)** | Full cosine decay to 0; 8,192 presentations of 512 examples, so validation CE watches overfitting |
| D. Adaptive stop | ≤ 1,024 | 64 after settling | Dropped: see above |
| E. Max-matched steps | 1,400 (22) | last 100 | Not chosen |

Settling is still **measured, as a logged diagnostic only** (D20): every 64 steps from 256, the Jaccard overlap of the hard survivor set (`z_det > 0`) with the set 64 steps earlier, and the relative change of expected-active over the same window. Nothing stops or changes on it. The report gives the first step after which both stay under the reference values (Jaccard ≥ 0.99, change < 2%) up to the freeze.

Checkpoints: 0, 8, 16, 32, 64 (gold's), 128, 256, 512, 768, 960 (last continuous-mask step) and 1,024 (final binary): 11 in all, each with greedy validation. Step 64 is a reference point under a different schedule, not gold's trajectory and not equal token exposure.

### D6. λ

**How Max chose λ.** He swept λ from 1e-4 to 3e-3 with up to 3 seeds, scored every run on held-out validation loss and on graded replies to evaluation prompts (does the model loop or repeat), and published "the sparsest run that loops no more than full fine-tuning" (`README.md@max`). So λ was picked by a quality floor first, then maximum sparsity, on his own data. For rank 0 his notes report λ 6e-4–8e-4 at LR 3e-4–5e-4 giving 82–122 neurons (`RANK0.md@max`). λ is a price per open neuron: each open gate must lower the training loss by more than about λ to stay open, so a higher λ means fewer survivors.

What that tells us: on his task about 100 neurons were enough to match full fine-tuning's quality, and the count moved smoothly with λ. It does not tell us the right λ here: different model, task, data size, and 250,880 candidates vs his 442,368. The survivor count we get at 6e-4 is itself a result.

**Decided [user 2026-11-23]: λ = 6e-4 for both ranks, no sweep.** If a run is degenerate (about 0 survivors, or more than 1,000), report it as such; nothing re-runs automatically, and any second λ is a new decision with its own run ID.

### D7. Data order beyond one epoch

Recommend: steps 1–64 consume gold's saved `inputs/order.json` exactly; epoch e ≥ 2 uses the 512 IDs reshuffled with `random.Random(42 + e − 1)`; record the full 1,024-step order (16 epochs) before training. Alternative: repeat gold's order every epoch (identical batches, but a periodic pattern). Max reshuffles each epoch.

### D8. Precision and runtime

FP32 weights, FP32 read factors and masks, no autocast, eager attention, `set_ieee_precision()` (IEEE for CUDA matmul and cuDNN, `highest`), CUBLAS workspace config, full determinism. A BF16 variant would be a separately named, separately qualified run and is not proposed.

### D9. Target and truncation contract

Gold's: labels from the frozen `training.items.json`, `<|im_end|>` supervised, trailing newline not supervised, no truncation. Max's `"<|im_end|>\n"` target and `MAX_LEN = 640` are recorded as differences not adopted.

### D10. Seeds

Separate and record: data order (gold's saved order, then D7), read-direction init (`seed + layer`, base 42, CPU `torch.Generator`, as in Max), mask noise (CUDA generator, seed 1042 proposed). A shared integer is not matched randomness with any other arm. A second seed (all three changed) is the first follow-up.

### D11. Validation set

**Role:** with λ fixed (D6) and a fixed step count (D5), validation is no longer used to choose anything in the training run. It gives the dense learning curve and accuracy at checkpoints (D20), shows whether the model overfits during the extra epochs, and ranks survivors in the reverts (D23). It costs about 15 min of GPU per arm. It is still needed because cohort150 is evaluated only twice, at 64 and final, by design, and a curve at every 16 steps on the reported set would turn it into a tuning set as soon as anyone looked at it to decide something.

No suitable set exists. Proposed: 128 GSM8K **train** questions, `random.Random(20261122)` sample from train indices excluding (a) `gsm8k-train-512`, (b) every `sequences-v2` calibration, discovery and held-out question, (c) any other train ID in a run record or the catalog. Train-split, because C ticket 14 plans to draw a held-out set from the **test** split minus prior uses, and this pilot should not consume it. Measures: gold-answer CE with deterministic masks (Max's selection quantity) every 16 steps (D20), and greedy flexible-v3 accuracy at checkpoints. **Blocked** until the `sequences-v2` ID lists are read from the artifact root (not in Git). Size 128 is a proposal.

### D12. Cohort150 use — decided

**[user 2026-11-23]: use cohort150 for evaluation only, labelled exploratory.** It is the frozen 150-question GSM8K test cohort every arm has been evaluated on, so the sparse arms are directly comparable with untuned, gold-sft, Sampling SFT and both GRPO arms. Nothing in this pilot is chosen on it. Each new evaluation is registered as a prior use in `catalog/cohorts.yaml`.

Why it was not called clean: its prior uses include greedy evaluation of gold-sft and online-grpo, GRPO behaviour runs, Stage 0, production-1 Stage D, and C attribution, which computed its scores and picked its frozen sets on these questions (`catalog/cohorts.yaml@catt`). **One caveat kept:** a result that combines "survivors overlap C's sets" with cohort150 accuracy is circular for the C side, because the same questions picked C's neurons.

### D13. Rank — decided

**[user 2026-11-23]: run both rank 1 and rank 0 (full rows), with everything else identical** (gold data and loss contract, FP32, shared mask per 4 examples, schedule, checkpoints, evaluations, dynamics figure). Rank 4, the published but self-described oversight, is not run.

| Arm | Status in Max's work | Trainable read parameters | Connection here |
|---|---|---|---|
| Rank 1 (`sparse-gold-sft-r1`) | Configured, **no released result** | 587,776 (28 layers × 2 parts × (1,536 + 8,960)) | Closest in shape to the r1 LoRA arms, but not equivalent |
| Rank 0 (`sparse-gold-sft-r0`) | **Intended method** (`RANK0.md`) | 770,703,360 (28 × 2 × 8,960 × 1,536) | Each neuron learns its own new gate and up rows |

Both arms have 250,880 mask logits. The arm names are proposed.

### D14. Microbatch and mask noise

**Decided [user 2026-11-23]: one shared mask per 4 examples (2 draws per step), microbatch 1.**

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

The catalog, IEEE helper, C outputs reader and neuron-audit helpers are on `c-attribution-pilot` only. **Recommend [proposed]: implement on `main` after `c-attribution-pilot` merges**, so the code builds on one reviewed base; if the merge is delayed, branch from `c-attribution-pilot@7667739` and rebase later. This design PR is docs-only against `main`.

### D18. Hardware — decided

**[user 2026-11-23]: one A100-80GB for both arms**, so the rank comparison does not mix runtimes. Rank 0 needs about 12 GB more than gold (FP32 rows, gradients and Adam state), which would have been tight on an L4. Gold and Sampling SFT trained on L4, so behaviour comparisons with gold record the runtime difference; neuron-list comparisons are unaffected. Ticket 15 measures memory and throughput for both ranks.

### D19. Comparison family — decided in scope

**[user 2026-11-23]: compare with C attribution and production-1 adapter attribution now, GRPO arms first. Write the comparison code so other arms' C attribution can be added later without code changes.**

Generic design: the comparison reads an **attribution source manifest**, one entry per (source, arm): source name, arm, score file and array name, its measured universe as a [28, 8960] boolean mask, the sign convention, and any frozen selections (named sets of `(layer, neuron)`). Every overlap, null and sign stratum is computed only inside that entry's universe. Adding an arm is a new manifest entry, not new code. Missing means unmeasured, never zero.

Run now (exploratory, predeclared), for each rewiring arm (r1, r0) and each GRPO arm (online, offline):
- **C attribution (`c-production-1`)**, full 250,880 universe: per-layer counts of the final binary survivor set S; |S ∩ T| for C's frozen `global64-reads` and `layer64-reads`, and for top-K by `dJdλ` with K ∈ {64, 256, 1,024, 4,096}; a layer-matched null of 10,000 random sets with S's per-layer sizes, seed 20261123; stratified by the sign of `dJdλ` (positive = locally helpful), never treated as a survivor probability; and the rank of survivors within C's per-layer distribution.
- **Production-1 adapter attribution**, GRPO arms only, on the one `down_proj` layer each arm covers: the same overlap and null restricted to that layer, using its trace and stop rankings.

Later (manifest entries only, when those attributions exist): C attribution for gold-sft and the Sampling SFT arms; production-1 single-layer rankings for those arms; joint-audit B/D lists per layer.

Not available today: a full-universe attribution for any SFT arm.

### D20. Training-dynamics logging cadence — decided

**[user 2026-11-23]: cheap metrics every step, validation CE every 16 steps.**

| Cadence | What is logged | Cost |
|---|---|---|
| Every step | training CE, L0 term, LR per group, grad norms, gate totals (expected-active, nonzero `z_det`), per-layer nonzero counts, gauge-invariant read-change norms (totals and survivor subset) | free (from tensors already in memory) |
| Every 16 steps | validation CE with deterministic masks; FP16 `logα` snapshot (0.5 MB) | 64 checks, about 10–15 min per arm |
| Every 64 steps from 256 | settle diagnostics (D5): hard-set Jaccard and expected-active change vs 64 steps earlier, in `logs/settle.jsonl`; diagnostic only | free |
| 11 checkpoints | greedy flexible-v3 accuracy on validation (128 questions); survivor list; churn vs previous checkpoint | about 35–45 min per arm |
| Step 64 and final | greedy cohort150 (evaluation only, D12) | about 10 min per arm |

Its value is descriptive, and one run cannot separate cause from co-movement.

### D21. Rank-1 vs rank-0 comparison

Requested by the researcher on 2026-11-23. It is descriptive: one seed per arm, and the arms differ in LR as well as rank (D1).

- **Survivors:** counts (expected-active, nonzero, binary) per arm; Jaccard overlap of the two final binary sets, plus overlap per layer; a layer-matched random reference for the overlap (10,000 draws, seed 20261124 **[proposed]**); and the neurons unique to each arm.
- **Against C attribution:** each arm's overlap with the `c-production-1` frozen sets and top-K lists (D19), plus the three-way intersection, all on the same universe and null.
- **Performance:** validation CE and greedy validation accuracy at each checkpoint, and cohort150 greedy at 64 and final. Arm vs arm is compared paired by question (paired bootstrap on question IDs, 10,000 draws), with gold-sft and untuned as references.
- **Hardening:** the validation CE change across the freeze, per arm.
- **Dynamics:** ticket 13's four panels for both arms side by side, on shared y axes.
- **Read-change size:** the same gauge-invariant per-neuron norm (D20), so rank-0 `‖w_i‖` and rank-1 `‖b_i‖‖a‖` measure the same quantity: the norm of the change to neuron i's read row.

### D22. Standalone analysis of each rank

Requested by the researcher on 2026-11-23: an analysis of what each rewiring arm did, apart from any attribution comparison. All CPU from saved files except the activation shift (one forward pass per arm). Full list in `spec.md` §Standalone analysis (ticket 12).

- **Selection:** survivor counts and per-layer distribution; gate dynamics and the settle step; survivor stability across checkpoints.
- **What survivors learned:** change size, gate vs up; rank 1: the shared `a` per layer and its cosine with gold LoRA's gate and up `A`; rank 0: effective rank of the survivors' change; activation shift vs base on GSM8K.
- **Behaviour:** validation CE and accuracy; cohort150 at 64 and final; invalid, capped and length; per-question flips vs gold and untuned.

Proposed definitions (each **[proposed]**):
- *Settle step*: from the D5 diagnostics, the first check after which Jaccard ≥ 0.99 and expected-active change < 2% hold at every later check up to 960; plus the first 16-step snapshot after which the hard set's Jaccard with the final set stays ≥ 0.99.
- *Effective rank*: participation ratio `(Σσ)² / Σσ²` of the singular values of the stacked survivor rows per layer and part, plus the number of singular values needed for 90% of `Σσ²`.
- *Activation shift*: on the 128 validation questions (prompt and gold answer, teacher-forced), mean and mean-absolute change of each survivor's post-SiLU gate × up activation, rewired vs base; plus the same for 1,000 random non-survivors per layer as a reference.

### D23. Causal checks — skipped for the pilot

**[user 2026-11-23]: skip for now.** Kept below as a follow-up that can run later on the saved final checkpoints, which hold everything needed (factors and masks). Max did the single-revert form: removing one survivor's new reads and measuring the validation-loss rise (L8.N344: +0.068, 2.7 times any other of the 89; `README.md@max`). If run, single-revert effects need FP64 accumulation and the FP32 floor reported (lessons 2026-11-13, 2026-11-16), and the control uses `c_attribution_selection.match_group`.

Not in the earlier scope; listed now because the researcher's analysis list implies them.  All run on the final binary model and on validation only; cohort150 only for the revert-all check, labelled exploratory.

| Check | What it does | Read-out | Est. GPU, per arm |
|---|---|---|---|
| Revert all | Remove every survivor's learned read change (model returns to base) | Sanity: validation CE and accuracy return to untuned within FP32 round-off | about 10 min (CE, greedy validation, greedy cohort150) |
| Single reverts | Revert one survivor at a time | Δ validation CE per survivor (teacher-forced only), giving an importance ranking | about 10–15 s each; about 20–40 min at 100–150 survivors. If there are more than 512 survivors, layer blocks only |
| Block reverts | Revert all survivors in one layer, then the top-k by single-revert importance (k = 1, 4, 16, 64) | Δ validation CE for every block; greedy accuracy for the 4 top-k blocks and the 4 largest layer blocks | about 40 min |
| Same-layer random control | Move each survivor's learned change `Δw_i` onto a random non-survivor in the same layer (20 draws) | Does location matter, or only the direction and size of the change? | about 15 min (CE for 20 draws, greedy for 3) |

Reverting random non-survivors alone would be a no-op (their change is zero), which is why the control transplants the survivors' changes instead. **Total, both arms: about 2–3 h** (arithmetic estimate, not measured).

### D24. Checkpoint storage — decided

**[user 2026-11-23]: store only the weights (read factors) and the masks (`logα`) at each checkpoint**, with a small seal (step, plan, runtime and code-commit hashes). No optimizer, scheduler or RNG state anywhere.

Why it came up: a full rank-0 checkpoint with Adam state would be about 9 GB, and 11 of them about 100 GB per arm, beyond the Runpod default container disk that already failed once (`experiment-lessons.md@catt`, 2026-11-12). Factors and masks are about 3.1 GB per rank-0 checkpoint (about 34 GB in all) and about 3.5 MB per rank-1 checkpoint.

Consequence: **a crash means a restart, not a resume.** An interrupted arm restarts from step 0 under the same plan; full determinism makes the restart reproduce the same trajectory (test 11). A completed and verified arm is kept, so a crash during rank 0 never reruns rank 1. Worst-case loss is one arm's training time (about 40–70 min for rank 1, 60–100 min for rank 0). The disk total sets the launch record's `disk` block, which `TransformersBackend.preflight` enforces.

### D25. A100 supersedes ADR 0018's L4 rule for these arms — approved in substance

**[user 2026-11-23]: L4 is not required for training.** ADR 0018 says "Training still requires L4 to match gold SFT". A short ADR (ticket 20) records that, for the sparse arms, the A100-80GB choice (D18) replaces that rule: the sparse arms are new arms, not gold SFT reruns, and their behaviour comparisons with gold carry the runtime difference.

## Not equivalent to the rank-1 LoRA arms

- LoRA: `ΔW = B A` on 7 projection types (attention included) and on `down_proj` (writes). Rewiring: changes only gate/up reads, writes frozen.
- LoRA: one adapter per module, no per-neuron selection. Rewiring: per-neuron gate on a per-layer shared direction.
- LoRA init: A Kaiming-uniform, B zero, both trained. Rewiring init: shared direction `randn/√d`, coefficients zero, masks `logα = 1`, all trained.
- Same seed integer does not give the same random draws.

## Compute (arithmetic estimate, not a measurement)

Assumptions: about 250 tokens per training example (C spec records 115-token mean prompts for test questions; gold answer length to be read from `lengths.json`), 2,000 tokens per step, about 8 × 1.54 × 10⁹ FLOPs per token with checkpointing, A100 FP32 IEEE peak 19.5 TFLOP/s at 30–50% utilisation. One A100-80GB (D18), 1,024 steps (D5).

| Item | Rank 1 | Rank 0 |
|---|---|---|
| One step | 2.5–4 s | 3.5–6 s (two extra 1,536 × 8,960 matmuls per layer: roughly +40% FLOPs) |
| 1,024 steps | 40–70 min | 60–100 min |
| Validation CE every 16 steps (64 checks) | 10–15 min | 10–15 min |
| Greedy validation at 11 checkpoints | 35–45 min | 35–45 min |
| Greedy cohort150 at 64 and 1,024 | about 10 min | about 10 min |
| Activation shift (D22) | about 5 min | about 5 min |
| Gold-sft references on validation (once), setup, preflight, export | about 35 min, shared | — |
| **Total, both arms in sequence** | | **about 5–5.5 h** (wider bounds 4.5–6 h) |
| Memory over gold | +0.84 M parameters, negligible | +770 M parameters, about +12 GB |

Ticket 15 replaces these with measured values before any launch record.

## Follow-ups, not in the minimal pilot

1. Second seed at the same settings (stability of S).
2. Rank 0 and rank 1 at matched survivor count, or at a shared LR; rank 4 only if the published model needs reproducing.
3. Read LR 1e-4 sensitivity (D1-A).
4. Causal checks (D23): revert all, single and block reverts, same-layer transplant control, about 2–3 h GPU for both arms, on the saved final checkpoints.
5. Same-parameterisation objective comparison on a fixed offline pool (CE on correct draws vs offline GRPO).
6. Gold vs Sampling-SFT k0/k2/k5 data under the same rewiring objective.
