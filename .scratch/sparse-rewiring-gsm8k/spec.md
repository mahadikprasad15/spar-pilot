# Sparse gold-SFT rewiring pilot — draft implementation spec

**Status:** needs-triage. Draft v1 (2026-11-22) from `design.md` and `decision-log.md`. Design and drafting only: no code, GPU run, threshold or budget is approved by this file. **[user]** = researcher decision; **[proposed]** = awaiting decision; **BLOCKED(Dn)** = cannot be implemented or frozen until decision Dn in `decision-log.md` is made.

Four stages are kept distinct throughout: **CPU implementation** (tests on tiny fixtures), **GPU qualification** (real model, no scientific run), **training/evaluation execution** (a launch record approved by the researcher), and **scientific interpretation** (report). Passing one says nothing about the next.

## Problem

Every existing arm adapts dense rank-1 LoRA modules, and the neuron-level evidence (C attribution, production-1, joint audit) describes how a `down_proj` adapter reads neuron activations. No arm selects which MLP neurons need their **inputs** changed. Max Loeffler's sparse rewiring does this with a per-neuron hard-concrete gate and an L0 penalty, leaving every write frozen. A gold-SFT version on the existing 512 examples gives (a) a new baseline arm and (b) a sparse, interpretable neuron set to compare with the GRPO attribution rankings.

## Solution

Two new arms, `sparse-gold-sft-r1` and `sparse-gold-sft-r0` (catalog names **[proposed]**), identical except for rank and read LR **[user 2026-11-23]**: the untuned model with a `MaskedReRead`-style module on every MLP, trained on gold's frozen training items with gold's loss contract and runtime plus Max's L0 objective, then hardened to a binary survivor set. Saved factors, masks and full recovery state. 1,024 steps, λ 6e-4, one A100-80GB. Validation-set monitoring, greedy cohort150 evaluation at 64 and final, a standalone analysis of each rank, a rank-1 vs rank-0 comparison, and a CPU comparison of each with C attribution and production-1 adapter attribution for the GRPO arms (generic over arms).

## User stories

1. As a researcher, I want both arms trained on gold's exact frozen token IDs and labels so data and format cannot explain differences from gold SFT.
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
- Hard concrete with β 2/3, γ −0.1, ζ 1.1; training noise: one uniform `u` per layer per group of 4 consecutive examples, drawn from a dedicated generator, cached, and reused by each single-example pass and by checkpoint recompute; `z` is recomputed from the cached `u` and the current `logα` (D14, [proposed]); deterministic gate `clip(σ(logα)(ζ−γ)+γ, 0, 1)`; expected-active `Σ σ(logα − β ln(−γ/ζ))`; freeze rule `z_det > 0` → binary. Ranks: 1 and 0 are both run **[user 2026-11-23]**; the module also supports k > 1. Rank 0 adds 770.7 M FP32 read parameters, so the memory test in ticket 15 covers it explicitly.
- An `off` switch returning the exact base MLP, and a per-neuron "remove new reads" intervention for later causal checks.

### Training

- Explicit loop adapted from `sampling_sft_training_backend.train` (not TRL), reusing `sft_backend.completion_loss`, `CompletionCollator` and `frozen_weight_hash`. Objective per step: Σ over microbatches of `completion_loss(…, denominator = step's supervised tokens)` + `λ · expected_active` (once per step).
- Optimizer: one AdamW, `foreach=False`, two groups. Values **BLOCKED(D1, D2, D3)**; proposed reads lr 3e-3 (rank 1) / 3e-4 (rank 0), masks lr 0.1, betas (0.9, 0.99), eps 1e-8, wd 0. Clip 1.0 on read factors only (D4). `LambdaLR` shared by both groups; schedule **[user 2026-11-23, D5]** 1,024 steps for both ranks: 20 warmup + cosine to 0 at 1,024, masks frozen at 960, final 64 steps with fixed masks.
- Settle diagnostics **[D5, logged only]**: every 64 steps from 256, Jaccard of the hard survivor set (`z_det > 0`) with the set 64 steps earlier, and relative change of expected-active over the window, in `logs/settle.jsonl`. Nothing stops or changes on them.
- λ **[user 2026-11-23, D6]** 6e-4 for both ranks, no sweep. A degenerate run (about 0 or more than 1,000 survivors) is reported, not re-run.
- Runtime (D8): FP32, `set_ieee_precision()`, eager attention, `use_cache=False`, gradient checkpointing (D15), full determinism, pinned packages from `sft_backend.PINS`.
- Logged every step **[user 2026-11-23, D20]** (`logs/steps.jsonl`): CE, L0 term, total, LR per group, read-factor grad norm before clipping, mask grad norm, examples seen, and the gate and read-change summaries listed under §Training dynamics, including per-layer nonzero counts and survivor-subset norms.
- Every 16 steps **[user 2026-11-23, D20]** (`logs/validation.jsonl`): validation CE with deterministic masks, plus a gate snapshot (`masks/trace/<step>.npz`: `logα` as FP16 for all 250,880 neurons, about 0.5 MB each).

### Training dynamics (figure requested by the researcher, 2026-11-23)

One figure with step on a shared x axis, built on CPU from the saved logs (ticket 13). Its purpose is descriptive: it shows how gate shutting, read-change size and performance move together within one run. Co-movement is not evidence that one causes the other.

| Panel | Quantities | Source and cadence |
|---|---|---|
| 1. Gates | expected-active `Σ P(z>0)`, nonzero deterministic count (`z_det > 0`), and binary survivors after the freeze, on a log y axis; a per-layer heatmap (layer × step) of nonzero-`z_det` counts | every step (totals and per-layer counts) |
| 2. Read-change size | the gauge-invariant per-neuron row change `‖Δw_i‖ = z_i ‖b_i‖ ‖a‖` (rank 1; `‖b_i A‖` for rank k; `‖w_i‖` for rank 0), summed and as a median, for gate and up separately; once over all neurons and once over the currently nonzero (survivor) set | every step |
| 3. Performance | validation CE (deterministic masks, dense); greedy flexible-v3 accuracy on validation at the checkpoints; cohort150 greedy only at 64 and final (D12); reference lines for untuned (step 0) and gold-sft step 64 evaluated on the same validation set | CE every 16 steps; accuracy at checkpoints |
| 4. Events | vertical line at the freeze step (960); settle diagnostics at each 64-step check; mask churn between consecutive checkpoints as the Jaccard overlap of their nonzero sets, plus counts that switched on and off | checkpoints |

- Raw `‖a‖` or `‖b‖` alone is not reported as a size: rescaling `a` by k and `b` by 1/k leaves the change the same.
- The gold reference lines need gold-sft checkpoints 0 and 64 evaluated on the new validation set (CE plus one greedy pass). That is a new, cheap use of existing adapters and is registered in the catalog.
- Every panel shows the real evaluation points as markers; nothing is interpolated between checkpoints.
- Checkpoints **[user 2026-11-23, D5]** 0/8/16/32/64/128/256/512/768/960/1024, each with unmerged factors, all `logα`, scheduler, data cursor and seal; optimizer and both RNG streams at the latest two only (D24 [proposed], rank 0 is about 9 GB per full checkpoint); 960 and 1,024 bracket the hardening. Greedy validation at every checkpoint. Resume only from a complete sealed checkpoint with matching plan and runtime hashes.

### Validation and cohort150 roles

λ and the step count are fixed, so nothing in training is chosen on any evaluation set. Validation (D11) gives the dense curve, overfitting watch and revert rankings. Cohort150 **[user 2026-11-23, D12]** is evaluation only, at 64 and final, labelled exploratory. Because C's scores and frozen sets were computed and selected on cohort150, a result that combines C overlap with cohort150 accuracy is labelled circular.

### Standalone analysis (D22, ticket 12)

Requested by the researcher on 2026-11-23, for each rank on its own. CPU from saved files, except the activation shift (one forward pass on validation per arm, inside ticket 17).

| Group | Measure | Definition | Source |
|---|---|---|---|
| Selection | Survivor counts | expected-active, nonzero `z_det` and binary, total and per layer (one gate per neuron covers both its gate and up reads) | `masks/<k>.json` |
| | Per-layer distribution | counts per layer, with a uniform-by-layer reference | final binary set |
| | Gate dynamics and settle step | `logα` histograms at checkpoints; settle step = from the D5 diagnostics, the first check after which both reference values hold to 960, plus the first 16-step snapshot after which Jaccard with the final set stays ≥ 0.99 **[proposed]** | `masks/trace/` |
| | Stability | Jaccard overlap of each checkpoint's nonzero set with the final binary set; when each survivor first turned on and stayed on | checkpoints |
| What survivors learned | Change size | gauge-invariant `‖Δw_i‖` for gate and up, distribution over survivors, ratio gate/up per layer | final factors |
| | Rank 1: the shared direction | `a` per layer and part; cosine of `a` with gold LoRA's gate and up `lora_A` rows in the same layer (sign-free, absolute cosine), against a random-direction reference | rewire state, gold step-64 adapter |
| | Rank 0: effective rank | per layer and part, participation ratio of the singular values of the stacked survivor rows, and the count reaching 90% of `Σσ²` **[proposed]**; compares with rank 1's forced rank 1 | final factors |
| | Activation shift | on the 128 validation items (prompt + gold answer, teacher-forced), mean and mean-absolute change of each survivor's `SiLU(gate)·up` vs base; 1,000 random non-survivors per layer as reference | one forward pass per arm |
| Behaviour | Validation | CE every 16 steps; greedy accuracy at 11 checkpoints | logs, eval |
| | Cohort150 | greedy flexible-v3 at 64 and final; invalid (no parseable answer), capped (hit 1,024 tokens), response length | eval |
| | Per-question flips | on cohort150, the 2×2 tables vs gold-sft and vs untuned (fixed, broken, both, neither), with paired bootstrap on question IDs | stored responses |

### Causal checks (D23, ticket 19) — proposed, not approved

On the final binary model, validation inputs only, except the revert-all sanity check, which also runs cohort150. Revert all (expect untuned within round-off); single reverts (validation CE only) for an importance ranking, layer blocks only if there are more than 512 survivors; block reverts per layer and top-k by importance (k = 1, 4, 16, 64), greedy for 8 blocks; same-layer random control that moves each survivor's change to a random non-survivor in the same layer (20 draws). Reverting random non-survivors alone is a no-op, hence the transplant. About 1–1.5 h per arm, arithmetic estimate. Needs the researcher's OK and its own launch record.

### Outputs

```text
artifacts/runs/sparse-rewiring/<run-id>/        one run ID per arm (r1, r0); both share one plan except rank and read LR
  config.json  runtime.json  status.json  logs/steps.jsonl  logs/validation.jsonl
  inputs/order.json  inputs/validation.items.json (+ exclusion record)
  checkpoints/checkpoint-<k>/  rewire_state.safetensors  optimizer.pt  scheduler.pt  rng.pt  complete.json
  masks/<k>.json             expected / nonzero / binary counts per layer; survivor list
  masks/trace/<step>.npz     logα snapshot every 16 steps (FP16), for the per-layer heatmap and churn
  dynamics/                  figure (PNG + SVG), its input table (CSV) and a provenance JSON
  hardening.json             rule, pre/post counts, validation CE before and after
  eval/<k>/greedy/           responses.jsonl  config.json  scores.json
  analysis/                  standalone tables and figures (D22), activation_shift.npz
  causal/                    only if D23 is approved: reverts, controls, configs, responses
  compare/                   overlap tables, null draws, figures (CPU), the attribution source manifest used
```

### Evaluation

- Validation: gold CE (deterministic masks) and greedy flexible-v3 at checkpoints, on the D11 set.
- Cohort150 greedy at 64 and final **[user 2026-11-23, D12, evaluation only]**: existing harness, cap 1,024, EOS 151645, explicit repetition penalty 1.0, flexible-v3; paired per-question comparison with the stored untuned, gold-sft, online- and offline-grpo runs, labelled with runtime differences. Every run writes its config and all responses (AGENTS.md).

### Comparison

- D21 (rank 1 vs rank 0, CPU): survivor counts; Jaccard overlap of the final binary sets and overlap per layer, with a layer-matched null; neurons unique to each arm; each arm's and the intersection's overlap with the C sets; validation CE and accuracy per checkpoint; cohort150 greedy at 64 and final, paired by question; CE change across hardening; the dynamics figure side by side on shared axes.

- D19 **[user 2026-11-23]**: GRPO arms first, generic over arms. The comparison reads an attribution source manifest: one entry per (source, arm) with the score file and array, its measured universe as a [28, 8960] boolean mask, the sign convention, and named frozen selections. Overlaps, nulls and sign strata are computed inside each entry's universe only; adding gold-sft or Sampling SFT C attribution later is a manifest entry, not code.
- Entries now: `c-production-1` for online-grpo and offline-grpo (`aggregate/<arm>/neurons.npz`, `freeze/<arm>.json`, full universe), and production-1 adapter attribution for the same two arms (one `down_proj` layer each). All read in place, never modified. Joins on zero-based `(layer, neuron)`; the loader asserts [28, 8960] and the index convention from the C freeze.
- Inputs from this effort: the final binary survivor set, and the continuous gates at the freeze step as a secondary.

### Registration

- Both arms, checkpoints, response sets and cohort prior uses in `catalog/` in the same change as the run records; `catalog verify` must pass on local entries.

### Code reuse (audit, 2026-11-23)

Every component names what it reuses. Paths are on `c-attribution-pilot7667739` (`catt`), which implementation builds on (D17). "New" is justified; nothing below is planned as new where an existing function does the job.

| Component | Reuses | New, and why |
|---|---|---|
| Gold data and loss | `sft.load_sft` (plan + hash checks), `sft_backend.CompletionCollator`, `sft_backend.completion_loss`, `sft_backend.PINS` | none |
| Base identity | `sft_backend.frozen_weight_hash`; `c_attribution_bundle.check_base_model` / `pinned_identities` (HF per-file pins) | none |
| Runtime precision | `neuron_audit_collector.set_ieee_precision`, called through `c_attribution_backend.TransformersBackend` (`precision_readback`, `precision_problems`; preflight refuses if any backend is not `ieee`) | none. **Not** `sampling_sft_training_backend.open_engine`, which sets the legacy `allow_tf32` flags: they leave cuDNN at TF32 in torch 2.9, and mixing them with the new API warns or raises |
| Training loop | the explicit loop in `sampling_sft_training_backend.train`, without its `open_engine` precision setup (FP32 check, context-limit refusal, `use_reentrant=False` checkpointing, base-hash check, `foreach=False` AdamW, resume binding); `sampling_sft_training.checkpoint` / `seal` / `load` / `event`; `sampling_sft_training_backend.rng_state` / `restore_rng`; `workflow._hash` for plan and runtime hashes | Two parameter groups, the warmup + cosine `LambdaLR`, the L0 term and the freeze at 960. That loop has one group and a constant LR, and no mask parameters. Adapted, not rewritten: every check it enforces is kept (test 19) |
| Rewire module | none exists (no masked read-change module in the repo) | `rewire_module.py`: the method itself. Clean-room (D16) |
| Hard-concrete gate, shared-mask cache | none | `hard_concrete.py`: Max's equations plus the D14 per-group `u` cache |
| Plan and launch record | `c_attribution_chain.typecheck` / `validate` / `seal_launch_record` / `load_launch_record` (typed YAML, the lesson that PyYAML reads `19.5e12` as text) | the plan schema's new fields only |
| Validation sampler | `catalog.load` / `find` for recorded train uses; the `sequences-v2` manifest reader in `neuron_audit_sequences` | the seeded exclusion sampler (about 40 lines); no train-split validation set exists |
| Validation CE | `completion_loss` on items built by the gold builder | none |
| Greedy evaluation | `c_attribution_generation.greedy` and `score` (flexible-v3 via `scoring.score_gsm8k_flexible_v3`), batching from `c_attribution_generation.batches`; protocol constants from `protocol` | A loader for rewire checkpoints. `sft_evaluation.evaluate_sft` is **not** reusable as is: it accepts only checkpoints 0/8/16/32/64 and loads PEFT adapters |
| Paired comparisons | `c_attribution_generation.mcnemar_exact`, `summarize_arm`; `c_attribution_stability.resample_counts` (paired bootstrap on question IDs) | none |
| Overlap and null | `neuron_audit_math.matched_null` (bins = layers), `exact_overlap_distribution`, `derive_seed`, `reference_interval`; `c_attribution_stability.top` / `overlap` / `spearman`; `c_attribution_selection.top_positive`, `read_freeze` | the attribution source manifest (D19): a small loader that turns each source into (universe, scores, frozen sets) |
| Gauge-invariant norms | `neuron_audit_math.gauge_transform` (test fixture for invariance) | the per-neuron norm itself (one line per rank) |
| Transplant control (D23) | `c_attribution_selection.match_group` (seeded group-wise matching, after the sequential-control lesson) | the transplant intervention |
| Sealing, persistence, export | `attribution_units.seal_run` / `verify_run` / `_encode` (NumPy-safe), `export`, `restore_summary` | none |
| Pod run, preflight, teardown | `c_attribution_chain.Chain` (precheck, `check_pod`, stop-record order, backup coverage before stop), `c_attribution_backend.TransformersBackend.preflight` (record-driven disk check `disk.min_free_gb` / `container_total_gb`, `HF_HOME`, offline worker, `neuron_audit_preflight.volume_identity`, private repo) and `runtime` (torch build, CUDA runtime, cuDNN and CUDA-library versions as the resume fingerprint), `c_attribution_chain.stop_commands`, `scripts/c_attribution_launch.sh` (watchdog with deadline disabled), the C-attribution runbook | only the stage list for this run. Writing a new runner would repeat about a dozen fixed Pod failures |
| Catalog | `catalog.verify`; `c_attribution_report.register` / `_append_prior_uses` (prior uses from sealed units, comments kept) | the new arm and response entries |
| Figures | `c_attribution_report._style` and `figures` conventions | the four dynamics panels and the analysis figures |
| Standalone analyses | `numpy.linalg.svd`; `neuron_audit_collector.load_saved_factors` to read gold's LoRA `A` for the cosine | settle step, participation ratio and activation-shift hooks (no existing equivalent) |

### FP32 and qualification: reuse of the fixing code

The project's precision and qualification failures, the code that fixed each, and whether this spec calls that code or reimplements it. Rule: call the fixing code directly. Where it is coupled to PEFT or to attribution, extend it with a parameter (keeping its existing callers bit-identical and their tests green) rather than copy it.

| Past failure (source) | Fixing code on `catt` | Here |
|---|---|---|
| FP32 weights alone did not stop TF32; a rewritten backend dropped the IEEE call (lessons 2026-11-20, ADR 0021) | `neuron_audit_collector.set_ieee_precision`; `c_attribution_backend.TransformersBackend.precision_readback` / `precision_problems` / `preflight` | **Calls it.** The run builds on `TransformersBackend` (load and save overridden for rewire checkpoints). Test 24 fails if any legacy `allow_tf32` setter runs |
| `model.double()` is not FP64 for Qwen2; FP32 finite differences hit the resolution floor (lessons 2026-11-13, 2026-11-16) | `attribution_instrument.full_fp64`; the FP64 reference via `TransformersBackend.reference` | **Calls it** if a single-revert Δ (D23) falls under the FP32 floor: re-measure that revert under `full_fp64`. Not needed for training |
| FP64 reference stayed live and inflated peak memory (lessons 2026-11-20) | `TransformersBackend.release`, plus dropping every owner (pattern in `c_attribution_stages.Stages.qualification`) | **Calls it** between gold references, rank 1 and rank 0 |
| Greedy no-op check put its prompt on CPU with the model on CUDA (lessons 2026-11-19) | `c_attribution_checks.noop_check` | **Calls it** for the zero-change identity (off switch and step 0 vs untuned). Extended with a parameter-name filter: its `adapter_hash` hashes only `lora_` names, so on a rewired model it would hash nothing and pass vacuously |
| Logits-processor check patched the wrong object (lessons 2026-11-19) | `c_attribution_checks.processor_check` / `resolve_processors` | **Calls it** to prove greedy evaluation resolves no repetition penalty or other processor |
| KV-cache decoding vs a forced forward can differ | `c_attribution_checks.cached_vs_forced` | **Calls it** on rewired checkpoints, because the module runs inside cached decoding |
| `pad_token_id` absent from Qwen2.5's config; fixtures hid it (lessons 2026-11-11) | `TransformersBackend.pad_token_id` from the tokenizer files | **Calls it** |
| `2.9.0+cu128` rejected by an exact version check (lessons 2026-11-11) | Superseded in the current C code: `TransformersBackend.runtime` records the actual build and CUDA libraries, and a resume under a different fingerprint is refused | **Calls the current code**, not the older `neuron_audit_preflight.check_interpreter` |
| Real preflight body passed a path where a dict belonged; YAML read `19.5e12` as text (lessons 2026-11-16) | `TransformersBackend.preflight` (disk, `HF_HOME`, offline worker, volume identity, private repo); `c_attribution_chain.typecheck` | **Calls them** |
| Qualification marked complete without sign-off; stage receipts reused under another runtime (lessons 2026-11-14, 2026-11-19) | `c_attribution_chain.Chain` sign-off handling; `c_attribution_stages.Stages.preflight` (reruns every attempt, binds runtime, bundle, base model and code commit) | **Calls them** (ticket 14) |
| Qualification row requirements impossible for some arms (lessons 2026-11-18) | `Stages._qualification_rows` (spread over length, required rows only where present) | **Same pattern**: longest example plus rows spread over length |
| Gold's own preflight: zero write at init, finite gradients, real optimizer state allocated then restored (Pilot 2) | `sft_backend.HFTrainingEngine.preflight` | **Calls it**, extended so the zero-change test and trainable set are parameters (it now uses PEFT's `disable_adapter` and `lora_B`). Gold's call is unchanged |
| Throughput and memory measured by hand | `attribution_gpu.StageMonitor`, `throughput`, `mfu`; `TransformersBackend.reset_peak` / `peak_memory` / `total_memory` with the record's `memory_fraction` | **Calls them** |
| vLLM vs HF score mismatch (ADR 0021, IEEE Triton) | `sampling_sft_engine` runtime controls | Not relevant: no vLLM; evaluation uses HF `generate` |

Attribution-only checks not reused because they test something this run does not compute: `surrogate_finite_difference`, the attribution identity check and `batch_invariance` (both take an attribution `Instrument`).

**Qualification (ticket 15)** therefore runs the existing harness: `Stages.preflight` and `TransformersBackend.preflight`, then `noop_check`, `processor_check`, `cached_vs_forced`, the extended `HFTrainingEngine.preflight` on the longest example for each rank, and the peak-memory rule, all sealed with `attribution_units.seal_run`. New checks are only the ones specific to this method: the shared-mask cache on the real model (same gradients with and without checkpoint recompute) and the D24 disk total.

### Lessons applied (`docs/agents/experiment-lessons.mdcatt`, ADRs)

Read on 2026-11-23: every entry in `docs/agents/experiment-lessons.md` (which exists only on `catt`, not on `main`), ADRs 0001–0023, `docs/experiments/status.md`, and the Pilot 2 SFT and C-attribution review notes. **Changed the design** marks the lessons that altered a decision; the rest are handled by reusing code that already carries the fix.

| Lesson (date) | How it is handled here | Effect |
|---|---|---|
| Runpod default container disk too small (2026-11-12) | Rank 0 recovery state is about 9 GB per checkpoint (770 M FP32 factors + 2 Adam moments). Eleven full checkpoints would be about 100 GB per arm. Storage policy D24: factors and `logα` at every checkpoint, full optimizer and RNG state only at the latest two. The disk size goes into the launch record's `disk` block, which the current `TransformersBackend.preflight` enforces | **Changed the design** (D24) |
| ADR 0018: "Training still requires L4 to match gold SFT" | **Superseded for these arms [user 2026-11-23]:** the researcher said L4 is not needed for training. A short ADR records that the A100-80GB choice (D18) replaces ADR 0018's L4 rule for the sparse arms (ticket 20) | Recorded (D25) |
| Deleting a helper's parameter does not release the model (2026-11-19) | Gold-sft reference evaluations run in a separate child process before rank 0 training, so no reference model can stay resident next to rank 0's extra 12 GB | **Changed the design** (run order) |
| FP32 finite differences have a resolution floor (2026-11-13, 2026-11-16) | Single-survivor reverts (D23) change validation CE by small amounts. Accumulate CE in FP64, report the FP32 floor next to every Δ, and label effects below it unresolved | **Changed the design** (D23) |
| Keep confirmation outcomes out of rehearsals (2026-11-13) | The disposable qualification run evaluates on validation only, never cohort150 | **Changed the design** (ticket 15) |
| A rewritten backend must carry over every setting (2026-11-20) | The loop is adapted from `sampling_sft_training_backend.train`; test 19 lists each check that loop makes and asserts the new one makes it too | reuse + test |
| Test the real backend boundary on CPU; fakes hid wrong arguments (2026-11-16) | Test 20: a real-path CPU chain on a saved tiny Qwen2, stubbing only CUDA counters, the volume, the Hub and the stop | reuse pattern |
| Type-check YAML values (2026-11-16) | `c_attribution_chain.typecheck` on the plan and launch record | reuse |
| Bind resumable stages and measurements to inputs, runtime and code commit (2026-11-11, 2026-11-19) | Plan, runtime and code-commit hashes in every checkpoint seal; resume refuses a mismatch and reruns preflight | reuse |
| Exercise recovery on an empty disk; reconstruct expected work (2026-11-11, 2026-11-14) | Test 11 resumes in a fresh directory from the volume copy; expected checkpoints are listed in the plan and compared on resume | test |
| Durability while running; stop only with full backup; closing records before stop; stop the right Pod (2026-11-20, 2026-11-22) | Reuse `c_attribution_chain` teardown, which carries these fixes; mirror at every checkpoint | reuse |
| HF cache off the volume; hash every download (2026-11-16) | Runbook step from the C-attribution Pod runbook | reuse |
| Version checks against `+cu128` builds (2026-11-11) | Superseded: `TransformersBackend.runtime` fingerprint | current code |
| Do not let fixtures supply what the real model lacks (`pad_token_id`, 2026-11-11) | Pad and EOS IDs taken from the pinned tokenizer files and recorded; fixtures leave `pad_token_id` unset | test |
| Test device boundaries on the real GPU path (2026-11-14, 2026-11-19) | Test 23: the cached `u` and `logα` live on the model device; a meta-device fixture fails if they do not; ticket 15 runs the real path | test |
| Treat malformed lineage as unresolved (2026-11-11) | Test 12 also feeds a malformed exclusion JSONL line and expects a refusal, not "no overlap" | test |
| Check whole-group feasibility for matched controls (2026-11-18) | Transplant control uses `match_group` | reuse |
| Separate a verified response file from a verified sampling policy (2026-11-18) | Flip tables vs GRPO arms use their greedy responses and keep the catalog's policy evidence labels | report label |
| Zero-signal fixtures must not become numerical failures (2026-11-19) | At checkpoint 0 every read change is zero: norms, ratios and Jaccard of empty sets are defined and tested, not divided by zero | test |
| A sealed manifest with no expected hashes verifies nothing (2026-11-19) | The input bundle (gold plan, order, C outputs, validation IDs) requires the full hash set | reuse (`c_attribution_bundle`) |
| Round-trip results through the actual serializer (2026-11-20) | `attribution_units._encode`; tests read back sealed reports | reuse |
| Record exposure from the artifact that recorded the use (2026-11-22) | Cohort150 and validation prior uses come from sealed evaluation units | reuse (`register`) |
| Report magnitudes beside signed averages; keep controls visible (2026-11-11) | Activation shift reports mean and mean-absolute; every overlap shows its layer-matched null | spec |
| ADR 0021 IEEE runtime; ADR 0001 / 0005 frozen protocol and flexible-v3; ADR 0004 OOM stops without fallback | IEEE precision set and read back; protocol unchanged; an OOM in qualification stops, with no automatic batch or precision change | reuse |

Not relevant here: PEFT `disable_adapter` unfreezing (no PEFT in training; gold's LoRA is only read on CPU), `model.double()` not being FP64 (no FP64 model reference is planned), watchdog deadlines (the C runner's no-cap mode is reused as is), and the Colab-specific stderr and preinstalled-package lessons (the run is on a Pod).

## Testing decisions (CPU, tiny Qwen2, no download)

1. Off-switch and zero-coefficient forward equal the base MLP bit-for-bit in FP32; logits equal across a whole tiny model.
2. Shapes and zero-based indexing: perturbing `b[j]` in layer l changes only neuron j's gate pre-activation in layer l.
3. Only `a`, `b`, `logα` require grad; base hash unchanged after a step (mutation: unfreezing a base weight fails).
4. Gradients: at init `a` gets zero gradient (b = 0) and `b` gets nonzero; `logα` gets only the L0 gradient; after one step all three are nonzero (mutation: detaching `z` fails).
5. Hard-concrete equations against hand-computed values; expected-active vs Monte-Carlo frequency within a declared sampling tolerance; freeze maps `logα > −ln 11` to 1.
6. Three counts and the export agree with an independent recomputation from saved `logα`.
7. Step loss equals an independent per-token CE sum over the step's supervised tokens with variable lengths (mutation: per-example mean fails).
8. First 64 steps consume gold's order; later orders match the frozen 1,024-step plan.
9. Clipping touches reads only; mask LR follows the shared schedule.
10. Checkpointed vs non-checkpointed backward identical for a fixed noise seed.
10a. Shared-mask equivalence (D14): the summed read and `logα` gradients of 4 single-example passes with one cached `u` equal one 4-row forward with that `u`, within FP32 round-off; a mutation that re-draws `u` per pass fails; the logged group boundaries match the step order.
10b. Rank 0 and rank 1 share one code path apart from the parameterisation: a rank-0 fixture with its rows set to `b_i a` reproduces a rank-1 fixture's forward and gradients with respect to the mask exactly.
11. Uninterrupted vs resumed (at 8, at 960 and at a fixed-mask step) give identical factors, masks and logs.
12. Validation sampler refuses an unreadable exclusion source and never returns an excluded ID.
13. Comparison: layer-matched null preserves per-layer sizes; overlap on a planted fixture; missing attribution values excluded, not zero.
14. Dynamics: the logged expected-active, nonzero and survivor counts equal a recomputation from the saved `logα` snapshots; the read-change norm is unchanged under `(a, b) → (k a, b / k)`; churn on a planted pair of mask sets; the figure builder refuses a log with missing steps instead of interpolating.

15. Comparison manifest: a fixture manifest with a third arm and a partial universe runs with no code change; a score outside the declared universe is refused; a planted overlap is recovered.
16. Standalone analysis: settle step on a planted trace; participation ratio on planted rank-1 and rank-3 rows; `a`-cosine is sign-free and gauge-invariant; flip tables on planted responses.
17. Causal checks (if D23 is approved): revert-all equals base logits bit-for-bit on the tiny model; the transplant control keeps per-layer counts and moves exactly `|S|` rows.

18. Settle diagnostics: Jaccard and expected-active change equal a recomputation from planted gate traces; the first check is at 256 and then every 64; the training trajectory is bit-identical with the diagnostics on or off (they never change the run).

19. Carried-over checks: the new loop refuses non-FP32 weights, a context overflow, a wrong base hash, and enables `use_reentrant=False` checkpointing and `foreach=False`, exactly as `sampling_sft_training_backend.train` does (one red case per check).
20. Real-path CPU chain: plan → preflight → train (3 steps, rank 1 and rank 0) → checkpoint → evaluate → seal → export → restore, on a saved tiny Qwen2 and the real launch-record YAML; only CUDA counters, volume, Hub and stop are stubbed.
21. Storage policy (D24): every checkpoint keeps factors and `logα`; only the latest two keep optimizer and RNG; resume from an older checkpoint is refused with a clear message; the planned disk total is computed from shapes and checked by preflight.
22. Pad and EOS IDs come from the tokenizer files; a fixture with `pad_token_id` in the model config is rejected as unrepresentative.
24. Precision: a run-path test fails if `torch.backends.cuda.matmul.allow_tf32` or `cudnn.allow_tf32` is assigned anywhere, and preflight refuses unless every `fp32_precision` reads `ieee`.
25. Extended checks keep old behaviour: `noop_check` and `HFTrainingEngine.preflight` give identical results for their existing callers (existing tests unchanged), and `noop_check` on a rewired fixture fails when the rewire parameters change inside the context.
23. Device placement: cached `u`, `logα` and the rewire factors are on the model device; a meta-device fixture fails on any CPU tensor in the forward.

GPU qualification (ticket 15) adds: real longest example forward/backward and peak memory, zero-update logits identity on the real model, a disposable few-step run (not the scientific run), throughput, save/restore.

## Out of scope

λ sweeps, rank 4 runs, second seeds, per-example mask ablation, objective comparisons, sampled evaluation, causal checks beyond D23, any change to existing arms, the C-attribution records or C-attribution ticket 14's held-out pool.

## Open decisions blocking a launch record

Decided by the researcher on 2026-11-23: D5 (1,024 steps, freeze at 960), D6 (λ 6e-4, no sweep), D12 (cohort150 evaluation only), D13 (rank 1 and rank 0), D18 (one A100-80GB), D19 scope (C and production-1, GRPO first, generic code), D20 cadence (cheap metrics every step, validation CE every 16), D22 scope.

Proposed and used as the working plan, but not yet explicitly signed off: D1 (read LR 3e-3 rank 1, 3e-4 rank 0), D2–D4 (Max's mask optimizer, betas, read-only clipping), D7 (epoch order), D8–D10 (FP32 IEEE, gold target contract, seeds), D14 (microbatch 1 with one mask per 4 examples), D15 (checkpointing), D16 (clean-room in `pilot_eval/`), D17 (implementation on `main` after the attribution branch merges), D21 details, D22 definitions (settle step, effective rank), D24 (checkpoint storage), and D25 (an ADR for the A100 departure from ADR 0018, ticket 20).

Blocked on data, not on a decision: D11's validation set needs the `sequences-v2` question IDs and gold's saved run records (ticket 01, needs the artifact root); the comparison needs the `c-production-1` and production-1 output files located (ticket 01).

Not approved: D23 causal checks (ticket 19), about 2–3 h more GPU. See `decision-log.md`.
