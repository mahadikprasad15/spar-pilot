# Adapter and neuron attribution — review draft v2 (first pass)

**Status:** needs-triage for execution; CPU implementation of tickets 01–05 approved by the user (2026-11-13). Draft v3 after review rounds 1 and 2 (`review-round-1.md`, `review-round-2.md`; dispositions in `review-response.md`, `review-response-round-2.md`). Nothing here approves a GPU run, a threshold or a budget. Proposed values are marked **[proposed]** and collected in `candidate-plan-v1.yaml`.

Human-readable companion: Claude Doc "D10 · Audit · Adapter and neuron attribution". This file is the review baseline; where they differ, this file wins.

This is a deliberately small first pass: it answers the three primary questions once, with explicit validity checks, and defers everything else (see §Deferred).

## Problem statement

The joint neuron and adapter audit is descriptive. The Stage 0 write heatmaps show where each arm writes, not what it needs. This study estimates, per arm, which rank-1 adapters (and which neuron terms in a `down_proj` adapter's read) its trace log-likelihood and its stop pull depend on, checks those first-order estimates against real removal, and checks the proxies against generation for two arms.

## Questions

Primary, preregistered:

1. **Separability (gold-sft).** Does removing the adapters that carry the stop pull reduce it beyond matched controls while leaving trace log-likelihood within a predeclared margin?
2. **Sharing across arms.** Are the two GRPO arms' attribution vectors more alike than either is to gold-sft, and do the Sampling SFT arms order by k in their similarity to gold-sft?
3. **Dense or sparse.** At each arm's top `down_proj` adapter, how much of the adapter's real effect does its natural 64-term read keep, compared with term-size and random 64-term reads?

Scope of every teacher-forced claim: questions the untuned model solves under flexible v3 and pilot decoding, read as that model's own generated tokens. Not all GSM8K questions, not newly acquired ability. Non-goals: claims about the objective alone (arms differ in data and trajectory); new training; base-model gradients.

## Arms and sources

- Arms: step-64 `gold-sft`, `online-grpo`, `offline-grpo`, `sampling-sft-k0`, `sampling-sft-k2`, `sampling-sft-k5`; untuned reference.
- Sources: the joint audit's sealed `inventory-v2`, reused unchanged. Unresolved arms or modules are reported missing, never zero.
- 196 rank-1 adapters per arm (7 projections × 28 layers; rank 1, alpha 1, dropout 0), dimensions from sealed metadata.
- Neuron attribution uses each checkpoint's current saved `a`; it does not depend on whether `a` moved in training. The post hoc Stage 0 init check's raw saved-factor cosine and drift (no sign alignment) may be quoted as a historical description only, labelled gauge-dependent: it is meaningful because a₀ and the trained `a` share one training parameterization, not because it is invariant.

## Data and provenance

| Set | Source | Size | Use | Visible before freeze? |
| --- | --- | --- | --- | --- |
| Calibration | `sequences-v2` calibration math rows | 25 | Colab and Pod rehearsal, sweeps, tolerance setting | Yes |
| Discovery | `sequences-v2` math, `discovery` | 200 | Attribution, all selection | Yes (attribution only) |
| Confirmation | `sequences-v2` math, `held_out` | 100 | Real ablation | Traces built blind; no model output computed or viewed before the freeze |
| Rehearsal generation | GSM8K test items outside the ADR 0001 cohort, seed 42 | 20 | Batch qualification, rehearsal | Yes |
| Generation | ADR 0001 GSM8K test cohort | 150 | Stage D | No, until the freeze |

Provenance: per-arm checked sources and unknowns are carried from the `sequences-v2` leakage record into this study's manifest and report. Online-grpo lineage is partial (its frozen plan is not local). Every statement is "no overlap with available sources", never "clean". Prior exposure: the joint audit computed descriptive activation statistics on the held-out rows and the user has seen that report; no quantity of this study was computed on them, and its hypotheses come from Stage 0 and pilot behaviour. The report labels confirmation as partially, not fully, independent.

**Base-correct traces.** For calibration, discovery and held-out prompts: untuned greedy generation under the pilot protocol (`max_new_tokens` 1024, FP32, the qualified generation batch), saving prompt IDs, generated IDs, stop reason and decoded text before any processing. Keep rows where `score_gsm8k_flexible_v3` is correct and uncapped. Teacher forcing uses exactly prompt IDs + generated IDs; no re-tokenization. Accepted and rejected counts, reasons, question length and response length are saved by split. Minimums **[proposed]** 120 discovery and 60 confirmation rows, checked after every mask exclusion; below that, stop and decide.

## Metrics

Behaviour (Stage D only): v3 accuracy, response tokens, capped rate, early-stop rate. Proxies (teacher-forced, differentiable):

**Trace log-likelihood.** With T_e the response-content target positions (every content token, including the first; excluding `<|im_end|>`, trailing newline, prompt, header and padding):

    L_e = (1/|T_e|) Σ_{t ∈ T_e} log p_θ(x_t | x_<t)     (logits read at position t−1)

**Stop pull.** With S the exact stop IDs passed to the generation decoder used in this study (sealed in 01 from the decoder config; for the pilot backend `eos_token_id` only, i.e. S = {`<|im_end|>` = 151645}; `<|endoftext|>` = 151643 is padding and is not a stop event) and P_e the prefix positions t whose next target token x_{t+1} is a content token before the start of the response's last non-empty line (token offsets; a token crossing that newline belongs to the last line):

    S_e = (1/|P_e|) Σ_{t ∈ P_e} log Σ_{s ∈ S} p_θ(s | x_≤t)

Rows with |P_e| = 0 are excluded from stop-pull analyses with a recorded reason; coverage is reported; an empty mean is never zero. Stop pull is a proxy for early termination, not an expected length.

**KL(e):** mean per-token KL(full arm ‖ condition) over T_e's logit positions.

Equal weight per example. Attribution = predicted change in the metric when the component is removed.

## Method

### Stage A+B — attribution (discovery)

**Gate.** Every `lora_B` module gets a forward hook returning g · out with g = 1. In pinned `peft==0.18.0` the branch is Δy = s · lora_B(lora_A(drop(x))) inline, so ⟨∂M/∂Δy, Δy⟩ = ⟨∂M/∂out, out⟩ and the scale cancels.

**Differentiable path.** All model and adapter weights stay frozen. The graph comes from `requires_grad_(True)` on the embedding output (an activation; nothing is trained). The captured `lora_B` outputs and `lora_A` outputs (c_t) are graph tensors. Each metric's gradients are taken with a separate `torch.autograd.grad(Σ_e M(e), captured_tensors, retain_graph=…)` call that returns fresh tensors; `.grad` accumulation is never used.

**Pass 1 (forward + 2 backwards).** One forward over a batch; backward of Σ_e L_e, then of Σ_e S_e (graph retained once). Per example, over every unpadded position (prompt, header, response):

    attr_i(e) = − Σ_t ⟨ ∂M(e)/∂out⁽ⁱ⁾_t , out⁽ⁱ⁾_t ⟩

and for each `down_proj` adapter the scalar g_{t} = ∂M(e)/∂c_t at every unpadded position (c_t = `lora_A` output), stored per metric.

Pass 1 also saves, per example, every adapter's Σ_t |c_t| and token count (for the discovery write ‖s b‖ · mean |c_t| used by matching) and every `down_proj` adapter's signed Σ_t c_t (for c̄), so selection needs no extra run.

**Pass 2 (forward only).** Recompute z_t at each `down_proj` input, one layer at a time, and contract:

    attr_j(e) = − a_j Σ_t z_{j,t} g_t

This is exact (z does not depend on the backward) and avoids holding 28 × tokens × 8,960 activations. Identity checked per example: Σ_j attr_j = attr_i for that adapter. Both quantities are gauge-invariant under (a, b) → (k a, b/k). Neuron attribution is removal of neuron j's term from the adapter read only, not neuron ablation.

Cost per arm: 2 forwards + 2 backwards over discovery. The output head is evaluated in checkpointed position chunks (as in `stage0_push_backend`), since full-vocabulary FP32 logits are ~10 GB at 16k tokens.

### Selection and analysis freeze

Before any model output on confirmation rows or the generation cohort is computed, freeze and hash, per arm and metric:

- Top-10 and top-5 adapters by |mean attribution| (ties: layer, then module order).
- **Matched controls**, per target: same module type; discovery write (‖s b‖ · mean |c_t| over unpadded positions) within ±20% **[proposed]**; sampled without replacement, seeded, from adapters outside that arm's top-10 sets for both metrics. Group control = the union of the per-target matches for that top 10. Unavailable matches are recorded and their targets dropped from paired comparisons; a group with any unavailable member is unavailable. No loosening.
- The top `down_proj` adapter and its neuron orderings: by signed attribution in the direction of the adapter's own predicted effect; by term size |a_j z_j|; 3 seeded random orders.
- c̄ for that adapter: signed, token-weighted mean of c_t over all unpadded discovery positions.
- The full analysis: contrasts, margins, bootstrap settings, Holm families, gate rules, concordance analysis.

### Stage C — real ablation (confirmation)

Removal sets the gate g = 0 (equivalent to s·b = 0; checked against an edited checkpoint). Per arm and metric:

| Condition | Count |
| --- | --- |
| Top-5 adapters removed singly | 5 |
| Their matched controls removed singly | ≤ 5 |
| Top-10 removed together; matched group removed together | 2 |
| Full removal of the top `down_proj` adapter (if not already among the singles) | ≤ 1 |
| Natural 64-term read at that adapter: by attribution, by term size, by 3 random orders | 5 |
| Constant read c_t → c̄ at every unpadded position | 1 |

At most 19 per arm and metric; ≤ 228 forward passes over confirmation rows in total. Per example: ΔL, ΔS, KL.

Retained effect for the 64-term reads: R = (M_sparse − M_removed) / (M_full − M_removed), unavailable when the 95% interval of M_full − M_removed includes 0 or is non-finite.

### Stage D — generation (ADR 0001 cohort)

| Arm | Conditions |
| --- | --- |
| gold-sft | full; top-10 trace-likelihood removed; its matched group; top-10 stop-pull removed; its matched group |
| online-grpo | full; top-10 trace-likelihood removed; its matched group |
| untuned | reference (new run; the early-stop threshold is recomputed from it) |

9 runs × 150 items, greedy, FP32, the qualified batch manifest, config and all responses as JSONL. Removal conditions run from edited PEFT checkpoints (chosen b zeroed), verified by hash and logits against the gate hook. Capped and invalid outputs are counted as their own outcomes. Item-level differences between this study's full-arm runs and the pilot results are reported.

## Statistics

Unit: question. Paired bootstrap, the same resampled questions across arms and conditions within a cohort, 10,000 resamples, seed 1234, percentile intervals **[proposed]**.

**Validity gate, per arm × metric** (prediction = fixed discovery mean attribution; group prediction = sum of members, labelled first-order): report Spearman, sign agreement, median absolute error and real-on-predicted slope over the top-5 and matched singles, with paired intervals; pooled values are summaries only. Pass: real effect has the predicted sign for ≥ 4 of 5 top singles and for the top-10 group **[proposed]**. This is a limited check on selected removals: a pass does not establish the ordering or magnitudes of all 196 causal effects, and the report says so. Finite differences (06b) check wiring; zeroing checks the nonlocal approximation; both are reported.

**If a gate fails** for an arm × metric: primary 2 comparisons involving it are reported as unvalidated, exploratory; primary 1 rests only on its directly measured contrasts; primary 3 always rests only on the measured 64-term intervention, and attribution concentration is described as attribution, not causal sparsity.

**Primary 1 (gold-sft, confirmation).** For the stop-pull set L and trace set A with matched groups, define per bootstrap resample D_S = ΔS(L) − ΔS(match_L), D_L = ΔL(L) − ΔL(match_L), D_A = ΔL(A) − ΔL(match_A), and r = |D_L| / |D_A|, all from the same resampled questions.
- (i) D_S < 0: its 95% interval excludes 0.
- (ii) The reference exists: the 95% interval of D_A excludes 0. Otherwise primary 1 is inconclusive.
- (iii) Equivalence: the one-sided 95% upper bound of r (95th percentile of the joint bootstrap) is < 0.25 **[proposed margin]**.
Supported if (i), (ii) and (iii). Contradicted if (i) and (ii) hold and the one-sided 95% lower bound of r is ≥ 0.25. Otherwise inconclusive. Stage D for gold-sft (length and early-stop change for L vs match_L, accuracy change) can confirm or fail to confirm; it cannot rescue a failed teacher-forced result. ρ(attr_L, attr_S) and top-10 overlap are descriptive.

**Primary 2.** Spearman on signed 196-vectors (average ranks for ties); arms with an unresolved module excluded and listed. Paired bootstrap of differences: (a) ρ(online, offline) − ρ(gold, online), (b) ρ(online, offline) − ρ(gold, offline); support needs both > 0. (c) ρ(k0, gold) − ρ(k2, gold), (d) ρ(k2, gold) − ρ(k5, gold); ordering needs both > 0. Per metric.

**Primary 3.** Per arm and metric: the paired contrasts R(attribution) − R(term size) and R(attribution) − mean R(random), each with a bootstrap interval; these two contrasts are the tests. Classification is descriptive and uses intervals, never point estimates: sparse at this adapter if the 95% lower bound of R(attribution) ≥ 0.70 **[proposed]**; dense if the 95% upper bound of R(attribution) < 0.30 in every arm; otherwise inconclusive; unavailable R stays unavailable. Descriptive alongside: neurons reaching 50% and 90% of Σ|attr_j| vs a random-order accumulation null (1,000 seeded orders), and cancellation Σ|attr_j| / |Σ attr_j|; both unavailable below a resolution floor set in 06b.

**Families.** Primary family per metric = primary 1's decision (one test: the conjunction above, with its own intervals) + primary 2's four differences + primary 3's two paired contrasts per arm; Holm at α = 0.05 over the family's p-values from the paired bootstrap (two-sided, percentile; for primary 1, the larger of the component p-values). Holm adjusts the tests; the threshold classifications in primary 3 and the margin rule's intervals are reported as computed. Everything else exploratory.

**Concordance (predeclared, exploratory).** For the 6 Stage D removal conditions: sign and magnitude agreement between confirmation ΔL, ΔS and generation Δaccuracy, Δlength, with intervals; disagreements listed. Behavioural claims are limited to generated arms and conditions.

## Precision

FP32 forward and backward, TF32 off, autocast off, eager attention, everywhere in this pass; FP64 accumulation for attribution products. TF32 is deferred.

## Units, outputs and operations

- Units: A+B per (arm, batch); C per (arm, condition); D per (arm, condition, batch). Each sealed with IDs, input, plan, code and runtime hashes; durable status; duplicates refused; resume verified by hash.
- Canonical artifact root is a parameter. Every unit saves config, logs, per-example structured results and a partial manifest.
- This study's manifest, export and restore are built (ticket 05) on the joint audit's `BlockStore` and `neuron_audit_export` primitives and tested on partial states.
- Caps come from the launch record and are enforced cumulatively at unit boundaries, including export time. The Pod is stopped at the end of every run after the verified volume copy (see §Overnight unattended chain).

## Run tiers

1. **CPU** (tickets 01–05): code with fixture tests; no GPU authorization implied.
2. **Colab rehearsal** (06a, needs the user's go-ahead): calibration rows and rehearsal generation items only, for `gold-sft`, `online-grpo` and untuned. Covering set: no-op hook; single removal; group removal; full `down_proj` removal; natural 64-term read; constant read; full vs edited-checkpoint generation on 2 items; a corrupted and an interrupted unit; online upload and readback of this study's partial outputs with child-process logs. Finds bugs; qualifies nothing.
3. **Overnight Pod chain** (06b → 07 → 08 → 09 → 10 → export → Pod stop, run unattended by the chain runner, ticket 11; needs the user's approval of the launch record below). Stages: preflight; qualification on calibration rows (finite differences, batch invariance, sweeps, generation batch qualification); blind base traces for discovery and held-out; attribution and freeze; Stage C; Stage D; report; verified export; Pod stop.

A small Pod rehearsal of the same chain (2 arms, a few units, an interruption with local state deleted and a verified restore, a real Pod stop) runs attended before the first overnight launch, unless the user waives it.

The joint audit's Colab and Pod lessons apply from the start (`docs/agents/experiment-lessons.md`).

## Overnight unattended chain

Precedent: the joint audit's approved unattended 03b → 04 chain (`frozen_plan_v1.approved.yaml`: automatic sign-off rule, cap fixed before launch, overrun rule, automatic Pod stop after verified backup). This study follows it, with one change: the Pod always ends stopped.

**Launch record (approved by the user before launch, sealed with the code commit).** Every [proposed] value becomes a fixed value or a fixed rule: tolerances (below), minimums, sweeps and selection rules, the generation batch rule, the statistics, the per-stage and total wall-time caps, and the Pod ID and stop command. Nothing is chosen at night by a person.

**Automatic sign-off after qualification** if every qualification check passes against the launch record's fixed tolerances and no review flag is raised; recorded as the rule's sign-off, not the user's. Any failure, flag, missing arm, unmet minimum or no feasible token budget stops the chain before production.

**Tolerances fixed before launch** **[proposed values for the launch record]**:
- Real-model finite differences (FP32, central difference on g with ε = 1e-2): median relative error ≤ 0.05 and maximum ≤ 0.20 over checked adapters and neurons.
- Σ_j attr_j = attr_i: relative error ≤ 1e-4 of Σ_j |attr_j|.
- Batch invariance on calibration rows: per-example |Δattr| ≤ 1e-4 × max |attr| in that arm and metric.
- Resolution floor for unavailable R and concentration: 10× the measured batch-invariance discrepancy.

**What does not stop the chain.** Analysis outcomes, including a failed validity gate, unavailable controls or inconclusive primaries, are recorded and reported; they never stop or alter the run.

**Caps and overrun.** Per-stage caps and a total cap from the launch record, enforced at unit boundaries including export time. After 10% of a stage's units and every further 10%, the projected stage time is checked; two consecutive projections above 2× the qualification projection or above the cap end the stage at a safe unit boundary with a verified partial export, and the chain proceeds to report-and-stop.

**End of run, success or failure.**
1. Write a morning summary (`run_summary.md` + JSON): stages completed, gates, primaries, failures with log pointers, utilization, wall time, cost estimate.
2. Copy sealed outputs to the persistent volume and verify by full hash.
3. Export to the private HF dataset with fresh readback; retry 3 times. A failed HF export is recorded but does not keep the Pod alive, because the verified volume copy persists.
4. Stop the Pod with the exercised stop command (try `runpodctl stop pod <id>`, then `runpodctl pod stop <id>`), logging its output.

**Independent watchdog.** At launch, a separate process (outside the chain's Python) sleeps until total cap + 30 min **[proposed]** and then copies whatever is on local disk to the volume and stops the Pod. A hung or crashed chain therefore cannot leave the Pod billing overnight.

## GPU utilization

Measured in every GPU unit by `pilot_eval/attribution_gpu.py`: NVML busy %, memory and power at 1 Hz (summaries need ≥ 30 s of samples), synchronized throughput after warm-up, padding fraction, and an MFU estimate. FLOP model: forward ≈ 2N per token; backward without weight gradients ≈ 2N; excludes attention-score FLOPs and padding; denominator = the measured device's FP32 dense peak. One profiler trace per stage. Busy ≥ 85% and memory 50–80% **[proposed]** are performance-investigation targets, not validity gates and never a reason to change a qualified configuration.

**Token-budget sweep** (06b; first probe in 06a): from 2k padded tokens upward (2k, 4k, 8k, 16k, 32k, 64k **[proposed]**), including singleton and longest-example probes, benchmarking the real path (pass 1 with two backwards, pass 2, writes). OOM aborts that candidate only; failed candidates are recorded; if none meets peak memory ≤ 80% of the device, 06b stops. The frozen budget comes from measured worst-case memory on the production device.

**Generation batch qualification** (06b): batch 1 vs candidates 8, 16, 32, 64 on the 20 rehearsal items through the full 1,024-token cap, longest prompts first, measuring KV-cache peak. A candidate qualifies if v3 correctness flips ≤ 1/20 and stop-reason changes ≤ 1/20 **[proposed]**; otherwise try the next smaller; batch 1 always qualifies. The batch manifest is frozen.

## Testing decisions

CPU, tiny random Qwen2 + PEFT 0.18.0, no downloads, red/green TDD:
- Gate hooks at g = 1 leave logits identical; g = 0 equals an edited checkpoint; hook gate equals an explicit scaled-branch gate with s ≠ 1.
- With every weight frozen, backward succeeds only via the embedding-output graph (a test asserts the frozen model without it has no graph); both metrics' gradients from separate `autograd.grad` calls each match their own finite differences, and running them in either order gives identical results.
- Stop pull with a single stop ID (151645) only; a two-ID configuration is refused unless labelled as a decoding change.
- Untuned (adapters disabled) → all attributions exactly 0; a zero-b adapter → 0.
- Finite differences for 5 adapters and 3 neurons, independent of the hook path.
- A fixture where an adapter affects the loss only through an earlier prompt token: attribution matches its finite difference.
- Σ_j attr_j = attr_i per example; pass 2 recompute equals a stored-z reference on a small model.
- Batch 1 vs 8 identical per example; gauge k ∈ {−2, 0.5}.
- Mutations (wrong layer, wrong branch, wrong sign, mean-not-sum, response-only position sum) each fail a test.
- Trace log-likelihood on a hand-computed sequence with distinct per-position values (first content token included, EOS excluded); stop-pull mask on single-line, `####`, prose and boundary-crossing fixtures; empty mask excluded, not zero.
- Routing and frozen parameters checked after every `set_adapter`.
- Matched sampler: excludes targets and top sets, without replacement, deterministic, unavailable rather than loosened.
- Freeze refuses confirmation or cohort outputs; units refuse duplicates; corrupted and interrupted units detected.

## Deferred

Gold-trace robustness; TF32; rescaled 64-term reads; cumulative top-1..10 curves; layer bands; other read sizes and drop-top-m; more layers per arm; per-example spot checks; generation for the other four arms and for neuron or constant-read conditions; drift on non-math text; head-level `o_proj` attribution.

## Open decisions (execution only; CPU implementation may proceed)

0. Approve the launch record (all values, caps, Pod ID, stop command) before the first overnight run; decide whether the attended Pod rehearsal can be waived.

1. Approve [proposed] values (minimums, tolerance, gate rule, margin, thresholds, sweeps, batch rule).
2. Approve the Colab rehearsal (06a) and a 06b budget.
3. Optional: freeze written predictions before attribution results are inspected.
