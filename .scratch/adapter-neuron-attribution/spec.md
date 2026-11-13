# Adapter and neuron attribution — review draft v1 (first pass)

**Status:** needs-triage. Draft for adversarial review. Nothing here approves a GPU run, a threshold or a budget. Proposed values are marked **[proposed]** and collected in `candidate-plan-v1.yaml`.

Human-readable companion: Claude Doc "D10 · Audit · Adapter and neuron attribution". This file is the review baseline; where they differ, this file wins.

This is a deliberately small first pass. It answers the three primary questions once, with one validity check, and leaves follow-ups (bands, cumulative curves, more neuron sizes, more arms in generation, gold-trace robustness) to a later round targeted at what it finds.

## Problem statement

The joint neuron and adapter audit (`.scratch/joint-neuron-adapter-audit/`) is descriptive: it measures activations, reads and writes, not what each arm's behaviour depends on. The Stage 0 effective-write heatmaps show where each arm writes, not which adapters are necessary. We need, per arm, which rank-1 adapters (and which neurons feeding a `down_proj` adapter's read) carry the accuracy effect and, for Gold SFT, the length collapse, with first-order estimates checked against real ablation.

## Questions

Primary, preregistered:

1. **Separability (Gold SFT).** Do different adapters carry the accuracy effect and the length effect?
2. **Sharing across arms.** Do arms rely on the same adapters? Do Sampling SFT k0/k2/k5 fall between Gold SFT and the GRPO arms, ordered by k?
3. **Dense or sparse.** How many neurons carry each arm's neuron-to-adapter attribution at `down_proj`, and does a 64-neuron read keep the top adapter's effect?

Non-goals: causal claims about the objective alone (arms differ in data and trajectory too); new training; base-model gradients.

## Arms and sources

- Arms: step-64 `gold-sft`, `online-grpo`, `offline-grpo`, `sampling-sft-k0`, `sampling-sft-k2`, `sampling-sft-k5`; untuned model as reference.
- Sources: the joint audit's sealed `inventory-v2`, reused unchanged. An arm it marks unresolved is reported missing, not zero.
- Rank 1, alpha 1, dropout 0, seven projections × 28 layers = 196 adapters per arm. Dimensions read from sealed metadata.
- Neuron-level results assume a fixed read (`a ≈ a₀`, from post hoc Stage 0 `scripts/neuron_audit_adapter_init_check.py`). Modules where `a` moved are reported separately.

## Data

| Set | Source | Size | Use |
| --- | --- | --- | --- |
| Discovery | `sequences-v2` math rows, `discovery` split (GSM8K train, leakage-screened) | 200 | Attribution and all selection |
| Confirmation | `sequences-v2` math rows, `held_out` split | 100 | Teacher-forced real ablation only |
| Generation | ADR 0001 GSM8K test cohort (150 items, seed 42) | 150 | Generation check, flexible v3 |

Why: not training questions (Gold SFT saw these gold solutions; attribution there partly measures memorization); select on discovery, test on confirmation (avoids selection inflating effects); generation on the ADR 0001 cohort (paired with every existing pilot number). Cleanliness status is inherited from `sequences-v2`.

**Teacher-forced text: base-correct traces.** The untuned model's greedy response to each discovery and held-out prompt (pilot protocol: greedy, `max_new_tokens` 1024), kept where `score_gsm8k_flexible_v3` marks it correct and uncapped, tokenized once with the frozen template. Same token IDs for every arm. If fewer than **[proposed]** 120 discovery or 60 confirmation rows remain, stop and decide. Gold traces are a later robustness check, not in this pass.

## Metrics

Two kinds. **Behaviour** (only in Stage D): v3 accuracy and response length from real generation. **Proxies** (Stages A–C): cheap, teacher-forced, differentiable, so attribution is possible; Stage D checks they track behaviour.

- **M_acc(e):** mean log p(next token) over `response_content`. Proxy for accuracy.
- **M_len(e):** mean of log(p(`<|im_end|>`) + p(first token of `####`)) over `response_content` positions excluding the final answer line and the last **[proposed]** 5 content tokens. Proxy for early stopping.
- **KL(e):** mean per-token KL(full arm ‖ ablated) over `response_content`. Size of an ablation's effect when M_acc and M_len barely move.
- **Generation:** v3 accuracy (ADR 0005, unchanged), mean response tokens, capped rate, early-stop rate = share of responses shorter than the untuned model's 10th-percentile length on the cohort **[proposed]**.

Equal weight per example. Attribution sign: predicted change in M when the component is removed.

## Method

### Stage A+B — attribution, one merged pass (discovery)

Per arm: one forward over the 200 discovery rows, then one backward of Σ_e M_acc(e) and one of Σ_e M_len(e) (graph retained between them). 6 forwards and 12 backwards in total, batched.

Adapter attribution, all 196 adapters, from hooks on each live LoRA branch output Δy⁽ⁱ⁾ = s·B(Ax):

    attr_i(e) = − Σ_t ⟨ ∂M(e)/∂Δy⁽ⁱ⁾_t , Δy⁽ⁱ⁾_t ⟩

Neuron-to-adapter attribution, all 28 `down_proj` adapters, from z_t (pre-hook on the PEFT-wrapped `mlp.down_proj`, as in `NeuronCollector`) and c_t (the `lora_A` output):

    attr_j(e) = − a_j Σ_t z_{j,t} ∂M(e)/∂c_t

This estimates removing neuron j's term from the adapter's read only; the neuron's native write is untouched. Both are gauge-invariant under (a, b) → (k a, b/k). Identity enforced per example: Σ_j attr_j = attr_i for each `down_proj` adapter.

Frozen weights; `requires_grad_(True)` on the embedding output so a graph exists; `retain_grad()` on hooked tensors; per-example values kept (sum-not-mean backward). z for 28 layers is held for one batch only and reduced in the backward hook.

### Selection freeze (discovery only, before any confirmation row is read)

Per arm and metric, freeze and hash:
- Top 10 adapters by |mean attr| (ties: layer, then module order).
- For each of the top 5: one write-matched random adapter (same module type, mean write ‖s b‖·mean|c_t| within ±20% **[proposed]**, seeded; unavailable if none). A write-matched random set of 10 for the group condition.
- The arm's top `down_proj` adapter (by |mean attr| on that metric) and its neuron rankings: by attr_j, by term size |a_j z_j| (joint audit list B), and 3 seeded random draws.

### Stage C — teacher-forced real ablation (confirmation)

Removal sets s·b = 0 for the chosen adapters (in-memory hook, checked equal to an edited checkpoint). Per arm and metric:

| Condition | Count |
| --- | --- |
| Top 5 removed, one at a time | 5 |
| Matched random removed, one at a time | 5 |
| Top 10 removed together | 1 |
| Matched random 10 removed together | 1 |

24 per arm, 144 total.

Neuron level, at each arm's top `down_proj` adapter (per metric): rebuild c_t from 64 neurons' terms, rescaled so mean |c_t| on discovery matches the full read, then score. Rankings: attribution, term size, 3 random. Plus constant read c_t → c̄ (pure steering limit). 6 per arm and metric, 72 total.

Each condition is one forward over the confirmation rows; record ΔM_acc, ΔM_len, KL per example.

### Stage D — generation check (ADR 0001 cohort)

Arms: `gold-sft` and `online-grpo`. Conditions: full arm; top-10 M_acc adapters removed; top-10 M_len adapters removed; matched random 10 removed. 8 runs × 150 items. The untuned run is reused from existing pilot results if its protocol and runtime match (else one more run). Edited adapters are new PEFT checkpoints with chosen b zeroed, verified by hash and by logits against the hook. Pilot protocol decoding; config and all responses saved as JSONL.

## Controls

| Control | Rules out |
| --- | --- |
| Write-matched random adapters (singles and group) | "Any adapter that writes this much matters" |
| Term-size and random neuron rankings | "Any 64 neurons, or just the big terms, would do" |
| Predicted vs real (top-5 + random singles) | "The first-order estimate is wrong" |
| Finite differences on the real model (qualification) | Wrong gradient wiring |
| Untuned model with hooks → all attributions 0 | Pipeline artefacts |
| Attribution vs write size across 196 | "Attribution restates the heatmaps" (descriptive) |

No seed replicates exist; arm differences are not compared against a run-to-run noise floor.

## Statistics

- Unit: question. Paired bootstrap, the same resampled questions across arms and conditions, 10,000 resamples, seed 1234 **[proposed]**.
- **Validity gate** (read first): Spearman between predicted and real ΔM over the 10 singles, pooled across arms per metric (60 points), ≥ 0.5 **[proposed]**, and sign agreement for the top-10 group in every arm. If it fails, primaries 1 and 2 rest on Stage C effects only, and attribution rankings are reported as unvalidated.
- **Primary 1** (Gold SFT): Spearman ρ(attr_acc, attr_len) across 196 with CI, and top-10 overlap. Supported if the CI upper bound < 0.3 **[proposed]**, overlap ≤ 3, and removing the top-10 M_len set lowers M_len more than the random 10 while M_acc stays within its CI; Stage D should show length rising more than under random removal. Contradicted if the CI lower bound > 0.6.
- **Primary 2:** Spearman between arms' 196-vectors per metric. Supported if ρ(online-grpo, offline-grpo) exceeds ρ(gold-sft, either GRPO arm) with non-overlapping CIs and ρ(sampling-sft-k, gold-sft) decreases with k.
- **Primary 3:** per arm, neurons holding 50% and 90% of Σ|attr_j| at each `down_proj` layer (shuffled-index baseline) and cancellation Σ|attr_j| / |Σ attr_j|. Sparse if the 64-neuron attribution read keeps ≥ 70% **[proposed]** of the top adapter's real effect and beats random and term-size; dense if it keeps < 30% in every arm (kill condition for a neuron-level story at rank 1).
- Everything else is exploratory and labelled so.

## Precision

As the joint audit: FP32 forward and backward, TF32 off, autocast off, eager attention, backend recorded; attribution products accumulated in FP64, stored FP32. Whether A+B and C may use TF32 is an open decision (see GPU utilization).

## Outputs and operations

- `runs/adapter-neuron-attribution/<stage>/<run_id>/`, sealed and hashed; one work unit per arm for A+B, per (arm, metric) for C, per condition for D; resume verified by hash.
- Sizes: adapter attribution ≈ 6 × 2 × 200 × 196 × 4 B ≈ 1 MB; neuron attribution ≈ 200 × 28 × 8,960 × 4 B ≈ 200 MB per arm and metric, ≈ 2.4 GB total (`BlockStore`).
- Pod, preflight and verified export reuse `neuron_audit_pod` and `neuron_audit_export`.

## Run tiers

Three tiers, each a gate for the next. Outputs of tiers 1 and 2 are labelled rehearsal and can never feed production.

1. **CPU tests** (local): every ticket's fixture tests, tiny random Qwen2 + PEFT, no downloads.
2. **Colab bug-finding notebook** (`notebooks/adapter-attribution-colab.ipynb`, ticket 05a; A100 40 GB or whatever Colab gives): the real model and real adapters on a small covering subset — `gold-sft` and `online-grpo`, 16 discovery and 8 confirmation base-correct rows, 10 generation items. Runs every stage end to end once: real-model finite differences and Σ_j attr_j identity, hook-vs-checkpoint equivalence, 3 Stage C conditions, 2 short generation runs, the GPU utilization sweep below, and an evidence upload. Its purpose is to find wiring, environment and shape bugs cheaply; it qualifies nothing.
3. **Pod** (Runpod A100 80 GB, ticket 05b then 06–08): preflight, qualification on the production device, a small Pod rehearsal with an interrupted resume (as in the joint audit), then the unattended production chain, verified export, Pod stop.

The joint audit's Colab and Pod rehearsal lessons (`docs/agents/experiment-lessons.md`) apply from the start: stream child stdout/stderr into the notebook; uninstall unused preinstalled packages before pinning torch; take `pad_token_id` from the pinned tokenizer config, not a fixture; bind resumable stages to the code commit; make rehearsal subsets cover every group later stages read; check container disk ≥ 60 GB free before installing.

## GPU utilization

The joint audit probed token budgets only up to 32k padded tokens, and the pilot evaluations generated at batch 1–2 (ADR 0004). On an 80 GB A100 that likely leaves most of the device idle. This study measures utilization explicitly and sizes batches from the measurements.

**What is measured, every stage, every run tier** (`pilot_eval/attribution_gpu.py`, ticket 05a):
- An NVML sampler thread at 1 Hz: GPU busy %, memory used, power. Summary per stage: median and 10th-percentile busy %, peak memory, share of samples below 50% busy.
- Throughput: real (unpadded) and padded tokens per second; padding fraction; generated tokens per second for Stage D.
- Model FLOP utilization (MFU): achieved FLOP/s ÷ device peak for the active precision. Forward ≈ 2·N FLOP per token, forward + backward ≈ 6·N, with N the parameter count from the model config. Busy % alone is not enough: a GPU can be 100% busy and still run at a few percent of peak.
- One `torch.profiler` trace of a steady-state batch per stage, to show CPU stalls (hook reductions, `.cpu()` syncs, block writes) if busy % is low.

**Expected ceilings (to state, not to hide).** A100 dense peak is roughly 19.5 TFLOP/s FP32 without TF32, 156 with TF32, 312 BF16. The joint audit's FP32/TF32-off policy therefore caps MFU at ~6% of BF16 peak before any inefficiency. At full FP32 utilization a 1.5B forward is ~6k tokens/s, so A+B and C are minutes, not hours; Stage D decoding is memory-bandwidth bound and is where batch size matters most.

**Levers, in order, none of which changes a scientific setting:**
1. **Token budget from a sweep, not a fixed cap.** Probe forward-only (Stage C) and forward + backward (A+B) at 16k, 32k, 64k, 128k, 256k padded tokens with length-bucketed packing; pick the largest with peak memory ≤ 80% of the device and no OOM on the longest-sequence stress batch **[proposed]**.
2. **Keep reductions on the GPU.** Per-example attribution sums and the Σ_j identity check stay on device; one async copy to pinned host memory per batch; block writes on a background thread. No per-layer `.item()` or `.cpu()` inside hooks.
3. **Load once.** Base model once; all six adapters via PEFT multi-adapter, switched with `set_adapter`; Stage C conditions switch hooks without reloading.
4. **Stage D batch size from a sweep** at 16, 32, 64, 150 sequences (length-sorted, left padding, per-sequence stop at EOS), choosing the largest that fits. Full-arm and ablated runs use the same batch setting, so within-study comparisons stay paired. Batch composition can change greedy outputs slightly (ADR 0004); this study's full-arm rerun is the reference, and its difference from the existing pilot numbers is reported.

**Targets [proposed], checked in 05b before production:** median busy ≥ 85% in A+B and C; peak memory 50–80% of the device at the chosen budget; padding fraction ≤ 15%. A miss is investigated with the profiler trace before production, not waived silently.

**Precision trade-off (open decision, not assumed).** TF32 would raise the throughput ceiling ~8×. Attribution and ablation compare whole-adapter effects, not tiny paired differences, so TF32 may be acceptable for A+B and C. Proposed: 05b measures attribution and ΔM on 20 rows under FP32 and TF32; the user decides. Stage D stays FP32 unless decided otherwise, because greedy generation is sensitive to numerics and is compared with FP32 pilot results.

## Compute (estimate; qualification replaces it)

At measured-not-assumed utilization: A+B and C together are likely under 1 GPU-h; D depends on the generation batch size (batch 1–2: several hours; batch 64+: plausibly under 1 h). Colab tier ≈ 1 h. The user sets a hard wall-time cap after 05b.

## Testing decisions

CPU, tiny random Qwen2 + PEFT, no downloads, red/green TDD:
- Hooks attached, nothing removed → logits identical to unhooked.
- Untuned → all attributions exactly 0.
- Finite differences for 5 adapters and 3 neurons.
- Σ_j attr_j = attr_i per example.
- Batch size 1 vs 8 identical per example.
- Gauge k ∈ {−2, 0.5} leaves attributions unchanged.
- Mutations (wrong layer, wrong sign, mean-not-sum backward) each fail a test.
- Removal hook == edited checkpoint; 64-neuron read with all neurons == full; constant read writes s·b·c̄; rescale factor correct.
- M_len mask excludes final answer line, padding, template and EOS positions; `####` tokenization fixture.
- Selection freeze refuses confirmation rows; matched sampler deterministic, returns unavailable rather than loosening.

## Deferred to a later round

Gold-trace robustness; cumulative top-1..10 curves; layer bands (necessity and sufficiency); other neuron sizes (8, 512) and the drop-top-m direction; more `down_proj` layers per arm; per-example spot checks; generation for the other four arms and for neuron conditions; drift on non-math text; head-level `o_proj` attribution.

## Open decisions

1. Approve [proposed] values.
2. Reuse the existing untuned generation results, or regenerate under the new runtime?
3. Freeze written predictions before attribution results are inspected (optional, as in the joint audit)?
4. TF32 for A+B and C after the 05b comparison, or FP32 throughout?
5. Approve the utilization targets and the token-budget and generation-batch sweeps.
