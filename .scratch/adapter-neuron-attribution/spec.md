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

As the joint audit: FP32 forward and backward, TF32 off, autocast off, eager attention, backend recorded; attribution products accumulated in FP64, stored FP32.

## Outputs and operations

- `runs/adapter-neuron-attribution/<stage>/<run_id>/`, sealed and hashed; one work unit per arm for A+B, per (arm, metric) for C, per condition for D; resume verified by hash.
- Sizes: adapter attribution ≈ 6 × 2 × 200 × 196 × 4 B ≈ 1 MB; neuron attribution ≈ 200 × 28 × 8,960 × 4 B ≈ 200 MB per arm and metric, ≈ 2.4 GB total (`BlockStore`).
- Pod, preflight and verified export reuse `neuron_audit_pod` and `neuron_audit_export`.

## Compute (estimate; qualification replaces it)

A+B ≈ 1 GPU-h; C ≈ 1 GPU-h (216 forward passes over ≤ 100 rows); D ≈ 2–3 GPU-h (8 HF FP32 generation runs). The user sets a hard wall-time cap after qualification.

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
