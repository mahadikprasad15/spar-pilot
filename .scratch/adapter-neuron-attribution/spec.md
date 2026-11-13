# Adapter and neuron attribution — review draft v1

**Status:** needs-triage. Draft for adversarial review. Nothing here approves a GPU run, a threshold or a budget. Proposed values are marked **[proposed]** and collected in `candidate-plan-v1.yaml`.

Human-readable companion: Claude Doc "D10 · Audit · Adapter and neuron attribution". This file is the review baseline; where they differ, this file wins and the doc is updated.

## Problem statement

The joint neuron and adapter audit (`.scratch/joint-neuron-adapter-audit/`) is descriptive: it measures activations, reads and writes, but not what each arm's behaviour depends on. The Stage 0 effective-write heatmaps show where each arm writes, not which adapters are necessary. We need, per arm, which rank-1 adapters (and which neurons feeding a `down_proj` adapter's read) carry the accuracy gain and, for Gold SFT, the length collapse, with estimates checked against real ablation.

## Questions

Three primary questions, preregistered; everything else is secondary and exploratory.

1. **Separability (Gold SFT).** Do different adapters carry the accuracy effect and the length effect?
2. **Sharing across arms.** Do arms rely on the same adapters? Do Sampling SFT k0/k2/k5 fall between Gold SFT and the GRPO arms, ordered by k?
3. **Dense or sparse.** At selected `down_proj` layers, how many neurons carry each arm's neuron-to-adapter attribution, and how much do positive and negative terms cancel?

Non-goals: causal claims about the training objective alone (arms differ in data and trajectory too); any new training; base-model gradients (`G = dL/dW` at step 0), which belong to a separate ticket set.

## Arms and sources

- Arms: untuned baseline (reference only); step-64 `gold-sft`, `online-grpo`, `offline-grpo`, `sampling-sft-k0`, `sampling-sft-k2`, `sampling-sft-k5`. Same names as the joint audit.
- Sources: the joint audit's sealed `inventory-v2` (model, tokenizer, template, adapter payloads, configs, scales, a₀). Reuse it; do not re-inventory. An arm the inventory marks unresolved is unresolved here too, and is reported as missing, not as zero.
- Rank 1, alpha 1, dropout 0, seven projections × 28 layers = 196 adapters per arm, per the inventory. Read dimensions from the sealed metadata, not constants.
- Precondition from post hoc Stage 0 (`scripts/neuron_audit_adapter_init_check.py`): which `down_proj` modules have `a ≈ a₀`. Stage B's neuron interpretation assumes a fixed read; modules where `a` moved are reported separately.

## Data

Reuse the joint audit's frozen `sequences-v2` so token IDs, masks and question splits are identical across both studies.

| Set | Source | Size | Use |
| --- | --- | --- | --- |
| Discovery | `sequences-v2` math rows, split `discovery` (GSM8K train, leakage-screened) | 200 | Attribution; all selection |
| Confirmation | `sequences-v2` math rows, split `held_out` | 100 | Teacher-forced real ablation only |
| Background | `sequences-v2` general rows (UltraChat first turns) | discovery/held-out as frozen | Drift to base only |
| Generation | ADR 0001 GSM8K test cohort (150 items, seed 42) | 150 | Generation check, flexible v3 |

Why disjoint sets:

- **Not training questions.** Gold SFT saw gold solutions verbatim; other arms saw rollouts on their prompts. Attribution there partly measures fit to remembered items and gives Gold SFT a log-probability advantage on gold text. `sequences-v2` already excludes the available training, rollout and selection IDs; its recorded cleanliness status carries over unchanged (unknown where provenance is incomplete).
- **Not select-and-test on the same questions.** Top-k adapters chosen on discovery are measured on confirmation, or the selection effect inflates the result.
- **Generation on the ADR 0001 cohort** so ablated accuracies are paired with every existing pilot number.

Text sets for teacher forcing (same token IDs for every arm):

- **Gold traces:** the frozen `sequences-v2` math rows as they are (GSM8K prompt + reference answer). Matches Gold SFT's training format.
- **Base-correct traces (new):** the untuned model's greedy response to the same prompt (pilot protocol decoding: greedy, `max_new_tokens` 1024), kept only where `score_gsm8k_flexible_v3` marks it correct and not capped. Neutral between arms. Frozen once, with its own manifest and hash, and never regenerated per arm.
- Primary trace set: base-correct. Gold is a robustness check. **[proposed]** If fewer than 120 discovery or 60 confirmation base-correct rows remain, stop and decide (do not silently top up from other splits).

## Metrics

All per example, equal example weighting, positions from frozen masks.

- **M_acc(e)** = mean over `response_content` positions of log p(next token). Higher is better.
- **M_len(e)** = mean over positions P_e of log(p(EOS) + p(first token of `####`)), where P_e = `response_content` positions excluding the final answer line and the last **[proposed]** 5 content tokens. Higher = stronger pull to stop early. The EOS id is the template's end-of-turn token `<|im_end|>`; the `####` tokenization is recorded once in the plan.
- **KL_full(e)** = mean per-token KL(full arm ‖ ablated arm) over `response_content`.
- **Drift(e)** = mean per-token KL(arm ‖ untuned) on background rows; secondary.
- **Generation:** accuracy by `score_gsm8k_flexible_v3` (ADR 0005, unchanged), mean response tokens, capped rate, and early-stop rate = fraction of responses shorter than the untuned model's 10th-percentile length on the same cohort **[proposed]**, frozen from the untuned run before any ablated run is scored.

Sign conventions: attribution = predicted change in M when the component is removed. Negative attribution on M_acc → component needed for fit; negative on M_len → component carries the early-stop pull.

## Method

### Stage A — adapter attribution (discovery)

Each adapter i has an implicit gate g_i = 1 on its branch output Δy⁽ⁱ⁾ = s·B(A x). Removing it is g_i: 1 → 0. First-order estimate per example:

    attr_i(e) = − Σ_t ⟨ ∂M(e)/∂Δy⁽ⁱ⁾_t , Δy⁽ⁱ⁾_t ⟩

Implementation: forward hook on each LoRA branch output (the `lora_B` output times scale, from the live PEFT module) that saves the tensor and calls `retain_grad()`. All base and adapter weights frozen; `requires_grad_(True)` on the embedding output so a graph exists. Backpropagate Σ_e M(e); reduce over tokens and hidden size keeping the example axis. One backward per (arm, metric, trace set).

Gauge: Δy is the physical write, so attr_i is invariant to (a, b) → (k a, b/k).

Layer selection for Stage B (frozen before Stage B runs): per arm and metric, rank `down_proj` adapters by |mean attr|; take the union over arms and both metrics of each arm's top 2, capped at 5 layers by total |attr| **[proposed]**. If an arm's top adapter overall is `o_proj`, record it; head-level o_proj attribution is secondary and only if the 1B head collection exists.

### Stage B — neuron-to-adapter attribution (discovery, selected layers)

At a selected `down_proj` adapter: z_t from the PEFT-wrapped `mlp.down_proj` pre-hook (as in `NeuronCollector`), c_t from the `lora_A` output with `retain_grad()`:

    attr_j(e) = − a_j Σ_t z_{j,t} ∂M(e)/∂c_t

This estimates removing neuron j's term from the adapter read only; the neuron's native write through W_down is untouched. It is not a neuron ablation. Gauge-invariant (a → k a scales ∂M/∂c by 1/k). Exact identity, enforced: Σ_j attr_j(e) = attr_i(e) for that adapter, within the plan's tolerance.

Can share Stage A's backward pass if the layers are known; default is a second pass after the layer freeze.

### Selection freeze (between B and C)

From discovery only, freeze and hash: per (arm, metric, trace set) the top 10 adapters by |mean attr| (ties by layer then module order), write-matched random controls, bands, Stage B neuron rankings and masked-read conditions. Confirmation rows are not read before this freeze (same rule as the joint audit's 05).

### Stage C — teacher-forced real ablation (confirmation)

Ablation sets s·b = 0 for the chosen adapters on the live model (in-memory hook; checked equal to an edited checkpoint).

Per arm and metric:
1. Top 10 singly.
2. Top 1..10 cumulatively; interaction = cumulative − Σ singles.
3. Write-matched random adapters singly and as a group of 10 (see Controls).
4. Bands: layers 11–13, 15–17, 20–24 (all modules); all `down_proj`; all `o_proj`. Necessity (remove band) and sufficiency (keep only the band on the untuned model).

Neuron level at Stage B layers (c_t rebuilt from a masked read):
- Necessity: drop top-m terms, m ∈ {8, 64, 512}.
- Sufficiency: keep top-m terms, rescaled so mean |c_t| on discovery matches the full read.
- Rankings: attribution (per metric), term size |a_j z_j| (joint audit list B), 3 seeded random draws.
- Constant read: c_t → discovery mean c̄ at that adapter (pure steering limit).

Record ΔM_acc, ΔM_len, KL_full per example.

### Stage D — generation (ADR 0001 cohort)

Per arm: full; untuned; top-10 M_acc adapters removed; top-10 M_len adapters removed (sampling and gold arms); 10 write-matched random removed; best necessity band removed; best sufficiency band alone; constant c̄ at all `down_proj` adapters. "Best" = largest discovery-predicted effect, chosen at the selection freeze. Edited adapters are written as new PEFT checkpoints with chosen b zeroed and verified by hash and by logits against the in-memory hook. Constant c̄ cannot be a LoRA checkpoint; it runs through a hooked HF generation path. Generation uses the pilot protocol (greedy, 1024 tokens) and writes config + all responses as JSONL (AGENTS.md).

## Controls

| Control | Rules out | Definition |
| --- | --- | --- |
| Write-matched random adapters | "Any adapter writing this much matters" | For each top-10 adapter, sample from the same module type with mean write ‖s b‖·mean|c_t| on discovery within ±20% **[proposed]**; 3 seeded draws; report unavailable if none qualify |
| Attribution vs write size | "Attribution restates the heatmaps" | Rank regression of |attr| on write size across 196; report residual outliers |
| Shuffled neurons | "Any m neurons carry this share" | Permute neuron indices per example; 1,000 draws **[proposed]** |
| Term-size ranking | "Important = big terms" | Masking curves under |a_j z_j| ranking vs attribution ranking |
| Shared-a₀ floor | "Arms overlap because a₀ is shared" | Cross-arm overlap under term-size ranking; only excess counts |
| Predicted vs real | "First-order estimate fails" | Spearman of Stage A vs Stage C for top-10 + random, per arm × metric |
| Per-example spot check | "Holds only on average" | Real ablation on 10 single examples for top 3 adapters |
| Trace-set agreement | "Depends on whose text" | Rank correlation base-correct vs gold |
| Zero-adapter run | Pipeline artefact | Untuned model with hooks: every attribution exactly 0 |
| Finite difference | Wrong gradient wiring | g_i = 1 → 1−ε real ΔM/ε vs attr_i, on the real model, 5 adapters |

No seed replicates exist; arm differences are not compared against a run-to-run noise floor. Stated as a limitation.

## Statistics

- Unit of resampling: question. Paired bootstrap, same resampled questions across arms and conditions, 10,000 resamples, seed 1234 **[proposed]**.
- Primary decision rules **[proposed, to freeze before Stage A results are read]**:
  1. Separability: Spearman ρ(attr_acc, attr_len) across 196 in Gold SFT. Supported if 95% CI upper bound < 0.3, top-10 overlap ≤ 3, and Stage C shows removing top-10 length adapters lowers M_len more than random while M_acc stays within its CI. Contradicted if CI lower bound > 0.6.
  2. Sharing: Spearman between arms' 196-vectors per metric. Supported if ρ(online-grpo, offline-grpo) > max ρ(gold-sft, GRPO arm) with non-overlapping CIs, and ρ(sampling-sft-k, gold-sft) decreasing in k.
  3. Dense/sparse: neurons holding 50%/90% of |attr|; cancellation Σ|attr_j| / |Σ attr_j|. Sparse if top-64 sufficiency keeps ≥ 70% of the real effect and beats random 64; dense (kill for neuron-level story) if ≥ 512 needed in every arm.
- Validity gates, checked before any primary is read: predicted-vs-real Spearman ≥ 0.5 per arm × metric (else primaries 1–2 use Stage C effects only); median bootstrap Jaccard of top-10 ≥ 0.5 (else rankings reported unstable); Stage C and D agree in sign for each reported condition.
- Secondary family (Holm within adapter-level and neuron-level families, labelled exploratory): concentration of |attr| in top 5/10/20; per-item link between top length-adapter attribution and that item's generation length; sign consistency; accuracy vs length neurons; math selectivity of top neurons (from joint audit A lists); attribution vs write outliers.
- Report effects with 95% CIs; generation effects in points and tokens, paired against the full arm.

## Precision and numerics

Follow the joint audit: FP32 forwards and backwards, TF32 off, autocast off, eager attention, recorded backend. Attribution products are accumulated in FP64 per example and stored FP32. Per-example results must not depend on batch composition beyond a measured tolerance (qualification).

## Storage and operations

- Outputs under `runs/adapter-neuron-attribution/<stage>/<run_id>/`, sealed, hashed, resumable per (arm, metric, trace set) work unit; reuse `BlockStore` for Stage B dense arrays (examples × 8,960 per layer).
- Sizes: Stage A ≈ 6 arms × 2 metrics × 2 sets × 200 × 196 × 4 B ≈ 1.9 MB. Stage B ≈ 200 × 8,960 × 4 B ≈ 7.2 MB per (arm, metric, set, layer); ≤ 5 layers → ≈ 0.86 GB.
- Pod operations, preflight, export and restore reuse the joint audit's `neuron_audit_pod` / `neuron_audit_export` contracts (one Pod, one writer, verified HF export before any stop).

## Compute (estimate; qualification replaces it)

Stage A+B ≈ 1–2 GPU-h; Stage C ≈ 1.5–3 GPU-h; Stage D ≈ 1.5–2 GPU-h with HF generation in FP32 (may be longer than the vLLM estimate in the companion doc). The user sets a hard wall-time cap after qualification, as in the joint audit.

## Testing decisions

CPU (tiny random Qwen2 + PEFT, no downloads; red/green TDD per AGENTS.md):
- Hooks attached, nothing ablated → logits bit-identical to unhooked.
- Untuned model → all attributions exactly 0.
- Finite-difference agreement for adapter attribution (5 adapters) and neuron attribution (3 neurons).
- Σ_j attr_j = attr_i.
- Per-example attributions identical at batch size 1 and 8 (CPU, FP64 fixture).
- Sum-not-mean backward: per-example values independent of batch size.
- Gauge: (a, b) → (k a, b/k), k ∈ {−2, 0.5} leaves attr_i and attr_j unchanged.
- Ablation hook == edited checkpoint logits; keep-all mask == full; keep-none == removed; constant-c̄ hook writes s b c̄.
- M_len masks exclude the final answer line, padding, template and EOS positions; `####` tokenization fixture.
- Selection freeze refuses confirmation rows; ties deterministic; random controls reproducible from seeds.
- Write-matched sampling returns unavailable (not a looser match) when no candidate qualifies.

Real GPU (ticket 05): finite difference on the real model; batch invariance tolerance; one arm end-to-end timing; Stage C sanity on 5 adapters.

## Open decisions

1. Approve [proposed] values: M_len exclusion window, base-correct minimums, layer-selection rule, write-match tolerance, bootstrap and null counts, primary thresholds, early-stop definition.
2. Does Stage B need `a ≈ a₀` for every selected module, or report moved modules separately (proposed)?
3. Include o_proj head-level attribution now or defer?
4. Generation path: HF (protocol-consistent, slower) vs vLLM (faster, new comparability check needed). Proposed: HF.
5. Whether to freeze written predictions before Stage A results are inspected (optional, as in the joint audit).
