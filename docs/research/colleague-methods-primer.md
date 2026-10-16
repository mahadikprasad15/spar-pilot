# Methods conversation guide: what to understand and what to share

This is a preparation guide for discussion, not a script for claiming expertise you do not have. For each answer: state the quantity, show the actual run config, name a check, and state the limit. Exact executed artifacts outrank this guide.

## 1. What did we actually train?

An existing **Qwen2.5-1.5B-Instruct** model, with its original weights frozen, plus rank-1 LoRA adapters on seven projection families in all28 decoder blocks. SFT and GRPO train separate adapters. They do not train separate full base models or replace the language-model head.

For a linear map W∈R^(d_out×d_in), the standard adapter is:

`y = W x + (alpha/r) B A x`

- A∈R^(r×d_in), B∈R^(d_out×r).
- **Rank r=1**: the update for that particular matrix has rank at most1. The whole model's nonlinear behavior is not rank1.
- **Alpha=1**: with the standard alpha/r rule and rank1, the multiplier is1. Alpha is not the learning rate. Verify no rank-stabilized or other alternative scaling in the actual adapter config.
- **Dropout=0**: no adapter-dropout noise; it does not mean the model has no other stochastic training components.
- A starts nonzero and B zero under the used initialization. This makes the initial **write** zero, although raw A·x can be nonzero.
- Target families: q/k/v/o projections in attention; gate/up/down projections in the MLP.28×7=196 intended adapted linear modules. These matrices need not have equal shapes.
- Trainable adapter parameter count is Σ_modules r(d_in+d_out). Quote the measured count from the actual model/manifest instead of guessing from “1.5B.”

**Why this design?** A small, constrained intervention makes local writes simple to decompose and holds adapter capacity comparable across arms. It limits what can be learned and is not evidence that results generalize to full fine-tuning or high-rank LoRA.

Show: actual `adapter_config.json`, checkpoint factor hashes, source/model revision, source audit and zero-adapter check.

## 2. FP32, BF16, FP16, quantization and TF32

These are different choices:

| Term | Meaning here | What to check |
|---|---|---|
| FP32 |32-bit floating-point weights/forward arithmetic in main SFT/GRPO recipes | Actual model/adapter dtype, autocast, optimizer state and trainer arguments |
| BF16 |16-bit floating-point format with less significand precision than FP32 | Some earlier baseline variants; distinguish them from matched FP32 baseline |
| FP16 |Another16-bit floating-point format | Distinguish a lens file stored in FP16 from model forward precision |
|4/8-bit quantization |Compressed numerical representation, often with scale/zero-point metadata and dequantization operations | Quantization configuration/loader flags; these main arms are not QLoRA |
| TF32 |A GPU arithmetic mode that can affect some operations using FP32 tensors | Explicit matmul/cuDNN settings, not just tensor dtype |
| FP64 accumulation |64-bit arithmetic for reductions/statistics | Does not make an earlier FP32 model forward exact or recover information already lost |
| Retention dtype |Precision used when writing captured arrays to disk | Pilot5 selected FP32 differences; independent of LoRA rank and model parameter count |

**Short answer:** “The main matched SFT/GRPO arms are unquantized FP32. We record forward settings separately from storage and reduction precision. An older BF16 baseline and the external lens's fitting/storage history are separately labeled.”

Why FP32? To reduce numerical ambiguity in a small-write measurement experiment, at a runtime/memory cost. It does not eliminate floating-point errors or batch/GPU sensitivity.

## 3. Training recipe versus batch/step budget

**SFT:**512 training problems;64 updates; microbatch1 and accumulation8; completion-only gold-solution loss including end-turn. LR1e-4, AdamW, constant schedule, no warmup, weight decay0, gradient clipping1.0; rank1/alpha1/dropout0. Source code uses token-summed completion loss normalized over the accumulation window; audit actual supervised labels/masks.

**GRPO:**512 problems;8 problems/update×8 sampled completions/problem=64 completions/update;64 updates and4,096 accepted completions. Group-normalized correctness rewards, DAPO loss denominator, one iteration, beta0, temperature1, top_p1, top_k0 in the saved run. Capped completions receive reward0 under the chosen procedure. Frozen cap and monitoring choices matter.

A shared optimizer-step count is not an equal training-token or FLOP budget. SFT sees gold completions; GRPO samples responses from its evolving policy. Do not attribute every difference solely to the objective.

Show: saved `sft.config.json` / `grpo.training.json`, frozen reviews, `resolved-trainer-args.json` or actual trainer-args file, ordered groups, step diagnostics and accepted response IDs. A plan name containing “batch8” is not enough to determine whether8 means evaluation items, training problems, microbatches or generated completions.

## 4. Layers and tokens

- A decoder block includes attention, MLP and residual paths. A projection module is one linear operation within it.
- Qwen has28 decoder blocks here, indexed0–27. **Block14 is the fifteenth block under zero-based indexing.** Including the embedding output gives29 capture sites.
- A token is a tokenizer unit, not generally a word. Preserve actual token IDs and tokenizer revision. Whitespace/partial-word/numeric tokens affect decoded lists.
- Content-relative position0 is the first content token. In a chat-formatted input it need not be absolute model position0. Pilot5 primary raw text avoids chat wrappers; Pilot3/4 explicitly mask content inside rendered sequences.
- Padding, instruction, question, gold-solution and FineWeb masks control the measurement population. Excluding padding from a sum does not prove every batched model operation is invariant.
- Fixed-context measurements supply identical known tokens to both models. They do not use each arm's freely generated continuation as if it were the same input.

Show: saved `input_ids`, `attention_mask`, `position_ids`, content/view positions, rendered prompt and sample audited masks.

## 5. Three different internal quantities

1. **Direct local module write:** `(alpha/r) B A x_adapted`, computed on that adapted pass's module input. It includes the local adapter branch, not the frozen module's response to an upstream input change.
2. **Propagated block-output change:** `h_adapted − h_base` for identical supplied tokens. This includes upstream and downstream interactions already accumulated by that site.
3. **Raw rank-1 coefficient:** `c=A x_adapted`. It is a scalar at rank1, but depends on the saved factorization. Replacing A by kA and B by B/k changes c while preserving the branch. Interpret c with its B/scaling metadata.

A small mean vector can reflect cancellation among large per-token changes. `norm(mean(delta))`, `mean(norm(delta))` and a mean of per-token relative norms are different estimands. Copy each report's exact numerator/denominator and token/equal-example weighting; do not label them all “activation change percent.”

Show: `activation_math.py`, fixed-write report definitions, per-example vectors/sums/counts and independent rank-1 reconstruction checks.

## 6. KL: what it does and does not say

At each fixed prefix, `KL(p_tuned || p_untuned) = Σ_v p_tuned(v) log[p_tuned(v)/p_untuned(v)]` over the full vocabulary, in nats. Direction matters. Logits at t−1 predict token t; the target token's content mask determines its view. No target0/padding/out-of-sequence terms are inserted.

KL answers how much the next-token distribution changed on these contexts. It does not establish accuracy loss, global policy distance, a causal mechanism or semantic task knowledge. GRPO beta0 means no training KL penalty; it does not disable post-training KL measurement. Equal KL, equal write norm and equal training steps are distinct comparisons.

Pilot4's supplemental coefficient/KL results remain unconfirmed in the last shared execution status. Do not present the implemented math as a completed empirical finding.

## 7. Scorers and behavior

Strict and flexible scorers read the same saved answer; they are not separate model generations. Flexible-v3 first preserves the existing valid strict final-line contract, then falls back to the v2 extraction rules. It prevents a gold-style final marker from being rejected merely because preceding reasoning has other numbers. Extraction does not receive the gold answer; numeric comparison is separate. It does not check reasoning validity or understand arbitrary prose.

The v3 revision happened after seeing outputs. Preserve and label it post hoc, show changed extractions, and report all selected checkpoints under the same scorer. Accuracy denominator includes invalid/capped responses. Lower strict score alone can be a formatting issue. Sampled mean correctness over8 draws is not pass@8.

Show: `scoring.py`, ADR0005, saved per-item extraction audits, scorer version and cap/invalid counts.

## 8. Pilot5 readouts and automatic relevance

For document d, primary vector `m_d = mean_{p=0..4}(h_adapted[d,p,14]−h_base[d,p,14])`; cohort vector is the equally weighted mean of m_d over200 documents.

The logit readout uses the pinned final RMS norm gain and output head. The Jacobian path first transports the vector through the qualified J14 map, then uses the declared final norm/head convention. These are readouts of a direction, not a claim that a model actually generated those tokens in context.

The top20 normalized unique positive-score strings are compared with200 task-enriched strings. Questions, gold solutions and saved GRPO completions create three separate references. Counts against exactly2M background tokens come from an approved bounded10,000-document FineWeb scan. Enrichment suppresses generic language; the final overlap is set membership, not another frequency ratio. Nonmembers are not automatically “general.”

Random-direction nulls ask how much overlap the same lens tends to produce without a learned direction. Random-adapter and base/adapted-activation baselines answer different questions. Whole-document bootstrap preserves correlated positions and arm pairing. Do not bootstrap tokens as independent observations.

Show: frozen references/procedure, source counts, analysis plan/draws, qualification receipts and exposure status. Raw top-token lists alone are not sufficient to establish task-specific or causal interpretation.

## 9. What configs actually share

A useful reproducibility packet has these pieces:

| File/object | What another researcher learns | What it does not provide alone |
|---|---|---|
| Scientific/frozen config | Model/tokenizer/dataset revisions; ordered examples; templates; decoding/scorer; seed; precision/batches; source paths/hashes; reviewed choices | Actual weights, data files, installed environment or results if they are referenced elsewhere |
| `adapter_config.json` | Rank, alpha, dropout, target modules, base identity and adapter options | Trained factor values |
| `adapter_model.safetensors` | Trained adapter factors | Full base weights, optimizer state or all training provenance |
| Runtime/trainer manifest | Actual dependency versions, GPU/CUDA, numerical flags and executed training arguments | Proof every runtime/config claim was correctly implemented |
| Raw responses / captured shards | Inputs/outputs needed to audit metrics and redo permitted analyses | License to redistribute or proof all other artifacts are intact |
| Result summary and report | Measured values, estimands, uncertainty and source linkage | A substitute for raw evidence or an independent replication |
| Completion seals/hashes | Content identity/inventory checks for the supported workflow | A signature proving scientific truth or resistance to a malicious rewrite |
| Checkpoint recovery state | Optimizer/scheduler/RNG/data position and adapter state | Full training recovery if only adapter files were copied |

Share originals inside the trusted team when appropriate, or separately named sanitized copies with a path mapping. Do not edit canonical configs just to make a share copy look tidy. A public clone does not include Git-ignored artifacts; provide an explicit artifact manifest and access route. Run a scoped credentials/data-sharing review before publishing an artifact bundle; none has been performed by this guide.

## 10. A defensible two-minute explanation

“We first made the behavior measurement reproducible, then trained a small SFT adapter and measured its internal changes on identical inputs. We added a GRPO arm with the same model, problems, rank and step budget, while recording the differences in training signal and generated-token budget. We compare behavior, local versus propagated writes, and—once complete—conditional KL and coefficients. Pilot5 asks whether the task is readable from mean changes on unrelated raw text, using qualified lenses and reference/null controls. This is one-seed descriptive/mechanistic evidence; readability is not yet a causal mechanism claim.”

It is appropriate to say “I used AI assistance for implementation; here are the exact design choices, tests, artifacts and limitations.” If you cannot explain a component yet, show its definition and identify the remaining question. Do not substitute an impressive codebase for understanding or hide the assistance.
