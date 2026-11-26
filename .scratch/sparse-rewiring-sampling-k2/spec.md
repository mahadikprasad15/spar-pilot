# Sparse Rewiring on Sampling SFT (`k = 2`): Specification

Status: design decisions `K2-D1`/`K2-D2`/`K2-D3` confirmed (2026-11-26); ready for Phase 0 CPU implementation (no paid GPU spend authorized by spec publication)
Date: 2026-11-26
Design reference: `.scratch/sparse-rewiring-sampling-k2/design.md`

---

## Problem Statement

Gold-SFT sparse rewiring showed that 11–12 MLP neurons recover >90% of Gold-SFT's validation CE drop on `Qwen2.5-1.5B-Instruct`, while reproducing Gold-SFT's ~14-point accuracy drop and shortening answers from ~220 to ~130 tokens. Multi-lens `down_proj` decoding (`R-Lens`, `J++`, `J-Lens`, `Logit Lens`) revealed that the shared core and cross-layer substitutes are heavily concentrated in formatting, markdown/LaTeX delimiters, newlines, EOS, and final-layer digits.

To separate **data-style compression** from **accuracy-relevant adaptation**, we need a matched sparse-rewiring experiment trained on the existing **Sampling SFT (`k = 2`)** targets (`477` covered items from `gsm8k-train-512`), evaluated on matched held-out `k = 2` validation targets (`val128`) and greedy accuracy (`val128` + `cohort150`), followed by a cleanly separated multi-lens `down_proj` and activation-shift ($\Delta a_j$) interpretation pass.

---

## Solution

Extend the existing `pilot_eval` sparse-rewiring workflow without mutating the sealed Gold `v1` schema:
1. **Stage 0 (CPU):** Add backward-compatible `sparse-rewiring/launch-record-v2` support for `recipe = sampling-sft-k2` (477 stored-ID training items, variable batch sizes 3–7 with `examples_per_mask = 4`, configurable `l0_lambda`, and `k = 2` LoRA reference), backed by CPU unit tests; parameterize the reusable `down_proj` multi-lens notebook & HTML report scripts.
2. **Stage 1 (GPU):** Generate and seal held-out `k = 2` targets on the exact 128 `gsm8k-train-val128` questions, measure the untuned and dense `sampling-sft-k2` (checkpoint-64) reference bar on `val128-k2` CE and `val128`/`cohort150` accuracy, and compute exact headroom $\Delta\text{CE}_{k=2}$ and scaled penalty $\lambda_{k=2}$.
3. **Stage 2 (GPU — Run A):** Train sparse rewiring (**Rank 1**, `read_lr = 3.0e-3`, `seed 42` first) at Gold's exact matched penalty $\lambda = 6.0 \times 10^{-4}$ for 1,024 steps (freeze after 960). If ≥ 4 survivors, run replication `seed 43` at matched $\lambda$.
4. **Stage 3 (GPU — Conditional Run B):** If Run A ends with **`< 4` survivors** (K2-D1), train **Rank 1 across 2 seeds (`seed 42` + `seed 43`)** at the Stage-1 calibrated $\lambda_{k=2}$ under the identical 1,024-step protocol.
5. **Stage 4 (Separate Post-Hoc Analysis):** Export compact survivor JSONs and `val128` activation shifts ($\Delta a_j$) on-pod for both Gold and `k = 2`, then run `Logit Lens`, `J-Lens`, `J++`, and `R-Lens` decoding, targeted GSM8K/style/EOS probes, and a Gold-vs-`k = 2` comparison HTML report.

---

## User Stories

1. As a researcher, I want Gold SFT's `launch-record-v1` schema and sealed artifacts left completely untouched, so adding `k = 2` never invalidates historical provenance.
2. As a researcher, I want `k = 2` sparse rewiring to consume the exact stored response token IDs from `l4-training-data-v1` (477 covered items, 35 marked missing slots in Gold's 64-batch order), so decode/re-encode drift cannot alter supervised examples.
3. As a researcher, I want held-out `k = 2` targets generated on the exact same 128 `gsm8k-train-val128` questions used in Gold sparse rewiring, so training keeps all 477 `k = 2` items (matching dense LoRA `k = 2`) while validation measures true held-out `k = 2` CE.
4. As a researcher, I want untuned and dense LoRA `sampling-sft-k2` (checkpoint-64) evaluated on `val128-k2` CE and `val128`/`cohort150` accuracy before training starts, so "% CE gain recovered" and headroom-scaled $\lambda_{k=2}$ are fixed prior to Run B.
5. As a researcher, I want Run A to test Gold's exact $\lambda = 6.0 \times 10^{-4}$ first—and if 1–3 neurons survive, I want their post-freeze (steps 961–1024) CE and accuracy reported cleanly rather than discarded.
6. As a researcher, I want Run B (headroom-scaled $\lambda_{k=2}$) triggered conditionally by an explicit pre-registered rule rather than ad-hoc reruns or extending training beyond 1,024 steps.
7. As a researcher, I want `k = 2` models judged primarily on `val128`/`cohort150` greedy accuracy and `val128-k2` CE (not Gold Val CE, which is logged strictly as a secondary descriptive metric).
8. As a researcher, I want multi-lens `down_proj` decoding (`Logit`, `J-Lens`, `J++`, `R-Lens`) and activation-shift sign ($\Delta a_j$) kept as a cleanly separated post-hoc stage reusing the same code/notebook/HTML report pipeline, so it runs either on the pod or locally on Mac with ~25 MB disk footprint.

---

## Implementation Decisions

### 1. Backward-compatible `v2` plan & schema (`pilot_eval/sparse_rewiring_plan.py`)
- Preserve `sparse-rewiring/launch-record-v1` and `PRODUCTION`/`RULES` unchanged for Gold SFT.
- Introduce `sparse-rewiring/launch-record-v2` adding explicit recipe binding:
  - `data.recipe`: `'gold-sft'` | `'sampling-sft-k2'`
  - `data.training_examples`: `477` for `k = 2` (verified against `l4-training-data-v1` manifest hash and 35 explicit uncovered IDs)
  - `data.targets`: `'sampling-sft-k2-stored-ids'` (stored IDs + `<|im_end|>` supervised, no truncation, context-limit refusal preserved)
  - `data.short_batch_policy`: preserve Gold's 64 batches for steps 1–64 (batch sizes 3–8, refusing any empty batch); epochs 2–16 reshuffle the 477 covered items per K2-D3 with `random.Random(42 + epoch - 1)`
  - `objective.l0_lambda`: explicit float per launch record (`6.0e-4` for Run A; Stage-1 derived $\lambda_{k=2}$ for Run B)
  - `inputs`: bind `k2_training_data`, `k2_lora_adapter` (`checkpoint-64`, `sha256: a28f52a0...`), `val128_k2_targets`, and `val128_gold_targets` (optional secondary CE cross-check)
- Short-batch mask grouping: with `examples_per_mask = 4`, partition any batch of size $B \in [3, 8]$ into contiguous groups of at most 4 (e.g., $B=7 \to [4, 3]$, $B=5 \to [4, 1]$, $B=3 \to [3]$), reusing `SharedMaskCache` and total-supervised-token loss denominator (`sft_backend.completion_loss`).

### 2. Stage 1 — Held-out `val128` `k = 2` target generation & reference bar
- Input cohort: exact `gsm8k-train-val128` question IDs (verified by `ids_sha256` against Gold's sparse-rewiring validation cohort).
- Sampling recipe: exact `sampling-sft` `k = 2` recipe (`J = 64`, `cap = 1024`, `temperature = 0.6`, `repetition_penalty = 1.1`, `top_p = 1.0`, gold-guided proposer + scorer prompt, `flexible-v3` automatic verdict + required single final `<|im_end|>` token, initial attempt + up to 3 bounded model retries, deterministic seed derivation).
- Output artifact: sealed `val128-k2-targets.json` retaining covered rows (stored token IDs + text + NLL), uncovered rows with terminal disposition, and manifest hash.
- Reference evaluation (IEEE FP32 child process, matching ADR 0021):
  - Evaluate `untuned` (step 0) and `sampling-sft-k2` (`checkpoint-64`) on `val128-k2` CE, `val128` greedy accuracy, `cohort150` greedy accuracy, and mean response length.
  - Record $\Delta\text{CE}_{k=2} = \text{CE}_{\text{untuned}}(k2) - \text{CE}_{\text{LoRA}}(k2)$ and derived $\lambda_{k=2} = 6.0 \times 10^{-4} \cdot (\Delta\text{CE}_{k=2} / 0.1777)$ in `references/k2_headroom.json`.

### 3. Stages 2 & 3 — Training protocol & conditional two-$\lambda$ rule
- Shared across Run A and Run B (identical to Gold sparse rewiring):
  - `Qwen/Qwen2.5-1.5B-Instruct`, FP32 IEEE, eager attention, `gradient_checkpointing` non-reentrant, `determinism = full`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `A100-80GB`.
  - Candidate set: all `28 × 8,960 = 250,880` MLP neurons (`gate_proj` + `up_proj` reads; `down_proj` frozen).
  - Hard-Concrete L0: $\beta = 2/3$, $\gamma = -0.1$, $\zeta = 1.1$, $\log\alpha_{\text{init}} = 1.0$, `once-per-step-expected-active`.
  - Optimizer: `AdamW(foreach=False)`, `betas = (0.9, 0.99)`, `eps = 1e-8`, `weight_decay = 0.0`; `mask_lr = 0.1` (unclipped); `read_lr = 3.0e-3` (Rank 1, clipped at norm `1.0`).
  - Schedule: `1,024` steps (`16` exact epochs × `64` steps/epoch: Gold's 64 batches for epoch 1; 29×8 + 35×7 reshuffled batches per epoch for epochs 2–16), linear warmup `20`, cosine to `0` at step `1,024`, hard binary freeze `z_det > 0` after step `960` (steps 961–1024 train frozen binary survivors).
  - Checkpoints: `[0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024]` (~36 MB total per Rank-1 seed).
  - Cadences: step metrics every 1, `val128-k2` CE (deterministic masks) every 16, FP16 `log_alpha` snapshots every 16, greedy `val128` at all 11 checkpoints, exploratory `cohort150` at steps `64` and `1024`.
- **Run A (`sparse-k2-sft-r1-matched-s42` [+ `-s43` if ≥ 4 survivors]):** `l0_lambda = 6.0e-4`. No extension beyond 1,024 steps if gates collapse.
- **Conditional Run B (`sparse-k2-sft-r1-scaled-s42`, `-s43`):** Triggered if Run A (`seed 42`) ends with **`< 4` survivors** (K2-D1), using Stage 1's pre-registered $\lambda_{k=2}$ without post-hoc sweeping.

### 4. Stage 4 — Separate Post-Hoc Multi-Lens & Activation-Sign Analysis
Kept completely separate from training chains so it can be run independently on the pod or locally:
- **On-pod lightweight export (`extract_survivor_shifts.py`, ~1 min on A100):**
  - Reads sealed final checkpoints for Gold (`sparse-gold-sft-r{0,1}-1`) and `k = 2` (Run A & Run B) on `sl-0`.
  - Exports small JSONs (`<200 KB`): surviving `(layer, idx)` sets + per-neuron `val128` activation shifts (`mean(Δa_j)`, `mean(|Δa_j|)`, `% positive`, `% negative`, and top-5 positive/negative firing token contexts).
- **Parameterized `down_proj` multi-lens decoder & report (`notebooks/decode_down_proj.py` + `notebooks/build_down_proj_report.py`):**
  - Supports `--model-dir` (local weights on pod, ~5s) OR `--stream-hf` (HTTP byte-range streaming on Mac, ~25 MB permanent cache, zero 3 GB model stored).
  - Applies all 4 lenses (`Logit Lens`, `J-Lens`, `J++`, `R-Lens` from `neuronpedia/jacobian-lens :: qwen2.5-1.5b-it/`), with orientation self-check, RMSNorm gain $\gamma$, vocab centering, and both `+v` / `−v` poles.
  - Produces per-experiment tables (Shared / R0-only / R1-only + full-arm Rank 0 vs. Rank 1), 18-token GSM8K/style/EOS z-score probe, active-pole sign annotation from `Δa_j`, and a **Gold SFT vs. Sampling SFT `k = 2` comparison** (index overlap, cross-experiment R-Lens cosine matrix, subspace principal angles, HTML report `docs/sparse-rewiring-k2-vs-gold-report.html`).

---

## Testing Decisions

- **CPU-only offline unit tests (Phase 0, no CUDA or network required):**
  1. `sparse_rewiring_plan` v1 backward-compatibility regression: existing Gold `launch-record-v1` tests pass unmodified.
  2. `sparse_rewiring_plan` v2 validation: accepts `sampling-sft-k2` (`477` items, custom `l0_lambda`, `k2` arm names), rejects `[proposed]` leaves, rejects empty batches, and verifies deterministic 1,024-step plan hashes.
  3. Short-batch `train_rewire` loss & mask-group oracle (batch sizes 3, 4, 5, 6, 7, 8 with `examples_per_mask = 4`): step completes, loss agrees with independent FP64 hand calculation (`atol=1e-6, rtol=1e-6`), gradients finite, read clipping applies to reads only, L0 applied once per step.
  4. `decode_down_proj.py` & `build_down_proj_report.py` regression test on Gold's existing 25 MB cache: reproduces identical HTML/JSON outputs and handles arbitrary survivor counts (including 1–3 survivors from a sparse Run A).
- **Real GPU evidence kept distinct from CPU tests:** Passing CPU tests validates software mechanics only; real `val128-k2` generation, reference headroom, Run A/B training, and `Δa_j` extraction require explicit GPU launch approval.

---

## Out of Scope

- Rerunning Gold SFT sparse rewiring or modifying `launch-record-v1` seals.
- Sampling SFT `k = 0` or `k = 5` sparse rewiring (deferred; `k = 5` has only ~0.005 nats training CE drop).
- Online/offline GRPO sparse rewiring or KL-distillation from GRPO (tracked in a separate future design/spec).
- Full $\lambda$-grid sweeps beyond the two pre-registered values (matched $6.0 \times 10^{-4}$ and conditional headroom-scaled $\lambda_{k=2}$).
- Multi-seed runs, leave-one-out greedy ablations, or L17–L19 SAE projection (deferred follow-ups).
- Any paid cloud launch, RunPod volume deletion, or HF push by virtue of this spec alone.
