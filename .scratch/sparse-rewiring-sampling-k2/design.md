# Sparse Rewiring on Sampling SFT (`k = 2`) — Working Design

Status: core experimental decisions (`K2-D1`, `K2-D2`, `K2-D3`) confirmed by the user on 2026-11-26 (`K2-D4`/`K2-D5` remain operational preflight items before GPU launch). No paid GPU job is authorized by this document alone.
Date: 2026-11-26
Parent experiments:
- Gold sparse rewiring (`sparse-gold-sft-r0-1`, `sparse-gold-sft-r1-1`; report `2026-11-26`)
- Sampling SFT `k = 2` (`l4-training-data-v1`, `l4-gate-v2`, `l4-night-v1-train-k2`)
- Frozen `down_proj` multi-lens interpretation (`docs/sparse-rewiring-down-proj-report.html`)

---

## 1. Purpose and Scientific Question

In the production Gold-SFT sparse-rewiring run, both Rank 0 (12 neurons, 94.4% CE gain recovered) and Rank 1 (11 neurons, 91.1% CE gain recovered) converged on an overlapping 5-neuron mid-late core (`L11 #8235`, `L16 #5569`, `L17 #6230`, `L17 #7054`, `L19 #5606`; $p \approx 3 \times 10^{-18}$) while reproducing Gold-SFT's ~14-point accuracy drop on `val128` (`80.5% → 67.2% / 66.4%`) and length compression (`220 → 130` tokens). Multi-lens `down_proj` decoding (`R-Lens`, `J++`, `J-Lens`, `Logit Lens`) showed that nearly all shared Gold survivors—and cross-layer substitutes like `L12 #6760 ↔ L6 #5466`—control markdown/LaTeX delimiters (`**`, `\[`, `boxed`), GSM8K tags (`####`, `<<`, `>>`), newlines (`\n`, `\n\n`), EOS (`<|endoftext|>`), and final-layer digits.

This experiment holds the **exact sparse-rewiring objective and optimization recipe fixed** and swaps **only the supervised completion distribution** from GSM8K Gold solutions to the verified **Sampling SFT (`k = 2`)** targets on the same 512 training problems (`gsm8k-train-512`).

**Core questions tested:**
1. **Sparsity under smaller style gap:** At Gold's exact per-nat penalty ($\lambda = 6.0 \times 10^{-4}$), how many neurons survive when the target completions already match Qwen2.5's own style and length?
2. **Matched-budget neuron identity (conditional Run B):** When $\lambda$ is scaled to `k = 2`'s measured CE headroom (yielding a comparable ~10–15 neuron budget if Run A collapses), do Gold's formatting/EOS/LaTeX-suppression switches (`L19 #5606`, `L16 #5569`, `L17 #7054`, `L12 #6760`, `L6 #5466`) drop out in favor of a distinct neuron set?
3. **Accuracy vs. CE:** Does sparse rewiring on `k = 2` recover dense LoRA `sampling-sft-k2`'s held-out `k = 2` CE **and** preserve/improve `val128` & `cohort150` greedy accuracy (avoiding Gold's −14 pt accuracy collapse)?
4. **Separate downstream mechanistic comparison:** Using the same multi-lens `down_proj` decoding (`Logit Lens`, `J-Lens`, `J++`, `R-Lens`) plus activation-shift signs ($\Delta a_j$), how do the `k = 2` survivors compare to Gold's 18 survivors in neuron index, J/R-space cosine similarity, subspace overlap, and GSM8K/style/EOS token roles?

---

## 2. Verified Existing Contracts & Measured Facts

Facts verified directly from local artifacts in `spar-pilot` and `~/spar/hf-sampling-sft/unpacked`:

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 28 layers, `d_model = 1536`, `d_mlp = 8960`, 250,880 candidate MLP neurons, frozen FP32 IEEE, eager attention).
- **Saved `k = 2` training targets:** Sealed in `Prasadmahadik/spar-sampling-sft-artifacts` and unpacked under `runs/sampling-sft/sampling-sft-v1/training-data/l4-training-data-v1` (bound by `plans/l4-night-v1-train-k2/sampling-sft.training.json`).
  - Exact same `gsm8k-train-512` problem IDs in Gold's exact 64-batch order.
  - **477 covered / 35 uncovered** (`automatic-verdicts-accepted-v1` policy); short batches of size 3–7 are preserved with the total-supervised-token denominator.
  - Stored response token IDs are authoritative—never retokenized or truncated.
- **Measured `k = 2` CE headroom vs. Gold SFT (`l4-gate-v2/results/summary.csv` & `l4-night-v1-train-k2/logs/steps.jsonl`):**
  - Gold SFT: untuned NLL `0.5861` on train-512 (median `114` tok); Val CE drops `0.5728 → 0.3951` (**`0.1777` nats headroom**).
  - Sampling SFT `k = 2`: untuned NLL `0.3600` on train-512 (median `245` tok; `121,876` supervised tokens, ~2× Gold's `63,388`); 64-step dense LoRA `l4-night-v1-train-k2` drops training loss from `0.3278` (first 8 steps) to `0.2909` (last 8 steps)—indicating **~0.037–0.070 nats/token headroom (~3×–5× smaller than Gold)**.
  - *(Why `k = 2` instead of `k = 5`: `k = 5` has untuned NLL `0.2910` and only `0.0048` nats training loss drop across 64 steps, leaving almost no CE signal for L0 selection.)*
- **Dense reference adapter:** `runs/sampling-sft/sampling-sft-v1/training/l4-night-v1-train-k2/checkpoints/checkpoint-64/adapter_model.safetensors` (`sha256: a28f52a0b61d0ee6140ece1a8fc4983fb1dcf4bd75db84c8594e5a9c73c27f73`).
- **Missing held-out `k = 2` targets:** `l4-gate-v2/results/answers.jsonl` covers only `gsm8k-train-512` (`477` rows). The 128 held-out training questions (`gsm8k-train-val128`) currently have **Gold targets only** and require a one-time `k = 2` sampling pass (Stage 1 below).

---

## 3. Experimental Pipeline

The experiment is cleanly split into **Core Training & Evaluation (Stages 1–3)** and a **Separate Post-Hoc Lens & Sign Analysis (Stage 4)**:

```text
Stage 0 (CPU prep)
  └─ Extend sparse_rewiring_plan (v2) for k=2 (477 items, short batches 3–7, parameterized λ) + unit tests

Stage 1 (GPU — Val128 k=2 Target Generation & Reference Bar)
  ├─ Run qualified k=2 sampler once on the exact 128 gsm8k-train-val128 questions
  ├─ Seal val128-k2-targets (expected ~115–120 covered items)
  ├─ Evaluate Reference Bar on val128-k2 CE + val128/cohort150 greedy accuracy:
  │    • Untuned (step 0)
  │    • Dense LoRA sampling-sft-k2 (checkpoint-64)
  └─ Compute exact headroom ΔCE_k2 = CE_untuned(k2) − CE_lora(k2) and λ_scaled = 6e-4 × (ΔCE_k2 / 0.1777)

Stage 2 (GPU — Run A: Matched λ = 6.0e-4)
  ├─ Train Rank 1 (read_lr = 3e-3, seed 42 first; add seed 43 if ≥ 4 survivors) at Gold's exact λ = 6.0e-4
  ├─ 1,024 steps, warmup 20 + cosine to 0, freeze z_det > 0 after step 960
  └─ Decision Gate (K2-D1): if < 4 survivors, log Run A's frozen-polish metrics and trigger Run B

Stage 3 (GPU — Conditional Run B: Headroom-Scaled λ_k2)
  ├─ Triggered if Run A yields < 4 survivors (K2-D1)
  └─ Train Rank 1 (read_lr = 3e-3) across 2 seeds (seed 42 + seed 43) at l0_lambda = λ_scaled

────────────────────────────────────────────────────────────────────────
Stage 4 (Separate Downstream Analysis — Lens Decoding & Δact Sign)
  ├─ 4a (On-pod, ~1 min): Export survivor JSONs + val128 mean(Δa_j), %pos/%neg, top contexts
  │       for BOTH Gold SFT (already on sl-0) and k=2 (Run A & Run B)
  └─ 4b (Local Mac or Pod): Run parameterized down_proj decoder (Logit, J-Lens, J++, R-Lens),
          GSM8K/style/EOS probe, and Gold-vs-k=2 overlap & HTML comparison report
```

---

## 4. Two-$\lambda$ Protocol & Why Additional Steps Do Not Substitute for Scaling $\lambda$

### Why extra steps (>1,024) do not rescue $\lambda$-induced collapse
Under Hard-Concrete L0, gate $j$ remains open at equilibrium only if turning neuron $j$ on lowers completion CE by **more than $\lambda$ nats/token**.
1. **Equilibrium threshold:** If `k = 2`'s total CE headroom is ~`0.04–0.07` nats and no individual neuron contributes $> 6 \times 10^{-4}$ nats/token, the global optimum of $\text{CE} + \lambda \sum_j \mathbb{E}[z_j > 0]$ is **0 open gates** regardless of step count.
2. **Irreversible Hard-Concrete clamping:** In Gold SFT, gates fell to `3` (R0) and `0` (R1) by step 64 and reopened by steps 128–672 only because Gold's large CE gradient (`0.178` nats) exceeded $\lambda$ before `log_alpha` saturated. Once `log_alpha` is pushed to `−8` to `−10` (clamped at exact zero), gradients vanish and running 2,048+ steps pushes gates deeper into zero rather than reopening them.

Therefore, we keep **1,024 steps** fixed (exact comparability to Gold SFT) and handle headroom via the **conditional two-$\lambda$ design**:

- **Run A (Matched $\lambda = 6.0 \times 10^{-4}$):** Always runs first. Exact apples-to-apples per-nat price vs. Gold SFT.
  - If **1–3 neurons survive**, steps 961–1024 (hard binary freeze) already train just those 1–3 neurons so we directly measure what CE and accuracy they achieve alone.
- **Conditional Run B (Headroom-Scaled $\lambda_{k=2}$):**
  - **Trigger `[proposed]`:** Run B triggers if either arm in Run A ends with **`0` survivors** (strict collapse) OR **`< 4` survivors** (too few for a meaningful cross-arm overlap or multi-lens functional comparison against Gold's 11–12).
  - **Value:** $\lambda_{k=2} = 6.0 \times 10^{-4} \times \frac{\Delta\text{CE}_{k=2}}{0.1777}$, computed strictly from Stage 1's sealed `val128-k2` reference bar (no post-hoc tuning on Run B outcomes).

---

## 5. Evaluation & Validation Contract

We **do not** judge `k = 2` sparse rewiring by Gold validation CE (improving on `k = 2` naturally diverges from Gold's terse format).
- **Primary metrics:**
  1. **Held-out `k = 2` Validation CE** on the Stage 1 `val128-k2` targets (deterministic masks, every 16 steps + step 0), plus **% CE gain recovered** relative to `(untuned − LoRA k=2)`.
  2. **Greedy accuracy & mean response length** on `gsm8k-train-val128` (at all 11 checkpoints `[0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024]`) and `gsm8k-test-cohort150` (exploratory, at steps `64` and `1024`), using `flexible-v3`, `cap = 1024`, `eos_token_id = 151645`.
- **Secondary descriptive cross-check:** Gold Val CE on `val128` (logged only to show divergence from Gold style, never used as a selection criterion).

---

## 6. Separate Downstream Analysis: Multi-Lens `down_proj` & Active Sign ($\Delta a_j$)

Kept cleanly decoupled from training completion (Stage 4 does not gate training success or require rerunning training):
1. **Same 4 lenses as Gold:** `Logit Lens`, `J-Lens`, `J++`, and `R-Lens` (`neuronpedia/jacobian-lens`, `qwen2.5-1.5b-it/`), reusing the parameterized notebook & HTML report pipeline (`notebooks/build_down_proj_report.py` generalized to any `--survivors-json`).
2. **Active pole sign ($\Delta a_j$):** One ~30-second pass on `sl-0` over `val128` exports `mean(Δa_j)`, `% pos / % neg`, and top-5 firing contexts for **both** Gold SFT and `k = 2` survivors, resolving the `+v` vs. `−v` sign ambiguity.
3. **Gold vs. `k = 2` comparative readouts:**
   - Exact neuron index intersection (e.g., whether Gold's 5 shared formatting/EOS switches survive under `k = 2`).
   - Cross-experiment R-Lens cosine heatmap & principal-angle subspace overlap between Gold's survivors and `k = 2`'s survivors.
   - Side-by-side 18-token GSM8K / Markdown / LaTeX / EOS z-score probe.

---

## 7. Decisions Log (Approved & Operational)

- **K2-D1 (`[user 2026-11-26]` Run B trigger):** Trigger Run B (headroom-scaled $\lambda_{k=2}$) if Run A ends with **`< 4` survivors** (0, 1, 2, or 3). Run A's frozen-polish (steps 961–1024) metrics are always reported even when 1–3 neurons survive.
- **K2-D2 (`[user 2026-11-26]` Architecture, LR & replication):** Use **Rank 1 (`read_lr = 3.0e-3`) across 2 seeds (`seed 42` + `seed 43`)** instead of Rank 0—saves ~34 GB of Rank-0 checkpoints (~36 MB/seed) while directly providing invariant core-neuron overlap and seed variance. Run `seed 42` first on Run A ($\lambda = 6.0 \times 10^{-4}$); run the second seed (`seed 43`) on the surviving $\lambda$ (Run A if ≥ 4 survivors, Run B if triggered).
- **K2-D3 (`[user 2026-11-26]` Epochs 2–16 batching for 477 items):** Keep **64 steps per epoch**—steps 1–64 follow Gold's exact 64 batches (with 35 uncovered slots omitted, batch sizes 3–8); epochs 2–16 reshuffle the 477 covered items into 64 batches of **7–8 items** (29 batches of 8 + 35 batches of 7 = 477) with `random.Random(42 + epoch - 1)`, preserving `1,024 steps = exactly 16 full epochs`.
- **K2-D4 (`[proposed]` Val128 `k = 2` generation engine):** Reuse the qualified `sampling-sft` `k = 2` sampler on A100 (or HF FP32 refinement loop if vLLM setup overhead outweighs 128 items); resolve at GPU preflight.
- **K2-D5 (`[proposed]` Storage & export):** Run on `A100-80GB` attached to `sl-0`; export compact survivor/metrics/`Δa_j` bundles + Rank-1 checkpoints (~36 MB/seed) to private HF repo `Prasadmahadik/spar-sampling-sft-artifacts`.
