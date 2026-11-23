# Sparse rewiring on GSM8K: filled design and decision register

**Status:** needs-triage. Design only. Filled 2026-11-22 from the researcher's design input (`design-input.md`, preserved unchanged). Nothing here approves code, a GPU run, a threshold or a budget. **[user]** = already decided by the researcher; **[proposed]** = Claude's recommendation awaiting a decision; **OPEN** = no recommendation possible from available evidence.

Companion files: `decision-log.md` (options and the recommended minimal pilot), `spec.md` (draft implementation spec), `issues/` (draft tickets), `map.md`.

## 0. How evidence is cited

| Tag | Meaning |
|---|---|
| `main@b1a541b` | spar-pilot `main` (this branch's base). The SFT/protocol/scoring files are byte-identical on `c-attribution-pilot` |
| `catt@7667739` | spar-pilot branch `c-attribution-pilot`, head on 2026-11-22. It contains every other attribution branch (`claude/bold-gauss-5qkzwi`, `neuron-audit-phase1`, `offline-grpo`, `adapter-attribution-spec` are all ancestors) and the `catalog/` |
| `max@a6dc314` | `maxsloef/owl-rewired-neurons` `a6dc314740acbf726dcac435b4e84eebbbde49d8`, still the public `main` head when cloned on 2026-11-22 |
| **catalog** | `catalog/*.yaml` on `catt@7667739`; its own `evidence.status` (verified / inferred / unknown) is carried over unchanged |
| **recomputed** | Claude recomputed the value on CPU from code alone in this session (no dataset or model download) |
| **not checked** | The artifact exists only on Drive, HF, the sl-0 volume or the researcher's machine. No device folder or artifact root was reachable from this session, so no artifact bytes were hashed here |

**Important limit.** This reconstruction reads code, specs, launch records and the catalog. It did not read `meta/resolved_trainer.json`, `inputs/order.json`, `training.items.json` or any run config, because the artifact root is not in Git (`docs/artifacts.md@main`). Every value marked "code" is what the pinned code would have produced; the saved run files govern and must be read in ticket 01 before any of these values are frozen.

## 1. Current repository state (2026-11-22)

- Active work: `c-attribution-pilot` (`catt@7667739`, 206 commits ahead of `main`). Production run `c-production-1` **finished** 08:43–10:21 UTC today, one attempt, verified backups, Pod stopped (`.scratch/c-attribution-pilot/issues/12-production-run.md@catt`); report and catalog registration done (`13-report-catalog.md@catt`). Ticket 19 (held-out follow-up on GSM8K **test** minus every prior use) is prepared, not launched.
- Nothing in this design writes to `c-attribution-pilot`, its records, its catalog entries or its artifacts. New files live only under `.scratch/sparse-rewiring-gsm8k/` on `claude/project-thread-4ft24f`.
- Attribution code (`pilot_eval/c_attribution*.py`, `attribution_*.py`, `neuron_audit_*.py`, `catalog.py`) is **not on `main`**. Implementation should start on `main` after `c-attribution-pilot` merges (decision D17).
- Session-start test run on `main@b1a541b`: see the PR description for the counts.

## 2. Reconstructed executed contracts

### 2.1 Model and tokenizer

| Item | Value | Source | Status |
|---|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` rev `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `catalog/arms.yaml@catt` header; `docs/methods.md@main`; C launch inputs record "`989aa798…` in both configs" (`c-attribution-pilot/spec.md@catt`) | recorded in run configs (per C spec); not rehashed here |
| Tokenizer | Same repo and revision; chat template sha256 `2e2d2512…` (prefix only in Git) | `c-attribution-pilot/spec.md@catt` §Data and provenance | recorded; full hash in source configs, not checked |
| Shape | 28 decoder blocks, MLP intermediate width 8,960 (`down_proj` input), hidden 1,536 | 28×8,960 used throughout C attribution (`c-attribution-pilot/spec.md@catt`); hidden 1,536 asserted in `pilot_eval/trace_lenses.py:113@catt` | from code; **confirm from the loaded `config.json` in ticket 02** |
| MLP form | Qwen2 MLP: `down(act(gate(x)) * up(x))`, SiLU, no MLP biases | Qwen2 architecture; to verify against pinned `transformers==4.57.6` `modeling_qwen2.py` | not checked here |
| Candidate neurons | 28 × 8,960 = 250,880, zero-based `(layer, neuron)`; neuron j = row j of `gate_proj`/`up_proj` = column j of `down_proj` = input j of the `down_proj` LoRA read | derived | to test (ticket 02) |
| "Untuned" | This checkpoint with adapters disabled / zero-B adapter `b26c524a…` | `catalog/arms.yaml@catt` | catalog |

### 2.2 Gold SFT (Pilot 2), canonical arm `gold-sft`

Canonical run: `runs/pilot-2/Qwen--Qwen2.5-1.5B-Instruct/gsm8k/train-512-seed-42/rank1-float32/pilot2-sft-fp32-l4-batch8-v1` (`catalog/arms.yaml@catt`). This is the **L4** variant (ADR 0004); the T4 plan is a separate, older variant.

| Area | Value | Source |
|---|---|---|
| Dataset | `openai/gsm8k` `main`, rev `740312add88f781978c0658806c59bc2815b9866`, split `train` | `catalog/cohorts.yaml@catt` `gsm8k-train-512` |
| Training IDs | 512 = `sorted(random.Random(42).sample(range(len(train)), 512))`, IDs `gsm8k:train:{index}` | `pilot_eval/sft.py:123,144@main` |
| ID digest | `e15548111c6ee7eb2035f1e7e9ad7e8df6af155b563819d1933ea73334f22b31` | catalog; **recomputed** identical from code with `len(train)=7473` |
| Order | `order = list(range(512)); random.Random(42).shuffle(order)`; SequentialSampler over that order; step s uses `order[(s-1)*8 : s*8]`; checked each step against actual exposures | `sft_backend.py:103-104,203-208,228-233@main` |
| Exposure | 64 steps × 8 = 512 presentations = exactly one epoch, no repeats | code; saved `logs/steps.jsonl` not checked |
| First batch (reconstruction) | `train:3104, 2902, 4885, 1746, 6609, 1148, 6833, 2308` | **recomputed**; must equal saved `inputs/order.json[0:8]` (ticket 01) |
| Prompt | `gsm8k_messages`: one user turn "Solve the following problem step by step. End your response with a final line in the form #### <number>.\n\nProblem: {q}", rendered with the tokenizer's chat template, `add_generation_prompt=True` (Qwen default system message included, known discrepancy kept) | `protocol.py:25-37@main`; `.scratch/pilot2-sft/spec.md@main` |
| Target | Raw GSM8K `answer` string (calculator annotations and `####` kept) as the assistant turn | `sft.py:131-133@main` |
| Tokens | `encode(full, add_special_tokens=False)`, cut after the single `<|im_end|>` (151645); trailing template newline **not** supervised | `sft.py:134-141@main` |
| Labels | `[-100]*prompt_tokens + input_ids[prompt_tokens:]`, last label must be EOS 151645 | `sft.py:146,71-76@main` |
| Truncation / packing | Forbidden / off; `max_length` = longest real sequence | `sft.py:141-142,171@main` |
| Loss | `CE(logits[:, :-1], labels[:, 1:], reduction='sum') / num_items_in_batch` = **summed over supervised tokens of the 8-example window, divided by that window's supervised-token count** (token-weighted) | `sft_backend.py:36-48@main`; `docs/experiments/pilot-2.md@main` |
| Model dtype / attention | FP32 weights and LoRA, `attn_implementation='eager'`, no autocast | `sft_backend.py:81-82,98@main` |
| TF32 | `SFTConfig(tf32=False)` (legacy HF switch; cuDNN TF32 not explicitly set by this backend; Sampling SFT later sets both explicitly) | `sft_backend.py:256@main`; `sampling_sft_training_backend.py:19-21@main` |
| Packages | torch 2.9.0, transformers 4.57.6, trl 0.26.2, peft 0.18.0, datasets 4.4.1, accelerate 1.12.0, huggingface-hub 0.36.0 | `sft_backend.py:21-23@main` (enforced) |
| Adapter | PEFT LoRA r=1, alpha=1, dropout 0, `init_lora_weights=True` (A Kaiming-uniform, B zero), 196 targets q/k/v/o/gate/up/down × 28; base frozen; trainable = `lora_*` only | `sft_backend.py:85-97@main`; `docs/methods.md@main` |
| Optimizer | `adamw_torch`, `foreach=False`, LR 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0, clip 1.0 (all trainables) | `sft.py:15-19@main`; `sft_backend.py:250-256@main` |
| Schedule | constant, 0 warmup, 64 steps | same |
| Batching | microbatch 1, accumulation 8 (effective 8), no padding inside a microbatch | same |
| Seeds | `seed=42`, `data_seed=42`, `enable_full_determinism(42)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `sft_backend.py:73,77,258@main` |
| Memory | gradient checkpointing (non-reentrant), `use_cache=False` | `sft_backend.py:93,257@main` |
| Checkpoints | 0, 8, 16, 32, 64 with optimizer/scheduler/RNG; step-64 adapter sha256 `a20a344e…` | `catalog/arms.yaml@catt` (5 hashes, from joint-audit inventory-v2) |
| Hardware | L4 (variant check enforces `'L4' in device`) | `sft_backend.py:114-116@main`; ADR 0004 |

**Conflicts to resolve (ticket 01):**
- `catalog/responses/gold-sft.greedy.test150.pilot2.yaml@catt`: the Pilot 2 evaluation config records adapter `26fdcb24…`, not the inventory step-64 hash `a20a344e…`; flexible-v3 0.647 there vs 0.66 in production-1 Stage D. Catalog status `inferred`, "checkpoint identity unverified". Its step-0 evaluation records `71ee7a33…`, not the zero adapter `b26c524a…`.
- The actual saved `meta/resolved_trainer.json` (TRL resolution of every argument), `inputs/order.json` and `logs/steps.jsonl` have not been read. The status ledger says "original full runtime/config provenance not independently reverified" (`docs/experiments/status.md@main`).

### 2.3 Sampling SFT k0 / k2 / k5

| Area | Value | Source |
|---|---|---|
| Recipe | Gold-guided proposer + k steps of greedy likelihood search + correctness filtering ("Sampling SFT recipe", not on-policy SFT) | `CONTEXT.md@main`; `docs/experiments/sampling-sft.md@main` |
| Data | `answers.jsonl` sha256 `71f09f5c…` (HF `spar-sampling-sft-artifacts`); k0 478 / k2 477 / k5 472 of 512 problems covered (34 / 35 / 40 uncovered) | catalog `sampling-sft-k*.training-data.yaml` (status `inferred`) |
| Order | Gold's 64 batches, original slots; a missing slot is skipped, so steps have 3–8 examples; denominator = that step's actual supervised tokens | `sampling_sft_training_backend.py:161-166@main` |
| Training | Gold primitives: same loss, AdamW 1e-4 constant, clip 1.0, microbatch 1, FP32, TF32 and cuDNN TF32 explicitly off, `highest` matmul precision, L4, from gold's exact checkpoint-0 adapter | `sampling_sft_training_backend.py:1-36,95-110@main`; ADR 0023 |
| Checkpoints | 0 and 64 in the catalog (`a28f52a0…` k2 etc.) | catalog |
| Eval | Greedy, cohort150, A100 FP32 batch 16 (Stage 0 sampling-behaviour), flexible-v3 | catalog `sampling-sft-k*.s64.greedy.test150.yaml` (verified) |

### 2.4 Online GRPO (Pilot 4), arm `online-grpo`

| Area | Value | Source |
|---|---|---|
| Data | Same 512 questions; 64 updates × 8 problems × 8 fresh draws from the current policy (4,096 accepted) | `docs/experiments/pilot-4.md@main`; catalog `online-grpo.training-rollouts` |
| Sampling | T 1.0, top-p 1, top-k 0, rep. penalty 1.0, cap 1,024, EOS 151645 | catalog |
| Objective | DAPO loss, group-scaled rewards, one iteration, beta 0; reward flexible-v3 (scorer sha256 `c958d2de…`), cap → 0 | pilot-4 page; catalog |
| Optimizer | AdamW 1e-4, betas 0.9/0.999, eps 1e-8, constant, no warmup/decay, clip 1 | pilot-4 page |
| Runtime | FP32, eager, TF32 off, A100 80GB | pilot-4 page |
| Checkpoints | 0/8/16/32/64; step 64 `70d2d56c…` | catalog |
| Lineage gaps | Training rollouts "not present locally or in either HF dataset; Drive copy assumed"; "online-grpo lineage is partial (its frozen plan is not local)" | catalog; `adapter-neuron-attribution/spec.md@catt` |

### 2.5 Offline GRPO, arm `offline-grpo`

| Area | Value | Source |
|---|---|---|
| Data | Fixed pool of untuned draws, 512 × 8 = 4,096 (128 historical groups + 384 new), same 64 ordered windows, no importance weights | catalog `offline-grpo.training-pool` (inferred); `.scratch/offline-grpo/spec.md@catt` |
| Update | "Use the verified Pilot 4 plan as the field-by-field reference" (same optimizer, reduction, checkpoints) from the exact checkpoint-0 adapter | `offline-grpo/spec.md@catt` §Implementation decisions |
| Checkpoints | 0/8/16/32/64; step 64 `66015518…` | catalog |
| Gap | Step-64 config hash differs from the original config; recovery of the actual step-64 config was required | `joint-neuron-adapter-audit/spec.md@catt` |

### 2.6 Behaviour evaluation

| Item | Value | Source |
|---|---|---|
| Cohort | `gsm8k-test-cohort150`: `sorted(random.Random(42).sample(range(1319), 150))`, IDs `gsm8k:test:{i}`, digest `79692f50…` | catalog; **recomputed** identical; `protocol.py:9-11@main`; ADR 0001 |
| Decoding | Greedy, `max_new_tokens` 1,024, EOS 151645, rep. penalty 1.0, explicit arguments | catalog policies; C launch record |
| Scorer | `score_gsm8k_flexible_v3(text, gold, capped)`, file sha256 `c958d2de…`; capped → 0; invalid stays in denominator | `pilot_eval/scoring.py:173@main`; ADR 0005 |
| Runtime variants | Pilot 2 L4 batch 2; Stage 0 A100 batch 16; C production A100 batch 16 | ADR 0004; catalog |
| Prior uses | Every arm's evaluations, Pilot 4 sampled ×8, production-1 Stage D, all C rehearsals, **C production attribution, selection and greedy ablation on (nearly) all 150 questions** | `catalog/cohorts.yaml@catt` |
| Status | Not held out for anything. Selection-on-cohort already occurred (C attribution). A new greedy evaluation is a further prior use and must be labelled exploratory | derived |

### 2.7 Existing neuron-level measurements available for comparison

| Study | Arms | Neuron object | Universe | Inputs / metric | Status |
|---|---|---|---|---|---|
| **C-attribution `c-production-1`** | online-grpo, offline-grpo (step 64) | `down_proj` LoRA **read term** j: `dJdλ_j = −removal_J_j`, removal = zeroing `a_j` in `lora_A` | all 28 × 8,960 per arm; per-question `S_q` [150, 28, 8960] FP64 + bootstrap; frozen top-64 global / hot / per-layer | sampled J (flexible-v3 correctness), own sampled ×8 on cohort150, all unpadded positions, leave-one-out weights | done today; outputs in `artifacts/imports/c-attribution-production[-final]/` and HF (not checked here) |
| **Adapter attribution production-1** | all 6 arms (step 64) | `down_proj` read terms of each arm's **top `down_proj` adapter only** | 8,960 terms of one layer per arm (layer per arm not in Git) | trace log-likelihood and stop pull, base-correct traces from `sequences-v2` (GSM8K **train**) discovery split | ran; `artifacts/imports/attribution-production-1/…/report.json` (not checked); ticket status lines not updated |
| **Joint neuron/adapter audit v1** | untuned + 6 arms | descriptive per-neuron lists A (activity selectivity), B (read-term size `|a_j z_j|`), D (activation change `|mean d_ij|`) | all 28 layers × 8,960 | `sequences-v2` discovery math vs background | production v1 done 2026-11-12 (lessons file); outputs not checked |

None of these measures how a neuron's **gate/up reads** should change. The C and production-1 objects are about how the `down_proj` adapter reads neuron j's **activation**. Sparse rewiring selects neurons whose **inputs** are changed. The join key `(layer, neuron)` is the same; the measured objects differ (design input §3 already states this).

### 2.8 Max's method at `max@a6dc314`

| Item | Value | Source |
|---|---|---|
| Arm | `--arm reread`: `MaskedReRead` replaces each layer's MLP; adds `z_i · Δgate_i(x)` and `z_i · Δup_i(x)`; down projection and all base weights frozen | `train/train.py:334-376` |
| Rank parameterisation | r=1: `a[d]` shared per layer per part, `b[n]`; r>1: `a[r,d]`, `b[n,r]`; r=0: full `w[n,d]` per part from zero | `train.py:342-364` |
| Init | `a ~ randn/sqrt(d)` with `torch.Generator().manual_seed(seed + layer)`; `b = 0`; `log_alpha = 1.0` (`--mask-init`) | `train.py:350-355,378-385,539` |
| Hard concrete | β=2/3, γ=−0.1, ζ=1.1; train: `u~U`, `s=σ((log u − log(1−u) + logα)/β)`; eval: `s=σ(logα)`; `z=clip(s(ζ−γ)+γ, 0, 1)`; `expected_on = Σ σ(logα − β log(−γ/ζ))` | `train.py:290-318` |
| Noise | one `z` vector per layer per forward call, shared by all rows and positions of that micro-batch | `train.py:301-314,366-376` |
| Positions | read change applies at every input position (prompt included) | `train.py:366-376` |
| Loss | per micro-batch mean CE over reply tokens, weighted `k/ntok` → token-weighted over the step; plus `λ · Σ expected_on` (split across micro-batches) | `train.py:711-731,860-869` |
| Targets | reply + `"<|im_end|>\n"` (**trailing newline supervised**); `MAX_LEN = 640` with **silent truncation** | `train.py:67,83-84` |
| Optimizer | one AdamW, groups {reads: `--lr`}, {masks: `--mask-lr` 0.1}; betas (0.9, `--beta2` 0.99), eps 1e-8, wd 0; or SGD momentum for masks with `--mask-opt sgd` | `train.py:694-698,559` |
| Clipping | `clip_grad_norm_(params, 1.0)` over **read factors only**; mask gradients unclipped | `train.py:871` |
| Schedule | `LambdaLR` on both groups: linear warmup 20 then cosine to `--steps` (default); `constant` and `plateau` available | `train.py:576-580,700-704` |
| Freeze | at step `steps − freeze_last + 1`: survivors `z_det() > 0` (i.e. `logα > −ln 11 ≈ −2.398`), set `logα=±20`, masks binary, no penalty, no mask updates | `train.py:842-850,865-866` |
| Export | `mask_summary.json`: neurons with `z() > 0` in eval mode (binary after freeze) and their `b` values | `train.py:949-957` |
| Precision | frozen weights `--dtype bfloat16` by default; read computations cast to FP32 and back | `train.py:601,608,366-376` |
| Model mode | `.eval()`, `requires_grad_(False)`, `gradient_checkpointing_enable` called. **Code-inspected risk:** HF decoder layers checkpoint only when `self.training`; the model stays in eval mode, so checkpointing may not have been active. Not verified. | `train.py:608-611` |
| Seeds | `--seed` (train/val split), `--init-seed` (data order, mask noise; default = seed), `--read-seed` (read init; default = init-seed) | `train.py:545,592-603` |
| Validation | 10% of distinct prompts (`VAL_SHARE`) or a supplied `--val` file; val loss with deterministic masks | `train.py:66,70-91,741-751` |
| Published case | `v2_reread_l06e-4_lr3e-3_s2`: rank 4, λ 6e-4, lr 3e-3, 1,400 steps, batch 8, micro 4 (default), freeze last 100, warmup 20 + cosine, mask lr 0.1, beta2 0.99, BF16, Qwen3-8B, 2,081 train / 262 val rows; expected-on 136 at step 1,300 → 89 binary survivors after freeze; val loss 1.2174; 834 s, peak 31.3 GB (L40S) | `README.md:61-68`; `results/train/v2_reread_l06e-4_lr3e-3_s2/train_log.json`, `mask_summary.json` |
| Rank caveat | Rank 4 was an **oversight**; the **intended** method is rank 0 (full rows), which needs a lower LR (3e-4; 3e-3 overfits) | `RANK0.md` |
| Rank 1 | Run names `reread_r1_l06e-4_lr3e-3_s{0,1,2}` are defined; **no rank-1 (or rank-16) result directory exists** in `results/train/` (62 run directories checked) | `train/modal_runs.py:42-44`; directory listing |
| Paper | None found in the repository; the related method paper is Louizos, Welling and Kingma 2018 (hard concrete / L0), not Max's | README; design input |

### 2.9 License and provenance

| Material | License | Consequence |
|---|---|---|
| `train/train.py` (upstream) | Apache-2.0, vendored from HF `Sauers/Qwen3-8B-tawny-owl-128-neurons` (`training/`) | Reusable with notice; the exact upstream HF revision is **not recorded** in `SOURCE.md` (pin it in ticket 03) |
| Max's local changes (reread, gains, add, …) | **No license file** in the repository; `SOURCE.md` lists changes but grants nothing for them; no SPDX headers | Default copyright applies. `MaskedReRead`/`setup_reread` and the freeze logic may be Max's own additions. **Do not vendor or copy code** without written permission or a license. A clean-room reimplementation from the documented method (and Louizos et al. 2018) is the safe route; cite Max's repo and commit |
| Upstream dataset (owl data) | CC BY-NC 4.0 | Not used here |
| neuron-lens labels | CC BY 4.0 | Not used here |

Recommendation **[proposed]**: reimplement in `pilot_eval/` from the method description, with a test suite that checks equivalence to Max's equations on fixtures (not to his code), and ask Max for a license if byte-for-byte reuse is wanted (decision D16).

## 3. Filled table (design input §5)

| Area | Reconstructed existing value | Relationship for the new run |
|---|---|---|
| Model/tokenizer | §2.1 | Same revision; shape verified from loaded config; zero-based indexing tested |
| Training IDs and ordering | §2.2: 512 IDs (digest recomputed), order from seed-42 shuffle, one epoch | Steps 1–64 use gold's exact 64 batches **[proposed]**; later epochs see later orders (D7) |
| Gold answers | Raw GSM8K answer, Qwen template with default system message, `<|im_end|>` supervised, trailing newline not | Reuse the frozen `training.items.json` token IDs (not re-tokenized) |
| Loss and masks | Shifted CE summed / window supervised tokens | Same; Max's reduction is mathematically the same (token-weighted per step); Max's trailing-newline label and 640 truncation are **not** adopted (D9) |
| Sequence limits | `max_length` = longest real example; no truncation | Same; preflight the longest example on the target GPU |
| Optimizer/runtime | FP32, eager, TF32 off (legacy flag), AdamW 0.9/0.999 1e-4, clip 1, microbatch 1 × 8 | FP32 + explicit IEEE precision (`neuron_audit_collector.set_ieee_precision@catt`, lesson 2026-11-19); optimizer per D1–D4 |
| Steps/seeds | 64 steps, seed 42, checkpoints 0/8/16/32/64 | Keep 0/8/16/32/64; add later checkpoints (D5); separate seeds (D10) |
| Existing LoRA | r1 α1, 7 projection types, PEFT init (A random, B zero), both factors trained | Not reused; rewiring arm has no LoRA attached. Differences listed in `decision-log.md` §Not equivalent |
| Behaviour evaluation | §2.6 | Same protocol; the cohort is not held out (D12) |
| Attribution | §2.7 | Now: C `c-production-1` per-read scores (full universe) and production-1 adapter attribution (one layer), GRPO arms first; other arms later as manifest entries (D19) |
| Prior exposure | `catalog/cohorts.yaml@catt` | Register every new use in the catalog when it happens |
| Validation | **None exists** for this purpose | New GSM8K **train** validation set disjoint from the 512 and from `sequences-v2` (D11) |
| Compute/artifacts | No measurement for this arm | Benchmark first (ticket 15); estimates in `decision-log.md` are arithmetic, not measurements |

## 4. Decision register

Full option tables are in `decision-log.md`. Summary:

| # | Decision | Recommendation | Status |
|---|---|---|---|
| D1 | Read-update LR | rank 1: 3e-3 (Max's published value); rank 0: 3e-4 (`RANK0.md`); 1e-4 (gold) as one sensitivity run | [proposed] |
| D2 | Mask optimizer | Max's: AdamW group, lr 0.1, same schedule as reads, unclipped | [proposed] |
| D3 | Betas | Max's (0.9, 0.99) for both groups; gold's 0.999 recorded as a difference | [proposed] |
| D4 | Clipping | 1.0 on read factors only (both projects clip at 1.0; masks unclipped as in Max) | [proposed] |
| D5 | Steps and schedule | 1,024 steps for both ranks, 20-step warmup + cosine to 1,024, freeze at 960, final 64 fixed-mask; settle metrics logged as diagnostics only; checkpoints 0/8/16/32/64/128/256/512/768/960/1024 | [user 2026-11-23] |
| D6 | λ | 6e-4 for both ranks, no sweep; a degenerate run is reported, not re-run | [user 2026-11-23] |
| D7 | Epoch order | Epoch 1 = gold's 64 batches exactly; epoch e>1 = gold IDs reshuffled with seed `42 + e − 1` | [proposed] |
| D8 | Precision | FP32 frozen weights, IEEE FP32 everywhere, no autocast, eager attention | [proposed] (matches gold) |
| D9 | Targets/truncation | Gold contract (no trailing newline, no truncation) | [proposed] |
| D10 | Seeds | data order = gold (42); read init seed 42 + layer; mask noise seed 1042 | [proposed] |
| D11 | Validation set | 128 GSM8K train questions, seeded, excluding the 512, `sequences-v2` and any other recorded train use; gold CE + greedy accuracy | [proposed]; blocked on `sequences-v2` IDs |
| D12 | Cohort150 use | Evaluation only, at 64 and final, exploratory; circular when combined with C overlap | [user 2026-11-23] |
| D13 | Rank | Both rank 1 and rank 0 (full rows); everything else identical | **[user 2026-11-23]** |
| D14 | Microbatch and mask noise | Microbatch 1 (gold), with one cached mask draw shared by each group of 4 examples: 2 draws per step, as in Max | [user 2026-11-23] |
| D15 | Gradient checkpointing | On (gold), RNG-preserving recompute tested | [proposed] |
| D16 | Code reuse | Clean-room reimplementation; ask Max for a license if copying is wanted | [proposed] |
| D17 | Code base | Implement on `main` after the attribution branch merges; otherwise branch from `c-attribution-pilot` | [proposed] |
| D18 | Hardware | One A100-80GB for both arms | [user 2026-11-23] |
| D19 | Comparison family | Now: survivors vs C attribution and production-1 adapter attribution, GRPO arms first; code generic over arms via a source manifest so other arms' C attribution is added later | [user 2026-11-23] |
| D21 | Rank comparison | Survivor Jaccard and per-layer overlap, overlap with C sets, paired accuracy, val CE, counts, dynamics side by side | [proposed] |
| D20 | Dynamics logging | Researcher-requested figure: gates, read-change norms and performance against step; cheap metrics every step, validation CE every 16 steps, greedy at checkpoints | [user 2026-11-23] |
| D22 | Standalone analysis | Selection, what survivors learned, behaviour, for each rank | [user 2026-11-23] scope; definitions [proposed] |
| D24 | Checkpoint storage | Read factors and `logα` only at every checkpoint; no optimizer state; a crash restarts the arm from step 0 | [user 2026-11-23] |
| D25 | A100 ADR | A100-80GB supersedes ADR 0018's L4 training rule for the sparse arms; ADR text in ticket 20 | [user 2026-11-23] approved in substance |
| D23 | Causal checks | Revert all, single and block reverts, same-layer transplant control | skipped for the pilot [user 2026-11-23]; later follow-up |

## 5. Recommended minimal pilot (once separately authorised)

Two sparse gold-SFT arms, **rank 1 and rank 0 [user 2026-11-23]**, from the untuned model, identical except rank and read LR. Each uses the 512 gold examples with gold's loss/token contract and FP32 runtime, Max's hard-concrete/L0 objective and mask optimizer with one mask draw per 4 examples (D14), and λ 6e-4 with no sweep (D6). Read LR is 3e-3 for rank 1 and 3e-4 for rank 0 [proposed]. Training runs 1,024 steps with the masks frozen at 960 (D5). The first 64 steps are gold's exact batches. Checkpoints are at 0/8/16/32/64/128/256/512/768/960/1024. Nothing in training is chosen on an evaluation set; cohort150 is evaluation only (D12). Cheap metrics are logged every step, validation CE every 16 steps (D20). Each arm gets a greedy cohort150 evaluation at step 64 and the final step. On CPU: a standalone analysis of each rank (D22), a rank-1 vs rank-0 comparison (D21), and a comparison of each with C attribution and production-1 adapter attribution for the GRPO arms (D19). Estimated GPU time: about 5–5.5 h for both arms in sequence on one A100 (arithmetic estimate, `decision-log.md` §Compute). No λ sweep, no rank 4, no second seed. Causal checks (D23) are a later follow-up.

## 6. Answers to the design input's questions (§9)

1. **Shared IDs and restrictions.** Training: all arms use `gsm8k-train-512` (digest verified). Evaluation and C attribution share `gsm8k-test-cohort150`, which is not held out. Production-1 attribution and the joint audit used `sequences-v2` GSM8K **train** questions (IDs only in artifacts). Unknown lineage: online-GRPO rollouts and frozen plan (not local), offline step-64 config, the 128 historical offline groups' generator.
2. **Gold SFT's actual settings.** §2.2. Differences in Max: trailing-newline label, 640 truncation, BF16, betas 0.99, warmup + cosine, mask group, unclipped masks, micro 4.
3. **What attribution produces.** C: full 28 × 8,960 `down_proj` read-term J gradients for the two GRPO arms. Production-1: one layer per arm, trace/stop metrics, all 6 arms. Joint audit: descriptive A/B/D lists for all layers and arms. None is a gate/up-read measure.
4. **Smallest budget.** Not measured. Arithmetic estimate in `decision-log.md`; ticket 15 measures it.
5. **Researcher decisions before a launchable spec.** D1, D11 (and its data), D21, D22 definitions. Decided 2026-11-23: D5 (1,024 fixed), D6, D12, D13, D18, D19 scope, D20 cadence.
6. **Max's rank-1 result.** Unavailable: configured in `modal_runs.py`, no result directory, no rank-1 row in `RANK0.md`.
7. **Reusable code.** In this repo: `sft.load_sft` and the frozen `training.items.json`; `sft_backend.completion_loss`, `CompletionCollator`, `frozen_weight_hash`; the explicit loop and recovery pattern in `sampling_sft_training_backend.train`; `neuron_audit_collector.set_ieee_precision@catt`; `scoring.score_gsm8k_flexible_v3`; the greedy evaluation used by Stage 0 / C generation; `catalog` register/verify; C-attribution `aggregate/<arm>/neurons.npz` and freeze files as comparison inputs. From Max: the method, not the code (§2.9).
