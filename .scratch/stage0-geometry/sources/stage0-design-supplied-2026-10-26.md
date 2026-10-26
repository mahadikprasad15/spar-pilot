# Stage 0 Design Doc: Geometry Experiments on Existing Adapters

Oct 26, 2026 · @Prasad Mahadik

## Summary

Stage 0 runs seven jobs on the adapters Pilots 2 and 4 already produced. The only new training-free generation is the missing base-model rollouts. The work splits cleanly by resource:

- **The GPU notebook** produces raw tensors: rollouts, teacher-forced log-probs, gradients at the base model, activations, KL and the E0.4 evaluations.
- **The CPU notebook** does all the analysis: SVDs, π, rotation, intruders, bootstrap intervals and plots.

Run order follows the data dependencies, not the causal chain, with one exception. The E0.5 prediction is written down before E0.2's π is looked at.

| Job | Answers | Resource | Depends on |
| --- | --- | --- | --- |
| R0 regenerate rollouts | input to E0.6a, E0.5 | GPU, generation | baseline-pass settings |
| S0 SVD cache | input to E0.2, E0.3, E0.4, E0.5 | CPU | base weights only |
| E0.6a dataset distance | C2 x-axis | GPU forward, then CPU | R0 |
| E0.5 push at base model | C2, predicts C3 | GPU forward+backward, then CPU | R0, S0 |
| E0.2 where b writes | C3 | CPU (plus one GPU activation pass) | S0 |
| E0.3 rotation and intruders | C3, picks E0.4's edit | CPU | S0 |
| E0.6b drift vs update size | C1, C5 | GPU forward (may already exist) | checkpoints |
| E0.4 edits and controls | C5 | CPU builds edits, GPU evaluates | S0, E0.2, E0.3 |
| E0.1 length interval | C1 | CPU | saved evaluations |

Everything here uses gold SFT (Pilot 2) and GRPO (Pilot 4), seed 42, rank 1, FP32. Sampling SFT and self-correct SFT as trained arms belong to Stage 2.

## What we have

The sources are the [Pilot 4 design doc](https://claude.ai/artifact/B2BkGcGwkYWtUJ1t4NTrwo) and the run manifests and folders in Drive. The doc gives intent; the manifests give what actually ran. Rows marked **check** are designed but not yet confirmed on disk.

| Artifact | Where | What it holds | Used by | Status |
| --- | --- | --- | --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct`, base hash recorded in the source audit | 196 linear modules (28 layers × q, k, v, o, gate, up, down) | all | have |
| SFT adapters | `runs/pilot-2/.../train-512-seed-42/rank1-float32/pilot2-sft-fp32-l4-batch8-v1/checkpoints/checkpoint-{0,8,16,32,64}` | rank 1, α 1, dropout 0, all 7 projections; trained FP32 on an L4, 8 problems per step | E0.2–E0.6b | have |
| GRPO adapters | `runs/pilot-4/.../pilot4-grpo-fp32-v1-arm-v1/checkpoints/checkpoint-{0,8,16,32,64}` | same adapter; G = 8, 8 prompts × 8 = 64 completions per step, T = 1, β = 0, num\_iterations 1, scale\_rewards = group, HF generation, FP32 on A100 | E0.2–E0.6b | have |
| Starting a₀ | checkpoint-0 of each arm (B = 0, A = a₀) | the exact initial read vectors | E0.5, E0.2 baseline | have; **check** SFT and GRPO a₀ are identical (same seed 42) |
| Random control adapter | `runs/pilot-4/.../random-fixed-writes/pilot4-grpo-fp32-v1-random-writes-v1/control-adapter` | random rank-1 adapter matched to GRPO's size | E0.2, E0.4 controls | have |
| Gold solutions | the 512 training indices in the run config | human solutions with `#### n` | E0.6a, E0.5 | have |
| Baseline pass | Pilot 4 baseline pass | untrained model, 8 samples per problem, T = 1, training length limit, both extractors | E0.6a, E0.5 | have for 128 problems; **check** location, engine, precision |
| Rollout table | Pilot 4 run, every step | 4,096 rows: step, problem, group, text, tokens, length, truncation, rewards, advantage | not used for distance (policy drifts) | have |
| Eval items | `plans/pilot2-sft-fp32-l4-batch8-v1/evaluation.items.json`; `paired-items.jsonl` (22 Oct folder) | 150 held-out GSM8K, per-item results, 13 human-confirmed corrections | E0.1, E0.4 | have |
| Fixed token sets | `runs/pilot-3/.../heldout150-control150-seed42/float32/pilot3-inputs-v1` | 150 GSM8K gold solutions teacher-forced, plus 150 FineWeb docs (sample-10BT, revision 9bb295dd, 128 content tokens each) | E0.2 activations, E0.6b, E0.4 | have |
| Write measurements | `pilot4-grpo-fp32-v1-{sft,grpo,random}-write-report-a10040-v2` | effective write per module, mean Δh, per-token coefficients aᵀx; all three remeasured on one A100 40GB | E0.6b x-axis | have |
| KL to base | designed for every checkpoint; possible home: `pilot4-grpo-fp32-v1-supplemental-tokens-a10040-v2` | KL(tuned ‖ base) on the fixed sets | E0.6b | **check** whether computed, and for both arms |

**Choices from earlier pilots that bind Stage 0:**

- **FP32 everywhere.** A BF16 → FP32 switch alone moved untuned accuracy by 25 points.
- **One A100 40GB setup** for every internal measurement. SFT trained on an L4, but its internals were remeasured on the A100. Keep every new GPU number on the same A100 type.
- **The flexible extractor defines "correct"** for rewards and evaluation. Strict is logged alongside.
- **The same 512 training problems** in both arms, each problem used once.
- **Paired bootstrap:** 10,000 draws, percentile, seed 42 (from the GRPO config). Pilot 3 used 2,000 replicates resampling examples.

## Shared settings and matching rules

Every job follows these rules. A violation makes its numbers incomparable with the pilots.

| Setting | Value | Why |
| --- | --- | --- |
| Precision | FP32 weights and compute; TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, same for cudnn) | TF32 changes results at about 1e-3, larger than the GPU-type differences |
| GPU | one GPU type within each experiment; A100 40GB wherever results are joined to the saved Pilot 3 A100 measurements (E0.6b), any GPU otherwise | matches the Pilot 3 remeasurement (a10040-v2) |
| Chat template, prompt | exactly as in the run manifests | the template changes token ids and log-probs |
| Correctness | flexible-v3, the corrected extractor (v2 rejected the gold #### format); strict logged; also record v2, in case GRPO's training reward used it | one definition of correct |
| Loss averaging | mean over answer tokens in the batch, prompt masked; match Pilot 2's SFT averaging | gold and own answers differ in length |
| Adapter loading | PEFT with the saved adapter configs; never re-instantiate by hand | keeps α, scaling and module names identical |
| Layout | `runs/stage0/<job>/...` with a manifest: inputs and hashes, git commit, library versions, GPU, dtype, seed | the same audit trail as Pilots 2–4 |
| Statistics | paired bootstrap over items, 10,000 draws, seed 42, percentile 95% | matches the GRPO config |
| Comparisons | always against chance, the init adapter, and the random control adapter | separates training from geometry that exists by default |

**Two naming conventions used below:**

- **U\_k, V\_k:** the top-k left and right singular vectors of a base weight W₀, with k the smallest number holding at least 50% of Σσ².
- **a, b:** each module's rank-1 adapter as stored in PEFT. `lora_A` is a 1 × d\_in row, so a = Aᵀ. `lora_B` is d\_out × 1, so b = B. The scaling α/r = 1 multiplies b aᵀ.

## R0: regenerate the missing base-model rollouts

The Pilot 4 baseline pass sampled 128 of the 512 training problems; the design doc deferred the other 384 to X1. R0 completes it. The new 384 must be indistinguishable in procedure from the existing 128, so the first step is reading the old settings, not choosing new ones.

**Why the rollout table can't stand in.** In training, problem i was sampled by the model as it was at step ⌈i/8⌉. Only the first 8 problems came from the untouched model.

1. **Read the old settings** from the baseline-pass manifest: problem indices, prompt and template, temperature (1.0), max new tokens, samples per problem (8), generation engine (HF or vLLM), dtype, sampling seed scheme, both extractor versions. The design doc allowed vLLM here, so do not assume HF.
2. **Confirm the 128 are inside the 512.** Intersect the baseline indices with the training indices from the Pilot 2 and Pilot 4 configs. All 128 should match.
3. **Generate the other 384 × 8 = 3,072 samples** with identical settings and the same engine and dtype. Seed each problem deterministically, for example `seed = 42 + problem_index`, and record it.
4. **Score** with flexible-v3 (it defines correct for every Stage 0 dataset), plus v2 and strict recorded alongside, and the same reward function (truncated answer → reward 0, as `capped_reward = 0`).
5. **Merge** into one file with the old schema: problem index, sample index, text, token ids, length, truncated flag, both rewards, source (old/new).
6. **Checks:**
   - 512 distinct problems, 8 samples each, no duplicates.
   - Truncated fraction below about 2%; otherwise the max length was too low.
   - Mean reward on the new 384 is close to the old 128. A gap of more than a few points suggests a setting mismatch, since the problems are random draws.
   - Re-generate 2 of the old problems with their recorded seeds and confirm identical text (HF) or matching reward rates (vLLM).

**Derived datasets** (used by E0.6a and E0.5):

| Name | Content | Stands for |
| --- | --- | --- |
| `gold` | the 512 gold solutions | gold SFT's data |
| `own_all` | all 4,096 samples | GRPO's raw data; distance near 0 |
| `own_correct` | samples with flexible reward 1 | what GRPO pushes up; future self-correct SFT data |
| `own_correct_one` | one correct sample per problem, chosen at random with seed 42 | a one-answer-per-problem match to `gold` |

Record **P(correct) per problem** as k/8. Problems with no correct sample drop out of the `own_correct` sets; record how many.

**Cost.** 3,072 generations in FP32 on one A100: about 15–40 minutes with HF generate, less with vLLM.

## S0: SVD cache of the base weights

E0.2, E0.3, E0.4 and E0.5 all need the same principal directions, so compute them once and save them. This needs only the base weights, not the model, so it runs on CPU from the safetensors files.

1. Load each of the 196 weight matrices W₀ in FP32. Shapes: q and o 1536 × 1536; k and v 256 × 1536; gate and up 8960 × 1536; down 1536 × 8960. Check each against the model config.
2. Full SVD: W₀ = U Σ Vᵀ (`torch.linalg.svd(W, full_matrices=False)`, float64 for the decomposition).
3. k = the smallest number of singular values with Σᵢ≤ₖ σᵢ² ≥ 0.5 Σ σ². Also save k at 30% and 70% as a sensitivity check.
4. Save per module: σ (all), U (all columns, up to 1536), V (all), k₅₀, and chance levels k/d\_out and k/d\_in.
5. **Checks:** reconstruction error ‖W₀ − UΣVᵀ‖/‖W₀‖ below 1e-6; UᵀU = I; the base-weight hash matches the source audit.

**Size.** Storing every singular vector of all 196 matrices would take about 6.6 GB. Most of that is directions nobody uses: the analyses only need the principal ones, the top k that hold 50% of the energy, usually a small part of each matrix. So save all the singular values (tiny), and only the top k₇₀ columns of U and V, which covers both k₅₀ and the 70% sensitivity check. If anything ever needs more columns, recompute that module's SVD; it takes seconds.

**Cost.** The largest SVDs (8960 × 1536) take seconds each on CPU; all 196 take a few minutes.

## E0.6a: dataset distance

**Question (C2, x-axis).** How far is each training dataset from the base model? The four datasets from R0 get a number each, and every later plot uses it as the x-axis.

**What it computes.** For each answer, at each answer token t, with the base model teacher-forced on the prompt plus the answer so far:

- surprise: −log p(yₜ)
- entropy: Hₜ = −Σᵥ p(v) log p(v), over the full vocabulary

Per dataset, average both per token. Distance = mean surprise − mean entropy.

```latex
\text{NLL} = \underbrace{\bar H}_{\text{model's own uncertainty}} + \underbrace{\mathrm{KL}(\text{data} \,\|\, \text{model})}_{\text{distance}}
```

**Steps.**

1. **GPU:** for every answer in `gold`, `own_all`, `own_correct` and `own_correct_one`, run one teacher-forced forward pass of the base model (no adapter). Save per token: token id, −log p(yₜ), Hₜ, and the rank of yₜ. Mask the prompt exactly as SFT did.
2. **CPU:** aggregate per dataset, per problem, and by position in the answer (first 32 tokens, middle, last 32).
3. **CPU:** for `own_correct`, compare its distance with the mean of −ln P(correct) over problems. They should roughly agree.

**Outputs.** Per dataset: NLL, entropy and distance per token, with bootstrap intervals over problems. Per problem: gold distance and P(correct). A table of the 50 gold tokens with the highest surprise.

**Prediction (write down before running).**

- `own_all` distance near 0, a code check.
- `own_correct` small, near the mean of −ln P(correct).
- `gold` much larger.
- Gold's surprise concentrated in format tokens (`####`, the final line, early stopping) as well as reasoning-style tokens.

**Comparability notes.**

- Per-token averaging matters: gold answers are shorter (median about 110 tokens vs about 196 for the model's own).
- `own_correct` has up to 8 answers per problem, weighting easy problems more. `own_correct_one` corrects that.

**Cost.** About 1,000 answers for `gold` and `own_correct_one`, plus up to 4,096 for `own_all`. All forward-only on the A100: about 10–20 minutes. Storage is per-token floats, a few tens of MB.

## E0.5: push at the base model

**Question (C2, and a prediction for C3).** Before any training, does gold data push the adapter harder, more consistently, and more into the principal directions than the model's own correct answers? And does that first push already point where the trained b ended up?

**What it computes.** With a = a₀ and b = 0, the gradient PyTorch reports for b is exactly G a₀ = Σₜ δₜ (xₜ · a₀): b's first push. No full G matrix is ever built. The gradient for a is exactly zero at this point, which is a free check.

**Steps.**

1. **GPU: load each arm's checkpoint-0** as a trainable PEFT model in FP32, dropout 0, `model.train()` off (eval mode, so nothing random). Confirm b = 0 and that the a-gradient comes out zero.
2. **GPU: build mini-batches** of 8 problems, the size Pilot 2 used per step. Fix the batch composition with seed 42 and save it.
3. **GPU, SFT loss:** per batch, compute the masked per-token mean cross-entropy on `gold`, then on `own_correct_one`, and backpropagate. Save b's gradient for all 196 modules, per batch. Never take an optimizer step. Zero gradients between batches.
4. **GPU, GRPO loss at step 0:** on the R0 rollouts (8 samples per problem), compute group advantages as Pilot 4 did: reward minus group mean, divided by group std, with all-correct and all-wrong groups giving 0. Then take the loss −Σ A·log p averaged per token as Pilot 4's loss type did, and save b's gradient per batch of 8 problems.
5. **CPU, size:** the norm of the mean push per module, for gold, own-correct and GRPO.
6. **CPU, consistency:** ‖mean push‖ divided by the mean of ‖batch push‖ per module. 1 means the batches agree; near 0 means they cancel.
7. **CPU, direction:** π\_out(mean push) = ‖U\_kᵀ g‖²/‖g‖² against chance k/d\_out.
8. **CPU, link to training:** cos(gold push, b\_SFT-64), cos(sign(gold push), b\_SFT-64), and cos(GRPO push, b\_GRPO-64), each against the cosine with a random vector of the same dimension.

**Order rule.** Do step 7 and write down the predicted SFT-vs-GRPO π\_out ordering **before** opening E0.2's π for the trained adapters.

**Outputs.** A per-module table with columns module, layer, type, dataset, size, consistency, π\_out, chance, and cosine with the trained b. Plus summaries by layer and module type.

**Prediction.**

- Gold push larger than own-correct push, and more consistent.
- Gold π\_out above own-correct's and above chance.
- GRPO push smallest and least consistent.
- If the a₀ vectors are identical across arms, any direction difference comes only from the data and loss.

**Comparability notes.**

- Use `own_correct_one`, so both datasets have one answer per problem on the same problems.
- Restrict to problems with at least one correct sample, and use the same problems for gold.

**Cost.** About 192 batches (64 per loss) of forward plus backward in FP32. A 1.5B model with only adapter gradients fits on the A100 with micro-batches of 2–4. Under 30 minutes in total. Storage: the b-gradient vectors are d\_out floats per module per batch, about 0.5 GB for all three losses; keep the per-batch vectors.

## E0.2: where each adapter writes

**Question (C3).** Does the trained b of SFT-64 point into the base layer's principal output directions more than GRPO-64's b does?

**Steps.**

1. **CPU:** load a and b for all 196 modules from SFT-64, GRPO-64, both checkpoint-0s, and the random control adapter.
2. **CPU:** compute π\_out(b) = ‖U\_kᵀ b‖²/‖b‖² and π\_in(a) = ‖V\_kᵀ a‖²/‖a‖² with the k from S0. Repeat at the 30% and 70% energy levels.
3. **CPU, baselines per module:**
   - chance k/d;
   - a at init (checkpoint-0) for π\_in;
   - the random control adapter;
   - 100 random Gaussian vectors, for the 5th–95th percentile band.
4. **CPU, trajectory:** repeat steps 2–3 for checkpoints 8, 16 and 32 of both arms. GCPO reports that spikes in principal overlap precede damage, so the path matters, not only the endpoint.
5. **GPU, activation-weighted input share:** run the base model on the fixed token sets (150 GSM8K gold, 150 FineWeb). Hook every module's input x, and accumulate per module: E\[(a·x)²\] and E\[(a∥·x)²\], where a∥ = V\_k V\_kᵀ a is a's principal part. Save the two sums per module, arm and token set, not the activations.
6. **CPU:** activation share = E\[(a∥·x)²\]/E\[(a·x)²\]. Report it for SFT, GRPO, a at init, and the random adapter.
7. **CPU:** summarize by layer and module type; plot π\_out(SFT) against π\_out(GRPO) per module, with the diagonal.

**Why step 5 is needed.** π\_in measures where a points. What matters for damage is how much of a's firing a·x comes through the principal directions on real inputs. Real activations are large along those directions, so even a random a fires mostly through them. Comparing with a at init separates what training learned from what the geometry gives for free. The cross-term E\[(a∥·x)(a⊥·x)\] means the two parts don't add up exactly; report the sum of the parts too.

**Outputs.** A per-module table of π\_out, π\_in, activation share, baselines and checkpoint. A scatter of SFT vs GRPO. Layer profiles.

**Prediction.**

- π\_in near chance for every arm, because a barely moved (cosine with init 0.967 for SFT, 0.987 for GRPO).
- π\_out for SFT above chance and above GRPO in most modules.
- Write down which module types first.

**Limit.** With one seed per arm, this is a first look; the seed noise floor comes in Stage 1. The random control adapter gives a size-matched null now.

**Cost.** Steps 1–4 and 6–7 run on CPU in minutes. Step 5 is one forward pass over 300 sequences per arm, about 5 minutes on the A100.

## E0.3: rotation and intruders

**Question (C3, and which edit E0.4 runs).** Did the adapter write into the base's existing main channels (SPACE's picture), or create a new strong channel (an intruder, Shuttleworth et al.'s picture)? No SPACE edit is involved here; this only measures.

**Steps (all CPU).**

1. **Size check per module:** s = ‖b‖‖a‖/σ\_k. A rank-1 update can reorder the spectrum only if it is comparable to the top singular values. If s < 0.1 in every module, intruders are impossible; report that and skip step 3.
2. **SVD of the trained weight** W = W₀ + b aᵀ, using the same full SVD as S0. Each matrix fits in memory and takes seconds, so no approximation is needed.
3. **Intruders:** for each of W's top 20 left singular vectors, take its best |cosine| with any of W₀'s left singular vectors (all of them, not only U\_k). Flag it as an intruder if that is below 0.5. Report the threshold sweep 0.3–0.7, as Shuttleworth et al. do.
4. **Rotation:** principal angles between span(U\_k of W₀) and span(U\_k of W); report the mean sin² and the largest angle. Do the same for the input side with V\_k.
5. **Repeat** for SFT-64, GRPO-64, the random control adapter, and the checkpoints 8–32 of both arms.

**Decision for E0.4.**

| E0.3 finds | E0.4 runs |
| --- | --- |
| s small everywhere, no intruders | the SPACE output- and input-side edits with controls |
| intruders in SFT, not GRPO | the SPACE edits plus intruder scaling |
| intruders in both arms | intruder scaling on both, as a pair |
| almost no rotation anywhere | the SPACE edits, expecting small effects; E0.6b becomes the main damage readout |

**Prediction.** s is small: a 64-step rank-1 update is tiny next to σ\_k, so expect no intruders and rotation of a fraction of a degree. If so, intruders are a full-fine-tuning or higher-rank phenomenon, and the SPACE picture is the one Stage 0 can test.

**Cost.** Minutes on CPU.

## E0.6b: drift on FineWeb vs update size

**Question (C1, C5).** How much did each adapter change the model on text it never trained on? Is SFT's drift larger than GRPO's at the same update size, or only because SFT's update is about twice as large?

**First, check what exists.** Pilot 4 was designed to save KL to base at every checkpoint on the fixed sets, and the Pilot 3 write reports have the effective write for all three products on the A100. If KL exists for both arms, at all five checkpoints, on the same FineWeb token set, E0.6b is pure CPU analysis.

**Steps (only what is missing).**

1. **GPU:** for each of the 10 checkpoints (2 arms × steps 0, 8, 16, 32, 64) and the random control adapter, teacher-force the 150 FineWeb docs (128 content tokens each) through the base model and the adapter model. At every content position compute KL(p\_adapter ‖ p\_base) over the full vocabulary, in float64 accumulation. Save per position.
2. **GPU, same pass:** the same on the 150 GSM8K gold solutions, as a task-text comparison.
3. **CPU:** mean KL per document, then the bootstrap over documents.
4. **CPU:** join with update size per checkpoint: the effective write ‖ΔW x‖/‖W x‖ from the Pilot 3 reports, plus the plain ‖b‖‖a‖ summed over modules.
5. **CPU:** plot KL (and, where available, greedy length change) against update size, one curve per arm, with the random adapter as a point.

**Checks.**

- Checkpoint 0 gives KL exactly 0, since b = 0.
- KL is never negative.
- Base log-probs match between runs to about 1e-6.

**Reading.**

- The curves overlap: size explains the arm difference.
- SFT sits above GRPO at the same size: SFT's update does extra damage per unit of change, and direction is the suspect (C5).
- The random adapter sits far below both at its size: direction matters, as the earlier 17–145× write result suggested.

**Cost.** 11 adapter states × 300 sequences of about 128–250 tokens, forward only: about 10 minutes on the A100. Storing full-vocabulary log-probs is not needed; keep per-position KL.

## E0.4: edits and controls

**Question (C5, causal).** If the principal part of SFT's adapter is deleted, do shortening and drift go away while accuracy stays? E0.2 can only show correlation; E0.4 changes one thing and re-measures.

**Variants** (built on CPU, evaluated on GPU):

| Id | Edit, applied to all 196 modules | Role |
| --- | --- | --- |
| V0a | SFT-64 unedited | reference, re-evaluated in this session |
| V0b | GRPO-64 unedited | reference, re-evaluated in this session |
| V1 | SFT, output side: b → b − U\_k U\_kᵀ b | stop writing into main channels |
| V2 | SFT, input side: a → a − V\_k V\_kᵀ a (SPACE's own edit) | stop firing on ordinary inputs |
| V3 | SFT, delete k random orthonormal directions on the output side (3 draws) | is the principal part special? |
| V4 | SFT with b scaled to V1's norm, direction kept | is it size, not direction? |
| V5 | GRPO, output side as V1 | should change little |
| V6 | SFT, shrink the intruder singular value by half (only if E0.3 finds intruders) | LoRA's own damage signature |

**Steps.**

1. **CPU:** build each variant's adapter by editing a or b in the saved PEFT tensors, and save it with a manifest naming the edit and k.
2. **CPU check:** an edit with k = 0 reproduces the original tensors byte for byte, and ‖U\_kᵀ b\_V1‖ is about 0.
3. **GPU:** greedy evaluation of every variant on the 150 items, with the exact Pilot 1 harness, max length and stop rule. Save per-item text, length, strict and flexible correctness.
4. **GPU:** FineWeb KL to the base for every variant, as in E0.6b.
5. **CPU:** paired comparisons against V0a: accuracy (bootstrap and McNemar), median length difference per item, and KL difference per document.
6. **CPU:** the 13 human corrections apply only to the original outputs. For edited variants, flag every item whose flexible extraction disagrees with strict, and hand-check those.

**Why V0a and V0b are re-evaluated.** SFT was evaluated on an L4 and GRPO on an A100. Every variant and both references must run on the same A100 in the same session, so the comparisons are paired on hardware too. Confirm that V0a reproduces the saved SFT-64 per-item results; if it doesn't, the difference is itself a finding about the evaluation.

**Reading.**

| Result | Meaning |
| --- | --- |
| V1 or V2 restores length and lowers KL, accuracy holds, V3 and V4 don't do the same | the principal part causes the damage and is separable from the lesson |
| Damage falls and accuracy falls too | lesson and damage are entangled |
| Nothing changes | the principal part is not where the damage lives |
| V3 or V4 does as well | size, not direction |
| V5 changes GRPO a lot | the edit does something unrelated to the SFT-vs-RL story |

**Cost.** About 9–11 variants × 150 greedy generations in FP32, roughly 1,500 generations plus KL passes: about 30–60 minutes on the A100.

## E0.1 remainder: length interval

The accuracy part is done (GRPO − SFT +11.3 \[+3.2, +19.5\]; SFT − base −7.3 \[−15.2, +0.5\]). These intervals used the normal approximation on the flip counts. Two pieces remain, both CPU only, from `paired-items.jsonl`:

1. **Confirm accuracy** with the paired bootstrap: 10,000 draws over items, seed 42, applying only the 13 human-confirmed corrections.
2. **Length:** per item, length(SFT) − length(base) and length(GRPO) − length(base), in tokens with the same tokenizer. Report the median paired difference with a bootstrap interval, plus the share of items that got shorter. Use the median, because lengths are skewed and truncation caps the long end.

**Prediction.** SFT's length drop has an interval far from 0 (about −85 tokens at the median). GRPO's includes 0.

## Compute plan

Group by resource, not by causal order. The GPU does only what needs a model in memory: generation, forward passes and gradients. It saves raw tensors to Drive. Everything that is arithmetic on saved tensors runs in a CPU notebook. Causal order still decides one thing: the E0.5 prediction is frozen before E0.2's π is opened.

&#91;embedded content: compute plan · 2 notebooks, 2 GPU sessions, 4 handoffs\]

Only two GPU sessions are needed. Session 1 bundles everything that needs the base model once loaded. Session 2 evaluates the E0.4 variants, and has to wait because they depend on E0.2 and E0.3.

| Step | GPU or CPU | Why | Rough time |
| --- | --- | --- | --- |
| Open-item checks | CPU | reading manifests and tensors | minutes |
| S0 SVD cache | CPU | weights only, no forward pass | minutes |
| E0.1 length interval | CPU | saved per-item results | seconds |
| R0 rollouts | GPU | generation | 15–40 min |
| E0.6a NLL and entropy | GPU, then CPU | teacher-forced forward over about 5,000 answers | 10–20 min |
| E0.5 gradients | GPU, then CPU | forward plus backward, 192 batches | under 30 min |
| E0.2 π | CPU | projections of saved a, b | minutes |
| E0.2 activation share | GPU | hooks on 300 fixed sequences | about 5 min |
| E0.3 | CPU | SVDs of W₀ + b aᵀ | minutes |
| E0.6b KL | GPU if missing, then CPU | forward over 300 sequences × 11 adapter states | about 10 min |
| E0.4 build | CPU | tensor edits | seconds |
| E0.4 evaluate | GPU | about 1,500 greedy generations plus KL | 30–60 min |

**Total GPU:** about 2–3 hours of A100 time across two sessions. At Colab's measured 5.4 units per hour, that is roughly 11–16 units.

**Why not run everything in causal order in one notebook?** Several reasons:

- It keeps the GPU idle during analysis, which is most of the wall-clock time.
- It mixes model-loading code with statistics, so an analysis bug forces a GPU re-run.
- A Colab disconnect loses both.

Saving raw tensors makes every analysis repeatable for free.

**Practical rules:**

- Each GPU cell writes its outputs and a manifest to Drive before the next cell starts, so a disconnect loses at most one step.
- Load the base model once per session and attach and detach adapters with PEFT, instead of reloading.
- Run the tiny-model gradient test and a 2-batch smoke run at the top of session 1, before the real jobs.

## Implementation spec

This section is the contract for whoever writes the code. Each job reads only the files listed as its inputs, writes only its own folder, and passes its acceptance tests before anything downstream starts. Where this section and an earlier section disagree, this section wins.

### Stop points

The implementer stops and reports at each point. Work resumes only after Prasad approves.

1. **After step 0 (open items).** Report every value from the open-items list, with the file it came from. In particular: the baseline pass's settings, the 384 remaining indices, the learning rate, loss averaging, the GRPO reward's extractor version, whether a₀ matches across arms, and which KL files exist.
2. **After R0.** Report the acceptance-test table and the mean reward on old 128 vs new 384.
3. **After GPU session 1.** Report the acceptance tests for E0.6a, E0.5, the E0.2 activation pass and E0.6b. No analysis numbers yet beyond the tests.
4. **Prediction freeze.** Prasad writes P3 and P4 into the prereg after seeing E0.5's summary, before E0.2's π is computed for the trained adapters.
5. **After E0.3.** Report the s ratios, intruder flags and rotation. Prasad picks the E0.4 variant list from the decision table.
6. **After GPU session 2.** Report the acceptance tests, then the E0.4 analysis.

### File layout

```
runs/stage0/
  common/
    stage0_common.py          # shared helpers, imported by both notebooks
    requirements.lock         # pinned versions, identical in both notebooks
  step0_checks/
    report.json               # every value from the open-items list + source file
  r0_rollouts/
    rollouts.parquet          # 4,096 rows: old 128 + new 384 problems
    datasets.json             # problem ids and sample ids for each derived dataset
    manifest.json
  s0_svd/
    svd.safetensors           # per module: sigma (all), U[:, :k70], V[:, :k70]
    k.parquet                 # per module: k30, k50, k70, d_in, d_out, chance
    manifest.json
  e01_length/
    results.json
  e06a_distance/
    tokens.parquet            # one row per answer token
    summary.parquet
    manifest.json
  e05_push/
    pushes.safetensors        # per loss/dataset, module, batch
    batches.json              # problem ids in every batch
    summary.parquet
    manifest.json
  e02_where/
    act_sums.safetensors      # GPU: activation sums per module, arm, token set
    pi.parquet
    manifest.json
  e03_rotation/
    results.parquet
    verdict.json
  e06b_drift/
    kl_positions.parquet
    summary.parquet
    manifest.json
  e04_edits/
    adapters/<variant>/       # PEFT adapter folders, one per variant
    eval_items.parquet
    kl_positions.parquet
    summary.parquet
    manifest.json
  DONE_<job>                  # empty marker written last; readers wait for it
```

**Every manifest records:**

- the paths and sha256 of all inputs;
- the git commit of `stage0_common.py`;
- library versions;
- GPU name, or `cpu`;
- dtype and the TF32 flags;
- seeds;
- start and end time;
- the acceptance-test results.

### Shared module: `stage0_common.py`

Both notebooks import these functions; neither re-implements them.

| Function | Does |
| --- | --- |
| `load_base(dtype=float32)` | base model and tokenizer, TF32 off, eval mode |
| `load_adapter(model, path, trainable=False)` | attaches a PEFT adapter; checks rank 1, α 1, dropout 0, 196 modules |
| `iter_modules(model)` | yields (layer, type, name) in a fixed order: layer 0–27, then q, k, v, o, gate, up, down |
| `get_ab(adapter_state, name)` | returns a = lora\_Aᵀ and b = lora\_B as float32 vectors |
| `format_example(problem, answer)` | the exact chat template and answer formatting used by Pilot 2, and the mask of loss tokens |
| `extract(text, version)` | flexible-v3, flexible-v2 or strict; imported from the Pilot 1 harness, not copied |
| `load_svd(name)` | sigma, U\_k, V\_k, k from the S0 cache |
| `pi(vec, basis)` | ‖basisᵀ v‖² / ‖v‖² |
| `paired_bootstrap(x, y, draws=10000, seed=42)` | percentile 95% interval on the mean paired difference |
| `write_manifest(folder, ...)`, `mark_done(job)`, `wait_done(job)` | provenance and handoff markers |

### Output schemas

Column names are fixed. Module names use the form `L{layer:02d}.{type}`, for example `L07.up`. Float columns are float32 unless marked; tensors are float32.

| File | Format | Contents |
| --- | --- | --- |
| `r0_rollouts/rollouts.parquet` | one row per sample | `problem_id, sample_idx, source` (old/new), `seed, prompt_sha256, text, token_ids, n_tokens, truncated, correct_v3, correct_v2, correct_strict, extracted_v3, gold_answer` |
| `r0_rollouts/datasets.json` | lists | `gold`: problem ids; `own_all`, `own_correct`, `own_correct_one`: (problem\_id, sample\_idx) pairs; `p_correct`: problem id → k/8; `dropped`: problems with no correct sample |
| `s0_svd/svd.safetensors` | tensors | keys `{module}/sigma`, `{module}/U`, `{module}/V` |
| `s0_svd/k.parquet` | one row per module | `module, layer, type, d_in, d_out, k30, k50, k70, energy_top1, chance_out, chance_in, recon_err` |
| `e01_length/results.json` | object | accuracy and length per comparison: `mean_diff, ci_low, ci_high, mcnemar_p, n_gain, n_loss`; for length also `median_diff, share_shorter` |
| `e06a_distance/tokens.parquet` | one row per answer token | `dataset, problem_id, sample_idx, position, rel_position, token_id, surprise, entropy, rank` (float64 for surprise and entropy) |
| `e06a_distance/summary.parquet` | one row per dataset | `dataset, n_answers, n_tokens, nll, entropy, distance, ci_low, ci_high`, and the paired gold − own\_correct\_one difference with its interval |
| `e05_push/pushes.safetensors` | tensors, length d\_out | keys `{loss}/{module}/batch{i:02d}`; loss is `sft_gold`, `sft_own1` or `grpo_own` |
| `e05_push/batches.json` | lists | the problem ids in every batch, per loss |
| `e05_push/summary.parquet` | one row per loss × module | `loss, module, mean_push_norm, consistency, pi_out, chance_out, pi_out_rand_p05, pi_out_rand_p95, cos_trained_b, cos_sign_trained_b, cos_rand_p95` |
| `e02_where/act_sums.safetensors` | scalars per key | `{arm}/{tokenset}/{module}/sum_full` and `/sum_par` (float64), plus `count` |
| `e02_where/pi.parquet` | one row per arm × checkpoint × module | `arm, step, module, pi_out, pi_in, act_share_gsm8k, act_share_fineweb, chance_out, chance_in, rand_p05, rand_p95, b_norm, a_norm` |
| `e03_rotation/results.parquet` | one row per arm × step × module | `arm, step, module, s_ratio, n_intruders_t03 … n_intruders_t07, rot_out_mean_sin2, rot_out_max_deg, rot_in_mean_sin2` |
| `e03_rotation/verdict.json` | object | which row of the decision table applies, with the counts behind it |
| `e06b_drift/kl_positions.parquet` | one row per position | `adapter, step, tokenset, doc_id, position, kl` (float64) |
| `e06b_drift/summary.parquet` | one row per adapter × step × tokenset | `adapter, step, tokenset, kl_mean, ci_low, ci_high, eff_write, ab_norm_sum` |
| `e04_edits/eval_items.parquet` | one row per variant × item | `variant, item_id, text, n_tokens, truncated, correct_v3, correct_strict, extracted_v3, needs_handcheck` |
| `e04_edits/summary.parquet` | one row per variant | `variant, acc, acc_diff_vs_V0a` and its interval, `mcnemar_p, median_len, len_diff` and its interval, `kl_fineweb, kl_diff` and its interval |

### Acceptance tests

Each test blocks the jobs downstream of it. Types: K = known answer, I = invariant, X = independent calculation, N = negative control.

| Job | Test | Type | Pass condition |
| --- | --- | --- | --- |
| common | the extractor is the Pilot 1 harness object; v3 scores the 150 gold solutions as correct | K | 150/150 under v3 (v2 gives 0/150, the known defect) |
| common | `load_adapter` on each checkpoint-0 gives outputs identical to the base model | I | greedy outputs token-identical on 5 items |
| common | tiny 2-layer model: b-gradient at b = 0 equals G a₀ built by hand from hooked δ and x; a-gradient is 0 | X | max relative error below 1e-5; a-gradient exactly 0 |
| R0 | counts | I | 512 problems × 8 samples, no duplicate (problem, sample) |
| R0 | old and new halves come from one procedure | I | mean v3 reward of new 384 within 5 points of the old 128; truncated fraction below 2% |
| R0 | reproducibility | K | re-generating 2 old problems with recorded seeds reproduces their text (HF) or their reward rate within noise (vLLM) |
| S0 | reconstruction | I | ‖W₀ − UΣVᵀ‖/‖W₀‖ below 1e-6 on the full SVD before truncation |
| S0 | orthonormality and hash | I | U\_kᵀU\_k = I to 1e-6; base-weight hash matches the source audit |
| E0.6a | own samples sit at distance 0 | K | own\_all distance within ±0.02 nats per token of 0 |
| E0.6a | surprise matches the loss | X | mean surprise on 8 gold answers equals the SFT loss computed by the training code on the same batch, to 1e-5 |
| E0.5 | starting state | I | b = 0 and the a-gradient is exactly 0 in every module |
| E0.5 | push matches a finite difference | X | on 3 modules, (L(b = εu) − L(0))/ε matches push·u within 1e-3 relative, ε = 1e-4 |
| E0.5 | dead groups | I | the GRPO loss gives exactly zero push on a batch made only of all-correct or all-wrong groups |
| E0.2 | π of a basis vector | K | π(U\_k\[:, 0\]) = 1; π of a vector orthogonal to U\_k = 0 |
| E0.2 | chance check | N | mean π of 100 random vectors within 10% of k/d per module |
| E0.2 | activation sums | I | a test vector lying inside V\_k gives sum\_par = sum\_full to 1e-6; checkpoint-0 a gives the same numbers in both arms if the a₀ match |
| E0.3 | zero update | K | an all-zero adapter gives 0 intruders and 0 rotation |
| E0.6b | KL basics | I | checkpoint 0 gives KL = 0 exactly; no negative values |
| E0.6b | agrees with saved numbers | X | where Pilot 4 saved KL, the new value matches within 1e-4 relative |
| E0.4 | edit identity | I | k = 0 edit reproduces the original tensors byte for byte; ‖U\_kᵀ b‖ = 0 after V1 |
| E0.4 | reference reproduces | K | V0a's per-item v3 results match the saved SFT-64 results; any mismatch is listed, not hidden |

### Details fixed here

These were open in the job sections.

- **own\_correct\_one:** for each problem with at least one v3-correct sample, take the correct sample with the lowest sample index after shuffling with `random.Random(42 + problem_id)`. This makes it deterministic and unbiased.
- **Matched problem set for E0.5:** use only problems that have an own\_correct\_one sample, for both the gold and own datasets. Batches are 8 consecutive problems from one seed-42 shuffle of that set, the same order for both datasets.
- **GRPO push:** advantages = (r − group mean)/(group std + 1e-4), with r the extractor version GRPO trained with (from step 0). Groups with zero std get advantage 0. Per-token averaging follows Pilot 4's loss type.
- **If the a₀ differ across arms:** compute E0.5 pushes twice, once with each arm's a₀, and compare each trained b with the push made from its own a₀. Report the cross-arm a₀ cosine.
- **V3 random directions:** for each module, draw a d\_out × k Gaussian matrix with `torch.Generator().manual_seed(1000 + draw)`, orthonormalize it by QR, and delete that subspace from b. Draws 0, 1, 2.
- **Hand-check in E0.4:** flag an item when v3 and strict disagree, or v3 finds no number. The flagged items are checked by Prasad, blind to variant, and corrections are recorded in `handcheck.csv` with the reason.
- **Generation for E0.4:** greedy, the Pilot 1 harness's max new tokens and stop rule, batch size fixed for all variants. Each variant runs right after reloading the base model state, to avoid adapter leakage.
- **Numerical settings:** float32 throughout, float64 accumulation for KL and the activation sums. `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### Handoff package

Give the implementer:

- this doc;
- the [Pilot 4 design doc](https://claude.ai/artifact/B2BkGcGwkYWtUJ1t4NTrwo);
- read access to the `runs/` folders listed in "What we have";
- the Pilot 1 harness code.

The first deliverable is `step0_checks/report.json`. Nothing runs on a GPU before stop point 1 is approved.

## Predictions to write down before running

Freeze these in the prereg ledger before the GPU notebook starts. E0.5's π prediction for E0.2 is the one that must be timestamped before E0.2's π is opened.

| # | Prediction | Measured by | Counts against it |
| --- | --- | --- | --- |
| P1 | `own_all` distance ≈ 0; `own_correct` ≈ mean −ln P(correct); `gold` much larger | E0.6a | own\_all clearly above 0: a pipeline bug, fix before anything else |
| P2 | gold push larger and more consistent than own-correct push; GRPO push smallest | E0.5 size, consistency | no difference: closeness doesn't shrink the push here (C2 fails) |
| P3 | π\_out(gold push) > π\_out(own-correct push) > chance | E0.5 direction | equal: data doesn't decide direction at step 0 |
| P4 | π\_out(b\_SFT) > π\_out(b\_GRPO) in most modules; state which types | E0.2 | reversed or both at chance (C3 fails at rank 1) |
| P5 | π\_in near chance for both arms; activation share similar for trained and init a | E0.2 | a carries learned structure despite its small rotation |
| P6 | s small everywhere; no intruders; rotation under a degree | E0.3 | intruders in SFT: switch E0.4's focus to intruder scaling |
| P7 | SFT's KL curve lies above GRPO's at matched update size | E0.6b | curves overlap: size explains the difference |
| P8 | V1 restores part of the length and lowers KL more than V3 and V4, with accuracy inside ±5 points | E0.4 | V3/V4 match V1 (size), or accuracy falls (entangled) |
| P9 | SFT median length drop interval excludes 0; GRPO's includes 0 | E0.1 | as stated |

## Open items to check in the files

Check all of these before the GPU notebook runs. Each one changes a step above.

- [ ] **Baseline pass:** its folder, problem indices, engine (HF or vLLM), dtype, max new tokens and seeds. All 128 problems are inside the 512 training indices. Sets every R0 setting.
- [ ] **a₀ match:** load `lora_A` from both checkpoint-0s; cosine should be 1.0 in every module if both used seed 42 the same way. Decides whether E0.5 compares arms from a shared start.
- [ ] **KL to base:** does it exist for both arms at all five checkpoints on the Pilot 3 FineWeb set? Look in `pilot4-grpo-fp32-v1-supplemental-tokens-a10040-v2` and the write-report folders. Decides how much of E0.6b runs on GPU.
- [ ] **Training hyperparameters from the manifests:** learning rate, warmup, max\_grad\_norm, SFT loss averaging, max completion length. Needed for E0.5's loss to match training.
- [ ] **SFT data formatting:** exactly how gold solutions were templated, and whether the end-of-turn token was in the loss. E0.5 and E0.6a must score the same tokens SFT trained on.
- [ ] **GRPO advantage details:** group std with or without a small epsilon; how truncated answers were scored. Needed for E0.5's GRPO push.
- [ ] **Random control adapter:** how it was built and what norm it matches.
- [ ] **Team 2 doc:** it was updated on 25 Oct. Check whether B0's precision, rank or module plan changed in a way that affects Stage 0.

**Do we need anything else?** Three small additions:

- **A tiny-model test file** that checks, on a 2-layer model, that the b-gradient at b = 0 equals G a₀ computed by hand, and that the a-gradient is zero.
- **A prereg entry** with P1–P9, dated before the GPU run.
- **One shared `stage0_common.py`** for loading adapters, the SVD cache, the extractor, and manifest writing, imported by both notebooks so they can't drift apart.
