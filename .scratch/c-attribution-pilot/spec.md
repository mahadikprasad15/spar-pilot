# Correctness (J) attribution pilot — online and offline GRPO

**Status:** needs-triage. Draft v2 (2026-11-18), revised after [review round 1](review-round-1.md); the response is in [review-response-round-1.md](review-response-round-1.md). Design only: no code, GPU run, threshold or budget is approved by this file. Values marked **[user]** are decisions recorded in review round 1. **[proposed]** marks a value not yet approved. Open decisions are listed at the end.

This pilot builds on `.scratch/adapter-neuron-attribution/` (instrument, gates, units, chain). Where the two differ, this file governs this pilot only. The production-1 study and its trace and stop metrics are unchanged.

## Problem

Production-1 ranked adapters by first-order effects on two teacher-forced proxies, both measured on the untuned model's own correct traces:

- **Trace log-likelihood.** Removing the top-10 adapters raised it in every arm, so it mostly measures distance from base-model text.
- **Stop pull.** It did not predict generated length.

Neither proxy targets correctness.

This pilot ranks the 196 rank-1 adapters, and the `down_proj` neuron-read terms of online and offline GRPO, by their local first-order effect on the **sampled probability of a correct answer**, J. It then removes the selected components and measures **greedy** accuracy on the same 150 questions, as an exploratory behavioural check.

## Questions

1. **Adapter ranking.** For each arm, what is dJ/dλ_i at λ = 1 for every adapter gate, and how uncertain is it?
2. **Neuron-read ranking.** For each `down_proj` adapter, how does that gradient split across its 8,960 read terms? Which reads rank highest within three scopes: global, hot layers and every layer?
3. **Stability (descriptive).** How similar are the rankings across two seeded halves of 75 questions each?
4. **Behaviour (exploratory, greedy).** On the same 150 questions, how does greedy accuracy change when each selected group is removed? Each group is compared with its intact model and with a matched control.
5. **Comparison.** How do online and offline GRPO differ in signed scores, top-set overlap and layer or module distribution? The answer is descriptive, from one seed per arm.

### Scope

| In this launch | Not in this launch |
|---|---|
| Greedy ablations on the full cohort150 | Sampled-ablation runs (deferred) |
| Top-10 adapter removal with a matched control | Top-10 neuron-read ablations **[user]** |
| Three top-64 neuron-read groups, each with a matched control | Combined adapter and neuron removals |
| | Removing harmful components |
| | The four SFT arms |
| | Held-out generalization (prepared as ticket 14, not launched) |
| | Changing production-1 metrics or results |

### How to read the results

- **Targets.** Attribution targets sampled J. The behavioural check measures greedy accuracy. A greedy result neither confirms nor refutes the sampled-J gradient. A direct test of J would need sampled intervention runs under the same policy, which is deferred.
- **Removal is not local.** Removing a whole adapter or a group of reads is a finite intervention, and its parts can interact. The sum of local predicted effects is only a first-order reference, reported beside the measured outcome under a distinct label.
- **Helpful and harmful.** These words are local and estimated. A positive dJ/dλ predicts that slightly strengthening the component near λ = 1 raises J. A negative value does not prove that full removal would raise accuracy. A component whose selected score is not positive is never labelled helpful.
- **Same-cohort results.** The 150 questions were used for selection and for earlier evaluations. Every behavioural result is labelled exploratory and is never described as held-out.
- **One seed per arm.** The two arms are scored on their own sampled responses, so their difficulty weighting differs. One seed per arm cannot establish an objective-specific or general mechanism.

## Data and provenance

Catalog ids come from `catalog/`. This branch now contains the catalog, merged from local `main` in `a43dca0` with the catalog commits `6444345`, `f3ebb9d`, `271acf3` and `6f6e2f0`. `python -m pilot_eval.catalog verify --root local=<artifacts>` reports 51 pass and 4 not-local.

**[user]** Attribution reuses the existing final-checkpoint sampled responses: 150 questions × 8 draws per arm, every draw and every outcome type kept. Reuse is conditional on the provenance checks below.

| Arm | Catalog id | sha256 | Adapter (per row) | Groups: all-correct / all-wrong / mixed | Mean reward | Capped | Invalid |
|---|---|---|---|---|---|---|---|
| online-grpo | `online-grpo.s64.sampled-x8.test150` | `4af7926a…` | `70d2d56c…` | 49 / 12 / 89 | 0.675 | 0 | 46 |
| offline-grpo | `offline-grpo.s64.sampled-x8.test150` | `a1cd79aa…` | `66015518…` | 57 / 13 / 80 | 0.693 | 1 | 39 |

### What the source run records say (checked 2026-11-18 against the local imports)

| Item | Status | Evidence |
|---|---|---|
| Base and tokenizer revision | Recorded | `989aa798…` in both configs |
| Prompt and chat template | Recorded | chat template sha256 `2e2d2512…`; dataset revision `740312ad…` |
| Checkpoint | Recorded | step-64 adapter and seal hashes; the per-row `adapter_sha256` matches |
| Runtime | Recorded | FP32 base and adapter; A100-SXM4-80GB; torch 2.9.0, transformers 4.57.6, PEFT 0.18.0; `use_cache` |
| Sampled policy | Recorded | `do_sample`, temperature 1.0, top_k 0, top_p 1.0, `num_beams` 1, cap 1,024, EOS 151645, 8 draws, `batch_size` 4 |
| Seeds | Recorded | 42 + ordered batch index |
| Generation engine | Recorded | `implementation.engine` = `e33ea870…`, byte-identical to `pilot_eval/grpo_eval_engine.py` on this branch |
| Repetition penalty and `no_repeat_ngram_size` | Inferred | Not in the configs. The engine calls `grpo_gpu._generate`, which passes `repetition_penalty=1.0` and `no_repeat_ngram_size=0` explicitly. The run configs do not hash `grpo_gpu.py`. Git shows its last change on 2026-10-09 (`3c859a6`), before both behaviour runs. |
| Resolved logits processors | Not verified | To be checked as described below |
| Per-token generation log-probs | Not saved | — |

Ticket 05 settles the two unverified rows on CPU. It loads the pinned `generation_config.json` (revision `989aa798…`), builds the processor and warper list that `generate` resolves from `_generate`'s explicit arguments, and checks that nothing modifies the logits. That means no repetition penalty, temperature, top-k or top-p processor; if any appears, its effect must be shown to be the identity.

The bundle records the inferred items as `inferred` with this evidence; they are not marked verified.

**If a policy or checkpoint mismatch is confirmed, it is material.** It is repaired, other suitable samples are used, or the report states that the on-policy interpretation is not established. It is never silently accepted.

### Rewards, tokens, cohort and baselines

- **Rewards.** Re-scored on CPU with `score_gsm8k_flexible_v3(text, gold, capped=capped)`; the scorer hashes to `c958d2de…`. The stored values agree with the re-score on 1,350 of 1,350 rows per arm. Capped and invalid rows score 0.
- **Tokens.** Each row stores `generated_token_ids`. An EOS row ends with the sampled EOS; a capped row has all 1,024 tokens. Prompts are the stored templated strings, tokenized with `add_special_tokens=False` (mean 115 tokens, max 216). Teacher-forced input is the prompt IDs followed by `generated_token_ids`: 422k tokens online and 405k offline, with the longest sequence 1,151 tokens.
- **Cohort.** `gsm8k-test-cohort150` (ADR 0001); its prior uses are in `catalog/cohorts.yaml`. Selection and the behavioural test use the same questions, so the behavioural results are exploratory.
- **Baselines.** Stored greedy runs: untuned 0.68, online step 64 0.76, offline step 64 0.7267. These are consistency references only. The ablation stage regenerates the intact and untuned greedy runs in its own qualified runtime and batching, and compares conditions within that run **[proposed: regenerate untuned, 150 answers]**.

## Estimand and estimator

For arm θ with adapter gates λ (λ_i = 1 is intact; the branch computes λ_i · s · B_i(A_i x)):

    J(λ) = (1/Q) Σ_q E_{y ~ π_λ(·|q)} [R(q, y)],   R ∈ {0, 1} = flexible-v3 correct (capped → 0)

### Per-response metric `seqlogp`

M(q, y) = Σ_{t ∈ G} log p_θ(y_t | q, y_<t), a **sum** over G, the generated positions:

- every sampled content token;
- plus the sampled EOS when the row stopped at EOS;
- or, for a capped row, all 1,024 tokens with no EOS term.

The sum is accumulated in FP64.

M uses the raw log-softmax of the FP32 logits. That equals the sampling distribution only under the recorded policy (T = 1, no truncation, no repetition penalty) and if no logits processor is active; the processor check above verifies the second condition.

### Positions

- **Scored positions:** G only.
- **Attributed positions:** every unpadded position, prompt prefill included. The existing contraction attr_i(e) = −Σ_t c_t ∂M/∂c_t already sums over them, and a gate changes prompt computations that affect later tokens.

### Per-response attribution

    attr_i(q, k) = −Σ_t c_t ∂M_qk/∂c_t = −∂M_qk/∂λ_i at λ = 1
    attr_j(q, k) = −a_j Σ_t z_{j,t} ∂M_qk/∂c_t          (neuron read j of a down_proj adapter)

with Σ_j attr_j = attr_i for each response.

### Leave-one-out weights (detached)

For question q with n_q = 8 draws:

    b_qk = (Σ_k' R_qk' − R_qk) / (n_q − 1),   w_qk = (R_qk − b_qk) / (Q · n_q)

- Questions are weighted equally, and draws are weighted equally within a question.
- All-correct and all-wrong groups stay in Q and contribute exactly 0.
- n_q = 1 is refused.
- Rewards and weights never enter the graph.

### Aggregation

No loss is formed and no optimizer step is taken.

    removal_J_c = Σ_qk w_qk · attr_c(q, k)     # first-order predicted ΔJ when component c goes 1 → 0
    dJdλ_c      = −removal_J_c                 # local gradient for strengthening c at λ = 1

Here c is an adapter or a neuron read.

**Per-question pre-aggregation.** Define S_q,c = Σ_k ((R_qk − b_qk)/n_q) · attr_c(q, k). Then removal_J_c = (1/Q) Σ_q S_q,c. Resampling whole questions keeps every group intact, so a question bootstrap on S_q is exactly the same as recomputing the leave-one-out weights inside each resample. This makes neuron-level uncertainty feasible without per-response neuron arrays (see Storage).

**Unbiasedness.** Under on-policy sampling, Σ w ∇M is an unbiased estimate of ∇J: this is the score-function identity with a leave-one-out baseline. The estimate is local and says nothing exact about full removal.

### Sign convention

This table must appear in code docstrings, outputs and report headers.

| Quantity | Meaning | Positive means |
|---|---|---|
| `attr_c(e)` (existing) | first-order Δ metric when c is removed (1 → 0) | removing raises the metric |
| `removal_J_c` | first-order ΔJ when c is removed | removal is predicted to **raise** J locally |
| `dJdλ_c` | ∂J/∂λ_c at λ = 1 | slightly strengthening c is predicted to raise J (**helpful**, locally) |

### Group prediction

For a removed group S, the "first-order sampled-J predicted removal effect" is Σ_{c∈S} removal_J_c, in J units, also shown as percentage points. It is a local sampled-J approximation. It is not a numerical prediction of greedy accuracy, and not a guarantee about full removal.

### Denominators

- **Concentration** is reported as a share of the **gross** total, Σ_c |dJdλ_c|.
- The **net** total, |Σ_c dJdλ_c|, is reported separately. It can approach 0 through cancellation and is never used as a denominator for shares.
- Both are stated wherever a share appears.

## Uncertainty and stability (descriptive; never a gate) [user]

### Question split

- The 150 question IDs are split into two groups of 75 with a recorded seed **[proposed: 20261118]**.
- All 8 draws of a question stay together, and the same split is used for both arms.
- The split's ID lists are sealed in the freeze inputs.

Per arm, from each half's S_q, report:

| Ranking | Summaries |
|---|---|
| Adapters (196) | Spearman; top-10 overlap; sign agreement among each half's top 20 |
| Global top-64 neuron reads | overlap count; sign agreement; layer distribution of each half's top 64 |
| Per hot layer and per down_proj layer: top-64 reads | overlap count and sign agreement, per layer and as an equal-layer mean |

Stability is descriptive. No threshold blocks or alters the pilot. After the split analysis, selection uses all 150 questions.

### Draw split [user, 2026-11-19]

Kept in addition to the question split. It asks a different question: for the same 150 questions, does the ranking depend on which samples the model happened to draw?

- Half A is draws 0–3 and half B is draws 4–7 of every question.
- Leave-one-out weights are recomputed **within each half** (n = 4, baseline from the other 3 draws of that half). A half whose 4 rewards are all equal contributes zero.
- Adapters: computed on CPU from the stored per-response attributions.
- Reads: the attribution pass also accumulates the two half-sums Σ_q S_q^A and Σ_q S_q^B, shape [2, 28, 8,960] per arm (small). Per-question half arrays are not stored, so there is no bootstrap on the draw split.
- The same summaries as the question split are reported, labelled "noisier: 4 draws per question per half".

### Bootstrap

**Adapters.** 10,000 resamples of whole question groups, seed 1234 **[proposed]**.

- Percentile intervals for dJdλ and rank intervals.
- Resampled question indices are shared by both arms (paired by question ID), so between-arm differences get paired intervals. Each arm keeps its own response groups.

**Neurons.**

1. Benchmark a chunked bootstrap first: a count matrix [B, 150] times S [150, columns], chunked over columns, with peak memory and wall time measured.
2. Priority order:
   - intervals for every selected read and its control;
   - the question-split summaries;
   - per-layer sums;
   - all-read intervals, only if the benchmark shows they are feasible within the cap.

All neuron and adapter intervals for candidates selected on this cohort are exploratory, pointwise intervals. They are not simultaneous or post-selection-confirmatory.

**Also reported:** J̄; informative (mixed) questions, 89 online and 80 offline; the top-10 adapter share of gross Σ|dJdλ|.

## Numerical precision: what is verified, measured, or open

| Item | Status | Source |
|---|---|---|
| FP32 attribution with an FP64 finite-difference reference (`full_fp64`, since `model.double()` leaves Qwen2's softmax, RMSNorm and RoPE in FP32) | Reproduced: agreement within 5e-5 on the CPU fixture and on Colab | lessons, 06a diagnostic 4; production-1 spec (user, 2026-11-16) |
| FP32 finite-difference floor ≈ ulp(\|M\|)/ε | Rule, measured in 06a r3: 60× disagreement on effects of about 1e-5 | lessons |
| Production-1 qualification | Passed: finite differences on gold-sft and online-grpo; batch invariance on all six arms; measured batch-invariance discrepancy 2.2e-6 (`trace` mean metric); resolution floor = 10× that | `attribution-production-1/.../qualification.json` |
| Production-1 tolerances | Finite differences: median relative error ≤ 0.05, max ≤ 0.20 (FP64 reference). Identity: \|Σ_j attr_j − attr_i\| ≤ 1e-4 · Σ_j \|attr_j\| (gross denominator). Batch invariance: ≤ 1e-4 × max\|attr\| | production-1 spec |
| Scale of `seqlogp` | **New risk.** \|M\| can reach hundreds of nats, against a mean of order 1 for `trace`. The FP32 finite-difference floor at ε = 1e-2 rises to about 1e-3 or more, so FP64 references are mandatory, not optional. | derived; to be measured |
| Signed cancellation in aggregates | **New risk.** Per-question S_q mixes signs; dJdλ_c can be much smaller than its gross contributions. | to be measured: report the gross/net ratio per component |
| Cached (generation) against teacher-forced (instrument) log p | **Unmeasured on real arms.** A per-token bound of 1e-3 would permit up to 1.024 nats per sequence over 1,024 tokens. That is a bound, not an observation. | rehearsal (ticket 11) |
| Storage precision | Per-question S stored as FP64; per-response adapter attributions as FP64; FP32 only for activation-derived term sizes | this spec |

**Process.** CPU tests and the rehearsal measure the floors and errors. The user then fixes the production tolerances in the launch record, before production. Production-1's finite-difference, identity and batch-invariance tolerances carry over unchanged. They are not loosened to pass: a failure stops the chain and is reported.

The user's no-gate decision covers statistical stability only. Input integrity and numerical qualification remain hard stops.

## Checks

### CPU, before any GPU run

Tiny random Qwen2 with PEFT 0.18.0, no downloads.

1. **`seqlogp` targets.** The mask includes the final EOS target. A capped row has 1,024 targets and no EOS term. An EOS-only response has one target. The value is a sum; the mean-not-sum mutation fails.
2. **Row builder.** It reproduces the counts from real-format fixtures and keeps every row. It refuses the following:
   - an adapter hash mismatch;
   - a duplicate `draw_id`;
   - an early EOS;
   - a malformed capped row;
   - a reward that disagrees with the scorer;
   - a scorer-hash mismatch.
3. **Leave-one-out weights.** Hand-computed fixtures. All-same groups give exactly 0. n_q = 1 is refused. Σ_k w_qk = 0 per question. Per-question pre-aggregation equals direct aggregation.
4. **Signs.** A synthetic helpful adapter gives dJdλ > 0 and removal_J < 0. On a 2-token-vocabulary exact enumeration, the exact ∇J equals the estimator's expectation.
5. **Surrogate finite differences.** F(λ) = Σ w_qk M_qk(λ), with the weights frozen, matches dJdλ for 5 adapters and 3 neurons, against an FP64 reference. This checks the implementation, not the estimator.
6. **Neuron identity.** Holds per response and on aggregates, with the gross denominator.
7. **Old metrics unchanged.** Existing instrument and metric tests pass unmodified. `trace` and `stop` outputs are bit-identical with and without `seqlogp`.
8. **Cached against teacher-forced log-probs.** Left-padded prefill with incremental KV-cache decoding of the saved tokens, against right-padded full forcing. Both per-token and per-sequence errors are reported.
9. **Logits processors.** With the pinned `generation_config.json`, `_generate`'s explicit sampled arguments resolve to processors that leave the logits unchanged.
10. **Interventions.** Zeroing lora_B, and zeroing a_j in lora_A, equal the `removal` hook and `neuron_gate(g = 0)` on logits, at prefill and at decode with a cache. Multi-read removal across layers is exact. Restore is bit-exact (weight hash).
11. **Matching.** The review's counterexample is a regression fixture: targets 0 and 1 with writes 1.00 and 1.40; candidates 2 and 3 with writes 1.16 and 0.85; tolerance 0.20. The new matcher finds 0→3 and 1→2. Infeasible groups are reported as unavailable, with the deficient targets named. Results are deterministic under a seed.

### GPU qualification

Real arms, a few rows each. The values are recorded, then compared with the launch record's fixed tolerances (automatic sign-off as in production-1).

- **(a) Cached against teacher-forced**, on 16 rows per arm chosen to include short and long rows and, where that arm has one, a capped row. Online GRPO has no capped rows, so record that absence rather than failing or fabricating one; include the available offline capped row, and cover capped sequences in the CPU fixture. Report max and median |Δ log p| per token and |Δ M| per row. This reconstructs cached decoding on the current runtime: the original generation log-probabilities were not saved, so it does not directly validate those original scores. Also report representative per-adapter attribution differences when attributions use cached or forced log-probs, if feasible.
- **(b) Replay** of one sealed sampled batch with the same seed, batch membership and padding. **Report-only.** A divergence is recorded with its first position; it becomes material only if it points to a policy or checkpoint mismatch.
- **(c) Surrogate finite differences** with the FP64 reference: 5 adapters and 3 neurons, at production-1's tolerances.
- **(d) Batch invariance** at the qualified token budget, at production-1's tolerance, reported together with the absolute floor.
- **(e) Identity** at production-1's rule (gross denominator).
- **(f) No-op.** The intervention machinery with an empty removal set gives logits bit-identical to intact, and greedy tokens identical on a qualification subset. After every condition the restored adapter hash equals the original.

## Selection

Selection is frozen and hashed before any generation unit exists.

### Measures

Both are computed from each arm's own 1,200 attribution rows, over every unpadded position, token-weighted:

- **adapter write** = ‖s·b‖ · mean_t |c_t|;
- **neuron-read size** = |a_j| · mean_t |z_{j,t}|.

They are accumulated in the attribution pass, so no separate run is needed.

### Targets per arm

| Group | Rule | Size |
|---|---|---|
| `top10-adapters` | the 10 largest positive dJdλ among the 196 adapters | 10 |
| `global64-reads` | the 64 largest positive dJdλ_j across all 28 × 8,960 `down_proj` reads | 64 |
| `hot64-reads` | the 64 largest positive dJdλ_j within each hot layer, removed together | 64 × \|H\| |
| `layer64-reads` | the 64 largest positive dJdλ_j within each of the 28 `down_proj` layers, removed together | up to 1,792 |

**Ties** are broken by layer, then module order or neuron index.

**Selection is positive only.** If a scope has fewer positive scores than the requested count, the shortfall is surfaced and the smaller positive group is frozen as it is, with its control matched to the actual size and the shortfall recorded in the freeze and the report [user, 2026-11-19]. It is never padded with non-positive scores, and the condition is not dropped.

**Read removals.** These remove read terms from the LoRA `down_proj` branch, by zeroing a_j in lora_A. They do not delete pretrained neurons, and they are separate from adapter removals.

**Hot layers H.** H is the same list for both arms. It must come from a recorded prior effective-write study, with its layer IDs, source run, measurement, context and file hash bound in the freeze inputs **before** the new J rankings are examined. H is never selected from the new J results.

**H = {11, 15, 16, 24}** [user, confirmed 2026-11-19]. The source is the step-64 effective-write figure on GSM8K solutions: the write ratio ‖ΔW·x‖ ÷ ‖W·x‖ per module, with the top 10 ranked per arm. Its values equal `artifacts/analysis/results-hub-v1/data/writes_modules.json` (sha256 `35747fda…`), context `gsm8k_solution`.

Rule: H is the set of layers whose `down_proj` is among that arm's top-10 modules, unioned over the two GRPO arms.

| Arm | Layers with `down_proj` in the top 10 (rank) |
|---|---|
| online | 15 (3), 16 (4), 11 (9) |
| offline | 16 (4), 15 (5), 24 (8) |

This gives 4 layers and 256 reads.

**Computed, not transcribed.** Selection derives H with `hot_layers(source, sha256, context="gsm8k_solution", step=64, k=10, module="down", arms=("online", "offline"), combine="union")`. It refuses a file whose hash differs, and writes H with every parameter into the launch record. The spec's list is the expected output, checked by a test.

**Why the union.** The same scope in both arms keeps the online-against-offline comparison of `hot64` free of a difference in layers, and the union contains each arm's own hot set. The cost is that each arm's group includes one layer that was not hot for that arm (24 for online, 11 for offline).

**Sensitivity to k (same file, recorded for the report):**

| k | online | offline | union | intersection |
|---|---|---|---|---|
| 5 | 15, 16 | 15, 16 | 15, 16 | 15, 16 |
| 10 (used) | 11, 15, 16 | 15, 16, 24 | 11, 15, 16, 24 | 15, 16 |
| 15 | 11, 15, 16 | 12, 15, 16, 24 | 11, 12, 15, 16, 24 | 15, 16 |
| 20 | 10, 11, 15, 16, 24 | 12–16, 24 | 10–16, 24 | 15, 16, 24 |

Layers 15 and 16 are hot at every k; 11 and 24 depend on the cutoff. The report therefore gives attribution summaries for `hot64` per layer, so the per-arm and intersection views can be read on CPU. The ablation removes all 256 reads together, so its effect is not split by layer.

### Matched controls [user]

| Targets | Control eligibility |
|---|---|
| Adapters | same module type; write within ±20% of the target's **[proposed]** |
| Reads | same layer; read size within ±20% of the target's **[proposed]** |

- **Excluded from candidacy:** every target of the arm in any group of the same kind (adapters or reads).
- **No rescaling** of targets or controls.

**Matching algorithm.** It is group-wise, not sequential:

1. Within each stratum (module type, or layer), build the bipartite graph of targets and eligible candidates.
2. Permute the candidate order with the recorded seed.
3. Run a deterministic maximum-cardinality matching (augmenting paths in target-rank order over the permuted candidates).

A group's control is "unavailable" only if the maximum matching is smaller than the group. Then the report names the deficient targets, the eligible-candidate count per target, and the largest feasible matching. Eligibility is never loosened. Matching is checked layer by layer, so an infeasible layer in `layer64-reads` never blocks the global or hot comparisons.

**Infeasible `layer64` control [user, 2026-11-19].** If any layer cannot supply a full matched control, the `layer64-reads` target is still run unchanged, and `match-layer64` is recorded as unavailable with the deficient layers and targets named. The target-minus-control contrast for `layer64` is then not reported; its paired Δ against intact is.

**Reported per control:**

- the achieved match ratios (min, median, max of control size over target size);
- the layer and module distribution;
- the control's own Σ removal_J.

**Also frozen and reported, not ablated:** the most harmful adapters (most negative dJdλ).

## Interventions

| Removal | Method |
|---|---|
| Adapter | lora_B of the chosen adapters set to 0, in memory, with save, restore and a restored-hash check (equal to the `removal` gate on logits) |
| Neuron read | a_j set to 0 in the chosen `down_proj` adapters' lora_A, i.e. c′ = c − a_j z_j at prefill and at every decode step. This is a weight edit, so it adds no runtime cost. |

Each condition applies exactly one group. Adapter and read removals are never combined.

## Generation-ablation stage (greedy only; same 150 questions; exploratory)

### Conditions per arm: 9

1. intact
2. `top10-adapters`
3. `match10-adapters`
4. `global64-reads`
5. `match-global64`
6. `hot64-reads`
7. `match-hot64`
8. `layer64-reads`
9. `match-layer64`

### Volume

| Item | Answers |
|---|---|
| Arms: 2 × 9 × 150 | 2,700 |
| Untuned, regenerated **[proposed]** | +150 |
| Second intact pass per arm, measuring greedy nondeterminism **[user, 2026-11-19]** | +300 |
| **Total** | **3,150** |

Skipped conditions (an unavailable control) reduce the count.

### Settings

- FP32, cap 1,024, EOS 151645, repetition penalty 1.0, explicit arguments over `generation_config`.
- One fixed question order and one fixed batch membership, identical across every condition and arm, at the qualified greedy batch size.
- Every unit saves responses JSONL and a config file (AGENTS.md).

### Report per condition

Bootstrap intervals use 10,000 resamples of questions, paired by question ID across conditions and arms.

- raw greedy accuracy;
- **paired** Δpp versus the same arm's intact run, with its interval;
- 100 × ablated/intact accuracy, undefined if intact is 0;
- target minus control (per-question paired difference), with its interval;
- gains and losses, and the exact McNemar p (descriptive);
- invalid and capped counts, and mean and median response tokens;
- the group's first-order sampled-J predicted removal effect, and the control's, side by side under their own label.

The untuned row is reported alongside. The intact repeat's flip count (questions whose greedy answer or correctness differs between the two identical intact passes) is reported per arm as the nondeterminism floor, next to every condition's gains and losses.

### Labels

Every table carries these labels:

- "exploratory: components were selected on these questions";
- "greedy accuracy is a different target from the sampled J used for attribution";
- "predicted effects are local first-order sampled-J approximations".

## Comparative reporting

- Online against offline signed adapter scores: scatter, and paired-bootstrap intervals on differences.
- Top-10 adapter overlap; layer × module distribution of the top sets.
- Global top-64 overlap; per-layer top-64 overlaps; the layer distribution of each arm's global top 64.
- Stability summaries for both arms side by side.
- The caveat: one seed per arm, own sampled responses, different difficulty weighting; no mechanism claim.

## Storage

Attribution units hold whole questions, so all 8 draws of a question are in the same unit. Inside a unit, micro-batches respect the qualified token budget.

| Contents | Size per arm |
|---|---|
| Per-response adapter attributions [n, 196], FP64 | small |
| Per-question S_q for adapters [q, 196], FP64 | small |
| Per-question S_q for reads [q, 28, 8,960], FP64 | about 300 MB |
| Draw-half read sums [2, 28, 8,960], FP64 | about 4 MB |
| Per-unit Σ\|c_t\|, Σ\|z_j\|, token counts | small |
| Per-response identity errors, gross and net | small |

Per-response neuron arrays are not stored: 1,200 × 250,880 would be about 1.2 GB in FP32. The identity is checked inside each unit instead.

## Outputs and operations

```text
artifacts/runs/c-attribution/<run-id>/
  config.json  launch.json  status.json  logs/
  inputs/                   sealed bundle manifest (ticket 08)
  rows/<arm>/rows.jsonl + manifest (source sha256, selects, counts, provenance status)
  attribution/units/        BlockStore per (arm, question block)
  aggregate/<arm>/          adapters.json, neurons.npz, split.json, bootstrap/, identity.json, numerics.json
  qualification/            checks (a)–(f), measured floors
  freeze/<arm>.json         hashed selections, hot-layer provenance, controls, split IDs, shortfalls
  generation/<arm>/<condition>/greedy/units + responses.jsonl + config.json
  report/                   report.md, report.json, figures
```

- Units are sealed and resumable, and duplicates are refused.
- Export and verified copy follow the production-1 chain. The Pod is stopped at the end.
- Every generation run, the freeze, and the new prior uses of cohort150 (including the exact rehearsal question IDs) are registered in `catalog/`.

## GPU workload (provisional; to be measured in the rehearsal)

| Stage | Size | Basis | Provisional |
|---|---|---|---|
| Qualification (a)–(f) | 16 rows per arm + FP64 copy | production-1 qualification: 12 min | 20–30 min |
| Attribution | 2,400 rows, 827k tokens, with per-question read accumulation | production-1: 975 tokens/s for two metrics | 15–25 min |
| Aggregate, split, bootstrap, freeze | CPU | not measured | benchmark in rehearsal |
| Greedy ablation | up to 3,150 answers | Stage D: 1,350 in 45 min (≈ 30 per minute) | 90–110 min |
| Export and copy | — | production-1 | 10–15 min |
| **Total** | | | **≈ 2.5–3.5 h, unconfirmed** |

The rehearsal still measures these, for planning (there are no time caps, per the user's decision):

- attribution throughput;
- bootstrap wall time and peak CPU and GPU memory;
- greedy throughput at the fixed batch;
- export and readback time.

The v1 estimate (2–2.5 h, 3 h cap) is withdrawn. There is no automatic handover.

## Run tier: Pod only, no Colab stage

The route is **CPU tickets 01–10 → one attended Pod rehearsal (11) → a fresh production launch (12)**. Colab is a fallback only, if no Pod is available. The reasons, and the Pod lessons built into tickets 08–11, are unchanged from v1:

- 120 GB disk;
- `HF_HOME` off the volume;
- hashed downloads;
- a sealed bundle verified against its seal;
- typed launch-record leaves;
- a new run ID and subtree per launch;
- a stop command that tries both syntaxes;
- watchdog ownership only after lock and precheck;
- empty-disk resume;
- budget at unit boundaries.

## Launch plan

1. **CPU tickets 01–10**, red/green. Report the full suite at session start.
2. **Attended Pod rehearsal (11)**, with approval. It measures floors, throughput and bootstrap cost, records the exact rehearsal question IDs, and exercises interruption and resume, export and a real Pod stop.
3. **Production launch record**, with approval. It contains a new run ID; tolerances set from the measured floors, with historical ones unchanged; seeds; the hot layers and their provenance; and the resolved open decisions.
4. **Production (12).** The order is qualification → rows → attribution → aggregate, split and freeze → greedy ablation → report → verified copy and export → Pod stop. Analysis outcomes never stop the chain. Failed qualification or input integrity does.
5. **Report and catalog (13)**, then a separate decision on the held-out follow-up (14).

A short preregistration before step 4 is recommended.

## Review process

- Spec rounds: `review-round-N.md` and `review-response-round-N.md`.
- One implementation review per ticket group: 01–05, 06–07, and 08–10.
- Each confirmed finding gets an entry in `docs/agents/experiment-lessons.md`.

## Deferred

- Sampled intervention runs: the direct test of J.
- The held-out follow-up (ticket 14).
- Top-10 neuron-read ablations.
- The four SFT arms.
- Integrated gradients along λ.
- Combined removals.
- Removing harmful components.
- B-type fixed-text contrasts.
- Importance weighting of off-policy sets.

## Decisions recorded 2026-11-18 (user)

- **Hot layers:** from the effective-write figure (see Selection). {11, 15, 16, 24} confirmed on 2026-11-19, computed by code from the hashed source with the union rule.
- **Tolerances:** accepted. Production-1's tolerances carry over unchanged. The cached-against-forced rule is set from the rehearsal's measurements.
- **No time caps.** The run is not stopped for elapsed time. The watchdog, the crash stop and the end-of-run Pod stop remain; ticket 09's budget enforcement is removed.
- **Seeds:** the user has no preference, so the proposed defaults are used and recorded: split 20261118, bootstrap 1234, and one matching seed per group from the launch record.

## Decisions recorded 2026-11-19 (user)

- **Infeasible `layer64` control:** run the target anyway; report the control as unavailable (see Matched controls).
- **Draw split:** kept in addition to the question split (see Uncertainty and stability).
- **Seeds:** the defaults above are confirmed.
- **Hot layers:** confirmed, computed by code (see Selection).
- **Positive shortfall:** freeze the smaller positive group, match its control to the actual size, record the shortfall; never pad, never drop.
- **Intact repeat:** yes, one second intact greedy pass per arm (+300 answers).

## Open decisions (for review round 2)

1. Resolved above (hot layers).
2. Resolved above (positive shortfall).
3. Resolved above (infeasible `layer64` control).
4. Resolved above (tolerances).
5. Resolved above (intact repeat).

All open decisions from round 1 are resolved.
6. Resolved above (no time caps).
7. Resolved above (draw split kept).
8. Resolved above (seeds).
