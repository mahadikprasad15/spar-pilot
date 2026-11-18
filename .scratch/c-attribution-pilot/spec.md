# Correctness (J) attribution pilot — online and offline GRPO

**Status:** needs-triage. Draft v1 (2026-11-18). Design only: no code, GPU run, threshold or budget is approved by this file. Proposed values are marked **[proposed]**; open decisions are listed at the end.

Builds on `.scratch/adapter-neuron-attribution/` (instrument, gates, matched controls, units, chain). Where they differ, this file governs this pilot only; the production-1 study and its trace/stop metrics are unchanged.

## Problem

The production-1 attribution ranked adapters by first-order effects on two teacher-forced proxies measured on the untuned model's own correct traces. Removing the top-10 adapters raised trace log-likelihood in every arm, so that metric mostly measures distance from base-model text. The stop-pull proxy did not predict generated length. Neither targets correctness.

This pilot attributes the **sampled probability of a correct answer**, J, to the 196 rank-1 adapters and to the `down_proj` neuron-read terms of online and offline GRPO. It then checks the rankings by removing the selected components and measuring greedy accuracy.

## Questions

1. **Adapter ranking.** For each arm, what is dJ/dλ_i for every adapter gate at λ = 1, with uncertainty?
2. **Neuron-read ranking.** For each `down_proj` adapter, how does that gradient split across its 8,960 read terms?
3. **Stability.** Is the 196-vector stable across draw halves and question halves?
4. **Behaviour (exploratory).** On the same 150 questions, does removing the top-10 adapters, the top-10 neuron reads or the top-64 neuron reads lower greedy accuracy more than size-matched controls?

Non-goals: the four SFT arms (no final-checkpoint sampled draws exist; see Deferred), causal claims beyond the tested removals, held-out generalization (prepared as a follow-up stage, not run in the initial launch), and changing the production-1 metrics or results.

## Data (verified 2026-11-18 unless marked)

Catalog ids from `catalog/` on `main`. Both files: 150 questions × 8 draws, step 64, policy T = 1.0, top_k 0, top_p 1.0, repetition penalty 1.0, max 1,024 new tokens, EOS 151645, seed 42 + ordered batch index (4 prompts × 8 draws per batch), FP32, A100, eager attention, PEFT 0.18.0, transformers 4.57.6.

| Arm | Catalog id | File (under `artifacts/`) | sha256 | Adapter (per row) | Groups all-correct / all-wrong / mixed | Mean reward | Capped | Invalid |
|---|---|---|---|---|---|---|---|---|
| online-grpo | `online-grpo.s64.sampled-x8.test150` | `imports/pilot4-behaviour-breakdown-2026-10-22/grpo/responses.jsonl` (select step 64, mode sampled) | `4af7926a…` | `70d2d56c…` | 49 / 12 / 89 | 0.675 | 0 | 46 |
| offline-grpo | `offline-grpo.s64.sampled-x8.test150` | `imports/offline-grpo-v1/runs/offline-grpo/offline-grpo-v1-behaviour/behaviour/results/responses.jsonl` (same select) | `a1cd79aa…` | `66015518…` | 57 / 13 / 80 | 0.693 | 1 | 39 |

- **Rewards.** Re-scored on CPU with `scoring.score_gsm8k_flexible_v3(text, gold, capped=capped)`. The scorer source on `main` and on this branch hashes to `c958d2de…`, the hash the behaviour configs record. Stored `reward` and stored `flexible.correct` agree with the re-score on 1,350/1,350 step-64 rows per arm (sampled + greedy). Capped rows score 0 (`cap_reward 0`); invalid rows score 0. **These rewards are reused; no re-scoring is needed for attribution.**
- **Tokens.** Each row stores `generated_token_ids`: the sampled tokens with the final EOS for EOS-terminated rows, all 1,024 for capped rows. `token_ids` equals `generated_token_ids` without EOS on every EOS row. Token 151643 never appears inside a generation. Prompts are the stored templated strings, tokenized with `add_special_tokens=False` (mean 115, max 216 tokens). Teacher-forced input = prompt IDs + `generated_token_ids`: 422k tokens online and 405k offline, longest sequence 1,151.
- **Cohort.** `gsm8k-test-cohort150` (ADR 0001). Its prior uses are listed in `catalog/cohorts.yaml`, including production-1 Stage D. No overlap with the 512 training questions. Attribution and the initial ablation both use this cohort, so **the ablation results are exploratory** (selection and test on the same questions).
- **Untuned greedy baseline.** `untuned.s0.greedy.test150` exists under the same settings (acc 0.68). The intact step-64 greedy runs exist too (online 0.76, offline 0.7267). They are consistency references only. The ablation stage regenerates intact and untuned greedy runs in its own runtime and batching, and compares within that run.

## Estimand and estimator

For arm θ with adapter gates λ (λ_i = 1 intact; the branch is λ_i · s · B_i(A_i x)):

    J(λ) = (1/Q) Σ_q E_{y ~ π_λ(·|q)} [R(q, y)],   R ∈ {0, 1} = flexible-v3 correct (capped → 0)

**Per-response metric (new, `seqlogp`).** M(q, y) = Σ_{t ∈ G} log p_θ(y_t | q, y_<t), summed (not averaged) over G = every generated position: every sampled content token plus the sampled EOS when the row stopped at EOS, and all 1,024 sampled tokens with no EOS term when capped. Raw log-softmax of the FP32 logits matches the sampling distribution, because T = 1, no top-k or top-p truncation, repetition penalty 1.0, and `grpo_gpu._generate` passes these explicitly over Qwen's `generation_config` defaults (rep 1.1, T 0.7, top_p 0.8, top_k 20).

**Positions.** The scored outputs are the generated positions G. The attributed positions are all unpadded positions, prompt prefill included, because the existing contraction attr_i(e) = −Σ_t c_t ∂M/∂c_t already sums over every unpadded position. A gate changes prompt computations that influence later tokens; restricting to response positions would be incomplete, and the existing response-only mutation test fails for that reason.

**Per-response attribution (existing machinery, new metric).** attr_i(q, k) = −Σ_t c_t ∂M_qk/∂c_t = −∂M_qk/∂λ_i at λ = 1. Neuron reads: attr_j(q, k) = −a_j Σ_t z_{j,t} ∂M_qk/∂c_t, with Σ_j attr_j = attr_i per response.

**Leave-one-out weights (detached).** For question q with n_q draws (n_q = 8 here):

    b_qk = (Σ_k' R_qk' − R_qk) / (n_q − 1),   w_qk = (R_qk − b_qk) / (Q · n_q)

Questions are weighted equally and draws equally within a question. All-correct and all-wrong groups stay in Q and contribute exactly 0. n_q = 1 is refused (no leave-one-out baseline). Rewards and weights never enter the graph.

**Aggregation (offline, no loss and no optimizer step).**

    removal_J_i = Σ_qk w_qk · attr_i(q, k)     # first-order predicted change in J when gate i goes 1 → 0
    dJdλ_i      = −removal_J_i                 # gradient for strengthening gate i at λ = 1

The same holds for every neuron read j: Σ_j dJdλ_j = dJdλ_i for each `down_proj` adapter. Because aggregation is linear, the identity holds whenever it holds per response; the pilot still checks it on the aggregates.

Under on-policy sampling, Σ w ∇M is an unbiased estimate of ∇J (the score-function identity with a leave-one-out baseline). It is local: it says nothing exact about removing a whole gate, which the ablation stage tests.

### Sign convention (must appear in code docstrings, outputs and report headers)

| Quantity | Meaning | Positive means |
|---|---|---|
| `attr_i(e)` (existing) | first-order Δ metric when gate i is removed (1 → 0) | removing raises the metric |
| `removal_J_i` | first-order ΔJ when gate i is removed | removing is predicted to **raise** J (component hurts correctness) |
| `dJdλ_i` | ∂J/∂λ_i at λ = 1 | strengthening raises J (component **helps** correctness) |

"Positive-attribution" components, the ones selected for removal, are those with the largest `dJdλ` (most helpful). Their removal is predicted to lower J. No code path minimizes a loss; if one is ever written as L = −Σ w·M, its gradient is −dJdλ.

## Uncertainty and stability (descriptive)

- **Bootstrap.** 10,000 resamples of questions, seed 1234 **[proposed]**; per adapter and per neuron, a percentile interval on dJdλ, recomputing leave-one-out weights within each resample.
- **Split-half by draws.** Draws 0–3 vs 4–7, leave-one-out within each half (n = 4). Report Spearman of the two 196-vectors, top-10 overlap, and sign agreement among the top 20.
- **Split-half by questions.** Two halves fixed by sorted ID parity. Same summaries.
- **Report also:** J̄, the number of informative (mixed) questions (89 / 80), and the fraction of |Σ dJdλ| coming from the top-10.
- **Rule for proceeding.** Not a gate. If draw-split Spearman is < 0.5 **[proposed]**, the report states that the ranking is unstable and the ablation is interpreted as a test of a noisy selection.

## Checks

**CPU (tiny random Qwen2 + PEFT 0.18.0, no downloads), before any GPU run:**

1. The `seqlogp` mask includes the final EOS target. A capped row includes all 1,024 targets with no EOS term. An EOS-only response has one target. A sum, not a mean (the mean-not-sum mutation fails).
2. Row builder: the counts in the Data table are reproduced from real-format fixtures; no filtering (wrong, invalid, capped, all-correct and all-wrong rows kept); refuses a row whose `adapter_sha256` ≠ the arm checkpoint, a duplicate `draw_id`, an EOS before the end of an EOS row, a capped row without exactly 1,024 tokens, or a reward that disagrees with the scorer re-score.
3. Leave-one-out weights: hand-computed fixtures; all-same groups give exactly 0; n_q = 1 refused; Σ_k w_qk = 0 per question.
4. Sign: a synthetic arm in which one adapter is constructed to raise P(correct) gives dJdλ > 0 and removal_J < 0. Exact enumeration on a 2-token vocabulary: the exact ∇J matches the estimator's expectation.
5. Finite differences on the fixed-sample surrogate F(λ) = Σ w_qk M_qk(λ) (weights frozen): the central difference over gates (`branch_gates`, `neuron_gate`) matches dJdλ for 5 adapters and 3 neurons. This checks the implementation, not whether the samples estimate the true ∇J.
6. Neuron identity: Σ_j dJdλ_j = dJdλ_i at `identity_rtol`, per response and aggregated.
7. Old metrics unchanged: every existing `test_attribution_instrument.py` / `test_attribution_metrics.py` test passes unmodified, and `trace` + `stop` outputs are bit-identical with and without `seqlogp` requested.
8. Cached vs teacher-forced: incremental KV-cache decoding with left padding (the generation path) and right-padded full teacher forcing (the instrument path) agree on per-token log-probs of the same tokens within tolerance on the tiny model.
9. Interventions: zeroing lora_B (adapter) and zeroing a_j in lora_A (neuron read) equal the `removal` hook and `neuron_gate(g = 0)` on logits, and multi-neuron removal across layers is exact.

**GPU qualification (real arms, a few rows; recorded, then automatic sign-off as in production-1):**

- (a) Cached vs teacher-forced log-prob agreement on 16 real rows per arm: report max and median |Δ log p| per token and |Δ M| per row. **[proposed]** tolerance: max |Δ log p| ≤ 1e-3.
- (b) Replay of one sealed sampled batch (same seed, batch composition, left padding): token-identical? **Report-only**; hardware and runtime differences can break identity without invalidating the scoring.
- (c) Surrogate finite differences with an FP64 reference (`full_fp64`, as decided 2026-11-16): 5 adapters + 3 neurons, existing tolerances (median relative error ≤ 0.05, max ≤ 0.20).
- (d) Batch invariance at the chosen token budget: per-response |Δattr| ≤ 1e-4 × max |attr|.
- (e) Identity at `identity_rtol` = 1e-4 **[proposed, as production-1]**.

## Selection (frozen and hashed before any generation)

Per arm, from the full-cohort aggregates:

- **Top-10 adapters** by dJdλ (largest positive). Ties broken by layer, then module order.
- **Matched adapter controls:** existing `attribution_ablation.matched_controls` — same module type, discovery write within ±20%, seeded, drawn from outside the arm's top-10. Here the write is ‖s·b‖ · mean |c_t| over these rows. Unavailable matches are recorded, not loosened.
- **Top-10 and top-64 neuron reads** by dJdλ_j, **pooled across all 28 `down_proj` adapters** **[proposed; open decision 1]**.
- **Matched neuron controls:** same layer as each target, term size |a_j| · mean |z_j| within ±20%, outside the top-64, seeded **[proposed; open decision 3]**.
- **Also frozen:** the most harmful adapters (most negative dJdλ). They are reported but not ablated in this launch.

## Interventions

- **Adapter removal:** lora_B of the chosen adapters set to 0, either as an edited checkpoint verified against the `removal` gate on probe logits (as Stage D) or as an in-memory weight edit with save and restore.
- **Neuron-read removal:** a_j set to 0 in the chosen `down_proj` adapters' lora_A. This is exactly c′ = c − a_j z_j, i.e. `neuron_gate` at g = 0, applied at prefill and every decode step.
- Adapter and neuron removals are not combined in this launch.

## Generation-ablation stage

**Initial launch (exploratory; greedy; same 150 questions):**

| Arm | Conditions |
|---|---|
| online-grpo | intact; top10-adapters; match10-adapters; top10-neurons; match10-neurons; top64-neurons; match64-neurons |
| offline-grpo | same 7 |
| untuned | greedy reference |

That is 15 runs × 150 = 2,250 greedy responses: FP32, the qualified batch rule from production-1, 1,024 cap, EOS 151645, repetition penalty 1.0. Every response is saved as JSONL with a config file (AGENTS.md).

Per condition, report:
- greedy accuracy with a Wilson interval;
- change vs the arm's intact run in percentage points;
- percentage of intact accuracy remaining (100 × ablated / intact);
- the top-condition minus its matched-control difference, with a paired bootstrap interval over questions;
- gains and losses (paired) and the exact McNemar p (descriptive);
- invalid and capped counts, and mean and median response tokens.

The untuned row is reported alongside. **Labels:** "exploratory — components were selected on the same questions", and "greedy accuracy is a different target from the sampled J used for attribution".

**Prepared follow-up (not launched):** the same conditions on other GSM8K test questions. That means the 1,319 test questions minus the 150-question cohort minus the 20 rehearsal items: a seeded 150 **[proposed]**. Measure sampled accuracy (8 draws, the attribution policy) and greedy accuracy separately, with the same summaries. This stage is the non-exploratory test.

## Outputs and operations

```text
artifacts/runs/c-attribution/<run-id>/
  config.json  launch.json  status.json  logs/
  rows/<arm>/rows.jsonl + manifest (source sha256, selects, counts)
  attribution/units/        BlockStore per (arm, batch): attr_seqlogp [n, 196], neuron reads per down_proj layer, sum|c|, token counts
  aggregate/<arm>/          adapters.json (dJdλ, removal_J, intervals, ranks), neurons.npz, stability.json, identity.json
  qualification/            checks (a)–(e)
  freeze/<arm>.json         hashed selections and controls
  generation/<arm>/<condition>/<mode>/units + responses.jsonl + config.json
  report/                   report.md, report.json, figures
```

- Units are sealed and resumable (existing `attribution_units` / BlockStore); duplicates are refused.
- Caps come from the launch record. Export and verified copy follow the production-1 chain; the Pod is stopped at the end.
- Every generation run and selection is registered in `catalog/` (response sets, plus the cohort's `prior_uses`). This branch does not yet contain `catalog/`: merge `main` first (open decision 8).

## GPU workload estimate (A100-SXM4-80GB)

| Stage | Size | Basis | Estimate |
|---|---|---|---|
| Qualification (a)–(e) | 16 rows per arm, FP64 finite-difference copy | production-1 qualification took 12 min | 15–25 min |
| Attribution (one metric) | 2,400 rows, 827k tokens | production-1 attribution ran 975 tokens/s for two metrics (1 forward, 2 backward, pass 2); one metric should be at least that fast | 12–15 min |
| Aggregation, bootstrap, freeze | CPU-side | — | < 5 min |
| Greedy ablation | 2,250 responses | production-1 Stage D ran 1,350 greedy in 45 min at the qualified batch (≈ 30 per minute) | 75–90 min |
| Export and copy | — | production-1 | 10–15 min |
| **Total** | | | **≈ 2–2.5 GPU-hours**; proposed caps: 3 h total, 45 min attribution, 2 h generation |

Follow-up held-out stage at full size: 2 arms × 7 conditions × 150 × 8 sampled + greedy ≈ 19,000 responses, about 5–10 h. A reduced version (intact, top10-adapters and match10-adapters; 100 questions × 8) is about 4,800 sampled + 600 greedy, roughly 2–3 h.

## Run tier: Pod only, no Colab stage [proposed]

Production-1 went CPU → Colab rehearsal (06a) → Pod rehearsals → production. Most Colab findings were Colab-specific (hidden child-process stderr, preinstalled torchvision, a 40 GB device) or numerical issues now settled: the FP64 finite-difference reference and the token-budget bounds. Every failure that cost a Pod launch was a boundary bug that no earlier tier exercised: a wrong preflight argument, a YAML number read as text, the HF cache on the volume, the stop command syntax, the container disk size, and per-volume chain state. Ticket 10 (a CPU whole-chain test through the real backend, with each recorded Pod failure put back) is designed to catch that class. This pilot reuses the qualified Pod tooling and is about 2 GPU-hours.

Proposed route: **CPU tickets 01–10 → one attended Pod rehearsal (11) → a fresh production launch (12)**. Colab is a fallback only, if no Pod is available.

Pod lessons built into the tickets:

- 120 GB container disk; preflight's real body checks free space, mount and volume identity (09).
- `HF_HOME` on the container disk, and every download hashed (08, 09).
- A sealed input bundle covering every bound input, verified against its seal (08).
- Launch-record values type-checked, nested leaves included (09).
- A new run ID and a separate volume subtree per launch (09).
- The stop command tries both syntaxes and logs its output, and the rehearsal exercises a real stop (09, 11).
- Watchdog ownership only after lock and precheck; a bounded recovery before stopping; no automatic rehearsal → production handover (09).
- Empty-disk resume, with expected work reconstructed before any stage is skipped (09, 10, 11).
- Budget enforced at unit boundaries, export time included (09).

## Launch plan

1. **CPU tickets 01–10**, red/green, committed after each green step; the full suite is run and reported at session start. Baseline on 2026-11-18: 804 passed, 42 skipped, 1 failed in `test_sampling_sft_sources.py` (the sampling-SFT notebook test; unrelated, pre-existing).
2. **Attended Pod rehearsal** (11; needs approval): 2 arms × 32 rows, qualification, freeze, small greedy and sampled generation, interruption and resume from an empty disk, export, clean restore, a real Pod stop.
3. **Production launch record** (approval): new run ID, caps, Pod ID and stop command, tolerances, seeds, resolved decisions.
4. **Production** (12): qualification → rows → attribution → aggregate and freeze → greedy ablation → report → verified copy and export → Pod stop. Analysis outcomes never stop the chain; failed qualification does.
5. **Report and catalog registration** (13), then a decision on the held-out follow-up (14).

Recommended before step 4: a short prereg recording predicted directions, e.g. top10 lowers accuracy more than match10, and the expected stability (the `preregister-experiment` skill).

## Review process

As for production-1:

- **Spec rounds:** `review-round-N.md` and `review-response-round-N.md` in this folder, until the open decisions are resolved and the status moves to ready-for-agent.
- **Implementation:** one review per ticket group (01–05, 06–07, 08–10) before the rehearsal, recorded the same way.
- **Lessons:** each review finding confirmed by a test or a run gets an entry in `docs/agents/experiment-lessons.md`.

## Deferred

The four SFT arms (they need 150 × 8 sampled draws from each final checkpoint: about 4,800 generations, roughly 1.5–3 h); integrated gradients along λ (several strengths); combined adapter + neuron removals; removing harmful components; B-type fixed-text contrasts on the untuned draws; importance weighting of off-policy sets.

## Open decisions

1. Neuron-read selection: pooled across all 28 `down_proj` adapters (proposed) or within each arm's top `down_proj` adapter only?
2. Ranking direction for this launch: helpful only (proposed), or also a harmful-removal condition?
3. Matched neuron control: same layer and term size ±20% (proposed), or same adapter only?
4. Qualification tolerances: keep production-1's finite-difference tolerances and identity rtol; cached-vs-forced max |Δ log p| ≤ 1e-3?
5. Replay token identity: report-only (proposed) or a gate?
6. Run tier: Pod only, after the CPU real-path test (proposed), or Colab first; caps (proposed: 3 h total).
7. Follow-up held-out stage: size (150 questions, 8 draws), and which conditions.
8. Merge `main` (catalog) into this branch before implementation (proposed: yes; no file overlap except `pyproject.toml`).
