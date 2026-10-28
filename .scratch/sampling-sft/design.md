# Sampling SFT arm — working design

Status: design confirmed by the user on 2026-10-28, including the final checklist below. No paid execution is authorized by this document; implementation requires a separate request.
Date: 2026-10-28

## Purpose and scope

Add a sampled-data SFT comparison to the existing gold-SFT and GRPO experiments. The supplied proposal constructs GSM8K answers using a gold-guided proposer and greedy likelihood refinement at k = 0, 2, 5, then trains rank-1 adapters with the saved gold-SFT optimization protocol. This is a proposed data-construction intervention; it does not by itself identify policy distance as the causal explanation of behavioral differences.

The supplied document is preserved unchanged in `sources/sampling-sft-design-supplied-2026-10-28.md`. It is the detailed proposal, not a record that every choice has been approved. This working design tracks verified facts and unresolved decisions. No spec or tickets have been generated.

## Verified existing contracts

- `pilot_eval/protocol.py`: pinned chat prompt requests step-by-step reasoning and a final #### number.
- `pilot_eval/sft.py`: gold targets retain the original GSM8K answer, including ####; prompt tokens are masked and one end-turn token is supervised. Overflow is rejected, not silently truncated.
- `pilot_eval/sft.py` and `sft_backend.py`: rank 1, alpha 1, dropout 0; seven projections across 28 layers; AdamW learning rate 1e-4, microbatch 1, accumulation 8, 64 optimizer steps, seed 42. This is 512 exposures, one pass. Use saved order and adapter payloads as authoritative identities.
- `sft_backend.completion_loss`: token-summed cross entropy divided by all supervised tokens in the accumulation window. Missing slots alter the objective and exposure contract; an entirely empty window is invalid.
- Actual Pilot 4 training config records a 1,024-token GRPO completion cap. The historical temperature-1 sampling baseline covers 128 questions, not 512. Stage 0 R0 code exists, but real expanded-cohort execution was not found locally.
- The saved SFT evaluation runtime is L4; this is not proof of its training hardware. Audit the training manifest before freezing a hardware requirement.

## Proposed pipeline

Source audit → immutable sampler/prompt plan → CPU correctness tests and bounded engine qualification → measured throughput/pilot → freeze production settings → resumable generation → immutable automatic grading plus separate human review → coverage/length/NLL gate → explicit choice of training arms → existing SFT trainer integration → durable adapters and handoff.

Evaluation and internal measurements are downstream consumers, not implied to have run when adapters are saved. Generation, scoring and training caps are separate settings. Expensive stages need a measured budget and explicit ceiling.

## Upstream fidelity and intended adaptations

Primary sources: https://github.com/aakaran/finetuning-with-sampling/blob/main/boost_math.py and https://arxiv.org/abs/2610.02140 .

The released code uses strict improvement in mean response-token log probability, whole-trace cuts, shorter regrowth, and end-token checks after the block's refinement steps. The decoded prefix inside the proposer user message plus the same IDs as assistant prefill is intentional upstream behavior. Fixed J=64 and a ceiling block count are adaptations. The paper's stochastic MH acceptance and theoretical distribution guarantees must not be attributed to this greedy port.

## Approved decisions — interview rounds 1–3

### Scientific scope and comparisons

This is a data-recipe comparison: gold SFT versus Sampling SFT k=0/2/5, with GRPO as a reference. No causal claim about on-policyness. The within-recipe k contrast varies search amount with gold conditioning and filtering fixed. Report pass rates and coverage separately per k; compare k arms on common covered problems. Plain NLL, length, format and leakage are descriptive candidate mechanisms, not established causal mediators. Sharpening relative to the model’s own samples is not measured in this run. Self-correct, random-correct versus searched, and length-matched controls are deferred.

### Generation and training scope

Generate all 512 problems for k=0/2/5, with data seed 0 and the declared deterministic per-request/retry seed derivation. After the gate and explicit approval, train k=0 and k=5 only, seed 42, matched to the verified gold-SFT protocol. This final scope answer supersedes the earlier round-1 suggestion to train all three. Retain k=2 data and gate row; any k=2 training needs separate approval. A claim that endpoint results differ meaningfully is not yet an executable rule.

Defer k=10, k=5+5 polish, extra seeds, self-correct SFT, evaluation, internals and Stage 0 follow-ups, each requiring its own approval/budget. No automatic arm merging. Intervals including zero mean inconclusive; an equivalence claim requires separately preregistered margins and an entire paired-bootstrap interval inside them. Illustrative margins in the reply are not approved thresholds.

### Gold training reuse and required captures

Gold SFT is not re-run. Sampling-SFT training uses gold's exact saved per-step problem order/batches with approved uncovered slots explicitly marked, its verified optimizer/configuration, checkpoint steps 0/8/16/32/64, and matching log fields. Use one training seed42. Training hardware is L4, FP32, with TF32 settings matching the actual gold-training metadata. These are preflight requirements: verify the original training manifest, not just an L4 evaluation manifest. Missing or contradictory evidence blocks training and is reported; do not silently substitute another GPU or redefine the historical match.

Required captures: PEFT-format adapters at 0/8/16/32/64; bit-identical checkpoint-0 payload versus gold (T15, including saved A0 and zero B); gold-matching training logs; and manifest with exact config, batches/slot omissions, seeds, source/runtime identities and hashes. Capture optimizer state and full per-step gradients optionally. Complete recovery state is mandatory only for a claimed mid-training resume; without it, recovery is a recorded restart from checkpoint0, not a weights-only resume.

### Coverage gate after Stage 3

Training must not start until coverage passes. Zero uncovered passes coverage only; other gates and explicit training approval remain required. With 1–5 uncovered per arm, keep marked empty slots and retain 64 nonempty optimizer steps. T16 must exercise batch sizes 3, 4, 5, 6, 7 with the existing total-supervised-token denominator: step completes, loss matches an independent hand computation, gradients are finite, and scheduler advances exactly once. Any failure stops for repair; there is no duplicate fallback. More than 5 uncovered blocks training and requires a revised plan. Never fill with gold or incorrect traces; never retry without a bound.

Missing slots retain problem id and reason in batches.json and coverage summaries. Gate comparisons use the intersection covered by all k0/k2/k5 arms, alongside full-coverage summaries per arm. Training uses each arm's own covered slots to stay as close as possible to historical gold SFT's 512 problems. Report uncovered lists and their union/intersection. Flag pairwise differences of more than 5 uncovered ids before training approval (symmetric difference). Eventual evaluation uses the same 150 held-out problems, but evaluation execution is outside this experiment's current scope.

### Termination and retries

Continue extension blocks while no stop token and response length is below cap, with safety limit 2 * ceil(cap/J). Preserve upstream within-block refinement ordering. Each generation attempt has exactly one end_state: finished, capped, block_budget_exhausted, engine_error. Proposed precedence when EOS also reaches cap: finished; stop-token accounting still needs qualification. Only finished, correctly graded traces can be eligible. Capped/exhausted attempts are ineligible and retryable: initial attempt plus at most three new-seed model-outcome retries. Engine errors are operational events: replay the same request/seed; after two failures stop and report rather than consume a model-outcome retry. The counter belongs to one problem’s generation or scoring request at one block, refinement step and model-retry index; initial failure plus one same-seed/runtime replay exhausts it.

The length-based loop intentionally extends cases that the supplied fixed-block algorithm would leave unfinished; a safety-exhausted trace can still remain unfinished. Every trace attempt, including model retries, has one terminal-state row. End-state counts sum to attempt records. Every problem per arm has exactly one final covered/uncovered disposition; a covered row references exactly one finished, correct attempt. An engine-error disposition is provisional while the run is blocked and cannot pass the coverage gate. Interruptions are events, not terminal model outcomes; partial blocks are discarded and redone on resume. Preserve operational request failures separately even if replay succeeds.

Empty generation without a real stop token, or missing/malformed scoring prompt_logprobs, is an engine/protocol error. Stop-only output is finished but incorrect and enters bounded model retries. Stop-only and early-stop proposals inside refinement remain legal score-based proposals; only the final trace is graded.

### Runtime freeze and replay

Sampling hardware is L40S. Choose settings at Stage 0.5; freeze before the 32-problem Stage 1 pilot. A different GPU requires an explicitly revised plan, not an implicit fallback. Record GPU type, driver, CUDA, Torch/vLLM versions, dtype, actual worker TF32 flags, max_num_seqs, max_num_batched_tokens, GPU memory utilization, prefix-cache setting and max_model_len; hash the block as runtime_hash. All sampling arms, model-outcome retries and resumes must match. A mismatch refuses resume. Changed settings require a new run_id and regeneration of all k arms; never mix runtime hashes in a comparison.

The 32-problem pilot qualification compares uninterrupted execution against kill-after-block-2/resume. Checkpoint at completed block boundaries only; discard partial scientific records and replay the partial block. Submit logical requests in fixed recorded order, by problem id then request type, while preserving extension/refinement dependencies. Clear prefix cache at each block boundary in qualification and production; within-block caching stays on. Record additional prefill work. Response IDs, scores, accept decisions, traces, steps and coverage match exactly; operational timing, PIDs, error and interruption events live separately.

If exact replay fails, stop before Stage 2. Requalify in order: disable prefix cache entirely; constrain scheduler/batch settings; enable available engine/framework deterministic modes. Each configuration change creates a new runtime_hash and qualification attempt; record throughput cost. Availability and effectiveness of these modes must be verified, not assumed. No numerical relaxation is authorized. Seeds alone do not guarantee reproducibility. Shared seed identities couple requests while paths coincide; they do not ensure identical outputs after arms diverge.

### Human eligibility review

Keep original flexible-v3 scores and generated text immutable. Separate confirmed human intended-answer labels may override eligibility for a finished trace when automatic extraction failed. They cannot edit the trace, supply a gold answer, or make unfinished text eligible. Record label provenance/revisions and distinguish agent suggestions from user-confirmed labels.

### Budget

Hard total ceiling USD25 including GPU and storage. Sub-caps: sampling/setup/tests/profile/pilot/retries/gate USD10 (the T1 baseline is deferred; the sub-cap is unchanged); training USD6; network volume and genuinely shared overhead USD9. Setup, idle, debugging and reruns count against the stage served, not the storage line.

Before every job, persist a conservative reservation in budget.json: GPU hourly rate times estimated hours times 1.5, plus setup-time cost. Include existing spent and outstanding reservations; launch only if the proposed sum stays below both the stage and total hard ceilings. Avoid counting setup twice if already in the runtime estimate. At 80% of any sub-cap or total, pause for approval before the next job even if headroom remains. Reconcile reservations to actual billed/estimated cost after each job; do not release an uncertain interrupted reservation without reconciliation. Store cumulative-cost snapshots in run_k{k}.json; shared charges count once in the global ledger.

Reserve storage for its expected lifetime, initially one month of approximately 60GB network volume (planning estimate USD4); extend if retained longer. Storage bills while Pods are stopped. Hourly rates and estimates in the supplied document remain unverified assumptions; obtain actual quotes before reserving a job. The USD15 top-up is account credit, not the budget.

After verified export to the network volume and a downloaded copy, the runner requests that its own Pod stop as its final cloud action. Persist stop_requested before the call; a surviving client/provider check records confirmed stopped separately. Do not claim the self-stopping process observed its own successful shutdown. Verify exported products with hashes/manifests; export failure cannot authorize deletion or unexported termination. Deleting network volumes or terminating an unexported Pod requires explicit user approval. Export-failure handling must preserve the budget ceiling and report before further paid work; never silently leave an unmonitored Pod running.

## Operational requirements already inherited from this repository

- Canonical storage under artifacts/plans, artifacts/runs and artifacts/reports; replace the proposal's outputs/sampling_sft root in the working implementation plan.
- Check existing products first; retain source hashes, resolved configuration, runtime, status, response IDs and logs. Resume only verified sealed units.
- Preserve automatic scores and original text; human labels and any selection override are separate versioned records.
- Saving optimizer state and per-step analysis gradients is optional under the final user-confirmed capture policy. Resuming training nevertheless requires a verified complete recovery checkpoint (adapter, optimizer, scheduler/progress and RNG state as applicable). If that state was not saved, do not resume from an adapter alone: preserve the interrupted attempt and restart from the verified checkpoint-0 under a new recorded attempt, subject to budget reservation and stage approval. Optional capture is not permission for an invalid resume.
- Prompt/J changes after the pilot require a new immutable version and a complete pilot rerun. Do not mutate an existing production plan.

## Baseline and measured gate quantities

The temperature-1 baseline is deferred. Do not generate 512×8 baseline samples, reuse historical samples, calculate typical-NLL offsets, or imply sharpening relative to the model’s own distribution was measured. Defer that baseline to later internals/Stage 0 work with separate approval and budget. Its proposed future settings are scorer prompt, temperature1, top-p1, no top-k restriction, repetition penalty1, cap1024. Current tables omit offset columns and explicitly mark that measurement not measured.

Gate rows cover gold and k0/k2/k5. A trace's NLL is the negative mean response-token log probability under the scorer prompt, including the actual stop token when present. Aggregate as an equal-problem mean of these answer-level values. Report length median and p90, explicitly counting the same retained response IDs. Report each arm's complete eligible coverage and the three-arm common-coverage comparison; include gold on the corresponding common problem IDs for paired comparisons.

For each k-arm contrast, bootstrap whole problems in the common cohort 10,000 times with local seed42. Use percentile95% intervals. The length estimand is the difference of arm medians, recomputed within every resample, not the median of paired differences (which may be a separately labelled secondary column). NLL contrasts use differences of equal-problem means. Preserve bootstrap configuration and cohort ids. No offset analysis runs without its baseline. Later E0.6a must match these definitions or document differences.

Leakage diagnostics: literal << occurrence rate; fraction of response word4-grams present in cleaned gold. Under-four-word responses have undefined overlap, not zero. Preserve numerator, denominator and undefined count. The exact deterministic word normalization belongs in the metric contract before implementation; it must not silently alter the approved word-based measure. All intervals and leakage/NLL summaries are descriptive, not causal mediation evidence. No equivalence rule or multiplicity-based success claim is introduced.

## Scoring qualification — approved T3 gates

Before Stage2, compare engine and independent HF FP32 scores on 20 traces chosen by a predeclared qualification fixture/selection plan. Include short and long traces, real-stop and capped cases, and a trace whose first generated token follows an empty response prefix. Qualification preparation must check that every required category is present; do not silently omit a category or search for traces that pass. If the pilot has no capped case, a bounded deliberately capped fixture must be identified in the fixture plan before the score comparison; missing fixtures block qualification rather than expand work silently.

For every trace, exact aligned token IDs and positions are mandatory; absolute mean log-probability difference <=0.001 nats/token and maximum absolute per-token difference <=0.01 nats. Include stop when actually present. Save signed/absolute mean error, maximum per-token error, worst position, aligned IDs and scores. No mean cancellation can hide a failing token. Investigate failures in order: prompt/template mismatch; slicing offset; dtype/TF32. Failure blocks Stage2. No gate changes without user approval.

This validates score agreement on the fixture, not cross-engine equality of near-tie acceptance decisions. Sampling/refinement uses one frozen engine. T3 is the score gate; the supplied T14 is batch replay, not a second score definition. CPU independent-math/slicing tests remain separate from this real-engine evidence.

## Pilot alarm and revision rules

J64 stays fixed. Report median/p90 lengths without an automatic J adjustment. Stop for review if initial-attempt unfinished fraction (capped plus block_budget_exhausted) >0.02; initial-attempt correctness <0.90; or the measured Stage2 reservation exceeds available budget headroom. Compute the initial-attempt metrics over the fixed32 problems; engine failures already stop and cannot disappear from a denominator. At n32 a single unfinished trace exceeds2%, and at most28 correct is below90%. Print each alarm's measured value, denominator and threshold, including alarms that do not fire.

Use pilot throughput and observed lengths to re-estimate the full-generation reservation. Prompt/J changes require user approval, new template/algorithm identity and run_id, and a complete pilot rerun. J belongs in the frozen algorithm hash. Never increase cap or rewrite a template automatically. Runtime changes follow the separate runtime_hash/requalification contract.

## Stage handoffs and scope boundaries

| Stage | Work and durable output | Gate |
| --- | --- | --- |
| Source preparation | Exact model/tokenizer/source hashes, train/eval IDs, gold config/order/A0 and resolved plan | Missing source identity blocks dependent work |
| 0 / 0.5 | CPU tests, bounded real-engine tests and throughput settings; runtime_hash and measured budget | Correctness, reservations and frozen settings |
| 1 | Fixed32 k5 pilot, replay evidence, T3 fixture scores, alarm report | Exact replay, score gates and alarm review |
| 2 / 3 | k0/2/5 on512; bounded retries, response/step records, events, automatic and reviewed labels | Terminal/disposition reconciliation and coverage |
| 4 | Plain-NLL/length/format/leakage gate sheet, common cohort, bootstrap intervals, budget | Explicit training approval; no baseline/offset |
| 5 | Seed42 k0/k5 on verified L4; gold-order slots; adapters0/8/16/32/64, logs and manifest; optional recovery state/gradients | T15 exact initial adapter, T16, verified trainer config and budget |
| 6 | Verified export manifests, network-volume and downloaded copy receipts; Pod-stop request/confirmation | Preserve durable products; no implicit evaluation/internal work |

## Implementation preflight and unresolved facts

These are source/engineering checks, not permission to choose different scientific controls: audit the actual gold-training runtime manifest; pin a compatible vLLM environment and decoding defaults; verify worker numeric flags; verify real stop-token return/cap semantics; define first-stop validation and same-boundary EOS/cap precedence; specify deterministic fixture identities, word normalization and p90 implementation; bound profiling jobs; implement atomic block sealing, writer ownership, interruption recovery, download receipts and external stop confirmation. Record any unsupported engine capability as blocked with evidence. A missing fact must not be filled from an unrelated evaluation manifest or an engine default.

The consolidated design is now confirmed. Implementation begins only on a separate request. No paid execution follows from this confirmation. The first spend, including sampling-GPU Stage0 tests and Stage0.5 profiling, requires its own explicit go-ahead after preflight. Separate training approval remains required after the gate. Any remaining policy choice exposed by preflight is brought back before affected execution.

## Evidence and limitations

A targeted local baseline check passed5 tests (6 deselected). The design-review full local regression completed:644 passed,39 skipped,4 warnings in2578.05 seconds. A later inherited implementation regression completed669 passed/39 skipped/4 warnings in2621.17 seconds; it collected before Ticket02 tests and is not a commit-isolated final suite claim. Some model-backed checks were skipped because the local Torch installation fails to import (missing libtorch_cpu.dylib). This is not full model-backed validation. Neither check is sampler qualification or scientific execution evidence. Timing and prices in the supplied proposal remain unverified estimates. No training, GPU sampling, cloud rental or upload was performed during this review. The supplied design remains unchanged in sources; this consolidated design supersedes its conflicting scope, coverage, baseline, merge, termination, replay and budget rules.

## Final user confirmation checklist — 2026-10-28

All items below are incorporated and supersede conflicting statements in the supplied source document.

- Data-recipe framing; no causal on-policyness claim; descriptive gates, baseline/offset not measured.
- Generate k0/k2/k5 with data seed0; train k0/k5 only after explicit gate approval, seed42. k2 training separately approved; no automatic merging.
- Do not rerun gold SFT. Reuse verified gold config, exact step batches, logs and checkpoints0/8/16/32/64; require L4 FP32/TF32 match and T15 checkpoint0 identity.
- Initial attempt plus at most3 model retries; coverage0 passes coverage only;1–5 requires T16 batches3–7 and marked omissions;>5 blocks. No duplication. Three-arm common gate cohort; arm-specific training coverage; symmetric-difference>5 flag.
- Length-based loop and doubled block safety limit; four attempt end states; same-request engine replay;512 final dispositions per completed arm; separate events.
- L40S sampling; runtime_hash after Stage0.5; same runtime for arms/retries/resumes; block-boundary state, fixed logical request order and prefix-cache reset; exact pilot resume before Stage2. Configuration changes create new attempts and regenerate all k arms.
- T3 on20 predeclared traces: exact alignment, mean error<=0.001 nats/token and max token error<=0.01. Fixed J64 and explicit pilot alarms; no automatic changes.
- Equal-problem NLL including stop; difference-of-medians bootstrap for length; approved whole-problem uncertainty and leakage diagnostics.
- Required adapter/identity/log/manifest captures; optimizer and per-step gradients optional, with strict recovery-state requirements for any claimed training resume.
- Deferred with separate approval: evaluation, internals, Stage0 research work (distinct from this sampler's Stage0 setup tests),512×8 baseline/offset,k10,k5+5,self-correct SFT,extra seeds.
- USD25 ceiling;10/6/9 sub-caps; reservations before launch;80% pause; automatic own-Pod stop only after verified export; explicit deletion/unexported termination approval.
- Design confirmation grants no paid-job permission. First sampling-GPU spend needs a separate post-preflight go-ahead.
