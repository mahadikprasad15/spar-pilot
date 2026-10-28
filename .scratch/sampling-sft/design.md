# Sampling SFT arm — working design

Status: design review; no implementation or paid execution authorized by this document.
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

## Approved decisions — interview round 1

### Scientific scope and comparisons

This is a data-recipe comparison: gold SFT versus Sampling SFT k=0/2/5, with GRPO as a reference. No causal claim about on-policyness. The within-recipe k contrast varies search amount with gold conditioning and filtering fixed. Report pass rates and coverage separately per k; compare k arms on common covered problems. Distance/NLL, sharpening, length, format and leakage are descriptive candidate mechanisms, not established causal mediators. Self-correct, random-correct versus searched, and length-matched controls are deferred.

### Generation and training scope

Generate all 512 problems for k=0/2/5. After the gate and explicit approval, train k=0 and k=5 only, seed 42, matched to the verified gold-SFT protocol. This final scope answer supersedes the earlier round-1 suggestion to train all three. Retain k=2 data and gate row; any k=2 training needs separate approval. A claim that endpoint results differ meaningfully is not yet an executable rule.

Defer k=10, k=5+5 polish, extra seeds, self-correct SFT, evaluation, internals and Stage 0 follow-ups, each requiring its own approval/budget. No automatic arm merging. Intervals including zero mean inconclusive; an equivalence claim requires separately preregistered margins and an entire paired-bootstrap interval inside them. Illustrative margins in the reply are not approved thresholds.

### Coverage gate after Stage 3

Training must not start until coverage passes. Zero uncovered passes coverage only; other gates and explicit training approval remain required. With 1–5 uncovered per arm, keep marked empty slots and retain 64 nonempty optimizer steps. T16 must exercise batch sizes 3, 4, 5, 6, 7 with the existing total-supervised-token denominator: step completes, loss matches an independent hand computation, gradients are finite, and scheduler advances exactly once. Any failure stops for repair; there is no duplicate fallback. More than 5 uncovered blocks training and requires a revised plan. Never fill with gold or incorrect traces; never retry without a bound.

Missing slots retain problem id and reason in batches.json and coverage summaries. Gate comparisons use the intersection covered by all k0/k2/k5 arms, alongside full-coverage summaries per arm. Training uses each arm's own covered slots to stay as close as possible to historical gold SFT's 512 problems. Report uncovered lists and their union/intersection. Flag pairwise differences of more than 5 uncovered ids before training approval (symmetric difference is the proposed formal definition). Eventual evaluation uses the same 150 held-out problems, but evaluation execution is outside this experiment's current scope.

### Termination and retries

Continue extension blocks while no stop token and response length is below cap, with safety limit 2 * ceil(cap/J). Preserve upstream within-block refinement ordering. Each generation attempt has exactly one end_state: finished, capped, block_budget_exhausted, engine_error. Proposed precedence when EOS also reaches cap: finished; stop-token accounting still needs qualification. Only finished, correctly graded traces can be eligible. Capped/exhausted attempts are ineligible and retryable: initial attempt plus at most three new-seed model-outcome retries. Engine errors are operational events: replay the same request/seed; after two failures stop and report rather than consume a model-outcome retry. The counter belongs to one problem’s generation or scoring request at one block, refinement step and model-retry index; initial failure plus one same-seed/runtime replay exhausts it.

The length-based loop intentionally extends cases that the supplied fixed-block algorithm would leave unfinished; a safety-exhausted trace can still remain unfinished. Every trace attempt, including model retries, has one terminal-state row. End-state counts sum to attempt records. Every problem per arm has exactly one final covered/uncovered disposition; a covered row references exactly one finished, correct attempt. An engine-error disposition is provisional while the run is blocked and cannot pass the coverage gate. Interruptions are events, not terminal model outcomes; partial blocks are discarded and redone on resume. Preserve operational request failures separately even if replay succeeds.

Empty generation without a real stop token, or missing/malformed scoring prompt_logprobs, is an engine/protocol error. Stop-only output is finished but incorrect and enters bounded model retries. Stop-only and early-stop proposals inside refinement remain legal score-based proposals; only the final trace is graded.

### Runtime freeze and replay

Choose settings at Stage 0.5; freeze before the 32-problem Stage 1 pilot. Record GPU type, driver, CUDA, Torch/vLLM versions, dtype, actual worker TF32 flags, max_num_seqs, max_num_batched_tokens, GPU memory utilization, prefix-cache setting and max_model_len; hash the block as runtime_hash. All sampling arms, model-outcome retries and resumes must match. A mismatch refuses resume. Changed settings require a new run_id and regeneration of all k arms; never mix runtime hashes in a comparison.

The 32-problem pilot qualification compares uninterrupted execution against kill-after-block-2/resume. Checkpoint at completed block boundaries only; discard partial scientific records and replay the partial block. Submit logical requests in fixed recorded order, by problem id then request type, while preserving extension/refinement dependencies. Clear prefix cache at each block boundary in qualification and production; within-block caching stays on. Record additional prefill work. Response IDs, scores, accept decisions, traces, steps and coverage match exactly; operational timing, PIDs, error and interruption events live separately.

If exact replay fails, stop before Stage 2. Requalify in order: disable prefix cache entirely; constrain scheduler/batch settings; enable available engine/framework deterministic modes. Each configuration change creates a new runtime_hash and qualification attempt; record throughput cost. Availability and effectiveness of these modes must be verified, not assumed. No numerical relaxation is authorized. Seeds alone do not guarantee reproducibility. Shared seed identities couple requests while paths coincide; they do not ensure identical outputs after arms diverge.

### Human eligibility review

Keep original flexible-v3 scores and generated text immutable. Separate confirmed human intended-answer labels may override eligibility for a finished trace when automatic extraction failed. They cannot edit the trace, supply a gold answer, or make unfinished text eligible. Record label provenance/revisions and distinguish agent suggestions from user-confirmed labels.

### Budget

Hard total ceiling USD25 including GPU and storage. Sub-caps: sampling/setup/tests/profile/pilot/retries/T1 baseline/gate USD10; training USD6; network volume and genuinely shared overhead USD9. Setup, idle, debugging and reruns count against the stage served, not the storage line.

Before every job, persist a conservative reservation in budget.json: GPU hourly rate times estimated hours times 1.5, plus setup-time cost. Include existing spent and outstanding reservations; launch only if the proposed sum stays below both the stage and total hard ceilings. Avoid counting setup twice if already in the runtime estimate. At 80% of any sub-cap or total, pause for approval before the next job even if headroom remains. Reconcile reservations to actual billed/estimated cost after each job; do not release an uncertain interrupted reservation without reconciliation. Store cumulative-cost snapshots in run_k{k}.json; shared charges count once in the global ledger.

Reserve storage for its expected lifetime, initially one month of approximately 60GB network volume (planning estimate USD4); extend if retained longer. Storage bills while Pods are stopped. Hourly rates and estimates in the supplied document remain unverified assumptions; obtain actual quotes before reserving a job. The USD15 top-up is account credit, not the budget.

After verified export to the network volume and a downloaded copy, the runner requests that its own Pod stop as its final cloud action. Persist stop_requested before the call; a surviving client/provider check records confirmed stopped separately. Do not claim the self-stopping process observed its own successful shutdown. Verify exported products with hashes/manifests; export failure cannot authorize deletion or unexported termination. Deleting network volumes or terminating an unexported Pod requires explicit user approval. Export-failure handling must preserve the budget ceiling and report before further paid work; never silently leave an unmonitored Pod running.

## Operational requirements already inherited from this repository

- Canonical storage under artifacts/plans, artifacts/runs and artifacts/reports; replace the proposal's outputs/sampling_sft root in the working implementation plan.
- Check existing products first; retain source hashes, resolved configuration, runtime, status, response IDs and logs. Resume only verified sealed units.
- Preserve automatic scores and original text; human labels and any selection override are separate versioned records.
- Required training resume state includes optimizer, scheduler and RNG state where applicable, not only adapters. Optional analysis gradients are distinct from required recovery state.
- Prompt/J changes after the pilot require a new immutable version and a complete pilot rerun. Do not mutate an existing production plan.

## Remaining technical verification

Pin vLLM version and compatible dependency environment; verify FP32/TF32 settings in the actual worker. Specify top-k and all decoding defaults. Verify stop-token return behavior, first-stop truncation and cap accounting without guessing a token from text. Define empty-generation handling, score slicing, context overflow, append/seal recovery and attempt ownership. Bound the throughput sweep before running it. Proposed numeric tolerances in the supplied document remain proposals, not approved acceptance gates.

## Evidence and limitations

A targeted local baseline check passed 5 tests (6 deselected). This is software evidence only, not sampler qualification or scientific results. Timing and prices in the supplied proposal are unverified estimates. No training, GPU sampling, cloud rental, or upload was performed during this review.

## Remaining decision frontier

- Fresh temperature-1 baseline on all 512 problems versus historical reuse; eligibility and scoring population.
- HF/vLLM score agreement gate and what numeric evidence is required before full generation.
- Pilot alarm actions, immutable prompt/J revisions and repeat qualification.
- Gate estimator, paired bootstrap, normalized-string leakage and explicit edge cases.
- Export/download receipt handoff and stop confirmation are engineering contracts to verify during implementation, not claims of executed cloud behavior.
