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

Training must not start until coverage passes. With zero uncovered, the coverage gate passes (other gates and explicit training approval still apply). With 1–5 uncovered per arm, the user proposes explicitly marked empty slots, conditional on T16 establishing the gold trainer's token-weighted reduction for smaller batches. A duplicate covered problem from the same step was proposed as fallback if T16 fails. More than 5 uncovered blocks training and requires inspection/revised plan. Never fill with gold or incorrect traces; never retry without a bound.

Pending: several missing problems can cluster in one step, so the batch can contain 3–7 rather than exactly 7 problems. Also, a failing correctness test must block implementation/execution under repository rules; it cannot automatically trigger duplication to bypass the failure. Resolve these details before freezing this branch.

### Termination and retries

Continue extension blocks while no stop token and response length is below cap, with safety limit 2 * ceil(cap/J). Preserve upstream within-block refinement ordering. Each generation attempt has exactly one end_state: finished, capped, block_budget_exhausted, engine_error. Proposed precedence when EOS also reaches cap: finished; stop-token accounting still needs qualification. Only finished, correctly graded traces can be eligible. Capped/exhausted attempts are ineligible and retryable: initial attempt plus at most three new-seed model-outcome retries. Engine errors are operational events: replay the same request/seed; after two failures stop and report rather than consume a model-outcome retry. The exact two-failure counter scope remains to be specified.

The length-based loop intentionally extends cases that the supplied fixed-block algorithm would leave unfinished; a safety-exhausted trace can still remain unfinished. Count conservation must be defined per attempt as well as per-problem final disposition; retries mean total attempt records can exceed 512. Empty-response classification remains open.

### Runtime freeze and replay

Choose settings at Stage 0.5; freeze before the 32-problem Stage 1 pilot. Record GPU type, driver, CUDA, Torch/vLLM versions, dtype, actual worker TF32 flags, max_num_seqs, max_num_batched_tokens, GPU memory utilization, prefix-cache setting and max_model_len; hash the block as runtime_hash. All sampling arms, model-outcome retries and resumes must match. A mismatch refuses resume. Changed settings require a new run_id and regeneration of all k arms; never mix runtime hashes in a comparison.

The pilot qualification compares uninterrupted execution with kill-after-block-2/resume and requires identical response IDs and scientific step records. Timing, PID and attempt bookkeeping are not deterministic scientific records. T8 reports batch dependence and does not authorize retuning. Seeds are necessary inputs, not a guarantee of exact replay; the qualification must establish replay on the pinned configuration. Shared seed identities couple requests only while their paths coincide; they do not ensure identical randomness or outputs after arms diverge.

### Budget

Hard total ceiling: USD25, including GPU and storage. Sub-caps: sampling/setup/tests/profile/pilot/retries/T1 baseline/gate USD10; training USD6; network volume plus rerun/debug slack USD9. Stop and report at 80% of any sub-cap or total before the next job; do not launch a job projected to cross a cap. The interpretation of the 80% approval checkpoint and allocation of rerun costs remains open.

Keep cumulative spend in budget.json and each arm run summary; shared costs must be counted once. Stop Pods when their stage ends. Storage can continue billing after a Pod stops, so reservation/lifetime and durable export/deletion policy need resolution. The supplied L40S/L4 prices and time estimates are planning assumptions, not verified quotes or authorized rentals. The USD15 runbook top-up is account credit, not the budget.

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

## Round 2 proposals — awaiting answers

- T16 tests 3–7 covered problems per accumulation window with the existing token denominator; a test failure blocks rather than selects duplication.
- One three-arm common-coverage gate cohort, plus arm-specific coverage summaries; retain each arm’s own training coverage if approved.
- Empty output without a stop is a protocol error; stop-only is a finished, grader-incorrect model outcome. Two consecutive failures of the same logical request trigger a stop. Attempt counts and final per-problem dispositions are separate tables.
- Freeze request grouping/order and persist logical scheduler state at block boundaries. Proposed prefix-cache reset at each boundary applies equally in qualification and production; qualify the cost. Exact scientific replay is required, excluding operational timestamps and process metadata. This is a proposal, not an established capability of vLLM.
- Human eligibility overrides are separate from immutable automatic scores and cannot alter generated text or completion state.
- Reserve job costs, include ongoing storage and idle/setup charges, pause at 80% for review, and count reruns against their real stage. Persistent-volume deletion remains separately approved.

Further technical frontier after these answers: baseline cohort/provenance, numeric score agreement gate, pilot alarm actions, and fully specified aggregation/uncertainty calculations.
