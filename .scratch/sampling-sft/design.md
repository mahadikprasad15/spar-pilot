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

## Decisions required before implementation

1. Claim: initially compare the resulting data-construction recipes; do not claim an isolated effect of on-policyness without further controls.
2. Coverage: choose a policy when bounded retries leave missing answers. Keeping empty slots cannot be described as the unchanged 512-exposure trainer. Recommended: stop at the training gate and report coverage; resolve a revised cohort explicitly if necessary.
3. Arm selection: a confidence interval containing zero is not evidence of equivalence. Recommended: remove automatic arm merging; use the gate sheet and budget to choose training arms explicitly.
4. Termination: distinguish EOS, token cap, block-budget exhaustion, empty response and engine failure. Recommended: unfinished block-budget exhaustion is ineligible and retryable under the same bounded attempt policy as other unfinished traces; engine failures remain operational failures, not wrong answers.
5. Replay: per-request seeds do not guarantee batch independence. Recommended: qualify a fixed pinned engine/runtime/scheduling configuration for restart/resume; changes create a new attempt identity. Report cross-batch differences rather than promising identical data across arbitrary hardware/batching.
6. Initial scope and budget: recommend k=0/2/5 only; defer k=10 and k=5+5. Separate approval of generation/gate work from later training; dollar and Colab-unit ceilings remain open.

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
