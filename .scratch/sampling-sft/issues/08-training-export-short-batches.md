# 08: Export the gold-matched training dataset and verify short batches

**What to build:** The user obtains a pretokenized training bundle using gold's exact per-step order and marked omissions, plus a readiness receipt proving checkpoint0 identity and T16 for batch sizes3–7. This slice independently verifies the training handoff without training a full arm or waiting for statistical-report presentation.

**Blocked by:** 01, 06.

**Status:** claimed

**Type:** task

## Acceptance criteria

- [x] Export a stored-ID training bundle in gold’s exact per-step order with each arm’s own covered slots; every omission retains problem ID/reason.
- [x] Verify saved tokenizer/prompt/masks and exactly one supervised end-turn; do not silently decode/re-encode or truncate targets.
- [x] T15 compares checkpoint0 with gold bit for bit, including A0 and zero B; seed equality alone cannot pass.
- [ ] T16 exercises batches3/4/5/6/7 with existing total-supervised-token loss denominator: step completes, independent hand loss agrees, gradients are finite and scheduler advances once.
- [x] Preserve64 nonempty steps and verified optimizer/configuration contracts; a failing test blocks for repair and never triggers duplication.
- [x] Publish restorable training-readiness evidence independent of gate presentation. No full training or optimizer path changes are implied by this handoff.
- [x] Use tiny offline CPU models/math through the public workflow, not mocked loss arithmetic; missing/broken local Torch is a reported missing check, not a pass.

## Testing and evidence

Tiny real CPU-model loss/gradient/scheduler checks, mask/end-turn and stored-ID checks, exact A0 comparison, all short-batch sizes and required metadata. Failure blocks readiness; no duplicate fallback. Full training still requires ticket07's approval product.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Implementation and remaining execution

Implemented `sampling-sft-training-data-{prepare,export,verify,check,status}`.
Exports retained IDs/masks and original 64-step slots for k0/k2/k5 with explicit
missing identities/reasons. T15 compares the original gold payload and all 196
rank-one A0/zero-B pairs bit for bit. Immutable plans/exports bind reviewed
coverage, source hashes and implementation. Changed data or initialization is
rejected; completed exports are restored without rewriting.

The separate offline CPU worker provides `full` and `kernel` modes. Full mode
constructs a tiny 28-layer Qwen with all 196 rank-one modules and runs the pinned
SFTTrainer for batch sizes 3–7 with accumulation 8. Acceptance is independently
recomputed from the saved FP64 oracle loss, token denominators, gradients and
optimizer/scheduler counters. Request identity, measurements and published
readiness are sealed and verified. Kernel-only or missing checks cannot mark
training ready. No tolerance changes, duplicate fallback or package installation
occur automatically. User approved loss atol=1e-6 and rtol=1e-6 on 2026-10-29.

**Remaining execution:** full T16 requires an existing pinned trainer environment;
it has not passed locally. A working local Torch interpreter supports the loss
kernel check but lacks Transformers/TRL/PEFT. The default interpreter has a
broken Torch import. Both limitations are recorded as missing checks rather
than qualification. No complete arm, model download, paid job or push is part of
this ticket. Real initialized model identity and L4 runtime are rechecked by
Ticket 09. Optional notebook sections 15–16 and the training-data runbook expose
the same public commands.

Validation: final focused run passed 7 tests in 173.63 seconds, with the existing
working Torch interpreter used for real loss/autograd/optimizer checks. The
largest kernel loss error was 8.056225420105534e-8. Tests cover exact IDs/masks/
slots, immutable initialization, missing full-trainer status, all five short
batch sizes, altered readiness rejection, independent acceptance reconstruction
and disabled/syntax-valid optional notebook cells. The relocated canonical
synthetic workflow passed fresh-process status verification. Baseline pytest
was interrupted after 49 passed, 11 skipped, 4 warnings in 286.66 seconds; this
is not a full-suite pass. Raw evidence: artifacts/validation/sampling-sft-ticket08/.
The remaining unchecked acceptance criterion is actual full pinned-trainer T16
execution, not a claim that the kernel-only result satisfies it.
