# Stage 0 production: behavior batch 16, internals batch 1

The primary ten-condition experiment uses the original 150 held-out behavior
questions and the original 150 GSM8K / 150 FineWeb fixed-context sequences.
No adapters or SVDs are rebuilt by this handoff.

Use the same live Colab GPU/session that produced the saved loader diagnostic
and capacity profile. Check out the committed release in the dedicated Colab
checkout, leaving Drive artifacts intact. Do not restart the kernel or change
hardware merely to load the new source. The batch workflow records the new
commit and verifies that hardware, software, session and original numerical
instrumentation still match. Freeze requires a clean committed checkout.

## Notebook order

After the existing setup and completed numerical diagnostic, use GPU sections
9B–9F, then12,12A and the existing collection cells. Primary batch16 mode skips
the legacy singleton freeze in10–11. Both notebooks set `USE_BATCH16` for the
primary path; conditional intruder workflows retain their separate names.
CPU report restoration selects the corresponding batch16 findings plan.

1. **9B, CPU:** `stage0-batch-prepare` validates the saved diagnostic, capacity
   receipt, tokenizer, catalogue and actual adapter files. It does not scan
   unused SVD/random-basis payloads or load a model.
2. **9C, GPU:** reconcile prior profiles using an actual cumulative unit reading,
   then reserve a bounded behavior-only agreement run. Full16 and tail6 shapes
   use the original fixed16 questions. All10 variants receive answer-position
   FP32 logit and cached-decode checks against singleton inference. V1 andV5
   receive greedy replay at both sizes:44 generated answers. Generation is left
   padded. Exact token IDs **and stop reason** pass. Any divergence saves the
   first differing position and common-prefix top-two gap. A near tie requires
   that gap below the measured common-prefix error, itself within the unchanged
   original numerical allowance. Other divergences stop. Counts are retained.
3. **9D–9E, CPU:** inspect saved evidence, approve its original3E numerical gate,
   reconcile agreement usage, then display the concrete production proposal.
   This includes the code commit, variants, decoding, batches, forecast and unit
   allowance. Send that proposal to the operator for approval.
4. **9F, CPU:** freeze only after that approval. Source/runtime/receipt changes
   are rejected. A separate bounded production reservation remains necessary.
5. **12/12A, CPU:** prepare100 behavioral durable units (1,500 answers; tail6),
   the singleton internals plan, and the full comparison registration.
6. Run/resume the existing behavioral collector, then separately budget and
   run/resume singleton KL/write collection. Confirmed receipts are reused;
   altered source/runtime/payloads stop. A changed runtime requires new evidence.

The public commands are also directly runnable with `python -m pilot_eval`.
`stage0-batch-{prepare,run,verify,status,approve,freeze}` exposes the notebook's
saved workflow. No notebook kernel state is the durable record.

## Precision and timing evidence

Decoder and actual model parameters remain FP32 with TF32/autocast off.
Fixed-context KL projects final hidden states through a separate FP64 copy of
the output head, then computes full-vocabulary log-softmax and KL in FP64.
Production chunks32 **token positions**; the vocabulary is never split.
The unchanged32-versus16 agreement gate is atol=rtol=1e-5.

On the existing L4, FP32 head chunk disagreement peaked at4.51465e-5 nats;
FP64 reduced it to3.55271e-15 on the identical hidden states. The primary
instrument now retains chunk-error coordinates and reports observed resolution
beside the smallest between-variant token-mean KL differences. This empirical
floor is not a statistical interval or a universal error bound.

A V0a singleton profile measured eight length-stratified GSM8K sequences and
eight FineWeb sequences, cold and warm. Extrapolating to all10 variants gives
4,967 seconds (1.38 GPU-hours,2.12 units at1.54/hour), including loading,
final integrity and allowances for controls/diagnostics. Source-verification
and Drive-write overhead were not profiled; other variants were not separately
timed. This is a forecast, not a completion guarantee. Behavioral batch16
throughput was87.55 generated tokens/second on repeated copies of two questions;
production length tails remain uncertain until the actual run.

## Earlier verification issues

PEFT0.18's substring-based adapter export selected both `V4` and `V4-input`.
The loader now selects exact final adapter-name segments and compares every
actual FP32 factor with its saved tensor. This repairs verification without
reconstructing adapters or weakening tensor equality.

`reconstruction_max_fraction` is the maximum absolute reconstruction difference
divided by `1e-6 + 1e-5 * abs(expected)`, for coefficients/direct branches.
Values0.20–0.52 use20–52% of the allowed numerical budget; they are not raw
relative errors or fractions of activation energy. Values above1 stop.

The shared cumulative ceiling remains20 Colab units. Never substitute an
estimated reading for the operator's actual cumulative usage or auto-renew an
exhausted reservation.
