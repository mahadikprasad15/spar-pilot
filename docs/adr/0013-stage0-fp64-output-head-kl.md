# Stage 0: FP64 output-head projection for fixed-context KL

Status: approved by the user on 2026-10-29; measured on the existing L4 session.

The original instrument projected FP32 final hidden states through the FP32
output head, then upcast logits for FP64 log-softmax and KL. Changing the number
of token contexts in each projection from 32 to 16 changed FP32 GEMM rounding.
The vocabulary is never split: each projected row contains all 151,936 logits.
TF32 and autocast were verified off. Identical chunk sizes repeated exactly.

On `gsm8k:test:1104`, maximum absolute chunk disagreement was
4.5146474470181275e-5 nats (KL 1.2553773077689416 versus 1.2554224542434118).
Projecting the identical hidden states with FP64 weights reduced maximum chunk
disagreement to 3.552713678800501e-15 nats. This isolates projection rounding;
neither token alignment nor the softmax normalizer changes between these probes.

For new Stage 0 fixed-context products, upcast hidden states and a separate
cached copy of output-head weights/bias before projection. Projection,
log-softmax, and KL use FP64. Decoder parameters, computation, stored hidden
states, and the actual model output-head parameters stay FP32. The additional
FP64 head copy is resident memory, separate from the bounded chunk workspace,
and is released when the measurement engine closes.

Use fixed production context chunk 32; chunk 16 remains a qualification check.
The existing agreement gate remains atol=rtol=1e-5. Changed implementation and
precision create a new artifact identity; never relabel old FP32 evidence.

The observed disagreement is an empirical numerical-resolution diagnostic,
not a universal error bound or a statistical confidence interval. Report it
alongside the smallest between-variant KL differences once those measurements
exist. A single variant's KL relative to the untuned model is not a substitute
for between-variant differences. The timing probe has not produced that matrix.

Canonical Drive evidence, relative to `artifacts/`:
- `runs/stage0/stage0-geometry-v1-drift-chunk-diagnosis-v1/profile/`
- `runs/stage0/stage0-geometry-v1-drift-fp64-timing-v1/profile/`

Timing is performance evidence only. The profile covers eight GSM8K inputs
spanning 126–551 tokens and eight 154-token FineWeb inputs, under V0a, with
cold/warm baseline-cache measurements and unchanged integrity checks. Projection
to ten variants is an estimate; source-verification and Drive-write overhead
remain unmeasured. No batch-schedule decision or production approval is implied.
