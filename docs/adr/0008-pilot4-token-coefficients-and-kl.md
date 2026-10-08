# Ticket 8: supplemental fixed-token coefficients and KL

Status: accepted scientific conventions; execution requires supplemental validation/profile.

## October 8 user approval

Save raw signed `c = A x` from the adapted pass's actual module input. Bind
coefficients to factor hashes and `alpha/r`; reconstruct the direct branch as
`(alpha/r) B c`. Coefficients are factorization-dependent, including sign.
Checkpoint zero can have nonzero coefficients because A is initialized randomly;
the B factor and direct contribution must be zero.

For KL(tuned || untuned), next target token t determines view membership;
paired next-token distributions are taken at context t-1. Exclude target zero
(no preceding supplied context), padding, and predictions beyond the saved
sequence. Preserve content masks exactly; a prompt-format context predicting
the first solution token can belong to the solution prediction view.

Retain per-token values and report both token-weighted and equal-example means
for all existing question/solution/user views. No generation, SFT retraining,
retokenization or overwrite of old evidence. Keep both arms' steps 0/8/16/32/64.
Use the approved public saved-workflow seam with independent tiny-model/math
oracles, plus separately saved target-GPU checks.

## Engineering approach

Build an additive separately named product from verified existing activation
execution configs. Bind source hashes, runtime, ordered batches and new
resource settings. Use the existing base/instrument integrity gates and atomic
locks/writers/seals. New head/coefficient checks and profiling are supplemental;
matching FP32 alone does not authorize inheriting their validation.

FP32 model/factors/hidden states and output-head logits; FP64 log-softmax, KL
accumulation and aggregates. Keep raw coefficients FP32, KL FP64, metadata in
JSON, arrays in NPZ shards. Chunk context positions **before** applying the
output head; retain no full-sequence vocabulary tensor. Configurable context
chunk and declared temporary workspace budget are frozen and checked before
allocation, then profiled on the actual GPU. This budget bounds output-head
workspace, not total decoder/model GPU use; total peak is reported separately.
Compare allowed chunks/batches under existing numerical-policy thresholds,
and do not widen those thresholds after a failed gate. Supplemental preflight
freezes reviewed evidence before production. Supplemental comparisons condition
on supplied contexts and do not estimate unconditional generated-policy KL.
