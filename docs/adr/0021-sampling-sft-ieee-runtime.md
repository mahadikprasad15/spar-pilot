# 0021: IEEE Triton runtime and fresh qualification

Date: 2026-10-31

The fixed20 A40 score diagnostic compared5995 tokens. The original vLLM vs HF
maximum absolute error was0.02118158, with34 tokens above0.01. The IEEE
Triton diagnostic reported a maximum error of0.00007104873657226562, zero
above0.01, and a maximum per-trace mean error of0.0000016743792136120206.
The worker reported the intended precision knob and source fingerprints. This
supports precision as a strong explanation for the observed score discrepancy;
it does not identify which compiled kernels actually executed.

New Sampling SFT runtime settings must explicitly specify IEEE Triton FP32
input precision, isolated per-engine compiler caches, and installed vLLM/Triton
source fingerprints. These controls are part of the runtime hash. The engine
sets the environment before importing Triton or vLLM, checks the effective knob,
source fingerprints, and cache paths in the actual inference worker, and saves
that evidence. The HF oracle also records and checks its local runtime. Existing
PyTorch FP32 and TF32-off settings remain mandatory. These controls apply to
qualification, pilot, generation and the HF gate scorer. The separate HF SFT
training backend is unchanged and has its own precision settings and evidence.

Create new settings and a fresh qualification name, recapture20 fixtures under
the new runtime, and repeat all qualification and replay stages. Preserve old
artifacts and verify them with their original pinned release. Do not reuse their
runtime identity or label the diagnostic as production qualification. The score
and replay gates, tolerances, source audit, model snapshot and budget controls
are unchanged. Model weights do not need to be downloaded again. Source or
runtime mismatch, unknown precision mechanism or premature Triton import blocks
the run. Record and back up the per-engine compiler-cache tree with the run.
