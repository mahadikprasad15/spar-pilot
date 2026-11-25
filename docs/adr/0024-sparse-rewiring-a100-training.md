# 0024: Sparse rewiring arms train on one A100-80GB

Date: 2026-11-23

The researcher decided that L4 is not required for training the sparse rewiring
arms (`sparse-gold-sft-r1` and `sparse-gold-sft-r0`). Both arms train, in
sequence, on one A100-80GB. For these two arms this supersedes ADR 0018's rule
that "Training still requires L4 to match gold SFT". The rule is unchanged for
gold SFT, Sampling SFT and any rerun of them.

## Why

- **Rank 0's memory.** Rank 0 adds 770.7 M FP32 read parameters
  (28 layers × gate and up × 8,960 × 1,536). With their gradients and two Adam
  moments that is about 12 GB more than gold SFT needs, which would be tight on
  an L4's 24 GB. Rank 1 adds only 0.84 M parameters, but runs on the same GPU so
  the rank comparison does not mix runtimes.
- **The existing Pod runbook.** The sparse arms reuse the C-attribution chain
  and its Pod runbook (`.scratch/c-attribution-pilot/pod-runbook.md`), which is
  written for one A100-SXM4-80GB with the `sl-0` Global Volume and checks the
  device before work starts. Writing an L4 variant would repeat Pod failures
  that runbook already fixes.
- **These are new arms, not gold reruns.** They read gold's frozen plan and
  batches, but nothing about them is a reproduction of gold's trained adapter,
  so matching gold's GPU is not needed to make them valid.

## Consequences

- Behaviour comparisons with gold SFT or Sampling SFT (both L4) carry the
  runtime difference and say so. Neuron-set comparisons, between the two ranks
  or with C attribution and adapter attribution, are not affected.
- The rest of the runtime contract is kept: FP32, TF32 and cuDNN TF32 off,
  `highest` matmul precision, pinned engine versions, and the GPU identity in
  the runtime fingerprint, verified in the worker.
- Hardware is still chosen explicitly and not qualified by analogy (ADR 0018).
  The A100 is qualified for these arms only by the GPU qualification run
  (ticket 15), which measures peak memory and throughput for both ranks. CPU
  fixtures never establish GPU qualification.
- This does not change which GPU any earlier plan, run or result used.
