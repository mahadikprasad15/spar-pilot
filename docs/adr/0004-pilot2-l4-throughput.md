# Pilot 2 L4 hardware and evaluation throughput variant

Approved in conversation: user can use an L4 and requests an explicit variant and
speed benchmark before continuing Pilot 2. Preserve the T4 plan and evidence.

The new named plan records hardware L4 and evaluation batch 2 (batch 1 is also
supported for a separately named L4 plan). FP32, training microbatch 1 with
accumulation 8, optimizer, data selection, prompts, scorers and generation caps
are unchanged. Use a fresh L4 preflight, baseline and checkpoint evaluations;
do not merge partial T4 responses into the L4 scientific run. Existing plans
without these fields retain the original T4/batch-1 meaning.

Before baseline, benchmark batches 1 and 2 on eight fixed questions selected
across prompt lengths from the same 150 held-out items. Warm up outside timed
measurement; synchronize CUDA before/after timing, reset peak allocation, and
save full generated outputs, token counts, stop reasons, timings, runtime and
input identity. Use the full 1024-token cap rather than a shortened proxy.
Report items/sec, tokens/sec, estimated 150-item and six-cell times, and output
differences across batches. These estimates exclude setup/training and are
conditional on this small sample; checkpoint lengths can differ.

The benchmark uses untuned weights, performs no optimizer updates and does not
write scientific evaluation responses. Same-config completed benchmark evidence
is verified and reused. OOM records evidence and stops, without automatic batch
or precision fallback. Batch output differences are reported, not hidden; the
chosen batch remains consistent for baseline and every adapter checkpoint.

Runtime checks continue to require the recorded GPU, library versions and code
commit. The L4 experiment is not locally GPU-verified; CPU tests use fakes at
the benchmark boundary and the existing tiny real model for training integration.

## Broader standalone profiling (approved follow-up)

The user requested a small batch-1/2/4/8 tester with OOM continuation and advice
on expert practice. `scripts/profile_inference.py` is independent of SFT plans
and GPU names; it saves diagnostics separately, preserves source inputs, and
reuses completed candidates. It recommends the fastest measured successful
batch without silently changing scientific settings. A new L4 plan may freeze
1/2/4/8 after reviewing memory headroom and output differences. Existing T4 and
L4 scientific runs retain their original settings. This broader diagnostic
handles OOM per candidate; the earlier fixed 1/2 SFT benchmark and scientific
training/evaluation still stop on OOM.
