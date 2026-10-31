# Bounded cache-on/cache-off scoring diagnostic

The first real A40 T3 score comparison failed the predeclared per-token0.01 limit.
The20 fixture inputs, original vLLM score run and HF oracle run remain sealed.
This diagnostic reuses and re-verifies those exact fixtures and both original
receipts. It runs two new, separately recorded vLLM jobs (cache on, then off),
with the same model, scorer inputs, GPU settings, batching and token positions.
The HF FP32 scores are the saved original oracle. Each new job retains its own
runtime/configuration hash, GPU worker evidence, raw token scores, completion
seal and job lifecycle records. A third comparison checks fresh cache-on against
the original vLLM scores to reveal failure to reproduce the starting point.

Run after environment and existing artifacts are restored, using a fresh name:

```bash
python -m pilot_eval.sampling_sft_cache_diagnostic \
  --output-root /workspace/artifacts --name a40-cache-diagnostic-v1 \
  --volume-id cmuptiucb000007l5dnohe861 --rate 0.60 --reviewer Prasad
```

The command launches two bounded, reviewed GPU jobs sequentially, printing their
logs, and saves the report under the source run's diagnostics/name directory.
It uses the existing per-request waiver; incomplete historical cost records are
not represented as settled. The per-job timeout is22.5 minutes; stopping and
retrying a failed launch requires a new name. If a previous job completed, rerun
with the same name verifies and reuses it. Back up the complete diagnostic tree,
plans, and job records to the private dataset and an independently verified local
copy before stopping the Pod. This command does not stop or delete resources.

Interpret the comparison as diagnostic, not causal proof. Cache-off agreement
supports a cache-related lead; similar on/off discrepancies point elsewhere;
non-reproduction of the original cache-on run calls for runtime/batching review.
The existing scientific gates, tolerance numbers and failed receipt are never
modified or overridden. A cached-off production candidate must be requalified.

## Batch-size diagnostic

For `--mode singleton`, the diagnostic runs one new vLLM scoring job against the
same sealed20 fixtures, changing only `max_num_seqs` from the saved runtime to1.
It retains the original cache setting, weights, token IDs, limits and settings.
The report compares original batched vLLM versus singleton vLLM, and each versus
the saved HF reference. This tests scheduler sequence concurrency, not all causes
of numerical differences: chunked prefill, kernel shapes and cache behavior can
also depend on the scheduler. Run under a new diagnostic name. Each result and
runtime identity is preserved, and the original T3 gate remains unchanged.

## Triton IEEE diagnostic

Under a fresh name, `--mode ieee` uses the original20 fixtures and scores once
with `TRITON_F32_DEFAULT=ieee`; it compares raw token scores against both saved
vLLM and HF results using the original0.01 token and0.001 mean limits. It
fingerprints the installed vLLM attention source and Triton precision mechanism,
then requires the actual inference worker to report the same source hashes,
environment, effective Triton knob and isolated cache. The child sets these
before importing Triton/vLLM. A new name supplies a fresh Triton and vLLM cache;
failed or interrupted attempts are preserved and require a new name. No package
files or existing runtime receipts are changed. For example:

```bash
python -m pilot_eval.sampling_sft_cache_diagnostic \
  --output-root /workspace/artifacts --name a40-ieee-diagnostic-v1 --mode ieee \
  --volume-id cmuptiucb000007l5dnohe861 --rate 0.60 --reviewer Prasad
```

The preflight stops if the installed source mechanism cannot be recognized.
A worker knob and source match do not prove that this particular attention
backend executed or that every kernel uses the default; inspect backend logs and
compiled kernels if the result needs a mechanistic interpretation. IEEE can be
slower and the existing22.5-minute job timeout still applies. A passing score
comparison does not qualify production or reverse the original failed gate.
The newer upstream [vLLM issue](https://github.com/vllm-project/vllm/issues/62582)
is motivation, not evidence that this particular version and backend were affected.
