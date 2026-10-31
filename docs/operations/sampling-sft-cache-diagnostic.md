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
