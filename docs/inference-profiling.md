# One-command inference batch profiler

This diagnostic runs on any single CUDA GPU. It does not require Pilot 2
training/preflight or a particular GPU name. Install the repo and its pinned
libraries first; do not run another model job concurrently.

From the repository checkout:

```bash
python scripts/profile_inference.py --config /content/drive/MyDrive/SPAR/spar-pilot/artifacts/plans/baseline-batch8-v1/gsm8k-0shot.config.json --output-root /content/drive/MyDrive/SPAR/spar-pilot/artifacts --name fp32-batch-sweep-v1
```

Default: FP32, batches 1/2/4/8, eight fixed questions spanning prompt lengths,
one measured repetition, short untimed kernel warmup per batch shape, and the
original full generation cap. The same prompts are used for every batch size.
It measures synchronized elapsed generation time, items/sec, output tokens/sec,
peak allocated/reserved memory and extrapolated 150-item/six-evaluation costs.
It saves responses, runtime/config, per-candidate evidence and status beneath
`artifacts/runs/diagnostics/inference-profile/<name>/`. It changes no source plan.

Only CUDA out-of-memory errors are treated as candidate failures. Temporaries
and allocator cache are released after an OOM; later candidates are attempted.
Other errors stop and are logged. Rerunning the same command verifies and reuses
finished candidates (including recorded OOMs); unfinished candidates repeat.
Use a new name for changed hardware, precision, sample size or repetition count.
Use `--sample-size 16 --repeats 2` under a new name for a stronger confirmation.

The fastest successful measured batch is provisional. Inspect memory headroom
and differences against the smallest successful batch before selecting it.
Eight prompts cannot guarantee full-cohort memory safety: generated length,
concurrent jobs and adapter/checkpoint behavior can change resource use.
Costs exclude model loading, installation and training. A single repetition
provides a quick screen, not a confidence interval or an industrial load test.

After selecting a batch, freeze it in a new named L4 scientific plan:

```bash
python -m pilot_eval sft-prepare --source-config SOURCE_CONFIG --name pilot2-sft-fp32-l4-batch4-v1 --hardware L4 --evaluation-batch-size 4 --output-root ARTIFACT_ROOT
```

L4 plans now accept evaluation batches 1/2/4/8. Existing immutable plans retain
their previous settings. The notebook has `EVALUATION_BATCH_SIZE` and `PLAN_NAME`
in section 1; choose those together before preparation. Do not mix earlier
batch-2 responses with new batch-4 outputs.

## Benchmarking practice

PyTorch documents warmup and CUDA synchronization as essentials of valid GPU
timing: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
NVIDIA GenAI-Perf distinguishes output-token throughput, request throughput,
time to first token and inter-token latency:
https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2630/user-guide/docs/perf_benchmark/genai-perf-README.html

For this offline research job, elapsed cohort time and memory are the first
useful measurements. A production serving benchmark also sweeps concurrency,
reports tail latency, repeats trials and uses representative input/output
length distributions. This script is a small screening benchmark, not a
replacement for production load testing.
