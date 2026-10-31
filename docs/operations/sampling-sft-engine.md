# Sampling SFT: bounded engine checks

Ticket 03 implements the instrument, not a qualified experiment result. CPU
tests inject tiny engines; they do not establish L40S throughput, numerical
agreement or exact replay. Use the CLI from a repository checkout. The optional
notebook only calls these commands.

## What runs

`bounded` runs one attempt on 1–64 predeclared problems, at k0, k2 or k5.
Problems are ordinals in the saved, ordered 512-row gold training corpus; the
original dataset IDs are retained in the plan. No dataset or model is downloaded.

Each active problem extends by up to 64 tokens under the gold-guided proposer.
The scorer uses the original training prompt, with no gold in its context.
Each of k refinements cuts anywhere in the retained response, regrows the
removed suffix and accepts only a strictly higher mean response log probability.
The partial response intentionally appears inside the proposer message and as
assistant-prefill IDs. Completion is checked after all refinements in a block;
an accepted EOS proposal can be overwritten. The cap is 1,024 tokens, with a
32-block safety bound. A stop token at the cap counts as finished.

Every generation/scoring batch has an ordered request record and deterministic
request seeds. Empty generation, missing probabilities, nonfinite values and
token/position mismatches replay the identical request once; a second failure
stops the run. These are engine failures, not model-outcome retries. This stage
does not implement the later initial-plus-three coverage retry policy.

## Prepare on CPU

Copy an explicit settings JSON, resolve its environment fields from preflight,
then prepare:

```sh
python -m pilot_eval sampling-sft-instrument-prepare --config plans/sampling-sft-v1/sampling-sft.plan.json --settings artifacts/plans/sampling-sft-v1/engine-settings.json --name engine-bounded-v1 --output-root artifacts
```

Settings schema:

```json
{
  "mode": "bounded",
  "k": 0,
  "problem_ids": [0, 1],
  "local_model": "inputs/models/MODEL_REVISION_SHA",
  "local_tokenizer": "inputs/tokenizers/TOKENIZER_REVISION_SHA",
  "runtime": {
    "vllm_version": "0.11.1",
    "torch_version": "2.9.0",
    "cuda": "RECORD_ACTUAL_VERSION",
    "driver": "RECORD_ACTUAL_VERSION",
    "gpu": "NVIDIA L40S",
    "dtype": "float32",
    "tf32": false,
    "worker_numeric_flags": {
      "matmul_tf32": false,
      "cudnn_tf32": false,
      "matmul_precision": "highest"
    },
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.85,
    "prefix_caching": true,
    "max_model_len": 4096,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "actual_stop_ids": [151645, 151643]
  }
}
```

These scheduler/context values are candidate examples, not measured choices.
No unresolved field inherits an engine default. Local snapshots must be staged
and independently verified against their audited Hub revisions before the first
paid job; the workflow freezes every local file hash. Their directory names must
be the immutable revision SHAs. New or changed snapshot files invalidate a plan.
The tokenizer's chat template, actual stop IDs and original scorer IDs are checked
again before real sampling. Context overflow stops instead of truncating.

The versioned adapter targets vLLM 0.11.1 and Transformers 4.57.6. This vLLM
release [pins Torch 2.9.0](https://github.com/vllm-project/vllm/blob/v0.11.1/requirements/cuda.txt).
Its [token input, prompt-logprob and worker APIs](https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/entrypoints/llm.py)
are used directly. The code makes no automatic installation or precision
fallback. A different engine release requires a reviewed adapter and new plan.
TF32 flags and FP32 weight dtypes are checked in the actual inference worker.

Worker inspection and peak-memory measurements use the importable
`pilot_eval.sampling_sft_engine.SamplingSFTWorkerExtension`, registered through
`worker_extension_cls`. Each `collective_rpc` carries a method name and returns
small plain-data results. Do not replace these calls with `apply_model(function)`:
vLLM V1's default serializer rejects function payloads. The first L40S smoke
loaded the model but failed at that boundary before producing responses.
`VLLM_ALLOW_INSECURE_SERIALIZATION` is not enabled as a workaround.

After this repair, prepare a new instrument/job identity: the code fingerprint
changes, and the failed attempt remains separate. CPU regression tests cover
extension registration, named-message transport, and dispatch to the resident
worker model. They do not replace a new real four-problem smoke. Reconcile the
previous job against the authoritative budget ledger on the network volume
before reserving another paid attempt.

## Run only inside an approved budgeted job

Use Ticket 02's request → reservation/approval → execution → export → stop
workflow. The request's command is an argv list such as:

```json
["python", "-m", "pilot_eval", "sampling-sft-instrument-run",
 "--config", "plans/engine-bounded-v1/sampling-sft.instrument.json",
 "--job", "runs/sampling-sft/sampling-sft-v1/jobs/engine-check/job.plan.json",
 "--output-root", "artifacts"]
```

Config and job paths are relative to the artifact root, or absolute. The output
root and settings-file paths are relative to the repository checkout, or absolute.
Include its `complete.json`, `results.json`, `responses.jsonl`, `requests.jsonl`,
`env.json` and each required block/timing file in the declared export outputs.
The generic lifecycle exports explicitly declared files, not undeclared folders.
Inspect the bounded plan before approving the output inventory and budget.
Diagnostics can be exported after failure without certifying successful work.

Direct real `instrument-run` without a matching active, reserved job refuses
before importing Torch or loading a model. A completed compatible instrument is
reused. Interrupted bounded attempts are preserved and require a new name;
Ticket 04 supplies and qualifies the pilot's block-boundary resume path.

The artifact directory is
`artifacts/runs/sampling-sft/<source-plan>/instrument/<name>/`:

| File | Meaning |
|---|---|
| `plan-seal.json` | Plan/code/source identity |
| `meta/status.json` | Explicit prepared/running/completed/failed status |
| `requests.jsonl` | Raw ordered requests, proposals, scores, decisions, errors |
| `blocks/*.json` | Retained states and per-problem RNG state after each completed block |
| `responses.jsonl` | Final retained IDs, text, score and terminal state |
| `env.json` | Actual worker dtype, flags, hardware and versions |
| `timings.jsonl` | Operational timing, separate from scientific equality |
| `results.json` | Structured bounded result, explicitly not production-ready |
| `failure.json` | Failure diagnostics when applicable |
| `complete.json` | Hashes sealing the completed payload |

## Score correctness: T3

Prepare two plans with modes `scores` and `oracle` and the same explicit
`fixtures` JSON path. Both plans bind the same sampling runtime. The oracle
loads the same local base snapshot separately with HF FP32/eager attention; it
computes a full-vocabulary log-softmax at each response-prediction position,
with the causal shift and actual stop token included.

Freeze exactly 20 distinct traces before either scorer runs. Each row contains
`trace_id`, `problem_id`, `prompt_ids`, `response_ids`, `categories`, the exact
`generation_request`, and `generation_record` (`path`, zero-based JSONL `line`,
and zero-based `output_index`). The record must match a sealed successful
generation log. Short-labelled fixtures must be shorter than all long-labelled
fixtures. Required categories are short, long, finished, capped and empty-prefix.
Capped means 1,024 retained tokens with no stop; empty-prefix must match an
actual request with `prefix_ids: []`. Missing categories block preparation;
a deliberately capped fixture, if needed, must be declared before comparison.

After both approved jobs finish, compare on CPU:

```sh
python -m pilot_eval sampling-sft-instrument-compare --config plans/engine-scores-v1/sampling-sft.instrument.json --oracle plans/engine-oracle-v1/sampling-sft.instrument.json --output-root artifacts
```

Every trace must have exact ID/position alignment, mean error ≤0.001 nats/token
and maximum per-token error ≤0.01 nats. `score-receipt.json` retains aligned
scores, worst position and errors, including failed gates. No threshold changes
or passing-trace search occurs. CPU-injected scores always remain unqualified.
Passing T3 establishes score correctness on the fixture, not agreement of
near-tie search decisions across engines.

## Bounded throughput candidates

Mode `profile` requires the same 64 fixed problems and k2, for one block only.
Prepare candidate plans before spending; vary `max_num_seqs` across
32/64/128/256/512 and explicitly declare the batched-token candidates. Each
candidate uses a separately reserved job. Record workload wall time, generated
and uncached input tokens, actual worker peak allocated/reserved memory and
first-extension batch-versus-singleton token differences.

Token rates are request-wall throughput proxies, not isolated prefill/decode
kernel timings. Auxiliary scorer decode is excluded from generated-token counts;
batch-dependence reruns occur after the timed workload. Rates and measured
lengths inform the next reservation, not an unqualified GPU time promise.

Selection is CPU-only. A review JSON names `candidates`, `score_config`,
`oracle_config`, `reviewer` and `notes`. Run `instrument-select --config` with
the fastest measured passing candidate and `--review` with that JSON. It checks
matching source inputs, actual GPU evidence, memory/batch evidence and T3 for
the selected runtime, then saves `runtime-selection.json`. Incomplete/failed
candidates must be diagnosed and the candidate set reviewed explicitly; the
software does not silently change settings after a failure. This selection is
Stage 0.5 evidence. Exact pilot restart/resume and alarm review still block
full generation.

### User-approved bounded-job budget waiver

For the 2026-10-31 A40 smoke, the user withdrew the original cost cap as an
execution blocker. A job approval can explicitly include:

```json
{"waive_budget_limits": true, "budget_sha256": "CURRENT_LEDGER_SHA256",
 "request_sha256": "EXACT_REQUEST_SHA256", "reviewer": "Prasad",
 "notes": "User withdrew original cap; historical accounting remains incomplete."}
```

These fields augment, rather than replace, paid execution and source-audit
approval. The stored reservation retains estimates/costs; cap-exceeded summaries
remain truthful. Missing historical records must be labelled incomplete, not
silently treated as zero. No environment variable or automatic fallback waives a
cap. A stale or unbound waiver is rejected. This supports the bounded instrument;
production generation and training retain their separate gate reviews.
