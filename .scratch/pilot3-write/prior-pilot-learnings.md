# Lessons from Pilots 1 and 2 for Pilot 3

These are design inputs for the Pilot 3 spec. The user confirmed the layered
verification boundaries on 2026-10-04. The interview remains
the record of questions and decisions; this document explains the safeguards.

| Earlier issue | Lesson | Pilot 3 safeguard |
|---|---|---|
| Flexible v2 rejected correct `####` answers after SFT. The apparent final accuracy collapse disappeared under v3. | A measurement bug can create the phenomenon being investigated. Toy examples alone are insufficient. | Hand-computed numerical fixtures, exact-zero checks, rank-1 identity checks, deliberately wrong controls, and tests of the resulting saved report. |
| The Qwen template inserted a system message despite the protocol specifying none. | Configured intent and the actual model input can differ. | Save and inspect rendered inputs, token IDs, masks and counted positions; verify actual inputs rather than only config labels. |
| Batch 8 passed a short audit but ran out of memory on long MMLU 5-shot prompts. | A short smoke test does not establish capacity for the longest inputs. | Benchmark the longest actual sequences with the baseline cache present; freeze a passing batch plan before production. |
| Slow serial evaluation prompted batching changes; a hardware whitelist then rejected a valid L4/batch-8 configuration. | Capacity depends on workload and capabilities, not just a GPU name. | Parameterized batches, measured forward-pass throughput and memory, explicit numerical agreement checks; avoid a GPU-family whitelist. |
| A notebook checked out an older commit without the profiler; imported Python code could also remain stale after checkout changes. | Notebook version, checkout version and loaded code are separate states. | Display code/dependency versions, check required entry points before expensive work, and provide explicit fresh-process/restart instructions. |
| Progress monitoring encountered an empty JSON file while the evaluation child remained active. | A monitoring read failure is not necessarily a computation failure. | Tolerate transient progress reads, track the child process independently, prevent duplicate writers, and strictly verify final artifacts. |
| Equal highest MMLU logits originally stopped a run; the revised policy retained ties as invalid and incorrect. | Undefined measurements need an explicit policy before running. | Preserve null ratios with counts and coverage; distinguish undefined denominators from nonfinite values or failed validation. |
| Batch changes required new plans; mixed recovery runs and old absolute paths became confusing. | Resume identity and artifact location must be explicit. | One shared artifact root, verified source manifests, frozen batch membership/settings, hashed completion markers and clear provenance. Preserve historical configs. |
| Historical pilot settings were unavailable, and our matched baseline differed. | A completed run is not automatically an identical reproduction. | Describe observed writes and uncertainty; preserve contextual historical references and avoid causal or replication claims unsupported by the inputs. |

## Proposed layered verification

1. Focused CPU tests with independently calculated expected values for masks,
   vector sums, norms, weighting and undefined ratios.
2. Workflow tests through public entry points and persisted artifacts: prepare,
   validate, measure, interrupt/resume and reconstruct the report. Controlled
   model/data fixtures keep these tests offline.
3. Tiny real Qwen + PEFT CPU integration tests, constructed locally, to check
   real hooks, adapter switching and padded batches without a model download.
4. Real GPU preflight on the selected workload for numerical checks, memory,
   throughput and the frozen production batch plan.

Each workflow scenario should target a meaningful behavior. Do not replace all
tests with one giant end-to-end test, or assert internal call sequences that
merely mirror the implementation. Passing tests supports instrument correctness;
it does not establish a scientific causal claim.
