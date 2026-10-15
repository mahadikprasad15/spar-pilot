# Ticket 04 implementation report

15 October 2026 · Source commit `ef3b98269151947c774f56404cf3020187a06180`.

**Bounded diagnostic implementation is ready; real CUDA execution and reviewed numerical qualification are pending.** No GPU/model download was used for local implementation verification.

## Approved scope

Prasad approved four fixed diagnostic documents (two legacy, two new), the base model and both saved step-zero adapters. He approved implementing diagnostic evidence first and reviewing GPU/FP16 acceptance limits afterward. The agreed public saved-workflow test seam and independent CPU math oracles remain unchanged. No tolerance or acceptance number was relaxed.

## Implementation

| Component | Responsibility |
|---|---|
| `trace_capture.py` | Paired content-position reductions; per-example pooled base/delta vectors; retained positions; FP64 moments; generic named hooks; logit-lens gain; conversion and paired bootstrap/rank sensitivity; structural zero checks |
| `trace_capture_backend.py` | Lazy pinned Torch/Transformers/PEFT boundary, one loaded model/two zero adapters, actual pre-norm hooks, exact base hashes and bounded numeric final-norm/head diagnostic |
| `trace_capture_workflow.py` | Canonical plan, sealed source inputs/used adapter files, fixed four-document selection, runtime-bound units, resume, failure coordinates, verification and honest pending-review status |
| CLI / notebook runner | Public `traces-capture-prepare`, `traces-capture-diagnose`, `traces-capture-verify`; active-GPU guard; completed CPU reuse |
| Shared notebook | Cells 14–16: CPU prepare, GPU diagnose/resume, CPU verify; persisted diagnostic selection |

The real workload is 12 singleton sequence forwards and one auxiliary base norm/head check. Capture collects all 29 sites together. Diagnostic scope is not a performance profile or production checkpoint sweep. Source checks verify sealed preparation and the actual step-zero adapter payloads; unrelated optimizer states and checkpoints are not rehashed.

## Verification

Session-start full suite: **336 passed, 35 skipped**, 132 warnings, 1020.54 seconds. This collected before the new Ticket 04 workflow. Final affected suite: **88 passed** (exact duration and log in the validation receipt). New Ticket 04 tests: **15 passing CPU tests**. Tests used synthetic inputs, a tiny independent hook model and an injected CPU backend; no pretrained download or grader API call. The pinned real CUDA backend has not yet been exercised in Colab, so these results are not real-model qualification.

Independent literal/full-array oracles cover content offsets, pre-final-norm named blocks, nonzero paired differences, primary pooling, base sum/outer sum, norm gain, positive-scale ranking, zero-vector behavior, conversion error and a constructed FP16 rank flip. Structural checks cover both rank-one zero paths and full projection coverage. Saved workflow tests cover interrupted conditions, skip/reuse, exact failed comparison coordinates, no failed seal, changed-plan refusal before GPU access, fresh CLI verification, completed CPU reroute and execution of actual notebook prepare/verify cells.

## Improved-design requirements

- **A7:** individual per-document positions 0–4 feed whole-document primary and secondary analysis; the reducer retains them at the six declared blocks. All-site primary pooled vectors are FP64.
- **A6:** FP64 base Σh/Σhhᵀ at block 14, positions 1–127; four-document count 508. These diagnostic moments do not substitute for production's 25,400-token moments.
- **Precision:** conversion evidence reports actual worst coordinates; FP16 overflow is failed candidate evidence, not an invalid-JSON infinity or a widened gate. The sensitivity utility uses identical document draws for original/converted vectors and reports rank changes. No real nonzero-adapter precision claim is made.

## Evidence and failure behavior

Plans and runs stay under the canonical artifact root. Unit seals bind plan/runtime/condition and payload hashes. Each new condition is sealed only after a matching frozen-base hash; interrupted unsealed work repeats. Input-plan corruption fails before model/runtime access. Nonfinite hooks and failed zero differences retain site/token/example coordinates when available. Base moments, summaries, payloads and seals are verified on CPU. Completion says `diagnostic_complete: true`, `qualified: false`, `state: awaiting-review`.

## Next execution

Fresh CPU **1–2 → 14**. Fresh GPU **1–2 → 4 → 15 → 16**. Share the summary and any failure evidence. Real duration is not benchmarked; setup, initial model download, base-hash reads and Drive saves are explicit costs. See [operating guide](docs/pilot5-capture-diagnostic.md) for workload, storage and exact recovery rules.

Ticket remains claimed until real execution and gate review. The original primary family still requires community Jacobian-lens qualification. Nonzero trained-adapter sensitivity and production resource/precision approval remain separately declared in Ticket 06. Neither those stages nor full production capture were silently added here.
