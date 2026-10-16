# Pilot 5 Tickets 05–06: inspection and bounded profile handoff

## Implementation boundary

Ticket 05 now has a CPU-only public inspection command and independent tiny-matrix/archive checks. The actual pinned community artifact was inspected locally without Torch or pretrained model inference. Its terminal status is **insufficient provenance**, not qualified. See `docs/pilot5-lens-and-profile.md` for source links and the exact unresolved fields.

Ticket 06 now has public `traces-profile-prepare`, `traces-profile-run` and `traces-profile-verify` commands, a one-model optional CUDA backend, and shared notebook cells 17–20. Prasad approved four existing diagnostic documents, four conditions (base/SFT64/GRPO64/existing-random64) and batches 1/4: 32 sequence forwards in 20 calls. This is a cost diagnostic only. Production freeze is deliberately not implemented as an approved state: numerical limits, lens disposition, production scope, budget and predictions are unresolved.

## Modules

| Module | Responsibility |
| --- | --- |
| `trace_lenses.py` | Restricted ZIP tensor inventory, SHA verification, matrix transport, sealed provenance status. Whitelisted tensor reconstruction metadata only; no arbitrary pickle globals. |
| `trace_community_lens.py` | Pinned Hub download and model-specific provenance claims. No model load or fitting. |
| `trace_profile.py` | Bound used source files, random construction and model metadata; freeze bounded inventory; save/verify/resume eight units; report measured costs and unthresholded batching/conversion diagnostics. |
| `trace_profile_backend.py` | Optional FP32 CUDA model with three existing endpoint adapters and hooks on all 29 sites. |
| `cli.py`, `notebook_session.py` | Public saved-workflow commands; GPU readiness only when new GPU work is needed; completed results verified on CPU. |
| Shared notebook | CPU inspect/prepare; GPU profile; CPU verify/review. Existing cells retain numbering. |

## Verification

Red/green tests were run before each slice: missing lens module; missing required inspection seal; changed provenance receipt; missing profile module; missing notebook commands. Current affected suite: **17 passed**, 8.31 seconds. It covers independent non-symmetric matrix orientation, restricted loader rejection, tampering, source binding, interrupted-unit recovery, changed-runtime rejection, CPU-only completion reuse, public CLI and notebook stage exposure. Unit tests do not download a model or import Torch.

Session-start full suite: **351 passed, 35 skipped**, 132 warnings, 1051.24 seconds. Separate notebook/session regression checks: **22 passed**, 27.83 seconds. Logs and the validation receipt are saved under `artifacts/runs/diagnostics/pilot5-tickets05-06-implementation-v1`. No local CUDA profiling or real nonzero sensitivity qualification is claimed. The local actual lens inspection is complete, while the new profile still needs its first Colab run.

## User-reported Ticket 04 evidence

The user supplied successful public Colab diagnose and verify outputs on a Tesla T4: 4 documents, 12 model calls, 29 sites, both zero adapters exactly unchanged, base unchanged, 508 base moment observations. The first-32-head-row base numeric-lens error was `8.366587680441739e-07`, with allowance `null`. These outputs say `diagnostic_complete: true`, `qualified: false`, `state: awaiting-review`; both commands exited 0. This is user-reported execution evidence, not a local rerun or an approved numerical allowance.

## Deliberate remaining work

No production capture, checkpoint sweep, learned-token reveal, lens fit or automatic threshold adjustment was added. Profile timings cannot be interpreted as acceptance. Nonzero readout precision/bootstrap sensitivity, production budget/predictions and scientific lens disposition remain review items.
