# Pilot 5 Tickets 07–08: capture and saved-readout implementation

## Delivered scope

- Reviewed production proposal/freeze interface with explicit qualification, lens, scope, budget and prediction gates.
- One-model raw capture with shared per-batch base references, all 29 named sites, required retained banks and FP64 moments.
- Immutable batch units, exact lossless-compression readback, local staging, partial-copy recovery, free-space checks, writer locks and durable verification.
- CPU norm/head/tokenizer preparation from a completed profile or frozen production contract, including BF16 weight extraction without Torch or decoder construction.
- Gated saved-vector readouts: direct positive/negative ranked tokens, normalized unique lists, scores, provenance, norm/coherence, separate controls and explicit zero/unsupported states.
- Public CLI and shared notebook stages 21–28. Readout units can pause by work-unit budget and resume independently.

The operating guide is `docs/pilot5-capture-and-readouts.md`.

## Boundary and pending real execution

No real production capture, learned-token reveal, numerical acceptance, lens fallback or new fine-tuning was executed locally. Real qualification receipt production remains work owned by Tickets 04/06; this release consumes that contract and refuses missing/unqualified receipts. The cost profile is not a qualification receipt. The inspected community Jacobian remains unqualified pending provenance resolution. A logit-only scope requires explicit approval; none was inferred.

The next safe user operation is CPU 1–2 → 21 after publication. Production launch remains blocked on actual reviewed qualification, lens disposition, endpoint/trajectory scope, retained precision, budget and user predictions/confidences. Numerical sensitivity can use the saved profile arrays; preparing pinned readout basis from that profile does not require production approval or another decoder capture.

## Modules and objects

| Module | Responsibility |
| --- | --- |
| `trace_production_plan.py` | Outcome-blind proposal; conservative storage candidates; explicit reviewed capture contract; parent/implementation/runtime identity. |
| `trace_production.py` | Batched capture, FP64 reductions, retained shards, staging/sync transactions, budget/space checks, resume, aggregate and CPU verifier. |
| `trace_production_backend.py` | Optional pinned FP32 CUDA model with the selected existing checkpoint adapters. |
| `trace_readout_assets.py` | Pinned norm/head/tokenizer basis, CPU tensor-byte reader, portable sealed asset bundle. |
| `trace_readouts.py` | Prereveal request gate, primary/secondary/control vector construction, chunked lens projection, ranked-token units and comparison table. |
| `cli.py`, `notebook_session.py` | Public operations, streamed output and completed-GPU-run CPU reuse. |

The capture plan is not an inference-plan substitute. Its primary identity is fixed by the approved design; secondary readout requests and the primary testing family require their own prereveal review. Ticket 09 owns statistical overlap/null/sample-size conclusions.

## Tests and evidence

Session-start full suite: **361 passed, 35 skipped**, 132 warnings, 1029.22 seconds. This was collected before the new Ticket 07/08 tests. Final affected suite: **41 passed**, 133.68 seconds. Logs, retained development failures and structured validation evidence are in `artifacts/runs/diagnostics/pilot5-tickets07-08-implementation-v1`. Explicit TF32 assignments in the optional backend were syntax-checked; no CUDA execution is claimed.

Independent CPU tests cover literal cohort sums, exact primary-vector/token-rank answers, non-symmetric Jacobian orientation, BF16 byte values, zero vectors, unsupported controls, source/receipt/runtime gates, disk failure, interrupted forward and sync recovery, writer conflict, payload corruption, no-decoder readouts, fresh-process CLI verification and notebook stages. Injected synthetic receipts are explicitly tagged and cannot launch a real CUDA backend. Tests download no model and call no grader.

Development regressions caught missing readout-output directory creation and an outdated completion-inventory check when JSONL/Markdown outputs were added. Both were repaired in code without changing any acceptance number. Their logs are retained with the validation evidence.

The new CUDA capture backend and real pinned-weight extraction path still need their bounded target-environment validation after the pending approvals. CPU fixture success is not real-model qualification.

## User-reported completed cost profile

The user supplied successful cells 19 and 20 on a Tesla T4:

- Eight units, 32 sequence forwards, 20 model calls; verification child exit 0.
- Model load: 38.48 seconds.
- Forward/capture/transfer total: 8.54 seconds.
- Base hashes total: 404.32 seconds.
- Raw profile arrays: 730,055,344 bytes.
- Batch-4 forward calls were approximately 0.66–0.69 seconds for four documents; batch-1 totals approximately 1.27–1.81 seconds for the same four documents.
- Maximum reported batch/singleton activation differences: base 0.00830078125, SFT 0.0078125, GRPO 0.00830078125, random 0.00537109375. No allowance was selected.

These are user-reported execution outputs, preserved under `artifacts/runs/pilot-5/user-reported-endpoint-profile-2026-10-16`. They establish profile completion, not numerical acceptance or statistical readability.

## Costs and operating limits

Production includes resumable batch sums/moments, so the proposal estimates their storage in addition to final retained banks. At batch 4 and FP16 token retention: endpoints approximately 11.86 GiB; trajectories approximately 28.31 GiB of arrays before lossless compression. Actual size and saves are measured. No automatic scientific scope reduction occurs.

Production currently performs full base hashes around each atomic batch. The measured T4 hash cost must be included in budget review; GPU arithmetic alone would substantially understate runtime. The command enforces time limits at batch boundaries per invocation and checks total durable usage. It does not promise a provider billing cap.

Durable verification covers the configured filesystem mount, readback, checksums and file flushes. It does not independently query a Drive remote-sync API. Completed stages verify on CPU before model access; loss of unsealed local staging can repeat only its unfinished batch. Different GPU/dependency settings require a reviewed compatible execution/bridge.
