# 01: Restore a clean session, audit sources and smoke-test Colab

**What to build:** A fresh readable notebook restores Pilot 5 state, audits existing sources, declares stable named-site and report/optional-grade contracts, and demonstrates environment/storage readiness before larger slices depend on Colab assumptions.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Public restore/inventory operations on an offline fixture show actual missing/pending/completed states without starting a model or older pilot.
- [x] The site contract distinguishes embedding output and block indices zero through 27, pre/post normalization and content/absolute positions.
- [x] CPU restoration works without importing GPU libraries; actual GPU text-model imports and optional package conflicts are checked in a bounded fresh-process path.
- [x] A recorded minimal fresh-Colab check mounts durable storage, writes/verifies a small artifact, disconnects/reconnects and reads it through the public restore path. Actual time is recorded, not promised.
- [x] A changed runtime, owned active child and ambiguous source selection fail with actionable messages before inference.
- [x] Limit the early grade contract to a schema version and not-requested/pending/validated status. Detailed grade fields belong to grading/report consumers, not this bootstrap ticket.
- [x] Define supported analysis capabilities without selecting outcome rules: per-position cohort means at all sites; whole-example primary-window pooled evidence at all sites; per-position example evidence at the six retained blocks. Other data demands require a separately reviewed capture extension.
- [x] Distinguish offline CPU completion from real-smoke evidence: this ticket cannot be reported as fully target-validated on CPU alone. The real GPU allocation and smoke duration are recorded.

## Implementation evidence · 2026-10-14

- Public interface: `pilot_eval.trace_session.TraceSession`; readable notebook: `notebooks/pilot-5-colab.ipynb`.
- CPU workflow tests: 10 passed. Each implemented slice had an observed red test before green and was committed separately.
- CPU source audit delegates to the existing sealed preparation validators, then checks arm kind and shared model/tokenizer/precision/base identity. Restoration records prior config audit status and never treats it as a fresh deep source audit.
- GPU readiness checks actual Qwen/PEFT/TRL imports in a fresh process with a 120-second bound; no model weights are loaded. Runtime changes require an explicit new qualified execution.
- Smoke interruption between payload and seal is recoverable. Fresh-kernel identity uses a random process token, so container PID reuse does not reject valid reconnection.
- Real Colab bootstrap smoke and fresh-runtime reconnection passed, as reported by the user through the public notebook commands. This resolves bootstrap validation; it does not qualify production numerical capture.
- Bootstrap writer locks coordinate processes sharing one filesystem lock implementation; they do not claim distributed locking across Colab VMs.

- Existing repository regression after restoring missing local test dependencies: 279 passed, 35 skipped, 0 failed (1013.78 seconds). Ten new bootstrap tests also passed. No acceptance tolerances changed.

## Target evidence received · 2026-10-14

The user reported the public smoke output and fresh-runtime validation `state: completed`. Observed runtime: Python 3.13.15, Torch 2.9.0, Transformers 4.57.6, PEFT 0.18.0, TRL 0.26.2, datasets 4.4.1, accelerate 1.12.0, huggingface-hub 0.36.0, CUDA 12.8, Tesla T4 (15,637,086,208 bytes). Smoke duration: 15.660914148 seconds, excluding dependency installation; no model inference.

The user subsequently reported the completed source audit: SFT and GRPO share Qwen/Qwen2.5-1.5B-Instruct with checkpoints 0/8/16/32/64; random uses 0/64. Receipts are saved under `artifacts/runs/pilot-5/pilot5-readable-traces-v1/bootstrap`. These are user-reported validator results, not independently fetched raw receipt contents.
