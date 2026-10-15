# Pilot 5 bootstrap

The dedicated shared notebook is `notebooks/pilot-5-colab.ipynb`. Cells 1–6 restore sources and check environment/storage assumptions; cells 7–9 prepare the raw corpus on CPU. It does not run Pilot 5 model inference, capture or readouts.

## Current resume point

The user has completed the source audits, import/storage smoke, fresh-runtime reconnection and verified 200-document raw corpus. In the updated notebook, use a fresh CPU runtime and run **1–2 → 10–12**, inspect the reference previews, then fill and run **13**. Skip 3–9. See [reference workflow](pilot5-prereveal-review.md). No GPU allocation is needed for this stage.

## Minimal Colab validation

1. Open the notebook. Run cells 1–2 to mount Drive and restore saved state.
2. Select a GPU for the import smoke. Run cells 4–5. No weights are downloaded, but dependency installation can take time; the recorded smoke duration covers imports and the small artifact round trip, not installation or idle allocation time.
3. When cell 5 finishes, restart the runtime. GPU allocation is unnecessary afterward. Run 1–2 and then 6 in the fresh kernel.
4. Save the output showing `Real-Colab bootstrap validation: completed`. The receipt and GPU identity remain under `artifacts/runs/pilot-5/<experiment>/bootstrap/`.

Cell 3 separately audits the existing prepared SFT, GRPO and random sources. It checks source seals through the established validators without loading a model. It is not needed to test the small storage round trip. Avoid repeating deep audits merely to restore notebook variables.

## Evidence and boundaries

- `selection.json`: explicit source paths; no source discovered by “latest” naming.
- `source-audit.json`: verified identities from the last explicit audit. Restore only checks the saved config fingerprint; capture must revalidate dependencies.
- `smoke.json` and `smoke-seal.json`: actual imported runtime, GPU, fresh-kernel token and sealed small payload.
- `reconnect.json`: fresh-process readback receipt, payload hash and verification duration.
- `meta/status.json`, `logs/run.log`: measured lifecycle evidence. The restore operation uses payload seals, rather than trusting this status file alone.

A pending or unsealed artifact does not establish target validation. Fixture GPU metadata in CPU tests does not count as a Colab run. Import compatibility does not establish numerical capture compatibility: qualification is a later stage. Storage smoke proves this small-file workflow on the observed mount; it is not a distributed-lock guarantee or a benchmark of large shard uploads.

Use a separately named experiment when the smoke runtime changes. Existing evidence is preserved. An active experiment child blocks readiness before model imports; the bootstrap never terminates it automatically.

The analysis contract exposes embedding output plus block outputs 0–27 before the final model norm, content-relative versus absolute input positions, and the approved retained-data capabilities. Statistical rules, checkpoint scope, storage precision, lens acceptance and grading settings remain later review gates.
