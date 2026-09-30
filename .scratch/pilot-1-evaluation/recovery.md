# Smaller-batch recovery and combined reporting

Status: resolved

User-approved implementation: derive a new smaller-batch plan from the exact existing frozen inputs, run the remaining three MMLU cells, retain completed GSM8K and zero-shot MMLU, and combine selected completed sources without moving their original artifacts. The partial source five-shot run is retained but not imported into the new execution configuration.

Commands: `fork-plan` and `report`. Same-notebook instructions are in `docs/colab-recovery.md`, also appended as sections 12–15 of the guided notebook. The update cell pins recovery code to `893c295770fe3d73d40290efc965c800cf2d275b`, which contains both commands and requires no dependency upgrades.

Verification: 66 CPU tests pass. Tests cover preserved revisions, cohort IDs, prompts and hashes; no external access for forks; idempotency; read-only source reporting; different recorded batch sizes; rejection of partial/corrupt/incompatible sources; and explicit system-message warnings. Recovery cells 2–4 ran locally with fake inference and actual CLI subprocesses. Notebook schema and all Python cells validate.

Limits: real Colab/GPU execution is still required. Batch size 2 reduces memory pressure but is not guaranteed to fit all prompts. Existing prompts and scorer semantics are unchanged, including the known no-system-message discrepancy. Reports distinguish verified completeness from protocol compliance.
