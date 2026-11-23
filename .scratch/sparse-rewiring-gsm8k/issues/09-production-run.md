# 09: Production run

**Status:** ready-for-human
**Type:** task
**Blocked by:** 08

**Needs the researcher's approval of ticket 08's launch record.** Two runs, rank 1 then rank 0, on the same GPU (D18), sharing one plan apart from rank and read LR. Each run: train with the dynamics logging (D20), harden, run validation (including the gold-sft step 0 and 64 references), greedy at 64 and 512, the activation-shift pass (D22), verified export. If λ 6e-4 is degenerate for an arm, stop that arm and return to the researcher with the D6 sweep plan; the sweep runs are separate run IDs under the same launch record only if it pre-approves them, Pod stop if a Pod is used. Analysis outcomes never stop the chain; integrity failures do.

## Comments

## Answer
