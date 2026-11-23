# 09: Production run

**Status:** ready-for-human
**Type:** task
**Blocked by:** 08

**Needs the researcher's approval of ticket 08's launch record.** Two runs, rank 1 then rank 0, on the same GPU (D18), sharing one plan apart from rank and read LR. Each run: train with the dynamics logging (D20), harden, run validation (including the gold-sft step 0 and 64 references), 1,024 steps with freeze at 960 (D5), greedy at 64 and final, the activation-shift pass (D22), verified export. A degenerate survivor count is reported, not re-run (D6). Pod stop if a Pod is used. Analysis outcomes never stop the chain; integrity failures do.

## Comments

## Answer
