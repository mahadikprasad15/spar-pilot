# 09: Produce automatic relevance and sample-size results

**What to build:** Apply frozen analysis definitions to saved readouts and retained examples, producing the primary results and declared layer/sample-size curves independently of human grading.

**Blocked by:** 08: Decode saved means and reveal controlled token lists.

**Status:** implemented (CPU fixture validation passed; real analysis pending)

- [x] Independent hand-worked vocabulary/null/paired-example cases match public analysis outputs.
- [x] Primary testing family, direct arm contrasts and secondary searches are separately reported; one arm passing a threshold does not prove an arm difference.
- [x] Whole-example draws preserve paired positions/conditions; stored precision and undefined SNR/noise cases are disclosed.
- [x] Report the predeclared primary result separately for legacy 150 and new 50 documents, with their counts and uncertainty/noise context. Lower readability in the smaller subset does not by itself establish outcome selection; any size-matched sensitivity uses frozen draws.
- [x] No missing convention receives an invented default after readouts are revealed; changes create an exploratory amendment.
- [x] Analysis persists its unit identities/draws and resumes meaningful costly work; report actual CPU/readout costs rather than claiming analysis is free.

## Implementation handoff — 2026-10-16

Saved CPU workflow implemented in `trace_analysis.py` and `trace_analysis_scores.py`, with public CLI and notebook26A/29/30. See `docs/pilot5-automatic-analysis.md`. Primary pooled four-test family, direct C3 contrasts, retained-example resampling/strata/N-curves and secondary logit layer/max-position results persist their draws and costly projection chunks. Real computation is pending verified capture/readouts. JLens analyses outside the accepted pooled block14 scope and the unqualified covariance-shaped null remain explicitly unavailable; endpoints-only production omits intermediate checkpoints. No numerical acceptance number was changed and no real learned vector was scored during implementation.

Validation: 32 affected tests passed in128.68s. Public workflow fixtures include independent score literals, fresh-process verification, null-before-readout ordering, interrupted chunk reuse, strict source/summary hash checks and matched max-five recomputation. No real primary result or human-grade claim.
