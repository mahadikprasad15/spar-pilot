# Execution status

Documentation review: 16 October 2026. This ledger separates software availability
from run completion and scientific interpretation. It uses preserved local reports
and supplied execution logs; it is not a fresh scan of Google Drive.

| Pilot / component | Software | Execution evidence | Remaining work / limit |
|---|---|---|---|
| 1: GSM8K/MMLU baselines | Implemented | Historical outputs and protocol variants exist | Choose exact run/config when citing; no single newly audited baseline aggregate |
| 2: SFT training and trajectory | Implemented | Saved adapters and preserved response audit through step 64 | Original full runtime/config provenance not independently reverified in this refresh |
| 3: SFT fixed-input writes | Implemented | Original T4 and separately named matched A10040 report evidence | Report-level review; original binary shards not rehashed here |
| 4: GRPO training | Implemented | 64 updates, 4,096 accepted draws; source completion recorded | One training seed |
| 4: Held-out behavior | Implemented | Five greedy checkpoints and endpoint sampling in preserved report | Cite scorer/decoding and item-level interval rule |
| 4: Matched SFT/GRPO/random writes | Implemented | A10040 reports preserved; comparison cell 27 later reported complete | Later comparison output not reimported locally |
| 4: Coefficients and KL | v2 implemented | v1 profile failed; bounded diagnostics support accepted batch-policy amendment | v2 profile, reviewed freeze, production measurement/verification not confirmed complete |
| 4: Final handoff | Implemented | Partial source/report snapshots | Final combined verification depends on remaining supplement |
| 5: Bootstrap and source audit | Implemented | Real-Colab fresh-runtime/storage check and source checkpoint audit supplied | Revalidate source identities before new capture |
| 5: Corpus and references | Implemented | 200-document corpus and three 200-string references supplied | Reference review/identities travel with analysis |
| 5: Lens/capture qualification | Implemented | Reviewed T4 batch-4 FP32 receipts supplied | Scoped acceptance; not semantic validation of JLens |
| 5: Endpoint production capture | Implemented | Cell 23 log reports durable completion; cell 24 reported complete | No fresh Drive verification performed for this page |
| 5: Readouts and automatic analysis | Implemented through Ticket 09 | Cell 25 last reported running | No completed learned-readout/relevance result available here |
| 5: Human grading / final report | Later tickets | Not established | Tickets 10–11 remain; low-level grading utilities are not completed study evidence |

## Evidence sources

- Pilot 2 and Pilot 4 behavior: local `artifacts/reports/pilot4-results-so-far-2026-10-13/inputs/` JSON snapshot.
- Matched write comparison: `artifacts/reports/pilot4-a10040-v2-review/` and the same snapshot's write results.
- Pilot 4 supplement: [accepted amendment](../adr/0010-pilot4-coefficient-batch-sensitivity.md) and supplied failed profile diagnostics.
- Pilot 5 capture: [normalized console summary](evidence/pilot5-capture-summary.json); independent cell-24 completion is a user report.
- Specs and ticket statuses record design/software progress. They are not substitutes for completion manifests.

When new results arrive, update this ledger with the run/config path, verifier
outcome and evidence date. Preserve failed execution records and older reports.
Do not convert “implemented” or a zero process exit alone into scientific success.


## Pilot 4 behavioural follow-up — 22 October 2026

This update concerns the saved-output follow-up only; the table above retains
its 16 October review scope for the other components.

The CPU source/scoring, paired-transition, sampled-reliability and audit software
are implemented and executed against named local outputs. The combined analyst
report is at
`artifacts/reports/pilot4-behaviour-findings-2026-10-22-v2/report.html`.
It verifies parent receipts and reconciles per-item counts and saved paired
bootstrap draws. No new training, generation or activation capture was performed.

The user classified 215 selected responses. Twenty-two numeric transcriptions
are human-confirmed, including all 16 assisted-review flags; 188 automatic
proposals are excluded from the correction overlay. This is completed targeted
assisted review, not an exhaustive human-scored accuracy audit or the original
complete-label freeze/reveal route. The nine corrected shared untuned responses
were explicitly rechecked.

Confirmed-case overlays change GRPO's counts from 102→114 to 111→117 of 150
(+4 pp instead of +8 pp), and SFT's from 102→99 to 111→100 (−7.33 pp).
Original sampled analyses remain automatically scored and unchanged. These
results do not establish reasoning quality or completion of Pilot 4's separate
KL/coefficient supplement.

See [workflow and evidence guide](../pilot4-behaviour-breakdown.md) and the
[Ticket 05 execution record](../../.scratch/pilot4-behaviour-breakdown/execution-ticket05.md).

## Stage 0 primary path — 27 October 2026

Tickets 01–07 implement source readiness, CPU spectral/edit preparation, bounded
loader qualification, paired greedy collection, fixed-context KL/effective writes,
and the CPU primary findings/review/report workflow. Implementation checks use
synthetic CPU fixtures. No real Stage 0 source/weight execution, GPU qualification,
paired collection or scientific finding is established by those tests.

The primary report can complete with targeted review pending and optional work
absent. Raw automatic scores and confirmed-case overlays are separately labelled.
The conditional intruder and optional explanatory branches remain separate
implementation/execution work. See [the Stage 0 guide](../stage0-geometry.md) and
[Ticket 07 execution record](../../.scratch/stage0-geometry/execution-ticket07.md).

### Conditional geometry extension

Ticket 08's CPU rotation/intruder report is implemented and locally validated.
It always measures all module/arm units and distinguishes chance-undecidable
modules from modules with no intruders. Real-weight execution is pending.
Ticket 09 will construct the conditional merged V6/V6c edits from verified
eligibility evidence; no such edit or behavioral result is established here.

### Optional initial pushes — 27 October2026

Ticket13 software is implemented: source-hashed initializations, matched CE
batches, initial GRPO gradients, bounded profile/production allocations,
resumable gradient artifacts, and CPU prediction-bound endpoint reports.
All11 ticket/handoff tests and42 existing affected regression tests passed across the
recorded CPU environments. Three actual Torch tests ran separately because the
regular local Torch installation is incomplete. No real GPU collection or
scientific alignment result is implied. CPU34–36 / GPU32–37 expose this branch;
see the [execution record](../../.scratch/stage0-geometry/execution-ticket13.md).

### Optional input energy — 27 October2026

Ticket14 implements one untuned base forward per fixed sequence, shared by
SFT/GRPO endpoint/zero and random input factors at three spectral cutoffs.
FP64 principal/rest/cross/independent-total sums, approved normalized projection
checks, bounded budgets/resume and CPU tables/plots are implemented. CPU37–39 /
GPU38–43 expose the branch. Test artifacts are synthetic software evidence;
real GPU measurements remain pending. See the
[execution record](../../.scratch/stage0-geometry/execution-ticket14.md).
