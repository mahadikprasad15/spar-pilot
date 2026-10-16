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
