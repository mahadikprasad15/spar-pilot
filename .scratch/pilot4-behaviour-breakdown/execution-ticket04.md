# Ticket 04: intended-answer audit handoff

2026-10-22. Software implemented and fixture round trip verified. The real packet
is prepared; **human review has not been performed**. No real labels, frozen
answers, revealed results or explanation annotations have been invented.

## Real preparation

- Parent: `artifacts/reports/pilot4-behaviour-transitions-2026-10-22-v2/`.
- Plan: `artifacts/plans/pilot4-behaviour-audit-2026-10-22-v1/behaviour.audit.json`.
- Run: `artifacts/runs/pilot-4/behaviour-audit/pilot4-behaviour-audit-2026-10-22-v1/`.
- State: `awaiting-review`, draft revision 0, all **215 unique responses pending**.
- SFT: 51 mandatory pairs + 20 controls = 71 pairs.
- GRPO: 32 mandatory pairs + 20 controls = 52 pairs.
- Total: 123 pairs, 246 endpoint references, 215 unique responses after verified
  content/input/token-metadata deduplication. All references remain private.

The required invalid/transition/extractor-disagreement cases were retained in
full. Each method's controls use a separate Python `random.Random(42)` on its
sorted remaining IDs. Pools, selections and reasons are persisted. No threshold
or workload cap was changed to reduce the packet.

The embedded reviewer data were checked against `reviewer/packet.json`: every
entry contains only opaque response ID, question and response. Packet bytes match
their configured SHA-256. No real frozen-label file or reveal directory exists.
Only the `reviewer` folder is appropriate for the initial reviewer handoff;
`private` contains gold/source/scorer mappings. This separation is not access
control, and writing style or prior exposure can still suggest the source.

## Reviewer workflow

1. Open `reviewer/review.html` locally. Enter a reviewer name or ID.
2. Identify the intended final answer: numeric transcription, absent, ambiguous
   or pending. Evidence and notes are available. Do not grade reasoning validity.
3. Download `answers.json` to preserve progress. Browser caching is temporary.
4. Save through notebook section 9 or the public `behaviour-audit-save` command.
   Then import its updated `reviewer/draft.json` before continuing the page.
5. Explicitly freeze complete labels in section 10. Pending labels block freezing.
6. Section 11 reveals gold/scorer comparisons and an explanation editor.
7. Section 12 saves named explanatory snapshots. Post-reveal answer corrections
   require a separate named, reasoned revision; original frozen labels stay intact.

The notebook remains an unpublished local release. Preparation and every later
stage restore paths from the saved configuration after setup/settings. No model
dependencies, generation or GPU allocation are needed.

## Verification evidence

- Session baseline, started before implementation: **442 passed, 35 skipped,
  4 warnings**, 1,652.73 seconds. This is not a full post-change suite. The known
  local Torch import failure accounts for some skipped legacy instrument checks.
- Final affected regression run: **85 passed**, 43.62 seconds, including
  **24 new audit cases** and the previous source/scoring/transition/sampling/session
  regressions.
- A fresh-process notebook fixture executed prepare, draft save, explicit freeze
  and reveal with Torch, Transformers, Datasets, PEFT, Accelerate and Hub imports
  blocked. Freeze without explicit enabling was rejected.
- Node-based DOM interaction checks exercised the generated page's actual draft
  download/import handlers, packet binding, revision restoration and valid-label
  progress. Literal angle-bracket text survives the embedded-JSON round trip.
- Tests exercise exact selection, small control pools, deduplication, omitted
  private fields, interrupted packet publication, sealed draft restoration,
  wrong/stale/malformed labels, freeze ordering, changed frozen/packet evidence,
  exposed corrections and source-bound explanation categories.
- Rendered browser QA was attempted on fixture-only data but unavailable because
  local browser access is disabled in Work settings. No visual-browser validation
  is claimed; the JavaScript interaction and structural checks passed.
- No real Colab run, human answer audit, population human-corrected accuracy,
  reasoning-quality judgment or inter-reviewer agreement is claimed.

The real workload is ready for the reviewer. Ticket 05 can publish a partial
combined report while this human work remains pending.
