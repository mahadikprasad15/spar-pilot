# Ticket 05 execution — 22 October 2026

## Delivered

- Public CPU command: `python -m pilot_eval behaviour-findings`.
- Combined data/recovery module: `pilot_eval/behaviour_findings.py`.
- Presentation and figure module: `pilot_eval/behaviour_findings_render.py`.
- Notebook: `notebooks/pilot-4-behaviour-breakdown.ipynb`, section 13.
- Workflow guide: `docs/pilot4-behaviour-breakdown.md`.
- Real report: `artifacts/reports/pilot4-behaviour-findings-2026-10-22-v2/report.html`.
- Structured results, Markdown, summary CSV, paired greedy JSONL, sampled-count
  JSONL, four SVG/PNG figure pairs, source configuration and completion receipts
  are retained alongside that report.

The report combines recorded/harmonized endpoint scores, original C/W/U flows,
direct paired contrasts, SFT churn/length/strata descriptions, eight-draw GRPO
reward and observed coverage, and the approved partial confirmed-case correction
analysis. It concludes with decisions about optional read-off or reasoning review;
it launches neither.

## Exact named sources

Artifact root: `artifacts/`.

- Scoring plan:
  `plans/pilot4-behaviour-breakdown-2026-10-22-v1/behaviour.analysis.json`.
- Paired transitions: `reports/pilot4-behaviour-transitions-2026-10-22-v2/`.
- Sampling: `reports/pilot4-behaviour-sampling-2026-10-22-v2/`.
- Audit plan: `plans/pilot4-behaviour-audit-2026-10-22-v1/behaviour.audit.json`.
- Confirmed corrections:
  `reports/pilot4-confirmed-extraction-corrections-2026-10-22-v1/`.

The frozen combined config enumerates input receipts and the transitive backup,
proposal and confirmation sources. Hashes, ordered question IDs, counts, paired
contrasts and bootstrap intervals are checked against saved per-response evidence.
No model/adapter weights or activation caches were loaded or scanned.

## Evidence state

- Automatic real-data analysis: completed, 600 greedy responses and 2,400 saved
  sampled responses on the existing 150-question endpoint cohorts.
- Human status classification: 215 unique selected responses (210 numeric,
  three ambiguous, two absent).
- Numeric transcriptions: six originally present plus 16 explicit assisted
  confirmations; the remaining 188 proposals are automatic and excluded from
  confirmed corrections.
- The nine corrected shared untuned responses were rechecked explicitly.
- Original complete-label freeze/reveal: **not performed**. This is completed
  targeted assisted review and a partial correction overlay, not exhaustive
  human-scored accuracy or a completed original full-transcription audit.
- Reviewer packet, original backup and prior scores remain unchanged. The
  analyst report is separate and intentionally contains gold comparisons.

## Findings retained without overstatement

| Measurement | SFT | GRPO |
|---|---|---|
| Original flexible-v3 counts, 0→64 | 102→99 / 150 | 102→114 / 150 |
| Confirmed-case overlay | 111→100 / 150 | 111→117 / 150 |
| Confirmed-case net change | −11 (−7.33 pp) | +6 (+4 pp) |
| Original gross gains / losses | 21 / 24 | 18 / 6 |

Original GRPO sampled flexible-v3 mean reward rises from 54.42% to 67.50%;
observed at-least-one-of-eight coverage rises from 90.67% to 92.00%. The latter
paired interval crosses zero. Last-number sensitivity results remain visible.
The sampled responses were not human-rescored, and greedy corrections are not
applied to sampled rewards. No reasoning-validity, new-ability, mutually exclusive
causal-share or cross-training-seed claim is made. Original bootstrap intervals
are not reused for corrected totals.

## Validation and recovery

- Red/green work used the approved public saved-workflow boundary. Twelve new
  findings cases cover partial reports, complete frozen selected audits,
  assisted provenance, changed backups/labels/parents/outputs, malformed unit
  receipts, per-item/summary reconciliation, saved uncertainty, and notebook
  fresh-session resume with model imports blocked.
- Post-change affected regressions: **104 passed in 72.08 seconds**.
- Notebook schema and all 18 report links/anchors verified.
- Four scientific figures inspected as PNGs; SVGs are also exported.
- HTML browser rendering was not inspected: local browser automation is disabled
  in Work settings. No browser-render or responsive screenshot claim is made.
- Notebook harness pin:
  `00feff448dca2ef7fd17143981dc915269c9bf0e`. Pinned CLI/backend/presentation
  bytes match the working sources. Notebook metadata includes stable cell IDs.
- Real report generated through the public command; a separate notebook process
  restored named paths and verified/reused the completed v2 report.
- Bounded stop/resume uses three independently sealed units. Changed evidence
  requires a new report identity; no stale output is silently reused.
- Full start-of-session baseline: **473 passed, 35 skipped, four warnings in
  1,749.41 seconds (29m09s)**. The warnings report the existing local Torch
  import failure (`libtorch_cpu.dylib` missing); the CPU report does not require
  Torch. This is the start-of-session collected suite, distinct from the
  post-change affected regression run above.

No GPU, new generation or external publication occurred. The repository changes
are committed locally; GitHub publication is not part of this execution.
