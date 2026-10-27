# Ticket 07: primary findings implementation

## Delivered workflow

CPU public commands:

- `stage0-findings-prepare`: freeze family, hypotheses and statistical procedure.
  Registration is before collection only when neither collection has sealed
  units, an attempt-start record or an active writer. Otherwise exploratory.
- `stage0-findings-run`, `status`, `verify`: 14 sealed paired comparisons and an
  immutable raw report. Verify outcomes, source snapshots and actual draw files.
- `stage0-findings-audit-{prepare,status,save,freeze,reveal}`: targeted blinded
  intended-answer packet, revisioned durable labels, exact historical reuse,
  explicit freeze/reveal and cap-preserving confirmed numeric overlays.
- `stage0-findings-report`: immutable review-status snapshots, preserving the raw
  report. Pending review and missing optional work are honest non-blocking states.

GPU notebook 12A registers the plan before collection; CPU sections 17–22 run
analysis/review/reporting. Collection, qualification and execution are not
performed by this slice. Notebook logic stays in named settings and public calls.

## Mathematical and evidence contract

- 10,000 PCG64 paired resamples, seed 42, percentile 95% intervals. Actual ID
  order and draw arrays are saved. The CPU NumPy version is recorded on analysis
  execution, independently from GPU preparation, and required on resume.
- Accuracy: flexible-v3 primary, strict alongside, gains/losses, transition IDs,
  invalid/cap counts and exact two-sided McNemar. Preservation only when lower
  bound strictly exceeds -0.05; boundary equality inconclusive. At n=150 the
  inconclusive risk is recorded before execution. Pointwise family limitations.
- Length: median of paired token differences, paired interval, shorter share and
  cap counts. No substituted difference of marginal medians or reasoning claim.
- KL: within-document mean, then document bootstrap, separated by corpus/view;
  token weighting remains separately visible. Module write differences are
  descriptive; no token/module pseudo-replication.
- Matching collection execution/runtime/batch/adapter identity is required for a
  join. Raw sources, completion receipts, copied inputs and interval derivations
  reconcile. Altered results are rejected even if their local seals are updated.
- Original raw result and each review snapshot retain JSON/JSONL/CSV, SVG/PNG,
  plot-input data, provenance and explicit links to question/document records.

## Human-review contract

- Select answer/bin changes including wrong-to-wrong extracted-number changes,
  invalid flexible endpoints and last-number discrepancies. Strict-only format
  failures are not mandatory manual review. Deduplicate exact prompt/text.
- Reuse only verified original frozen labels/revisions or confirmed-only assisted
  reports. The latter is needed for the previous Pilot 4 partial workflow. Keep
  its recorded code provenance rather than demanding that its saved UI equal
  today's repaired UI. Independently verify the frozen packet, source records,
  label chain, confirmation/proposal receipts and case-level correction result.
  Unconfirmed LLM proposals never become human labels.
- Exact prompt/response text required; no fuzzy matching. Conflicting confirmed
  numbers stop reuse. New and reused labels retain one reviewer identity (trim
  incidental surrounding whitespace in reviewer IDs only).
- Save snapshots with base revision, reject stale/unknown/invalid entries, freeze
  complete labels before reveal. Numeric labels change only those exact cases;
  cap zero remains. Absent/ambiguous labels stay visible with raw scores retained.
- Mixed overlays are not exhaustive human accuracy, reasoning validation or
  preservation claims. Raw tables remain available unchanged.

## TDD / validation record

- Public command test initially red: missing `stage0-findings-prepare`.
- Raw report/oracle slice: two tests green; review/report snapshot test then red
  for missing `report` entrypoint before its implementation.
- Review slice: five tests green. Notebook handoff test red before new cells.
- Expanded analysis/notebook suite: ten tests green in 86.29 seconds.
- Reused-reviewer test red before enforcing the approved single-reviewer identity.
  Historical frozen-audit loader and reused-identity tests then green.
- Read-only live compatibility check of the selected saved Pilot 4 confirmed-only
  assisted report succeeded; original source artifacts were unchanged.
- Final affected regression and release pin: recorded at closeout below.

## Remaining execution

Real Stage 0 source/weight execution, GPU qualification, paired collections,
scientific report and new human labels remain pending. Local fixtures are
synthetic software validation and are labelled accordingly. No GPU time,
training, source edits, Drive upload or GitHub publication is implied.

## Closeout, 27 October 2026

- Starting affected Stage 0 baseline: 49 passed in 58.13s.
- Final affected regression: **73 passed in 193.02s**, covering new findings,
  drift, greedy evaluation, qualification, workflow, and existing behavior-findings
  reconciliation. This is an affected-suite result, not a full-suite claim.
- Code compiles; notebook cells parse with no outputs; diff whitespace check clean.
  SVG and PNG figure inspected, including the corrected legend placement.
- Local synthetic raw/reviewed fixtures and verification metadata are preserved
  under `artifacts/validation/stage0-ticket07/`. No Drive upload or GitHub push.
- All primary Tickets 01–07 are software-complete; conditional/optional branches
  remain separate. No Stage 0 scientific execution is established by this record.

- Notebook scientific implementation pin: `ebad4290a30e780ade3a0ecace4cb887e791024c`.
