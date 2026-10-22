# 04: Run the targeted blinded audit through frozen labels and reveal

**What to build:** A usable single-reviewer workflow that selects the agreed
cases, exports a genuinely blinded intended-answer packet, saves review progress,
freezes complete labels, and separately reveals gold/scorer comparisons for
error attribution. Demonstrate the whole round trip with CPU fixtures.

**Blocked by:** 02 — paired per-question transitions and diagnostic outcomes.

**Status:** resolved

**Parent:** Behavioural gain breakdown specification.

**Execution:** Software round trip verified on fixtures. Real packet prepared;
all 215 intended-answer labels remain pending. No real freeze/reveal completed.

- [x] Per method, select every flexible-v3 endpoint-bin change, every pair with
      a flexible-invalid endpoint, and every pair with an endpoint numeric-value
      or validity disagreement between flexible-v3 and last-number extraction.
- [x] Add 20 remaining stable pairs per method, or all if fewer, from sorted
      IDs with a separate local seed-42 generator. Save pools, selections and
      inclusion reasons; strict-only format failures are not all mandatory.
- [x] Display mandatory/control pair counts and unique-response workload before
      review. Do not silently truncate mandatory cases to meet a guessed budget.
- [x] Deduplicate identical verified shared responses with all source references
      preserved. Export opaque response IDs, question and response only; hide
      method/checkpoint, endpoint grouping, gold, scorer decisions, selection
      reasons and revealing filenames/order from the reviewer packet/view.
- [x] Support intended numeric answer, absent, ambiguous and pending labels,
      evidence text and notes. No line-by-line reasoning-validity label is required.
- [x] Draft labels save incrementally and survive fresh notebook sessions.
      Reject unknown/duplicate IDs, malformed entries and wrong packet identities.
- [x] Explicit complete-label freeze precedes reveal; pending labels block a
      completed freeze. Preserve the immutable packet/label receipt.
- [x] Stage two compares the frozen intended answer with gold and scorer outputs
      and supports intermediate-number, missed-format, conflicting/ambiguous,
      cap/truncation, other and uncertain error explanations with evidence.
- [x] Post-reveal corrections use a separately identified, reasoned revision
      explicitly marked as exposed; frozen labels are never silently overwritten.
- [x] Report selected-case counts and examples only, with targeted selection and
      partial blinding disclosed. Do not claim exact whole-cohort human accuracy,
      reasoning-validity rates or inter-reviewer agreement.
- [x] Public-command tests cover exact selection, controls, deduplication, actual
      packet-field omission, label binding/resume, invalid freeze attempts,
      immutability, reveal ordering and versioned corrections.
- [x] Keep software round-trip completion distinct from the real reviewer's
      outstanding labels; do not invent human annotations to finish the ticket.

## Comments

2026-10-22: User authorized implementation of Ticket 04. Real labels remain human work.

2026-10-22: Software implemented, 24 new audit tests and 85 affected regressions
pass. Real packet prepared under `pilot4-behaviour-audit-2026-10-22-v1`;
215 unique responses, 0 labels. See `../execution-ticket04.md`.
