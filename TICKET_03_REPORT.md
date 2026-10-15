# Ticket 03 implementation report

15 October 2026 · **Implementation complete; real reference preparation/review pending.**

This report separates CPU fixture verification from scientific execution. No model inference, learned readout, real null distribution or grader API request was run for this implementation.

## 1. Ticket 03 verbatim and approved decisions

Original ticket at implementation source commit `d7d1d24a67c7fa812113ff99ec5e4112c3478780`:

````markdown
# 03: Freeze vocabulary and inference commitments before outcome reveal

**What to build:** Review and freeze the task-enriched vocabulary and outcome-dependent analysis/grade rules while capture qualification and lens inspection proceed independently. The background vocabulary uses the actual declared corpus; this is why the corpus artifact is a genuine dependency.

**Blocked by:** 02: Prepare and freeze the raw corpus.

**Status:** claimed (partial safeguards implemented; scientific approvals pending)

- [ ] Public vocabulary preparation uses the fixed training-solution and background source, with explicit counts/smoothing/exclusions/ties and no learned token lists as input.
- [ ] Freeze reveal-only decisions: vocabulary, null/correction/threshold rules, sample-size draw counts/seeds and SNR estimator within the allowed retained-data family, plus blinding/grade rubric. Reference the primary/capture contract; do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.
- [ ] Prior exposure records cover earlier pilots and first looks: named reviewer/decision-maker, which outputs or token lists were seen, when known, and unknown history. Scalar Pilot 4 plots are distinguished from learned lens lists; do not invent an earlier readout.
- [ ] Freeze legacy-150/new-50 robustness analysis and its uncertainty/sample-size interpretation before reveal.
- [ ] Missing approval blocks outcome publication; an existing-data preview or bounded adapted preview cannot bypass this barrier.
- [ ] Outcomes revealed before freeze are marked exploratory and prior exposure is recorded; they cannot be relabelled confirmatory by a later freeze.
- [ ] Every allowed analysis is a function of the retained-data capability contract. Requests for unsupported per-position uncertainty at other sites are rejected before analysis rather than silently creating a dependency on capture or promising a rerun-free result. Capture does not wait for vocabulary approval.
- [ ] The notebook reveal barrier prevents accidental output through supported operations, not deliberate file access. Record its scope and any known bypass/exposure.

## Implementation progress · 2026-10-15

- Public CPU exposure-recording/inspection and corpus-bound draft preparation are implemented in `pilot_eval/trace_review.py` and the CLI. Red tests preceded each committed slice.
- Immutable event IDs preserve prior history; changed payloads fail verification. Scalar results, learned token lists and other outputs are explicitly distinguished. Unknown history/date must be stated.
- Availability validation rejects per-position example analyses outside blocks 14 and 17–21 and rejects all-site pooled-window requests other than positions 0–4. Capture ownership is unchanged.
- Review drafts bind verified corpus inputs and keep all six scientific sections empty. They never grant reveal. No production freeze, vocabulary ranking, readout consumer or exploratory/confirmatory classification has been implemented.
- Prasad has been asked to approve the proposed vocabulary rule, review the remaining settings and confirm the real corpus run. No answer/approval is inferred from elapsed time.
- Ticket remains unresolved; Ticket 02's real pinned-data run remains unconfirmed. See `docs/pilot5-prereveal-review.md` for exact boundaries.

````

That historical progress text is superseded by the implementation below. The approved user design is `.scratch/pilot5-readable-traces/ticket03-scoring-design-v1.1.md` (SHA-256 `1d2bcdfaa1fb4275b384f18eb1a8fd3ce3b65927637bda0468f202f4827d7636`). Amendments A1–A11 override §§1–11. Prasad explicitly approved the defaults in `TICKET_03_PREFLIGHT.md`; the complete decision record is retained there and in the spec amendment.

Approved primary: raw form, block 14, checkpoint 64, positions 0–4 pooled into one vector. Questions are the primary reference; gold solutions and all accepted GRPO completions are secondary. Background uses the pinned first 10,000 FineWeb documents, excludes measurement and duplicate text identities, and selects exactly 2M raw tokens with local seed 20261019. Normalization is NFKC/strip/lower; all added/special IDs and replacement-character strings are excluded. Informative-prior log odds uses α0=1000 (sensitivity 100/10000), yA≥5, z>0, top 200. Readouts select 20 unique strings from at most 1,000 raw ranked IDs.

The procedure records isotropic unit-Gaussian nulls (1,000, seed 20261015), paired bootstrap, Holm families, SNR, N-curves and matched legacy/new checks. Missing Jacobian qualification blocks the original primary family. Grading is optional and local: 106 lists per mode, naming before disclosed-task relevance, with keys outside the public package and settlement before condition joins. Unknown exposure remains unverified/exploratory. No scientific threshold was relaxed to pass tests.

## 2. Implemented files, interfaces and commit

Implementation source: `d7d1d24a67c7fa812113ff99ec5e4112c3478780`. The notebook pins this source commit; subsequent release metadata/docs do not change it.

| File | Public responsibility |
|---|---|
| `pilot_eval/trace_reference_workflow.py` | `prepare_reference_run`, `verify_reference_run`, approved procedure and exact source lineage |
| `pilot_eval/trace_vocabulary.py` | Native tokenizer inventory, counts/log odds, background resume, reference freeze and review |
| `pilot_eval/trace_scoring.py` | `score_vector`, `ScoringSession`, normalized lists, overlap/soft-rank, null generators and zero gate |
| `pilot_eval/trace_statistics.py` | Null/C3 p-values, Holm, paired bootstrap, SNR, N-curves, stratum checks, sealed draw runner |
| `pilot_eval/trace_grading.py` | Blind packaging, agreement, naming isolation, raw-input persistence and settlement reuse |
| `pilot_eval/cli.py` | `traces-reference-prepare`, `traces-reference-verify`, `traces-reference-review` |
| `notebooks/pilot-5-colab.ipynb` | Shared notebook cells 10–13: restore → prepare → verify/preview → explicit review |

The saved workflow binds source bytes and implementation identities, verifies completed artifacts before reuse, seals background chunks every 250 documents, and repeats only unsealed scan work. Canonical config is also saved under `artifacts/plans/<experiment>-overlap-v1/overlap.config.json`. Reference review records `state: frozen`. A verified scoring session hashes parents at entry/exit and checks file-version guards before repeated readouts, avoiding a full parent scan for every null draw.

## 3. Reconciliations and deviations

All scope reconciliations were explicitly approved in preflight. Three references and a separate 2M-token background supersede the original single-solution-reference wording. Canonical output maps to `artifacts/runs/pilot-5/<experiment>/analysis/ticket03/`.

A1 limits this implementation to reusable algorithms and synthetic verification. Consequently real lens-specific null scores are deferred until qualified lens/base inputs exist; the generator, seeds, procedure and math are implemented now. Reference preparation is runnable but has not yet been executed on the real background. Qualified capture and lens integration belong to later tickets. No absent result is filled with fixture output.

C3 uses the requested bootstrap-tail formula. A bounded synthetic diagnostic is implemented, but it does not establish a calibrated exact test or nominal error rate for the eventual ranking statistic. The report must retain this limitation. Raw grade and naming inputs are stored in independently sealed per-input/session runs so a resumed analysis preserves earlier settlements.

## 4. Verification

Session-start full regression before these new slices: **311 passed, 35 skipped**, 132 warnings, 1033.80 seconds. Final affected regression: **73 passed** (see saved release log for timing). These are distinct runs; 73 is not a claim that the whole enlarged repository suite was rerun.

Each slice used a failing test before implementation, followed by a green commit. All tests are CPU-only and use synthetic data/tiny tokenizer fixtures; no model download. The public reference workflow includes a 10,000-document synthetic stream and exactly 2M selected raw token IDs, fresh-process CLI verify/review, and actual notebook prepare/verify-cell execution against saved fixtures.

| Design gate | Result / evidence |
|---|---|
| §10.1 Normalization, exclusions, 20 unique strings | PASS: vocabulary and scoring tests; duplicate variants merge, special/partial bytes are excluded |
| §10.2 Independent log-odds oracle to 1e-9; rare/frequent ranking | PASS: independent numeric literals, minimum counts and positive-z rules |
| §10.3 Add-one contrast | PASS: same toy counts demonstrate reversed rare-token preference |
| §10.4 Freeze required; byte/parent tampering refused | PASS: freeze, reviewed-scoring and parent-change tests |
| §10.5 Seed determinism | PASS: null/bootstrap/background/package and saved-chunk replay tests |
| §10.6 Plus-one null p-value | PASS: explicit hand case above all null draws |
| §10.7 Matched maximum over positions | PASS: null and secondary-bootstrap tests |
| §10.8 Package blindness | PASS: only list_id/tokens public; key separate; exact 106-condition inventory |
| §10.9 κ | PASS: identical/independent labels plus constant/undefined edge cases |
| §10.10 Exact step-zero | PASS: exact-zero finite gate rejects even a small nonzero delta |
| A11 scale invariance | PASS: two synthetic linear lens callbacks, positive scaling 7.3 |
| A11 C3 formula | PASS: literal synthetic d* distribution and bounded reproducible diagnostic |
| A11 degenerate cases | PASS: zero vectors/lists, short lists, constant soft-rank, undefined SNR and κ cases |
| A11 exact background | PASS: 2M synthetic tokens, exclusion/dedup, interruption/resume and tamper refusal |

Additional recovery tests cover fixed accumulation order for identical N-curve memberships, incomplete/NaN null-family refusal, persisted grader inputs, settled-analysis reuse, naming-session receipts, code-identity changes and the outer frozen lifecycle status. Logs and metadata: `artifacts/validation/pilot5-ticket03-implementation-2026-10-15/`.

## 5. Frozen references

**Real reference freeze hash and top30.md: not produced yet.** No toy tokens are presented as real task-enriched vocabulary. Notebook 10–12 builds and previews all three references; cell 13 records review only after inspection. Review short vocabularies and possible dominance by names, format fragments or calculator notes before accepting. A changed reviewed freeze requires a new identity.

## 6. Inputs and counts

Preflight source snapshots (metadata retained; large input files are not committed):

| Input snapshot path | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/validation/pilot5-ticket03-preflight/inputs/training.items.json` | 3,879,974 | `f27d0d75740438adafcdc3bf56104ac6fee06f80784366e5fc4e1a2c5850d260` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/grpo-responses.jsonl` | 19,461,879 | `ce0a340d03e23b2ffeda163825ebdc7146b52eb6b9978fbcae37f865403e76ef` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/tokenizer_config.json` | 7,305 | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/traces.corpus.json` | 8,474 | `8729de8f9e5c8bf0b27c3f95e318222b3a54c051a233e4be2276a4ac5ca48a11` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/documents.jsonl` | 1,021,775 | `ee9c9bdbb55f9db596e3edeb47ba9d14ab5e91d3ab4b28d8c834e3e9e9e0f7c2` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/grpo.training.json` | 17,536 | `958123b24611e667e474b1d1d87e4d68ab03d63128ef0c8a2028c2cedd712958` |
| `artifacts/validation/pilot5-ticket03-preflight/inputs/sft.config.json` | 16,257 | `e70679ab52600e5cca8c33daa1ee7c6206ed309daa5796346a9bc97c24da87e1` |

Original Drive IDs/URLs and validation outcomes are in `artifacts/validation/pilot5-ticket03-preflight/input-inventory.json` and `input-checks.json`. These snapshots establish inspected text/config identities; they do not constitute a fresh validation of every optimizer/checkpoint payload.

Observed preflight structural counts: **512** unique SFT training rows; **4,096** unique accepted GRPO draws covering the same 512 sources with matching prompts/gold; **200** sealed measurement documents, each retaining 128 content positions. Questions/gold refer to the exact training set, not held-out answers. GRPO texts include correct and incorrect completions. The production builder also checks the saved GRPO response completion seal.

Tokenizer/model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. FineWeb revision: `9bb295ddab0e05d785b879661af7260fed5140fc`. Actual normalized A-token counts and real background counts have **not yet been measured**. B=2,000,000 is the approved target and the fixture-tested requirement, not an observed real-run count.

## 7. Primary results

**Not run; Tickets 08–09 own real qualified readouts and analysis.**

| Family | Required rows | Observed overlap / null percentile / raw p / Holm p / CI / SNR |
|---|---|---|
| C1/C2 | SFT and GRPO × logit and Jacobian lenses (4) | Not available |
| C3 | GRPO minus SFT, per lens (2) | Difference, interval and Holm p not available |

No claim about task relevance, SFT versus GRPO, or mechanism follows from passing implementation tests.

## 8. Secondary results

**Not run.** Planned: separate positions/max-over-five, other references, word/format-number decomposition, soft-rank, supported sites/checkpoints/forms, isotropic and optional activation-shaped nulls, N-curves and matched legacy/new distributions. Unsupported retained-data requests are rejected. Legacy/new membership is not a randomized exposure treatment; its comparison is descriptive. Optional assistant form remains separate.

## 9. Learned token lists

**Not produced.** Actual raw IDs, decoded repr strings, normalized strings and valid-list denominators will be retained with qualified readout identity. Zero vectors produce an explicitly undefined readout rather than a fabricated top-token interpretation.

## 10. Grading package

No real grading package or grade evidence exists. Synthetic packaging tests pass for **106 lists per mode**: 36 delta lists, 30 base/adapted activation lists and 40 null lists. Public lists omit conditions; private salted keys are stored separately. Actual file hashes become available only after package creation. No grader API calls were made. Raw labels, naming choices, isolation attestations and reviewer settlements are persisted before accepted condition-level results; attestations alone do not prove isolation.

## 11. Open execution gates and limits

1. Run fresh CPU notebook **1–2 → 10–12**; inspect real top30 previews and actual counts. Then explicitly fill/run **13**. Send the verification summary and preview for review.
2. Qualify capture and the community Jacobian lens; request approval before fitting any replacement lens. No silent fallback or reduced primary family.
3. Capture durable per-document block-14 positions 0–4; optional base moments from positions 1–127 need 25,400 contexts. Profile FP64 accumulation and storage in Tickets 04–07.
4. Generate and seal real null/analysis arrays from the frozen procedure and qualified inputs before interpreting learned outcomes.
5. Preserve C3 calibration caveat; determine whether stronger inferential claims need a separately reviewed calibrated procedure.

Real CPU background duration has not been benchmarked. The bounded scan emits progress every 250 sealed source documents and resumes after interruption. GPU costs are zero for this reference stage. Existing Pilot 4 training, adapters and measurement artifacts are reused, not retrained. Checksums detect accidental changes; they are not a malicious-tampering guarantee. The supported-operation reveal gate cannot prevent deliberate direct file access.
