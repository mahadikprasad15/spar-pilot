# Ticket 03 preflight

Prepared 15 October 2026. **Approved by Prasad on 15 October 2026 through the explicit preflight-defaults reply. Phase 2 is authorized.**

Design reviewed: `.scratch/pilot5-readable-traces/ticket03-scoring-design-v1.1.md` (byte-identical copy of supplied v1.1; SHA-256 `1d2bcdfaa1fb4275b384f18eb1a8fd3ce3b65927637bda0468f202f4827d7636`). Amendments A1–A11 override the original sections. The defaults below were approved in the explicit preflight reply; real reference artifacts still require preparation and review.

## 1. Ticket 03 verbatim

Path: `.scratch/pilot5-readable-traces/issues/03-freeze-vocabulary-and-reveal-policy.md`  
Commit: `196e9b7fd147792cf2f7a12f2839e32f35a04418`  
File SHA-256: `39ae4cd8a269daa346ae1af99460fa0d8dd4da2291ebf1dd7e36c8942556aa36`

The following is the complete file at that commit, including its implementation-progress notes:

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

## 2. Mapping table

The matching ticket text below is quoted exactly. Status describes the amended design, not the superseded v1.0 wording.

| Design section | Matching ticket text (quote) | Status | Note |
|---|---|---|---|
| §2 Inputs; A3/A6/A7 | “Public vocabulary preparation uses the fixed training-solution and background source, with explicit counts/smoothing/exclusions/ties and no learned token lists as input.” | design adds | Adds primary question reference, GRPO-completion reference and a separate exactly-2M-token background. Real vectors/lenses are downstream inputs, not requirements to run synthetic Ticket 03 tests. |
| §3 Normalization | “with explicit counts/smoothing/exclusions/ties” | design adds | Matching changes from the earlier token-ID overlap proposal to NFKC/strip/lower strings; preserve raw IDs and decode evidence too. |
| §4 References / A10 | “Public vocabulary preparation uses the fixed training-solution and background source” | design adds | Three references, informative-prior log-odds, minimum count, positive-z rule, sensitivity variants and hash-sealed lists. |
| §5 Scores / A10 | “Freeze reveal-only decisions: vocabulary, null/correction/threshold rules” | design adds | Defines overlap, word/format decomposition and soft-rank; shorter lists have an amended denominator. Zero-valid-list behavior remains open. |
| §6 Conditions / A2/A4/A5/A6 | “Reference the primary/capture contract; do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.” | conflict | A2 preserves the existing pooled primary. A6 adds a base-covariance capture requirement that must be carried to capture-owner tickets. A4's real-null timing still conflicts with lens/capture dependencies. |
| §7 Bootstrap / A5/A7/A8 | “Freeze legacy-150/new-50 robustness analysis and its uncertainty/sample-size interpretation before reveal.” | design adds | Specifies paired draws, stratified results, N-matched check and SNR. Preserve B=1,000 for §7 primary-score intervals versus B=2,000 for A5 C3 and A8 strata unless amended. |
| §8 Grading / A9/A10 | “plus blinding/grade rubric” | design adds | Two modes, isolated sessions and synthetic package/agreement testing. Actual grading/unblinding stays with Ticket 10. Exact baseline list identities remain to enumerate. |
| §9 Outputs / A1 | “Capture does not wait for vocabulary approval.” | conflict | Original output tree and results-oriented report require a canonical-root mapping and phase labels. Ticket 03 publishes procedures/references, not production learned results. |
| §10 Tests / A11 | “Every allowed analysis is a function of the retained-data capability contract.” | design adds | Adds independent numerical oracles and synthetic public-workflow tests. Tests of implementations are distinct from actual lens/capture qualification and inferential calibration. |

The ticket additionally requires named exposure history, exploratory labeling after early reveal, support-capability checks and a scoped supported-operation barrier. These remain requirements even though the new design does not repeat all of them.

## 3. Conflicts

### 3.1 Existing ticket history is stale, not an execution failure

Ticket progress says: “Ticket 02's real pinned-data run remains unconfirmed.” The later Ticket 02 record and the user-pasted public verifier say `corpus_complete: true`, 200 examples, exit code 0. Treat that sentence as historical; update it when this preflight is approved. The underlying inputs have now also been downloaded and checked below.

### 3.2 New references and background amend the previous design

Ticket: “The background vocabulary uses the actual declared corpus; this is why the corpus artifact is a genuine dependency.” Design A3: “until the total reaches **exactly 2,000,000 tokens**.” These can be reconciled by binding a separately prepared background to the frozen 200-document exclusion set. Do not substitute the old 200 prefixes for the new 2M-token background or rebuild the measurement cohort.

Ticket: “fixed training-solution and background source”. Design §4.1: `ref_questions` is “**Primary.**” Gold and GRPO-output references become secondary. Approval adopts this explicit scientific amendment; no vocabulary has yet been frozen under either rule.

### 3.3 Primary/capture ownership remains separate

Ticket: “do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.” A2: “**Primary: one pooled vector**” agrees with ADR 0011. A6: “accumulate Σh and Σhhᵀ in float64” adds capture evidence. Record that requirement for Tickets 04–07 and budget it there; Ticket 03 must not silently change capture execution or retained precision. A7's five per-document position vectors are already within planned block-14 retained capability; actual storage/quantization qualification remains with capture.

### 3.4 Real nulls cannot all precede their inputs

A1: “using synthetic vectors only. It does not compute learned readouts or scientific results.” A4: “The full null distributions (isotropic, and activation-shaped if available) are computed and frozen in Ticket 03, before any learned vector exists.”

Real isotropic lens scores require qualified real lenses; the activation-shaped null requires later base statistics. Proposed resolution, requiring approval: Ticket 03 freezes the generator, seeds, normalization, scoring and synthetic tests. Ticket 08/09 produces sealed real null score arrays from the frozen procedure and qualified inputs before publishing learned scores. No learned norm is used. If a qualified real lens becomes available earlier, base-independent isotropic computation may be a separately recorded operation; it is not a prerequisite to marking the procedure implementation green.

### 3.5 Output root and report phase

§9: “Write all files under `runs/pilot-5-readable-traces/analysis/ticket03/`”. Existing canonical layout uses `<ARTIFACT_ROOT>/runs/pilot-5/<experiment>/...`. Proposed path mapping is in section 6; it must be explicit.

§11 asks for “Results, primary”, “The token lists themselves”, and the frozen-reference contents. A1 instead says “It does not compute learned readouts or scientific results.” Ticket 03's implementation report must label future result sections **not run / owned by Ticket 08–10**, not fabricate values or wait for capture. Real reference lists appear only after the approved builder is actually run; a fixture list is never presented as the real frozen vocabulary.

### 3.6 Grading supplement remains optional

Ticket includes “plus blinding/grade rubric”; A1 implements package/agreement algorithms with synthetic data. Existing Ticket 10 owns actual grading. Build and test reusable algorithms here, execute later with explicit grader submission approval and session isolation; no API calls in this ticket.

## 4. Reuse plan

| Existing module / function | Reuse scope |
|---|---|
| `pilot_eval/trace_corpus.py`: `load_corpus` | Verify the existing frozen measurement corpus and bind exclusion identities. No retokenization/resampling of the 200 documents. |
| `pilot_eval/trace_corpus.py`: `PinnedRawTokenizer`, `HFCorpusDependencies.load_tokenizer`, `stream_fineweb` | Load exact pinned tokenizer JSON and dataset revision without model imports. Extend normalization inventory to identify all added token IDs, not only special IDs. Retain tokenizer bytes/hashes. |
| `pilot_eval/trace_review.py`: `record_exposure`, `inspect_review`, `prepare_review`, `validate_analysis_requests` | Preserve prior exposure and capability checks; add versioned reviewed plans/freeze checks. Existing drafts remain drafts; do not turn empty fields into approval. |
| `pilot_eval/trace_session.py`: `TraceSession.artifact`, restore/selection conventions | Root-bounded paths, explicit source selections and cheap restore; no repeated broad checkpoint audit for text-counting work. |
| `pilot_eval/sft.py`: `read_artifact`, source schema used by `load_sft` | Exact 512 question/gold fields and saved identity. Avoid invoking full training validation solely to tokenize text; verify the actual text-parent seals and state verification scope. |
| `pilot_eval/grpo_training.py`: `verify_training` | Reuse full verification where appropriate to establish accepted history. Do not replace it with a line count; the preflight only checked bounded downloaded text/config relationships. |
| `pilot_eval/workflow.py`: `_hash`, `_save_frozen`; `pilot_eval/training.py`: `file_hash`, `run_lock`; `pilot_eval/run.py`: `_write_json`, `_write_state` | Canonical JSON identities, byte hashes, immutable plans, atomic outputs, single-writer locks and explicit lifecycle logs. |
| `pilot_eval/notebook_session.py`: `NotebookSession.run`; `pilot_eval/cli.py`: `main` | Public notebook/CLI seam, streamed logs, liveness versus completion distinction and owned-child exit handling. |
| Existing lens code | No actual logit/Jacobian implementation exists in `pilot_eval/` yet. Do not claim it can be reused. Ticket 03 takes an injected, explicit linear-readout interface with tiny independent CPU matrices; qualified real adapters belong to Tickets 05/08. |

Planned new modules (implementation only after approval):

- `trace_vocabulary.py`: normalized token inventory, background preparation/counts, enrichment and reference freeze.
- `trace_scoring.py`: normalized unique-list extraction, overlap/soft-rank and null procedures.
- `trace_statistics.py`: paired whole-document bootstrap, N-curves, SNR and correction; no raw capture logic.
- `trace_grading.py`: blinded package, grade validation/agreement and guarded key joining; no API submission.
- Extend `trace_review.py` for reviewed procedure schemas and freeze/reveal enforcement; do not duplicate capture ownership.

Proposed public commands extend `traces-review-*` for prepare/inspect/freeze and add named vocabulary preparation/verification operations. Tests remain at the approved saved-workflow seam, plus independent tiny CPU numerical oracles. Model, tokenizer, lens and dataset access are explicit external dependency boundaries. Data stages and expensive resampling must persist resumable units, not silently restart a large all-or-nothing stage.

## 5. Inputs found

Artifact root in Colab: `/content/drive/MyDrive/SPAR/spar-pilot/artifacts/`. Paths below are relative to that root unless stated otherwise. Local evidence snapshots: `artifacts/validation/pilot5-ticket03-preflight/inputs/`. Exact downloaded bytes were hashed locally; raw large snapshots will not be committed. Metadata/hashes are in `input-inventory.json`; bounded checks are in `input-checks.json`.

| Input | Exact source path / artifact | Bytes | SHA-256 of payload bytes | Status / scope |
|---|---|---:|---|---|
| I1 tokenizer | `runs/pilot-5/pilot5-readable-traces-v1-raw-corpus-v1/corpus/inputs/tokenizer/tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` | Found on Drive; byte hash matches the earlier successful Colab receipt. Revision in corpus agrees with I1. |
| I1 tokenizer config | `runs/pilot-5/pilot5-readable-traces-v1-raw-corpus-v1/corpus/inputs/tokenizer/tokenizer_config.json` | 7,305 | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` | Found. Same pinned model/tokenizer revision; no model weights downloaded. |
| I2 questions / I3 gold | `plans/pilot2-sft-fp32-l4-batch8-v1/training.items.json` | 3,879,974 | `f27d0d75740438adafcdc3bf56104ac6fee06f80784366e5fc4e1a2c5850d260` | Found 512 records. Fields are `question` and `gold`; `gold` preserves the SFT answer text. Both logical view hashes are recorded separately in the JSON inventory. |
| I2/I3 parent config | `plans/pilot2-sft-fp32-l4-batch8-v1/sft.config.json` | 16,257 | `e70679ab52600e5cca8c33daa1ee7c6206ed309daa5796346a9bc97c24da87e1` | Downloaded training payload matches its canonical content hash. |
| I4 accepted completions | `runs/pilot-4/Qwen--Qwen2.5-1.5B-Instruct/gsm8k/train-512-seed-42/rank1-float32/pilot4-grpo-fp32-v1-arm-v1/results/responses.jsonl` | 19,461,879 | `ce0a340d03e23b2ffeda163825ebdc7146b52eb6b9978fbcae37f865403e76ef` | Found 4,096 rows / unique draw IDs; use `text`, not prompt-plus-completion `token_ids`. All 512 source indices and gold fields agree with SFT records. Full checkpoint/seal revalidation is separate. |
| I4 parent config | `plans/pilot4-grpo-fp32-v1-arm-v1/grpo.training.json` | 17,536 | `958123b24611e667e474b1d1d87e4d68ab03d63128ef0c8a2028c2cedd712958` | Found; frozen parent chain is not fully downloaded or revalidated in this preflight. |
| I5 exclusion parent | `plans/pilot5-readable-traces-v1-raw-corpus-v1/traces.corpus.json` | 8,474 | `8729de8f9e5c8bf0b27c3f95e318222b3a54c051a233e4be2276a4ac5ca48a11` | Found; measurement corpus identity is fixed and unchanged. |
| I5 measurement exclusions | `runs/pilot-5/pilot5-readable-traces-v1-raw-corpus-v1/corpus/inputs/documents.jsonl` | 1,021,775 | `ee9c9bdbb55f9db596e3edeb47ba9d14ab5e91d3ab4b28d8c834e3e9e9e0f7c2` | Found 200 rows, each with 128 content positions; canonical row hash agrees with corpus config. These are exclusions, not the new background. |
| I5 new background | Not created; proposed prepared background under the approved analysis root | — | — | Missing, expected new CPU preparation. Exactly 2M raw token IDs required; candidate scan bound awaits approval. |
| I6 learned mean vectors | No Pilot 5 production capture run exists | — | — | Missing, expected downstream Ticket 07 output. No old Pilot 4 aggregate is substituted. |
| I7 final-norm gain / unembedding | Pinned model revision named by I1; no qualified local lens weights/interface yet | — | — | Not materialized/qualified; belongs to lens qualification, not fabricated from tokenizer metadata. |
| I7 community lens | `qbz506/prabodha-lenses/qwen2.5-1.5b-instruct.pt` | — | — | Specified remote identity only. Exact artifact revision/bytes/SHA-256 and compatibility remain Ticket 05 work. No unsafe/unqualified tensor deserialization performed. |
| I8 per-document positions/pooled deltas | No Pilot 5 capture yet | — | — | Missing, expected downstream. A7 requires separate positions 0–4, not only pooled vectors. |
| I9 random-adapter mean delta | Existing random adapter source was audited, but no Pilot 5 raw-cohort delta has been captured | — | — | Missing delta, not missing adapter. Do not regenerate the adapter or substitute Pilot 4 means from another cohort. |

Source files are identified by exact Drive IDs/URLs in the JSON inventory. No signed download credentials are stored there. The source design and ticket snapshot have the hashes stated above.

Checks performed now: 512 training rows; 4,096 unique accepted draw IDs; GRPO rows cover 512 training indices and their gold fields agree with SFT; 200 measurement rows and 128 content positions each; tokenizer hashes agree with the user's Colab verifier. The shared model/tokenizer revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the corpus records FineWeb revision `9bb295ddab0e05d785b879661af7260fed5140fc`.

Important schema distinction: `grpo.training.json` does not itself contain `training_items_sha256`. Its absence is not a failed hash comparison; full linkage follows its frozen-parent contract. This preflight checks downloaded source-index/gold consistency without claiming that it repeated the complete sealed training audit. Question/prompt consistency result is preserved verbatim in `input-checks.json`.

Session regression already completed before this preflight: 306 passed, 35 skipped. Subsequent affected set: 48 passed. Those are prior run results, not a new claim that the proposed scoring implementation passed tests; no such code exists yet. This phase changed documentation/evidence only.

## 6. Open decisions

There are no remaining literal `[DECIDE]` choices after A2 overrides §6.1: the primary is pooled. The table below records residual definitions and conflicts; proposed defaults become binding only when Prasad approves this preflight.

| Decision | Proposed default for approval | Consequence / alternative |
|---|---|---|
| Finite background sampling frame | Scan the first **10,000 source documents** in the pinned stream, reconstruct full-text exclusion identities for all 200 measurement documents, deduplicate with SHA-256 of `' '.join(text.split())`, drop <64-token documents, shuffle eligible source indices with a separate `random.Random(20261019)`, then consume first-1,024-token slices until exactly 2M raw token IDs. Stop and ask if insufficient. | Bounded CPU work; no full-dataset shuffle or silent pool expansion. No corpus-content outcomes used. Record original and normalized full-text identities, ordered indices, per-document retained token counts and final partial-document cut. |
| Null timing / dependencies | Freeze the real procedure here and test on synthetic matrices; seal real isotropic null scores after real lens qualification and real activation-shaped null scores only after base moments exist, before learned-score publication. | Resolves A1/A4 conflict; actual distributions are not invented or a prerequisite to procedure implementation. Unit-normalize nonzero draws; zero/invalid covariance cases are unavailable with diagnostics. |
| Artifact root | `<ARTIFACT_ROOT>/runs/pilot-5/<experiment>/analysis/ticket03/`, with the §9 subdirectories; immutable plans under `<ARTIFACT_ROOT>/plans/<name>/`. Keep `grading_key/` outside exported `grading/`. | Matches existing experiment selection/resume paths. Original literal §9 root is an alternative requiring an explicit mapping. No ad-hoc root-level arrays. |
| Zero valid list strings | Score undefined (NaN + reason), preserve raw ranked candidates/valid count, and do not report a p-value from an undefined observation. Identical rule for nulls; if null draws are invalid, report insufficient null evidence rather than silently resampling. | Handles 0/0 without inventing a zero-relevance score or changing the fixed null sample size. |
| Constant-label agreement | Compute κ whenever its denominator is defined; always flag a constant grader and report percent agreement. If chance agreement is 1, κ is undefined and agreement/reason remain. | Corrects A10's statement that any constant grader makes κ undefined. If the user instead wants agreement-only for all constant-grader cases, label that as policy, not a mathematical undefined result. |
| C3 inferential status | Implement A5 exactly, preserving ties in both tails. Add bounded synthetic null/shift calibration diagnostics; do not claim exact test calibration from a formula unit test. No new acceptance tolerance is selected here. | Any inferential concern must be reviewed before scientific significance claims. Do not silently swap in another p-value. Numerical unit-oracle tolerance remains the design's 1e-9. |
| Baseline inputs and grading inventory | Fix baseline input to the same 200 FineWeb measurement documents. Proposed grading inventory: learned SFT/GRPO/random × 6 vectors (positions 0–4 + pooled) × 2 lenses = 36; base once + adapted SFT/GRPO × 5 positions × 2 lenses = 30; 20 isotropic null lists per lens = 40; **106 lists per mode**. Random adapted-activation baseline omitted unless separately requested. | Removes outcome-dependent source switching and resolves which adapted arm the “two baselines” phrase refers to. Naming and relevance use fresh isolated sessions; key not exported with lists. |
| Readout sign, pooling and list identity | Positive-score descending lists define primary overlap. Negative lists may be retained later as labeled secondary outputs, not added to the primary family. Per-position max remains secondary. Use token-ID tie break before normalized-string dedup. | The current design's “top” ranking does not explicitly define a combined positive/negative family. Never search either sign and silently select the better one. |
| Random draw reuse and salts | Reuse each declared random-vector draw across references/arms when its site/null/form contract matches; record draw IDs, seed and exact RNG/version. Persist grading salt once in the private key tree; inject a fixed test salt for byte-determinism tests. | Avoid repeated null forward/readout work and preserve pairing. No fresh salt or changed seed on resume. Qualified numeric precision/runtime is recorded; actual cross-runtime bit identity is not promised. |
| Remaining statistical conventions | Preserve specified seeds/counts. §7 primary-score bootstrap B=1,000; A5 C3 and A8 stratum bootstrap B=2,000. Use average ranks for Spearman ties; the combined-count support includes valid strings with yA+yB≥5 even if yA<5. At N=200 without replacement all subsets are the full cohort; report this degeneracy. | No silent unification of draw counts. Sensitivity α0=100/10,000 and max-position results are secondary, not additional primary tests. State quantile/numerical library conventions in the frozen config. |
| Primary-family availability | Keep the declared four-test family. Missing/incompatible Jacobian qualification blocks the originally planned primary release and requires a separately approved scope amendment; it does not silently become a two-test primary experiment. | Preserves community-lens approval and existing lens-failure stop rule. Synthetic tests can still finish while lens qualification is pending. |

Capture follow-through: A6's 1,536×1,536 FP64 second-moment matrix is 18 MiB, plus the first-moment vector and counts; GPU compute/transfer cost still requires profiling. Do not describe the whole operation as free because the output array is small. Covariance construction, PSD/numerical handling and retained-precision checks must be qualified by the capture/lens owners before using that secondary null. Missing moments disable only that secondary analysis with an explicit reason.

Exposure follow-through: no Pilot 5 learned lens outputs have been produced by this workflow. Do not infer the user's complete prior exposure history; record it explicitly through the existing exposure interface. The supported-operation reveal barrier is not an access-control claim against direct file access.

**Approval boundary:** approve or amend this preflight before Phase 2. This document does not mark Ticket 03 resolved, launch capture, download model/lens weights, score learned activations, call a grader API or publish scientific results.

PREFLIGHT READY — waiting for approval

Approval receipt: “Approve the preflight defaults”. All section 6 defaults are accepted; real-input, lens and capture qualification remain independent gates.
