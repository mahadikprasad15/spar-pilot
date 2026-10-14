# Pilot 5 revised ticket proposal

Status: approved for individual publication after the second supplied review corrections. Public saved-workflow test seam is approved. Scientific execution approvals remain explicit and are not supplied by ticket publication.

Implementation blockers below are contracts or artifacts, not an obligation to finish human grading or a full production run. Capture-plan ownership and reveal-policy ownership are distinct; no reveal setting may expand the frozen retention requirements. Target-runtime qualification records remain separate from offline test evidence.

## 1. Restore a clean session, audit sources and smoke-test Colab

**Blocked by:** None.

**What it delivers:** A fresh readable notebook restores Pilot 5 state, audits existing sources, declares stable named-site and report/optional-grade contracts, and demonstrates environment/storage readiness before larger slices depend on Colab assumptions.

**Done when:**

- [ ] Public restore/inventory operations on an offline fixture show actual missing/pending/completed states without starting a model or older pilot.
- [ ] The site contract distinguishes embedding output and block indices zero through 27, pre/post normalization and content/absolute positions.
- [ ] CPU restoration works without importing GPU libraries; actual GPU text-model imports and optional package conflicts are checked in a bounded fresh-process path.
- [ ] A recorded minimal fresh-Colab check mounts durable storage, writes/verifies a small artifact, disconnects/reconnects and reads it through the public restore path. Actual time is recorded, not promised.
- [ ] A changed runtime, owned active child and ambiguous source selection fail with actionable messages before inference.
- [ ] Limit the early grade contract to a schema version and not-requested/pending/validated status. Detailed grade fields belong to grading/report consumers, not this bootstrap ticket.
- [ ] Define supported analysis capabilities without selecting outcome rules: per-position cohort means at all sites; whole-example primary-window pooled evidence at all sites; per-position example evidence at the six retained blocks. Other data demands require a separately reviewed capture extension.
- [ ] Distinguish offline CPU completion from real-smoke evidence: this ticket cannot be reported as fully target-validated on CPU alone. The real GPU allocation and smoke duration are recorded.

## 2. Prepare and freeze the raw corpus

**Blocked by:** 1.

**What it delivers:** Build the shared 200-document raw cohort, exact 128-content-token mapping and immutable input identities through readable preparation/review cells. Corpus selection can complete independently of vocabulary review.

**Done when:**

- [ ] Public preparation recovers the original 150 document identities where eligible, applies an explicitly reviewed extension procedure and freezes the exact added 50. Selection is outcome-blind and never prefers documents because their previous results were clean or strong.
- [ ] Preserve a legacy-150/new-50 stratum label through inputs, retained evidence and reports; do not assert new documents are unseen without checking/recording prior exposure.
- [ ] Hand-worked token/offset cases and actual tokenizer metadata verify content positions and special-token/BOS behaviour; mismatches are surfaced rather than silently replaced.
- [ ] Reuse verifies token/text/source hashes; identical preparation reuses saved evidence without resampling.
- [ ] No learned lens output or grading is emitted by corpus preparation.

## 3. Freeze vocabulary and inference commitments before outcome reveal

**Blocked by:** 2.

**What it delivers:** Review and freeze the task-enriched vocabulary and outcome-dependent analysis/grade rules while capture qualification and lens inspection proceed independently. The background vocabulary uses the actual declared corpus; this is why the corpus artifact is a genuine dependency.

**Done when:**

- [ ] Public vocabulary preparation uses the fixed training-solution and background source, with explicit counts/smoothing/exclusions/ties and no learned token lists as input.
- [ ] Freeze reveal-only decisions: vocabulary, null/correction/threshold rules, sample-size draw counts/seeds and SNR estimator within the allowed retained-data family, plus blinding/grade rubric. Reference the primary/capture contract; do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.
- [ ] Prior exposure records cover earlier pilots and first looks: named reviewer/decision-maker, which outputs or token lists were seen, when known, and unknown history. Scalar Pilot 4 plots are distinguished from learned lens lists; do not invent an earlier readout.
- [ ] Freeze legacy-150/new-50 robustness analysis and its uncertainty/sample-size interpretation before reveal.
- [ ] Missing approval blocks outcome publication; an existing-data preview or bounded adapted preview cannot bypass this barrier.
- [ ] Outcomes revealed before freeze are marked exploratory and prior exposure is recorded; they cannot be relabelled confirmatory by a later freeze.
- [ ] Every allowed analysis is a function of the retained-data capability contract. Requests for unsupported per-position uncertainty at other sites are rejected before analysis rather than silently creating a dependency on capture or promising a rerun-free result. Capture does not wait for vocabulary approval.
- [ ] The notebook reveal barrier prevents accidental output through supported operations, not deliberate file access. Record its scope and any known bypass/exposure.

## 4. Qualify bounded paired capture and logit-lens mathematics

**Blocked by:** 2.

**What it delivers:** Run a declared diagnostic capture through the public workflow; validate named sites, primary pooling, paired differences and the logit readout using independent oracles. Save bounded numerical evidence without exposing learned task-token lists before ticket 3.

**Done when:**

- [ ] Independent tiny-model/full-array answers agree with streamed per-position sums and named-site capture.
- [ ] Both zero adapters pass the reviewed bounded identity checks, with unchanged base weights and finite/reconstruction checks.
- [ ] Retained precision has explicit quantization evidence and no claim that approximate shards are exact.
- [ ] Logit-lens norm gain, positive scaling and zero-vector behaviour match independent toy examples.
- [ ] GPU diagnostic failures save comparison/site/token coordinates, observed values and allowances without requiring another full run to name the failed check.
- [ ] Identity/reconstruction/nonfinite failures stop correctness qualification. They do not become null results or trigger threshold widening.

## 5. Inspect and qualify community Jacobian-lens status

**Blocked by:** 1.

**What it delivers:** Resolve the external lens risk early from the stable model/site contract, artifact metadata and independent tiny/base-model checks. Produce a reusable status artifact: qualified, incompatible, insufficient provenance or pending further qualification.

**Done when:**

- [ ] Public inspection binds the explicit artifact revision/hash, source sites, shapes and target/norm conventions rather than trusting its filename.
- [ ] Declared independent matrix and bounded base-model qualification checks run without production capture or learned task-trace reveal.
- [ ] Capture-site requirements are explicit even if readout qualification is unresolved; incompatible or unproven artifacts are rejected clearly.
- [ ] Predeclared lens-failure action: save the failed status and diagnostics, stop model work and make the bounded stage safe to release, then await Prasad’s decision on logit-only scope or a separately budgeted fit. Neither starts automatically; CPU source/vocabulary/fixture work may continue. No paid runtime is held while awaiting this decision.
- [ ] A terminal incompatible status is a valid ticket output; successful qualification is not assumed merely because the implementation ticket is complete.

## 6. Profile and freeze the explicit production plan and predictions

**Blocked by:** 4, 5.

**What it delivers:** Use numerical capture qualification and lens status to measure actual resource costs and freeze a reviewed production inventory. Vocabulary selection is not a computational prerequisite; predictions and capture-affecting scientific commitments precede production.

**Done when:**

- [ ] The public profile declares a bounded matrix before launch and reports loading/forward/hash/save/transfer timings, memory peaks and staging/durable storage needs.
- [ ] Conditions are enumerated, not implied: base reference, selected SFT/GRPO checkpoints including the final endpoints, random-64 and separately declared bounded zero checks.
- [ ] The existing random adapter's seed, construction, per-module norm target/realized values, source hashes and parent GRPO identity are bound. No new control is silently generated.
- [ ] This ticket is the sole owner of capture-affecting commitments: sites, content positions, pooling bank, conditions, batches, precision, retention and runtime. It also owns the single primary-vector definition and predictions. Review explicitly selects endpoints versus trajectory; unresolved choices block freeze.
- [ ] Freeze retention as a superset of the allowed analysis family: full position-wise cohort sums at all sites, durable per-position example deltas at blocks 14 and 17–21, and FP64 per-example pooled primary-window deltas/baselines plus required norm/count scalars at all 29 sites. Reveal policy may choose statistics on these objects but may not ask for unsupported bytes.
- [ ] Verify the actual hidden width from pinned model metadata. At width 1,536, all-site FP64 pooled vectors cost 67.97 MiB per condition (200 examples); six-block FP16 position-wise vectors cost 450 MiB per condition before compression. Preserve the approved token retention, qualify its precision and include additive pooled evidence in the budget.
- [ ] User/team prediction confidences and primary definitions are frozen before production; synthetic qualification predictions are not substituted.
- [ ] Lens status is an input. A qualified community lens proceeds; failed qualification follows ticket 5’s predeclared save/stop/release/ask route. Prasad owns the scientific fallback decision. Diagnostic capture does not fabricate Jacobian evidence.
- [ ] Prasad owns budget/scope approval. An unaffordable profile saves diagnostics, stops paid work before production, allows independent CPU review to continue and presents reductions in a predeclared order: defer optional form/trajectory, reduce secondary readout work, choose a smaller passing batch for memory, then propose separately reviewed corpus/site/retention changes. No automatic scientific reduction or post-failure tolerance retuning.
- [ ] Review states the next CPU/GPU grouping and exact durable-release boundary; no source-doc timing promise is treated as a measurement.

## 7. Capture, resume and durably verify raw evidence

**Blocked by:** 6.

**What it delivers:** One complete capture operation runs missing immutable batch/condition identities, resumes after a disconnect or partial write, and verifies required durable evidence before release.

**Done when:**

- [ ] Public capture uses one model, includes the frozen random condition, shares only compatible base references, and saves all-site sums and declared retained differences.
- [ ] Interrupted and uninterrupted controlled runs yield the same accepted inventory/aggregates; only incomplete units repeat and valid source bytes remain unchanged.
- [ ] Temporary/staged writes cannot publish incomplete units. Resume checks unit hashes; corruption is rejected or repaired under an explicit policy without silent acceptance.
- [ ] Free-space preflight and actionable disk/write/sync failures preserve accepted units and suppress a premature release message.
- [ ] Durable synchronization reuses valid copied units; final verification checks required retained shards, identities and seals before announcing GPU release and waits for child exit.
- [ ] Show actual phase, condition, completed/sealed units, throughput and estimate; separate hash/sync work from forward progress.
- [ ] Reuse existing writer locks with one bounded public conflict check and a clear error; no separate broad contention framework is required.

## 8. Decode saved means and reveal controlled token lists

**Blocked by:** 7, 3.

**What it delivers:** After the vocabulary/inference commitments are frozen, read saved means, build primary/secondary vectors and expose the direct token lists with qualified readouts and controls. No decoder capture repeats.

**Done when:**

- [ ] Public readout refuses to publish learned lists until the reveal barrier is satisfied, including earlier preview entrypoints.
- [ ] Pooled primary and selected secondary vectors preserve site/position/form identities; zero/unsupported/unqualified vectors remain explicit.
- [ ] Save positive/negative ranked token IDs, decoded strings and scores, norm/coherence and vector/lens provenance.
- [ ] Random-adapter, random-direction and base/adapted-mean controls remain distinct; do not recapture data to generate Gaussian-vector controls.
- [ ] Side-by-side logit/Jacobian display uses qualified artifacts or an explicit approved partial mode; useful readout units resume without decoder inference.

## 9. Produce automatic relevance and sample-size results

**Blocked by:** 8.

**What it delivers:** Apply frozen analysis definitions to saved readouts and retained examples, producing the primary results and declared layer/sample-size curves independently of human grading.

**Done when:**

- [ ] Independent hand-worked vocabulary/null/paired-example cases match public analysis outputs.
- [ ] Primary testing family, direct arm contrasts and secondary searches are separately reported; one arm passing a threshold does not prove an arm difference.
- [ ] Whole-example draws preserve paired positions/conditions; stored precision and undefined SNR/noise cases are disclosed.
- [ ] Report the predeclared primary result separately for legacy 150 and new 50 documents, with their counts and uncertainty/noise context. Lower readability in the smaller subset does not by itself establish outcome selection; any size-matched sensitivity uses frozen draws.
- [ ] No missing convention receives an invented default after readouts are revealed; changes create an exploratory amendment.
- [ ] Analysis persists its unit identities/draws and resumes meaningful costly work; report actual CPU/readout costs rather than claiming analysis is free.

## 10. Prepare blinded grading and add validated grade evidence

**Blocked by:** 8.

**What it delivers:** Prepare an opaque-ID bundle and separate mapping, validate imported LLM/human grades, compute reviewed agreement and attach a source-bound graded supplement when available. Human work does not gate the automatic report.

**Done when:**

- [ ] Public bundle/import operations exclude condition labels, enforce declared rubric/task/model/cost settings and reject duplicate/misaligned/incomplete grade inventories before unblinding.
- [ ] No external submission occurs without explicit authorization; imported grades record provenance and human sample selection.
- [ ] Agreement metrics use independent worked examples; unblinding is recorded after grade validation.
- [ ] A validated supplement can be attached to the report through the predeclared grade contract; incomplete grades remain pending and do not overwrite automatic evidence.

## 11. Publish the automatic research report and verified handoff

**Blocked by:** 9.

**What it delivers:** Produce a coherent automatic-results research report from verified captures/readouts/statistics, with a grade section marked pending/not requested until a validated supplement is available.

**Done when:**

- [ ] Public reporting succeeds without human/LLM grades and records the exact evidence coverage.
- [ ] Figures, tables, primary claims, limitations and registered prediction outcomes agree with saved inputs; incomplete evidence, unavailable lens and genuine null are distinct states.
- [ ] A later validated grade supplement adds a source-bound section without changing frozen automatic evidence or falsely labelling missing grades complete.
- [ ] Fresh CPU restoration/reporting makes no decoder capture calls; optional verified export states inventory, storage duplication and cost.

## 12. Add the optional assistant-form script and its isolation check

**Blocked by:** 11.

**What it delivers:** A separate optional launcher, disabled by default, uses the same documents and shared scientific code while preserving raw artifacts and creating a new formatted-input/base-reference execution.

**Done when:**

- [ ] Public extension preparation validates content/header/absolute-position mapping and distinct prompt/template/token identities.
- [ ] The user sees additional workload and qualifies/freezes it independently; raw activations are never used as formatted baselines.
- [ ] The optional path exercises capture/resume/durable sync/readout/report on a small controlled fixture and records its own bounded real-environment isolation check.
- [ ] Failure leaves raw completion/results unchanged; its script is not called by default setup, raw execution or required raw release validation.

## 13. Validate the completed raw notebook in fresh Colab

**Blocked by:** 11.

**What it delivers:** Run final raw-path deployment regression after the early environment smoke, using a bounded real capture/readout, durable sync and reconnect/reuse demonstration.

**Done when:**

- [ ] Record the actual clean environment, imports, source inventory, bounded workload, durable verification and reconnect reuse through public stage calls.
- [ ] Reviewer check: code is readable with explicit imports/named settings, purpose/input/output notes and no required repair cells. Keep this judgment separate from automatic checks such as parsing cells and rejecting expression-valued string fields.
- [ ] Incompatible runtime and active-child failures are actionable before inference; interruption cleans up its owned child.
- [ ] The check neither runs nor waits for optional assistant work or human grading. Those remain independent status entries.
- [ ] Separate offline implementation tests from target-GPU validation and from the actual full scientific experiment; no blanket guarantee about every Colab allocation.

## Revised graph and review notes

Main raw path: 1 → 2 → 4 → 6 → 7 → 8 → 9 → 11 → 13.

Additional genuine inputs: 1 → 5 → 6; 2 → 3 → 8. Grading branches 8 → 10; optional assistant work branches 11 → 12. Neither branch gates required raw reporting/validation. When vocabulary is slow, capture can finish while learned readout reveal remains blocked.

Splitting the former corpus/vocabulary ticket yields **13 tickets** (not 12); splitting capture additionally would yield 14. Capture is retained as one narrow complete outcome using existing artifact/process primitives, with bounded failure checks rather than a new general-purpose framework.

The phrase “each check has a public command” means the approved public workflow seam. Exact command names are established during implementation; these acceptance criteria do not invent a second CLI or bind tests to private helpers.

Second-review corrections are incorporated. The supplied review authorizes publication after these changes; scientific execution choices remain pending at their owning review gates.

## Human review ownership and wait states

Prasad is the decision owner; mentor/team input may inform his approval but is not invented as an extra gate. Vocabulary/inference approval pauses learned readout reveal while qualified capture and other non-revealing work can continue. Lens failure pauses the affected execution scope, persists diagnostics and releases model/GPU resources before asking for a fallback decision; independent CPU design/vocabulary work can continue. Budget rejection pauses production and releases the paid runtime; the saved profile remains available for CPU review of reductions. Capture/source correctness failure stops affected work and requires diagnosis, not a scientific scope fallback. No timeout is approval.
