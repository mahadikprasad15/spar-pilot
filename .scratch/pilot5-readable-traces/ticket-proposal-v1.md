# Pilot 5 ticket proposal

Status: awaiting user approval of granularity and blocking edges. These are proposed vertical slices, not published implementation issues. Public workflow test seam is approved.

## 1. Restore a clean session and audit source readiness

**Blocked by:** None.

**What it delivers:** A readable initial notebook path restores saved Pilot 5 selections, inspects existing adapter/model/corpus/lens candidates, shows missing or incompatible evidence and next pending work on CPU, and never launches a model or repeats older pilots. Demonstrate from a fresh kernel and offline source fixture. Include explicit source selection and stable stage identities.

## 2. Prepare and freeze the raw corpus and relevance vocabulary

**Blocked by:** 1.

**What it delivers:** From verified sources, build the 200-document raw corpus with exact 128-content-token mapping and the reviewed task-enriched vocabulary, and expose review/freeze in readable cells. Recover original document identities, persist selection and content hashes, detect BOS/alignment/eligibility mismatches, and block unresolved selection/scoring settings. No GPU or outcome readouts.

## 3. Qualify a bounded paired capture and logit-lens preview

**Blocked by:** 2.

**What it delivers:** A small reviewed diagnostic workload captures named embedding/block sites and per-position differences, checks both zero adapters and unchanged-base identities, and produces real ranked logit-lens tokens with norms/coherence. Demonstrate mathematical correctness through independent tiny CPU models; expose bounded target-GPU qualification and coordinate-specific errors. Retained-precision checks belong here.

## 4. Inspect and qualify the community Jacobian lens

**Blocked by:** 3.

**What it delivers:** Load a explicitly selected, hash-bound artifact, validate model/site/target conventions, run declared compatibility and bounded independent readout checks, and expose ranked tokens beside the logit preview. Reject unsupported artifacts clearly and offer a separately approved fit proposal without starting it. The preview is independently demoable; no production capture required.

## 5. Profile readiness and freeze the reviewed execution budget

**Blocked by:** 3, 4.

**What it delivers:** The actual GPU environment passes fresh-process text-model imports and runtime checks; a declared bounded representative workload produces phase timings, memory peaks, staging/durable storage estimates and total cost projections. A review freezes explicit checkpoints, batches, precision, retention, analysis approvals and stop/resume boundaries. Missing scientific choices block freeze. No forced regeneration of earlier measurements or unbounded benchmark matrix.

## 6. Capture, interrupt, resume and durably verify the raw experiment

**Blocked by:** 5.

**What it delivers:** Execute only missing frozen batch/condition units using one model and shared compatible base references, stream durable progress, persist all-site position sums and retained per-example deltas, resume after interruption, synchronize required evidence and issue a GPU-release message only after durable verification and child exit. Demonstrate corruption, writer contention, disk/sync failure and source-preservation behaviour through public commands.

## 7. Decode saved means and produce controlled readout lists

**Blocked by:** 6.

**What it delivers:** Read verified capture data without rerunning the model, build pooled primary and declared secondary vectors, apply qualified lenses, and save directly inspectable positive/negative tokens and scores. Random-adapter, random-vector and mean-activation controls have distinct provenance; zero/unavailable vectors are explicit. Readout stages resume their costly units and report actual progress.

## 8. Produce automatic relevance and sample-size analysis

**Blocked by:** 7.

**What it delivers:** From controlled readouts and retained examples, apply the reviewed vocabulary, primary null/correction and whole-example bootstrap/SNR rules to produce primary tables, layer curves and N=25/50/100/200 curves. Separate descriptive results from approved inference, account for retained precision, expose direct arm contrasts and state missing/undefined settings. No private implementation-based expected results.

## 9. Prepare blinded grading and import validated grades

**Blocked by:** 7.

**What it delivers:** Prepare an opaque-ID bundle and separate unblinding map, display reviewed rubric/model/tasks/call budget, import validated LLM/human grades and compute the declared agreement. No external call is made without explicit authorization. Missing grades, duplicate/misaligned lists and premature unblinding are rejected. This is independently usable while automatic analysis proceeds.

## 10. Publish the research report and verified handoff

**Blocked by:** 8, 9.

**What it delivers:** Join verified capture, lenses, automatic analysis and available grading into a readable research narrative, plots, exact tables, prediction outcomes and optional verified export. Distinguish partial evidence, pending human work, failed qualification and genuine nulls. Expose fresh CPU report restoration, provenance and archive/storage costs. Never label incomplete required grading as a complete graded experiment.

## 11. Add the optional assistant-form experiment script

**Blocked by:** 10.

**What it delivers:** A separate optional notebook launcher, disabled by default, prepares assistant content on the same documents, qualifies the new mapping/runtime, captures a distinct base reference and runs the shared readout/analysis/report path under a new identity. Demonstrate independent interruption/resume and preservation of raw artifacts. This extension does not gate the main raw report.

## 12. Validate the complete notebook in a fresh Colab session

**Blocked by:** 10, 11.

**What it delivers:** Record a bounded real fresh-runtime validation covering environment install/imports, source inventory, missing-stage selection, a qualified tiny capture/readout, durable sync, disconnect/reconnect and reuse. Check readable code, no expression-valued form fields or repair cells, compatible runtime rejection, owned-child cleanup and optional assistant isolation. This validates the deployment surface; it does not replace full scientific production or claim every possible Colab allocation is supported.

## Why these boundaries

Each slice ends in a visible artifact or complete user operation and includes its notebook/API/state/testing path. Environment and recovery are present from the first relevant slice rather than postponed to a cleanup ticket. The two grading paths can proceed independently after readouts. The optional assistant extension and final Colab smoke are independently deliverable after the raw end-to-end path. No broad prefactor is justified from current evidence; reuse shared instrument and artifact primitives additively.

Review requested: is this granularity right, are the blockers genuine, and should any slices be merged or split? No individual issue files are published until the breakdown is approved.
