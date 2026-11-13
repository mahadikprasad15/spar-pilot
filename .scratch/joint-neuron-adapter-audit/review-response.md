# Response to external adversarial review

This document records the disposition of every point in the supplied review. The accompanying spec and tickets are drafts; no numerical tolerance, expensive run, or incomplete source has been approved by this response.

## Blocking points

- **B1 — Forward precision:** Adopt FP32 production forwards, TF32 off and no autocast; BF16 is diagnostic only. A small effective-write ratio and BF16 spacing make the concern plausible, but do not by themselves prove that every smaller effect disappears. The real-GPU qualification measures the effect at the actual instrument seam.
- **B2 — Stored paired differences:** Adopt FP32 per-example summaries, rather than FP16 as a production option. This avoids FP16 storage rounding dominating a later subtraction; FP32 is still finite precision and is not presented as perfect.
- **B3 — Silent adapter load:** Compare saved/live factor keys, tensors, shapes, scales and target routing for every arm; independently check disabled-base and active-branch/logit behavior. **Qualification:** a truly zero factor, read or tiny logit difference can be legitimate, so blindly requiring nonzero ΔW*x on every module and batch would incorrectly reject valid cases. Use designed probes and explicit zero/unresolved reporting; unexplained missing keys, inactive routing or missing weights fail.
- **B4 — Signs and scale:** Treat raw signed reads as checkpoint-gauge dependent, prefer physical vector writes, and require an explicit alignment convention and near-zero/orthogonal exclusions for signed cross-arm comparisons. Within-arm magnitude ranks are invariant to nonzero common rank-one rescaling. Even comparing a to its initial value depends on the checkpoint's factor convention and verified lineage; do not assume the sign could not have changed.
- **B5 — Activity-matched null:** Add at least 1,000 within-layer, without-replacement draws matching the randomized list's frozen base-activity-bin counts. Specify the eligible universe, fixed versus randomized side, pooled statistic and underfilled handling. Its conditional randomization interval is not a confidence interval or a cure for other confounds.
- **B6 — Missing ticket:** Confirmed: `05-selection.md` actually contained a duplicate of 04. It now contains a real selection ticket. Ticket 03 has been split into 03a/03b.
- **B7 — Partial export:** Choose the configured private HF dataset repository and Hub API with sealed manifest and independently downloaded readback. Ticket 07 depends only on the 03a artifact contract and is a prerequisite for 03b/04, not dependent on the final report. It can export partial results after any stage or failure; failure to verify export prevents automated stop.

## High-priority points

- **H1:** Define primary D as |mean over examples of adapted-minus-base mean |z||; retain signed change and add relative D with a declared activity floor. Compare B∩D under both.
- **H2:** Freeze identical example membership, order, padding and position/mask hashes across arms. Measure repeats, alternate batch compositions, singleton comparisons and approximately ten FP64 examples. **Qualification:** matched shapes do not force errors to cancel; the measured floor is sample- and model-dependent. The suggested 5× flag is an exploratory display, not a universal correctness or significance criterion. Exact tolerances and floor formula remain to be agreed before the measurement.
- **H3:** Add independently captured PEFT `lora_A` output, independent gate/up reconstruction and actual adapter-branch comparison; use shifted mask, wrong layer and sign mutations as negative controls.
- **H4:** Replace long-form neuron parquet with immutable dense FP32 row blocks and a sealed example index. The spec contains an explicit size formula and illustrative estimate; actual storage and temporary-space estimates must use measured counts.
- **H5:** Check interpreter, prefix, pinned packages, resolved HF cache and offline model/tokenizer loading in the actual launcher and each actual worker. Keep online HF access checks separate and explicitly online; offline environment must be set before Hub import.
- **H6:** Confirmed locally: the imported offline-GRPO run has step-0 and step-64 tensor files and trainer states, plus the internals result used by the heatmap. Neither checkpoint directory contains `adapter_config.json`. Ticket 01 starts there and must find/verify the true configuration and source chain before treating them as load-ready.
- **H7:** Split CPU instrument 03a from real-GPU qualification 03b. Benchmark and select the frozen batch/head plan in 03b, before production 04; exporter 07 is available before that expensive stage.
- **H8:** Declare a descriptive primary family and exact equal-layer overlap-fraction pooling, plus per-layer and arm/arm-pair results. Near-miss, relative D, smaller budgets and other views are secondary. Multiple comparisons remain exploratory.
- **H9:** Define held-out estimands for fixed A selectivity, B read-term sizes and D paired changes/sign agreement, each compared to discovery-frozen matched controls; no held-out re-ranking.

## Medium and operational points

- Use a host/PID/heartbeat writer record and stop for stale state. **Qualification:** `O_EXCL` on an unverified network/FUSE backend alone does not establish cross-client mutual exclusion. Require verified semantics, an external conditional lease, or explicit operator-enforced exclusivity; never silently take over.
- Check output and temp create/write/read/rename/fsync/delete capabilities; report a volume's claimed free space separately from usable quota. Unbuffer logs and save timestamped work-unit progress; stale age is a warning, not proof of a hang.
- Hash source bytes once per inventory/session unless immutable content identity is actually guaranteed; verify sealed unit hashes on reuse. Size/mtime by itself cannot detect same-size corruption. Avoid recursive whole-history rehashing at every unit.
- Freeze first-turn extraction and disjoint background groups; define exact tokens/masks before qualification. Batch membership/padding is frozen **after** benchmarking, avoiding a circular dependency between data preparation and a GPU-selected batch size.
- Reduce layers immediately, benchmark memory and treat the proposed 32k-token bound as an initial probe only. Output size includes all arms, datasets, positions, indexes and temporary copies; do not treat the per-token c_t estimate as total storage.

## Remaining review decisions

The spec deliberately does not invent numeric tolerances, the exact 5× floor denominator, source datasets/revisions, activity floors/binning, bootstrap choices, 1B cost thresholds, and incomplete-arm policy. Those choices need a reviewed frozen plan before the relevant real run. They do not prevent adversarial review of the implementation contracts.
