# Stage 0 vertical-slice proposal

Status: approved by the user on 2026-10-26.
All 15 individual ready-for-agent issues are published in the configured local tracker.

Each slice adds public saved-workflow commands, durable verified outputs,
CPU-only workflow tests and the applicable sections of the shared CPU/GPU
notebooks. A completed slice is independently inspectable or runnable. There
is no separate notebook per ticket and no artificial foundational refactor.
Dependencies below are software/product capabilities, not a demand to execute
every ancestor's optional scientific workload on real data.

## Primary intervention path

1. **Source audit, fresh-runtime restoration and budget status**
   - **Blocked by:** None.
   - **Delivers:** Select source runs in the CPU notebook, audit actual source
     metadata/tensors and available products, save a parameterized Stage 0 plan,
     then restore it in a new process and inspect source readiness and cumulative
     budget. No GPU call occurs. Missing optional sources do not block endpoints.
   - **Acceptance:** Source evidence is distinguished from assumptions; changed
     source payloads invalidate identity; actual A0 comparisons are reported when
     available, not inferred from seeds. Persist status, exact source references
     and credit observations. Restore retains the total 20-unit ceiling and never
     interprets missing observations as zero expenditure. Existing behavior and
     release paths remain intact.

2. **Verified spectral cache and endpoint projection report**
   - **Blocked by:** 01.
   - **Delivers:** Run CPU SVD once per selected base module, resume partial cache
     creation, and produce endpoint projection tables/plots against initialization,
     chance and random references. Save bases sufficient for principal edits and
     floored intruder subspaces, not just the top-20.
   - **Acceptance:** SVD reconstruction/orthogonality and payload identity gates
     pass or preserve failure evidence. k30/k50/k70 and dimensions reconcile.
     Zero factors are undefined where appropriate. Tiny explicit matrices provide
     independent workflow oracles. A rerun reuses verified module units.
     Cache preparation does not automatically reveal trained overlaps; report
     reveal respects the preregistration boundary, while the primary edit path
     can consume bases without waiting for optional gradient predictions.

3. **Principal edits and per-module magnitude/random controls**
   - **Blocked by:** 02.
   - **Delivers:** The CPU notebook builds and verifies V0a/V0b, V1/V2, three V3
     draws, V4, the analogous V2 size control and V5; users inspect an immutable
     variant catalogue with hashes and norms before evaluation.
   - **Acceptance:** k=0 identity is byte-preserving; projected components satisfy
     declared numerical gates; per-module effective-update sizes match their
     controls and directions remain correct. Seeds, construction, partial progress
     and dense independent matrix checks are retained. Existing adapters unchanged.

4. **Bounded GPU qualification and measured production freeze**
   - **Blocked by:** 03.
   - **Delivers:** The GPU notebook restores the selected catalogue, verifies the
     pinned runtime/durable storage, runs bounded instrument and mixed-loader
     diagnostics, reports costs and approval requests, then produces a frozen
     executable plan only with valid qualification and budget approvals.
   - **Acceptance:** Sixteen-sequence common-prefix diagnostics, negative control,
     hard 1e-3 stop, proposed 3E gate, >=1000x sensitivity, near-tie exceptions and
     separate approval receipts follow the spec. Un-halved partial merges never
     double-apply the adapter. Credit reservations include diagnostics and total
     prior usage; unsafe jobs are refused before spending beyond the ceiling.
     Real-Colab evidence is distinct from mocked CPU checks. Test fixtures exercise
     all gates without a model download.

5. **Resumable paired greedy intervention evaluation**
   - **Blocked by:** 04.
   - **Delivers:** Execute the frozen principal variants and fresh SFT/GRPO
     references on the 150 Pilot 4 questions, resume sealed batches after an
     interruption, and produce a CPU-readable behavioral result with all responses.
   - **Acceptance:** Exact prompt/template/cap, fixed batches and scorer contracts;
     JSONL response/token/length/cap records; completed-unit counters; source and
     runtime mismatch rejection; unit durability and one-writer safety. References
     belong to the same comparison runtime. Restart cannot reset the credit ledger
     or silently combine incompatible sessions. Tiny backends demonstrate the
     whole collect/interrupt/verify/inspect path.

6. **Paired fixed-context KL and effective-write results**
   - **Blocked by:** 04. Independent of 05.
   - **Delivers:** Collect and verify per-position KL and same-runtime effective
     writes for frozen variants on the fixed GSM8K/FineWeb sequences, then inspect
     document/module summaries on CPU without collecting full-vocabulary arrays.
   - **Acceptance:** Counted-position and view mappings are preserved; KL direction
     and accumulation are correct; direct matrix oracles reconstruct write
     measurements; switching/integrity/zero controls retain declared gates.
     Sealed batches resume safely. Saved historical values serve only compatible
     checks. Joined behavioral/KL claims require matching comparison provenance.

7. **Primary findings, targeted review and non-inferiority report**
   - **Blocked by:** 05, 06.
   - **Delivers:** CPU commands generate paired accuracy/length/KL tables and plots,
     export the targeted intended-answer audit, incorporate confirmed labels and
     render an inspectable primary report with raw tables and evidence links.
   - **Acceptance:** Lower 95% bound strictly above -5pp is required for an accuracy
     preservation claim; n=150 inconclusive risk is preregistered. All specified
     controls and individual random draws remain visible. Labels apply only to
     identical text; pending review and raw/overlay outcomes stay distinguishable.
     Incomplete optional work or not-applicable V6 does not block this report.
     Per-item and document sums reconcile with reported intervals and figures.

## Conditional intruder extension

8. **Gap-aware rotation and chance-aware intruder report**
   - **Blocked by:** 02. Independent of 03-07.
   - **Delivers:** CPU analysis computes updated-matrix singular geometry,
     principal-angle rotation, update/gap diagnostics and intruder tables with
     explicit eligibility for conditional edits in either arm.
   - **Acceptance:** Always measure, with no small-update skip. Top-20 detection,
     m=max(k50,40), floored k30/k70 sensitivities, norm thresholds 0.3-0.7,
     sqrt(m/d_out) and undecidable status follow the approved rule. Candidate
     top-40 control overlaps, singular values and gaps are retained. Invariance
     under within-subspace basis changes and zero-update cases are independently
     verified; undefined and degenerate boundaries are explicit.

9. **Matched merged-weight intruder catalogues**
   - **Blocked by:** 03, 08.
   - **Delivers:** Build full V6, matched V6/V6c and ratio-restricted variants for
     SFT and/or GRPO where eligible flags exist; otherwise publish a verified
     not-applicable result. Catalogue integrates with the existing variant runner.
   - **Acceptance:** Halve only eligible primary flagged singular values. Controls
     are distinct non-flagged top-40 directions, assigned by minimum total absolute
     singular-value difference with index tie-break. Preserve pair ratios, module
     exclusions and [0.8,1.25] sensitivity masks without rematching. Full matrices
     and per-module adapter disable masks reconstruct the intended operator;
     original sources and full V6 remain separate from matched variants.

10. **Conditional intruder evaluation and findings extension**
    - **Blocked by:** 07, 09.
    - **Delivers:** Reuse the qualification, budget, behavioral/KL collection and
      reporting workflow for conditional merged variants, with fresh references
      when required, and add a separately identified findings extension.
    - **Acceptance:** Mixed loaders pass approved qualification; all control masks,
      merged matrices and representations are bound in the frozen plan. Credit
      accounting is cumulative, not reset for this extension. Reports distinguish
      full/matched/sensitivity cohorts, undecidable/skipped modules, and either
      arm's eligible edits. No extension runs if the catalogue is not applicable.

## Optional supporting branches

11. **Audited untuned rollout completion and derived cohorts**
    - **Blocked by:** 01.
    - **Delivers:** Recover historical raw groups/settings, generate only missing
      eight-draw groups under a frozen measured-budget plan, and publish verified
      gold/own_all/own_correct/own_correct_one memberships with cohort summaries.
    - **Acceptance:** Complete unique 512x8 identities when source compatibility
      supports combining groups; frozen historical ordering/seeds; strict/v2/v3
      and cap outcomes retained; old/new rates descriptive only. Preserve common
      input compatibility evidence or separately identify incompatible products.
      Zero-correct questions and deterministic one-correct selection are explicit.
      This branch cannot block principal interventions or consume reserved budget.

12. **Teacher-forced NLL, entropy and excess-surprise report**
    - **Blocked by:** 11.
    - **Delivers:** Score each unique answer once on GPU and summarize the approved
      dataset memberships on CPU, with per-token evidence and problem-level
      intervals, using a frozen measured-budget plan.
    - **Acceptance:** SFT masking and stopping treatment audited; NLL primary,
      descriptive excess surprise secondary, independently named intervals and
      separate whole-answer negative-log correctness probabilities. Preserve the
      +/-0.02 own_all gate and its sampling premises. No invalid KL identity,
      own-correct agreement test or duplicate subset forwards. Resume preserves
      membership, token mappings and counter/credit accounting.

13. **Initial gradient pushes and training-alignment report**
    - **Blocked by:** 02, 11. Independent of 12.
    - **Delivers:** Run frozen matched gold/own batches and historical GRPO loss
      diagnostics at saved initialization, retain per-batch gradients/pushes, and
      report their size, consistency and geometric/endpoint alignments on CPU.
    - **Acceptance:** No optimizer update; proper masked CE and DAPO reductions
      across microbatches; detached historical advantages; dead-group zero pushes;
      gradient sign and factor-gauge conventions correct. Independent tiny-model
      reconstruction/finite differences retain approved gates. Differing A0 and
      selected-cohort sizes are explicit. Conditional predictions are frozen
      before revealing trained-overlap results, without blocking the primary path.
      If overlaps were already inspected, label subsequent interpretation exploratory
      rather than claiming a prospective prediction.

14. **Activation-energy decomposition report**
    - **Blocked by:** 02, 04.
    - **Delivers:** In one base-model pass per frozen sequence, collect all
      selected factors' principal, rest, cross and total energies; the CPU
      notebook reports bounded headline fractions and signed interference terms.
    - **Acceptance:** float64 sufficient sums and counts, correct token views,
      independent reconstruction, undefined zero denominators, base-input scope,
      and resumable verified units. All factors share captured base inputs rather
      than requiring separate arm forwards. Work runs only in approved spare budget.

15. **Optional checkpoint trajectory and drift-size report**
    - **Blocked by:** 02, 06.
    - **Delivers:** Select original intermediate checkpoints and verified reusable
      products, collect only missing same-runtime KL/write measurements when
      approved, and publish CPU geometry/drift-size trajectories plus saved-output
      endpoint length analyses.
    - **Acceptance:** Historical/new compatibility is evidence-backed or separated;
      checkpoint/sequence identities and denominators are explicit; raw coefficient
      sensitivity retains prior policy boundaries. Observational size-matched curves
      do not become causal direction claims. Optional failures/missing products
      remain isolated from the primary report and total budget remains cumulative.

## Proposed dependency summary

- Primary: 01 -> 02 -> 03 -> 04 -> (05 and 06) -> 07.
- Conditional: 02 -> 08; (03 and 08) -> 09; (07 and 09) -> 10.
- Supporting: 01 -> 11 -> 12; (02 and 11) -> 13;
  (02 and 04) -> 14; (02 and 06) -> 15.

Granularity and dependency edges were approved before publication. Source
verification and real execution evidence remain separately tracked from software
ticket completion.
