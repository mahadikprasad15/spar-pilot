# 05: Inspect and qualify community Jacobian-lens status

**What to build:** Resolve the external lens risk early from the stable model/site contract, artifact metadata and independent tiny/base-model checks. Produce a reusable status artifact: qualified, incompatible, insufficient provenance or pending further qualification.

**Blocked by:** 01: Restore a clean session, audit sources and smoke-test Colab.

**Status:** claimed (inspection implemented; actual artifact has insufficient provenance)

- [ ] Public inspection binds the explicit artifact revision/hash, source sites, shapes and target/norm conventions rather than trusting its filename.
- [ ] Declared independent matrix and bounded base-model qualification checks run without production capture or learned task-trace reveal.
- [ ] Capture-site requirements are explicit even if readout qualification is unresolved; incompatible or unproven artifacts are rejected clearly.
- [ ] Predeclared lens-failure action: save the failed status and diagnostics, stop model work and make the bounded stage safe to release, then await Prasad’s decision on logit-only scope or a separately budgeted fit. Neither starts automatically; CPU source/vocabulary/fixture work may continue. No paid runtime is held while awaiting this decision.
- [ ] A terminal incompatible status is a valid ticket output; successful qualification is not assumed merely because the implementation ticket is complete.

## Actual inspection · 2026-10-16

Pinned revision `bc6aec86a96e7e75ce286dfb8f3e59aaaca50cce`, lens SHA `6a23a7460139a0d3b73fdcc3a204bdee57475223bdfa72a284e6c7c60ed3a9cc`: 27 finite 1536×1536 FP16 matrices, source keys 0–26, n_prompts 16. Primary block 14 exists. Exact fitting model revision and fitted target/norm provenance remain unresolved. Saved status: insufficient-provenance, qualified false. No paid base-model check or fallback fitting was launched. See the handoff report for sources and constraints.

## Explicit remaining gate and repair · 2026-10-16

The inspection/profile handoff exposed `qualified: false` but did not clearly track the missing executable real-matrix qualification producer. Ticket 05 remains **partial**. A bounded CPU workflow now reuses the four saved profile documents, actual J14 and pinned norm/head: independent FP64 arithmetic checks plus nonzero FP16/FP32 top-20/bootstrap sensitivity, with sealed resume and a separate explicit review receipt. Notebook sections 20A–20C precede production. No learned lists are revealed and no new decoder calls or fitting are performed. See `docs/pilot5-jlens-qualification.md`.

**Still awaiting execution/review:** real-artifact diagnostic results, explicit numerical acceptance limits and provenance acceptance. This bounded block14/positions0–4 receipt does not qualify all layers, establish semantic lens fidelity, or replace Ticket 04/06 capture qualification and production review. Original inspection evidence is not rewritten.
