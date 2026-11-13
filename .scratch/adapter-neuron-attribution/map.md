# Adapter and neuron attribution — map

## Notes
- First pass only; deferred items are listed at the end of `spec.md`.
- Reuses the joint audit's inventory-v2, sequences-v2, FP32 policy and Pod/export chain.
- Stage 0 heatmaps = write size; this study = necessity (attribution + real ablation).
- Neuron attribution = removing a neuron's term from a `down_proj` adapter's read, not a neuron ablation.
- Proxies (M_acc, M_len, KL) drive attribution and Stage C; v3 accuracy and length are measured only in Stage D.

## Decisions so far
- 2026-11-13 (user): shrink to a first pass — merged A+B pass, base-correct traces only, top-5 singles + top-10 group in Stage C, one 64-neuron read per arm, generation for gold-sft and online-grpo only, no drift metric.

- 2026-11-13 (user): GPU is underused; measure utilization (busy %, MFU, tokens/s) in every stage and size batches by sweep. Run tiers: Colab bug-finding notebook, then Pod qualification and production.

- 2026-11-13: review round 1 (Codex) answered in `review-response.md`; draft v2 of spec, plan and tickets (rehearsals on calibration rows only; z recomputed in a forward-only pass; trace log-likelihood and stop pull replace M_acc/M_len; per-arm gates, equivalence margin, Holm families; FP32 throughout; tickets renumbered 01–10).

- 2026-11-13: review round 2 (Codex) recommends CPU implementation without another design cycle; three corrections (frozen-model gradient path, exact stop IDs, uncertainty in primary decisions) made in draft v3. User approved CPU implementation of 01–05.

- 2026-11-13 (user): the Pod run must be overnight and unattended — qualification, traces, attribution, ablation, generation and report as one chain from an approved launch record, ending with the Pod stopped. Added §Overnight unattended chain, ticket 11, report code moved to ready-for-agent.

- 2026-11-13: review round 3 (Codex, overnight chain and reuse) adopted in full: scalar-read attribution for all adapters (no full-width gradients), export contract extended to this study's files, bounded-time teardown with sealed-only copies and requested-vs-verified stop, explicit resume rule, hard caps with advisory projections, adapt-not-reuse notes on old numeric code.

- 2026-11-13: ticket 02 resolved — `attribution_metrics.py` (target-position masks shifted by one; stop pull NaN when empty; KL full ‖ condition). See `issues/02-metrics.md` §Answer.

- 2026-11-13: ticket 05 resolved — `attribution_units.py` (BlockStore units, sealed run directories, persisted hard-cap budget, export with declared expected units and runs) and `attribution_gpu.py`. See `issues/05-units-export.md` §Answer.

- 2026-11-13: ticket 01 resolved — `attribution_traces.py` (prompt IDs from the frozen prefix, raw responses sealed before scoring, decoder-sealed stop IDs, target-position masks, minimum after mask exclusions, confirmation blind until the freeze). Open question on the rehearsal-item draw rule in `issues/01-trace-preparation.md` §Comments.

- 2026-11-13: ticket 03 resolved — `attribution_instrument.py` (lora_A scalar reads, embedding-output graph, one autograd.grad per metric, checkpointed head at read positions, pass-2 neuron contraction with the Σ_j identity; oracle and FD checks). PEFT 0.18 `disable_adapter()` unfreezes adapters on exit (lesson recorded). See `issues/03-attribution-instrument.md` §Answer.

- 2026-11-13: ticket 04 resolved — `attribution_ablation.py` (lora_B-gate removal == edited checkpoint, natural 64-term and constant reads, matched controls, discovery-only sealed freeze, ≤19 conditions with recorded skips, Stage C vs the same arm unablated). Open question on random-order seeds in `issues/04-ablation-engine.md` §Comments.

- 2026-11-13: ticket 10 (code) resolved — `attribution_report.py` (gate, primaries 1–3 by interval rules, Holm per metric, concordance, concentration, render) on fixtures. Interpretations to confirm in `issues/10-report.md` §Comments.

- 2026-11-13: ticket 11 resolved — `attribution_chain.py` (sealed launch record, rule sign-off, persisted caps, teardown on every path ending in a requested stop, resume rule, watchdog), `attribution_stages.py` (06b–09 stages from 01–05 and 10, CPU end-to-end rehearsal test), `attribution_backend.py`, `scripts/attribution_{watchdog,overnight}.sh`. New launch-record fields listed in `issues/11-overnight-chain-runner.md` §Comments.

## Fog
- Actual utilization of the joint audit's run (no NVML record exists).
- How many base-correct rows survive.
- Whether a ≈ a₀ in each arm's top `down_proj` module.
- Whether first-order estimates track real removal of whole adapters (gate).
- HF FP32 generation time for 8 × 150 items.
