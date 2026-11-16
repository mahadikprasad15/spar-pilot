# 06a: Colab rehearsal

**Status:** needs-triage — draft v3
**Type:** task
**Blocked by:** 01–05
**Authorization:** user go-ahead and a Colab cap. Outputs labelled `rehearsal-colab`; never feed production.

**What to build (approved now, CPU session) and run (needs go-ahead):** `notebooks/adapter-attribution-colab.ipynb` on calibration rows and the 20 rehearsal generation items only, for `gold-sft`, `online-grpo` and untuned.

- [ ] Environment per the joint audit's Colab lessons (stream child output, remove conflicting preinstalled packages, pinned tokenizer pad ID, record runtime).
- [ ] Base traces for calibration rows via 01.
- [ ] Covering set: no-op hook; single removal; group removal; full `down_proj` removal; natural 64-term read; constant read; full vs edited-checkpoint generation (2 items); a corrupted and an interrupted unit.
- [ ] Real-model finite differences (3 adapters, 3 neurons per arm) and the Σ_j identity.
- [ ] First token-budget probe from 2k upward with singleton and longest examples; record OOMs.
- [ ] Online upload and readback of this study's partial outputs; child-process logs captured.
- [ ] Every failure fixed with a regression test and recorded in `docs/agents/experiment-lessons.md`.

## Comments

### 2026-11-13 — notebook written, not run (implementing agent)

- `notebooks/adapter-attribution-colab.ipynb` (19 cells) and its driver `pilot_eval/attribution_rehearsal.py`. Driver tests run on CPU with a tiny backend and a mocked Hub: `tests/test_attribution_rehearsal.py`; notebook scope checks: `tests/test_attribution_notebook.py`. Neither has run on a GPU.
- **Joint-audit Colab lessons applied:** child output streamed into the cells; torchvision and torchaudio removed before pinning; pad ID taken from the pinned tokenizer (`ChatTokenizer`); runtime recorded; the evidence-upload receipt printed in the cell.
- **Inputs:** calibration rows and the 20 rehearsal items only. The notebook refuses to start until the user sets the Colab cap, the 20 rehearsal item indices (ticket 01 question), the random-order seeds and the null seed (ticket 04 question), and the step timeouts (ticket 11). Tolerances and sweeps are copied unchanged from the candidate plan and only exercise the checks.
- **Still needs:** the user's go-ahead and a cap, as before.

### 2026-11-14 — operating rules for the first run (user + Codex)

- **Supervised.** Run the notebook once and don't restart automatically; if a step fails, run the evidence cell and report.
- **A test, not production.** Before trusting the backup, check that the summary lists every step and that the clean restore returned it.
- **Inputs.** The tokenizer directory is explicit (cell 6, the inventory's pinned tokenizer revision).
- **What to learn:** GPU execution (incl. the device fix from round 4), memory use per token budget, and runtime per step. Fix only what this run reveals, then decide on an overnight launch.

### 2026-11-16 — run r1 (user)

- Cells 1–6 passed on an A100-SXM4-40GB (FP32 peak 19.5e12 picked up automatically; rehearsal record sealed at `480234a`).
- Cell 7 stopped in `bind_sources`: the GSM8K test parquet is absent from the prep snapshot, which the message wrongly called "differs". The evidence upload worked; it held 0 files because nothing had been written.
- Fixed: the message now says "missing" (with a test), and new notebook cell 4b fetches both GSM8K parquets at the manifest's pinned revision with hash verification. Next: rerun all cells.

### 2026-11-16 — run r1, second attempt (user)

- **Run:** after the GSM8K fetch (hashes matched), the covering set ran on the A100. Export, readback and clean restore succeeded (`complete`, 6 files, revision `ea082b61`).
- **Root failure:** `calibration_traces` failed because `verify_sources` used `neuron_audit_inventory.verify`, which rebuilds the inventory from the joint audit's original source files. A restored bundle can't satisfy that anywhere but the original disk.
- **Cascade:** every dependent step then failed confusingly, and `attribution:*` reported ok with zero rows.
- **Fixed:**
  - inventory-v2 is now checked by hash against its own seal (`attribution_traces.inventory_integrity`), with the inputs actually used bound by `bind_sources`, as the ticket's "verify by hash" asks;
  - dependent steps are `skipped` with "needs calibration_traces";
  - attribution with no rows fails;
  - skipped steps count as missing work and give a non-zero exit.

### 2026-11-16 — run r2 (user)

- **Passed on the A100-40GB, both arms:** calibration traces; attribution (one row per unit); finite differences; corrupt/interrupted units; token-budget probe.
- **Failed, out of GPU memory (36 GB held), both arms:** `conditions` and `generation`. Both started with a calibration-only selection that re-ran attribution (two backward passes) over all calibration rows in one batch.
- **Fixed:**
  - the selection reads the units the attribution step had sealed (no GPU work, cached per arm);
  - interventions are evaluated in batches packed to the smallest sweep budget;
  - the driver frees cached CUDA memory after every step.
- Production is not affected: its selection is the frozen discovery freeze built from sealed units, and Stage C already runs in packed batches.

### 2026-11-16 — run r3 (user): every step passed; the finite-difference gate would fail

- **All 11 steps ok; export and clean restore `complete` (135 files, revision `c5ee6c08`).**
  - No-op hook: ΔL = 0 and KL = 0 exactly.
  - Edited checkpoints: logits bit-identical to the gate.
  - Interventions ordered sensibly: group > single ≈ full `down_proj` > 64-neuron read > constant read.
- **Attribution memory (A100-40GB):** about 15 GB fixed plus about 2.5 MB per token. 1,808 tokens used 21.4 GB; 4,068 used 31.6 GB; 8k and above ran out of memory. Throughput was about 1,100 tokens/s at 2–4k tokens.
- **Time:** calibration traces (25 questions, batch 1) took 303 s; generation dominates.
- **Finite differences against the proposed tolerances:**
  - gold-sft: median 0.008, max **60.8**;
  - online-grpo: median 0.029, max **1.92**.
  On the Pod the automatic sign-off would stop at qualification. **Not resolved and not loosened.**
- **Diagnostic:** `finite_difference_check` now reports every comparison and the 10 worst, with an FP32 noise estimate (`d106108`; reporting only). A Colab diagnostic cell was given to the user. Pending: whether the large errors are rounding noise on tiny effects or a real error. Any change to the check is the user's decision.

### 2026-11-16 — finite-difference diagnosis and method change (user decision)

- **Diagnostic 2 (gold-sft):** the same FP32 attributions checked against FP32 ε=0.01 / FP32 ε=0.1 / FP64 ε=0.01 references. Max relative error 60.8 / 0.151 / 0.078; adapters agree to ≤ 0.0005 in FP64. The attribution is correct, and the FP32 reference could not resolve single-neuron effects. Full per-check table: `06a-r3-fd-diagnosis.md`.
- **Decision (user):** the qualification's finite-difference reference runs in FP64 (ε = 0.01), with the tolerances unchanged.
- **Implemented:**
  - `finite_difference_check(reference_model=…)`: every perturbed forward pass (weights, activations, gates, log-softmax, metric means) runs on an FP64 copy of the arm; the attribution stays the FP32 instrument's. The reference is verified to be FP64 and to hold the same adapter weights. The result records its precision and which part ran in which precision.
  - `Stages._fd_check` measures the check's peak memory on the device. `fd_memory:<arm>` is a qualification check, so a reference that doesn't fit stops the chain before production.
  - The launch record must state `finite_difference.precision`.
  - The 06a driver runs the same check.
- **Pending:** online-grpo with the FP64 reference (diagnostic 3, given to the user). The FP64 check's memory on the Pod is measured by the supervised Pod run's qualification.

### 2026-11-16 — diagnostic 3 (online-grpo): the FP64 reference still fails; decision needed

- **Pod check, FP64 reference:**
  - median 0.0021, **max 0.926**, so it fails the proposed max of 0.20;
  - peak memory 27.4 GB on the 40 GB card (fits the 80% rule);
  - the FP32 reference gave median 0.029 and max 1.925.
- **Where it fails:** all six failures are trace-metric single neurons with effects of 6e-8 to 1.3e-5. The absolute errors are 6e-7 to 2.7e-6. That is the FP32 attribution's own rounding floor, which adapter-level trace comparisons also show (~1e-6 to 2e-6 absolute).
- **What is clean:** every adapter (trace and stop) and every stop-pull neuron is at ≤ 0.09. Full table: `06a-r3-fd-diagnosis.md`.
- **Open decision (user):** how to handle checks whose true effect is at the FP32 attribution's precision floor. Options are on this ticket's thread in the conversation of 2026-11-16 and summarised in `06a-r3-fd-diagnosis.md`. No tolerance has been changed.
