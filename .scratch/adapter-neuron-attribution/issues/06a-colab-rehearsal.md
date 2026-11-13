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
