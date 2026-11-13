# 06a: Colab rehearsal

**Status:** needs-triage — draft v2
**Type:** task
**Blocked by:** 01–05
**Authorization:** user go-ahead and a Colab cap. Outputs labelled `rehearsal-colab`; never feed production.

**What to build and run:** `notebooks/adapter-attribution-colab.ipynb` on calibration rows and the 20 rehearsal generation items only, for `gold-sft`, `online-grpo` and untuned.

- [ ] Environment per the joint audit's Colab lessons (stream child output, remove conflicting preinstalled packages, pinned tokenizer pad ID, record runtime).
- [ ] Base traces for calibration rows via 01.
- [ ] Covering set: no-op hook; single removal; group removal; full `down_proj` removal; natural 64-term read; constant read; full vs edited-checkpoint generation (2 items); a corrupted and an interrupted unit.
- [ ] Real-model finite differences (3 adapters, 3 neurons per arm) and the Σ_j identity.
- [ ] First token-budget probe from 2k upward with singleton and longest examples; record OOMs.
- [ ] Online upload and readback of this study's partial outputs; child-process logs captured.
- [ ] Every failure fixed with a regression test and recorded in `docs/agents/experiment-lessons.md`.

## Comments
