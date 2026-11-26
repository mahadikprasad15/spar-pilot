# 04: Ablation engine and condition builder

**Status:** resolved (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_ablation.py`.

- [ ] Removal via g = 0 for any set; equal to an edited checkpoint (chosen b zeroed, hashed).
- [ ] Matched controls per target: same module type, write tolerance, without replacement, seeded, excluding both metrics' top-10 sets; group = union of per-target matches; unavailable recorded, never loosened.
- [ ] Natural 64-term read (no rescaling) by a given neuron order; constant read c̄ at every unpadded position.
- [ ] Builder reads only a sealed freeze; refuses confirmation- or cohort-derived inputs.
- [ ] Generation from edited checkpoints with a frozen batch manifest (left padding, per-sequence stop).
- [ ] Tests: all-neuron read == full; empty removal == full; constant read writes s·b·c̄; checkpoint == gate; sampler exclusions, determinism, unavailable path; manifest batch order reproduced.

## Comments

### 2026-11-13 — question for the user (implementing agent)

- **Seeds for the 3 random neuron orders and the 1,000-order null.** The spec and candidate plan say "3 seeded random orders" and "1,000 seeded orders" but give no seeds. The only seeds the plan lists are 20000 (matched controls) and 1234 (bootstrap). I have not chosen any: `neuron_orders` and `build_freeze` require the three seeds as inputs and refuse a missing value, so the launch record must state them. Please give the seeds, or a rule such as "20001–20003; null orders from `default_rng(20004)`".

## Answer

2026-11-13 — `pilot_eval/attribution_ablation.py`, tests `tests/test_attribution_ablation.py` (15, tiny random Qwen2 + PEFT 0.18.0, CPU).

- **Removal** (`removal`) is a forward hook on `lora_B` that multiplies its output by 0, i.e. gate g = 0 on the scaled branch. An empty removal leaves logits bit-identical. `edit_checkpoint` writes a copy with the chosen b zeroed and records source, edited and config hashes; `verify_edit` re-checks it. Loading the edited checkpoint gives the same logits as the gate (atol 1e-6) and the same greedy generations, in the frozen manifest's batch order.
- **Natural read** (`natural_read`): c_t → Σ_{j∈keep} a_j z_{j,t}, unrescaled (checked against z and a). All neurons gives the full model; no neurons gives removal.
- **Constant read** (`constant_read`): c_t → c̄ everywhere. The branch output is s·b·c̄. With right padding the padded positions cannot affect real ones.
- **Selection.** `top_k` ranks by |mean attribution| with ties broken by index (layer, then module order). `discovery_write` = |s|·‖b‖·mean |c_t|. `matched_controls` picks, per target, the same module type within ±tolerance, outside both metrics' top-10 sets, without replacement, seeded. Unavailable matches are recorded and never loosened; any unavailable member makes the group unavailable. `neuron_orders` gives three orders:
  - by signed attribution in the direction of the adapter's own predicted effect;
  - by term size |a_j| Σ_t |z_j| (pass 2 now saves this);
  - three seeded permutations.
- **Freeze.** `build_freeze` refuses any input not from discovery and records the settings and analysis block. `seal_freeze` writes the sealed run directory `attribution/freeze`. `conditions` reads only a sealed freeze and builds at most 19 conditions per arm and metric:
  - 5 top singles, 5 matched singles, top-10 group and matched group;
  - full removal of the top `down_proj` adapter when it is not one of the singles;
  - five 64-term reads and the constant read.
  An unavailable match is a `skipped` condition with reason `matched-control-unavailable`, never an error.
- **Stage C** (`StageC.evaluate`) gives per-example ΔL, ΔS and KL(full arm ‖ condition) against the same arm unablated: two FP32 forwards, logits only at read positions in chunks, KL summed over the vocabulary in FP64. Values match the full-logit `attribution_metrics` path; an empty stop mask gives ΔS = NaN.
- The read size (64), tolerance (0.20) and control seed (20000) are passed in from the launch record and are not defaults here. The random-order seeds are open (see the question above).
- Stage D's ADR 0001 cohort prompts are built in the chain from `protocol.build_gsm8k_items` and the pinned tokenizer (ticket 11).

### 2026-11-14 — user decision

- Random neuron orders use seeds 20001, 20002, 20003, and the 1,000-order concentration null uses seed 20004; all in the 2000x selection block next to the matched-control seed 20000. They are recorded in `launch-record-draft.yaml` and the 06a notebook.
