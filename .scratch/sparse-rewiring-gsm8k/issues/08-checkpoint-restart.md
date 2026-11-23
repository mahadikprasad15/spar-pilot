# 08: Checkpoints and restart

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 07
**Group:** E

Checkpoints at 0/8/16/32/64/128/256/512/768/960/1024 holding only read factors, `logα` and a seal (D24, decided). No mid-run resume: an interrupted arm restarts from step 0; a completed, verified arm is kept. Reuse `sampling_sft_training.checkpoint` / `seal` for writing and sealing; its optimizer and RNG recovery paths are not used.

- [ ] Red/green: spec tests 11 and 21.
- [ ] Expected checkpoints listed in the plan; a restored completed arm is verified against them.

## Comments

## Answer
