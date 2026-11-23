# 08: Checkpoints, resume and storage policy

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 07
**Group:** E

Checkpoints at 0/8/16/32/64/128/256/512/768/960/1024 with the D24 storage policy; resume bound to plan, runtime and code commit; resume from a fresh directory using the volume copy. Reuse `sampling_sft_training.checkpoint` / `seal` / `load` and `rng_state` / `restore_rng`.

- [ ] Red/green: spec tests 11 and 21.
- [ ] Expected checkpoints listed in the plan and compared on resume.

## Comments

## Answer
