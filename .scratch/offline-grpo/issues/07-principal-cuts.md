# 07: Principal cuts and paired evaluation

Status: ready-for-agent
Blocked by: 04 Behaviour and offline source handoff; 06 Internal measurements and comparisons

## What to build

Apply the sealed sampling-SFT cut definitions to the offline checkpoint and evaluate the resulting adapters on the paired held-out cohort using the established evaluation contract.

## Acceptance criteria

- [ ] All eight conditions are included: original, principal-output cut, principal-input cut, three random-direction controls, output magnitude control and input magnitude control; ranks and construction match the sealed plan.
- [ ] The paired evaluation covers 150 questions per condition, retaining answers and scores; reuse the original endpoint only when its complete evaluation contract matches.
- [ ] CPU fixtures verify condition and control identities, cohort pairing and endpoint reuse decisions; constructing adapters or passing fixtures alone is not evidence of behavioural effects.
