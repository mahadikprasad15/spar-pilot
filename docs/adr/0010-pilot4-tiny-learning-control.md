# Pilot 4 tiny learning-control revisions

Status: approved by user, October 7, 2026.

The random frozen output layer restricted attainable binary reward. V2 uses
an output-row difference of unit norm orthogonal to the initial prompt hidden
state, giving an equal-probability start; the output layer stays frozen.
V2 still failed the early-versus-late window rule because learning already
occurred inside steps 1-5. V3 compares exact pre-training expected reward with
mean rollout reward at steps 16-20. The two-token reward expectation equals
probability of token 1. Both seeds and signs must change by at least 0.2.
Task, training settings and threshold remain fixed. Both revisions were
approved before their acceptance runs; earlier failures remain preserved.
This is a post-diagnosis engineering-test revision, not scientific evidence.
