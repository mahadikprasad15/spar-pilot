# Record exact top-logit ties as invalid under scorer v2

Approved by the user on September 30, 2026 after a bf16 MMLU run stopped on
`mmlu:clinical_knowledge:161` (B and C both 29.375; gold A).

New logit configs explicitly record `scorer_version: mmlu-logits-v2` and
`logit_tie_policy: invalid`. A tied maximum records all tied letters, no chosen
letter, invalid status and incorrect credit. It remains in the accuracy
denominator and evaluation continues. Reports include tie count and rate.
Nonfinite logits and malformed outputs still stop evaluation.

Configs without these fields retain the original stop policy. Recovery uses a
new plan, preserves source artifacts, and reuses verified raw logits, including
the tied batch saved in the error log. No model, prompts, sample, precision,
batch size or decoding settings change. Completed text and GSM8K runs remain
the selected sources for a combined report. The scoring version is separate
from the unchanged v1 prompt/data protocol.
