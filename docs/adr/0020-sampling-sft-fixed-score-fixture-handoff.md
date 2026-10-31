# 0020: Fixed score fixtures and staged qualification handoff

Date: 2026-10-31

The A40 four-problem k0 smoke completed with zero engine errors. It produced17
extension requests, fewer than the20 required by T3, and no capped response.
The user approved building the qualification handoff, including a deliberately
capped case. Do not duplicate, hand-edit or relabel old requests to pass T3.

Add an instrument mode `fixtures`:20 predeclared distinct problems, k0,
empty-prefix generation requests with max_tokens1024. The handoff fixes the20
smallest sorted ordinals in the existing seed0,32-problem pilot cohort before
launch. Nineteen requests retain ordinary stopping. The last request records
qualification_min_tokens1024; the vLLM adapter forwards this to min_tokens,
which suppresses stopping until the requested cap. Require exactly1024 returned
tokens and no stop IDs; failure replays the identical batch once then stops.

These are scoring fixtures, not observations of ordinary model stopping or
training candidates. Set training_eligible=false and retain the exact forced
control in the generation record. No normal sampler request changes. A malformed
forced control is rejected. All scoring uses the original scorer prompt and raw
model probabilities, including actual stop tokens where present. Retokenize the
scorer prompts independently in real execution.

Freeze all20 traces before either scorer runs. Label the shortest observed
length(s) short and longest length(s) long, by a predeclared length-only rule.
Require distinct short/long lengths and at least one finished natural request.
Keep every selected problem; no filtering on correctness, score agreement or
visual appeal. Retain the real log coordinate, request, IDs and sealed source.
T3's mean0.001 and per-token0.01 limits and exact alignment are unchanged.

The explicit qualification-stage command prepares/runs one stage at a time:
fixtures, four-problem k2, the two scorers, one fixed scheduler profile,
selection, and the32-problem k5 uninterrupted/kill/resume/compare workflow.
A single profile qualifies the supplied configuration, not its optimality;
additional scheduler candidates remain separately measurable. No automatic
hardware, context, prompt, tolerance or algorithm fallback is permitted.

Preparation is CPU-only. --run explicitly executes the paid stage under the
existing reviewed waiver, source/Pod/storage checks and timeout. Preserve the
SIGKILL receipt and verify dead ownership before recovery. CPU evidence cannot
qualify real-CUDA replay. Every scientific stage is still bounded; neither full
512-arm generation nor training is launched. Export/independent download and
Pod-stop completion remain separate explicit lifecycle steps.
