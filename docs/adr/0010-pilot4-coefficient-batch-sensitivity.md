# ADR 0010: Fixed-batch coefficient measurements and descriptive batch sensitivity

Status: accepted by the user on 2026-10-14; target-GPU profile remains required.

## Evidence and decision

The A10040 supplemental profile failed the old coefficient singleton/batch gate at
GRPO step 0: maximum difference 3.325939178466797e-5. Identical batched repeats
agreed exactly. Four probes confirmed matching valid input IDs, attention masks,
and position IDs, while module input activations differed. Independent FP64 dot
products accounted for almost all observed coefficient differences. The first
upstream operation responsible has not been located; this is evidence of batch
sensitivity, not proof that every padding operation is correct.

The user explicitly approved this amendment after the failed gate. Raw FP32
signed A*x on the actual adapted input remains the measurement. Production keeps
identical frozen input batches across SFT and GRPO. Batch-versus-singleton
coefficient differences are descriptive sensitivity evidence, including counts,
maximum differences and bounded worst token/module cases. They are not an
acceptance test or a claim of numerical invariance.

All original tolerance numbers remain unchanged. Repeat and context-chunk
coefficient agreement remain gates. KL agreement, exact mapping/factor identity,
finite/shape checks, independent coefficient/branch reconstruction, zero-adapter
identity and frozen-base integrity remain gates for every compared pass.

## Artifacts and limitations

Use protocol pilot4-token-measurements-v2 and a separately named supplemental
product. Preserve the failed v1 plan and its diagnostic record. Bind the explicit
comparison policy in prepared/frozen plans, profiles and reports. The old code
pin can still inspect v1 evidence; new code rejects incompatible plans.

Profiles retain sealed per-checkpoint sensitivity JSON and the final report
exposes this evidence and its limit. Coefficients cannot be claimed invariant to
batch membership. No SFT/GRPO retraining, behavioural regeneration or fixed-write
rerun is authorized or required by this coefficient-policy amendment. A new
GPU runtime still needs a compatible source execution; this amendment does not
relax the existing hardware/runtime policy.
