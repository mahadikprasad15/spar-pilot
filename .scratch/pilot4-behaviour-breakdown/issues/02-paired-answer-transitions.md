# 02: Explain endpoint gains and losses with paired transition reports

**What to build:** A saved, inspectable report of which questions improve or
regress between checkpoints 0 and 64, alongside direct paired method comparisons
and SFT churn/length descriptions. The notebook displays the report from saved
inputs, with exact item-level evidence behind every count.

**Blocked by:** 01 — verified source and harmonized scoring workflow.

**Status:** ready-for-agent

**Parent:** Behavioural gain breakdown specification.

- [ ] Consume verified scoring products and retain all source/parent identities;
      stale or incompatible inputs cannot be accepted silently.
- [ ] Produce strict and flexible-v3 correct/wrong/invalid 3×3 tables for SFT
      and GRPO, with last-number diagnostic tables and contributing question IDs.
- [ ] Report gross gains, gross losses, churn and net change. Net W-related and
      U-related flows sum to the total change; never divide gross gains by net
      +12 and interpret the result as a causal percentage of the gain.
- [ ] Show within-method endpoint differences and the direct paired
      GRPO64-minus-SFT64 accuracy difference with clearly named cohorts.
- [ ] Use 10,000 paired question-ID percentile-bootstrap resamples, seed 42 and
      the existing interval convention; record denominators and resampling
      identity. Keep both records of a pair together.
- [ ] Retain distinct baselines and disclose overlap-cohort restrictions or
      compatibility limits. Do not infer an arm difference from two separate
      within-arm intervals or mix decoding uncertainty with training-seed variation.
- [ ] Publish SFT gains/losses and descriptive views of endpoint response lengths,
      exact gold calculator-annotation counts and nonempty-line counts. Include
      denominators and do not call these causal or validated difficulty measures.
- [ ] Save per-question rows, structured totals, tables, readable plots, receipts
      and explicit completion state; a fresh notebook session restores them.
- [ ] Hand-computable fixtures at the public command seam verify flow accounting,
      paired alignment/resampling, partial-overlap disclosure, distinct baselines,
      source-change rejection and idempotent report reconstruction.
- [ ] Automatic answer transitions are not labelled validated reasoning gains.
