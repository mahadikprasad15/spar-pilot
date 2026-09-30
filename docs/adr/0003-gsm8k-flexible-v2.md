# Deterministic offline GSM8K flexible v2

Approved by the user after reviewing the GSM8K extraction diagnosis. Strict and
flexible v1 stay unchanged. V2 is a separate, post hoc measurement.

Extraction never receives the gold answer. It inspects the final paragraph and
contiguous immediately preceding paragraphs containing explicit answer cues
(`####`, `\boxed{`, `Final answer:`, `answer is`). A final-answer phrase delimits
the conclusion from earlier calculations in the same paragraph. Explicit
markers retain surrounding final-paragraph prose so conflicting answers are
not silently overridden.

The region must contain one distinct numeric value; repeated equal values are
allowed. Units, currency symbols and prose can surround it. Context quantities
are not interpreted semantically: `36 hours in 4 weeks` remains invalid.
Numbers use the existing exact integer/decimal/fraction grammar. Spaces after
thousands commas are allowed only before a three-digit group. Comparison uses
exact rational values, so 3.0 equals 3 and 1/2 equals 0.5. Percent suffixes are
units (5% is the answer quantity 5, not 0.05). Spelled-out numbers, scientific
notation and LaTeX fractions are outside this initial grammar. Arithmetic
expressions, malformed numbers/boxes, conflicting or ambiguous quantities,
missing numbers and capped responses are invalid and incorrect. V2 does not
claim to validate the reasoning or to understand all natural-language answers.

Each result records version, normalized extracted answer, extraction rule,
status, invalid reason and correctness. Offline rescoring verifies source
responses and old scores, copies source files into a new immutable report,
records source hashes/config and preserves every old/new item score. The
original run, combined report and model responses are untouched. Every item
remains in the denominator. The 46% diagnostic review is not an acceptance
target, and it was not blinded. Results are explicitly labelled post hoc.
