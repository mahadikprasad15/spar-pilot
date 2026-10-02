# Flexible v3: preserve the strict final-line contract

The user requested an audit and scoring improvement after the SFT trajectory
showed strict 64.7% versus flexible-v2 1.3%, and explicitly stated that an answer
passing strict should pass flexible. Audit verified the saved 750 pairs and
recomputed existing strict/v2 scores. At step 32, 83 strict-correct outputs were
v2-invalid; at step 64, all 97 strict-correct outputs were v2-invalid. All of
these failures were labelled conflicting_answers. Gold-style solutions put
reasoning and #### on adjacent lines; v2 scans the entire final paragraph and
misclassifies intermediate quantities as competing final answers.

V3 first applies the *existing strict acceptance conditions*, without gold:
one final #### numeric line, exactly one marker, no multiple recognized answer
phrases, and no disagreement with a recognized answer phrase. When accepted,
normalize that final number and return strict_final_line. Otherwise use the
unchanged v2 extraction rules for prose/boxes. Capped outputs remain invalid.
Numeric extraction never receives gold; comparison is a separate operation.

This intentionally inherits strict's scope of contradiction checking. It does
not claim to understand all prose, detect every contradictory claim or validate
reasoning. A valid strict answer remains accepted regardless of paragraph
spacing. Unit tests cover gold-style examples, explicit competing final markers,
contradictory answer phrases, malformed fractions, caps and gold independence.

Leave strict/v1/v2 and prior reports unchanged. Apply v3 to baseline and every
checkpoint, publish a separately named immutable CPU report with source hashes,
per-item decisions and paired intervals. Label this post hoc, because the rule
was revised after viewing outcomes. No model generation or training repeats.
The uploaded paired file alone does not independently verify model/runtime or
training integrity; its score/content consistency and repeated item identities
can be checked. Preserve this evidence boundary in the new report.
