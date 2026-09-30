import pytest


@pytest.mark.parametrize("text,gold,answer", [
    ("Final answer: It takes 3 bolts.", "3", "3"),
    ("Final answer: 8 centimeters shorter.", "8", "8"),
    ("The total is $140. #### 140", "140", "140"),
    ("#### 8, 000", "8000", "8000"),
    ("Work gives 99.\n\n\\boxed{160}", "160", "160"),
    ("Therefore, the answer is 3.0.", "3", "3"),
    ("Therefore, Melanie started with 18 vacuum cleaners.", "18", "18"),
    ("Earlier reasoning used 72.\n\nFinal answer: $1,400.", "1400", "1400"),
    ("Final answer: 1/2 liters.", "0.5", "1/2"),
    ("Final answer: -0.50.", "-1/2", "-1/2"),
])
def test_v2_accepts_final_numeric_answers(text, gold, answer):
    from pilot_eval.scoring import score_gsm8k_flexible_v2
    result = score_gsm8k_flexible_v2(text, "#### " + gold)
    assert result["status"] == "valid"
    assert result["correct"] is True
    assert result["extracted"] == answer
    assert result["scorer_version"] == "gsm8k-flexible-v2"
    assert result["extraction_rule"]


@pytest.mark.parametrize("text", [
    "Final answer: 36 hours in 4 weeks.",
    "Final answer: 140. #### 160",
    "Final answer: 140.\n\n#### 160",
    "#### 12\n#### 72",
    "Earlier reasoning gives 72.\n\nI cannot decide.",
    "Final answer: 12,34 dollars.",
    "Final answer: 1/0",
    r"\boxed{x + 3}",
    "Final answer: 2 + 3 = 5",
    "",
])
def test_v2_rejects_ambiguous_missing_and_malformed_answers(text):
    from pilot_eval.scoring import score_gsm8k_flexible_v2
    result = score_gsm8k_flexible_v2(text, "#### 72")
    assert result["status"] == "invalid"
    assert result["correct"] is False
    assert result["invalid_reason"]


def test_v2_extraction_is_independent_of_gold_and_caps_are_invalid():
    from pilot_eval.scoring import score_gsm8k_flexible_v2
    right = score_gsm8k_flexible_v2("Final answer: 3 bolts.", "#### 3")
    wrong = score_gsm8k_flexible_v2("Final answer: 3 bolts.", "#### 4")
    assert {k: v for k, v in right.items() if k != "correct"} == {k: v for k, v in wrong.items() if k != "correct"}
    assert wrong["correct"] is False
    assert score_gsm8k_flexible_v2("#### 3", "#### 3", capped=True)["status"] == "invalid"
