import math
import pytest

from pilot_eval.scoring import score_gsm8k, score_mmlu_logits, score_mmlu_text


def test_gsm8k_final_marker_scores_strict_and_flexible():
    result = score_gsm8k("Work: 48 + 24 = 72.\n#### 72", "#### 72")

    assert result["strict"]["correct"] is True
    assert result["flexible"]["correct"] is True


def test_gsm8k_answer_phrase_is_flexible_only():
    result = score_gsm8k("I get 6 times 12. Final answer: 72", "#### 72")

    assert result["strict"]["correct"] is False
    assert result["flexible"]["correct"] is True


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("#### 1,200", "#### 1200"),
        ("#### 1/2", "#### 0.5"),
        ("#### -0.50", "#### -1/2"),
    ],
)
def test_gsm8k_compares_exact_numeric_values(response, gold):
    assert score_gsm8k(response, gold)["strict"]["correct"] is True


def test_gsm8k_flexible_accepts_only_number_on_final_line():
    result = score_gsm8k("6 times 12 equals 72.\n72.", "#### 72")

    assert result["strict"]["correct"] is False
    assert result["flexible"]["correct"] is True


def test_gsm8k_multiple_final_markers_are_invalid():
    result = score_gsm8k("#### 12\n#### 72", "#### 72")

    assert result["strict"]["status"] == "invalid"
    assert result["flexible"]["correct"] is False


def test_mmlu_generated_letter_is_scored():
    result = score_mmlu_text(" B", "B")

    assert result == {"choice": "B", "status": "valid", "correct": True}


def test_mmlu_answer_phrase_is_scored():
    assert score_mmlu_text("The answer is C.", "C")["correct"] is True


def test_mmlu_choice_can_precede_an_explanation():
    assert score_mmlu_text("B. Because the second option follows the rule.", "B")["correct"] is True


def test_mmlu_logits_choose_highest_scored_letter():
    result = score_mmlu_logits({"A": 0.2, "B": 1.4, "C": 0.1, "D": -0.3}, "B")

    assert result == {"choice": "B", "status": "valid", "correct": True}


def test_mmlu_logit_tie_stops_scoring():
    with pytest.raises(ValueError, match="tie"):
        score_mmlu_logits({"A": 1.0, "B": 1.0, "C": 0.1, "D": 0.0}, "B")
@pytest.mark.parametrize("text", ["#### 1/0", "Final answer: 1/0", "Final answer: 1; Final answer: 2", "#### 1\nFinal answer: 2"])
def test_malformed_or_contradictory_numeric_answers_are_invalid(text):
    from pilot_eval.scoring import score_gsm8k
    assert score_gsm8k(text, "#### 2")["flexible"]["status"] == "invalid"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_logits_stop_scoring(value):
    from pilot_eval.scoring import score_mmlu_logits
    with pytest.raises(ValueError, match="finite"):
        score_mmlu_logits(dict(A=value, B=1., C=0., D=0.), "B")


@pytest.mark.parametrize("text", ["C ... actually B", "A or B", "No answer"])
def test_mmlu_ambiguous_or_missing_answers_remain_incorrect(text):
    assert score_mmlu_text(text, "B")["correct"] is False


@pytest.mark.parametrize("text", ["#### 50%", "#### 5 kg", "Reasoning gives 5", "#### 5\nMore text"])
def test_gsm8k_rejects_units_percent_and_nonfinal_numbers(text):
    result = score_gsm8k(text, "#### 5")
    assert result["strict"]["correct"] is False
    assert result["flexible"]["correct"] is False
