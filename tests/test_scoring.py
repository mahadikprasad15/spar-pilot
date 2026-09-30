import pytest

from pilot_eval.scoring import score_gsm8k, score_mmlu_text


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
