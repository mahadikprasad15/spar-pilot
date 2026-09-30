import pytest

from pilot_eval.scoring import score_gsm8k


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
