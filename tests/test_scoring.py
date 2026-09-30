from pilot_eval.scoring import score_gsm8k


def test_gsm8k_final_marker_scores_strict_and_flexible():
    result = score_gsm8k("Work: 48 + 24 = 72.\n#### 72", "#### 72")

    assert result["strict"]["correct"] is True
    assert result["flexible"]["correct"] is True
