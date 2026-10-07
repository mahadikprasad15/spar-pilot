"""Real offline model checks at the public training-window boundary."""
import math
import pytest


def test_group_advantages_use_sample_std_and_dead_groups_are_zero():
    from pilot_eval.grpo_algorithm import group_advantages
    advantages = group_advantages([0, 0, 0, 0, 1, 1, 1, 1] + [1] * 8)
    expected = .5 / (math.sqrt(2 / 7) + .0001)
    assert advantages[:4] == pytest.approx([-expected] * 4)
    assert advantages[4:8] == pytest.approx([expected] * 4)
    assert advantages[8:] == [0] * 8
    assert sum(advantages[:8]) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        group_advantages([1] * 7)
