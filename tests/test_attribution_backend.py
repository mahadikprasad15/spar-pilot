"""Ticket 11: pure parts of the real backend (CPU; the CUDA loader itself runs only in 06a/06b)."""
import pytest

from pilot_eval import attribution_backend as ab
from pilot_eval.protocol import select_gsm8k_indices


def test_cohort_is_adr_0001():
    assert ab.cohort_indices(1319) == select_gsm8k_indices(1319, 150, 42)


def test_rehearsal_draw_rule_user_2026_11_14():
    import random
    cohort = set(select_gsm8k_indices(1319, 150, 42))
    expected = sorted(random.Random(42).sample([i for i in range(1319) if i not in cohort], 20))
    assert ab.rehearsal_draw(1319) == expected and not cohort & set(expected) and len(set(expected)) == 20


def test_rehearsal_ids_must_be_explicit_distinct_and_outside_the_cohort():
    cohort = set(ab.cohort_indices(1319))
    outside = [i for i in range(1319) if i not in cohort][:20]
    assert ab.rehearsal_indices(outside, 1319) == outside
    with pytest.raises(ValueError, match='overlap'):
        ab.rehearsal_indices(outside[:19] + [min(cohort)], 1319)
    for bad in (None, outside[:19], outside[:19] + outside[:1], outside[:19] + [5000]):
        with pytest.raises(ValueError):
            ab.rehearsal_indices(bad, 1319)
