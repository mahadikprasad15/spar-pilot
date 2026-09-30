from pilot_eval.protocol import select_gsm8k_indices, select_mmlu_indices


def test_gsm8k_cohort_selection_is_repeatable_and_unique():
    first = select_gsm8k_indices(total=10, count=3, seed=42)
    second = select_gsm8k_indices(total=10, count=3, seed=42)

    assert first == second
    assert len(first) == len(set(first)) == 3
    assert all(0 <= index < 10 for index in first)


def test_mmlu_cohort_balances_subjects_and_is_repeatable():
    sizes = {"biology": 8, "history": 6}
    first = select_mmlu_indices(sizes, count_per_subject=2, seed=42)
    second = select_mmlu_indices(sizes, count_per_subject=2, seed=42)

    assert first == second
    assert set(first) == set(sizes)
    assert all(len(set(indices)) == 2 for indices in first.values())
