"""Public cohort and prompt protocol."""

import random


def select_gsm8k_indices(total: int, count: int = 150, seed: int = 42) -> list[int]:
    """Select a stable, ordered GSM8K evaluation cohort."""
    return sorted(random.Random(seed).sample(range(total), count))


def select_mmlu_indices(
    subject_sizes: dict[str, int], count_per_subject: int = 20, seed: int = 42
) -> dict[str, list[int]]:
    """Select an equal-size cohort from each MMLU subject."""
    rng = random.Random(seed)
    return {
        subject: sorted(rng.sample(range(subject_sizes[subject]), count_per_subject))
        for subject in sorted(subject_sizes)
    }
