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


def build_gsm8k_prompt(question: str, tokenizer) -> str:
    """Render the GSM8K v1 chat prompt."""
    content = (
        "Solve the following problem step by step. End your response with a final line "
        "in the form #### <number>.\n\nProblem: " + question
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
