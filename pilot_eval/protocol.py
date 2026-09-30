"""Public cohort and prompt protocol."""

import json
import os
import random
from pathlib import Path


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


def build_mmlu_prompt(
    question: str, choices: list[str], examples: list[dict], tokenizer
) -> str:
    """Render MMLU v1 with an open assistant-side answer prefix."""
    def formatted_item(item_question: str, item_choices: list[str]) -> str:
        if len(item_choices) != 4:
            raise ValueError("MMLU requires four choices")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", item_choices))
        return f"Question: {item_question}\n{options}"

    parts = ["Choose the correct answer. Reply with only A, B, C, or D."]
    for example in examples:
        parts.append(
            formatted_item(example["question"], example["choices"])
            + f"\nAnswer: {example['answer']}"
        )
    parts.append(formatted_item(question, choices))
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "\n\n".join(parts)},
            {"role": "assistant", "content": "Answer:"},
        ],
        tokenize=False,
        continue_final_message=True,
    )


def choice_token_ids(tokenizer, prompt: str) -> dict[str, int]:
    """Get A/B/C/D continuation IDs after the rendered MMLU prompt."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    choices = {}
    for letter in "ABCD":
        full = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
        if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"{letter} continuation must be exactly one token in context")
        choices[letter] = full[-1]
    if len(set(choices.values())) != 4:
        raise ValueError("A/B/C/D continuations must use distinct token IDs")
    return choices


def persist_cohort(
    output_root: Path, *, dataset: str, revision: str, name: str,
    indices: list[int] | dict[str, list[int]], seed: int,
) -> dict:
    """Save a cohort once and reject later changes to its membership."""
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "name": name,
        "seed": seed,
        "selection_method": "python-random-sample-without-replacement-sorted",
        "indices": indices,
    }
    path = Path(output_root) / "cohorts" / dataset / revision / name / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != manifest:
            raise ValueError("cohort mismatch with frozen manifest")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return manifest
