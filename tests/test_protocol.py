import pytest

from pilot_eval.protocol import (
    build_gsm8k_prompt,
    build_gsm8k_items,
    build_mmlu_prompt,
    choice_token_ids,
    persist_cohort,
    select_gsm8k_indices,
    select_mmlu_indices,
)


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


def test_gsm8k_prompt_uses_one_user_message_and_chat_template():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages == [{
                "role": "user",
                "content": "Solve the following problem step by step. End your response with a final line in the form #### <number>.\n\nProblem: What is 6 times 12?",
            }]
            return "rendered prompt"

    assert build_gsm8k_prompt("What is 6 times 12?", FakeTokenizer()) == "rendered prompt"


def test_mmlu_prompt_prefills_assistant_after_subject_example():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "continue_final_message": True}
            assert messages == [
                {"role": "user", "content": (
                    "Choose the correct answer. Reply with only A, B, C, or D.\n\n"
                    "Question: 2 + 2?\nA. 1\nB. 4\nC. 5\nD. 9\nAnswer: B\n\n"
                    "Question: 1 + 1?\nA. 1\nB. 2\nC. 3\nD. 4"
                )},
                {"role": "assistant", "content": "Answer:"},
            ]
            return "rendered Answer:"

    example = {"question": "2 + 2?", "choices": ["1", "4", "5", "9"], "answer": "B"}
    prompt = build_mmlu_prompt("1 + 1?", ["1", "2", "3", "4"], [example], FakeTokenizer())

    assert prompt == "rendered Answer:"


def test_mmlu_choice_ids_follow_the_rendered_prefix():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {
                "prompt": [7, 8],
                "prompt A": [7, 8, 11],
                "prompt B": [7, 8, 12],
                "prompt C": [7, 8, 13],
                "prompt D": [7, 8, 14],
            }[text]

    assert choice_token_ids(FakeTokenizer(), "prompt") == {
        "A": 11, "B": 12, "C": 13, "D": 14,
    }


def test_mmlu_choice_validation_rejects_multitoken_continuation():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            ids = {"prompt": [7], "prompt A": [7, 11, 99]}
            return ids.get(text, [7, 12])

    with pytest.raises(ValueError, match="one token"):
        choice_token_ids(FakeTokenizer(), "prompt")


def test_frozen_cohort_rejects_changed_indices(tmp_path):
    first = persist_cohort(
        tmp_path / "artifacts", dataset="gsm8k", revision="rev-1",
        name="test-150", indices=[1, 4, 7], seed=42,
    )

    assert first["indices"] == [1, 4, 7]
    assert persist_cohort(
        tmp_path / "artifacts", dataset="gsm8k", revision="rev-1",
        name="test-150", indices=[1, 4, 7], seed=42,
    ) == first
    with pytest.raises(ValueError, match="cohort mismatch"):
        persist_cohort(
            tmp_path / "artifacts", dataset="gsm8k", revision="rev-1",
            name="test-150", indices=[1, 4, 8], seed=42,
        )


def test_gsm8k_items_keep_source_index_and_gold_solution():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "CHAT:" + messages[0]["content"]

    rows = [
        {"question": "one?", "answer": "work\n#### 1"},
        {"question": "two?", "answer": "work\n#### 2"},
    ]

    items = build_gsm8k_items(rows, [1], FakeTokenizer())

    assert items == [{
        "id": "gsm8k:test:1", "source_index": 1,
        "prompt": "CHAT:Solve the following problem step by step. End your response with a final line in the form #### <number>.\n\nProblem: two?",
        "gold": "work\n#### 2",
    }]
