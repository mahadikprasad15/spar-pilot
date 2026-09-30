from pilot_eval.protocol import build_gsm8k_prompt, select_gsm8k_indices, select_mmlu_indices


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
