import pytest

from pilot_eval.config import validate_config


def test_config_requires_exact_revisions_and_run_provenance():
    config = {
        "run_id": "20260930T120000Z-a1b2c3",
        "experiment": "pilot-1",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "adapter": None,
        "dataset": "gsm8k",
        "dataset_path": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": "b" * 40,
        "evaluation_split": "test",
        "fewshot_split": None,
        "cohort": "test-150-seed-42",
        "variant": "baseline",
        "prompt_template": "gsm8k-v1",
        "seed": 42,
        "scorer": "gsm8k",
        "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 4,
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "quantization": None,
        "deterministic": True,
    }

    assert validate_config(config) == config

    config["model_revision"] = "main"
    with pytest.raises(ValueError, match="model_revision"):
        validate_config(config)


def test_adapter_revision_is_required_only_for_adapter_runs():
    config = {
        "run_id": "20260930T120000Z-a1b2c3", "experiment": "pilot-1",
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40, "adapter": "org/adapter",
        "dataset": "gsm8k", "dataset_path": "openai/gsm8k", "dataset_config": "main",
        "dataset_revision": "b" * 40, "evaluation_split": "test", "fewshot_split": None,
        "cohort": "test-150-seed-42", "variant": "adapter", "prompt_template": "gsm8k-v1",
        "seed": 42, "scorer": "gsm8k",
        "decoding": {"do_sample": False, "max_new_tokens": 1024}, "batch_size": 4,
        "dtype": "bfloat16", "attention_implementation": "eager",
        "quantization": None, "deterministic": True,
    }

    with pytest.raises(ValueError, match="adapter_revision"):
        validate_config(config)
