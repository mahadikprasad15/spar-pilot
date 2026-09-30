"""Run configuration validation."""

import re


_SHA = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED = {
    "run_id", "experiment", "model", "model_revision", "tokenizer_revision",
    "adapter", "dataset", "dataset_path", "dataset_config", "dataset_revision",
    "evaluation_split", "fewshot_split", "cohort", "variant",
    "prompt_template", "seed", "scorer", "decoding", "batch_size", "dtype",
    "attention_implementation", "quantization", "deterministic",
}


def validate_config(config: dict) -> dict:
    """Reject configurations that cannot identify a reproducible run."""
    missing = sorted(_REQUIRED - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    for field in ("model_revision", "tokenizer_revision", "dataset_revision"):
        if not isinstance(config[field], str) or not _SHA.fullmatch(config[field]):
            raise ValueError(f"{field} must be an exact 40-character commit SHA")
    if config["adapter"]:
        if not _SHA.fullmatch(config.get("adapter_revision", "")):
            raise ValueError("adapter_revision must be an exact 40-character commit SHA")
    elif config.get("adapter_revision") is not None:
        raise ValueError("adapter_revision must be null when adapter is null")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    if config["seed"] < 0:
        raise ValueError("seed must be non-negative")
    if config["scorer"] not in {"gsm8k", "mmlu_text", "mmlu_logits"}:
        raise ValueError("unsupported scorer")
    return config
