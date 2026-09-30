"""Run configuration validation."""

import re
from pathlib import Path


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
        if Path(config["adapter"]).is_absolute():
            if not re.fullmatch(r"[0-9a-f]{64}", config.get("adapter_sha256") or ""):
                raise ValueError("local adapter requires adapter_sha256")
        elif not _SHA.fullmatch(config.get("adapter_revision") or ""):
            raise ValueError("adapter_revision must be an exact 40-character commit SHA")
    elif config.get("adapter_revision") is not None:
        raise ValueError("adapter_revision must be null when adapter is null")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    if config["seed"] < 0:
        raise ValueError("seed must be non-negative")
    if config["scorer"] not in {"gsm8k", "mmlu_text", "mmlu_logits"}:
        raise ValueError("unsupported scorer")
    if "logit_tie_policy" in config or "scorer_version" in config:
        if (config["scorer"] != "mmlu_logits" or config.get("logit_tie_policy") != "invalid"
                or config.get("scorer_version") != "mmlu-logits-v2"):
            raise ValueError("revised logit policy requires mmlu-logits-v2 and invalid ties")
    dataset = "gsm8k" if config["scorer"] == "gsm8k" else "mmlu"
    if config["dataset"] != dataset or config["prompt_template"] != f"{dataset}-v1":
        raise ValueError("dataset or prompt template does not match scorer")
    if config.get("shots", 0) not in (0, 5):
        raise ValueError("pilot supports zero or five shots")
    if config["decoding"].get("do_sample") is not False:
        raise ValueError("sampled decoding requires a future protocol")
    if config["scorer"] != "mmlu_logits" and config["decoding"].get("max_new_tokens") != (1024 if dataset == "gsm8k" else 32):
        raise ValueError("generation cap does not match pilot protocol")
    if config["dtype"] not in ("bfloat16", "float16") or config["quantization"] is not None:
        raise ValueError("pilot requires unquantized bf16 or fp16")
    if config["deterministic"] is not True:
        raise ValueError("deterministic execution required")
    if config.get("checkpoint_interval_batches", 1) < 1:
        raise ValueError("checkpoint interval must be positive")
    return config
