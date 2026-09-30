"""Hugging Face model boundary for GPU evaluation."""

from dataclasses import dataclass


@dataclass
class HFBackend:
    model: object
    tokenizer: object


def _default_dependencies():
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return type("Dependencies", (), {
        "tokenizer_factory": AutoTokenizer,
        "model_factory": AutoModelForCausalLM,
        "adapter_config_factory": PeftConfig,
        "adapter_model_factory": PeftModel,
        "dtype_values": {"bfloat16": torch.bfloat16, "float16": torch.float16},
        "set_deterministic": staticmethod(torch.use_deterministic_algorithms),
    })()


def load_hf_backend(config: dict, dependencies=None) -> HFBackend:
    """Load one pinned causal LM and an optional compatible PEFT adapter."""
    dependencies = dependencies or _default_dependencies()
    dependencies.set_deterministic(config["deterministic"])
    tokenizer = dependencies.tokenizer_factory.from_pretrained(
        config["model"], revision=config["tokenizer_revision"],
    )
    model = dependencies.model_factory.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        torch_dtype=dependencies.dtype_values[config["dtype"]],
        attn_implementation=config["attention_implementation"],
        device_map="auto",
    )
    if config.get("adapter"):
        adapter_config = dependencies.adapter_config_factory.from_pretrained(
            config["adapter"], revision=config["adapter_revision"],
        )
        if adapter_config.base_model_name_or_path != config["model"]:
            raise ValueError(
                f"adapter base model {adapter_config.base_model_name_or_path!r} does not match "
                f"configured base model {config['model']!r}"
            )
        model = dependencies.adapter_model_factory.from_pretrained(
            model, config["adapter"], revision=config["adapter_revision"], is_trainable=False,
        )
    model.eval()
    return HFBackend(model=model, tokenizer=tokenizer)
