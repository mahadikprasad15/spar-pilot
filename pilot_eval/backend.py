"""Hugging Face model boundary for GPU evaluation."""

from dataclasses import dataclass
from contextlib import nullcontext
import os


@dataclass
class HFBackend:
    model: object
    tokenizer: object
    inference_context: object = nullcontext

    def generate_batch(self, prompts: list[str], decoding: dict) -> list[dict]:
        """Generate continuations and return text plus auditable stop metadata."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(self.model.device)
        input_width = inputs["input_ids"].shape[1]
        with self.inference_context():
            generated = self.model.generate(
                **inputs, do_sample=decoding["do_sample"], num_beams=1, num_return_sequences=1,
                max_new_tokens=decoding["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        outputs = []
        for row in generated:
            sliced = row[input_width:]
            continuation = sliced.tolist() if hasattr(sliced, "tolist") else list(sliced)
            eos_index = next(
                (index for index, token_id in enumerate(continuation) if token_id in eos_ids),
                None,
            )
            tokens = continuation if eos_index is None else continuation[:eos_index]
            outputs.append({
                "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "token_count": len(tokens),
                "stop_reason": (
                    "eos" if eos_index is not None
                    else "cap" if len(continuation) >= decoding["max_new_tokens"]
                    else "other"
                ),
            })
        return outputs

    def choice_logits_batch(
        self, prompts: list[str], token_ids: list[dict[str, int]],
    ) -> list[dict[str, float]]:
        """Read next-token logits for each item's validated A/B/C/D tokens."""
        if len(prompts) != len(token_ids):
            raise ValueError("prompts and choice token IDs must have equal length")
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(self.model.device)
        with self.inference_context():
            logits = self.model(**inputs).logits
        return [
            {
                letter: float(logits[row_index, -1, candidate_id].item())
                for letter, candidate_id in item_token_ids.items()
            }
            for row_index, item_token_ids in enumerate(token_ids)
        ]


def _default_dependencies():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
        "inference_context": staticmethod(torch.inference_mode),
        "set_seed": staticmethod(__import__("transformers").set_seed),
    })()


def load_hf_backend(config: dict, dependencies=None) -> HFBackend:
    """Load one pinned causal LM and an optional compatible PEFT adapter."""
    dependencies = dependencies or _default_dependencies()
    dependencies.set_deterministic(config["deterministic"])
    if hasattr(dependencies, "set_seed"):
        dependencies.set_seed(config.get("seed", 42))
    tokenizer = dependencies.tokenizer_factory.from_pretrained(
        config["model"], revision=config["tokenizer_revision"],
    )
    if hasattr(tokenizer, "eos_token_id"):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
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
    return HFBackend(model=model, tokenizer=tokenizer,
                     inference_context=getattr(dependencies, "inference_context", nullcontext))
