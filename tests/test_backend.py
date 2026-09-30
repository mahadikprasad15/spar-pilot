import pytest

from pilot_eval.backend import load_hf_backend


class _Factory:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def from_pretrained(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.value


def test_loader_rejects_adapter_for_a_different_base_model():
    tokenizer = object()
    model = object()
    adapter_config = type("AdapterConfig", (), {
        "base_model_name_or_path": "another/model",
    })()
    dependencies = type("Dependencies", (), {
        "tokenizer_factory": _Factory(tokenizer),
        "model_factory": _Factory(model),
        "adapter_config_factory": _Factory(adapter_config),
        "adapter_model_factory": _Factory(object()),
        "dtype_values": {"bfloat16": "bf16"},
        "set_deterministic": staticmethod(lambda enabled: None),
    })()
    config = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "adapter": "org/adapter",
        "adapter_revision": "b" * 40,
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "quantization": None,
        "deterministic": True,
    }

    with pytest.raises(ValueError, match="base model"):
        load_hf_backend(config, dependencies)
