import pytest

from pilot_eval.backend import HFBackend, load_hf_backend


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


def test_backend_generates_only_new_tokens_and_reports_eos():
    class Batch(dict):
        def to(self, device):
            return self

    class InputIds:
        shape = (1, 3)

    class Row(list):
        def tolist(self):
            return list(self)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, prompts, **kwargs):
            assert kwargs == {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
            return Batch(input_ids=InputIds())

        def decode(self, token_ids, skip_special_tokens):
            assert skip_special_tokens is True
            return "answer"

    class Model:
        device = "cuda"

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 4
            return [Row([1, 2, 3, 7, 8, 99])]

    outputs = HFBackend(Model(), Tokenizer()).generate_batch(
        ["prompt"], {"do_sample": False, "max_new_tokens": 4}
    )

    assert outputs == [{"text": "answer", "token_count": 2, "stop_reason": "eos"}]


def test_backend_reads_four_choice_logits_at_next_token():
    class Batch(dict):
        def to(self, device):
            return self

    class Value:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Logits:
        def __getitem__(self, key):
            row, position, token_id = key
            assert position == -1
            return Value(row * 100 + token_id)

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert prompts == ["p1", "p2"]
            return Batch(input_ids="ids", attention_mask="mask")

    class Model:
        device = "cuda"

        def __call__(self, **inputs):
            return type("Output", (), {"logits": Logits()})()

    scores = HFBackend(Model(), Tokenizer()).choice_logits_batch(
        ["p1", "p2"],
        [
            {"A": 11, "B": 12, "C": 13, "D": 14},
            {"A": 21, "B": 22, "C": 23, "D": 24},
        ],
    )

    assert scores == [
        {"A": 11, "B": 12, "C": 13, "D": 14},
        {"A": 121, "B": 122, "C": 123, "D": 124},
    ]


def test_loader_sets_left_padding_and_inference_context():
    from contextlib import contextmanager
    entered = []
    @contextmanager
    def inference():
        entered.append(True)
        yield
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 99})()
    model = type("Model", (), {"eval": lambda self: self})()
    dependencies = type("Dependencies", (), {
        "tokenizer_factory": _Factory(tokenizer), "model_factory": _Factory(model),
        "dtype_values": {"bfloat16": "bf16"},
        "set_deterministic": staticmethod(lambda enabled: None),
        "inference_context": staticmethod(inference),
    })()
    config = dict(model="org/model", model_revision="a"*40, tokenizer_revision="a"*40,
                  adapter=None, dtype="bfloat16", attention_implementation="eager", deterministic=True)
    backend = load_hf_backend(config, dependencies)
    assert tokenizer.padding_side == "left"
    with backend.inference_context():
        pass
    assert entered == [True]
