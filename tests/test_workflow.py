import json
from pathlib import Path

import pytest

from pilot_eval.workflow import prepare_plan, execute_config


class Tokenizer:
    model_max_length = 4096
    eos_token_id = 99
    pad_token_id = 99
    chat_template = "fake chat v1"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        if text.endswith(tuple(" " + letter for letter in "ABCD")):
            return self.encode(text[:-2]) + [10 + "ABCD".index(text[-1])]
        return [1] * (len(text) // 10 + 1)


class Dependencies:
    tokenizer = Tokenizer()

    def resolve(self, repo, kind):
        return "a" * 40

    def load_tokenizer(self, model, revision):
        return self.tokenizer, 4096

    def load_dataset(self, path, name, revision):
        if name == "main":
            return {"test": [dict(question=f"Compute {i}", answer="#### 2") for i in range(160)]}
        rows = [dict(subject=f"subject_{s:02}", question=f"Question {i}", choices=["a", "b", "c", "d"], answer=0)
                for s in range(57) for i in range(25)]
        dev = [dict(subject=f"subject_{s:02}", question=f"Example {i}", choices=["a", "b", "c", "d"], answer=0)
               for s in range(57) for i in range(5)]
        return {"test": rows, "dev": dev}

    def runtime(self, config):
        return {"device": "fake GPU", "versions": {"transformers": "fake"}}

    def load_backend(self, config):
        class Backend:
            def generate_batch(self, prompts, decoding):
                text = "#### 2" if config["scorer"] == "gsm8k" else "A"
                return [dict(text=text, token_count=2, stop_reason="eos") for _ in prompts]

            def choice_logits_batch(self, prompts, tokens):
                return [dict(A=1., B=0., C=0., D=0.) for _ in prompts]
        return Backend()


def test_prepare_freezes_five_cells_and_audit_execution_resumes(tmp_path):
    deps = Dependencies()
    paths = prepare_plan(tmp_path, "baseline-v1", dependencies=deps)
    assert len(paths) == 5
    configs = [json.loads(Path(path).read_text()) for path in paths]
    assert {(c["scorer"], c["shots"]) for c in configs} == {
        ("gsm8k", 0), ("mmlu_text", 0), ("mmlu_text", 5), ("mmlu_logits", 0), ("mmlu_logits", 5)}
    assert len(configs[0]["prompt_indices"]) == 150
    assert sum(map(len, configs[1]["prompt_indices"].values())) == 1140
    assert configs[0]["adapter"] is None
    assert prepare_plan(tmp_path, "baseline-v1", dependencies=deps) == paths
    for path in paths:
        result = execute_config(path, tmp_path, audit_items=2, dependencies=deps)
        assert result["total"] == 2
        assert execute_config(path, tmp_path, audit_items=2, dependencies=deps) == result
    records = list(tmp_path.glob("runs/**/responses.jsonl"))
    assert len(records) == 5
    record = json.loads(records[0].read_text().splitlines()[0])
    assert "source_index" in record and "prompt_tokens" in record


def test_prepare_rejects_context_overflow_without_truncation(tmp_path):
    class SmallContext(Dependencies):
        def load_tokenizer(self, model, revision):
            return self.tokenizer, 10
    with pytest.raises(ValueError, match="context"):
        prepare_plan(tmp_path, "small", dependencies=SmallContext())


def test_execute_rejects_changed_inputs_before_model_load(tmp_path):
    paths = prepare_plan(tmp_path, "baseline-v1", dependencies=Dependencies())
    config = json.loads(paths[0].read_text())
    inputs = tmp_path / config["items_path"]
    items = json.loads(inputs.read_text())
    items[0]["gold"] = "#### 999"
    inputs.write_text(json.dumps(items))
    with pytest.raises(ValueError, match="hash"):
        execute_config(paths[0], tmp_path, dependencies=Dependencies())


def test_local_adapter_is_content_pinned_and_changes_are_rejected(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fake weights")
    paths = prepare_plan(tmp_path / "artifacts", "adapter-v1", adapter=str(adapter), dependencies=Dependencies())
    config = json.loads(paths[0].read_text())
    assert len(config["adapter_sha256"]) == 64
    (adapter / "adapter_model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="adapter hash"):
        execute_config(paths[0], tmp_path / "artifacts", dependencies=Dependencies())


def test_cli_prepares_and_runs_an_audit_with_fake_boundaries(tmp_path, capsys):
    from pilot_eval.cli import main
    assert main(["prepare", "--plan", "cli-v1", "--output-root", str(tmp_path)], dependencies=Dependencies()) == 0
    paths = capsys.readouterr().out.strip().splitlines()
    assert len(paths) == 5
    assert main(["run", "--config", paths[0], "--audit-items", "2", "--output-root", str(tmp_path)], dependencies=Dependencies()) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2
