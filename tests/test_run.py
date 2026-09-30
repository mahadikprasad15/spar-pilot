import json

import pytest

from pilot_eval.run import run_evaluation


def test_gsm8k_run_saves_response_and_summary(tmp_path):
    class FakeBackend:
        def generate_batch(self, prompts, decoding):
            assert prompts == ["prompt for item 7"]
            return [{"text": "Worked it out.\n#### 72", "token_count": 8, "stop_reason": "eos"}]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen2.5-1.5b-instruct",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 1,
    }
    items = [{"id": "gsm8k:test:7", "prompt": "prompt for item 7", "gold": "#### 72"}]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")
    run_dir = tmp_path / "artifacts/runs/pilot-1/qwen2.5-1.5b-instruct/gsm8k/test-150/baseline/run-1"
    records = [json.loads(line) for line in (run_dir / "results/responses.jsonl").read_text().splitlines()]

    assert records[0]["score"]["strict"]["correct"] is True
    assert summary["strict_accuracy"] == 1.0


def test_completed_run_is_immutable_on_repeat(tmp_path):
    class FakeBackend:
        calls = 0

        def generate_batch(self, prompts, decoding):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("completed run generated again")
            return [{"text": "#### 72", "token_count": 3, "stop_reason": "eos"}]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 1,
    }
    items = [{"id": "gsm8k:test:7", "prompt": "question", "gold": "#### 72"}]
    backend = FakeBackend()

    first = run_evaluation(config, items, backend, tmp_path / "artifacts")
    second = run_evaluation(config, items, backend, tmp_path / "artifacts")
    path = tmp_path / "artifacts/runs/pilot-1/qwen/gsm8k/test-150/baseline/run-1/results/responses.jsonl"

    assert first == second
    assert len(path.read_text().splitlines()) == 1


def test_interrupted_run_resumes_only_missing_items(tmp_path):
    class FlakyBackend:
        def generate_batch(self, prompts, decoding):
            if prompts == ["q2"]:
                raise RuntimeError("GPU interrupted")
            assert prompts == ["q1"]
            return [{"text": "#### 1", "token_count": 2, "stop_reason": "eos"}]

    class RecoveryBackend:
        def generate_batch(self, prompts, decoding):
            assert prompts == ["q2"]
            return [{"text": "#### 2", "token_count": 2, "stop_reason": "eos"}]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 1,
    }
    items = [
        {"id": "q1", "prompt": "q1", "gold": "#### 1"},
        {"id": "q2", "prompt": "q2", "gold": "#### 2"},
    ]
    root = tmp_path / "artifacts"

    with pytest.raises(RuntimeError, match="interrupted"):
        run_evaluation(config, items, FlakyBackend(), root)
    run_dir = root / "runs/pilot-1/qwen/gsm8k/test-150/baseline/run-1"
    failed_status = json.loads((run_dir / "meta/status.json").read_text())
    assert failed_status["state"] == "failed"
    assert failed_status["completed"] == 1
    summary = run_evaluation(config, items, RecoveryBackend(), root)
    path = root / "runs/pilot-1/qwen/gsm8k/test-150/baseline/run-1/results/responses.jsonl"

    assert len(path.read_text().splitlines()) == 2
    assert summary["strict_accuracy"] == 1.0
    assert json.loads((run_dir / "meta/status.json").read_text())["state"] == "completed"


def test_run_rejects_duplicate_expected_item_ids(tmp_path):
    class UnusedBackend:
        def generate_batch(self, prompts, decoding):
            raise AssertionError("duplicate IDs should fail before inference")

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 1,
    }
    items = [
        {"id": "same", "prompt": "q1", "gold": "#### 1"},
        {"id": "same", "prompt": "q2", "gold": "#### 2"},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        run_evaluation(config, items, UnusedBackend(), tmp_path / "artifacts")
