import json

import pytest

from pilot_eval.run import run_evaluation


@pytest.mark.parametrize("damage", ["duplicate", "unexpected", "changed-prompt", "truncated", "missing"])
def test_resume_rejects_damaged_completed_artifacts(tmp_path, damage):
    class Backend:
        def generate_batch(self, prompts, decoding):
            return [{"text": "#### 1", "token_count": 2, "stop_reason": "eos"}]

    config = dict(run_id="r", experiment="pilot-1", model="org/qwen", dataset="gsm8k",
                  cohort="test", variant="baseline", scorer="gsm8k", batch_size=1,
                  decoding={"do_sample": False, "max_new_tokens": 1024})
    items = [{"id": "q1", "source_index": 7, "prompt": "q", "gold": "#### 1"}]
    root = tmp_path / "artifacts"
    run_evaluation(config, items, Backend(), root)
    path = next(root.glob("runs/**/responses.jsonl"))
    record = json.loads(path.read_text())
    if damage == "duplicate":
        path.write_text(path.read_text() * 2)
    elif damage == "unexpected":
        record["id"] = "alien"
        path.write_text(json.dumps(record) + "\n")
    elif damage == "changed-prompt":
        items[0]["prompt"] = "different"
    elif damage == "missing":
        path.write_text("")
    else:
        path.write_text('{"id":')
    with pytest.raises(ValueError):
        run_evaluation(config, items, Backend(), root)


def test_tie_persists_invalid_output_and_failed_status(tmp_path):
    class Backend:
        def choice_logits_batch(self, prompts, token_ids):
            return [{"A": 2., "B": 2., "C": 0., "D": 0.}]
    config = dict(run_id="r", experiment="pilot-1", model="qwen", dataset="mmlu",
                  cohort="test", variant="baseline", scorer="mmlu_logits", batch_size=1,
                  decoding={"do_sample": False})
    items = [dict(id="q", source_index=7, subject="bio", prompt="p", gold="A",
                  choice_token_ids=dict(zip("ABCD", range(4))))]
    with pytest.raises(ValueError, match="tie"):
        run_evaluation(config, items, Backend(), tmp_path)
    errors = json.loads(next(tmp_path.glob("runs/**/errors.jsonl")).read_text())
    assert errors["item"]["source_index"] == 7
    assert errors["output"]["A"] == 2.
    assert errors["status"] == "invalid"
    assert not list(tmp_path.glob("runs/**/results.json"))


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
    manifest = json.loads((run_dir / "meta/run_manifest.json").read_text())
    assert manifest["run_id"] == "run-1"
    assert (run_dir / "logs/run.log").exists()


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


def test_mmlu_text_run_saves_choice_and_accuracy(tmp_path):
    class FakeBackend:
        def generate_batch(self, prompts, decoding):
            return [{"text": " B", "token_count": 1, "stop_reason": "eos"}]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "mmlu", "cohort": "balanced-1140", "variant": "baseline-5shot",
        "scorer": "mmlu_text", "decoding": {"do_sample": False, "max_new_tokens": 32},
        "batch_size": 1,
    }
    items = [{"id": "mmlu:biology:7", "subject": "biology", "prompt": "Answer:", "gold": "B"}]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")
    path = tmp_path / "artifacts/runs/pilot-1/qwen/mmlu/balanced-1140/baseline-5shot/run-1/results/responses.jsonl"
    record = json.loads(path.read_text())

    assert record["score"]["correct"] is True
    assert summary["accuracy"] == 1.0


def test_mmlu_logit_run_saves_four_way_decision(tmp_path):
    class FakeBackend:
        def choice_logits_batch(self, prompts, token_ids):
            assert token_ids == [{"A": 11, "B": 12, "C": 13, "D": 14}]
            return [{"A": 0.1, "B": 2.0, "C": 0.0, "D": -1.0}]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "mmlu", "cohort": "balanced-1140", "variant": "baseline-5shot",
        "scorer": "mmlu_logits", "decoding": {"do_sample": False}, "batch_size": 1,
    }
    items = [{
        "id": "mmlu:biology:7", "subject": "biology", "prompt": "Answer:",
        "gold": "B", "choice_token_ids": {"A": 11, "B": 12, "C": 13, "D": 14},
    }]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")
    path = tmp_path / "artifacts/runs/pilot-1/qwen/mmlu/balanced-1140/baseline-5shot/run-1/results/responses.jsonl"
    record = json.loads(path.read_text())

    assert record["generated_text"] is None
    assert record["candidate_scores"]["B"] == 2.0
    assert record["score"]["correct"] is True
    assert summary["accuracy"] == 1.0


def test_gsm8k_summary_reports_both_scorers_and_response_lengths(tmp_path):
    class FakeBackend:
        def generate_batch(self, prompts, decoding):
            return [
                {"text": "#### 1", "token_count": 8, "stop_reason": "eos"},
                {"text": "Final answer: 2", "token_count": 4, "stop_reason": "eos"},
            ]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 2,
    }
    items = [
        {"id": "q1", "prompt": "q1", "gold": "#### 1"},
        {"id": "q2", "prompt": "q2", "gold": "#### 2"},
    ]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")

    assert summary["strict_correct"] == 1
    assert summary["flexible_correct"] == 2
    assert summary["total"] == 2
    assert summary["mean_response_tokens"] == 6
    assert summary["median_response_tokens"] == 6


def test_gsm8k_summary_counts_caps_and_accuracy_interval(tmp_path):
    class FakeBackend:
        def generate_batch(self, prompts, decoding):
            return [
                {"text": "#### 1", "token_count": 2, "stop_reason": "eos"},
                {"text": "#### 2", "token_count": 1024, "stop_reason": "cap"},
            ]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "gsm8k", "cohort": "test-150", "variant": "baseline",
        "scorer": "gsm8k", "decoding": {"do_sample": False, "max_new_tokens": 1024},
        "batch_size": 2,
    }
    items = [
        {"id": "q1", "prompt": "q1", "gold": "#### 1"},
        {"id": "q2", "prompt": "q2", "gold": "#### 2"},
    ]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")

    assert summary["cap_count"] == 1
    assert summary["strict_invalid_count"] == 1
    assert summary["strict_wilson_95"][0] < 0.5 < summary["strict_wilson_95"][1]


def test_mmlu_summary_reports_subject_counts_and_sampling_range(tmp_path):
    class FakeBackend:
        def generate_batch(self, prompts, decoding):
            return [
                {"text": letter, "token_count": 1, "stop_reason": "eos"}
                for letter in ["A", "B", "A", "A"]
            ]

    config = {
        "run_id": "run-1", "experiment": "pilot-1", "model": "qwen",
        "dataset": "mmlu", "cohort": "balanced-1140", "variant": "baseline-0shot",
        "scorer": "mmlu_text", "decoding": {"do_sample": False, "max_new_tokens": 32},
        "batch_size": 4, "seed": 42,
    }
    items = [
        {"id": "bio-1", "subject": "biology", "prompt": "q1", "gold": "A"},
        {"id": "bio-2", "subject": "biology", "prompt": "q2", "gold": "A"},
        {"id": "hist-1", "subject": "history", "prompt": "q3", "gold": "A"},
        {"id": "hist-2", "subject": "history", "prompt": "q4", "gold": "A"},
    ]

    summary = run_evaluation(config, items, FakeBackend(), tmp_path / "artifacts")

    assert summary["accuracy"] == 0.75
    assert summary["subjects"]["biology"] == {"correct": 1, "total": 2}
    assert summary["subjects"]["history"] == {"correct": 2, "total": 2}
    assert summary["sampling_interval_95"] == [0.5, 1.0]
