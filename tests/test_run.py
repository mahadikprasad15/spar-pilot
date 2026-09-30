import json

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
