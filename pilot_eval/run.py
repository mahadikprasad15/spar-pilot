"""Public run and resume interface."""

import json
import os
from pathlib import Path

from pilot_eval.scoring import score_gsm8k


def run_evaluation(config: dict, items: list[dict], backend, output_root: Path) -> dict:
    """Evaluate and persist item-level results under the canonical artifact root."""
    run_dir = Path(output_root).joinpath(
        "runs", config["experiment"], config["model"], config["dataset"],
        config["cohort"], config["variant"], config["run_id"],
    )
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    summary_path = results_dir / "results.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("run config mismatch")
        if summary_path.exists():
            return json.loads(summary_path.read_text())
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    responses_path = results_dir / "responses.jsonl"
    records = [json.loads(line) for line in responses_path.read_text().splitlines()] if responses_path.exists() else []
    completed = {record["id"] for record in records}
    pending = [item for item in items if item["id"] not in completed]
    batch_size = config["batch_size"]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = backend.generate_batch([item["prompt"] for item in batch], config["decoding"])
        if len(outputs) != len(batch):
            raise ValueError("model returned the wrong number of outputs")
        batch_records = [
            {
                "id": item["id"], "prompt": item["prompt"], "gold": item["gold"],
                "generated_text": output["text"], "token_count": output["token_count"],
                "stop_reason": output["stop_reason"],
                "score": score_gsm8k(output["text"], item["gold"], output["stop_reason"] == "cap"),
            }
            for item, output in zip(batch, outputs)
        ]
        with responses_path.open("a") as stream:
            for record in batch_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records.extend(batch_records)
    summary = {
        "strict_accuracy": sum(record["score"]["strict"]["correct"] for record in records) / len(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
