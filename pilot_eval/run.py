"""Public run and resume interface."""

import json
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
    outputs = backend.generate_batch([item["prompt"] for item in items], config["decoding"])
    records = []
    for item, output in zip(items, outputs):
        records.append({
            "id": item["id"], "prompt": item["prompt"], "gold": item["gold"],
            "generated_text": output["text"], "token_count": output["token_count"],
            "stop_reason": output["stop_reason"],
            "score": score_gsm8k(output["text"], item["gold"], output["stop_reason"] == "cap"),
        })
    (results_dir / "responses.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    summary = {
        "strict_accuracy": sum(record["score"]["strict"]["correct"] for record in records) / len(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
