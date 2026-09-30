"""Public run and resume interface."""

import json
import math
import os
import random
import statistics
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval.scoring import score_gsm8k, score_mmlu_logits, score_mmlu_text


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_state(run_dir: Path, state: str, completed: int, total: int, error: str | None = None) -> None:
    status = {
        "state": state,
        "completed": completed,
        "total": total,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        status["error"] = error
    _write_json(run_dir / "meta/status.json", status)
    _write_json(run_dir / "checkpoints/progress.json", {"completed": completed, "total": total})
    log_path = run_dir / "logs/run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(json.dumps(status, sort_keys=True) + "\n")


def _wilson_95(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half_width, center + half_width]


def _subject_sampling_interval_95(records: list[dict], seed: int) -> list[float]:
    grouped = {}
    for record in records:
        grouped.setdefault(record["subject"], []).append(int(record["score"]["correct"]))
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        sampled = [
            rng.choice(values)
            for subject in sorted(grouped)
            for values in [grouped[subject]]
            for _ in values
        ]
        draws.append(sum(sampled) / len(sampled))
    draws.sort()
    return [draws[49], draws[1949]]


def run_evaluation(config: dict, items: list[dict], backend, output_root: Path) -> dict:
    """Evaluate and persist item-level results under the canonical artifact root."""
    item_ids = [item["id"] for item in items]
    if not items or config["batch_size"] < 1:
        raise ValueError("nonempty cohort and positive batch size required")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("duplicate expected item IDs")
    components = [config[key] for key in ("experiment", "dataset", "cohort", "variant", "run_id")]
    if config["model"] in (".", "..") or any(not value or value in (".", "..") or "/" in value or "\\" in value for value in components):
        raise ValueError("unsafe artifact path component")
    run_dir = Path(output_root).joinpath(
        "runs", config["experiment"], config["model"].replace("/", "--"), config["dataset"],
        config["cohort"], config["variant"], config["run_id"],
    )
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    summary_path = results_dir / "results.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("run config mismatch")
    else:
        _write_json(config_path, config)
    inputs_path = run_dir / "inputs/items.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != items:
        raise ValueError("run input mismatch")
    if not inputs_path.exists():
        _write_json(inputs_path, items)
    if "cohort_manifest" in config and not (run_dir / "inputs/cohort.json").exists():
        _write_json(run_dir / "inputs/cohort.json", config["cohort_manifest"])
    manifest_path = run_dir / "meta/run_manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, {
            "run_id": config["run_id"],
            "config": config,
            "responses_path": "results/responses.jsonl",
            "summary_path": "results/results.json",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
    responses_path = results_dir / "responses.jsonl"
    records = [json.loads(line) for line in responses_path.read_text().splitlines()] if responses_path.exists() else []
    expected = {item["id"]: item for item in items}
    completed = set()
    for record in records:
        item = expected.get(record.get("id"))
        if item is None or record["id"] in completed:
            raise ValueError("duplicate or unexpected saved item ID")
        if any(record.get(key) != value for key, value in item.items()):
            raise ValueError("saved response provenance mismatch")
        if config["scorer"] == "mmlu_logits":
            score = score_mmlu_logits(record["candidate_scores"], item["gold"], config.get("logit_tie_policy", "stop"))
        else:
            scorer = score_gsm8k if config["scorer"] == "gsm8k" else score_mmlu_text
            score = scorer(record["generated_text"], item["gold"], record["stop_reason"] == "cap")
        if score != record["score"]:
            raise ValueError("saved score mismatch")
        completed.add(record["id"])
    errors_path = run_dir / "logs/errors.jsonl"
    if errors_path.exists():
        errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
        if any(error.get("status") == "invalid" for error in errors):
            raise ValueError("unresolved invalid item error; inspect errors and use a new run ID")
    saved_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if saved_summary is not None:
        if completed != set(item_ids) or saved_summary.get("responses_sha256") != hashlib.sha256(responses_path.read_bytes()).hexdigest():
            raise ValueError("completed run responses are incomplete or changed")
    pending = [item for item in items if item["id"] not in completed]
    batch_size = config["batch_size"]
    if saved_summary is None:
        _write_state(run_dir, "running", len(records), len(items))
    try:
        for start in range(0, len(pending), batch_size):
            item = output = None
            outputs = []
            batch = pending[start : start + batch_size]
            prompts = [item["prompt"] for item in batch]
            if config["scorer"] == "mmlu_logits":
                outputs = backend.choice_logits_batch(
                    prompts, [item["choice_token_ids"] for item in batch]
                )
            else:
                outputs = backend.generate_batch(prompts, config["decoding"])
            if len(outputs) != len(batch):
                raise ValueError("model returned the wrong number of outputs")
            batch_records = []
            for item, output in zip(batch, outputs):
                record = {**item, "subject": item.get("subject")}
                if config["scorer"] == "mmlu_logits":
                    record.update({
                        "generated_text": None, "candidate_scores": output,
                        "choice_token_ids": item["choice_token_ids"],
                        "score": score_mmlu_logits(output, item["gold"], config.get("logit_tie_policy", "stop")),
                    })
                else:
                    record.update({
                        "generated_text": output["text"], "token_count": output["token_count"],
                        "stop_reason": output["stop_reason"],
                        "score": (
                            score_gsm8k(output["text"], item["gold"], output["stop_reason"] == "cap")
                            if config["scorer"] == "gsm8k"
                            else score_mmlu_text(output["text"], item["gold"], output["stop_reason"] == "cap")
                        ),
                    })
                batch_records.append(record)
            with responses_path.open("a") as stream:
                for record in batch_records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.extend(batch_records)
            if (start // batch_size + 1) % config.get("checkpoint_interval_batches", 1) == 0:
                _write_state(run_dir, "running", len(records), len(items))
    except (Exception, KeyboardInterrupt) as exc:
        _write_state(run_dir, "failed", len(records), len(items), str(exc))
        with errors_path.open("a") as stream:
            stream.write(json.dumps({"error": str(exc), "item": locals().get("item"),
                                    "output": locals().get("output"),
                                    "batch_items": locals().get("batch"), "batch_outputs": locals().get("outputs"),
                                    "status": "invalid" if isinstance(exc, ValueError) else "recoverable"},
                                   sort_keys=True) + "\n")
        raise
    if {record["id"] for record in records} != set(item_ids) or len(records) != len(items):
        raise ValueError("incomplete run")
    if config["scorer"] == "gsm8k":
        total = len(records)
        strict_correct = sum(record["score"]["strict"]["correct"] for record in records)
        flexible_correct = sum(record["score"]["flexible"]["correct"] for record in records)
        lengths = [record["token_count"] for record in records]
        summary = {
            "total": total,
            "strict_correct": strict_correct,
            "flexible_correct": flexible_correct,
            "strict_accuracy": strict_correct / total,
            "flexible_accuracy": flexible_correct / total,
            "strict_wilson_95": _wilson_95(strict_correct, total),
            "flexible_wilson_95": _wilson_95(flexible_correct, total),
            "strict_invalid_count": sum(record["score"]["strict"]["status"] == "invalid" for record in records),
            "flexible_invalid_count": sum(record["score"]["flexible"]["status"] == "invalid" for record in records),
            "cap_count": sum(record["stop_reason"] == "cap" for record in records),
            "mean_response_tokens": statistics.mean(lengths),
            "median_response_tokens": statistics.median(lengths),
        }
    else:
        subjects = {}
        for record in records:
            counts = subjects.setdefault(record["subject"], {"correct": 0, "total": 0})
            counts["total"] += 1
            counts["correct"] += int(record["score"]["correct"])
        correct = sum(record["score"]["correct"] for record in records)
        summary = {
            "accuracy": correct / len(records),
            "correct": correct,
            "total": len(records),
            "subjects": subjects,
            "sampling_interval_95": _subject_sampling_interval_95(records, config.get("seed", 42)),
            "invalid_count": sum(record["score"]["status"] == "invalid" for record in records),
            "cap_count": sum(record.get("stop_reason") == "cap" for record in records),
        }
    if config["scorer"] == "mmlu_logits" and config.get("logit_tie_policy") == "invalid":
        summary["scorer_version"] = "mmlu-logits-v2"
        summary["logit_tie_policy"] = "invalid"
        summary["tie_count"] = sum(bool(record["score"].get("tied_choices")) for record in records)
        summary["tie_rate"] = summary["tie_count"] / summary["total"]
    summary["cap_rate"] = summary["cap_count"] / summary["total"]
    if config["scorer"] == "gsm8k":
        summary["strict_invalid_rate"] = summary["strict_invalid_count"] / summary["total"]
        summary["flexible_invalid_rate"] = summary["flexible_invalid_count"] / summary["total"]
        summary["historical_reference"] = {"accuracy": .640, "mean_response_tokens": 288,
            "strict_accuracy_gap": summary["strict_accuracy"] - .640,
            "flexible_accuracy_gap": summary["flexible_accuracy"] - .640,
            "mean_response_tokens_gap": summary["mean_response_tokens"] - 288}
    else:
        summary["invalid_rate"] = summary["invalid_count"] / summary["total"]
        summary["historical_reference"] = {"accuracy": .570, "accuracy_gap": summary["accuracy"] - .570}
    summary["comparison_note"] = "New sampled protocol; historical gaps are descriptive, not reproduction gates."
    summary["responses_sha256"] = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    if saved_summary is not None:
        if summary != saved_summary:
            raise ValueError("saved summary differs from response-derived aggregate")
        return summary
    _write_json(summary_path, summary)
    _write_state(run_dir, "completed", len(records), len(items))
    return summary
