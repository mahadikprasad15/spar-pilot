"""CPU-only, immutable rescoring of saved GSM8K responses."""

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from pilot_eval.config import validate_config
from pilot_eval.recovery import _name
from pilot_eval.run import _wilson_95
from pilot_eval.scoring import score_gsm8k, score_gsm8k_flexible_v2
from pilot_eval.workflow import _save_frozen


def _save_bytes(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError("frozen rescore artifact mismatch; use a new report name")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def rescore_gsm8k(responses_path, config_path, summary_path, output_root, name):
    paths = dict(responses=Path(responses_path), config=Path(config_path), summary=Path(summary_path))
    contents = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in contents.items()}
    config = validate_config(json.loads(contents["config"]))
    old = json.loads(contents["summary"])
    records = [json.loads(line) for line in contents["responses"].splitlines()]
    if config["scorer"] != "gsm8k" or config.get("audit_items"):
        raise ValueError("rescoring requires a full GSM8K run")
    if hashes["responses"] != old["responses_sha256"]:
        raise ValueError("source response hash mismatch")
    if not records or len(records) != old["total"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("source responses incomplete or duplicate")
    if [r["source_index"] for r in records] != config["prompt_indices"]:
        raise ValueError("source indices mismatch")
    if [r["id"] for r in records] != config["cohort_manifest"]["item_ids"]:
        raise ValueError("source item IDs mismatch")
    for record in records:
        if not isinstance(record["generated_text"], str) or record["stop_reason"] not in ("eos", "cap", "other"):
            raise ValueError("invalid source response fields")
        if score_gsm8k(record["generated_text"], record["gold"], record["stop_reason"] == "cap") != record["score"]:
            raise ValueError("source score mismatch")
    for scorer in ("strict", "flexible"):
        correct = sum(r["score"][scorer]["correct"] for r in records)
        invalid = sum(r["score"][scorer]["status"] == "invalid" for r in records)
        if (old[f"{scorer}_correct"] != correct or old[f"{scorer}_accuracy"] != correct / len(records)
                or old[f"{scorer}_invalid_count"] != invalid):
            raise ValueError("source summary mismatch")
    if (old["cap_count"] != sum(r["stop_reason"] == "cap" for r in records)
            or old["mean_response_tokens"] != statistics.mean(r["token_count"] for r in records)):
        raise ValueError("source length/cap summary mismatch")
    rescored, changes = [], []
    for record in records:
        new = score_gsm8k_flexible_v2(record["generated_text"], record["gold"], record["stop_reason"] == "cap")
        rescored.append({**record, "score": {**record["score"], "flexible_v2": new}})
        previous = record["score"]["flexible"]
        if any(previous.get(key) != new.get(key) for key in ("extracted", "status", "correct")):
            changes.append(dict(id=record["id"], generated_text=record["generated_text"], gold=record["gold"],
                                old_flexible=previous, flexible_v2=new))
    scores = [r["score"]["flexible_v2"] for r in rescored]
    correct = sum(s["correct"] for s in scores)
    invalid = sum(s["status"] == "invalid" for s in scores)
    summary = dict(total=len(records), scorer_version="gsm8k-flexible-v2", post_hoc=True,
                   old_strict_correct=old["strict_correct"], old_flexible_correct=old["flexible_correct"],
                   old_strict_accuracy=old["strict_accuracy"], old_flexible_accuracy=old["flexible_accuracy"],
                   flexible_v2_correct=correct, flexible_v2_accuracy=correct / len(records),
                   flexible_v2_wilson_95=_wilson_95(correct, len(records)),
                   flexible_v2_invalid_count=invalid, flexible_v2_invalid_rate=invalid / len(records),
                   invalid_reasons=dict(Counter(s["invalid_reason"] for s in scores if s["status"] == "invalid")),
                   extraction_rules=dict(Counter(s["extraction_rule"] for s in scores if s["status"] == "valid")),
                   changed_extraction_count=len(changes),
                   recovered_correct_count=sum(r["score"]["flexible_v2"]["correct"] and not r["score"]["flexible"]["correct"] for r in rescored),
                   lost_correct_count=sum(r["score"]["flexible"]["correct"] and not r["score"]["flexible_v2"]["correct"] for r in rescored),
                   source_hashes=hashes,
                   note="Post hoc offline scorer revision. Strict and flexible v1 remain unchanged; historical gaps are descriptive.")
    destination = Path(output_root) / "reports" / _name(name)
    for key, filename in (("responses", "responses.jsonl"), ("config", "config.json"), ("summary", "results.json")):
        _save_bytes(destination / "inputs" / filename, contents[key])
    _save_frozen(destination / "config.json", dict(source_config=config, scorer_version="gsm8k-flexible-v2",
                  source_paths={key: str(path.resolve()) for key, path in paths.items()}, source_hashes=hashes))
    for filename, values in (("responses.jsonl", rescored), ("changes.jsonl", changes)):
        _save_bytes(destination / "results" / filename, ''.join(json.dumps(r, sort_keys=True) + '\n' for r in values).encode())
    summary["rescored_responses_sha256"] = hashlib.sha256((destination / "results/responses.jsonl").read_bytes()).hexdigest()
    _save_frozen(destination / "results/results.json", summary)
    low, high = summary["flexible_v2_wilson_95"]
    markdown = (f"# {name}\n\n{summary['note']}\n\n"
                "| Scorer | Correct | Accuracy |\n|---|---:|---:|\n"
                f"| Strict v1 | {old['strict_correct']}/{len(records)} | {old['strict_accuracy']:.3%} |\n"
                f"| Flexible v1 | {old['flexible_correct']}/{len(records)} | {old['flexible_accuracy']:.3%} |\n"
                f"| Flexible v2 | {correct}/{len(records)} | {correct / len(records):.3%} |\n\n"
                f"V2 95% Wilson range: [{low:.3%}, {high:.3%}]. Invalid: {invalid}/{len(records)}.\n\n"
                f"Recovered correct: {summary['recovered_correct_count']}; lost correct: {summary['lost_correct_count']}. "
                f"Changed extractions: {len(changes)}.\n\n"
                "All items remain in the denominator. No model inference was performed. "
                "The source run's prompts, precision and protocol discrepancies remain applicable. "
                "See responses.jsonl and changes.jsonl for every decision.\n")
    _save_bytes(destination / "results/report.md", markdown.encode())
    _save_frozen(destination / "meta/status.json", dict(state="completed", completed=len(records), total=len(records)))
    return summary
