"""Read-only verification of source runs and a combined, provenance-preserving report."""

import hashlib
import json
from pathlib import Path

from pilot_eval.recovery import CELLS, _inside, _name, plan_cells
from pilot_eval.run import run_evaluation
from pilot_eval.workflow import _hash, _save_frozen


_GLOBAL = ("model", "model_revision", "tokenizer_revision", "adapter", "adapter_revision",
           "adapter_sha256", "dtype", "attention_implementation", "quantization", "deterministic",
           "seed", "protocol_version", "context_limit", "chat_template_sha256")
_DATASET = ("dataset_path", "dataset_config", "dataset_revision", "evaluation_split", "cohort",
            "cohort_manifest", "prompt_indices", "prompt_template")


def _same(left, right, fields):
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"selected sources have mismatched {field}")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _NoInference:
    def generate_batch(self, *args):
        raise ValueError("report source is incomplete; reporting never generates missing responses")

    choice_logits_batch = generate_batch


def build_report(output_root, name, selection):
    """Select exactly one completed source per cell; retain its config and response identity."""
    root = Path(output_root).resolve()
    destination = root / "reports" / _name(name)
    if set(selection) != set(CELLS):
        raise ValueError("selection must name all five pilot cells exactly once")
    plans = {plan: plan_cells(root, plan) for plan in set(selection.values())}
    selected = [(cell, selection[cell], *plans[selection[cell]][cell]) for cell in CELLS]
    dataset_configs, prompts, runtimes = {}, {}, []
    entries, discrepancies = [], []
    first = selected[0][3]
    logits = [config for _, _, _, config in selected if config["scorer"] == "mmlu_logits"]
    _same(logits[0], logits[1], ("logit_tie_policy", "scorer_version"))
    for cell, plan, config_path, config in selected:
        _same(first, config, _GLOBAL)
        for key in ("do_sample", "num_beams", "num_return_sequences", "eos_token_id", "pad_token_id"):
            _same(first["decoding"], config["decoding"], (key,))
        dataset = config["dataset"]
        if dataset in dataset_configs:
            _same(dataset_configs[dataset], config, _DATASET)
        dataset_configs[dataset] = config
        items = json.loads(_inside(root, config["items_path"]).read_text())
        if _hash(items) != config["items_sha256"]:
            raise ValueError(f"planned input hash mismatch: {cell}")
        comparison = [{key: value for key, value in item.items() if key != "choice_token_ids"} for item in items]
        prompt_key = (dataset, config["shots"])
        if prompt_key in prompts and prompts[prompt_key] != comparison:
            raise ValueError("text and logit sources have mismatched rendered prompts or items")
        prompts[prompt_key] = comparison
        if any("<|im_start|>system" in item["prompt"] for item in items):
            discrepancies.append({"cell": cell, "issue": "Rendered prompt contains a system message; protocol v1 specifies none."})
        run_dir = root.joinpath("runs", config["experiment"], config["model"].replace("/", "--"),
                                dataset, config["cohort"], config["variant"], config["run_id"])
        status = json.loads((run_dir / "meta/status.json").read_text())
        if status.get("state") != "completed":
            raise ValueError(f"source run is not completed: {cell}")
        saved = json.loads((run_dir / "config.json").read_text())
        if any(saved.get(key) != value for key, value in config.items()) or saved.get("audit_items"):
            raise ValueError(f"saved run config mismatch or audit source: {cell}")
        if "runtime" not in saved:
            raise ValueError("source run lacks runtime provenance")
        runtime = {key: value for key, value in saved["runtime"].items() if key != "resolved_generation"}
        if runtimes and runtime != runtimes[0]:
            raise ValueError("selected sources have mismatched runtime provenance")
        runtimes.append(runtime)
        if json.loads((run_dir / "inputs/items.json").read_text()) != items:
            raise ValueError("saved run input mismatch")
        if json.loads((run_dir / "inputs/cohort.json").read_text()) != config["cohort_manifest"]:
            raise ValueError("saved run cohort mismatch")
        if not (run_dir / "meta/run_manifest.json").is_file() or not (run_dir / "results/results.json").is_file():
            raise ValueError("source run lacks completion artifacts")
        # Reuse the run's integrity checks and response-derived aggregation, with no model boundary.
        summary = run_evaluation(saved, items, _NoInference(), root)
        entries.append({"cell": cell, "plan": plan, "run_path": str(run_dir.relative_to(root)),
                        "config_path": str(config_path.relative_to(root)), "batch_size": config["batch_size"],
                        "config_sha256": _sha(run_dir / "config.json"),
                        "responses_sha256": summary["responses_sha256"],
                        "summary_sha256": _sha(run_dir / "results/results.json"), "summary": summary})
        if config.get("scorer_version"):
            entries[-1].update(scorer_version=config["scorer_version"], logit_tie_policy=config["logit_tie_policy"])
    report = {"state": "completed", "name": name, "selection": selection, "cells": entries,
              "protocol_discrepancies": discrepancies, "protocol_compliant": not discrepancies,
              "note": "Combined source runs; batch size is reported per cell. Historical gaps are descriptive. "
                      "Completion does not imply protocol compliance or identical reproduction."}
    _save_frozen(destination / "inputs/selection.json", selection)
    _save_frozen(destination / "results/results.json", report)
    lines = [f"# {name}", "", report["note"], "",
             "| Measurement | Accuracy | 95% range | Batch | Source plan |",
             "|---|---:|---|---:|---|"]
    for entry in entries:
        summary = entry["summary"]
        measurements = (("strict", "flexible") if entry["cell"] == "gsm8k-0shot" else (None,))
        for scorer in measurements:
            label = f"GSM8K {scorer}" if scorer else entry["cell"]
            accuracy = summary[f"{scorer}_accuracy"] if scorer else summary["accuracy"]
            interval = summary[f"{scorer}_wilson_95"] if scorer else summary["sampling_interval_95"]
            lines.append(f"| {label} | {accuracy:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] | {entry['batch_size']} | {entry['plan']} |")
    gsm = entries[0]["summary"]
    revised = [entry for entry in entries if entry.get("scorer_version") == "mmlu-logits-v2"]
    if revised:
        lines.extend(["", "## Logit scoring v2", "",
                      "Ties count as invalid and incorrect and remain in the accuracy denominator."])
        for entry in revised:
            lines.append(f"- {entry['cell']}: {entry['summary']['tie_count']} ties ({entry['summary']['tie_rate']:.3%}); scorer mmlu-logits-v2.")
    lines.extend(["", f"GSM8K mean/median response tokens: {gsm['mean_response_tokens']} / {gsm['median_response_tokens']}.",
                  "", "## Source runs", ""])
    for entry in entries:
        lines.append(f"- {entry['cell']}: `{entry['run_path']}`")
    lines.extend(["", "## Protocol audit", ""])
    lines.extend(f"- {item['cell']}: {item['issue']}" for item in discrepancies)
    if not discrepancies:
        lines.append("No system-message discrepancy detected; this is not a substitute for manual audit.")
    markdown = "\n".join(lines) + "\n"
    markdown_path = destination / "results/report.md"
    if markdown_path.exists() and markdown_path.read_text() != markdown:
        raise ValueError("frozen report mismatch")
    if not markdown_path.exists():
        markdown_path.write_text(markdown)
    _save_frozen(destination / "meta/status.json", {"state": "completed", "cells": len(entries),
                                                   "protocol_compliant": not discrepancies})
    return report
