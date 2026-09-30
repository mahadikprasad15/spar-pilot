"""Fork execution settings while preserving frozen scientific inputs."""

import json
import re
from pathlib import Path

from pilot_eval.config import validate_config
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.scoring import score_mmlu_logits


CELLS = ("gsm8k-0shot", "mmlu_text-0shot", "mmlu_text-5shot", "mmlu_logits-0shot", "mmlu_logits-5shot")


def _name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("artifact name must contain only letters, digits, underscores and hyphens")
    return name


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes root")
    return path


def plan_cells(root, name):
    manifest = json.loads((root / "plans" / _name(name) / "manifest.json").read_text())
    result = {}
    for relative in manifest["configs"]:
        path = _inside(root, relative)
        config = validate_config(json.loads(path.read_text()))
        cell = f"{config['scorer']}-{config['shots']}shot"
        if cell in result:
            raise ValueError("duplicate plan cell")
        result[cell] = (path, config)
    if set(result) != set(CELLS):
        raise ValueError("plan must contain the five pilot cells")
    return result


def fork_plan(output_root, source, plan_name, batch_size):
    """Copy validated frozen inputs; never download or transfer response records."""
    root = Path(output_root).resolve()
    _name(plan_name)
    if source == plan_name or batch_size < 1:
        raise ValueError("fork requires a new plan name and positive batch size")
    cells = plan_cells(root, source)
    source_dir = root / "plans" / source
    destination = root / "plans" / plan_name
    manifest = json.loads((source_dir / "manifest.json").read_text())
    options = {**manifest["options"], "batch_size": batch_size}
    validated = []
    for cell in CELLS:
        _, config = cells[cell]
        items = json.loads(_inside(root, config["items_path"]).read_text())
        if _hash(items) != config["items_sha256"]:
            raise ValueError(f"input hash mismatch: {cell}")
        validated.append((cell, config, items))
    pins = json.loads((source_dir / "pins.json").read_text())
    _save_frozen(destination / "pins.json", {**pins, "options": options})
    paths = []
    for cell, config, items in validated:
        items_path = destination / f"{cell}.items.json"
        _save_frozen(items_path, items)
        forked = {**config, "batch_size": batch_size, "run_id": f"{plan_name}-{cell}",
                  "items_path": str(items_path.relative_to(root)), "derived_from_plan": source}
        validate_config(forked)
        path = destination / f"{cell}.config.json"
        _save_frozen(path, forked)
        paths.append(path)
    _save_frozen(destination / "manifest.json", {"options": options, "source_plan": source,
                                                "configs": [str(path.relative_to(root)) for path in paths]})
    return paths


def revise_logits(output_root, source, plan_name):
    """Freeze scorer v2 with a verified cache of source logits; leave sources intact."""
    root = Path(output_root).resolve()
    _name(plan_name)
    if source == plan_name:
        raise ValueError("scoring revision requires a new plan name")
    cells = plan_cells(root, source)
    destination = root / "plans" / plan_name
    prepared = []
    for cell in CELLS:
        _, config = cells[cell]
        items = json.loads(_inside(root, config["items_path"]).read_text())
        if _hash(items) != config["items_sha256"]:
            raise ValueError("source input hash mismatch")
        cache, runtime, sources = {}, None, {}
        run_dir = root.joinpath("runs", config["experiment"], config["model"].replace("/", "--"),
                                config["dataset"], config["cohort"], config["variant"], config["run_id"])
        if config["scorer"] == "mmlu_logits" and run_dir.exists():
            expected = {item["id"]: item for item in items}
            def read(path):
                content = path.read_text()
                sources[str(path.relative_to(root))] = _hash(content)
                return content
            saved = json.loads(read(run_dir / "config.json"))
            if any(saved.get(k) != v for k, v in config.items()) or "runtime" not in saved:
                raise ValueError("source run config mismatch")
            runtime = saved["runtime"]
            if json.loads(read(run_dir / "inputs/items.json")) != items:
                raise ValueError("source run input mismatch")
            def add(item, scores):
                if expected.get(item.get("id")) != item:
                    raise ValueError("source logit provenance mismatch")
                score_mmlu_logits(scores, item["gold"], "invalid")
                value = {"item": item, "candidate_scores": scores}
                if item["id"] in cache and cache[item["id"]] != value:
                    raise ValueError("conflicting source logits")
                cache[item["id"]] = value
            responses = run_dir / "results/responses.jsonl"
            if responses.exists():
                for line in read(responses).splitlines():
                    record = json.loads(line)
                    item = expected.get(record.get("id"))
                    if item is None or any(record.get(k) != v for k, v in item.items()):
                        raise ValueError("source response provenance mismatch")
                    if record["id"] in cache:
                        raise ValueError("duplicate source response")
                    if record["score"] != score_mmlu_logits(record["candidate_scores"], item["gold"], config.get("logit_tie_policy", "stop")):
                        raise ValueError("source score mismatch")
                    add(item, record["candidate_scores"])
            errors = run_dir / "logs/errors.jsonl"
            if errors.exists():
                for line in read(errors).splitlines():
                    error = json.loads(line)
                    if error.get("status") != "invalid":
                        continue
                    if error.get("error") != "top-logit tie makes the item invalid":
                        raise ValueError("source has an invalid error other than a logit tie")
                    if not score_mmlu_logits(error["output"], error["item"]["gold"], "invalid").get("tied_choices"):
                        raise ValueError("source error is not a top-logit tie")
                    batch, outputs = error["batch_items"], error["batch_outputs"]
                    if not batch or len(batch) != len(outputs):
                        raise ValueError("incomplete source error batch")
                    if not any(item == error["item"] and scores == error["output"] for item, scores in zip(batch, outputs)):
                        raise ValueError("source tie is absent from saved error batch")
                    for item, scores in zip(batch, outputs):
                        add(item, scores)
        replay = {"records": list(cache.values()), "runtime": runtime, "source_hashes": sources}
        prepared.append((cell, config, items, replay))
    source_dir = root / "plans" / source
    manifest = json.loads((source_dir / "manifest.json").read_text())
    _save_frozen(destination / "pins.json", json.loads((source_dir / "pins.json").read_text()))
    paths = []
    for cell, config, items, replay in prepared:
        items_path = destination / f"{cell}.items.json"
        _save_frozen(items_path, items)
        revised = {**config, "run_id": f"{plan_name}-{cell}", "derived_from_plan": source,
                   "items_path": str(items_path.relative_to(root))}
        if config["scorer"] == "mmlu_logits":
            replay_path = destination / f"{cell}.replay.json"
            _save_frozen(replay_path, replay)
            revised.update(scorer_version="mmlu-logits-v2", logit_tie_policy="invalid",
                           replay_path=str(replay_path.relative_to(root)), replay_sha256=_hash(replay),
                           replayed_item_ids=[r["item"]["id"] for r in replay["records"]])
        validate_config(revised)
        path = destination / f"{cell}.config.json"
        _save_frozen(path, revised)
        paths.append(path)
    _save_frozen(destination / "manifest.json", {"options": manifest["options"], "source_plan": source,
                                                "configs": [str(p.relative_to(root)) for p in paths]})
    return paths
