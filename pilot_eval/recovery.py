"""Fork execution settings while preserving frozen scientific inputs."""

import json
import re
from pathlib import Path

from pilot_eval.config import validate_config
from pilot_eval.workflow import _hash, _save_frozen


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
