import json
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import prepare_plan, execute_config
from test_workflow import Dependencies


def test_fork_plan_preserves_frozen_inputs_without_external_access(tmp_path, capsys):
    original = prepare_plan(tmp_path, "batch8", batch_size=8, dependencies=Dependencies())
    before = {path: path.read_bytes() for path in (tmp_path / "plans/batch8").iterdir()}
    args = ["fork-plan", "--source", "batch8", "--plan", "batch2", "--batch-size", "2", "--output-root", str(tmp_path)]
    assert main(args) == 0
    forked = [Path(line) for line in capsys.readouterr().out.splitlines()]
    assert len(forked) == 5
    for old_path, new_path in zip(original, forked):
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        assert new["batch_size"] == 2
        assert new["derived_from_plan"] == "batch8"
        for key in ("model_revision", "dataset_revision", "prompt_indices", "items_sha256", "decoding", "dtype", "scorer"):
            assert new[key] == old[key]
        assert (tmp_path / new["items_path"]).read_bytes() == (tmp_path / old["items_path"]).read_bytes()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert main(args) == 0  # idempotent
