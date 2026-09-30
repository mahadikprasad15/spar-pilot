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


@pytest.fixture
def selected_runs(tmp_path):
    deps = Dependencies()
    old = prepare_plan(tmp_path, "batch8", batch_size=8, dependencies=deps)
    assert main(["fork-plan", "--source", "batch8", "--plan", "batch2", "--batch-size", "2", "--output-root", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "plans/batch2/manifest.json").read_text())
    new = [tmp_path / relative for relative in manifest["configs"]]
    selected = old[:2] + new[2:]
    for path in selected:
        execute_config(path, tmp_path, dependencies=deps)
    selection = {f"{json.loads(path.read_text())['scorer']}-{json.loads(path.read_text())['shots']}shot":
                 "batch8" if index < 2 else "batch2" for index, path in enumerate(selected)}
    selection_file = tmp_path / "selection.json"
    selection_file.write_text(json.dumps(selection))
    return tmp_path, selection_file, selected


def test_combined_report_references_verified_sources_and_batch_sizes(selected_runs, capsys):
    root, selection, paths = selected_runs
    args = ["report", "--selection", str(selection), "--name", "combined", "--output-root", str(root)]
    assert main(args) == 0
    report = json.loads((root / "reports/combined/results/results.json").read_text())
    assert report["state"] == "completed"
    assert [cell["batch_size"] for cell in report["cells"]] == [8, 8, 2, 2, 2]
    assert [cell["summary"]["total"] for cell in report["cells"]] == [150, 1140, 1140, 1140, 1140]
    assert all((root / cell["run_path"] / "results/responses.jsonl").exists() for cell in report["cells"])
    assert len(list(root.glob("runs/**/responses.jsonl"))) == 5
    assert "Batch" in (root / "reports/combined/results/report.md").read_text()
    assert main(args) == 0


def test_report_rejects_partial_or_tampered_source_runs(selected_runs):
    root, selection, paths = selected_runs
    run_dir = next(root.glob("runs/**/batch2-mmlu_text-5shot"))
    status_file = run_dir / "meta/status.json"
    original_status = status_file.read_bytes()
    status = json.loads(original_status)
    status["state"] = "failed"
    status_file.write_text(json.dumps(status))
    args = ["report", "--selection", str(selection), "--name", "rejected", "--output-root", str(root)]
    assert main(args) == 1
    assert not (root / "reports/rejected/results/results.json").exists()
    status_file.write_bytes(original_status)
    responses = run_dir / "results/responses.jsonl"
    with responses.open("a") as stream:
        stream.write(responses.read_text().splitlines()[0] + "\n")
    assert main(args) == 1
    assert not (root / "reports/rejected/results/results.json").exists()
