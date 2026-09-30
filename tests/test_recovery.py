import json
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import prepare_plan, execute_config
from test_workflow import Dependencies, Tokenizer


def test_revise_logits_reuses_saved_and_failed_batch_without_changing_source(tmp_path, capsys):
    deps = Dependencies()
    paths = prepare_plan(tmp_path, "old", batch_size=2, dependencies=deps)
    class Tied(Dependencies):
        def load_backend(self, config):
            class Backend:
                calls = 0
                def choice_logits_batch(self, prompts, tokens):
                    self.calls += 1
                    return ([dict(A=3., B=2., C=1., D=0.) for _ in prompts] if self.calls == 1
                            else [dict(A=1., B=2., C=2., D=0.) for _ in prompts])
            return Backend()
    with pytest.raises(ValueError, match="tie"):
        execute_config(paths[3], tmp_path, dependencies=Tied())
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    args = ["revise-logits", "--source", "old", "--plan", "v2", "--output-root", str(tmp_path)]
    assert main(args) == 0
    manifest = json.loads((tmp_path / "plans/v2/manifest.json").read_text())
    new = [tmp_path / p for p in manifest["configs"]]
    config = json.loads(new[3].read_text())
    assert config["replayed_item_ids"] and len(config["replayed_item_ids"]) == 4
    class Remaining(Dependencies):
        inferred = 0
        def load_backend(self, config):
            class Backend:
                def choice_logits_batch(self, prompts, tokens):
                    Remaining.inferred += len(prompts)
                    return [dict(A=3., B=2., C=1., D=0.) for _ in prompts]
            return Backend()
    summary = execute_config(new[3], tmp_path, dependencies=Remaining())
    assert summary["total"] == 1140
    assert summary["tie_count"] == 2
    assert summary["correct"] == 1138
    assert Remaining.inferred == 1136
    assert main(args) == 0
    assert execute_config(new[3], tmp_path, dependencies=Remaining()) == summary
    assert all(p.read_bytes() == content for p, content in before.items())


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
    class SystemTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>system\nDefault system message\n<|im_end|>\n" + super().apply_chat_template(messages, **kwargs)
    deps.tokenizer = SystemTokenizer()
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
    before = {path: path.read_bytes() for path in (root / "runs").rglob("*") if path.is_file()}
    assert main(args) == 0
    report = json.loads((root / "reports/combined/results/results.json").read_text())
    assert report["state"] == "completed"
    assert [cell["batch_size"] for cell in report["cells"]] == [8, 8, 2, 2, 2]
    assert [cell["summary"]["total"] for cell in report["cells"]] == [150, 1140, 1140, 1140, 1140]
    assert all((root / cell["run_path"] / "results/responses.jsonl").exists() for cell in report["cells"])
    assert len(list(root.glob("runs/**/responses.jsonl"))) == 5
    assert report["protocol_compliant"] is False
    assert len(report["protocol_discrepancies"]) == 5
    assert "Batch" in (root / "reports/combined/results/report.md").read_text()
    assert main(args) == 0
    assert all(path.read_bytes() == value for path, value in before.items())


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


def test_report_rejects_incompatible_selected_cohorts(selected_runs, capsys):
    root, selection, paths = selected_runs
    path = paths[2]
    config = json.loads(path.read_text())
    config["dataset_revision"] = "b" * 40
    path.write_text(json.dumps(config))
    args = ["report", "--selection", str(selection), "--name", "mismatch", "--output-root", str(root)]
    assert main(args) == 1
    assert "mismatched dataset_revision" in capsys.readouterr().err
    assert not (root / "reports/mismatch/results/results.json").exists()


def test_report_exposes_revised_scorer_and_reuses_completed_sources(selected_runs):
    root, selection_file, paths = selected_runs
    assert main(["revise-logits", "--source", "batch2", "--plan", "v2", "--output-root", str(root)]) == 0
    class NoModel(Dependencies):
        def load_backend(self, config):
            raise AssertionError("completed source logits must be reused")
    manifest = json.loads((root / "plans/v2/manifest.json").read_text())
    for relative in manifest["configs"][3:]:
        execute_config(root / relative, root, dependencies=NoModel())
    selection = json.loads(selection_file.read_text())
    selection.update({cell: "v2" for cell in ("mmlu_logits-0shot", "mmlu_logits-5shot")})
    selection_file.write_text(json.dumps(selection))
    assert main(["report", "--selection", str(selection_file), "--name", "v2-report", "--output-root", str(root)]) == 0
    report = json.loads((root / "reports/v2-report/results/results.json").read_text())
    assert report["cells"][3]["scorer_version"] == "mmlu-logits-v2"
    assert "Ties count as invalid and incorrect" in (root / "reports/v2-report/results/report.md").read_text()
