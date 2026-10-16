"""Capture acceptance through the saved CPU workflow; no decoder downloads."""

import json
import pytest
from test_trace_lens_qualification import inputs
from test_trace_capture_workflow import Engine, Dependencies


def test_capture_review_cannot_be_replaced_by_profile_completion(tmp_path, monkeypatch):
    from pilot_eval.trace_capture_qualification import (
        prepare,
        diagnose,
        verify,
        approve,
    )
    from pilot_eval.trace_capture_workflow import run_diagnostic
    from pilot_eval.training import file_hash

    from test_trace_profile import Engine as ProfileEngine

    class BatchDrift(ProfileEngine):
        def capture(self, rows, condition):
            values = super().capture(rows, condition)
            return {k: v + (0.25 if len(rows) == 4 else 0) for k, v in values.items()}

    monkeypatch.setattr("test_trace_lens_qualification.Engine", BatchDrift)
    profile, inspection, assets = inputs(tmp_path)
    p = json.loads(profile.read_text())
    run_diagnostic(
        tmp_path / p["diagnostic_path"], tmp_path, dependencies=Dependencies(Engine())
    )
    plan = prepare(
        profile,
        assets,
        tmp_path,
        "capture-check",
        batch_size=4,
        storage_dtype="float32",
        bootstrap_draws=8,
        seed=42,
    )
    first = diagnose(plan, tmp_path, max_new_units=1)
    assert first["completed_units"] == 1 and not first["qualified"]
    result = diagnose(plan, tmp_path)
    assert result["diagnostic_complete"] and result["decoder_forwards"] == 0
    assert not result["qualified"] and result["reused_units"] == 1
    assert result["capture"]["zero_identity"] and result["capture"]["base_integrity"]
    assert result["capture"]["batch_max_absolute_difference"] == 0.25
    assert result["capture"]["retention_max_absolute_error"] == 0
    assert verify(plan, tmp_path) == result
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    review = json.loads((directory / "review.draft.json").read_text())
    path = directory / "review.fixture.json"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="approval"):
        approve(plan, path, tmp_path)
    review.update(
        approved=True,
        reviewer="fixture",
        notes="Synthetic acceptance only",
        bounded_scope_accepted=True,
    )
    review["limits"] = dict(
        batch_max_absolute_difference=0.25,
        retention_max_absolute_error=0,
        numeric_lens_max_score_error=1e-10,
        cohort_max_score_error=1e-6,
        bootstrap_max_score_error=1e-6,
        cohort_min_top20_overlap=1,
        bootstrap_min_top20_overlap=1,
    )
    path.write_text(json.dumps(review))
    receipt_path = approve(plan, path, tmp_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["qualified"] and receipt["execution_kind"] == "synthetic-cpu"
    assert receipt["protocol"] == "pilot5-capture-qualification-v1"
    assert all(v["passed"] for v in receipt["checks"].values())
    assert receipt["profile_sha256"] == file_hash(profile)
    # The producer's receipt is accepted by the actual production-freeze seam.
    from pilot_eval.trace_production_plan import propose_capture, freeze_capture

    proposal = propose_capture(profile, tmp_path, "production")
    production = json.loads(proposal.read_text())["review_template"]
    production.update(
        approved=True,
        reviewer="fixture",
        notes="Synthetic workflow integration",
        proposal_sha256=file_hash(proposal),
        conditions=["base", "sft-64", "grpo-64", "random-64"],
        batch_size=4,
        storage_dtype="float32",
        scope="endpoints",
        qualification_path=str(receipt_path.relative_to(tmp_path)),
        qualification_sha256=file_hash(receipt_path),
        lens_mode="logit-only",
        partial_mode_approved=True,
        predictions={
            a: dict(statement="Synthetic fixture only", confidence=0.5)
            for a in ["sft", "grpo"]
        },
        budget=dict(approved=True, max_seconds=10000, max_durable_bytes=100000000),
    )
    production_review = tmp_path / "production-review.json"
    production_review.write_text(json.dumps(production))
    frozen = freeze_capture(proposal, production_review, tmp_path)
    assert json.loads(frozen.read_text())["mode"] == "frozen-production-capture"
    import subprocess, sys

    fresh = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot_eval",
            "traces-capture-qualification-verify",
            "--config",
            str(plan),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert fresh.returncode == 0, fresh.stderr
    # A stricter reviewed decision is saved as failed, never silently widened.
    review["limits"]["batch_max_absolute_difference"] = 0
    path2 = directory / "review.strict.json"
    path2.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="gate failed"):
        approve(plan, path2, tmp_path)
    assert json.loads(receipt_path.read_text())["qualified"]
    # Completed evidence is checked, not trusted based on status alone.
    (directory / "units/operator/result.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        verify(plan, tmp_path)


def test_capture_qualification_cli_and_notebook_handoff(tmp_path):
    import ast
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot_eval",
            "traces-capture-qualification-prepare",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    notebook = json.loads(Path("notebooks/pilot-5-colab.ipynb").read_text())
    headings = [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    ]
    assert any("## 21A." in h for h in headings)
    assert any("## 21B." in h for h in headings)
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))


def test_notebook_pin_contains_capture_qualification_commands():
    import re
    import subprocess
    from pathlib import Path

    notebook = json.loads(Path("notebooks/pilot-5-colab.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    commit = re.search(r"HARNESS_COMMIT = '([0-9a-f]{40})'", source).group(1)
    cli = subprocess.check_output(
        ["git", "show", commit + ":pilot_eval/cli.py"], text=True
    )
    assert "traces-capture-qualification-" in cli
    module = subprocess.check_output(
        ["git", "show", commit + ":pilot_eval/trace_capture_qualification.py"],
        text=True,
    )
    assert "pilot5-capture-qualification-v1" in module


def test_capture_reports_literal_fp16_conversion_error_without_accepting_it(
    tmp_path, monkeypatch
):
    from test_trace_profile import Engine as ProfileEngine
    from pilot_eval.trace_capture_workflow import run_diagnostic
    from pilot_eval.trace_capture_qualification import prepare, diagnose, approve

    class ConstantBase(ProfileEngine):
        def capture(self, rows, condition):
            return {k: v * 0 for k, v in super().capture(rows, condition).items()}

    monkeypatch.setattr("test_trace_lens_qualification.Engine", ConstantBase)
    profile, _, assets = inputs(tmp_path)
    p = json.loads(profile.read_text())
    run_diagnostic(
        tmp_path / p["diagnostic_path"], tmp_path, dependencies=Dependencies(Engine())
    )
    plan = prepare(
        profile,
        assets,
        tmp_path,
        "fp16-check",
        batch_size=4,
        storage_dtype="float16",
        bootstrap_draws=8,
        seed=42,
    )
    result = diagnose(plan, tmp_path)
    # IEEE representations: f32 .12345670163631439 → f16 .12347412109375.
    assert result["capture"]["retention_max_absolute_error"] == 0.00001741945743560791
    assert not result["qualified"]
    folder = tmp_path / json.loads(plan.read_text())["run_path"]
    review = json.loads((folder / "review.draft.json").read_text())
    review.update(
        approved=True,
        reviewer="fixture",
        notes="Strict literal error rejection",
        bounded_scope_accepted=True,
    )
    review["limits"] = dict(
        batch_max_absolute_difference=0,
        retention_max_absolute_error=0,
        numeric_lens_max_score_error=1e-10,
        cohort_max_score_error=1,
        bootstrap_max_score_error=1,
        cohort_min_top20_overlap=0,
        bootstrap_min_top20_overlap=0,
    )
    path = folder / "review.strict.json"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="retention"):
        approve(plan, path, tmp_path)
    # Mutating an upstream payload must fail before any reuse/computation.
    assets.write_text("{}")
    with pytest.raises(ValueError, match="parent hash"):
        diagnose(plan, tmp_path)
