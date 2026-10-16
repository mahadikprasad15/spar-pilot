"""Public saved qualification seam; synthetic fixtures, no Torch or downloads."""

import json
import zipfile
from pathlib import Path
import numpy as np
import pytest
from test_trace_profile import prepared, Engine, Deps
from test_trace_lenses import archive


def inputs(root):
    from pilot_eval.trace_profile import run_profile
    from pilot_eval.trace_readout_assets import prepare_assets
    from pilot_eval.trace_lenses import inspect_lens
    from pilot_eval.training import file_hash

    class NonzeroEngine(Engine):
        def capture(self, rows, condition):
            values = super().capture(rows, "base")
            if condition != "base":
                values = {
                    key: value + np.float32(0.1234567) for key, value in values.items()
                }
            return values

    profile = prepared(root)
    run_profile(profile, root, dependencies=Deps(NonzeroEngine()))
    lens = root / "lens.pt"
    archive(lens)
    with zipfile.ZipFile(lens) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    entries["lens/data.pkl"] = (
        entries["lens/data.pkl"]
        .replace(b"}K\x00ctorch", b"}K\x0ectorch")
        .replace(b"]K\x00a", b"]K\x0ea")
    )
    with zipfile.ZipFile(lens, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    inspection = inspect_lens(
        lens,
        root,
        "lens-inspect",
        expected_sha256=file_hash(lens),
        provenance=dict(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            model_revision=None,
            source_site="block-output",
            target=None,
        ),
        expected_model="Qwen/Qwen2.5-1.5B-Instruct",
        expected_revision="model-pin",
        expected_width=2,
        primary_block=14,
    )

    class Assets:
        def load(self, config):
            return dict(
                execution_kind="synthetic-cpu",
                norm=np.array([1.0, 2.0], dtype=np.float32),
                head=np.array(
                    [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=np.float32
                ),
                epsilon=1e-6,
                decoded={0: " first", 1: "second", 2: "third", 3: "fourth", 4: "fifth"},
                excluded_ids=[],
                origin_files={"fixture": "fixture"},
            )

    assets = prepare_assets(
        profile, root, "qualification-assets", dependencies=Assets()
    )
    return profile, inspection, assets


def test_lens_diagnostics_reuse_profile_with_no_decoder_and_require_explicit_review(
    tmp_path,
):
    from pilot_eval.trace_lens_qualification import prepare, diagnose, verify, approve
    from pilot_eval.training import file_hash

    profile, inspection, assets = inputs(tmp_path)
    plan = prepare(
        profile,
        inspection,
        assets,
        tmp_path,
        "lens-qualification",
        bootstrap_draws=8,
        seed=42,
    )
    # One finished unit is reusable after an interrupted execution.
    first = diagnose(plan, tmp_path, max_new_units=1)
    assert first["state"] == "incomplete" and first["completed_units"] == 1
    result = diagnose(plan, tmp_path)
    assert result["diagnostic_complete"] and result["qualified"] is False
    assert (
        result["decoder_forwards"] == 0
        and result["new_units"] == 12
        and result["reused_units"] == 1
    )
    assert result["bootstrap_draws"] == 8 and result["source_block"] == 14
    assert verify(plan, tmp_path) == result
    op = result["operator"]
    assert op["maximum_score_error"] < 1e-12
    np.testing.assert_allclose(
        op["probe_transported_prefix"], [12, 26]
    )  # J=[[1,2],[3,4]], fixed probe [2,5]
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            dict(
                approved=True,
                reviewer="fixture-review",
                notes="fixture acceptance",
                diagnostic_sha256=file_hash(
                    tmp_path
                    / json.loads(plan.read_text())["run_path"]
                    / "results/results.json"
                ),
                model_identity_accepted=True,
                target_convention_accepted=True,
                provenance_limitations_accepted=True,
                batch_size=4,
                storage_dtype="float32",
                limits=None,
            )
        )
    )
    with pytest.raises(ValueError, match="limits"):
        approve(plan, review, tmp_path)
    assert "readouts" not in json.dumps(result) and "decoded" not in json.dumps(result)
    again = diagnose(plan, tmp_path)
    assert again == result


def test_changed_lens_or_profile_is_rejected_before_numeric_work(tmp_path):
    from pilot_eval.trace_lens_qualification import prepare, diagnose

    profile, inspection, assets = inputs(tmp_path)
    plan = prepare(
        profile, inspection, assets, tmp_path, "changed", bootstrap_draws=8, seed=42
    )
    (tmp_path / "lens.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        diagnose(plan, tmp_path)


def test_public_cli_and_reviewed_receipt_preserve_failed_decisions(tmp_path):
    import subprocess
    import sys
    from pilot_eval.trace_lens_qualification import prepare, diagnose, approve
    from pilot_eval.training import file_hash

    profile, inspection, assets = inputs(tmp_path)
    plan = prepare(
        profile, inspection, assets, tmp_path, "reviewed", bootstrap_draws=8, seed=42
    )
    diagnose(plan, tmp_path)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot_eval",
            "traces-lens-qualification-verify",
            "--config",
            str(plan),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    review = json.loads((directory / "review.draft.json").read_text())
    review.update(
        approved=True,
        reviewer="fixture",
        notes="Strict synthetic checks only",
        model_identity_accepted=True,
        target_convention_accepted=True,
        provenance_limitations_accepted=True,
        batch_size=4,
        storage_dtype="float32",
        limits=dict(
            operator_max_score_error=1e-12,
            conversion_max_absolute_error=0,
            cohort_max_score_error=0,
            cohort_min_top20_overlap=1,
            bootstrap_max_score_error=0,
            bootstrap_min_top20_overlap=1,
        ),
    )
    review_path = tmp_path / "strict-review.json"
    review_path.write_text(json.dumps(review))
    receipt_path = approve(plan, review_path, tmp_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["qualified"] is True and receipt["qualified_source_layers"] == [14]
    assert receipt["execution_kind"] == "synthetic-cpu"
    assert (
        file_hash(receipt_path)
        == json.loads(
            receipt_path.with_name("qualification.json.seal.json").read_text()
        )["sha256"]
    )
    # Lossy FP16 conversion must fail a predeclared exact-conversion fixture gate.
    review["storage_dtype"] = "float16"
    failed_review = tmp_path / "failed-review.json"
    failed_review.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="gate failed"):
        approve(plan, failed_review, tmp_path)
    decisions = [
        json.loads(p.read_text())
        for p in (directory / "reviews").glob("*/qualification.json")
    ]
    assert sorted(r["qualified"] for r in decisions) == [False, True]


def test_notebook_exposes_cpu_qualification_before_production():
    import ast

    notebook = json.loads(Path("notebooks/pilot-5-colab.ipynb").read_text())
    cells = notebook["cells"]
    qualification = [
        c
        for c in cells
        if c.get("metadata", {}).get("pilot5_stage") == "lens-qualification"
    ]
    code = [c for c in qualification if c["cell_type"] == "code"]
    assert len(code) == 3
    joined = "\n".join("".join(c["source"]) for c in code)
    assert "traces-readout-assets" in joined
    assert "traces-lens-qualification-" in joined and "['diagnose', 'verify']" in joined
    assert "traces-lens-qualification-approve" in joined
    assert "--bootstrap-draws" in joined and "--seed" in joined
    for cell in code:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    assert cells.index(qualification[-1]) < next(
        i for i, c in enumerate(cells) if "## 21." in "".join(c["source"])
    )
