"""Ticket09 public saved-analysis seam, tiny CPU artifacts only."""

import json
import numpy as np
import pytest
from test_trace_readouts import evidence


def fixture(tmp_path):
    from pilot_eval.trace_vocabulary import freeze_references, approve_references
    from pilot_eval.trace_reference_workflow import approved_procedure
    from pilot_eval.trace_readouts import prepare_readouts

    capture, assets, _, review = evidence(tmp_path, dual=True)
    ref = dict(
        strings=["both"],
        word=["both"],
        format_number=[],
        scores=[
            dict(string="both", y_a=10, y_b=5, z=2),
            dict(string="first", y_a=5, y_b=8, z=1),
        ],
    )
    refs = freeze_references(
        tmp_path / "analysis-refs",
        {n: ref for n in ["ref_questions", "ref_gold", "ref_grpo"]},
        parameters=dict(procedure=approved_procedure()),
        input_files={"capture": capture},
    )
    (refs.parent / "top30.md").write_text("Independent reference fixture")
    approve_references(
        refs, reviewer="fixture", notes="Before reveal", exposure_status="not-seen"
    )
    r = json.loads(review.read_text())
    r["references_sha256"] = __import__(
        "pilot_eval.training", fromlist=["file_hash"]
    ).file_hash(refs)
    review.write_text(json.dumps(r))
    readouts = prepare_readouts(
        capture, assets, refs, review, tmp_path, "analysis-readouts"
    )
    return readouts


def test_nulls_are_durable_before_primary_scores_and_analysis_resumes(tmp_path):
    from pilot_eval.trace_analysis import (
        prepare_analysis,
        run_nulls,
        verify_nulls,
        run_analysis,
        verify_analysis,
    )
    from pilot_eval.trace_readouts import run_readouts

    readouts = fixture(tmp_path)
    plan = prepare_analysis(
        readouts,
        tmp_path,
        "automatic",
        test_counts={
            "null": 8,
            "primary": 8,
            "c3": 8,
            "stratum": 8,
            "matched": 4,
            "ncurve": 3,
            "sizes": [25, 50, 100, 200],
        },
    )
    assert json.loads(plan.read_text())["execution_kind"] == "synthetic-cpu"
    with pytest.raises((ValueError, FileNotFoundError), match="null|complete"):
        run_analysis(plan, tmp_path, max_new_units=1)
    first = run_nulls(plan, tmp_path, max_new_chunks=1)
    assert not first["nulls_complete"] and first["decoder_forwards"] == 0
    nulls = run_nulls(plan, tmp_path)
    assert nulls["nulls_complete"] and verify_nulls(plan, tmp_path) == nulls
    run_readouts(readouts, tmp_path)
    partial = run_analysis(plan, tmp_path, max_new_units=1)
    assert not partial["analysis_complete"]
    final = run_analysis(plan, tmp_path)
    assert final["analysis_complete"] and final["reused_units"] >= 1
    assert final["decoder_forwards"] == 0 and final["synthetic_only"]
    assert len(final["primary"]) == 4 and len(final["contrasts"]) == 2
    # Literal fixture: every nonzero endpoint delta is [.25,.25], all five
    # vocabulary strings are valid; 'both' is one hit among five, hence .2.
    assert all(row["overlap"] == 0.2 for row in final["primary"])
    assert all(row["difference"] == 0 for row in final["contrasts"])
    assert all(row["p_value"] == 1 for row in final["contrasts"])
    assert all(row["interval"] == [0, 0] for row in final["contrasts"])
    assert verify_analysis(plan, tmp_path) == final
    assert run_analysis(plan, tmp_path) == final
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    assert (directory / "results/report.md").is_file()
    assert (directory / "results/scores.jsonl").is_file()
    (directory / "results/report.md").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        verify_analysis(plan, tmp_path)
