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
    import subprocess, sys

    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot_eval",
            "traces-analysis-verify",
            "--config",
            str(plan),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["primary"] == final["primary"]
    # Literal fixture: every nonzero endpoint delta is [.25,.25], all five
    # vocabulary strings are valid; 'both' is one hit among five, hence .2.
    assert all(row["overlap"] == 0.2 for row in final["primary"])
    assert all(
        row["p_value"] == 1 and row["holm_p_value"] == 1 for row in final["primary"]
    )
    assert all(row["snr"] is None for row in final["primary"])
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


def test_batched_analysis_scores_match_independent_ticket03_scorer():
    from pilot_eval.trace_analysis_scores import LensScorer
    from pilot_eval.trace_scoring import score_vector
    from pilot_eval.trace_capture import logit_lens
    from pilot_eval.trace_vocabulary import TokenInventory

    inventory = TokenInventory(
        {0: " alpha", 1: "Alpha", 2: "beta", 3: " gamma", 4: "<special>", 5: "4"},
        special_ids=[4],
    )
    ref = dict(
        strings=["alpha", "4"],
        word=["alpha"],
        format_number=["4"],
        scores=[
            dict(string="alpha", y_a=6, y_b=1, z=4),
            dict(string="beta", y_a=5, y_b=5, z=0),
            dict(string="4", y_a=7, y_b=2, z=3),
        ],
    )
    references = {n: ref for n in ["ref_questions", "ref_gold", "ref_grpo"]}
    norm = np.array([1.0, 2.0])
    head = np.array([[1, 0], [1, 0], [0, 1], [-1, 1], [9, 9], [1, -1.0]])
    scorer = LensScorer(norm, head, 1e-6, inventory, references, batch_size=2)
    vectors = np.array([[1, 2], [-2, 1], [0, 0], [3, 6.0]])
    actual = scorer.many(vectors)
    for v, row in zip(vectors, actual):
        expected = score_vector(
            v,
            lambda x: logit_lens(x, norm_weight=norm, unembedding=head, epsilon=1e-6),
            inventory,
            references=references,
            synthetic=True,
        )
        assert row["state"] == expected["state"]
        if expected["state"] == "scored":
            assert row["strings"] == expected["strings"]
            assert row["list_ids"] == expected["list_ids"]
            for name in references:
                for metric, value in expected["scores"][name].items():
                    assert row["scores"][name][metric] == pytest.approx(value)
    assert actual[0]["scores"] == actual[3]["scores"]


def test_secondary_max_recomputes_each_resampled_position_before_taking_max(tmp_path):
    from pilot_eval.trace_analysis import prepare_analysis, run_nulls, run_analysis
    from pilot_eval.trace_readouts import run_readouts

    readouts = fixture(tmp_path)
    plan = prepare_analysis(
        readouts,
        tmp_path,
        "secondary",
        test_counts={
            "null": 4,
            "primary": 4,
            "c3": 4,
            "stratum": 4,
            "matched": 2,
            "ncurve": 2,
            "sizes": [25, 50, 100, 200],
        },
    )
    run_nulls(plan, tmp_path)
    run_readouts(readouts, tmp_path)
    result = run_analysis(plan, tmp_path)
    assert len(result["secondary_max"]) == 6
    for row in result["secondary_max"]:
        assert row["classification"] == "secondary" and row["lens"] == "logit"
        assert row["interval"] == [0.2, 0.2] and row["overlap"] == 0.2
        assert row["null_p_value"] is not None
    assert all(r["full_cohort_degeneracy"] for r in result["ncurves"] if r["n"] == 200)


def test_analysis_cli_and_notebook_preserve_prereveal_null_order():
    import ast, subprocess, sys
    from pathlib import Path

    command = subprocess.run(
        [sys.executable, "-m", "pilot_eval", "traces-analysis-prepare", "--help"],
        capture_output=True,
        text=True,
    )
    assert command.returncode == 0, command.stderr
    notebook = json.loads(Path("notebooks/pilot-5-colab.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    nulls = next(i for i, s in enumerate(sources) if "## 26A." in s)
    decode = next(i for i, s in enumerate(sources) if s.startswith("## 27."))
    assert nulls < decode
    assert any("## 29." in s for s in sources) and any("## 30." in s for s in sources)
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    decode_code = sources[decode + 1]
    assert "'traces-analysis-nulls-verify'" in decode_code


def test_batched_kernel_top20_membership_has_literal_task_contrast():
    from pilot_eval.trace_analysis_scores import LensScorer
    from pilot_eval.trace_vocabulary import TokenInventory

    inventory = TokenInventory({i: "token" + str(i) for i in range(21)})
    head = np.array([[1.0, 0.0]] + [[0.0, 1.0]] * 20)
    reference = dict(strings=["token0"], word=["token0"], format_number=[], scores=[])
    score = LensScorer(np.ones(2), head, 1e-6, inventory, {"ref_questions": reference})
    left, right = score.many([[1, 0], [0, 1]])
    assert left["scores"]["ref_questions"]["overlap"] == 0.05
    assert right["scores"]["ref_questions"]["overlap"] == 0
    assert left["list_ids"] == list(range(20))
    assert right["list_ids"] == list(range(1, 21))


def test_full_vocabulary_rank_kernel_preserves_ties_and_primary_score_rules():
    from pilot_eval.trace_analysis_scores import LensScorer
    from pilot_eval.trace_vocabulary import TokenInventory

    # A large tied candidate boundary: only the lowest IDs may survive.
    inventory = TokenInventory({i: "word" + str(i) for i in range(1002)})
    ref = dict(
        strings=["word0"],
        word=["word0"],
        format_number=[],
        scores=[
            dict(string="word0", y_a=5, y_b=1, z=1),
            dict(string="word1", y_a=5, y_b=1, z=1),
            dict(string="word2", y_a=5, y_b=1, z=3),
        ],
    )
    scorer = LensScorer(np.ones(2), np.zeros((1002, 2)), 1e-6, inventory, {"r": ref})
    scores = np.zeros(1002)
    scores[2] = 2
    result = scorer.score_logits(scores, norm=1)
    assert result["list_ids"] == [2, 0, 1] + list(range(3, 20))
    assert result["scores"]["r"]["soft_rank"] == pytest.approx(1, abs=1e-9, rel=0)
    assert result["scores"]["r"]["overlap"] == 0.05
