"""Saved readout consumer: reveal barriers, literal ranks, no decoder inference."""

import json
from pathlib import Path
import numpy as np
import pytest
from test_trace_production import frozen, CaptureEngine, Dependencies


def evidence(root, *, reviewed=True, dual=False):
    from pilot_eval.trace_production import run_capture
    from pilot_eval.trace_readout_assets import prepare_assets
    from pilot_eval.trace_vocabulary import freeze_references, approve_references
    from pilot_eval.training import file_hash

    path, _ = frozen(root, dual=dual)
    run_capture(path, root, dependencies=Dependencies(CaptureEngine()))

    class Assets:
        def load(self, config):
            return dict(
                execution_kind="synthetic-cpu",
                norm=np.ones(2, dtype=np.float32),
                head=np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=np.float32),
                epsilon=1e-6,
                decoded={0: " first", 1: " second", 2: " minus", 3: " negative", 4: " both"},
                excluded_ids=[],
                origin_files={},
            )

    assets = prepare_assets(path, root, "assets", dependencies=Assets())
    refs = {name: dict(strings=["both"]) for name in ["ref_questions", "ref_gold", "ref_grpo"]}
    references = freeze_references(
        root / "references", refs, parameters={"fixture": True}, input_files={"production": path}
    )
    (references.parent / "top30.md").write_text("independent reference fixture")
    if reviewed:
        approve_references(
            references, reviewer="fixture", notes="Reference inspection", exposure_status="not-seen"
        )
    review = root / "readout-review.json"
    review.write_text(
        json.dumps(
            dict(
                approved=True,
                reviewer="fixture",
                notes="Predeclared readouts",
                capture_sha256=file_hash(path),
                references_sha256=file_hash(references),
                primary_family="fixture family; no inferential claims",
                requests=[
                    dict(
                        name="sft-primary",
                        condition="sft-64",
                        role="delta",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                    ),
                    dict(
                        name="grpo-primary",
                        condition="grpo-64",
                        role="delta",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                    ),
                    dict(
                        name="base-control",
                        condition="base",
                        role="base",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                    ),
                    dict(
                        name="zero",
                        condition="base",
                        role="delta",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                    ),
                    dict(
                        name="random-reference",
                        condition="random-64",
                        role="delta",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                    ),
                    dict(
                        name="gaussian",
                        condition="sft-64",
                        role="random-direction",
                        site="block-14-output",
                        positions=[0, 1, 2, 3, 4],
                        seed=7,
                    ),
                ],
                lens_names=["logit"],
                partial_mode_approved=True,
            )
        )
    )
    if dual:
        r = json.loads(review.read_text())
        r.update(lens_names=["logit", "jacobian"], partial_mode_approved=False)
        review.write_text(json.dumps(r))
    return path, assets, references, review


def test_readout_refuses_unreviewed_references_before_loading_arrays(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts

    path, assets, references, review = evidence(tmp_path, reviewed=False)
    with pytest.raises(ValueError, match="reference review"):
        prepare_readouts(path, assets, references, review, tmp_path, "readouts")
    assert not (tmp_path / "runs/pilot-5/readouts/readouts/complete.json").exists()


def test_readout_ranks_saved_primary_vectors_and_keeps_controls_distinct(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts, run_readouts, verify_readouts

    path, assets, references, review = evidence(tmp_path)
    plan = prepare_readouts(path, assets, references, review, tmp_path, "readouts")
    result = run_readouts(plan, tmp_path)
    assert result["readouts_complete"] and result["decoder_forwards"] == 0
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    sft = json.loads((directory / "units/sft-primary/result.json").read_text())
    assert sft["readouts"]["logit"]["positive"]["raw_top_tokens"][0]["token_id"] == 4
    assert sft["readouts"]["logit"]["negative"]["raw_top_tokens"][0]["token_id"] == 2
    np.testing.assert_allclose(np.load(directory / "units/sft-primary/vector.npy"), [0.25, 0.25])
    assert sft["coherence"] == pytest.approx(1.0)
    zero = json.loads((directory / "units/zero/result.json").read_text())
    assert zero["state"] == "undefined" and zero["reason"] == "zero-vector"
    roles = [
        json.loads((directory / "units" / n / "result.json").read_text())["role"]
        for n in ["base-control", "random-reference", "gaussian"]
    ]
    assert roles == ["base", "delta", "random-direction"]
    assert verify_readouts(plan, tmp_path) == result
    assert run_readouts(plan, tmp_path) == result
    (directory / "units/sft-primary/vector.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        verify_readouts(plan, tmp_path)


def test_readouts_resume_only_missing_units_and_report_unsupported_control(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts, run_readouts

    path, assets, references, review = evidence(tmp_path)
    data = json.loads(review.read_text())
    data["requests"].append(
        dict(
            name="unsupported-base-window",
            condition="base",
            role="base",
            site="block-14-output",
            positions=[127],
        )
    )
    review.write_text(json.dumps(data))
    plan = prepare_readouts(path, assets, references, review, tmp_path, "resume-readouts")
    first = run_readouts(plan, tmp_path, max_new_units=2)
    assert not first["readouts_complete"] and first["new_units"] == 2
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    original = (directory / "units/sft-primary/result.json").read_bytes()
    second = run_readouts(plan, tmp_path)
    assert second["new_units"] == 5 and second["reused_units"] == 2
    assert (directory / "units/sft-primary/result.json").read_bytes() == original
    unsupported = json.loads((directory / "units/unsupported-base-window/result.json").read_text())
    assert unsupported["state"] == "unavailable" and unsupported["readouts"] == {}


def test_cpu_readout_weight_loader_handles_bfloat16_without_torch(tmp_path):
    from pilot_eval.trace_readout_assets import read_weight

    header = json.dumps({"weight": dict(dtype="BF16", shape=[1, 2], data_offsets=[0, 4])}).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes.fromhex("c03f00c0"))
    np.testing.assert_array_equal(read_weight(path, "weight"), [[1.5, -2.0]])
    bad = json.dumps(
        {"weight": dict(dtype="BF16", shape=[1000, 1000], data_offsets=[0, 4])}
    ).encode()
    path.write_bytes(len(bad).to_bytes(8, "little") + bad + bytes(4))
    with pytest.raises(ValueError, match="byte|shape"):
        read_weight(path, "weight")


def test_public_readout_cli_resumes_and_verifies_without_decoder(tmp_path):
    import subprocess, sys
    from pilot_eval.trace_readouts import prepare_readouts

    capture, assets, refs, review = evidence(tmp_path)
    plan = prepare_readouts(capture, assets, refs, review, tmp_path, "cli-readout")
    common = ["--config", str(plan), "--output-root", str(tmp_path)]
    result = subprocess.run(
        [sys.executable, "-m", "pilot_eval", "traces-readouts-run", *common],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "pilot_eval", "traces-readouts-verify", *common],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["decoder_forwards"] == 0


def test_qualified_dual_readout_uses_source_matrix_orientation(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts, run_readouts

    capture, assets, refs, review = evidence(tmp_path, dual=True)
    plan = prepare_readouts(capture, assets, refs, review, tmp_path, "dual-readouts")
    run_readouts(plan, tmp_path)
    directory = tmp_path / json.loads(plan.read_text())["run_path"]
    result = json.loads((directory / "units/sft-primary/result.json").read_text())
    assert result["readouts"]["logit"]["negative"]["raw_top_tokens"][0]["token_id"] == 2
    assert result["readouts"]["jacobian"]["negative"]["raw_top_tokens"][0]["token_id"] == 3


def test_shared_notebook_has_ordered_capture_and_readout_stages():
    import ast

    notebook = json.loads(Path("notebooks/pilot-5-colab.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("pilot5_stage") != "lens-qualification"]
    assert len(cells) == 28
    for number, command in [
        (21, "traces-production-propose"),
        (22, "traces-production-freeze"),
        (23, "traces-production-run"),
        (24, "traces-production-verify"),
        (25, "traces-readout-assets"),
        (26, "traces-readouts-prepare"),
        (27, "traces-readouts-run"),
        (28, "traces-readouts-verify"),
    ]:
        assert repr(command) in cells[number - 1]
        assert "saved_selection" in cells[number - 1]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_readout_basis_can_be_prepared_from_profile_before_production(tmp_path):
    from pilot_eval.trace_readout_assets import prepare_assets, verify_assets
    from test_trace_profile import prepared, Deps, Engine
    from pilot_eval.trace_profile import run_profile

    profile = prepared(tmp_path)
    run_profile(profile, tmp_path, dependencies=Deps(Engine()))

    class Basis:
        def load(self, config):
            return dict(
                execution_kind="synthetic-cpu",
                norm=np.ones(2, dtype=np.float32),
                head=np.eye(2, dtype=np.float32),
                epsilon=1e-6,
                decoded={0: "a", 1: "b"},
                excluded_ids=[],
                origin_files={},
            )

    path = prepare_assets(profile, tmp_path, "prequalification-basis", dependencies=Basis())
    result = verify_assets(path, tmp_path)
    assert result["decoder_forwards"] == 0 and result["compute_device"] == "cpu"
    assert result["source_path"] == str(profile.relative_to(tmp_path))


def test_primary_window_qualification_does_not_authorize_other_jacobian_windows(tmp_path):
    from pilot_eval.trace_readouts import prepare_readouts, run_readouts
    capture, assets, refs, review = evidence(tmp_path, dual=True)
    data = json.loads(review.read_text())
    data['requests'].append(dict(name='unqualified-window', condition='sft-64', role='delta',
        site='block-14-output', positions=[0]))
    review.write_text(json.dumps(data))
    plan = prepare_readouts(capture, assets, refs, review, tmp_path, 'scoped-readouts')
    run_readouts(plan, tmp_path)
    directory = tmp_path / json.loads(plan.read_text())['run_path']
    result = json.loads((directory / 'units/unqualified-window/result.json').read_text())
    assert result['readouts']['logit']['state'] == 'decoded'
    assert result['readouts']['jacobian'] == dict(state='unavailable', reason='outside qualified source/window scope')
