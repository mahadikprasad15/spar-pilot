"""Saved production capture, using an independent tiny CPU model boundary."""

import json
from pathlib import Path
import numpy as np
import pytest
from test_trace_profile import prepared, Deps, Engine


def frozen(root, *, approved=True, dual=False):
    from pilot_eval.trace_profile import run_profile
    from pilot_eval.trace_production_plan import propose_capture, freeze_capture
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash

    profile = prepared(root)
    run_profile(profile, root, dependencies=Deps(Engine()))
    proposal = propose_capture(profile, root, "production")
    data = json.loads(proposal.read_text())
    # Independently authored fixture qualification; never accepted by real CUDA.
    qualification = root / "fixture-qualification.json"
    qualification.write_text(
        json.dumps(
            dict(
                protocol="pilot5-capture-qualification-v1",
                qualified=True,
                execution_kind="synthetic-cpu",
                profile_sha256=file_hash(profile),
                model_revision=data["model_revision"],
                batch_size=4,
                storage_dtype="float32",
                checks={
                    k: dict(passed=True, evidence="independent fixture only")
                    for k in [
                        "zero_identity",
                        "base_integrity",
                        "numeric_lens",
                        "batching",
                        "retention",
                        "ranking",
                        "bootstrap",
                    ]
                },
            )
        )
    )
    (qualification.parent / (qualification.name + ".seal.json")).write_text(
        json.dumps(dict(sha256=file_hash(qualification)))
    )
    review = root / "capture-review.json"
    review.write_text(
        json.dumps(
            dict(
                approved=approved,
                reviewer="fixture-only",
                notes="Synthetic qualification and budget; never scientific evidence.",
                proposal_sha256=file_hash(proposal),
                conditions=["base", "sft-64", "grpo-64", "random-64"],
                batch_size=4,
                storage_dtype="float32",
                scope="endpoints",
                qualification_path=str(qualification),
                qualification_sha256=file_hash(qualification),
                lens_mode="logit-only",
                partial_mode_approved=True,
                lens_qualification_path=None,
                predictions={
                    arm: {"statement": "fixture prediction", "confidence": 0.5}
                    for arm in ["sft", "grpo"]
                },
                budget=dict(approved=True, max_seconds=10000, max_durable_bytes=100000000),
                primary=dict(
                    form="raw",
                    site="block-14-output",
                    step=64,
                    positions=[0, 1, 2, 3, 4],
                    pooling="positions-within-example-then-examples",
                ),
            )
        )
    )
    if dual:
        from test_trace_lenses import archive
        import zipfile

        lens = root / "lens.pt"
        archive(lens)
        with zipfile.ZipFile(lens) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        parts["lens/data.pkl"] = (
            parts["lens/data.pkl"]
            .replace(b"}K\x00ctorch._utils", b"}K\x0ectorch._utils")
            .replace(b"]K\x00a", b"]K\x0ea")
        )
        with zipfile.ZipFile(lens, "w") as z:
            for n, v in parts.items():
                z.writestr(n, v)
        qlens = root / "qualified-lens.json"
        qlens.write_text(
            json.dumps(
                dict(
                    qualified=True,
                    model_revision=data["model_revision"],
                    lens_path="lens.pt",
                    lens_sha256=file_hash(lens),
                    d_model=2,
                    source_layers=[14],
                    qualified_source_layers=[14],
                    positions=[0, 1, 2, 3, 4],
                    source_site="block-output",
                    target="final-block-output-before-norm",
                    evidence_files={"lens.pt": file_hash(lens)},
                )
            )
        )
        qlens.with_name(qlens.name + ".seal.json").write_text(
            json.dumps(dict(sha256=file_hash(qlens)))
        )
        r = json.loads(review.read_text())
        r.update(
            lens_mode="logit-and-jacobian",
            partial_mode_approved=False,
            lens_qualification_path=str(qlens),
        )
        review.write_text(json.dumps(r))
    return freeze_capture(proposal, review, root), profile


class CaptureEngine:
    def __init__(self, stop=None):
        self.calls = []
        self.stop = stop
        self.closed = False

    def integrity(self):
        return "fixture-base"

    def capture(self, rows, condition):
        from pilot_eval.trace_capture_workflow import SITES

        self.calls.append((condition, [r["id"] for r in rows]))
        if len(self.calls) == self.stop:
            raise RuntimeError("fixture disconnect")
        b = np.stack([np.asarray(r["input_ids"], dtype=np.float32) for r in rows])[:, :, None]
        b = np.concatenate([b, b + 1], axis=2)
        shift = {"base": 0, "sft-64": 0.25, "grpo-64": 0.5, "random-64": -0.125}[condition]
        return {s: b + i + shift for i, s in enumerate(SITES)}

    def close(self):
        self.closed = True


class Dependencies:
    def __init__(self, engine):
        self.instance = engine

    def runtime(self):
        return dict(execution_kind="synthetic-cpu", device="fixture")

    def engine(self, config, root):
        return self.instance


def test_capture_requires_review_and_fixture_cannot_qualify_real_execution(tmp_path):
    from pilot_eval.trace_production_plan import freeze_capture, load_execution

    with pytest.raises(ValueError, match="approval"):
        frozen(tmp_path, approved=False)
    review = tmp_path / "capture-review.json"
    data = json.loads(review.read_text())
    data["approved"] = True
    review.write_text(json.dumps(data))
    proposal = tmp_path / "plans/production/traces.production.proposal.json"
    path = freeze_capture(proposal, review, tmp_path)
    assert load_execution(path, tmp_path)["execution_kind"] == "synthetic-cpu"
    from pilot_eval.trace_production import run_capture

    class Real(Dependencies):
        def runtime(self):
            return dict(execution_kind="real-cuda", device="fixture-forbidden")

    e = CaptureEngine()
    with pytest.raises(ValueError, match="execution kind"):
        run_capture(path, tmp_path, dependencies=Real(e))
    assert not e.calls


def test_production_resume_matches_uninterrupted_sums_and_durable_release(tmp_path):
    from pilot_eval.trace_production import run_capture, verify_capture, load_aggregate

    path, _ = frozen(tmp_path)
    first = CaptureEngine(stop=6)
    with pytest.raises(RuntimeError, match="disconnect"):
        run_capture(path, tmp_path, dependencies=Dependencies(first))
    assert first.closed
    directory = tmp_path / json.loads(path.read_text())["run_path"]
    assert (directory / "units/000000/complete.json").exists()
    assert not (directory / "complete.json").exists()
    second = CaptureEngine()
    result = run_capture(path, tmp_path, dependencies=Dependencies(second))
    assert len(second.calls) == 49 * 4
    assert result["durable_verified"] and result["gpu_release_safe"]
    assert result["examples"] == 200 and result["base_moment_count"] == 25400
    assert result["new_batches"] == 49 and result["reused_batches"] == 1
    sums = load_aggregate(path, tmp_path)
    np.testing.assert_allclose(sums["sft-64.position_sum.block-14-output"], np.full((128, 2), 50))
    np.testing.assert_allclose(sums["grpo-64.position_sum.block-14-output"], np.full((128, 2), 100))
    assert verify_capture(path, tmp_path) == result

    class NoGPU:
        def runtime(self):
            raise AssertionError("completed capture must verify on CPU")

    assert run_capture(path, tmp_path, dependencies=NoGPU()) == result
    shard = directory / "units/000000/arrays.npz"
    shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        verify_capture(path, tmp_path)


def test_production_disk_failure_preserves_no_premature_release(tmp_path):
    from pilot_eval.trace_production import run_capture

    path, _ = frozen(tmp_path)

    class NoSpace(Dependencies):
        def free_bytes(self, path):
            return 0

    e = CaptureEngine()
    with pytest.raises(OSError, match="space"):
        run_capture(path, tmp_path, dependencies=NoSpace(e))
    assert not e.calls
    directory = tmp_path / json.loads(path.read_text())["run_path"]
    assert not (directory / "complete.json").exists()
    assert json.loads((directory / "meta/status.json").read_text())["state"] == "failed"


def test_production_refuses_unqualified_gpu_change_before_model(tmp_path):
    from pilot_eval.trace_production import run_capture

    path, _ = frozen(tmp_path)

    class OtherGPU(Dependencies):
        def runtime(self):
            return dict(execution_kind="synthetic-cpu", device="different GPU")

    engine = CaptureEngine()
    with pytest.raises(ValueError, match="qualified runtime"):
        run_capture(path, tmp_path, dependencies=OtherGPU(engine))
    assert not engine.calls


def test_public_production_cli_verifies_complete_fixture_without_model(tmp_path):
    import subprocess, sys
    from pilot_eval.trace_production import run_capture

    path, _ = frozen(tmp_path)
    run_capture(path, tmp_path, dependencies=Dependencies(CaptureEngine()))
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "pilot_eval",
            "traces-production-verify",
            "--config",
            str(path),
            "--output-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert output.returncode == 0, output.stdout + output.stderr
    assert json.loads(output.stdout)["gpu_release_safe"] is True


def test_freeze_rejects_unqualified_receipt_not_just_profile_success(tmp_path):
    from pilot_eval.trace_production_plan import freeze_capture
    from pilot_eval.training import file_hash

    with pytest.raises(ValueError, match="approval"):
        frozen(tmp_path, approved=False)
    review = tmp_path / "capture-review.json"
    r = json.loads(review.read_text())
    r["approved"] = True
    qualification = Path(r["qualification_path"])
    q = json.loads(qualification.read_text())
    q["qualified"] = False
    qualification.write_text(json.dumps(q))
    qualification.with_name(qualification.name + ".seal.json").write_text(
        json.dumps(dict(sha256=file_hash(qualification)))
    )
    r["qualification_sha256"] = file_hash(qualification)
    review.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="Qualified capture receipt"):
        freeze_capture(
            tmp_path / "plans/production/traces.production.proposal.json", review, tmp_path
        )


def test_sync_failure_reuses_verified_stage_without_repeating_its_forward(tmp_path, monkeypatch):
    import shutil
    from pilot_eval.trace_production import run_capture

    path, _ = frozen(tmp_path)
    original = shutil.copyfile
    failed = False

    def interrupted_copy(source, destination, *args, **kwargs):
        nonlocal failed
        if not failed and str(destination).endswith("arrays.npz.partial"):
            failed = True
            Path(destination).write_bytes(b"partial")
            raise OSError("simulated sync failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", interrupted_copy)
    first = CaptureEngine()
    with pytest.raises(OSError, match="sync failure"):
        run_capture(path, tmp_path, dependencies=Dependencies(first))
    assert len(first.calls) == 4 and first.closed
    directory = tmp_path / json.loads(path.read_text())["run_path"]
    assert not (directory / "units/000000/complete.json").exists()
    second = CaptureEngine()
    result = run_capture(path, tmp_path, dependencies=Dependencies(second))
    assert len(second.calls) == 196 and result["gpu_release_safe"]


def test_capture_writer_conflict_stops_before_model(tmp_path):
    from pilot_eval.trace_production import run_capture
    from pilot_eval.training import run_lock

    path, _ = frozen(tmp_path)
    directory = tmp_path / json.loads(path.read_text())["run_path"]
    engine = CaptureEngine()
    with run_lock(directory), pytest.raises(RuntimeError, match="another process"):
        run_capture(path, tmp_path, dependencies=Dependencies(engine))
    assert not engine.calls


def test_capture_freeze_requires_actual_prediction_confidences(tmp_path):
    from pilot_eval.trace_production_plan import freeze_capture

    with pytest.raises(ValueError, match="approval"):
        frozen(tmp_path, approved=False)
    review = tmp_path / "capture-review.json"
    r = json.loads(review.read_text())
    r["approved"] = True
    r["predictions"] = {"sft": {"statement": "fixture"}, "grpo": {"statement": "fixture"}}
    review.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="confidence"):
        freeze_capture(
            tmp_path / "plans/production/traces.production.proposal.json", review, tmp_path
        )
