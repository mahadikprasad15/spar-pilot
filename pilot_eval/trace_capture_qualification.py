"""Saved-profile capture diagnostics and explicit reviewed acceptance.

CPU only. No default numerical limits, new model capture, or learned-token reveal.
The qualification is bounded instrument evidence, not whole-corpus validation.
"""

import json
import math
from pathlib import Path
import numpy as np
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_profile import load_profile, verify_profile, _arrays, CONDITIONS
from pilot_eval.trace_capture_workflow import verify_diagnostic
from pilot_eval.trace_readout_assets import verify_assets
from pilot_eval.trace_lens_qualification import _operator, _sensitivity, _unit
from pilot_eval.trace_vocabulary import TokenInventory
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name

LIMITS = {
    "batch_max_absolute_difference",
    "retention_max_absolute_error",
    "numeric_lens_max_score_error",
    "cohort_max_score_error",
    "bootstrap_max_score_error",
    "cohort_min_top20_overlap",
    "bootstrap_min_top20_overlap",
}
UNITS = ["capture", "operator", "sft-64", "grpo-64", "random-64"]


def implementation():
    folder = Path(__file__).parent
    return {
        n: file_hash(folder / n)
        for n in [
            "trace_capture_qualification.py",
            "trace_lens_qualification.py",
            "trace_capture.py",
            "trace_readouts.py",
            "trace_vocabulary.py",
        ]
    }


def prepare(
    profile_path,
    assets_path,
    output_root,
    name,
    *,
    batch_size,
    storage_dtype,
    bootstrap_draws,
    seed,
):
    root = Path(output_root).resolve()
    safe_name(name)
    profile_path, assets_path = [_inside(root, p) for p in [profile_path, assets_path]]
    profile = load_profile(profile_path, root)
    verify_profile(profile_path, root)
    zero_path = _inside(root, profile["diagnostic_path"])
    verify_diagnostic(zero_path, root)
    assets = verify_assets(assets_path, root)
    runtime = json.loads((root / profile["run_path"] / "meta/runtime.json").read_text())
    if (
        type(batch_size) is not int
        or batch_size not in [1, 4]
        or storage_dtype not in ["float16", "float32"]
        or type(bootstrap_draws) is not int
        or bootstrap_draws < 1
        or type(seed) is not int
        or len(profile["examples"]) != 4
    ):
        raise ValueError("Explicit profiled batch, precision, draws and seed required.")
    if (
        any(
            assets[k] != profile[k]
            for k in ["model", "model_revision", "tokenizer_revision", "width"]
        )
        or assets["execution_kind"] != runtime["execution_kind"]
    ):
        raise ValueError("Capture profile/readout basis incompatibility.")
    parents = {}
    # Bind existing seals and their payload inventories; no prior file is changed.
    paths = [profile_path, assets_path, zero_path]
    for directory in [
        root / profile["run_path"],
        root / json.loads(zero_path.read_text())["run_path"],
        assets_path.parent,
    ]:
        marker = directory / "complete.json"
        paths.append(marker)
        paths.extend(directory / n for n in json.loads(marker.read_text())["files"])
        if (directory / "units").exists():
            for unit in (directory / "units").iterdir():
                if unit.is_dir():
                    paths.extend(
                        unit / n
                        for n in ["arrays.npz", "evidence.json", "complete.json"]
                    )
    for path in paths:
        parents[str(path.relative_to(root))] = file_hash(path)
    config = dict(
        schema_version=1,
        mode="capture-qualification-diagnostic",
        name=name,
        profile_path=str(profile_path.relative_to(root)),
        assets_path=str(assets_path.relative_to(root)),
        diagnostic_path=str(zero_path.relative_to(root)),
        model=profile["model"],
        model_revision=profile["model_revision"],
        execution_kind=runtime["execution_kind"],
        batch_size=batch_size,
        storage_dtype=storage_dtype,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        width=profile["width"],
        positions=[0, 1, 2, 3, 4],
        run_path=f"runs/pilot-5/{name}/capture-qualification",
        parents=parents,
        implementation=implementation(),
    )
    path = root / "plans" / name / "traces.capture-qualification.json"
    with run_lock(path.parent):
        _save_frozen(path, config)
        _save_frozen(
            root / config["run_path"] / "meta/run_manifest.json",
            dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)),
        )
    return path


def _load(path, root):
    path = _inside(root, path)
    config = json.loads(path.read_text())
    if (
        config["mode"] != "capture-qualification-diagnostic"
        or config["implementation"] != implementation()
    ):
        raise ValueError(
            "Capture qualification implementation changed; use a new name."
        )
    if json.loads(
        (root / config["run_path"] / "meta/run_manifest.json").read_text()
    ) != dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)):
        raise ValueError("Capture qualification manifest mismatch.")
    for relative, digest in config["parents"].items():
        if file_hash(_inside(root, relative)) != digest:
            raise ValueError("Capture qualification parent hash mismatch.")
    return config


def _capture_metrics(config, profile, root):
    zero = verify_diagnostic(root / config["diagnostic_path"], root)
    # Both exact-zero identities and all recorded base-integrity checks are mandatory.
    checks = zero["checks"]
    if set(checks) != {"base", "sft-zero", "grpo-zero"}:
        raise ValueError("Missing zero diagnostic conditions.")
    zero_ok = all(r.get("exact_zero") is True for r in checks.values())
    base_ok = all(r.get("base_unchanged") is True for r in checks.values())
    directory = root / profile["run_path"]
    batch_error = retention_error = 0.0
    worst_batch = worst_retention = None
    batch = config["batch_size"]
    one_base = _arrays(directory, "b1-base")
    batch_base = _arrays(directory, f"b{batch}-base")
    for condition in CONDITIONS:
        one = _arrays(directory, f"b1-{condition}")
        selected = _arrays(directory, f"b{batch}-{condition}")
        if set(one) != set(selected) or set(one) != set(one_base):
            raise ValueError("Profile activation inventory differs.")
        for key in one:
            a, b = one[key].astype(np.float64), selected[key].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite profile capture.")
            error = float(np.max(np.abs(a - b)))
            if error > batch_error:
                batch_error, worst_batch = error, dict(
                    condition=condition,
                    example_site=key,
                    coordinate=list(
                        map(int, np.unravel_index(np.abs(a - b).argmax(), a.shape))
                    ),
                )
            # Qualify retained deltas across all captured sites and positions, not only primary.
            delta = b - batch_base[key].astype(np.float64)
            stored = delta.astype(config["storage_dtype"]).astype(np.float64)
            if not np.isfinite(stored).all():
                raise ValueError("Retention conversion overflow.")
            error = float(np.max(np.abs(delta - stored)))
            if error > retention_error:
                retention_error, worst_retention = error, dict(
                    condition=condition,
                    example_site=key,
                    coordinate=list(
                        map(
                            int,
                            np.unravel_index(
                                np.abs(delta - stored).argmax(), delta.shape
                            ),
                        )
                    ),
                )
    return dict(
        zero_identity=zero_ok,
        base_integrity=base_ok,
        batch_max_absolute_difference=batch_error,
        retention_max_absolute_error=retention_error,
        worst_batch=worst_batch,
        worst_retention=worst_retention,
        numeric_lens=zero.get("numeric_lens"),
        scope="Four saved documents; batching/retention at all 29 sites and 128 positions. "
        "Readout/ranking/bootstrap at block14 pooled positions0–4 only.",
    )


def diagnose(path, output_root, *, max_new_units=None):
    root = Path(output_root).resolve()
    config = _load(path, root)
    directory = root / config["run_path"]
    if max_new_units is not None and (
        type(max_new_units) is not int or max_new_units < 1
    ):
        raise ValueError("Positive bounded unit count required.")
    with run_lock(directory):
        if (directory / "complete.json").exists():
            return verify(path, root)
        profile = load_profile(root / config["profile_path"], root)
        folder = (root / config["assets_path"]).parent
        assets = verify_assets(root / config["assets_path"], root)
        norm = np.load(folder / "norm.npy", mmap_mode="r", allow_pickle=False)
        head = np.load(folder / "head.npy", mmap_mode="r", allow_pickle=False)
        raw = json.loads((folder / "inventory.json").read_text())
        inventory = TokenInventory(
            {int(k): v for k, v in raw["decoded"].items()},
            special_ids=raw["excluded_ids"],
        )
        matrix = np.eye(
            config["width"], dtype=np.float64
        )  # Logit lens: no Jacobian transport.
        draws = np.random.default_rng(config["seed"]).integers(
            0, 4, size=(config["bootstrap_draws"], 4)
        )
        draw_path = directory / "bootstrap-indices.npy"
        if draw_path.exists():
            if not np.array_equal(np.load(draw_path, allow_pickle=False), draws):
                raise ValueError("Bootstrap schedule changed.")
        else:
            np.save(draw_path, draws, allow_pickle=False)
        rows = {n: _unit(directory, n, config) for n in UNITS}
        reused = sum(v is not None for v in rows.values())
        new = 0
        try:
            for name in UNITS:
                if rows[name] is not None:
                    continue
                if max_new_units is not None and new >= max_new_units:
                    _write_state(directory, "incomplete", reused + new, len(UNITS))
                    return dict(
                        qualified=False,
                        state="incomplete",
                        completed_units=reused + new,
                    )
                _write_state(directory, "running", reused + new, len(UNITS))
                print(
                    f"Capture qualification {reused+new}/{len(UNITS)} sealed; {name}; decoder forwards0.",
                    flush=True,
                )
                if name == "capture":
                    row = _capture_metrics(config, profile, root)
                elif name == "operator":
                    row = _operator(
                        config, profile, root, matrix, norm, head, assets["epsilon"]
                    )
                else:
                    row = _sensitivity(
                        config,
                        profile,
                        root,
                        config["batch_size"],
                        name,
                        config["storage_dtype"],
                        matrix,
                        norm,
                        head,
                        assets["epsilon"],
                        inventory,
                        draws,
                    )
                unit = directory / "units" / name
                _write_json(unit / "result.json", row)
                _write_json(
                    unit / "complete.json",
                    dict(
                        config_sha256=_hash(config),
                        result_sha256=file_hash(unit / "result.json"),
                    ),
                )
                rows[name] = row
                new += 1
            _load(path, root)
            result = dict(
                diagnostic_complete=True,
                qualified=False,
                state="awaiting-review",
                decoder_forwards=0,
                compute_device="cpu",
                new_units=new,
                reused_units=reused,
                capture=rows["capture"],
                operator=rows["operator"],
                measurements=[rows[n] for n in UNITS[2:]],
                learned_lists_revealed=False,
                note="Logit checks here; a separate scoped JLens receipt remains required for dual-lens production.",
            )
            _write_json(directory / "results/results.json", result)
            _save_frozen(
                directory / "review.draft.json",
                dict(
                    approved=False,
                    reviewer="",
                    notes="",
                    diagnostic_sha256=file_hash(directory / "results/results.json"),
                    bounded_scope_accepted=False,
                    limits={k: None for k in sorted(LIMITS)},
                ),
            )
            files = [
                "results/results.json",
                "review.draft.json",
                "bootstrap-indices.npy",
                "meta/run_manifest.json",
            ]
            _write_json(
                directory / "complete.json",
                dict(
                    config_sha256=_hash(config),
                    files={n: file_hash(directory / n) for n in files},
                ),
            )
            _write_state(directory, "awaiting-review", len(UNITS), len(UNITS))
            return result
        except BaseException as exc:
            _write_state(directory, "failed", reused + new, len(UNITS), str(exc))
            raise


def verify(path, output_root):
    root = Path(output_root).resolve()
    config = _load(path, root)
    directory = root / config["run_path"]
    marker = json.loads((directory / "complete.json").read_text())
    expected = {
        "results/results.json",
        "review.draft.json",
        "bootstrap-indices.npy",
        "meta/run_manifest.json",
    }
    if marker["config_sha256"] != _hash(config) or set(marker["files"]) != expected:
        raise ValueError("Capture qualification completion inventory mismatch.")
    for n, digest in marker["files"].items():
        if file_hash(directory / n) != digest:
            raise ValueError("Capture qualification completion hash mismatch.")
    for name in UNITS:
        if _unit(directory, name, config) is None:
            raise ValueError("Missing capture qualification unit.")
    return json.loads((directory / "results/results.json").read_text())


def approve(path, review_path, output_root):
    root = Path(output_root).resolve()
    config = _load(path, root)
    result = verify(path, root)
    directory = root / config["run_path"]
    review_path = _inside(root, review_path)
    review = json.loads(review_path.read_text())
    if (
        review.get("approved") is not True
        or not review.get("reviewer", "").strip()
        or not review.get("notes", "").strip()
        or review.get("bounded_scope_accepted") is not True
        or review.get("diagnostic_sha256")
        != file_hash(directory / "results/results.json")
    ):
        raise ValueError("Explicit named, hash-bound bounded-scope approval required.")
    limits = review.get("limits")
    if (
        not isinstance(limits, dict)
        or set(limits) != LIMITS
        or any(
            type(v) not in [int, float] or not math.isfinite(v) or v < 0
            for v in limits.values()
        )
        or any(
            limits[k] > 1
            for k in ["cohort_min_top20_overlap", "bootstrap_min_top20_overlap"]
        )
    ):
        raise ValueError("Explicit finite limits required; none are inferred.")
    c = result["capture"]
    measured_numeric = result["operator"]["maximum_score_error"]
    if config["execution_kind"] == "real-cuda":
        numeric = c["numeric_lens"]
        if not numeric or not math.isfinite(numeric["max_absolute_error"]):
            raise ValueError(
                "Real capture requires its actual GPU numeric-lens evidence."
            )
        measured_numeric = max(measured_numeric, numeric["max_absolute_error"])
    rows = result["measurements"]

    def stable(field, threshold):
        return all(r[field] is not None and r[field] >= threshold for r in rows)

    checks = dict(
        zero_identity=c["zero_identity"],
        base_integrity=c["base_integrity"],
        batching=c["batch_max_absolute_difference"]
        <= limits["batch_max_absolute_difference"],
        retention=c["retention_max_absolute_error"]
        <= limits["retention_max_absolute_error"],
        numeric_lens=measured_numeric <= limits["numeric_lens_max_score_error"]
        and result["operator"]["zero_vector_undefined"],
        ranking=all(
            not r["cohort"]["undefined"]
            and r["cohort"]["top20_overlap"] is not None
            and r["cohort"]["top20_overlap"] >= limits["cohort_min_top20_overlap"]
            and r["cohort"]["max_score_error"] <= limits["cohort_max_score_error"]
            for r in rows
        ),
        bootstrap=stable(
            "bootstrap_minimum_top20_overlap", limits["bootstrap_min_top20_overlap"]
        )
        and all(
            not r["bootstrap_undefined"]
            and r["bootstrap_maximum_score_error"]
            <= limits["bootstrap_max_score_error"]
            for r in rows
        ),
    )
    failures = [k for k, v in checks.items() if not v]
    evidence = {
        str(_inside(root, path).relative_to(root)): file_hash(_inside(root, path)),
        **config["parents"],
        str(review_path.relative_to(root)): file_hash(review_path),
    }
    for n in ["complete.json", "results/results.json", "bootstrap-indices.npy"]:
        evidence[str((directory / n).relative_to(root))] = file_hash(directory / n)
    for name in UNITS:
        for n in ["result.json", "complete.json"]:
            p = directory / "units" / name / n
            evidence[str(p.relative_to(root))] = file_hash(p)
    receipt = dict(
        protocol="pilot5-capture-qualification-v1",
        qualified=not failures,
        failures=failures,
        profile_sha256=file_hash(root / config["profile_path"]),
        model_revision=config["model_revision"],
        execution_kind=config["execution_kind"],
        batch_size=config["batch_size"],
        storage_dtype=config["storage_dtype"],
        reviewer=review["reviewer"],
        notes=review["notes"],
        limits=limits,
        checks={
            k: dict(
                passed=bool(v),
                evidence=str((directory / "results/results.json").relative_to(root)),
            )
            for k, v in checks.items()
        },
        evidence_files=evidence,
        scope=c["scope"],
        limitations=[
            "Four-document endpoint evidence; no intermediate-checkpoint or full-corpus qualification.",
            "Batching gate is absolute activation drift, not a cross-batch ranking-invariance claim.",
            "Readout sensitivity here covers logit lens at primary window; dual lens needs separate JLens receipt.",
        ],
    )
    output = directory / "reviews" / _hash(review) / "qualification.json"
    with run_lock(directory / "reviews"):
        _save_frozen(output, receipt)
        _save_frozen(
            output.with_name(output.name + ".seal.json"), dict(sha256=file_hash(output))
        )
    if failures:
        raise ValueError(
            f"Reviewed capture qualification gate failed: {failures}. Saved {output}; limits unchanged."
        )
    return output
