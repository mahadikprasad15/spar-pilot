"""Outcome-blind production proposal and explicit reviewed contract boundary.

This module consumes qualification receipts; it does not manufacture them from
profile completion or choose any numerical allowance on the reviewer's behalf.
"""

import json
import math
from pathlib import Path
from pilot_eval.trace_corpus import _inside, load_corpus
from pilot_eval.trace_profile import load_profile, verify_profile, endpoint
from pilot_eval.trace_capture_workflow import SITES
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _save_frozen, _hash
from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name

RETAINED = [f"block-{i}-output" for i in [14, 17, 18, 19, 20, 21]]
PRIMARY = dict(
    form="raw",
    site="block-14-output",
    step=64,
    positions=[0, 1, 2, 3, 4],
    pooling="positions-within-example-then-examples",
)


def implementation():
    folder = Path(__file__).parent
    return {
        n: file_hash(folder / n)
        for n in [
            "trace_production_plan.py",
            "trace_production.py",
            "trace_production_backend.py",
            "trace_capture.py",
        ]
    }


def storage_estimate(examples, width, conditions, batch_size, storage_dtype):
    """Exact array-byte upper estimate before lossless compression or metadata."""
    import math
    import numpy as np

    batches = math.ceil(examples / batch_size)
    sites = 29
    tokens = 128
    position = (sites * tokens * width * 8 + sites * tokens * 8) * conditions * batches
    pooled = (2 * sites * width * 8 + 2 * sites * 8) * examples * conditions
    retained = 6 * tokens * width * np.dtype(storage_dtype).itemsize * examples * conditions
    moments = (width + width * width) * 8 * batches
    aggregate = (
        sites * tokens * width * 8 + sites * tokens * 8 + 2 * sites * width * 8 + 2 * sites * 8
    ) * conditions + (width + width * width) * 8
    return dict(
        unit_arrays_before_compression=position + pooled + retained + moments,
        aggregate_arrays=aggregate,
        total_arrays_before_compression=position + pooled + retained + moments + aggregate,
        metadata_and_npy_headers_additional=True,
        compression="lossless NPZ; size not assumed",
    )


def propose_capture(profile_path, output_root, name):
    root = Path(output_root).resolve()
    safe_name(name)
    profile_path = _inside(root, profile_path)
    profile = load_profile(profile_path, root)
    verify_profile(profile_path, root)
    diagnostic = json.loads(_inside(root, profile["diagnostic_path"]).read_text())
    corpus_path = _inside(root, diagnostic["source_paths"][0])
    corpus, rows = load_corpus(corpus_path, root)
    paths = dict(
        sft=diagnostic["source_paths"][1],
        grpo=diagnostic["source_paths"][2],
        random=profile["random_path"],
    )
    parents = dict(profile["parents"])
    sources = {}
    conditions = {"base": None}
    parents[str(profile_path.relative_to(root))] = file_hash(profile_path)
    profile_complete = root / profile["run_path"] / "complete.json"
    parents[str(profile_complete.relative_to(root))] = file_hash(profile_complete)
    for arm, relative in paths.items():
        source, evidence = endpoint(relative, root, arm)
        parents.update(evidence)
        sources[arm] = source
        for step, checkpoint in source["source_evidence"]["checkpoints"].items():
            if int(step) not in [0, 8, 16, 32, 64]:
                raise ValueError("Unsupported checkpoint inventory.")
            conditions[f"{arm}-{step}"] = checkpoint
    proposal = dict(
        schema_version=1,
        name=name,
        mode="production-capture-proposal",
        profile_path=str(profile_path.relative_to(root)),
        corpus_path=str(corpus_path.relative_to(root)),
        model=profile["model"],
        model_revision=profile["model_revision"],
        tokenizer_revision=profile["tokenizer_revision"],
        width=profile["width"],
        expected_base_hash=profile["expected_base_hash"],
        parents=parents,
        corpus_items_sha256=corpus["items_sha256"],
        examples=rows,
        available_conditions=conditions,
        sites=SITES,
        content_tokens=128,
        retained_sites=RETAINED,
        primary=PRIMARY,
        profiled_batches=profile["batch_sizes"],
        production_ready=False,
        storage_candidates={
            f"{scope}-b{batch}-{dtype}": storage_estimate(
                len(rows), profile["width"], count, batch, dtype
            )
            for scope, count in [("endpoints", 4), ("trajectory", 10)]
            for batch in profile["batch_sizes"]
            for dtype in ["float16", "float32"]
        },
        pending=[
            "qualified numerical/retention/readout receipts",
            "scope",
            "budget",
            "predictions",
        ],
        review_template=dict(
            approved=False,
            reviewer="",
            notes="",
            proposal_sha256=None,
            conditions=None,
            batch_size=None,
            storage_dtype=None,
            scope=None,
            qualification_path=None,
            qualification_sha256=None,
            lens_mode=None,
            partial_mode_approved=False,
            lens_qualification_path=None,
            predictions=None,
            budget=None,
            primary=PRIMARY,
        ),
    )
    path = root / "plans" / name / "traces.production.proposal.json"
    with run_lock(path.parent):
        _save_frozen(path, proposal)
    return path


def _receipt(path, root):
    path = _inside(root, path)
    payload = json.loads(path.read_text())
    seal = path.with_name(path.name + ".seal.json")
    if json.loads(seal.read_text()) != {"sha256": file_hash(path)}:
        raise ValueError("Qualification receipt seal hash mismatch.")
    return path, payload


def freeze_capture(proposal_path, review_path, output_root):
    root = Path(output_root).resolve()
    proposal_path = _inside(root, proposal_path)
    review_path = _inside(root, review_path)
    p = json.loads(proposal_path.read_text())
    review = json.loads(review_path.read_text())
    if (
        review.get("approved") is not True
        or not review.get("reviewer", "").strip()
        or not review.get("notes", "").strip()
    ):
        raise ValueError("Explicit named production approval is required.")
    if review.get("proposal_sha256") != file_hash(proposal_path):
        raise ValueError("Production review proposal hash mismatch.")
    profile = load_profile(p["profile_path"], root)
    verify_profile(p["profile_path"], root)
    if any(file_hash(_inside(root, k)) != v for k, v in p["parents"].items()):
        raise ValueError("Production proposal parent hash mismatch.")
    selected = review["conditions"]
    batch = review["batch_size"]
    dtype = review["storage_dtype"]
    if (
        not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or selected[0:1] != ["base"]
        or not {"sft-64", "grpo-64", "random-64"} <= set(selected)
        or not set(selected) <= set(p["available_conditions"])
        or any(c.endswith("-0") for c in selected)
    ):
        raise ValueError(
            "Declare base, learned endpoints and existing random; zero checks are bounded separately."
        )
    if (
        type(batch) is not int
        or batch not in p["profiled_batches"]
        or dtype not in ["float16", "float32"]
    ):
        raise ValueError("Explicit profiled batch and qualified retained dtype required.")
    if review["primary"] != PRIMARY:
        raise ValueError("Primary identity differs from the approved design.")
    scope = review["scope"]
    expected = {"base", "sft-64", "grpo-64", "random-64"}
    if scope == "trajectory":
        expected |= {f"{a}-{s}" for a in ["sft", "grpo"] for s in [8, 16, 32]}
    if scope not in ["endpoints", "trajectory"] or set(selected) != expected:
        raise ValueError("Selected conditions differ from the explicit endpoint/trajectory scope.")
    qualification_path, q = _receipt(review["qualification_path"], root)
    checks = [
        "zero_identity",
        "base_integrity",
        "numeric_lens",
        "batching",
        "retention",
        "ranking",
        "bootstrap",
    ]
    if (
        file_hash(qualification_path) != review["qualification_sha256"]
        or q.get("protocol") != "pilot5-capture-qualification-v1"
        or q.get("qualified") is not True
        or q.get("profile_sha256") != file_hash(_inside(root, p["profile_path"]))
        or q.get("model_revision") != p["model_revision"]
        or q.get("batch_size") != batch
        or q.get("storage_dtype") != dtype
        or q.get("execution_kind") not in ["real-cuda", "synthetic-cpu"]
        or any(
            q.get("checks", {}).get(k, {}).get("passed") is not True
            or not q["checks"][k].get("evidence")
            for k in checks
        )
    ):
        raise ValueError(
            "Qualified capture receipt with all named evidence checks is required; profile completion is insufficient."
        )
    # The qualification owner must bind measured evidence and reviewed allowances.
    # Synthetic receipts are confined to injected CPU fixtures by the runtime gate.
    if q["execution_kind"] == "real-cuda":
        if not q.get("reviewer") or not q.get("limits") or not q.get("evidence_files"):
            raise ValueError(
                "Real qualification requires reviewer, explicit limits and measured evidence hashes."
            )
        for relative, digest in q["evidence_files"].items():
            if file_hash(_inside(root, relative)) != digest:
                raise ValueError("Qualification measured evidence hash mismatch.")
    parents = {
        **p["parents"],
        str(proposal_path.relative_to(root)): file_hash(proposal_path),
        str(review_path.relative_to(root)): file_hash(review_path),
        str(qualification_path.relative_to(root)): file_hash(qualification_path),
        str(
            qualification_path.with_name(qualification_path.name + ".seal.json").relative_to(root)
        ): file_hash(qualification_path.with_name(qualification_path.name + ".seal.json")),
        **q.get("evidence_files", {}),
    }
    if review["lens_mode"] == "logit-only":
        if review.get("partial_mode_approved") is not True:
            raise ValueError("Logit-only partial scope requires explicit scientific approval.")
    elif review["lens_mode"] == "logit-and-jacobian":
        lens_path, lens = _receipt(review["lens_qualification_path"], root)
        if lens.get("qualified") is not True or lens.get("model_revision") != p["model_revision"]:
            raise ValueError("Qualified compatible Jacobian receipt required.")
        from pilot_eval.trace_lenses import archive_inventory

        artifact = _inside(root, lens["lens_path"])
        inventory = archive_inventory(artifact)
        if (
            file_hash(artifact) != lens["lens_sha256"]
            or inventory["d_model"] != p["width"]
            or inventory["source_layers"] != lens["source_layers"]
            or 14 not in inventory["source_layers"]
            or lens.get("source_site") != "block-output"
            or lens.get("target") != "final-block-output-before-norm"
        ):
            raise ValueError("Qualified Jacobian source/shape/target contract mismatch.")
        parents[str(artifact.relative_to(root))] = file_hash(artifact)
        for relative, digest in lens.get("evidence_files", {}).items():
            if file_hash(_inside(root, relative)) != digest:
                raise ValueError("Lens evidence hash mismatch.")
        parents[str(lens_path.relative_to(root))] = file_hash(lens_path)
        parents[str(lens_path.with_name(lens_path.name + ".seal.json").relative_to(root))] = (
            file_hash(lens_path.with_name(lens_path.name + ".seal.json"))
        )
        parents.update(lens.get("evidence_files", {}))
    else:
        raise ValueError("Declare qualified dual-lens or explicitly approved logit-only scope.")
    budget = review["budget"]
    if (
        budget.get("approved") is not True
        or any(
            type(budget.get(k)) not in [int, float]
            or not math.isfinite(budget[k])
            or budget[k] <= 0
            for k in ["max_seconds", "max_durable_bytes"]
        )
        or not isinstance(review["predictions"], dict)
        or not all(review["predictions"].get(k) for k in ["sft", "grpo"])
    ):
        raise ValueError("Reviewed positive budget and actual SFT/GRPO predictions required.")
    for arm in ["sft", "grpo"]:
        prediction = review["predictions"][arm]
        if (
            not isinstance(prediction, dict)
            or not isinstance(prediction.get("statement"), str)
            or not prediction["statement"].strip()
            or type(prediction.get("confidence")) not in [float, int]
            or not math.isfinite(prediction["confidence"])
            or not 0 <= prediction["confidence"] <= 1
        ):
            raise ValueError(
                "Each actual prediction requires a statement and confidence between 0 and 1."
            )
    adapters = {}
    for condition in selected[1:]:
        cp = p["available_conditions"][condition]
        folder = _inside(root, cp["path"])
        seal = folder / "complete.json"
        if file_hash(seal) != cp["complete_sha256"]:
            raise ValueError("Selected checkpoint seal hash mismatch.")
        saved = json.loads(seal.read_text())
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            if file_hash(folder / name) != saved["files"][name]:
                raise ValueError("Selected checkpoint payload hash mismatch.")
            parents[str((folder / name).relative_to(root))] = file_hash(folder / name)
        parents[str(seal.relative_to(root))] = file_hash(seal)
        adapters[condition] = cp["path"]
    rows = p["examples"]
    batches = [list(range(i, min(i + batch, len(rows)))) for i in range(0, len(rows), batch)]
    config = {
        k: p[k]
        for k in [
            "name",
            "model",
            "model_revision",
            "tokenizer_revision",
            "width",
            "expected_base_hash",
            "sites",
            "retained_sites",
            "primary",
            "examples",
        ]
    }
    qualified_runtime = json.loads((root / profile["run_path"] / "meta/runtime.json").read_text())
    config.update(
        qualified_runtime={
            k: v for k, v in qualified_runtime.items() if k not in ["git_commit", "profile_batches"]
        },
        schema_version=1,
        mode="frozen-production-capture",
        execution_kind=q["execution_kind"],
        conditions=selected,
        adapters=adapters,
        seed=42,
        batches=batches,
        batch_size=batch,
        storage_dtype=dtype,
        parents=parents,
        implementation=implementation(),
        run_path=f'runs/pilot-5/{p["name"]}/production',
        review=review,
        unit_policy="one compatible batch, shared base, all frozen conditions; only whole sealed batches accepted",
        corruption_policy="reject changed sealed units; retry incomplete unpublished units only",
        capture_precision="float32",
        accumulation="float64",
        content_tokens=128,
        storage_estimate=storage_estimate(len(rows), p["width"], len(selected), batch, dtype),
    )
    path = root / "plans" / p["name"] / "traces.production.json"
    with run_lock(path.parent):
        _save_frozen(path, config)
        _save_frozen(
            root / config["run_path"] / "meta/run_manifest.json",
            dict(config_path=str(path.relative_to(root)), config_sha256=_hash(config)),
        )
    return path


def load_execution(path, output_root):
    root = Path(output_root).resolve()
    path = _inside(root, path)
    config = json.loads(path.read_text())
    if (
        config.get("mode") != "frozen-production-capture"
        or config["implementation"] != implementation()
    ):
        raise ValueError(
            "Production implementation changed; preserve old execution and review a new one."
        )
    if json.loads((root / config["run_path"] / "meta/run_manifest.json").read_text()) != dict(
        config_path=str(path.relative_to(root)), config_sha256=_hash(config)
    ):
        raise ValueError("Production plan manifest mismatch.")
    for relative, digest in config["parents"].items():
        if file_hash(_inside(root, relative)) != digest:
            raise ValueError("Production parent hash mismatch.")
    return config
