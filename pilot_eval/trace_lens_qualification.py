"""Bounded real-artifact JLens diagnostics from saved profile activations.

No decoder construction, inference or learned-token publication. Preparation,
measurement and explicit acceptance are separate saved operations. Numerical
limits are supplied by the reviewer, never inferred from measured errors.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_profile import load_profile, verify_profile, _arrays
from pilot_eval.trace_lenses import verify_lens_inspection, read_matrix, transport
from pilot_eval.trace_readout_assets import verify_assets
from pilot_eval.trace_readouts import _project
from pilot_eval.trace_vocabulary import TokenInventory
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name

CONDITIONS = ["sft-64", "grpo-64", "random-64"]
BATCHES = [1, 4]
DTYPES = ["float16", "float32"]
SOURCE_BLOCK = 14


def _implementation():
    folder = Path(__file__).parent
    return {
        name: file_hash(folder / name)
        for name in [
            "trace_lens_qualification.py",
            "trace_lenses.py",
            "trace_readouts.py",
            "trace_readout_assets.py",
            "trace_vocabulary.py",
        ]
    }


def prepare(
    profile_path,
    inspection_path,
    assets_path,
    output_root,
    name,
    *,
    bootstrap_draws,
    seed,
):
    """Reuse the approved four documents; explicitly record bootstrap settings."""
    root = Path(output_root).resolve()
    safe_name(name)
    if type(bootstrap_draws) is not int or bootstrap_draws < 1 or type(seed) is not int:
        raise ValueError(
            "Explicit positive bootstrap draw count and integer seed required."
        )
    profile_path, inspection_path, assets_path = [
        _inside(root, p) for p in (profile_path, inspection_path, assets_path)
    ]
    profile = load_profile(profile_path, root)
    verify_profile(profile_path, root)
    inspection = verify_lens_inspection(inspection_path, root)
    assets = verify_assets(assets_path, root)
    original = json.loads((inspection_path.parent.parent / "config.json").read_text())
    if (
        inspection["status"] == "incompatible"
        or SOURCE_BLOCK not in inspection["source_layers"]
        or inspection["d_model"] != profile["width"]
        or len(profile["examples"]) != 4
        or any(
            assets[k] != profile[k]
            for k in ["model", "model_revision", "tokenizer_revision", "width"]
        )
        or original["expected_model"] != profile["model"]
        or original["expected_revision"] != profile["model_revision"]
    ):
        raise ValueError("Inspection/profile/readout basis incompatibility.")
    runtime = json.loads((root / profile["run_path"] / "meta/runtime.json").read_text())
    if assets["execution_kind"] != runtime["execution_kind"]:
        raise ValueError("Synthetic/real source mismatch.")
    paths = [profile_path, inspection_path, assets_path, root / original["lens_path"]]
    for directory in [
        root / profile["run_path"],
        inspection_path.parent.parent,
        assets_path.parent,
    ]:
        paths.append(directory / "complete.json")
    for unit in (root / profile["run_path"] / "units").iterdir():
        if unit.is_dir():
            paths.extend(
                unit / n
                for n in ["complete.json", "evidence.json", "arrays.npz"]
                if (unit / n).is_file()
            )
    parents = {str(p.relative_to(root)): file_hash(p) for p in paths}
    config = dict(
        schema_version=1,
        mode="bounded-jacobian-qualification",
        compute_device="cpu",
        name=name,
        run_path=f"runs/pilot-5/{name}/lens-qualification",
        profile_path=str(profile_path.relative_to(root)),
        inspection_path=str(inspection_path.relative_to(root)),
        assets_path=str(assets_path.relative_to(root)),
        lens_path=original["lens_path"],
        lens_sha256=inspection["lens_sha256"],
        source_layers=inspection["source_layers"],
        qualified_source_layers=[SOURCE_BLOCK],
        model=profile["model"],
        model_revision=profile["model_revision"],
        width=profile["width"],
        execution_kind=runtime["execution_kind"],
        source_block=SOURCE_BLOCK,
        positions=[0, 1, 2, 3, 4],
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        parents=parents,
        implementation=_implementation(),
        decoder_forwards=0,
        qualification_scope="raw block14 pooled positions0–4; four profile documents only",
        provenance_supplement=dict(
            status="published-procedure-supported; original fitting-run manifest absent",
            source_commit="f835828b235c8f1c027b4205a80220cabf26f145",
            fitting_code="https://github.com/SharathSPhD/prabodha/blob/f835828b235c8f1c027b4205a80220cabf26f145/scripts/experiments/fit_lenses.py",
            reader_code="https://github.com/SharathSPhD/prabodha/blob/f835828b235c8f1c027b4205a80220cabf26f145/vendor/jacobian-lens/jlens/hf.py",
            model_history="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commits/main",
            inference="Published fit targets final block; source0–26 inventory agrees. Canonical-model history supports expected revision; exact original run hash is unavailable.",
            limitations=[
                "No original run manifest binding binary to fitting invocation.",
                "Published fit uses BF16; supplied matrices are FP16.",
                "Published fitting default excludes first16 positions; primary readout uses0–4.",
            ],
        ),
        note="Diagnosis may use an unqualified candidate; only explicit review can issue a scoped receipt.",
    )
    path = root / "plans" / name / "traces.lens-qualification.json"
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
        config["mode"] != "bounded-jacobian-qualification"
        or config["implementation"] != _implementation()
    ):
        raise ValueError(
            "Qualification implementation changed; preserve old evidence and use a new name."
        )
    directory = _inside(root, config["run_path"])
    if json.loads((directory / "meta/run_manifest.json").read_text()) != dict(
        config_path=str(path.relative_to(root)), config_sha256=_hash(config)
    ):
        raise ValueError("Qualification plan identity mismatch.")
    for relative, digest in config["parents"].items():
        if file_hash(_inside(root, relative)) != digest:
            raise ValueError("Qualification parent hash mismatch.")
    verify_profile(root / config["profile_path"], root)
    verify_lens_inspection(root / config["inspection_path"], root)
    verify_assets(root / config["assets_path"], root)
    return config


def _units():
    return ["operator"] + [
        f"b{b}-{c}-{dtype}" for b in BATCHES for c in CONDITIONS for dtype in DTYPES
    ]


def _unit(directory, name, config):
    folder = directory / "units" / name
    if not (folder / "complete.json").exists():
        return None
    seal = json.loads((folder / "complete.json").read_text())
    if seal != dict(
        config_sha256=_hash(config), result_sha256=file_hash(folder / "result.json")
    ):
        raise ValueError("Qualification unit hash mismatch.")
    return json.loads((folder / "result.json").read_text())


def _oracle(vector, matrix, norm, head, epsilon):
    # Distinct FP64 reduction path, independent of transport() and _project().
    transported = np.einsum("ij,j->i", matrix, vector, dtype=np.float64, optimize=False)
    scale = math.sqrt(
        math.fsum(float(x) * float(x) for x in transported) / len(vector) + epsilon
    )
    normalized = transported * norm / scale
    return np.array(
        [
            math.fsum(float(a) * float(b) for a, b in zip(row, normalized))
            for row in head
        ]
    )


def _operator(config, profile, root, matrix, norm, head, epsilon):
    base = _arrays(root / profile["run_path"], "b1-base")
    site = f"block-{SOURCE_BLOCK}-output"
    vectors = [2.0 + 3.0 * np.arange(config["width"], dtype=np.float64)]
    vectors.extend(
        base[f"{i}.{site}"][config["positions"]].astype(np.float64).mean(axis=0)
        for i in range(4)
    )
    ids = list(range(min(32, len(head))))
    reference_head = np.asarray(head[ids], dtype=np.float64)
    max_error = 0.0
    worst = None
    for i, v in enumerate(vectors):
        actual = _project(transport(v, matrix), norm, head, epsilon, ids)
        expected = _oracle(v, matrix, norm, reference_head, epsilon)
        difference = np.abs(actual - expected)
        k = int(difference.argmax())
        if worst is None or float(difference[k]) > max_error:
            max_error = float(difference[k])
            worst = dict(
                probe=i,
                output_row=ids[k],
                actual=float(actual[k]),
                reference=float(expected[k]),
            )
    # A zero difference must not be turned into an arbitrary tied token list.
    from pilot_eval.trace_capture import logit_lens

    zero_undefined = (
        logit_lens(
            np.zeros(config["width"]),
            norm_weight=norm,
            unembedding=reference_head,
            epsilon=epsilon,
        )
        is None
    )
    return dict(
        maximum_score_error=max_error,
        worst=worst,
        zero_vector_undefined=zero_undefined,
        probes=len(vectors),
        output_rows=len(ids),
        probe_transported_prefix=transport(vectors[0], matrix)[:8].tolist(),
        reference="FP64 einsum transport plus math.fsum RMS/head; actual production transport/_project path",
        qualified=False,
    )


def _scores(vectors, matrix, norm, head, epsilon, ids):
    moved = vectors @ matrix.T
    norms = np.sqrt(np.mean(moved * moved, axis=1) + epsilon)
    normalized = moved * norm / norms[:, None]
    result = np.empty((len(vectors), len(ids)), dtype=np.float64)
    for start in range(0, len(ids), 2048):
        index = ids[start : start + 2048]
        result[:, start : start + len(index)] = (
            normalized @ np.asarray(head[index], dtype=np.float64).T
        )
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite qualification scores.")
    return result


def _rank_set(scores, ids, inventory):
    chosen = set()
    for index in np.lexsort((ids, -scores))[:1000]:
        word = inventory.normalize(int(ids[index]))
        if word is not None:
            chosen.add(word)
        if len(chosen) == 20:
            break
    return chosen


def _sensitivity(
    config,
    profile,
    root,
    batch,
    condition,
    dtype,
    matrix,
    norm,
    head,
    epsilon,
    inventory,
    draws,
):
    directory = root / profile["run_path"]
    base = _arrays(directory, f"b{batch}-base")
    adapted = _arrays(directory, f"b{batch}-{condition}")
    site = f"block-{SOURCE_BLOCK}-output"
    differences = np.stack(
        [
            adapted[f"{i}.{site}"][config["positions"]].astype(np.float64)
            - base[f"{i}.{site}"][config["positions"]].astype(np.float64)
            for i in range(4)
        ]
    )
    stored = differences.astype(dtype).astype(np.float64)
    if not np.isfinite(stored).all():
        raise ValueError("Retained precision conversion overflow.")
    before = differences.mean(axis=1)
    after = stored.mean(axis=1)
    # Repeated bootstrap multisets have identical means: at most35 unique weight
    # vectors for four documents, even with1000 draws. Compute each readout once.
    weights = np.vstack(
        [
            np.ones(4, dtype=np.int64),
            np.stack([np.bincount(row, minlength=4) for row in draws]),
        ]
    )
    unique, inverse = np.unique(weights, axis=0, return_inverse=True)
    left = unique @ before / 4.0
    right = unique @ after / 4.0
    ids = np.array(sorted(inventory.decoded), dtype=np.int64)
    lscore = _scores(left, matrix, norm, head, epsilon, ids)
    rscore = _scores(right, matrix, norm, head, epsilon, ids)
    comparisons = []
    for a, b, x, y in zip(left, right, lscore, rscore):
        if not np.any(a) or not np.any(b):
            comparisons.append(
                dict(
                    undefined=True,
                    reason="zero-vector",
                    top20_overlap=None,
                    max_score_error=float(np.max(np.abs(x - y))),
                    vector_error_l2=float(np.linalg.norm(a - b)),
                )
            )
            continue
        # Both signs are inspected numerically without returning IDs or strings.
        overlaps = []
        counts = []
        for sign in [1, -1]:
            ls = _rank_set(sign * x, ids, inventory)
            rs = _rank_set(sign * y, ids, inventory)
            overlaps.append(len(ls & rs) / len(ls) if ls else None)
            counts.extend([len(ls), len(rs)])
        comparisons.append(
            dict(
                undefined=any(v is None for v in overlaps),
                top20_overlap=(
                    min(v for v in overlaps if v is not None)
                    if any(v is not None for v in overlaps)
                    else None
                ),
                minimum_valid_strings=min(counts),
                max_score_error=float(np.max(np.abs(x - y))),
                vector_error_l2=float(np.linalg.norm(a - b)),
            )
        )
    cohort = comparisons[inverse[0]]
    bootstrap = [comparisons[i] for i in inverse[1:]]
    valid = [r["top20_overlap"] for r in bootstrap if r["top20_overlap"] is not None]
    return dict(
        batch_size=batch,
        condition=condition,
        storage_dtype=dtype,
        examples=4,
        positions=config["positions"],
        source_block=SOURCE_BLOCK,
        conversion_max_absolute_error=float(np.max(np.abs(differences - stored))),
        cohort=cohort,
        bootstrap_minimum_top20_overlap=min(valid) if valid else None,
        bootstrap_maximum_score_error=max(r["max_score_error"] for r in bootstrap),
        bootstrap_undefined=sum(r["undefined"] for r in bootstrap),
        bootstrap_draws=bootstrap,
        unique_multisets=len(unique),
        learned_lists_revealed=False,
        qualified=False,
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
            return _verify_loaded(config, root)
        names = _units()
        saved = {n: _unit(directory, n, config) for n in names}
        reused = sum(v is not None for v in saved.values())
        new = 0
        profile = load_profile(root / config["profile_path"], root)
        assets = json.loads((root / config["assets_path"]).read_text())
        folder = (root / config["assets_path"]).parent
        norm = np.load(folder / "norm.npy", mmap_mode="r", allow_pickle=False)
        head = np.load(folder / "head.npy", mmap_mode="r", allow_pickle=False)
        raw = json.loads((folder / "inventory.json").read_text())
        inventory = TokenInventory(
            {int(k): v for k, v in raw["decoded"].items()},
            special_ids=raw["excluded_ids"],
        )
        matrix = read_matrix(
            root / config["lens_path"],
            SOURCE_BLOCK,
            expected_sha256=config["lens_sha256"],
        )
        draws = np.random.default_rng(config["seed"]).integers(
            0, 4, size=(config["bootstrap_draws"], 4)
        )
        directory.mkdir(parents=True, exist_ok=True)
        draw_path = directory / "bootstrap-indices.npy"
        if draw_path.exists():
            if not np.array_equal(np.load(draw_path, allow_pickle=False), draws):
                raise ValueError("Saved bootstrap schedule changed.")
        else:
            np.save(draw_path, draws, allow_pickle=False)
        try:
            for name in names:
                if saved[name] is not None:
                    continue
                if max_new_units is not None and new >= max_new_units:
                    _write_state(directory, "incomplete", reused + new, len(names))
                    return dict(
                        state="incomplete",
                        completed_units=reused + new,
                        total_units=len(names),
                        qualified=False,
                    )
                _write_state(directory, "running", reused + new, len(names))
                print(
                    f"JLens diagnostic {reused+new}/{len(names)} sealed; measuring {name}; decoder forwards0.",
                    flush=True,
                )
                if name == "operator":
                    result = _operator(
                        config, profile, root, matrix, norm, head, assets["epsilon"]
                    )
                else:
                    batch = int(name[1])
                    dtype = name.rsplit("-", 1)[1]
                    condition = name[3 : -(len(dtype) + 1)]
                    result = _sensitivity(
                        config,
                        profile,
                        root,
                        batch,
                        condition,
                        dtype,
                        matrix,
                        norm,
                        head,
                        assets["epsilon"],
                        inventory,
                        draws,
                    )
                folder = directory / "units" / name
                _write_json(folder / "result.json", result)
                _write_json(
                    folder / "complete.json",
                    dict(
                        config_sha256=_hash(config),
                        result_sha256=file_hash(folder / "result.json"),
                    ),
                )
                saved[name] = result
                new += 1
            result = dict(
                diagnostic_complete=True,
                compute_device="cpu",
                qualified=False,
                state="awaiting-review",
                decoder_forwards=0,
                source_block=SOURCE_BLOCK,
                positions=config["positions"],
                bootstrap_draws=config["bootstrap_draws"],
                seed=config["seed"],
                operator=saved["operator"],
                measurements=[saved[n] for n in names[1:]],
                new_units=new,
                reused_units=reused,
                learned_lists_revealed=False,
                provenance_supplement=config["provenance_supplement"],
                scope="Four saved documents; primary block14/window0–4 only. No lens-fitting-quality or causal guarantee.",
            )
            _write_json(directory / "results/results.json", result)
            template = dict(
                approved=False,
                reviewer="",
                notes="",
                diagnostic_sha256=file_hash(directory / "results/results.json"),
                model_identity_accepted=False,
                target_convention_accepted=False,
                provenance_limitations_accepted=False,
                batch_size=None,
                storage_dtype=None,
                limits=dict(
                    operator_max_score_error=None,
                    conversion_max_absolute_error=None,
                    cohort_max_score_error=None,
                    cohort_min_top20_overlap=None,
                    bootstrap_max_score_error=None,
                    bootstrap_min_top20_overlap=None,
                ),
            )
            _save_frozen(directory / "review.draft.json", template)
            files = [
                "meta/run_manifest.json",
                "results/results.json",
                "bootstrap-indices.npy",
                "review.draft.json",
            ]
            files.extend(
                f"units/{n}/{f}"
                for n in names
                for f in ["result.json", "complete.json"]
            )
            _write_json(
                directory / "complete.json",
                dict(
                    config_sha256=_hash(config),
                    files={n: file_hash(directory / n) for n in files},
                ),
            )
            _write_state(directory, "awaiting-review", len(names), len(names))
            return result
        except BaseException as exc:
            _write_state(directory, "failed", reused + new, len(names), str(exc))
            raise


def verify(path, output_root):
    root = Path(output_root).resolve()
    config = _load(path, root)
    return _verify_loaded(config, root)


def _verify_loaded(config, root):
    directory = root / config["run_path"]
    marker = json.loads((directory / "complete.json").read_text())
    required = {
        "meta/run_manifest.json",
        "results/results.json",
        "bootstrap-indices.npy",
        "review.draft.json",
    }
    required |= {
        f"units/{n}/{f}" for n in _units() for f in ["result.json", "complete.json"]
    }
    if marker["config_sha256"] != _hash(config) or set(marker["files"]) != required:
        raise ValueError("Qualification completion inventory mismatch.")
    for relative, digest in marker["files"].items():
        if file_hash(_inside(directory, relative)) != digest:
            raise ValueError("Qualification payload hash mismatch.")
    return json.loads((directory / "results/results.json").read_text())


def approve(path, review_path, output_root):
    root = Path(output_root).resolve()
    config = _load(path, root)
    result = _verify_loaded(config, root)
    directory = root / config["run_path"]
    review_path = _inside(root, review_path)
    review = json.loads(review_path.read_text())
    if (
        review.get("approved") is not True
        or not review.get("reviewer", "").strip()
        or not review.get("notes", "").strip()
        or review.get("diagnostic_sha256")
        != file_hash(directory / "results/results.json")
        or any(
            review.get(k) is not True
            for k in [
                "model_identity_accepted",
                "target_convention_accepted",
                "provenance_limitations_accepted",
            ]
        )
    ):
        raise ValueError(
            "Named, hash-bound provenance and diagnostic approval required."
        )
    keys = {
        "operator_max_score_error",
        "conversion_max_absolute_error",
        "cohort_max_score_error",
        "cohort_min_top20_overlap",
        "bootstrap_max_score_error",
        "bootstrap_min_top20_overlap",
    }
    limits = review.get("limits")
    if (
        not isinstance(limits, dict)
        or set(limits) != keys
        or any(
            type(v) not in [int, float] or not math.isfinite(v) or v < 0
            for v in limits.values()
        )
        or any(
            limits[k] > 1
            for k in ["cohort_min_top20_overlap", "bootstrap_min_top20_overlap"]
        )
    ):
        raise ValueError(
            "Explicit finite numerical limits required; none are inferred."
        )
    batch = review.get("batch_size")
    dtype = review.get("storage_dtype")
    if type(batch) is not int or batch not in BATCHES or dtype not in DTYPES:
        raise ValueError("Select diagnosed batch and storage precision.")
    failures = []
    if (
        not result["operator"]["zero_vector_undefined"]
        or result["operator"]["maximum_score_error"]
        > limits["operator_max_score_error"]
    ):
        failures.append("operator")
    selected = [
        r
        for r in result["measurements"]
        if r["batch_size"] == batch and r["storage_dtype"] == dtype
    ]
    for row in selected:
        if (
            row["conversion_max_absolute_error"]
            > limits["conversion_max_absolute_error"]
            or row["cohort"]["max_score_error"] > limits["cohort_max_score_error"]
            or row["cohort"]["top20_overlap"] is None
            or row["cohort"]["top20_overlap"] < limits["cohort_min_top20_overlap"]
            or row["bootstrap_maximum_score_error"]
            > limits["bootstrap_max_score_error"]
            or row["bootstrap_minimum_top20_overlap"] is None
            or row["bootstrap_minimum_top20_overlap"]
            < limits["bootstrap_min_top20_overlap"]
            or row["bootstrap_undefined"]
        ):
            failures.append(row["condition"])
    evidence = {
        str((directory / n).relative_to(root)): file_hash(directory / n)
        for n in ["complete.json", "results/results.json", "bootstrap-indices.npy"]
    }
    evidence.update(config["parents"])
    evidence[str(review_path.relative_to(root))] = file_hash(review_path)
    receipt = dict(
        protocol="pilot5-jacobian-qualification-v1",
        compute_device="cpu",
        qualified=not failures,
        failures=failures,
        reviewer=review["reviewer"],
        limits=limits,
        execution_kind=config["execution_kind"],
        model=config["model"],
        model_revision=config["model_revision"],
        lens_path=config["lens_path"],
        lens_sha256=config["lens_sha256"],
        source_layers=config["source_layers"],
        qualified_source_layers=[SOURCE_BLOCK],
        source_site="block-output",
        target="final-block-output-before-norm",
        positions=config["positions"],
        batch_size=batch,
        storage_dtype=dtype,
        evidence_files=evidence,
        provenance_supplement=config["provenance_supplement"],
        note="Qualified only for the declared four-document primary-window checks; not a full capture qualification receipt.",
    )
    # Reviewed decisions receive separate identities; never overwrite a failed or
    # superseded receipt to turn a gate green under the same name.
    output = directory / "reviews" / _hash(review) / "qualification.json"
    with run_lock(directory / "reviews"):
        _save_frozen(output, receipt)
        _save_frozen(
            output.with_name(output.name + ".seal.json"), dict(sha256=file_hash(output))
        )
    if failures:
        raise ValueError(
            f"Reviewed qualification gate failed: {failures}. Saved {output}; original limits unchanged."
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ["prepare", "diagnose", "verify", "approve"]:
        sub = commands.add_parser(operation)
        sub.add_argument("--output-root", type=Path, required=True)
        if operation == "prepare":
            for field in ["profile", "inspection", "assets"]:
                sub.add_argument("--" + field, type=Path, required=True)
            sub.add_argument("--name", required=True)
            sub.add_argument("--bootstrap-draws", type=int, required=True)
            sub.add_argument("--seed", type=int, required=True)
        else:
            sub.add_argument("--config", type=Path, required=True)
        if operation == "diagnose":
            sub.add_argument("--max-new-units", type=int)
        if operation == "approve":
            sub.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        result = prepare(
            args.profile,
            args.inspection,
            args.assets,
            args.output_root,
            args.name,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed,
        )
    elif args.operation == "diagnose":
        result = diagnose(
            args.config, args.output_root, max_new_units=args.max_new_units
        )
    elif args.operation == "verify":
        result = verify(args.config, args.output_root)
    else:
        result = approve(args.config, args.review, args.output_root)
    print_summary(result)


def print_summary(result):
    if isinstance(result, Path):
        print(result)
    else:
        # Keep notebook output compact; full measurements and draws stay durable.
        brief = {
            k: v
            for k, v in result.items()
            if k not in ["measurements", "provenance_supplement"]
        }
        if "measurements" in result:
            keys = [
                "batch_size",
                "condition",
                "storage_dtype",
                "conversion_max_absolute_error",
                "cohort",
                "bootstrap_minimum_top20_overlap",
                "bootstrap_maximum_score_error",
                "bootstrap_undefined",
                "unique_multisets",
            ]
            brief["measurement_summary"] = [
                {key: row[key] for key in keys} for row in result["measurements"]
            ]
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
