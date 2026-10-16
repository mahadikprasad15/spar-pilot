"""Saved-vector readouts with explicit prereveal and lens qualification gates."""

import json
from pathlib import Path
import numpy as np
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_production_plan import load_execution, PRIMARY, _receipt
from pilot_eval.trace_production import verify_capture, load_aggregate
from pilot_eval.trace_readout_assets import verify_assets
from pilot_eval.trace_vocabulary import load_references, TokenInventory
from pilot_eval.trace_lenses import read_matrix, transport
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name


def _implementation():
    folder = Path(__file__).parent
    return {
        n: file_hash(folder / n)
        for n in [
            "trace_readouts.py",
            "trace_readout_assets.py",
            "trace_lenses.py",
            "trace_vocabulary.py",
        ]
    }


def propose_readouts(capture_path, references_path, output_root, name):
    """Save a draft with fixed primary requests; no reveal or approval is inferred."""
    root = Path(output_root).resolve()
    safe_name(name)
    capture_path = _inside(root, capture_path)
    references_path = _inside(root, references_path)
    load_execution(capture_path, root)
    load_references(references_path, require_review=True)
    proposal = dict(
        approved=False,
        reviewer="",
        notes="",
        capture_sha256=file_hash(capture_path),
        references_sha256=file_hash(references_path),
        primary_family=None,
        requests=[
            dict(
                name=arm + "-primary",
                condition=arm + "-64",
                role="delta",
                site=PRIMARY["site"],
                positions=PRIMARY["positions"],
            )
            for arm in ["sft", "grpo"]
        ],
        lens_names=["logit", "jacobian"],
        partial_mode_approved=False,
    )
    path = root / "plans" / name / "traces.readout-review.draft.json"
    with run_lock(path.parent):
        _save_frozen(path, proposal)
    return path


def prepare_readouts(capture_path, assets_path, references_path, review_path, output_root, name):
    root = Path(output_root).resolve()
    safe_name(name)
    capture_path = _inside(root, capture_path)
    assets_path = _inside(root, assets_path)
    references_path = _inside(root, references_path)
    review_path = _inside(root, review_path)
    # Prereveal checks precede activation-array/head access or token decoding.
    references = load_references(references_path, require_review=True)
    capture = load_execution(capture_path, root)
    verify_capture(capture_path, root)
    assets = verify_assets(assets_path, root)
    review = json.loads(review_path.read_text())
    if (
        review.get("approved") is not True
        or not review.get("reviewer", "").strip()
        or not review.get("notes", "").strip()
        or not review.get("primary_family")
        or review.get("capture_sha256") != file_hash(capture_path)
        or review.get("references_sha256") != file_hash(references_path)
    ):
        raise ValueError("Explicit bound prereveal review and primary family required.")
    if any(
        assets[k] != capture[k]
        for k in ["model", "model_revision", "tokenizer_revision", "width", "execution_kind"]
    ):
        raise ValueError("Readout assets belong to a different pinned model/tokenizer.")
    lenses = review["lens_names"]
    if not lenses or len(lenses) != len(set(lenses)) or not set(lenses) <= {"logit", "jacobian"}:
        raise ValueError("Unsupported readout lens inventory.")
    if lenses != ["logit", "jacobian"]:
        if (
            capture["review"]["lens_mode"] != "logit-only"
            or review.get("partial_mode_approved") is not True
            or lenses != ["logit"]
        ):
            raise ValueError("Qualified dual-lens or explicitly approved partial mode required.")
    jacobian = None
    if "jacobian" in lenses:
        lens_path, jacobian = _receipt(capture["review"]["lens_qualification_path"], root)
        if (
            jacobian.get("qualified") is not True
            or jacobian.get("model_revision") != capture["model_revision"]
        ):
            raise ValueError("Jacobian lens is unqualified or incompatible.")
        if (
            jacobian.get("target") != "final-block-output-before-norm"
            or jacobian.get("source_site") != "block-output"
        ):
            raise ValueError("Jacobian source/target/norm convention unresolved.")
        lens_file = _inside(root, jacobian["lens_path"])
        if file_hash(lens_file) != jacobian["lens_sha256"]:
            raise ValueError("Jacobian lens hash mismatch.")
    requests = review["requests"]
    names = []
    primary_arms = set()
    for request in requests:
        safe_name(request["name"])
        names.append(request["name"])
        if (
            request["condition"] not in capture["conditions"]
            or request["site"] not in capture["sites"]
        ):
            raise ValueError("Readout requests unavailable capture condition/site.")
        pos = request["positions"]
        if (
            not pos
            or len(pos) != len(set(pos))
            or any(type(i) is not int or not 0 <= i < 128 for i in pos)
        ):
            raise ValueError("Invalid readout content positions.")
        if request["role"] not in ["delta", "base", "adapted", "random-direction"]:
            raise ValueError("Unknown readout vector role.")
        if request["role"] == "random-direction" and type(request.get("seed")) is not int:
            raise ValueError("Random direction needs an explicit local seed.")
        if (
            request["role"] == "delta"
            and request["site"] == PRIMARY["site"]
            and pos == PRIMARY["positions"]
        ):
            primary_arms.add(request["condition"])
    if len(names) != len(set(names)) or not {"sft-64", "grpo-64"} <= primary_arms:
        raise ValueError("Declare both unique primary arm requests.")
    parents = {
        str(p.relative_to(root)): file_hash(p)
        for p in [capture_path, assets_path, references_path, review_path]
    }
    for n in ["freeze-seal.json", "review.json", "review-seal.json", "top30.md"]:
        path = references_path.parent / n
        parents[str(path.relative_to(root))] = file_hash(path)
    if jacobian:
        for p in [lens_path, lens_path.with_name(lens_path.name + ".seal.json"), lens_file]:
            parents[str(p.relative_to(root))] = file_hash(p)
        parents.update(jacobian.get("evidence_files", {}))
    config = dict(
        schema_version=1,
        mode="saved-trace-readouts",
        name=name,
        run_path=f"runs/pilot-5/{name}/readouts",
        capture_path=str(capture_path.relative_to(root)),
        assets_path=str(assets_path.relative_to(root)),
        references_path=str(references_path.relative_to(root)),
        parents=parents,
        requests=requests,
        lens_names=lenses,
        jacobian=jacobian,
        implementation=_implementation(),
        review=review,
        inference_status=references["review"]["inference_status"],
        execution_kind=capture["execution_kind"],
        ranking=dict(raw_candidates=1000, normalized_unique=20, tie_break="ascending token ID"),
        decoder_forwards=0,
    )
    path = root / "plans" / name / "traces.readouts.json"
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
    directory = root / config["run_path"]
    if (
        config.get("mode") != "saved-trace-readouts"
        or config["implementation"] != _implementation()
    ):
        raise ValueError("Readout implementation changed; use a reviewed new execution.")
    if json.loads((directory / "meta/run_manifest.json").read_text()) != dict(
        config_path=str(path.relative_to(root)), config_sha256=_hash(config)
    ):
        raise ValueError("Readout plan manifest mismatch.")
    for relative, digest in config["parents"].items():
        if file_hash(_inside(root, relative)) != digest:
            raise ValueError("Readout parent hash mismatch.")
    load_references(root / config["references_path"], require_review=True)
    return config


def _unit(directory, name, identity):
    folder = directory / "units" / name
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    seal = json.loads(marker.read_text())
    if seal["identity"] != identity or set(seal["files"]) != {"result.json", "vector.npy"}:
        raise ValueError("Readout unit inventory changed.")
    for n, digest in seal["files"].items():
        if file_hash(folder / n) != digest:
            raise ValueError("Readout unit hash mismatch.")
    return json.loads((folder / "result.json").read_text())


def _vector(request, aggregate, count):
    condition = request["condition"]
    site = request["site"]
    positions = request["positions"]
    role = request["role"]
    delta = (
        aggregate[f"{condition}.primary_delta_sum.{site}"] / count
        if positions == PRIMARY["positions"]
        else aggregate[f"{condition}.position_sum.{site}"][positions].mean(axis=0) / count
    )
    if role in ["base", "adapted"]:
        if positions != PRIMARY["positions"]:
            return None, "base/adapted means retained only for pooled positions 0–4"
        baseline = aggregate[f"{condition}.primary_base_sum.{site}"] / count
        return baseline + (delta if role == "adapted" else 0), None
    if role == "random-direction":
        noise = np.random.default_rng(request["seed"]).normal(size=delta.shape)
        return noise / np.linalg.norm(noise) * np.linalg.norm(delta), None
    return delta, None


def _project(vector, norm, head, epsilon, ids):
    normalized = norm.astype(np.float64) * vector / np.sqrt(np.mean(vector * vector) + epsilon)
    result = np.empty(len(ids), dtype=np.float64)
    # Bounded FP64 workspace; no full vocabulary × width FP64 materialization.
    for start in range(0, len(ids), 4096):
        index = ids[start : start + 4096]
        result[start : start + len(index)] = np.asarray(head[index], dtype=np.float64) @ normalized
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite readout scores.")
    return result


def _rank(scores, ids, inventory):
    order = np.lexsort((ids, -scores))
    raw = []
    normalized = []
    seen = set()
    for index in order[:20]:
        token = int(ids[index])
        raw.append(
            dict(token_id=token, decoded=inventory.decoded[token], score=float(scores[index]))
        )
    for index in order[:1000]:
        token = int(ids[index])
        value = inventory.normalize(token)
        if value is None or value in seen:
            continue
        normalized.append(
            dict(
                token_id=token,
                decoded=inventory.decoded[token],
                string=value,
                score=float(scores[index]),
            )
        )
        seen.add(value)
        if len(normalized) == 20:
            break
    return dict(
        raw_top_tokens=raw,
        normalized_top_tokens=normalized,
        valid_count=len(normalized),
        short_list=len(normalized) < 20,
    )


def verify_readouts(path, output_root):
    root = Path(output_root).resolve()
    config = _load(path, root)
    directory = root / config["run_path"]
    verify_capture(root / config["capture_path"], root)
    verify_assets(root / config["assets_path"], root)
    marker = json.loads((directory / "complete.json").read_text())
    if marker["config_sha256"] != _hash(config) or set(marker["files"]) != {
        "results/results.json",
        "meta/run_manifest.json",
        "results/readouts.jsonl",
        "results/readouts.md",
    }:
        raise ValueError("Readout completion identity/inventory mismatch.")
    for n, digest in marker["files"].items():
        if file_hash(directory / n) != digest:
            raise ValueError("Readout completion hash mismatch.")
    for request in config["requests"]:
        if (
            _unit(directory, request["name"], dict(config_sha256=_hash(config), request=request))
            is None
        ):
            raise ValueError("Missing readout unit.")
    return json.loads((directory / "results/results.json").read_text())


def run_readouts(path, output_root, *, max_new_units=None):
    root = Path(output_root).resolve()
    config = _load(path, root)
    directory = root / config["run_path"]
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 1):
        raise ValueError("Readout work budget must be a positive unit count.")
    with run_lock(directory):
        if (directory / "complete.json").exists():
            return verify_readouts(path, root)
        capture = load_execution(root / config["capture_path"], root)
        observed = verify_capture(root / config["capture_path"], root)
        assets = verify_assets(root / config["assets_path"], root)
        folder = (root / config["assets_path"]).parent
        # File payloads are verified once at entry and again before final release.
        with np.load(
            root / capture["run_path"] / "results/aggregate.npz", allow_pickle=False
        ) as saved:
            aggregate = {k: saved[k] for k in saved.files}
        norm = np.load(folder / "norm.npy", mmap_mode="r", allow_pickle=False)
        head = np.load(folder / "head.npy", mmap_mode="r", allow_pickle=False)
        raw = json.loads((folder / "inventory.json").read_text())
        inventory = TokenInventory(
            {int(k): v for k, v in raw["decoded"].items()}, special_ids=raw["excluded_ids"]
        )
        ids = np.array(sorted(inventory.decoded))
        completed = 0
        reused = 0
        matrices = {}
        watched = [
            folder / n
            for n in ["head.npy", "norm.npy", "inventory.json", "config.json", "complete.json"]
        ]
        capture_dir = root / capture["run_path"]
        watched += [capture_dir / "results/aggregate.npz", capture_dir / "complete.json"]

        def version(p):
            stat = p.stat()
            return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

        versions = {p: version(p) for p in watched}
        try:
            for request in config["requests"]:
                if any(version(p) != v for p, v in versions.items()):
                    raise ValueError("Readout source changed during verified session.")
                name = request["name"]
                identity = dict(config_sha256=_hash(config), request=request)
                if _unit(directory, name, identity) is not None:
                    completed += 1
                    reused += 1
                    continue
                if max_new_units is not None and completed - reused >= max_new_units:
                    _write_state(directory, "paused", completed, len(config["requests"]))
                    return dict(
                        readouts_complete=False,
                        decoder_forwards=0,
                        units=completed,
                        new_units=completed - reused,
                        reused_units=reused,
                    )
                _write_state(directory, "running", completed, len(config["requests"]))
                vector, reason = _vector(request, aggregate, observed["examples"])
                output = dict(
                    **request,
                    form="raw",
                    state="unavailable" if reason else "decoded",
                    reason=reason,
                    inference_status=config["inference_status"],
                    execution_kind=config["execution_kind"],
                    readouts={},
                    coherence=None,
                    vector_source="FP64 cohort sums; retained FP16 token shards are not used for cohort means",
                    primary=request["role"] == "delta"
                    and request["condition"] in ["sft-64", "grpo-64"]
                    and request["site"] == PRIMARY["site"]
                    and request["positions"] == PRIMARY["positions"],
                )
                if vector is None:
                    vector = np.array([], dtype=np.float64)
                output["norm"] = float(np.linalg.norm(vector))
                if not reason and not np.any(vector):
                    output.update(state="undefined", reason="zero-vector")
                elif not reason:
                    for lens in config["lens_names"]:
                        v = vector
                        if lens == "jacobian":
                            site = request["site"]
                            layer = int(site.split("-")[1]) if site.startswith("block-") else None
                            if layer not in config["jacobian"]["source_layers"]:
                                output["readouts"][lens] = dict(
                                    state="unavailable", reason="unsupported source site"
                                )
                                continue
                            qualified_layers = config["jacobian"].get("qualified_source_layers")
                            qualified_positions = config["jacobian"].get("positions")
                            if (
                                qualified_layers is not None and layer not in qualified_layers
                                or qualified_positions is not None and request["positions"] != qualified_positions
                            ):
                                output["readouts"][lens] = dict(
                                    state="unavailable", reason="outside qualified source/window scope"
                                )
                                continue
                            if layer not in matrices:
                                matrices[layer] = read_matrix(
                                    root / config["jacobian"]["lens_path"],
                                    layer,
                                    expected_sha256=config["jacobian"]["lens_sha256"],
                                )
                            v = transport(vector, matrices[layer])
                        if not np.any(v):
                            output["readouts"][lens] = dict(
                                state="undefined", reason="zero-transport"
                            )
                            continue
                        scores = _project(v, norm, head, assets["epsilon"], ids)
                        output["readouts"][lens] = dict(
                            state="decoded",
                            positive=_rank(scores, ids, inventory),
                            negative=_rank(-scores, ids, inventory),
                        )
                    # Position-window coherence requires the corresponding per-example norms.
                    if request["role"] == "delta":
                        if request["positions"] == PRIMARY["positions"]:
                            key = f'{request["condition"]}.primary_delta_norm_sum.{request["site"]}'
                            denominator = (
                                float(aggregate[key]) / observed["examples"]
                                if key in aggregate
                                else None
                            )
                        elif len(request["positions"]) == 1:
                            denominator = (
                                float(
                                    aggregate[
                                        f'{request["condition"]}.position_norm_sum.{request["site"]}'
                                    ][request["positions"][0]]
                                )
                                / observed["examples"]
                            )
                        else:
                            denominator = None
                        output["coherence"] = output["norm"] / denominator if denominator else None
                unit = directory / "units" / name
                unit.mkdir(parents=True, exist_ok=True)
                np.save(unit / "vector.npy", vector, allow_pickle=False)
                _write_json(unit / "result.json", output)
                _write_json(
                    unit / "complete.json",
                    dict(
                        identity=identity,
                        files={n: file_hash(unit / n) for n in ["vector.npy", "result.json"]},
                    ),
                )
                completed += 1
                print(
                    f'Readout: {completed}/{len(config["requests"])} sealed; {name}; no decoder inference.',
                    flush=True,
                )
            # Recheck prereveal parents and immutable assets before completion.
            _load(path, root)
            verify_assets(root / config["assets_path"], root)
            verify_capture(root / config["capture_path"], root)
            result = dict(
                readouts_complete=True,
                decoder_forwards=0,
                units=completed,
                new_units=completed - reused,
                reused_units=reused,
                inference_status=config["inference_status"],
                execution_kind=config["execution_kind"],
                lens_names=config["lens_names"],
                requests=config["requests"],
                capture_sha256=file_hash(root / config["capture_path"]),
                note="Ranked readout evidence only; primary testing, nulls and arm contrasts belong to Ticket 09.",
            )
            records = [
                json.loads((directory / "units" / r["name"] / "result.json").read_text())
                for r in config["requests"]
            ]
            report = [
                "# Saved trace readouts",
                "",
                f"Inference status: {config['inference_status']}. Decoder forwards: 0.",
                "",
                "| Request | Lens | Positive raw tokens | Negative raw tokens |",
                "| --- | --- | --- | --- |",
            ]
            for record in records:
                for lens, row in record["readouts"].items():
                    if row["state"] != "decoded":
                        report.append(
                            f"| {record['name']} | {lens} | {row['state']} | {row.get('reason','')} |"
                        )
                        continue

                    def text(sign):
                        return ", ".join(
                            repr(t["decoded"]).replace("|", "\\|").replace("\n", " ")
                            for t in row[sign]["raw_top_tokens"]
                        )

                    report.append(
                        f"| {record['name']} | {lens} | {text('positive')} | {text('negative')} |"
                    )
                if not record["readouts"]:
                    report.append(
                        f"| {record['name']} | — | {record['state']} | {record['reason']} |"
                    )
            (directory / "results").mkdir(parents=True, exist_ok=True)
            (directory / "results/readouts.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            )
            (directory / "results/readouts.md").write_text("\n".join(report) + "\n")
            _write_json(directory / "results/results.json", result)
            _write_json(
                directory / "complete.json",
                dict(
                    config_sha256=_hash(config),
                    files={
                        n: file_hash(directory / n)
                        for n in [
                            "results/results.json",
                            "meta/run_manifest.json",
                            "results/readouts.jsonl",
                            "results/readouts.md",
                        ]
                    },
                ),
            )
            _write_state(directory, "completed", completed, len(config["requests"]))
            return result
        except BaseException as exc:
            _write_state(directory, "failed", completed, len(config["requests"]), str(exc))
            raise
