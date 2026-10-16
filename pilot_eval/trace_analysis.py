"""Ticket09 saved automatic relevance, nulls and paired document inference.

No model inference. All costly projection chunks have durable identities. Real
counts come only from the approved reference procedure; tiny overrides are
restricted to synthetic artifacts. Missing qualified scope stays unavailable.
"""

import json
import math
import time
from pathlib import Path
import numpy as np
from pilot_eval.trace_corpus import _inside
from pilot_eval.trace_readouts import _load as load_readouts, verify_readouts
from pilot_eval.trace_production_plan import load_execution
from pilot_eval.trace_production import verify_capture
from pilot_eval.trace_readout_assets import verify_assets
from pilot_eval.trace_vocabulary import load_references, TokenInventory
from pilot_eval.trace_reference_workflow import approved_procedure
from pilot_eval.trace_lenses import read_matrix
from pilot_eval.trace_analysis_scores import LensScorer
from pilot_eval.trace_statistics import null_pvalue, holm, c3_pvalue, snr
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name

ARMS = ["sft-64", "grpo-64"]
SITE = "block-14-output"
LENSES = ["logit", "jacobian"]


def implementation():
    folder = Path(__file__).parent
    return {
        n: file_hash(folder / n)
        for n in [
            "trace_analysis.py",
            "trace_analysis_scores.py",
            "trace_scoring.py",
            "trace_statistics.py",
        ]
    }


def prepare_analysis(readouts_path, output_root, name, *, test_counts=None):
    root = Path(output_root).resolve()
    safe_name(name)
    path = _inside(root, readouts_path)
    r = load_readouts(path, root)
    capture = load_execution(root / r["capture_path"], root)
    verify_capture(root / r["capture_path"], root)
    refs = load_references(root / r["references_path"], require_review=True)
    if refs["parameters"].get("procedure") != approved_procedure():
        raise ValueError(
            "Analysis needs the exact approved reference procedure; no invented defaults."
        )
    if (
        r["lens_names"] != LENSES
        or capture["review"]["lens_mode"] != "logit-and-jacobian"
    ):
        raise ValueError("The primary four-test family requires both qualified lenses.")
    j = r["jacobian"]
    if (
        j.get("qualified_source_layers") is not None
        and 14 not in j["qualified_source_layers"]
        or j.get("positions") is not None
        and j["positions"] != [0, 1, 2, 3, 4]
    ):
        raise ValueError("Jacobian primary window is outside qualified scope.")
    procedure = approved_procedure()
    counts = dict(
        null=procedure["null"]["draws"],
        primary=procedure["bootstrap"]["primary_draws"],
        c3=procedure["bootstrap"]["c3_draws"],
        stratum=procedure["bootstrap"]["stratum_draws"],
        matched=procedure["matched_legacy"]["draws"],
        ncurve=procedure["ncurves"]["draws"],
        sizes=procedure["ncurves"]["sizes"],
    )
    if test_counts is not None:
        if r["execution_kind"] != "synthetic-cpu" or set(test_counts) != set(counts):
            raise ValueError(
                "Draw-count overrides are restricted to synthetic CPU fixtures."
            )
        counts = test_counts
    if any(
        type(v) is not int or v < 1 for k, v in counts.items() if k != "sizes"
    ) or counts["sizes"] != [25, 50, 100, 200]:
        raise ValueError("Invalid declared analysis counts/sizes.")
    requests = r["requests"]
    for arm in ARMS:
        if not any(
            q["condition"] == arm
            and q["role"] == "delta"
            and q["site"] == SITE
            and q["positions"] == [0, 1, 2, 3, 4]
            for q in requests
        ):
            raise ValueError("Both frozen primary readout requests are required.")
    parents = {str(path.relative_to(root)): file_hash(path), **r["parents"]}
    for relative in [r["capture_path"], r["assets_path"]]:
        parents[relative] = file_hash(root / relative)
    config = dict(
        schema_version=1,
        mode="trace-automatic-analysis",
        name=name,
        run_path=f"runs/pilot-5/{name}/automatic-analysis",
        readouts_path=str(path.relative_to(root)),
        capture_path=r["capture_path"],
        assets_path=r["assets_path"],
        references_path=r["references_path"],
        execution_kind=r["execution_kind"],
        inference_status=r["inference_status"],
        procedure=procedure,
        counts=counts,
        implementation=implementation(),
        parents=parents,
        chunk_size=16,
        quantile_method="linear",
        positive_only=True,
        primary_family="SFT/GRPO × logit/Jacobian; pooled0–4 block14 step64 ref_questions",
        covariance_null=dict(
            state="unavailable",
            reason="No reviewed covariance/PSD qualification receipt; stored moments alone do not qualify this secondary null.",
        ),
        exposures=dict(
            reference_review=refs["review"]["exposure_status"],
            null_timing="record actual readout completion state before null freeze",
        ),
    )
    output = root / "plans" / name / "traces.analysis.json"
    with run_lock(output.parent):
        _save_frozen(output, config)
        _save_frozen(
            root / config["run_path"] / "meta/run_manifest.json",
            dict(
                config_sha256=_hash(config), config_path=str(output.relative_to(root))
            ),
        )
    return output


def _load(path, root):
    path = _inside(root, path)
    c = json.loads(path.read_text())
    if (
        c["mode"] != "trace-automatic-analysis"
        or c["implementation"] != implementation()
    ):
        raise ValueError(
            "Analysis implementation changed; preserve evidence and use a new name."
        )
    if json.loads(
        (root / c["run_path"] / "meta/run_manifest.json").read_text()
    ) != dict(config_sha256=_hash(c), config_path=str(path.relative_to(root))):
        raise ValueError("Analysis manifest mismatch.")
    for n, h in c["parents"].items():
        if file_hash(_inside(root, n)) != h:
            raise ValueError("Analysis parent hash mismatch.")
    load_readouts(root / c["readouts_path"], root)
    return c


def _scorers(c, root):
    r = load_readouts(root / c["readouts_path"], root)
    a = verify_assets(root / c["assets_path"], root)
    folder = (root / c["assets_path"]).parent
    norm = np.load(folder / "norm.npy", mmap_mode="r", allow_pickle=False)
    head = np.load(folder / "head.npy", mmap_mode="r", allow_pickle=False)
    raw = json.loads((folder / "inventory.json").read_text())
    inventory = TokenInventory(
        {int(k): v for k, v in raw["decoded"].items()}, special_ids=raw["excluded_ids"]
    )
    refs = load_references(root / c["references_path"], require_review=True)[
        "references"
    ]
    j = r["jacobian"]
    matrix = read_matrix(root / j["lens_path"], 14, expected_sha256=j["lens_sha256"])
    return {
        lens: LensScorer(
            norm,
            head,
            a["epsilon"],
            inventory,
            refs,
            matrix=matrix if lens == "jacobian" else None,
            batch_size=c["chunk_size"],
        )
        for lens in LENSES
    }


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _read_unit(directory, name, c):
    folder = directory / "units" / name
    seal = folder / "complete.json"
    if not seal.exists():
        return None
    marker = json.loads(seal.read_text())
    if marker != dict(
        config_sha256=_hash(c), result_sha256=file_hash(folder / "result.json")
    ):
        raise ValueError("Analysis unit hash mismatch.")
    return json.loads((folder / "result.json").read_text())


def _save_unit(directory, name, c, value):
    folder = directory / "units" / name
    _write_json(folder / "result.json", _clean(value))
    _write_json(
        folder / "complete.json",
        dict(config_sha256=_hash(c), result_sha256=file_hash(folder / "result.json")),
    )


def _complete(directory, c, result, files):
    _write_json(directory / "results/results.json", _clean(result))
    names = ["results/results.json", *files]
    _write_json(
        directory / "complete.json",
        dict(
            config_sha256=_hash(c), files={n: file_hash(directory / n) for n in names}
        ),
    )


def _verify(directory, c, required):
    marker = json.loads((directory / "complete.json").read_text())
    if marker["config_sha256"] != _hash(c) or set(marker["files"]) != set(required) | {
        "results/results.json"
    }:
        raise ValueError("Analysis completion inventory mismatch.")
    for n, h in marker["files"].items():
        if file_hash(directory / n) != h:
            raise ValueError("Analysis completion hash mismatch.")
    return json.loads((directory / "results/results.json").read_text())


def _null_units(c):
    # JLens individual positions are not inferred qualified from its pooled receipt.
    for lens, form in [
        ("logit", "pooled"),
        ("jacobian", "pooled"),
        ("logit", "max-five"),
    ]:
        for start in range(0, c["counts"]["null"], c["chunk_size"]):
            yield f"{lens}-{form}-{start:06d}", lens, form, start


def run_nulls(path, output_root, *, max_new_chunks=None):
    root = Path(output_root).resolve()
    c = _load(path, root)
    directory = root / c["run_path"] / "nulls"
    if max_new_chunks is not None and (
        type(max_new_chunks) is not int or max_new_chunks < 1
    ):
        raise ValueError("Positive chunk budget required.")
    with run_lock(directory):
        if (directory / "complete.json").exists():
            return verify_nulls(path, root)
        r = load_readouts(root / c["readouts_path"], root)
        readout_dir = root / r["run_path"]
        timing = dict(
            readouts_completed_before_null_freeze=(
                readout_dir / "complete.json"
            ).exists(),
            learned_unit_seals_present=any(
                (readout_dir / "units").glob("*/complete.json")
            ),
            limitation="Supported saved artifacts only; no claim about unrecorded manual exposure.",
        )
        _save_frozen(directory / "exposure-at-start.json", timing)
        capture = load_execution(root / c["capture_path"], root)
        dimension = capture["width"]
        seed = c["procedure"]["null"]["seed"]
        # Separate seeded schedules for pooled and matched five-position statistics.
        vectors = {
            form: np.random.default_rng(seed).standard_normal(
                (c["counts"]["null"], positions, dimension)
            )
            for form, positions in [("pooled", 1), ("max-five", 5)]
        }
        for form, v in vectors.items():
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            npy = directory / f"{form}-vectors.npy"
            if npy.exists():
                if not np.array_equal(np.load(npy, allow_pickle=False), v):
                    raise ValueError("Null schedule changed.")
            else:
                np.save(npy, v, allow_pickle=False)
        scorers = _scorers(c, root)
        new = reused = 0
        start_time = time.monotonic()
        try:
            for name, lens, form, start in _null_units(c):
                if _read_unit(directory, name, c) is not None:
                    reused += 1
                    continue
                if max_new_chunks is not None and new >= max_new_chunks:
                    _write_state(
                        directory, "incomplete", new + reused, len(list(_null_units(c)))
                    )
                    return dict(
                        nulls_complete=False,
                        decoder_forwards=0,
                        new_chunks=new,
                        reused_chunks=reused,
                    )
                values = vectors[form][start : start + c["chunk_size"]]
                rows = scorers[lens].many(values.reshape(-1, dimension))
                positions = values.shape[1]
                draws = []
                for i in range(len(values)):
                    selected = rows[i * positions : (i + 1) * positions]
                    scores = {
                        ref: {
                            metric: (
                                max(r["scores"][ref][metric] for r in selected)
                                if all(
                                    r["scores"][ref][metric] is not None
                                    for r in selected
                                )
                                else None
                            )
                            for metric in [
                                "overlap",
                                "overlap_word",
                                "overlap_format",
                                "soft_rank",
                            ]
                        }
                        for ref in scorers[lens].references
                    }
                    draws.append(
                        dict(
                            draw=start + i,
                            scores=scores,
                            valid=all(r["state"] == "scored" for r in selected),
                        )
                    )
                _save_unit(directory, name, c, dict(lens=lens, form=form, draws=draws))
                new += 1
                print(
                    f"Analysis nulls: {new+reused}/{len(list(_null_units(c)))} chunks sealed; {name}.",
                    flush=True,
                )
            _load(path, root)
            result = dict(
                nulls_complete=True,
                decoder_forwards=0,
                new_chunks=new,
                reused_chunks=reused,
                seconds_this_attempt=time.monotonic() - start_time,
                seed=seed,
                draws=c["counts"]["null"],
                exposure=timing,
                activation_shaped=c["covariance_null"],
                learned_lists_revealed=False,
            )
            files = [
                "exposure-at-start.json",
                "pooled-vectors.npy",
                "max-five-vectors.npy",
            ]
            files.extend(
                f"units/{n}/{f}"
                for n, _, _, _ in _null_units(c)
                for f in ["result.json", "complete.json"]
            )
            _complete(directory, c, result, files)
            _write_state(directory, "completed", len(files), len(files))
            return result
        except BaseException as exc:
            _write_state(
                directory, "failed", new + reused, len(list(_null_units(c))), str(exc)
            )
            raise


def verify_nulls(path, output_root):
    root = Path(output_root).resolve()
    c = _load(path, root)
    d = root / c["run_path"] / "nulls"
    files = ["exposure-at-start.json", "pooled-vectors.npy", "max-five-vectors.npy"]
    for n, _, _, _ in _null_units(c):
        if _read_unit(d, n, c) is None:
            raise ValueError("Missing null chunk.")
        files.extend(f"units/{n}/{f}" for f in ["result.json", "complete.json"])
    return _verify(d, c, files)


def _documents(c, root):
    capture = load_execution(root / c["capture_path"], root)
    verify_capture(root / c["capture_path"], root)
    ids = [r["id"] for r in capture["examples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate capture document IDs.")
    pooled = {a: [] for a in [*ARMS, "random-64", "base"]}
    base = []
    positions = {a: [] for a in [*ARMS, "random-64"]}
    for i, indices in enumerate(capture["batches"]):
        with np.load(
            root / capture["run_path"] / "units" / f"{i:06d}" / "arrays.npz",
            allow_pickle=False,
        ) as z:
            for arm in pooled:
                pooled[arm].append(z[f"{arm}.primary_delta.{SITE}"].copy())
                if arm in positions:
                    positions[arm].append(
                        z[f"{arm}.retained_delta.{SITE}"][:, :5, :].astype(np.float64)
                    )
            base.append(z[f"base.primary_base.{SITE}"].copy())
    return (
        capture,
        ids,
        {k: np.concatenate(v) for k, v in pooled.items()},
        np.concatenate(base),
        {k: np.concatenate(v) for k, v in positions.items()},
    )


def _schedule(c, ids, strata):
    n = len(ids)
    seed = c["procedure"]["bootstrap"]["seed"]
    counts = c["counts"]
    rng = np.random.default_rng(seed)
    primary = rng.integers(0, n, size=(counts["primary"], n))
    c3 = np.random.default_rng(seed).integers(0, n, size=(counts["c3"], n))
    groups = {
        g: [i for i in range(n) if strata[i] == g] for g in ["legacy-150", "new-50"]
    }
    schedules = dict(primary=primary, c3=c3)
    for g, indices in groups.items():
        if not indices:
            raise ValueError("Both preserved corpus strata are required.")
        schedules[g] = np.asarray(indices)[
            np.random.default_rng(seed).integers(
                0, len(indices), size=(counts["stratum"], len(indices))
            )
        ]
    legacy = groups["legacy-150"]
    matched = c["procedure"]["matched_legacy"]["size"]
    rng = np.random.default_rng(c["procedure"]["matched_legacy"]["seed"])
    schedules["matched"] = np.stack(
        [
            np.sort(rng.choice(legacy, size=matched, replace=False))
            for _ in range(counts["matched"])
        ]
    )
    rng = np.random.default_rng(c["procedure"]["ncurves"]["seed"])
    for size in counts["sizes"]:
        if size > n:
            raise ValueError("Declared N-curve size exceeds retained cohort.")
        schedules[f"n{size}"] = np.stack(
            [
                np.sort(rng.choice(n, size=size, replace=False))
                for _ in range(counts["ncurve"])
            ]
        )
    return schedules


def _analysis_units(c):
    yield "observed", "observed", None, 0
    for lens in LENSES:
        for arm in ARMS:
            for kind in [
                "primary",
                "legacy-150",
                "new-50",
                "matched",
                *[f"n{x}" for x in c["counts"]["sizes"]],
            ]:
                total = (
                    c["counts"]["primary"]
                    if kind == "primary"
                    else (
                        c["counts"]["stratum"]
                        if kind in ["legacy-150", "new-50"]
                        else (
                            c["counts"]["matched"]
                            if kind == "matched"
                            else c["counts"]["ncurve"]
                        )
                    )
                )
                for start in range(0, total, c["chunk_size"]):
                    yield f"{lens}-{arm}-{kind}-{start:06d}", kind, (lens, arm), start
        for start in range(0, c["counts"]["c3"], c["chunk_size"]):
            yield f"{lens}-paired-{start:06d}", "c3", lens, start
    # All-site pooled layer curves are descriptive; JLens outside block14 is unavailable.
    yield "layer-curves", "layers", None, 0
    for arm in ARMS:
        yield f"logit-{arm}-max-five", "max-five", arm, 0
        for start in range(0, c["counts"]["primary"], c["chunk_size"]):
            yield f"logit-{arm}-max-bootstrap-{start:06d}", "max-bootstrap", arm, start


def _interval(values):
    return (
        np.quantile(values, [0.025, 0.975], method="linear").tolist()
        if values and all(v is not None for v in values)
        else [None, None]
    )


def _null_values(c, root, lens, ref, form="pooled"):
    values = []
    for n, l, f, _ in _null_units(c):
        if l == lens and f == form:
            values.extend(
                r["scores"][ref]["overlap"] if r["valid"] else None
                for r in _read_unit(root / c["run_path"] / "nulls", n, c)["draws"]
            )
    return values


def run_analysis(path, output_root, *, max_new_units=None):
    root = Path(output_root).resolve()
    c = _load(path, root)
    directory = root / c["run_path"]
    verify_nulls(path, root)
    if max_new_units is not None and (
        type(max_new_units) is not int or max_new_units < 1
    ):
        raise ValueError("Positive unit budget required.")
    with run_lock(directory):
        if (directory / "complete.json").exists():
            return verify_analysis(path, root)
        verify_readouts(root / c["readouts_path"], root)
        capture, ids, pooled, base, positions = _documents(c, root)
        strata = [r["stratum"] for r in capture["examples"]]
        schedules = _schedule(c, ids, strata)
        schedule_path = directory / "draws.npz"
        if schedule_path.exists():
            with np.load(schedule_path, allow_pickle=False) as z:
                if set(z.files) != set(schedules) or any(
                    not np.array_equal(z[k], v) for k, v in schedules.items()
                ):
                    raise ValueError("Analysis paired document schedule changed.")
        else:
            np.savez_compressed(schedule_path, **schedules)
        _save_frozen(
            directory / "document-order.json",
            dict(
                ids=ids,
                strata=strata,
                retained_precision=capture["storage_dtype"],
                primary_precision="FP64 pooled summaries",
            ),
        )
        scorers = _scorers(c, root)
        new = reused = 0
        start_time = time.monotonic()
        units = list(_analysis_units(c))
        try:
            for name, kind, selection, start in units:
                if _read_unit(directory, name, c) is not None:
                    reused += 1
                    continue
                if max_new_units is not None and new >= max_new_units:
                    _write_state(directory, "incomplete", new + reused, len(units))
                    return dict(
                        analysis_complete=False,
                        decoder_forwards=0,
                        new_units=new,
                        reused_units=reused,
                    )
                started = time.monotonic()
                if kind == "observed":
                    rows = []
                    for lens, scorer in scorers.items():
                        for arm in [*ARMS, "random-64"]:
                            for group, idx in [
                                ("all", np.arange(len(ids))),
                                *[
                                    (g, np.flatnonzero(np.array(strata) == g))
                                    for g in ["legacy-150", "new-50"]
                                ],
                            ]:
                                vector = pooled[arm][idx].mean(axis=0)
                                rows.append(
                                    dict(
                                        lens=lens,
                                        condition=arm,
                                        group=group,
                                        n=len(idx),
                                        snr=_clean(snr(pooled[arm][idx])),
                                        **scorer.many([vector])[0],
                                    )
                                )
                        for role, v in [
                            ("base", base.mean(axis=0)),
                            *[
                                (a + "-adapted", (base + pooled[a]).mean(axis=0))
                                for a in ARMS
                            ],
                        ]:
                            rows.append(
                                dict(
                                    lens=lens,
                                    condition=role,
                                    group="all",
                                    n=len(ids),
                                    interpretation="secondary baseline on same FineWeb documents",
                                    **scorer.many([v])[0],
                                )
                            )
                    value = dict(rows=rows)
                elif kind in [
                    "primary",
                    "legacy-150",
                    "new-50",
                    "matched",
                ] or kind.startswith("n"):
                    lens, arm = selection
                    draw = schedules[kind][start : start + c["chunk_size"]]
                    vectors = np.stack([pooled[arm][idx].mean(axis=0) for idx in draw])
                    score = scorers[lens].many(vectors)
                    value = dict(
                        lens=lens,
                        condition=arm,
                        kind=kind,
                        rows=[
                            dict(
                                draw=start + i,
                                snr=_clean(snr(pooled[arm][idx])),
                                counts={
                                    g: int(np.sum(np.array(strata)[idx] == g))
                                    for g in ["legacy-150", "new-50"]
                                },
                                **row,
                            )
                            for i, (idx, v, row) in enumerate(zip(draw, vectors, score))
                        ],
                    )
                elif kind == "c3":
                    lens = selection
                    draw = schedules["c3"][start : start + c["chunk_size"]]
                    left = scorers[lens].many(
                        np.stack([pooled[ARMS[0]][idx].mean(axis=0) for idx in draw])
                    )
                    right = scorers[lens].many(
                        np.stack([pooled[ARMS[1]][idx].mean(axis=0) for idx in draw])
                    )
                    value = dict(
                        lens=lens,
                        rows=[
                            dict(
                                draw=start + i,
                                differences={
                                    ref: (
                                        a["scores"][ref]["overlap"]
                                        - b["scores"][ref]["overlap"]
                                        if a["scores"][ref]["overlap"] is not None
                                        and b["scores"][ref]["overlap"] is not None
                                        else None
                                    )
                                    for ref in scorers[lens].references
                                },
                            )
                            for i, (a, b) in enumerate(zip(left, right))
                        ],
                    )
                elif kind == "layers":
                    with np.load(
                        root / capture["run_path"] / "results/aggregate.npz",
                        allow_pickle=False,
                    ) as z:
                        rows = []
                        for arm in ARMS:
                            vectors = [
                                z[f"{arm}.primary_delta_sum.{site}"] / len(ids)
                                for site in capture["sites"]
                            ]
                            scored = scorers["logit"].many(vectors)
                            rows.extend(
                                dict(
                                    condition=arm,
                                    lens="logit",
                                    site=site,
                                    classification="secondary",
                                    **row,
                                )
                                for site, row in zip(capture["sites"], scored)
                            )
                    value = dict(
                        rows=rows,
                        jacobian_other_layers="unavailable: primary-only qualification",
                    )
                elif kind == "max-five":
                    values = positions[selection]
                    means = values.mean(axis=0)
                    row = scorers["logit"].many(means)
                    scores = {
                        ref: (
                            max(r["scores"][ref]["overlap"] for r in row)
                            if all(r["scores"][ref]["overlap"] is not None for r in row)
                            else None
                        )
                        for ref in scorers["logit"].references
                    }
                    value = dict(
                        condition=selection,
                        lens="logit",
                        classification="secondary",
                        observed=scores,
                        per_position=row,
                        per_position_snr=[
                            _clean(snr(values[:, i, :])) for i in range(5)
                        ],
                        jacobian="unavailable: pooled-only qualification",
                    )
                elif kind == "max-bootstrap":
                    values = positions[selection]
                    draw = schedules["primary"][start : start + c["chunk_size"]]
                    means = np.stack([values[idx].mean(axis=0) for idx in draw])
                    scores = scorers["logit"].many(means.reshape(-1, capture["width"]))
                    rows = []
                    for i, idx in enumerate(draw):
                        five = scores[i * 5 : (i + 1) * 5]
                        rows.append(
                            dict(
                                draw=start + i,
                                scores={
                                    ref: (
                                        max(r["scores"][ref]["overlap"] for r in five)
                                        if all(
                                            r["scores"][ref]["overlap"] is not None
                                            for r in five
                                        )
                                        else None
                                    )
                                    for ref in scorers["logit"].references
                                },
                                position_snr=[
                                    _clean(snr(values[idx, j, :])) for j in range(5)
                                ],
                            )
                        )
                    value = dict(
                        condition=selection,
                        lens="logit",
                        classification="secondary",
                        rows=rows,
                    )
                _save_unit(
                    directory,
                    name,
                    c,
                    dict(**value, seconds=time.monotonic() - started),
                )
                new += 1
                print(
                    f"Analysis: {new+reused}/{len(units)} durable units; {name}; elapsed {(time.monotonic()-start_time)/60:.1f}min.",
                    flush=True,
                )
            return _finish(path, c, root, new, reused, time.monotonic() - start_time)
        except BaseException as exc:
            _write_state(directory, "failed", new + reused, len(units), str(exc))
            raise


def _finish(path, c, root, new, reused, seconds):
    directory = root / c["run_path"]
    observed = _read_unit(directory, "observed", c)["rows"]

    def draws(lens, arm, kind):
        values = []
        for n, k, s, _ in _analysis_units(c):
            if k == kind and s == (lens, arm):
                values.extend(_read_unit(directory, n, c)["rows"])
        return values

    primary = []
    contrasts = []
    robustness = []
    curves = []
    ref = "ref_questions"
    for lens in LENSES:
        for arm in ARMS:
            obs = next(
                r
                for r in observed
                if r["lens"] == lens and r["condition"] == arm and r["group"] == "all"
            )
            overlap = obs["scores"][ref]["overlap"]
            null = _null_values(c, root, lens, ref)
            finite = overlap is not None and all(v is not None for v in null)
            test = (
                null_pvalue(overlap, null)
                if finite
                else dict(p_value=None, null_percentile=None)
            )
            bootstrap = [
                r["scores"][ref]["overlap"] for r in draws(lens, arm, "primary")
            ]
            primary.append(
                dict(
                    lens=lens,
                    condition=arm,
                    reference=ref,
                    overlap=overlap,
                    interval=_interval(bootstrap),
                    snr=obs["snr"],
                    **test,
                )
            )
            for group in ["legacy-150", "new-50"]:
                row = next(
                    r
                    for r in observed
                    if r["lens"] == lens
                    and r["condition"] == arm
                    and r["group"] == group
                )
                values = [r["scores"][ref]["overlap"] for r in draws(lens, arm, group)]
                robustness.append(
                    dict(
                        lens=lens,
                        condition=arm,
                        group=group,
                        n=row["n"],
                        overlap=row["scores"][ref]["overlap"],
                        interval=_interval(values),
                        snr=row["snr"],
                        interpretation="robustness only; not exposure effect",
                    )
                )
            matched = [r["scores"][ref]["overlap"] for r in draws(lens, arm, "matched")]
            new_score = next(
                r["overlap"]
                for r in robustness
                if r["lens"] == lens
                and r["condition"] == arm
                and r["group"] == "new-50"
            )
            robustness.append(
                dict(
                    lens=lens,
                    condition=arm,
                    group="matched-legacy",
                    n=50,
                    draws=len(matched),
                    new_score_percentile=(
                        100 * float(np.mean(np.asarray(matched) <= new_score))
                        if new_score is not None and all(v is not None for v in matched)
                        else None
                    ),
                    interpretation="size-matched sensitivity; no exposure-effect inference",
                )
            )
            for size in c["counts"]["sizes"]:
                rows = draws(lens, arm, f"n{size}")
                values = [r["scores"][ref]["overlap"] for r in rows]
                curves.append(
                    dict(
                        lens=lens,
                        condition=arm,
                        n=size,
                        draws=len(rows),
                        median=(
                            float(np.median(values))
                            if all(v is not None for v in values)
                            else None
                        ),
                        range_5_95=(
                            np.quantile(values, [0.05, 0.95], method="linear").tolist()
                            if all(v is not None for v in values)
                            else [None, None]
                        ),
                        snr_median=(
                            float(np.median([r["snr"] for r in rows]))
                            if all(r["snr"] is not None for r in rows)
                            else None
                        ),
                        full_cohort_degeneracy=size == 200,
                    )
                )
        paired = []
        for n, k, s, _ in _analysis_units(c):
            if k == "c3" and s == lens:
                paired.extend(_read_unit(directory, n, c)["rows"])
        for reference in ["ref_questions", "ref_gold", "ref_grpo"]:
            differences = [r["differences"][reference] for r in paired]
            a = next(
                r
                for r in observed
                if r["lens"] == lens
                and r["condition"] == ARMS[0]
                and r["group"] == "all"
            )["scores"][reference]["overlap"]
            b = next(
                r
                for r in observed
                if r["lens"] == lens
                and r["condition"] == ARMS[1]
                and r["group"] == "all"
            )["scores"][reference]["overlap"]
            contrasts.append(
                dict(
                    lens=lens,
                    reference=reference,
                    classification="primary" if reference == ref else "secondary",
                    difference=a - b if a is not None and b is not None else None,
                    interval=_interval(differences),
                    p_value=(
                        c3_pvalue(differences)
                        if all(v is not None for v in differences)
                        else None
                    ),
                    interpretation="declared bootstrap-tail statistic; nominal calibration not established",
                )
            )
    valid = all(r["p_value"] is not None for r in primary)
    if valid:
        for row, adjusted in zip(primary, holm([r["p_value"] for r in primary])):
            row["holm_p_value"] = adjusted
    else:
        for row in primary:
            row["holm_p_value"] = None
    main = [r for r in contrasts if r["classification"] == "primary"]
    if all(r["p_value"] is not None for r in main):
        for row, p in zip(main, holm([r["p_value"] for r in main])):
            row["holm_p_value"] = p
    nulls = verify_nulls(path, root)
    inference = c["inference_status"]
    if (
        nulls["exposure"]["readouts_completed_before_null_freeze"]
        or nulls["exposure"]["learned_unit_seals_present"]
    ):
        inference = "exploratory: nulls frozen after learned readout artifacts"
    secondary_max = []
    for arm in ARMS:
        obs = _read_unit(directory, f"logit-{arm}-max-five", c)
        boot = []
        for n, k, sel, _ in _analysis_units(c):
            if k == "max-bootstrap" and sel == arm:
                boot.extend(_read_unit(directory, n, c)["rows"])
        for reference, value in obs["observed"].items():
            null = _null_values(c, root, "logit", reference, form="max-five")
            p = (
                null_pvalue(value, null)["p_value"]
                if value is not None and all(v is not None for v in null)
                else None
            )
            secondary_max.append(
                dict(
                    condition=arm,
                    lens="logit",
                    classification="secondary",
                    reference=reference,
                    overlap=value,
                    interval=_interval([r["scores"][reference] for r in boot]),
                    null_p_value=p,
                    interpretation="Descriptive matched max-five search; outside primary testing family.",
                )
            )
    result = dict(
        analysis_complete=True,
        decoder_forwards=0,
        synthetic_only=c["execution_kind"] == "synthetic-cpu",
        primary=primary,
        contrasts=main,
        secondary_contrasts=[
            r for r in contrasts if r["classification"] == "secondary"
        ],
        secondary_max=secondary_max,
        robustness=robustness,
        ncurves=curves,
        inference_status=inference,
        primary_family_complete=valid,
        new_units=new,
        reused_units=reused,
        seconds_this_attempt=seconds,
        limitations=[
            "One arm passing a threshold does not establish an arm difference.",
            "C3 uses approved bootstrap-tail statistic; no exact/nominal calibration claim.",
            "Primary resampling uses FP64 pooled summaries; position analyses use disclosed retained precision.",
            "JLens other-layer and per-position/max-five results are unavailable under pooled-block14 qualification.",
            "Secondary activation-shaped null lacks covariance/PSD qualification.",
            "Intermediate checkpoints were deliberately excluded from endpoint production.",
            "Undefined vectors/SNR/soft ranks stay undefined; no replacement/resampling.",
        ],
    )
    records = [
        dict(
            classification=(
                "primary"
                if r["condition"] in ARMS and r["group"] == "all"
                else "secondary"
            ),
            **r,
        )
        for r in observed
    ]
    records.extend(_read_unit(directory, "layer-curves", c)["rows"])
    for arm in ARMS:
        records.append(_read_unit(directory, f"logit-{arm}-max-five", c))
    lines = [
        "# Pilot 5 automatic relevance",
        "",
        f"Inference status: {inference}",
        "",
        "## Primary pooled block14 / positions0–4 / step64 / ref_questions",
        "",
        "| Arm | Lens | Overlap | 95% interval | Null p | Holm p | SNR |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for r in primary:
        lines.append(
            f"| {r['condition']} | {r['lens']} | {r['overlap']} | {r['interval']} | {r['p_value']} | {r.get('holm_p_value')} | {r['snr']} |"
        )
    lines += [
        "",
        "## Direct SFT minus GRPO contrasts",
        "",
        "Passing only one arm’s gate does not establish an arm difference.",
        "",
    ]
    lines.extend(
        f"- {r['lens']}: difference {r['difference']}; interval {r['interval']}; Holm p {r.get('holm_p_value')}."
        for r in main
    )
    lines += ["", "## Limitations", ""] + ["- " + x for x in result["limitations"]]
    _write_json(directory / "results/results.json", _clean(result))
    (directory / "results/scores.jsonl").write_text(
        "".join(json.dumps(_clean(r), allow_nan=False) + "\n" for r in records)
    )
    (directory / "results/report.md").write_text("\n".join(lines) + "\n")
    _load(path, root)
    verify_readouts(root / c["readouts_path"], root)
    files = [
        "draws.npz",
        "document-order.json",
        "results/scores.jsonl",
        "results/report.md",
    ]
    files.extend(
        f"units/{n}/{f}"
        for n, _, _, _ in _analysis_units(c)
        for f in ["result.json", "complete.json"]
    )
    _complete(directory, c, result, files)
    _write_state(directory, "completed", len(files), len(files))
    return result


def verify_analysis(path, output_root):
    root = Path(output_root).resolve()
    c = _load(path, root)
    verify_nulls(path, root)
    verify_readouts(root / c["readouts_path"], root)
    directory = root / c["run_path"]
    files = [
        "draws.npz",
        "document-order.json",
        "results/scores.jsonl",
        "results/report.md",
    ]
    for n, _, _, _ in _analysis_units(c):
        if _read_unit(directory, n, c) is None:
            raise ValueError("Missing analysis unit.")
        files.extend(f"units/{n}/{f}" for f in ["result.json", "complete.json"])
    return _verify(directory, c, files)
