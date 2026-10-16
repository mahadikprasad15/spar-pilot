"""Self-contained norm/head/token inventory; no decoder construction or forwards."""

import json
from pathlib import Path
import numpy as np
from pilot_eval.trace_production_plan import load_execution
from pilot_eval.trace_corpus import _inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name


def weight_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Incomplete safetensor header.")
        length = int.from_bytes(prefix, "little")
        if not 1 <= length <= 100000000 or length + 8 > path.stat().st_size:
            raise ValueError("Invalid safetensor header byte count.")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensor metadata.")
    return length, header


def read_weight(path, key):
    """Read supported pinned tensor bytes without importing Torch, including BF16."""
    import math

    length, header = weight_header(path)
    row = header[key]
    dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(row.get("dtype"))
    shape = row.get("shape")
    offsets = row.get("data_offsets")
    if (
        dtype is None
        or not isinstance(shape, list)
        or not shape
        or any(type(i) is not int or i < 1 for i in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(i) is not int for i in offsets)
        or offsets[0] < 0
        or offsets[1] - offsets[0] != math.prod(shape) * np.dtype(dtype).itemsize
        or offsets[1] + 8 + length > Path(path).stat().st_size
    ):
        raise ValueError("Invalid readout weight shape, dtype or byte offsets.")
    values = np.memmap(
        path, dtype=dtype, mode="r", offset=8 + length + offsets[0], shape=tuple(shape)
    )
    if row["dtype"] == "BF16":
        return (np.asarray(values, dtype=np.uint32) << 16).view(np.float32)
    return np.array(values, dtype=np.float32, copy=True)


class HFReadoutAssets:
    def load(self, config):
        from huggingface_hub import hf_hub_download, list_repo_files
        from pilot_eval.trace_vocabulary import TokenInventory

        model = config["model"]
        revision = config["model_revision"]
        files = set(list_repo_files(model, revision=revision))
        origin = {}
        downloaded = {}

        def download(name, rev=revision):
            if (name, rev) not in downloaded:
                path = Path(hf_hub_download(model, name, revision=rev))
                origin[name] = file_hash(path)
                downloaded[name, rev] = path
            return downloaded[name, rev]

        metadata = json.loads(download("config.json").read_text())
        if metadata.get("hidden_size") != config["width"] or metadata.get("model_type") != "qwen2":
            raise ValueError("Readout basis requires the pinned Qwen2 RMS-norm architecture.")
        if "model.safetensors.index.json" in files:
            index = json.loads(download("model.safetensors.index.json").read_text())
            mapping = index["weight_map"]
        elif "model.safetensors" in files:
            weight = download("model.safetensors")
            _, header = weight_header(weight)
            mapping = {k: "model.safetensors" for k in header if k != "__metadata__"}
        else:
            raise ValueError("Pinned model lacks supported safetensor weights.")
        head_key = "lm_head.weight"
        if head_key not in mapping:
            if metadata.get("tie_word_embeddings") is not True:
                raise ValueError("Missing untied output head.")
            head_key = "model.embed_tokens.weight"
        selected = {}
        for key in ["model.norm.weight", head_key]:
            if key not in mapping:
                raise ValueError(f"Missing readout tensor {key}.")
            path = download(mapping[key])
            selected[key] = read_weight(path, key)
        tokenizer_path = download("tokenizer.json", config["tokenizer_revision"])
        inventory = TokenInventory.from_file(tokenizer_path)
        return dict(
            execution_kind="real-cuda",
            norm=selected["model.norm.weight"],
            head=selected[head_key],
            epsilon=metadata["rms_norm_eps"],
            decoded=inventory.decoded,
            excluded_ids=sorted(inventory.excluded),
            origin_files=origin,
        )


def _prepare_assets(capture_path, output_root, name, *, dependencies=None):
    root = Path(output_root).resolve()
    safe_name(name)
    capture_path = _inside(root, capture_path)
    source = json.loads(capture_path.read_text())
    if source.get("mode") == "bounded-endpoint-profile":
        from pilot_eval.trace_profile import load_profile, verify_profile

        config = load_profile(capture_path, root)
        verify_profile(capture_path, root)
        runtime = json.loads((root / config["run_path"] / "meta/runtime.json").read_text())
        config = {**config, "execution_kind": runtime["execution_kind"]}
    else:
        config = load_execution(capture_path, root)
    directory = root / "runs/pilot-5" / name / "readout-assets"
    with run_lock(directory):
        if (directory / "complete.json").exists():
            result = verify_assets(directory / "config.json", root)
            if any(
                result[k] != config[k]
                for k in [
                    "model",
                    "model_revision",
                    "tokenizer_revision",
                    "width",
                    "execution_kind",
                ]
            ):
                raise ValueError("Readout asset model/tokenizer identity changed; use a new name.")
            return directory / "config.json"
        _save_frozen(
            directory / "meta/run_manifest.json",
            dict(
                source_path=str(capture_path.relative_to(root)),
                source_sha256=file_hash(capture_path),
                implementation_sha256=file_hash(Path(__file__)),
            ),
        )
        _write_state(directory, "running", 0, 1)
        loaded = (dependencies or HFReadoutAssets()).load(config)
        if loaded["execution_kind"] != config["execution_kind"]:
            raise ValueError("Synthetic/real readout asset source mismatch.")
        norm = np.asarray(loaded["norm"])
        head = np.asarray(loaded["head"])
        if (
            norm.dtype != np.float32
            or head.dtype != np.float32
            or norm.shape != (config["width"],)
            or head.ndim != 2
            or head.shape[1] != config["width"]
            or not np.isfinite(norm).all()
            or not np.isfinite(head).all()
            or not np.isfinite(loaded["epsilon"])
            or loaded["epsilon"] <= 0
        ):
            raise ValueError("Invalid FP32 final norm/head asset.")
        decoded = {int(k): v for k, v in loaded["decoded"].items()}
        if not decoded or min(decoded) < 0 or max(decoded) >= len(head):
            raise ValueError("Tokenizer inventory outside output-head rows.")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "norm.npy", norm, allow_pickle=False)
        np.save(directory / "head.npy", head, allow_pickle=False)
        _write_json(
            directory / "inventory.json", dict(decoded=decoded, excluded_ids=loaded["excluded_ids"])
        )
        metadata = dict(
            schema_version=1,
            mode="pinned-readout-assets",
            model=config["model"],
            model_revision=config["model_revision"],
            tokenizer_revision=config["tokenizer_revision"],
            source_path=str(capture_path.relative_to(root)),
            source_sha256=file_hash(capture_path),
            width=config["width"],
            head_rows=len(head),
            epsilon=float(loaded["epsilon"]),
            execution_kind=config["execution_kind"],
            origin_files=loaded["origin_files"],
            compute_device="cpu",
            decoder_forwards=0,
            implementation_sha256=file_hash(Path(__file__)),
        )
        _save_frozen(directory / "config.json", metadata)
        _write_json(
            directory / "complete.json",
            dict(
                files={
                    n: file_hash(directory / n)
                    for n in ["config.json", "norm.npy", "head.npy", "inventory.json"]
                }
            ),
        )
        _write_state(directory, "completed", 1, 1)
        return directory / "config.json"


def prepare_assets(capture_path, output_root, name, *, dependencies=None):
    safe_name(name)
    directory = Path(output_root).resolve() / "runs/pilot-5" / name / "readout-assets"
    try:
        return _prepare_assets(capture_path, output_root, name, dependencies=dependencies)
    except BaseException as exc:
        _write_state(directory, "failed", 0, 1, str(exc))
        raise


def verify_assets(path, output_root):
    root = Path(output_root).resolve()
    path = _inside(root, path)
    directory = path.parent
    metadata = json.loads(path.read_text())
    marker = json.loads((directory / "complete.json").read_text())
    if path.name != "config.json" or set(marker["files"]) != {
        "config.json",
        "norm.npy",
        "head.npy",
        "inventory.json",
    }:
        raise ValueError("Readout asset seal omits required inventory.")
    for name, digest in marker["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError("Readout asset hash mismatch.")
    if metadata["implementation_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Readout asset implementation changed.")
    if file_hash(_inside(root, metadata["source_path"])) != metadata["source_sha256"]:
        raise ValueError("Readout capture source hash mismatch.")
    return metadata
