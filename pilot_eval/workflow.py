"""Freeze a protocol plan, then execute its cells through external HF boundaries."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path

from pilot_eval.backend import load_hf_backend
from pilot_eval.config import validate_config
from pilot_eval.protocol import (
    build_gsm8k_items, build_mmlu_items, persist_cohort,
    select_gsm8k_indices, select_mmlu_indices,
)
from pilot_eval.run import run_evaluation


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _save_frozen(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def adapter_digest(path):
    """Hash the contents and relative names of all files in a local adapter."""
    path = Path(path)
    if not (path / "adapter_config.json").is_file():
        raise ValueError("local adapter must contain adapter_config.json")
    digest = hashlib.sha256()
    files = sorted(file for file in path.rglob("*") if file.is_file())
    for file in files:
        digest.update(str(file.relative_to(path)).encode() + b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class HFDependencies:
    def resolve(self, repo, kind):
        from huggingface_hub import HfApi
        api = HfApi()
        return (api.model_info(repo) if kind == "model" else api.dataset_info(repo)).sha

    def load_tokenizer(self, model, revision):
        from transformers import AutoConfig, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        context = AutoConfig.from_pretrained(model, revision=revision).max_position_embeddings
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, min(context, tokenizer.model_max_length)

    def load_dataset(self, path, name, revision):
        from datasets import load_dataset
        return load_dataset(path, name, revision=revision)

    def runtime(self, config):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("pilot 1 requires exactly one visible CUDA GPU")
        if config["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU lacks bf16 support; prepare a separate float16 plan")
        versions = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "datasets", "peft", "accelerate", "huggingface-hub")}
        from transformers import GenerationConfig
        resolved = GenerationConfig(**config["decoding"], repetition_penalty=1.0, no_repeat_ngram_size=0).to_dict()
        return {"versions": versions, "python": sys.version, "platform": platform.platform(),
                "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "deterministic_algorithms": True, "resolved_generation": resolved}

    def load_backend(self, config):
        return load_hf_backend(config)


def prepare_plan(output_root, plan_name, *, model="Qwen/Qwen2.5-1.5B-Instruct",
                 dtype="bfloat16", batch_size=4, adapter=None, dependencies=None):
    """Prepare five pinned cells without loading model weights."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan_name):
        raise ValueError("plan name must contain only letters, digits, underscores and hyphens")
    if dtype not in ("bfloat16", "float16") or batch_size < 1:
        raise ValueError("invalid precision or batch size")
    root = Path(output_root).resolve()
    deps = dependencies or HFDependencies()
    plan_dir = root / "plans" / plan_name
    plan_manifest = plan_dir / "manifest.json"
    options = dict(model=model, dtype=dtype, batch_size=batch_size, adapter=adapter)
    if plan_manifest.exists():
        saved = json.loads(plan_manifest.read_text())
        if saved["options"] != options:
            raise ValueError("plan options mismatch; choose a new plan name")
        return [root / path for path in saved["configs"]]
    # Partial preparations reuse already-resolved immutable commits.
    pins_path = plan_dir / "pins.json"
    if pins_path.exists():
        pins = json.loads(pins_path.read_text())
        if pins["options"] != options:
            raise ValueError("partial plan options mismatch")
    else:
        pins = {"options": options, "model": deps.resolve(model, "model"),
                "gsm8k": deps.resolve("openai/gsm8k", "dataset"),
                "mmlu": deps.resolve("cais/mmlu", "dataset"), "adapter_revision": None}
        if adapter:
            if Path(adapter).is_dir():
                pins["adapter_sha256"] = adapter_digest(adapter)
            else:
                pins["adapter_revision"] = deps.resolve(adapter, "model")
        _save_frozen(pins_path, pins)
    tokenizer, context = deps.load_tokenizer(model, pins["model"])
    gsm = deps.load_dataset("openai/gsm8k", "main", pins["gsm8k"])
    mmlu = deps.load_dataset("cais/mmlu", "all", pins["mmlu"])
    grouped = {}
    for split in ("test", "dev"):
        grouped[split] = {}
        for row in mmlu[split]:
            grouped[split].setdefault(row["subject"], []).append(dict(row))
    if len(grouped["test"]) != 57:
        raise ValueError("MMLU must contain exactly 57 test subjects")
    gsm_indices = select_gsm8k_indices(len(gsm["test"]))
    mmlu_indices = select_mmlu_indices({subject: len(rows) for subject, rows in grouped["test"].items()})
    manifests = {}
    for dataset, indices, name in (("gsm8k", gsm_indices, "test-150-seed-42"),
                                  ("mmlu", mmlu_indices, "balanced-1140-seed-42")):
        manifest = persist_cohort(root, dataset=dataset, revision=pins[dataset], name=name, indices=indices, seed=42)
        source = [dict(row) for row in gsm["test"]] if dataset == "gsm8k" else grouped
        manifests[dataset] = {**manifest, "evaluation_split": "test", "source_sha256": _hash(source),
                              "item_ids": ([f"gsm8k:test:{i}" for i in indices] if dataset == "gsm8k" else
                                           [f"mmlu:{s}:{i}" for s in sorted(indices) for i in indices[s]])}
        _save_frozen(root / "cohorts" / dataset / pins[dataset] / name / "source.json", manifests[dataset])
    paths = []
    for scorer, shots in (("gsm8k", 0), ("mmlu_text", 0), ("mmlu_text", 5), ("mmlu_logits", 0), ("mmlu_logits", 5)):
        dataset = "gsm8k" if scorer == "gsm8k" else "mmlu"
        cell = f"{scorer}-{shots}shot"
        items = (build_gsm8k_items(gsm["test"], gsm_indices, tokenizer) if dataset == "gsm8k" else
                 build_mmlu_items(grouped["test"], grouped["dev"], mmlu_indices, shots, tokenizer, scorer == "mmlu_logits"))
        maximum = 1024 if dataset == "gsm8k" else 32
        for item in items:
            item["prompt_tokens"] = len(tokenizer.encode(item["prompt"], add_special_tokens=False))
            item["source_split"] = "test"
            item["dataset_revision"] = pins[dataset]
            item["dev_source_indices"] = list(range(shots))
            if item["prompt_tokens"] + (1 if scorer == "mmlu_logits" else maximum) > context:
                raise ValueError(f"context overflow: {item['id']} ({item['prompt_tokens']} prompt tokens, limit {context})")
        items_path = plan_dir / f"{cell}.items.json"
        _save_frozen(items_path, items)
        config = {
            "run_id": f"{plan_name}-{cell}", "experiment": "pilot-1", "protocol_version": "v1",
            "model": model, "model_revision": pins["model"], "tokenizer_revision": pins["model"],
            "adapter": str(Path(adapter).resolve()) if adapter and Path(adapter).is_dir() else adapter,
            "adapter_revision": pins["adapter_revision"], "adapter_sha256": pins.get("adapter_sha256"),
            "dataset": dataset, "dataset_path": "openai/gsm8k" if dataset == "gsm8k" else "cais/mmlu",
            "dataset_config": "main" if dataset == "gsm8k" else "all", "dataset_revision": pins[dataset],
            "evaluation_split": "test", "fewshot_split": "dev" if shots else None,
            "cohort": manifests[dataset]["name"], "cohort_manifest": manifests[dataset],
            "prompt_indices": gsm_indices if dataset == "gsm8k" else mmlu_indices,
            "shots": shots, "prompt_template": f"{dataset}-v1", "seed": 42, "scorer": scorer,
            "decoding": {"do_sample": False, "max_new_tokens": maximum, "num_beams": 1,
                         "num_return_sequences": 1, "eos_token_id": tokenizer.eos_token_id,
                         "pad_token_id": tokenizer.pad_token_id},
            "batch_size": batch_size, "checkpoint_interval_batches": 1, "dtype": dtype,
            "attention_implementation": "eager", "quantization": None, "deterministic": True,
            "variant": f"{'adapter' if adapter else 'baseline'}-{dtype}-{cell}",
            "context_limit": context, "chat_template_sha256": _hash(tokenizer.chat_template),
            "items_path": str(items_path.relative_to(root)), "items_sha256": _hash(items),
        }
        validate_config(config)
        path = plan_dir / f"{cell}.config.json"
        _save_frozen(path, config)
        paths.append(path)
    _save_frozen(plan_manifest, {"options": options, "configs": [str(path.relative_to(root)) for path in paths]})
    return paths


def execute_config(config_path, output_root, *, audit_items=None, dependencies=None):
    """Run frozen inputs, or a separately named fixed-prefix real-item audit."""
    root = Path(output_root).resolve()
    config = validate_config(json.loads(Path(config_path).read_text()))
    items_path = (root / config["items_path"]).resolve()
    if not items_path.is_relative_to(root):
        raise ValueError("items path must be within artifact root")
    items = json.loads(items_path.read_text())
    if _hash(items) != config["items_sha256"]:
        raise ValueError("input hash mismatch")
    if config.get("adapter_sha256") and adapter_digest(config["adapter"]) != config["adapter_sha256"]:
        raise ValueError("local adapter hash mismatch")
    if audit_items is not None:
        if not 1 <= audit_items <= len(items):
            raise ValueError("audit size must be between one and cohort size")
        items = items[:audit_items]
        config = {**config, "run_id": config["run_id"] + f"-audit-{audit_items}",
                  "audit_items": audit_items, "items_sha256": _hash(items),
                  "prompt_indices": [{"id": item["id"], "source_index": item["source_index"]} for item in items]}
    deps = dependencies or HFDependencies()
    config = {**config, "runtime": deps.runtime(config)}

    class LazyBackend:
        backend = None

        def _get(self):
            if self.backend is None:
                self.backend = deps.load_backend(config)
            return self.backend

        def generate_batch(self, *args):
            return self._get().generate_batch(*args)

        def choice_logits_batch(self, *args):
            return self._get().choice_logits_batch(*args)

    return run_evaluation(config, items, LazyBackend(), root)
