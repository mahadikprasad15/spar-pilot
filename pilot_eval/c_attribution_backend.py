"""The C-attribution model backend (ticket 09): local files only, FP32, eager attention, explicit device.

Everything is resolved through the sealed input bundle (ticket 08): the adapters, the pinned tokenizer files and
``generation_config.json`` by role under the restored artifact root, the base model from ``inputs.model_dir``
(downloaded at the pinned revision, its files checked against the bundle in preflight). The same class runs the
CPU whole-chain test on a tiny saved Qwen2 (ticket 10); only the CUDA memory counters, the volume mount and the
Hub check are injectable, as ``probes``.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def cuda_libraries():
    """Installed ``nvidia-*`` pip distributions (the CUDA libraries torch's wheel pins) and their versions."""
    from importlib import metadata
    names = {}
    for dist in metadata.distributions():
        name = (dist.metadata['Name'] or '').lower()
        if name.startswith('nvidia-') and name != 'nvidia-ml-py':
            names[name] = dist.version
    return dict(sorted(names.items()))


def nvidia_driver():
    try:
        import pynvml
        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:  # no GPU or no NVML: recorded as unknown
        return None


class TransformersBackend:
    def __init__(self, record, volume_root, *, device='cuda', probes=None, repo_root=REPO):
        self.record, self.volume_root, self.device = record, Path(volume_root), device
        self.inputs = record['inputs']
        self.root = Path(self.inputs['artifact_root'])
        self.bundle = json.loads(Path(self.inputs['bundle']).read_text())
        self.model_dir = Path(self.inputs['model_dir'])
        self.repo_root = Path(repo_root)  # the code checkout whose catalog/ the bundle's revision must equal
        self.probes = dict(probes or {})
        from pilot_eval.c_attribution_bundle import PINNED_BASE_MODEL
        # HF's identities for the pinned revision; only the CPU world substitutes its own (a test stand-in).
        self.pinned_base_model = self.probes.get('pinned_base_model', PINNED_BASE_MODEL)
        tokenizer_file = self._path('tokenizer:tokenizer.json')
        self.generation_config_path = self._path('generation-config')
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_file.parent, local_files_only=True)
        self.eos_token_id = int(record['generation']['eos_token_id'])
        self.pad_token_id = self._tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError('the pinned tokenizer has no pad token')
        # IEEE FP32 for every backend (CUDA matmul, cuDNN, mkldnn) plus 'highest', set before any model work and
        # read back; preflight reads it again and refuses a change (review 06-10, P1; ADR 0021, neuron audit).
        from pilot_eval.neuron_audit_collector import set_ieee_precision
        self.precision = set_ieee_precision()

    def _entry(self, role):
        entry = next((e for e in self.bundle['files'] if e['role'] == role), None)
        if entry is None:
            raise ValueError(f'the input bundle has no {role}')
        return entry

    def _path(self, role):
        return self.root / self._entry(role)['path']

    def adapter_dir(self, arm):
        return self._path(f"{self.inputs['sources'][arm]['adapter_role']}:adapter_model.safetensors").parent

    # --- identity ---------------------------------------------------------------------------------

    def tokenizer_identity(self):
        from pilot_eval.c_attribution_bundle import MODEL_REVISION
        from pilot_eval.workflow import _hash
        return {'revision': MODEL_REVISION, 'chat_template_sha256': _hash(self._tokenizer.chat_template)}

    def hf_tokenizer(self):
        return self._tokenizer

    def encode(self, prompt):
        return self._tokenizer(prompt, add_special_tokens=False)['input_ids']

    def decode(self, ids):
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def runtime(self):
        import peft
        import torch
        import transformers
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        return dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                    python=sys.version.split()[0], dtype='float32', attention='eager', device=self.device,
                    device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                    # the CUDA runtime, cuDNN and the pip-installed CUDA libraries torch brings (cuBLAS, cuDNN, ...):
                    # they choose kernels, so a resume on another Pod must match them (user, 2026-11-20)
                    cuda_runtime=torch.version.cuda, cudnn=cudnn, cuda_libraries=cuda_libraries(),
                    tokenizer=self.tokenizer_identity(), autocast=False, precision=self.precision_readback())

    def host(self):
        """Facts of the machine, recorded but not part of the resume fingerprint: the NVIDIA driver belongs to the
        RunPod host and cannot be installed from the Pod; it rarely changes the numbers (user, 2026-11-20)."""
        probe = self.probes.get('driver')
        return {'driver': probe() if probe else nvidia_driver()}

    def precision_readback(self):
        from pilot_eval.neuron_audit_collector import _legacy_matmul_precision, precision_backends
        try:
            legacy = _legacy_matmul_precision()
        except RuntimeError as exc:  # torch refuses the read when the backend settings disagree
            legacy = f'unreadable ({str(exc)[:80]})'
        return dict(precision_backends(), matmul_precision=legacy)

    def precision_problems(self):
        observed = self.precision_readback()
        return [f'IEEE FP32 precision not in effect: {k}={v}' for k, v in sorted(observed.items())
                if v != ('highest' if k == 'matmul_precision' else 'ieee')]

    # --- preflight ----------------------------------------------------------------------------------

    def preflight(self):
        """Disk, CUDA, HF_HOME off the volume, offline worker, volume identity on its own mount, private repo."""
        problems = self.precision_problems()
        disk = self.record['disk']
        work = Path(self.inputs['work_dir'])
        usage = self.probes.get('disk_usage', shutil.disk_usage)(work)
        if usage.free < disk['min_free_gb'] * 1e9 or usage.total < 0.95 * disk['container_total_gb'] * 1e9:
            problems.append(f"container disk {usage.total / 1e9:.0f} GB with {usage.free / 1e9:.0f} GB free; need a "
                            f"{disk['container_total_gb']} GB disk with >= {disk['min_free_gb']} GB free")
        cuda = self.probes.get('cuda_available')
        if cuda is None:
            import torch
            cuda = torch.cuda.is_available()
        if not cuda:
            problems.append('CUDA unavailable')
        hf_home = os.environ.get('HF_HOME')
        if not hf_home:
            problems.append('HF_HOME is not set (it must point at the container disk, off the volume)')
        elif Path(hf_home).resolve().is_relative_to(self.volume_root.resolve()):
            problems.append(f'HF_HOME {hf_home} is on the volume; put the model cache on the container disk')
        if os.environ.get('HF_HUB_OFFLINE') != '1':
            problems.append('worker is not offline (HF_HUB_OFFLINE=1 required; exports run in a child process)')
        volume_check = self.probes.get('volume_identity')
        if volume_check is None:
            from pilot_eval.neuron_audit_preflight import volume_identity
            _, volume_problems = volume_identity(self.volume_root, expected_id=self.record['volume_identity'],
                                                 local_dir=work)
        else:
            volume_problems = volume_check()
        problems += volume_problems
        hub_check = self.probes.get('private_repo')
        try:
            if hub_check is None:
                from pilot_eval.neuron_audit_export import OnlineHubProcess
                from pilot_eval.sampling_sft_hf import private
                private(OnlineHubProcess(timeout=120), self.record['export_repo'])
            else:
                hub_check(self.record['export_repo'])
        except Exception as exc:  # reported; the chain stops before any GPU work
            problems.append(f'export repo check: {type(exc).__name__}: {exc}')
        if not (self.repo_root / 'catalog').is_dir():
            problems.append('missing catalog: the code checkout has no catalog/')
        return problems

    # --- models -------------------------------------------------------------------------------------

    def _base(self, dtype):
        import torch
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=getattr(torch, dtype),
                                                    attn_implementation='eager', local_files_only=True)

    def load(self, arm, dtype='float32'):
        """The arm's PEFT model; refuses an adapter file that differs from the bundle or a live/saved mismatch."""
        import torch
        from peft import PeftModel
        from pilot_eval.neuron_audit_collector import check_live_adapter
        directory = self.adapter_dir(arm)
        role = f"{self.inputs['sources'][arm]['adapter_role']}:adapter_model.safetensors"
        actual = hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest()
        if actual != self._entry(role)['sha256']:
            raise ValueError(f'{arm}: adapter file differs from the sealed bundle ({actual})')
        model = PeftModel.from_pretrained(self._base('float32').to(self.device), directory, is_trainable=False,
                                          torch_device=self.device)
        config = json.loads((directory / 'adapter_config.json').read_text())
        live = check_live_adapter(model, directory, expected_scale=config['lora_alpha'] / config['r'])
        if live['status'] != 'pass':
            raise ValueError(f"{arm}: live adapter check failed: {live['reasons']}")
        model = model.to(getattr(torch, dtype))
        for p in model.parameters():
            p.requires_grad_(False)
        return model.eval()

    def reference(self, arm):
        """FP64 copy for the finite-difference reference (used inside ``full_fp64``)."""
        return self.load(arm, dtype='float64')

    def release(self, model=None):
        """Collect and empty the CUDA cache. The caller must drop its own references first (``del model``):
        deleting this parameter does not free the caller's object (review 06-10, P1)."""
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def adapter_norms(self, arm, adapters):
        """‖b‖ and the scale alpha/r per adapter, read from the sealed safetensors (no model load)."""
        from safetensors.numpy import load_file
        directory = self.adapter_dir(arm)
        config = json.loads((directory / 'adapter_config.json').read_text())
        saved = load_file(str(directory / 'adapter_model.safetensors'))
        norms = []
        for layer, projection in adapters:
            group = 'self_attn' if projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
            key = f'base_model.model.model.layers.{layer}.{group}.{projection}.lora_B.weight'
            import numpy as np
            norms.append(float(np.linalg.norm(saved[key].astype(np.float64))))
        return norms, [config['lora_alpha'] / config['r']] * len(adapters)

    # --- memory ---------------------------------------------------------------------------------------

    def reset_peak(self):
        if 'reset_peak' in self.probes:
            return self.probes['reset_peak']()
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        if 'peak_memory' in self.probes:
            return self.probes['peak_memory']()
        import torch
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    def total_memory(self):
        if 'total_memory' in self.probes:
            return self.probes['total_memory']()
        import torch
        return torch.cuda.get_device_properties(0).total_memory
