"""Ticket 11: real model backend for the attribution chain (Colab / Pod; CUDA, local files only).

Reuses the joint audit's ``neuron_audit_qualify.TransformersBackend`` loader (eager attention, explicit dtype,
local files), ``ChatTokenizer`` (pinned files, saved chat template), ``set_ieee_precision`` and
``check_live_adapter``. All weights FP32 and frozen; TF32 off; one arm model resident at a time.

Launch-record ``inputs`` block:

    artifact_root, inventory_dir, sequences_dir, model_dir, tokenizer_dir, gsm8k_test_parquet,
    adapter_dirs: {arm: dir}, max_new_tokens: 1024, work_dir,
    stage_d.pilot_results: {arm: {path, filters}}   # pilot responses, re-scored with flexible v3

The 20 rehearsal items are ``rehearsal_draw`` (seed 42, outside the ADR 0001 cohort; user 2026-11-14).

The ADR 0001 cohort is ``protocol.select_gsm8k_indices(len(test), 150, 42)``. Edited checkpoints are verified
by hash and by *bit-identical* logits against the g = 0 gate (a zeroed b adds exactly the 0 the gate adds).
"""
import json
from pathlib import Path

from pilot_eval import attribution_ablation as ab
from pilot_eval import attribution_traces as at
from pilot_eval.protocol import gsm8k_messages, select_gsm8k_indices


def cohort_indices(total):
    return select_gsm8k_indices(total, 150, 42)


def rehearsal_draw(total, *, count=20, seed=42):
    """The 20 rehearsal items (user 2026-11-14): a seed-42 draw from the test indices outside ADR 0001, by the
    cohort's own mechanism: sorted(random.Random(42).sample(non-cohort indices, 20))."""
    import random
    cohort = set(cohort_indices(total))
    return sorted(random.Random(seed).sample([i for i in range(total) if i not in cohort], count))


def rehearsal_indices(ids, total, *, count=20):
    ids = [int(i) for i in ids or []]
    cohort = set(cohort_indices(total))
    if len(ids) != count or len(set(ids)) != count or any(not 0 <= i < total for i in ids):
        raise ValueError(f'rehearsal_generation_ids must list {count} distinct GSM8K test indices')
    if cohort & set(ids):
        raise ValueError(f'rehearsal items overlap the ADR 0001 cohort: {sorted(cohort & set(ids))}')
    return ids


class TransformersBackend:
    def __init__(self, record, volume_root, *, device='cuda'):
        from pilot_eval.neuron_audit_collector import set_ieee_precision
        from pilot_eval.neuron_audit_qualify import TransformersBackend as Loader
        from pilot_eval.neuron_audit_sequences import ChatTokenizer, _inside, _read_parquet
        self.record, self.volume_root, self.device = record, Path(volume_root), device
        self.inputs = inputs = record['inputs']
        self.root = Path(inputs['artifact_root']).resolve()
        self.path = lambda value: _inside(self.root, value)  # noqa: E731
        self.adapter_dirs = {arm: self.path(d) for arm, d in inputs['adapter_dirs'].items()}
        missing = sorted(set(record['expected_arms']) - set(self.adapter_dirs))
        if missing:
            raise ValueError(f'no adapter directory for arms {missing}')
        # The base model snapshot is hash-checked at download and may live in the HF cache outside the root.
        model_dir = Path(inputs['model_dir']) if Path(inputs['model_dir']).is_absolute() else self.path(inputs['model_dir'])
        self.loader = Loader(model_dir, self.adapter_dirs, self.path(inputs['tokenizer_dir']),
                             record['expected_arms'][0], device=device)
        self.tokenizer = ChatTokenizer(self.path(inputs['tokenizer_dir']))
        if self.tokenizer.pad_token_id is None:
            raise ValueError('the pinned tokenizer has no pad token')
        self.precision = set_ieee_precision()
        self.test_rows = _read_parquet(self.path(inputs['gsm8k_test_parquet']))
        self._rehearsal = rehearsal_indices(rehearsal_draw(len(self.test_rows),
                                                           count=record['generation_batch_rule']['items']),
                                            len(self.test_rows), count=record['generation_batch_rule']['items'])
        self._sequences = None
        self._current, self._untuned = (None, None), None

    # --- identity and runtime -----------------------------------------------------------------

    def runtime(self):
        import torch
        return dict(model=str(self.inputs['model_dir']), torch=torch.__version__, dtype='float32', tf32=False,
                    autocast=False, attention='eager', precision=self.precision,
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                    tokenizer=self.tokenizer.identity())

    def decoder(self):
        # The pilot decoder passes only tokenizer.eos_token_id; pad is padding, not a stop.
        return dict(eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)

    def token_text(self, token_id):
        return self.tokenizer.token_text(token_id)

    def decode(self, ids):
        return self.tokenizer.tokenizer.decode(list(ids), skip_special_tokens=True)

    def decoding(self):
        return dict(mode='greedy', max_new_tokens=self.inputs['max_new_tokens'], precision='fp32',
                    padding='left', eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.0, path='hf')

    def sources(self):
        return at.verify_sources(self.path(self.inputs['inventory_dir']), self.path(self.inputs['sequences_dir']),
                                 self.root)

    def unresolved_arms(self):
        return sorted(a for a, r in self.sources()['provenance'].items() if r.get('unresolved_modules'))

    def preflight(self):
        import os
        import torch
        from pilot_eval.neuron_audit_export import OnlineHubProcess
        from pilot_eval.neuron_audit_pod import disk_problems
        from pilot_eval.sampling_sft_hf import private
        problems = list(disk_problems(self.inputs['work_dir']))
        if not torch.cuda.is_available():
            problems.append('CUDA unavailable')
        if os.environ.get('HF_HUB_OFFLINE') != '1':
            problems.append('worker is not offline (HF_HUB_OFFLINE=1 required; exports run in a child process)')
        identity = self.volume_root / 'volume_identity.json'
        if not identity.is_file() or json.loads(identity.read_text()).get('identity') != self.record['volume_identity']:
            problems.append('persistent volume identity does not match the launch record')
        try:
            self.sources()
        except ValueError as exc:
            problems.append(str(exc))
        try:
            private(OnlineHubProcess(timeout=120), self.record['export_repo'])
        except Exception as exc:  # reported, then the chain stops before production
            problems.append(f'export check: {type(exc).__name__}: {exc}')
        return problems

    # --- models ---------------------------------------------------------------------------------

    def _freeze(self, model):
        for p in model.parameters():
            p.requires_grad_(False)
        return model.eval()

    def load(self, arm):
        from pilot_eval.neuron_audit_collector import check_live_adapter
        if self._current[0] == arm:
            return self._current[1]
        if self._current[1] is not None:
            self.loader.release(self._current[1])
            self._current = (None, None)
        model = self._freeze(self.loader.load(arm, 'float32'))
        config = json.loads((self.adapter_dirs[arm] / 'adapter_config.json').read_text())
        live = check_live_adapter(model, self.adapter_dirs[arm], expected_scale=config['lora_alpha'] / config['r'])
        if live['status'] != 'pass':
            raise ValueError(f'{arm}: live adapter check failed: {live["reasons"]}')
        self._current = (arm, model)
        return model

    def untuned(self):
        if self._untuned is None:
            self._untuned = self._freeze(self.loader.fresh_base('float32'))
        return self._untuned

    def untuned_generator(self, size):
        return at.HFGenerator(self.untuned(), pad_token_id=self.tokenizer.pad_token_id,
                              eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.inputs['max_new_tokens'],
                              decode=self.decode)

    def stage_d_generator(self, arm, adapters, size, edit_dir, *, probe_ids):
        """Generator for one Stage D run; an edited checkpoint is checked against the gate on ``probe_ids``,
        an allowed prompt supplied by the caller (a rehearsal item in rehearsals, never a blind cohort item)."""
        import torch
        from peft import PeftModel
        if arm is None:
            return self.untuned_generator(size), None
        model = self.load(arm)
        edit = None
        if adapters:
            names = [ab.adapter_modules(model)[i][:2] for i in adapters]
            edit = ab.edit_checkpoint(self.adapter_dirs[arm], edit_dir, names)
            ab.verify_edit(edit_dir, edit)
            edited = self._freeze(PeftModel.from_pretrained(self.loader.fresh_base('float32'), edit_dir,
                                                            is_trainable=False, torch_device=self.device))
            probe = torch.tensor([list(probe_ids)], device=self.device)
            with torch.no_grad():
                with ab.removal(model, adapters):
                    gated = model(input_ids=probe, use_cache=False).logits
                direct = edited(input_ids=probe, use_cache=False).logits
            if not torch.equal(gated, direct):
                raise ValueError(f'{arm}: edited checkpoint logits differ from the gate (max '
                                 f'{float((gated - direct).abs().max())})')
            edit['logits_equal_gate'] = True
            model = edited
        return at.HFGenerator(model, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                              max_new_tokens=self.inputs['max_new_tokens'], decode=self.decode), edit

    def parameter_count(self):
        """N in the FLOP model: base-model parameters (the rank-1 adapters are negligible)."""
        return sum(p.numel() for n, p in self.untuned().named_parameters())

    def synchronize(self):
        self.loader.synchronize()

    def device_memory(self):
        import torch
        return torch.cuda.get_device_properties(0).total_memory

    def reset_peak(self):
        self.loader.reset_peak()

    def peak_memory(self):
        return self.loader.peak_memory()

    def is_oom(self, exc):
        return self.loader.is_oom(exc)

    # --- prompts and gold ------------------------------------------------------------------------------

    def _sequence_rows(self):
        if self._sequences is None:
            from pilot_eval.neuron_audit_qualify import read_sequences
            self._sequences = read_sequences(self.path(self.inputs['sequences_dir']))
        return self._sequences

    def prompt_rows(self, split):
        return at.prompt_rows(self._sequence_rows(), split)

    def _test_items(self, indices):
        out = []
        for index in indices:
            text = self.tokenizer.render(gsm8k_messages(self.test_rows[index]['question']), add_generation_prompt=True)
            ids, _ = self.tokenizer.encode(text)
            out.append(dict(id=f'gsm8k:test:{index}', prompt_ids=ids, source_index=index))
        return out

    def rehearsal_items(self):
        return self._test_items(self._rehearsal)

    def cohort_items(self):
        return self._test_items(cohort_indices(len(self.test_rows)))

    def gold(self, split):
        if split == 'rehearsal':
            return {f'gsm8k:test:{i}': self.test_rows[i]['answer'] for i in self._rehearsal}
        if split == 'cohort':
            return {f'gsm8k:test:{i}': self.test_rows[i]['answer'] for i in cohort_indices(len(self.test_rows))}
        return at.load_gold(self.path(self.inputs['sequences_dir']), self.root, self.prompt_rows(split))
