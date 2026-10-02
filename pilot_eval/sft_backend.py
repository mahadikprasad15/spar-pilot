"""Pinned TRL/PEFT training boundary. Optional libraries load on use only."""

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from pathlib import Path
from collections.abc import Sized
from typing import cast

from pilot_eval.run import _write_json, _write_state
from pilot_eval.training import seal_checkpoint
from pilot_eval.workflow import HFDependencies


PINS = {'torch': '2.9.0', 'transformers': '4.57.6', 'trl': '0.26.2',
        'peft': '0.18.0', 'datasets': '4.4.1', 'accelerate': '1.12.0',
        'huggingface-hub': '0.36.0'}


def frozen_weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        if 'lora_' not in name:
            digest.update(name.encode() + b'\0')
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def completion_loss(outputs, labels, num_items_in_batch=None):
    """Summed shifted CE / total supervised tokens in the accumulation window."""
    import torch
    shifted = labels[..., 1:].contiguous()
    denominator = num_items_in_batch if num_items_in_batch is not None else (shifted != -100).sum()
    if int(denominator) <= 0:
        raise ValueError('no supervised completion tokens')
    loss = torch.nn.functional.cross_entropy(
        outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.shape[-1]),
        shifted.view(-1), ignore_index=-100, reduction='sum') / denominator
    if not torch.isfinite(loss):
        raise ValueError('nonfinite training loss')
    return loss


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(row['input_ids']) for row in rows)
        inputs, labels, masks = [], [], []
        for row in rows:
            size = len(row['input_ids'])
            inputs.append(row['input_ids'] + [self.pad_token_id] * (width - size))
            labels.append(row['labels'] + [-100] * (width - size))
            masks.append([1] * size + [0] * (width - size))
        return dict(input_ids=torch.tensor(inputs), labels=torch.tensor(labels),
                    attention_mask=torch.tensor(masks), example_ids=[row['id'] for row in rows])


class HFTrainingEngine:
    """Real model boundary; supplied tiny models allow offline CPU integration."""

    def __init__(self, config, rows, *, model=None, tokenizer=None):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, enable_full_determinism
        self.config, self.rows = config, rows
        enable_full_determinism(config['seed'])
        if model is None:
            self.runtime(config)  # Check pinned environment before loading weights.
            tokenizer, _ = HFDependencies().load_tokenizer(config['model'], config['tokenizer_revision'])
            model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['model_revision'],
                dtype=torch.float32, attn_implementation='eager', device_map={'': 0})
        self.tokenizer = tokenizer
        targets = sorted(name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.split('.')[-1] in config['adapter']['projections'])
        expected = {f'model.layers.{layer}.{group}.{projection}'
            for layer in range(28) for group, projections in [
                ('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                ('mlp', ['gate_proj', 'up_proj', 'down_proj'])] for projection in projections}
        if set(targets) != expected:
            raise ValueError('adapter target modules must be the intended 196 projections')
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0,
            target_modules=targets, bias='none', task_type='CAUSAL_LM', init_lora_weights=True))
        if any(p.requires_grad != ('lora_' in name) for name, p in model.named_parameters()):
            raise ValueError('unexpected trainable/frozen parameters')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('all weights must be FP32')
        model.enable_input_require_grads()
        self.model, self.targets = model, targets
        self.base_hash = frozen_weight_hash(model)
        self.order = list(range(len(rows)))
        random.Random(config['seed']).shuffle(self.order)
        self.seen = []

    @staticmethod
    def runtime(config):
        versions = {name: importlib.metadata.version(name) for name in PINS}
        if any(versions[name].split('+')[0] != version for name, version in PINS.items()):
            raise ValueError(f'Pilot 2 requires pinned training dependencies: {PINS}; actual: {versions}')
        runtime_config = {**config, 'decoding': config.get('decoding', config.get('source_config', {}).get('decoding'))}
        runtime = HFDependencies().runtime(runtime_config)
        import torch
        hardware = config.get('hardware', 'T4')
        if hardware not in ('T4', 'L4') or hardware not in runtime['device']:
            raise ValueError(f'this protocol requires the selected {hardware}; record a variant for another GPU')
        runtime['versions'] = versions
        runtime['compute_capability'] = list(torch.cuda.get_device_capability(0))
        try:
            runtime['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            raise ValueError('repository commit unavailable')
        return runtime

    def preflight(self, directory):
        import torch
        from transformers import enable_full_determinism
        model = self.model
        longest = max(self.rows, key=lambda row: len(row['input_ids']))
        inputs = torch.tensor([longest['input_ids'][:min(8, longest['prompt_tokens'])]], device=model.device)
        model.eval()
        with torch.no_grad():
            adapted = model(input_ids=inputs, use_cache=False).logits
            with model.disable_adapter():
                untuned = model(input_ids=inputs, use_cache=False).logits
        zero = torch.equal(adapted, untuned) and all(
            torch.count_nonzero(p).item() == 0 for name, p in model.named_parameters() if 'lora_B' in name)
        if not zero:
            raise ValueError('initial adapter does not have zero write')
        del adapted, untuned
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        collator = CompletionCollator(self.tokenizer.pad_token_id if self.tokenizer else 0)
        denominator = 8 * sum(v != -100 for v in longest['labels'][1:])
        loss_value = 0.0
        for _ in range(8):
            batch = collator([longest])
            batch.pop('example_ids')
            labels = batch.pop('labels').to(model.device)
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()}, use_cache=False)
            loss = completion_loss(outputs, labels, denominator)
            loss_value += loss.item()
            loss.backward()
            del outputs, loss
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
        if not finite:
            raise ValueError('nonfinite/missing preflight gradients')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                      error_if_nonfinite=True)
        # Allocate real optimizer state in this disposable preflight, then restore
        # adapter parameters exactly so preflight cannot become a scientific update.
        trainable = [p for p in model.parameters() if p.requires_grad]
        originals = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0, foreach=False)
        optimizer.step()
        peak = torch.cuda.max_memory_allocated() if model.device.type == 'cuda' else None
        with torch.no_grad():
            for param, original in zip(trainable, originals):
                param.copy_(original)
        model.zero_grad(set_to_none=True)
        del optimizer, originals
        enable_full_determinism(self.config['seed'])
        evidence = dict(finite_loss=math.isfinite(loss_value), finite_gradients=finite,
                        zero_write=zero, target_count=len(self.targets), targets=self.targets,
                        base_sha256=self.base_hash, peak_memory_bytes=peak,
                        longest_sequence_tokens=len(longest['input_ids']), loss=loss_value)
        if frozen_weight_hash(model) != self.base_hash:
            raise ValueError('preflight changed base weights')
        return evidence

    def train(self, directory, resume):
        import torch
        from datasets import Dataset
        from torch.utils.data import SequentialSampler
        from transformers import Trainer, TrainerCallback
        from trl import SFTConfig, SFTTrainer
        engine = self
        directory = Path(directory)
        checkpoint_root = directory / 'checkpoints'
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        logs_path = directory / 'logs/steps.jsonl'
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        resume_step = int(resume.name.split('-')[-1]) if resume else 0
        if logs_path.exists():
            logs = [json.loads(line) for line in logs_path.read_text().splitlines()]
            logs = [log for log in logs if log['step'] <= resume_step]
            logs_path.write_text(''.join(json.dumps(log) + '\n' for log in logs))

        class OrderedTrainer(SFTTrainer):
            def _get_train_sampler(self, train_dataset=None):
                dataset = train_dataset if train_dataset is not None else self.train_dataset
                if dataset is None:
                    raise ValueError('training dataset is required')
                return SequentialSampler(cast(Sized, dataset))

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                engine.seen.extend(inputs.pop('example_ids'))
                # Use the explicit CE objective without TRL's optional full-vocab
                # entropy diagnostics, which materially increase peak memory.
                return Trainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError('nonfinite/missing training gradients')

            def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
                if getattr(optimizer, 'step_was_skipped', False):
                    raise ValueError('optimizer update was skipped')
                if any(not torch.isfinite(p).all() for p in engine.model.parameters() if p.requires_grad):
                    raise ValueError('nonfinite adapter weights after optimizer update')

            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                expected = [engine.rows[i]['id'] for i in engine.order[(step - 1) * 8:step * 8]]
                if engine.seen != expected:
                    raise ValueError('actual example exposures differ from frozen order')
                control.should_save = step in engine.config['checkpoint_steps']
                _write_state(directory, 'running', step, 64)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    if any(not math.isfinite(float(logs[k])) for k in ['loss', 'grad_norm']):
                        raise ValueError('nonfinite training log')
                    record = dict(step=state.global_step, loss=logs['loss'],
                                  learning_rate=logs['learning_rate'], grad_norm=logs['grad_norm'],
                                  example_ids=list(engine.seen))
                    with logs_path.open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                    engine.seen.clear()

            def on_save(self, args, state, control, **kwargs):
                seal_checkpoint(checkpoint_root / f'checkpoint-{state.global_step}', state.global_step)

        opt = self.config['optimizer']
        args = SFTConfig(output_dir=str(checkpoint_root), max_steps=64,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=opt['learning_rate'], lr_scheduler_type='constant', warmup_steps=0,
            optim=opt['name'], optim_args='foreach=False', adam_beta1=opt['beta1'],
            adam_beta2=opt['beta2'], adam_epsilon=opt['epsilon'], weight_decay=0.0,
            max_grad_norm=1.0, bf16=False, fp16=False, tf32=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            full_determinism=True, seed=42, data_seed=42, dataloader_num_workers=0,
            dataloader_pin_memory=False, dataloader_drop_last=False, group_by_length=False,
            remove_unused_columns=False, ignore_data_skip=False, logging_steps=1,
            logging_nan_inf_filter=False, save_strategy='steps', save_steps=8,
            save_only_model=False, report_to='none', packing=False,
            max_length=self.config['max_length'], completion_only_loss=True,
            dataset_kwargs={'skip_prepare_dataset': True}, shuffle_dataset=False,
            use_cpu=self.model.device.type == 'cpu')
        dataset = Dataset.from_list([dict(id=self.rows[i]['id'], input_ids=self.rows[i]['input_ids'],
            labels=self.rows[i]['labels']) for i in self.order])
        trainer = OrderedTrainer(model=self.model, args=args, train_dataset=dataset,
            processing_class=self.tokenizer, data_collator=CompletionCollator(
                self.tokenizer.pad_token_id if self.tokenizer else 0),
            compute_loss_func=completion_loss, callbacks=[Audit()])
        _write_json(directory / 'meta/resolved_trainer.json', args.to_dict())
        _write_json(directory / 'inputs/order.json', [self.rows[i]['id'] for i in self.order])
        if resume is None:
            trainer.create_optimizer_and_scheduler(64)
            zero = checkpoint_root / 'checkpoint-0'
            trainer.save_model(str(zero))
            trainer._save_optimizer_and_scheduler(str(zero))
            trainer._save_rng_state(str(zero))
            trainer.state.save_to_json(str(zero / 'trainer_state.json'))
            seal_checkpoint(zero, 0)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        records = [json.loads(line) for line in logs_path.read_text().splitlines()]
        if [r['step'] for r in records] != list(range(1, 65)):
            raise ValueError('missing or duplicated successful step logs')
        with (directory / 'logs/steps.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'learning_rate', 'grad_norm', 'example_ids'])
            writer.writeheader()
            writer.writerows(records)
        return dict(successful_steps=trainer.state.global_step,
                    example_exposures=sum(len(r['example_ids']) for r in records),
                    base_sha256_before=self.base_hash, base_sha256_after=frozen_weight_hash(self.model),
                    target_count=len(self.targets), checkpoint_steps=[0, 8, 16, 32, 64])

    def close(self):
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
