"""Fixed-pool input adapter for the existing GRPO training engine."""
from pilot_eval.grpo_baseline import _score
from pilot_eval.grpo_train_engine import GRPOTrainingEngine
from pilot_eval.training import file_hash
from pathlib import Path
import json


def verify_initial_adapter(directory, weights_sha256, config_sha256):
    directory = Path(directory)
    if (file_hash(directory / 'adapter_model.safetensors') != weights_sha256
            or file_hash(directory / 'adapter_config.json') != config_sha256):
        raise ValueError('checkpoint-0 adapter or configuration hash mismatch')


def pool_window(rows):
    """Validate and convert one already-verified pool window to online inputs."""
    if len(rows) != 8 or len({r['id'] for r in rows}) != 8:
        raise ValueError('eight distinct pool questions required')
    prompts, outputs, seen = [], [], set()
    for row in rows:
        prompts.append(row['input_ids'])
        if not row['input_ids'] or len(row['draws']) != 8:
            raise ValueError('complete pool group required')
        for index, draw in enumerate(row['draws']):
            if (draw['group_id'] != row['id'] or draw['draw_index'] != index
                    or draw['draw_id'] != f"{row['id']}:draw:{index}" or draw['draw_id'] in seen):
                raise ValueError('pool draw identity or order mismatch')
            seen.add(draw['draw_id'])
            tokens = draw['generated_token_ids']
            if not tokens or draw['completion_mask'] != [1] * len(tokens):
                raise ValueError('pool completion mask incompatible with online window')
            if draw['capped'] != (draw['stop_reason'] == 'cap'):
                raise ValueError('pool cap status mismatch')
            output = {key: draw[key] for key in ('text', 'token_ids', 'generated_token_ids', 'stop_reason')}
            scored = _score(row, output, index)
            if scored['reward'] != draw['reward_v3']:
                raise ValueError('pool reward differs from frozen scorer')
            outputs.append(output)
    return prompts, outputs


class OfflineGRPOTrainingEngine(GRPOTrainingEngine):
    """Use the online update unchanged, replacing only the draw source."""
    def __init__(self, plan, settings, directory, *, checkpoint_zero,
                 weights_sha256, config_sha256, sampler=None):
        verify_initial_adapter(checkpoint_zero, weights_sha256, config_sha256)
        saved = json.loads((Path(checkpoint_zero) / 'adapter_config.json').read_text())
        adapter = plan['adapter']
        if (saved.get('r') != adapter['rank'] or saved.get('lora_alpha') != adapter['alpha']
                or saved.get('lora_dropout') != adapter['dropout']):
            raise ValueError('checkpoint-0 adapter configuration differs from plan')
        targets = saved.get('target_modules')
        if not isinstance(targets, list) or {str(x).split('.')[-1] for x in targets} != set(adapter['projections']):
            raise ValueError('checkpoint-0 adapter targets differ from plan')
        super().__init__(plan, settings, directory, sampler=sampler)
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        weights = load_file(str(Path(checkpoint_zero) / 'adapter_model.safetensors'),
                            device=str(self.model.device))
        expected = get_peft_model_state_dict(self.model)
        if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
            raise ValueError('checkpoint-0 adapter target mapping mismatch')
        if any(self.torch.count_nonzero(value) for key, value in weights.items() if 'lora_B' in key):
            raise ValueError('checkpoint-0 adapter has nonzero B weights')
        loaded = set_peft_model_state_dict(self.model, weights)
        if loaded.unexpected_keys:
            raise ValueError('checkpoint-0 adapter has unexpected keys')
        actual = get_peft_model_state_dict(self.model)
        if any(not self.torch.equal(actual[k], weights[k]) for k in weights):
            raise ValueError('checkpoint-0 adapter load mismatch')

    def train_window(self, rows, step):
        prompts, outputs = pool_window(rows)
        converted = [dict(row, prompt_tokens=len(prompt)) for row, prompt in zip(rows, prompts)]
        self._pool_outputs = outputs
        try:
            records, log = super().train_window(converted, step)
        finally:
            del self._pool_outputs
        self._check_finite_state()
        for record, row in zip(records, (draw for group in rows for draw in group['draws'])):
            record['pool_provenance'] = {k: row[k] for k in ('origin', 'seed', 'batch_id') if k in row}
        return records, log

    def _sample(self, rows):
        return self._pool_outputs

    def _check_finite_state(self):
        def finite(value):
            if self.torch.is_tensor(value):
                return bool(self.torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            return True
        if not finite(self.optimizer.state_dict()):
            raise ValueError('nonfinite optimizer state')
        if not all(bool(self.torch.isfinite(p).all()) for p in self.model.parameters() if p.requires_grad):
            raise ValueError('nonfinite adapter parameters')

    def check_integrity(self):
        result = super().check_integrity()
        self._check_finite_state()
        return result
