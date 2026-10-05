"""FP32 Qwen/PEFT activation instrument, usable with a locally constructed model."""

from contextlib import contextmanager
import time

import numpy as np

from pilot_eval.activation_prepare import PROJECTIONS, VIEWS
from pilot_eval.sft_backend import frozen_weight_hash


class InstrumentFailure(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ActivationEngine:
    """One model; hooks live only during a pass and summaries reduce on device."""

    def __init__(self, *, model, expected_base_hash, expected_layers=28,
                 expected_alpha=1, validation_limit=16):
        import torch
        self.torch = torch
        self.model = model
        self.expected_base_hash = expected_base_hash
        self.expected_alpha = expected_alpha
        self.validation_limit = validation_limit
        self._closed = False
        self.profile_timings = False
        self._initial_adapter = model.active_adapters[0]
        self._initial_training = model.training
        self._initial_grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
        self._initial_adapter_weights = {name: p.detach().cpu().clone()
                                        for name, p in model.named_parameters() if 'lora_' in name}
        self.body = model.get_base_model().model
        self.layers = list(self.body.layers)
        if len(self.layers) != expected_layers:
            raise InstrumentFailure('unexpected decoder block count')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise InstrumentFailure('activation instrument requires FP32 parameters')
        if self.base_hash() != expected_base_hash:
            raise InstrumentFailure('loaded frozen-base hash differs from source')
        self.modules = []
        for layer_index, layer in enumerate(self.layers):
            for projection in PROJECTIONS:
                group = layer.self_attn if projection in PROJECTIONS[:4] else layer.mlp
                module = getattr(group, projection)
                if not hasattr(module, 'lora_A') or module.merged:
                    raise InstrumentFailure('missing or merged adapted linear module')
                if module.active_adapters != ['default']:
                    raise InstrumentFailure('expected one default adapter')
                if (module.lora_A['default'].weight.shape[0] != 1
                        or module.lora_B['default'].weight.shape[1] != 1
                        or module.scaling['default'] != expected_alpha
                        or getattr(module.lora_dropout['default'], 'p', 0) != 0
                        or module.lora_variant):
                    raise InstrumentFailure('unsupported rank/scaling/dropout/LoRA variant')
                self.modules.append((layer_index, projection, module))
        actual = {id(m) for m in model.modules() if hasattr(m, 'lora_A')}
        if actual != {id(m) for _, _, m in self.modules}:
            raise InstrumentFailure('unexpected adapter outside intended projections')
        self._precision = torch.get_float32_matmul_precision()
        self._tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        model.eval()

    def base_hash(self):
        return frozen_weight_hash(self.model)

    def _check_finite(self, *values):
        if any(not self.torch.isfinite(value).all().item() for value in values):
            raise InstrumentFailure('nonfinite activations or contribution')

    def _batch(self, rows):
        torch = self.torch
        if not rows or len({row['id'] for row in rows}) != len(rows):
            raise InstrumentFailure('nonempty unique batch required')
        width = max(len(row['input_ids']) for row in rows)
        tokens, attention, masks = [], [], {view: [] for view in VIEWS}
        pad = self.model.config.pad_token_id
        if pad is None:
            raise InstrumentFailure('explicit padding token required')
        for row in rows:
            size = len(row['input_ids'])
            if size == 0 or row['attention_mask'] != [1] * size:
                raise InstrumentFailure('invalid frozen sequence attention mask')
            tokens.append(row['input_ids'] + [pad] * (width - size))
            attention.append([1] * size + [0] * (width - size))
            for view in VIEWS:
                if len(row['masks'][view]) != size:
                    raise InstrumentFailure('input/content mask alignment mismatch')
                masks[view].append(row['masks'][view] + [False] * (width - size))
            if any(sum(row['masks'][view][i] for view in VIEWS) > 1 for i in range(size)):
                raise InstrumentFailure('overlapping content masks')
        device = next(self.model.parameters()).device
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        attention = torch.tensor(attention, dtype=torch.long, device=device)
        positions = (attention.cumsum(dim=1) - 1).clamp(min=0)
        return {'input_ids': tokens, 'attention_mask': attention, 'position_ids': positions}, {
            view: torch.tensor(value, dtype=torch.bool, device=device) for view, value in masks.items()}

    @contextmanager
    def _hooks(self, registrations):
        handles = []
        try:
            for module, kind, callback in registrations:
                handles.append(getattr(module, kind)(callback))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _forward(self, inputs):
        with self.torch.no_grad(), self.torch.autocast(next(self.model.parameters()).device.type, enabled=False):
            self.body(**inputs, use_cache=False, return_dict=True)

    def capture_reference(self, rows):
        inputs, masks = self._batch(rows)
        blocks = {}
        registrations = []
        for index, layer in enumerate(self.layers):
            def capture(module, args, output, index=index):
                tensor = output[0] if isinstance(output, tuple) else output
                self._check_finite(tensor)
                if tensor.shape[:2] != inputs['input_ids'].shape:
                    raise InstrumentFailure('block output/input shape mismatch')
                if index in blocks:
                    raise InstrumentFailure('decoder block invoked twice')
                blocks[index] = tensor.detach().clone()
            registrations.append((layer, 'register_forward_hook', capture))
        with self._hooks(registrations), self.model.disable_adapter():
            self._forward(inputs)
        if sorted(blocks) != list(range(len(self.layers))):
            raise InstrumentFailure('missing decoder block outputs')
        return {'rows': rows, 'inputs': inputs, 'masks': masks, 'blocks': blocks}

    def _switch(self, checkpoint):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        weights = load_peft_weights(str(checkpoint), device=str(next(self.model.parameters()).device))
        # Embeddings are frozen and excluded by the verified module inventory.
        # Explicit False also prevents PEFT's auto-export Hub config lookup.
        expected = get_peft_model_state_dict(self.model, save_embedding_layers=False)
        if weights.keys() != expected.keys() or any(
                weights[key].shape != expected[key].shape or weights[key].dtype != self.torch.float32
                or not self.torch.isfinite(weights[key]).all().item() for key in weights):
            raise InstrumentFailure('checkpoint parameter identity/shape/precision mismatch')
        result = set_peft_model_state_dict(self.model, weights)
        if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
            raise InstrumentFailure('checkpoint did not load all adapter parameters')
        self.model.set_adapter('default')
        self.model.eval()

    def _positions(self, mask):
        # Round-robin examples before second tokens: fixed coverage across the batch.
        by_example = [self.torch.where(row)[0].tolist() for row in mask]
        positions = [(example, tokens[offset]) for offset in range(max(map(len, by_example), default=0))
                     for example, tokens in enumerate(by_example) if offset < len(tokens)]
        return positions if self.validation_limit is None else positions[:self.validation_limit]

    def _profile_clock(self):
        if not self.profile_timings:
            return 0.
        if next(self.model.parameters()).is_cuda:
            self.torch.cuda.synchronize()
        return time.perf_counter()

    def measure(self, checkpoint, reference, *, step):
        torch = self.torch
        started = self._profile_clock()
        rank1_seconds = 0.
        self._switch(checkpoint)
        repeated = self.capture_reference(reference['rows'])
        invariant = all(torch.equal(reference['blocks'][i], repeated['blocks'][i]) for i in reference['blocks'])
        del repeated
        if not invariant:
            raise InstrumentFailure('disabled-adapter reference changed after checkpoint switch')
        if self.base_hash() != self.expected_base_hash:
            raise InstrumentFailure('frozen-base hash changed after checkpoint switch')
        before_forward = self._profile_clock()
        pre_validation_seconds = before_forward - started
        arrays, diagnostics = {}, []
        masks = reference['masks']
        examples = len(reference['rows'])
        layers, hidden = len(self.layers), self.model.config.hidden_size
        for key in ['count', 'base_norm_sum', 'delta_norm_sum']:
            arrays['block_' + key] = np.zeros((examples, 3, layers), dtype=np.float64)
        for key in ['base_sum', 'delta_sum']:
            arrays['block_' + key] = np.zeros((examples, 3, layers, hidden), dtype=np.float64)
        for key in ['count', 'base_norm_sum', 'delta_norm_sum', 'ratio_sum', 'defined_count']:
            arrays['module_' + key] = np.zeros((examples, 3, layers, 7), dtype=np.float64)
        seen_blocks, seen_modules = set(), set()
        exact_zero = True

        def store(prefix, index, ordinary, delta):
            nonlocal exact_zero
            self._check_finite(ordinary, delta)
            for vi, view in enumerate(VIEWS):
                mask = masks[view]
                if step == 0 and torch.count_nonzero(delta[mask]).item():
                    exact_zero = False
                    raise InstrumentFailure('step 0 has nonzero write', {'gate': 'exact-zero', 'module': index})
                base, change = ordinary.double(), delta.double()
                base_norm, delta_norm = base.norm(dim=-1), change.norm(dim=-1)
                values = {'count': mask.sum(dim=1), 'base_norm_sum': (base_norm * mask).sum(dim=1),
                          'delta_norm_sum': (delta_norm * mask).sum(dim=1)}
                if prefix == 'block':
                    values.update(base_sum=(base * mask[..., None]).sum(dim=1),
                                  delta_sum=(change * mask[..., None]).sum(dim=1))
                else:
                    defined = mask & (base_norm != 0)
                    ratio = torch.zeros_like(base_norm)
                    ratio[defined] = delta_norm[defined] / base_norm[defined]
                    values.update(ratio_sum=ratio.sum(dim=1), defined_count=defined.sum(dim=1))
                for key, value in values.items():
                    destination = (slice(None), vi, index) if prefix == 'block' else (slice(None), vi, *index)
                    arrays[prefix + '_' + key][destination] = value.cpu().numpy()

        registrations = []
        for layer_index, layer in enumerate(self.layers):
            def block_hook(module, args, output, index=layer_index):
                if index in seen_blocks:
                    raise InstrumentFailure('duplicate decoder hook')
                seen_blocks.add(index)
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.shape != reference['blocks'][index].shape:
                    raise InstrumentFailure('adapted/reference block shape mismatch')
                store('block', index, reference['blocks'][index], tensor - reference['blocks'][index])
            registrations.append((layer, 'register_forward_hook', block_hook))
        for layer_index, projection, module in self.modules:
            mi = PROJECTIONS.index(projection)
            captured = {}
            def ordinary_hook(m, args, output, captured=captured):
                captured['ordinary'] = output
            def branch_hook(m, args, output, captured=captured):
                captured['branch'] = output
            def module_hook(m, args, output, captured=captured, index=(layer_index, mi)):
                if index in seen_modules:
                    raise InstrumentFailure('duplicate adapted module hook')
                seen_modules.add(index)
                nonlocal rank1_seconds
                validation_started = self._profile_clock()
                x = args[0]
                ordinary, branch = captured.pop('ordinary'), captured.pop('branch')
                scale = m.scaling['default']
                # Rank-1 identity: full-shape dot product followed by an outer product.
                expected = torch.nn.functional.linear(x, m.lora_A['default'].weight) * \
                    m.lora_B['default'].weight[:, 0] * scale
                actual = branch * scale
                self._check_finite(expected, actual, ordinary, output)
                for view in VIEWS:
                    positions = self._positions(masks[view])
                    if not positions:
                        continue
                    bi, ti = zip(*positions)
                    exp, obs = expected[list(bi), list(ti)], actual[list(bi), list(ti)]
                    on, off = output[list(bi), list(ti)], ordinary[list(bi), list(ti)]
                    branch_error = (obs - exp).abs()
                    subtraction_error = ((on - off) - exp).abs()
                    direct_limit = 1e-6 + 1e-5 * exp.abs()
                    rounding = 4 * (2 ** -23) * (on.abs() + off.abs())
                    evidence = {'layer': index[0], 'projection': PROJECTIONS[index[1]], 'view': view,
                                'positions': [[reference['rows'][b]['id'], t] for b, t in positions],
                                'branch_max_error': float(branch_error.max()),
                                'subtraction_max_error': float(subtraction_error.max()),
                                'branch_max_fraction': float((branch_error / direct_limit).max()),
                                'subtraction_max_fraction': float((subtraction_error / (direct_limit + rounding)).max()),
                                'below_resolution_coordinates': int((exp.abs() <= rounding).sum()),
                                'coordinate_count': exp.numel()}
                    diagnostics.append(evidence)
                    if (branch_error > direct_limit).any() or (subtraction_error > direct_limit + rounding).any():
                        raise InstrumentFailure('rank-1 identity validation failed', evidence)
                rank1_seconds += self._profile_clock() - validation_started
                store('module', index, ordinary, actual)
            registrations.extend([(module.base_layer, 'register_forward_hook', ordinary_hook),
                                  (module.lora_B['default'], 'register_forward_hook', branch_hook),
                                  (module, 'register_forward_hook', module_hook)])
        forward_started = self._profile_clock()
        with self._hooks(registrations):
            self._forward(reference['inputs'])
        forward_finished = self._profile_clock()
        if len(seen_blocks) != layers or len(seen_modules) != layers * 7:
            raise InstrumentFailure('incomplete hook coverage')
        if self.base_hash() != self.expected_base_hash:
            raise InstrumentFailure('frozen-base hash changed during measurement')
        finished = self._profile_clock()
        return {'arrays': arrays, 'timing': {
                'validation_seconds': pre_validation_seconds + rank1_seconds + finished - forward_finished,
                'adapted_forward_and_reductions_seconds': forward_finished - forward_started - rank1_seconds,
                'scope': 'synchronized; validation includes switch, disabled pass, hashes and rank-1 checks; forward includes reductions and zero/finite checks'
                } if self.profile_timings else None, 'validation': {'exact_zero': exact_zero if step == 0 else None,
                'rank1_passed': True, 'reference_invariant': invariant, 'module_count': len(seen_modules),
                'base_sha256': self.expected_base_hash, 'sample_limit': self.validation_limit,
                'block_hooks': [f'model.layers.{index}' for index in range(layers)],
                'module_hooks': [f"model.layers.{index}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
                                 for index, projection, _ in self.modules],
                'block_position': 'decoder-block-output-before-final-model-norm',
                'positions': diagnostics, 'thresholds': {'atol': 1e-6, 'rtol': 1e-5, 'rounding_factor': 4}}}

    def close(self):
        if self._closed:
            return
        self.model.set_adapter(self._initial_adapter)
        with self.torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in self._initial_adapter_weights:
                    parameter.copy_(self._initial_adapter_weights[name].to(parameter.device))
                parameter.requires_grad_(self._initial_grad_flags[name])
        self.model.train(self._initial_training)
        self.torch.set_float32_matmul_precision(self._precision)
        self.torch.backends.cuda.matmul.allow_tf32 = self._tf32
        self._initial_adapter_weights.clear()
        self._closed = True
