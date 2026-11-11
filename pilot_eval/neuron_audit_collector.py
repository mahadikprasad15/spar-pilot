"""Ticket 03a FP32 ``down_proj``-input collector and live-adapter checks (CPU-testable).

Reuses the Pilot 3 ``ActivationEngine`` conventions: right padding with position IDs from
the attention cumulative sum, scoped hook registration, ``no_grad`` FP32 forwards with
autocast off, TF32 off, ``disable_adapter()`` for the untuned arm and the frozen-base
parameter hash. Unlike that engine, dimensions and layers are read from the model config,
nothing assumes 28 layers, and each identity is checked against an independent path:

- z (pre-hook on the PEFT-wrapped ``mlp.down_proj``) vs ``act(gate) * up`` rebuilt from
  separately hooked ``gate_proj``/``up_proj`` outputs, which include their own adapters;
- the PEFT ``lora_A`` output vs a.z, with a read from the *saved* checkpoint file;
- the PEFT ``lora_B`` output times scale vs s*b*c, with b from the saved file;
- (diagnostic) ``down_proj`` output minus ``base_layer`` output vs s*b*c.

Per example, position group and layer the collector stores FP32 mean z and mean |z|
(FP64-accumulated); adapted arms also store the mean signed term a_j z_j and mean
|a_j z_j|, plus per-token c_t at every masked position. Rows with a zero count are NaN, not
zero. Untuned-arm adapter fields are absent, not zero.

This is the CPU artifact contract; it is not a real-GPU qualification (03b).
"""
from contextlib import contextmanager

import numpy as np

from pilot_eval import neuron_audit_math as nm
from pilot_eval.sft_backend import frozen_weight_hash

GROUPS = ['response_content', 'reply_start', 'termination']
BASE_STATS = ['mean_z', 'mean_abs_z']
ADAPTER_STATS = ['mean_term', 'mean_abs_term']


PRECISION_BACKENDS = ['cuda.matmul', 'cudnn.conv', 'cudnn.rnn', 'mkldnn.matmul', 'mkldnn.conv', 'mkldnn.rnn']


def _backend(path):
    import torch
    node = torch.backends
    for part in path.split('.'):
        node = getattr(node, part)
    return node


def precision_backends():
    """Read the torch>=2.9 per-backend FP32 precision settings (new API only; never legacy allow_tf32)."""
    try:
        return {path: _backend(path).fp32_precision for path in PRECISION_BACKENDS}
    except AttributeError as exc:
        raise CollectorFailure('per-backend fp32_precision API missing; pinned torch 2.9.0 required') from exc


def _legacy_matmul_precision(value=None):
    """torch.{set,get}_float32_matmul_precision are legacy in torch 2.9 (warn-once deprecation notice).

    Kept because the run plan requires 'highest'; verified locally not to alter the per-backend
    'ieee' values. Only this notice is suppressed; legacy allow_tf32 setters are never used.
    """
    import warnings
    import torch
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Please use the new API settings to control TF32')
        if value is not None:
            torch.set_float32_matmul_precision(value)
        return torch.get_float32_matmul_precision()


def set_ieee_precision():
    """IEEE FP32 for CUDA matmul, cuDNN and CPU mkldnn conv/rnn/matmul plus 'highest'; read back and verify.

    A global torch.backends.fp32_precision leaves cuDNN conv/rnn at 'tf32' in torch 2.9.0, so
    each backend is set explicitly. Legacy allow_tf32 setters are not used (mixing APIs warns/raises).
    """
    import torch
    for path in PRECISION_BACKENDS:
        _backend(path).fp32_precision = 'ieee'
    observed = dict(precision_backends(), matmul_precision=_legacy_matmul_precision('highest'))
    if any(v != 'ieee' for k, v in observed.items() if k != 'matmul_precision') or \
            observed['matmul_precision'] != 'highest':
        raise CollectorFailure('IEEE FP32 precision settings did not take effect', observed)
    return observed


class CollectorFailure(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def load_saved_factors(adapter_dir, layers, scale, projection='down_proj'):
    """Saved rank-one ``down_proj`` factors read from the checkpoint file, not the live model."""
    from pilot_eval.trace_readout_assets import read_weight
    path = f'{adapter_dir}/adapter_model.safetensors'
    out = {}
    for layer in layers:
        group = 'self_attn' if projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
        base = f'base_model.model.model.layers.{layer}.{group}.{projection}'
        a, b = read_weight(path, base + '.lora_A.weight'), read_weight(path, base + '.lora_B.weight')
        if a.ndim != 2 or a.shape[0] != 1 or b.ndim != 2 or b.shape[1] != 1:
            raise CollectorFailure('saved factors are not rank one', dict(layer=layer))
        out[layer] = dict(a=a[0].astype(np.float32), b=b[:, 0].astype(np.float32))
    return dict(scale=float(scale), layers=out)


class NeuronCollector:
    def __init__(self, model, *, tolerances, expected_base_hash=None, validation_examples=2, dtype='float32',
                 pad_token_id=None):
        """``pad_token_id``: explicit right-padding id (from the pinned tokenizer); falls back to the model config,
        and refuses to guess when neither has one."""
        import torch
        self.torch = torch
        self.model = model
        self.tolerances = dict(tolerances)
        for key in ['z_reconstruction_relative_to_layer_max', 'identity_atol', 'identity_rtol']:
            if not isinstance(self.tolerances.get(key), (int, float)):
                raise CollectorFailure(f'tolerance {key} must be supplied from the plan')
        self.validation_examples = validation_examples
        self.pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
        self.pad_source = 'explicit (tokenizer)' if pad_token_id is not None else 'model config' 
        if dtype not in ('float32', 'float64', 'bfloat16'):
            raise CollectorFailure('collector dtype must be float32 (production), float64 (reference) or '
                                   'bfloat16 (labelled 03b diagnostic only)')
        self.dtype_name, self.dtype = dtype, getattr(torch, dtype)
        if any(p.dtype != self.dtype for p in model.parameters()):
            raise CollectorFailure(f'collector requires {dtype} parameters')
        self.body = model.get_base_model().model
        self.layers = list(self.body.layers)
        config = model.config
        if len(self.layers) != config.num_hidden_layers:
            raise CollectorFailure('decoder block count differs from config')
        self.width = config.intermediate_size
        for index, layer in enumerate(self.layers):
            down = layer.mlp.down_proj
            if down.in_features != self.width:
                raise CollectorFailure('down_proj input width differs from config', dict(layer=index))
            if hasattr(down, 'lora_A'):
                dropout = down.lora_dropout['default'] if 'default' in down.lora_dropout else None
                if down.merged or getattr(dropout, 'p', 0) not in (0, None):
                    raise CollectorFailure('merged adapter or nonzero dropout on down_proj', dict(layer=index))
        self.expected_base_hash = expected_base_hash
        if expected_base_hash is not None and frozen_weight_hash(model) != expected_base_hash:
            raise CollectorFailure('loaded frozen-base hash differs from source')
        self.flags = dict(precision_backends(), matmul_precision=_legacy_matmul_precision())
        set_ieee_precision()
        model.eval()

    def runtime(self):
        torch = self.torch
        return dict(torch=torch.__version__, dtype=self.dtype_name, autocast=False, torch_compile=False,
                    fp32_precision=precision_backends(), matmul_precision=_legacy_matmul_precision(),
                    attention=getattr(self.model.config, '_attn_implementation', None),
                    training=self.model.training, layers=len(self.layers), intermediate_size=self.width,
                    num_attention_heads=self.model.config.num_attention_heads, pad_token_id=self.pad_token_id,
                    pad_source=self.pad_source)

    def close(self):
        torch = self.torch
        for path, value in self.flags.items():
            if path != 'matmul_precision':
                _backend(path).fp32_precision = value
        _legacy_matmul_precision(self.flags['matmul_precision'])

    # --- batching (ActivationEngine._batch conventions, position-list masks) ----------------

    def batch(self, rows, *, padded_width=None):
        torch = self.torch
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise CollectorFailure('nonempty unique batch required')
        width = max(len(r['input_ids']) for r in rows)
        if padded_width is not None:
            if padded_width < width:
                raise CollectorFailure('padded width below the longest sequence')
            width = padded_width
        pad = self.pad_token_id
        if pad is None:
            raise CollectorFailure('explicit padding token required')
        tokens, attention = [], []
        masks = {g: torch.zeros((len(rows), width), dtype=torch.bool) for g in GROUPS}
        for i, row in enumerate(rows):
            size = len(row['input_ids'])
            tokens.append(list(row['input_ids']) + [pad] * (width - size))
            attention.append([1] * size + [0] * (width - size))
            seen = set()
            for group in GROUPS:
                positions = row['masks'][group]
                if any(not 0 <= p < size for p in positions):
                    raise CollectorFailure('mask position outside the sequence', dict(id=row['id'], group=group))
                if group == 'response_content' and not positions:
                    raise CollectorFailure('empty primary mask', dict(id=row['id']))
                masks[group][i, positions] = True
                if group != 'termination':
                    if seen & set(positions):
                        raise CollectorFailure('overlapping response/reply-start masks', dict(id=row['id']))
                    seen |= set(positions)
        device = next(self.model.parameters()).device
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        attention = torch.tensor(attention, dtype=torch.long, device=device)
        positions = (attention.cumsum(dim=1) - 1).clamp(min=0)
        inputs = dict(input_ids=tokens, attention_mask=attention, position_ids=positions)
        return inputs, {g: m.to(device) for g, m in masks.items()}

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
        torch = self.torch
        with torch.no_grad(), torch.autocast(next(self.model.parameters()).device.type, enabled=False):
            return self.model(**inputs, use_cache=False, return_dict=True).logits

    # --- collection ------------------------------------------------------------------------

    def collect(self, rows, *, factors=None, padded_width=None, validate=True, heads=None, head_factors=None):
        """One frozen batch. ``factors`` None means the untuned arm (adapter disabled).

        ``heads`` (Phase 1B) lists layers whose ``o_proj`` input is summarized per attention head:
        mean per-head L2 norm and, for adapted arms with ``head_factors`` (saved ``o_proj`` factors),
        mean per-head read a_h.x_h and its absolute value. Their sum over heads is checked against
        the independently captured PEFT ``o_proj`` lora_A output (ADR 0015 bound).
        """
        torch = self.torch
        inputs, masks = self.batch(rows, padded_width=padded_width)
        examples = len(rows)
        counts = {g: masks[g].sum(dim=1).cpu().numpy().astype(np.int64) for g in GROUPS}
        union = masks['response_content'] | masks['reply_start'] | masks['termination']
        arrays, token_parts, validation = {}, [], []
        fired = {}
        captured = {}

        def store_means(group, layer, stat, total):
            mean = (total / torch.from_numpy(np.maximum(counts[group], 1)).to(total)[:, None]).cpu().numpy()
            mean[counts[group] == 0] = np.nan
            # Production stores FP32; the FP64 reference keeps FP64 for the D-floor comparison.
            arrays[(group, layer, stat)] = mean.astype(np.float32 if self.dtype_name == 'float32' else np.float64)

        def note(key):
            fired[key] = fired.get(key, 0) + 1

        registrations = []
        for index, layer in enumerate(self.layers):
            mlp = layer.mlp
            down = mlp.down_proj
            per = captured.setdefault(index, {})

            def pre_hook(module, args, index=index):
                note(('z', index))
                z = args[0]
                if z.shape != (examples, inputs['input_ids'].shape[1], self.width) or z.dtype != self.dtype:
                    raise CollectorFailure('unexpected down_proj input shape/dtype', dict(layer=index))
                if not torch.isfinite(z).all():
                    raise CollectorFailure('nonfinite down_proj input', dict(layer=index))
                zd = z.detach().double()
                for group in GROUPS:
                    m = masks[group].to(zd)[..., None]
                    store_means(group, index, 'mean_z', (zd * m).sum(dim=1))
                    store_means(group, index, 'mean_abs_z', (zd.abs() * m).sum(dim=1))
                if factors is not None:
                    a = torch.from_numpy(factors['layers'][index]['a']).to(z.device).double()
                    terms = zd * a
                    for group in GROUPS:
                        m = masks[group].to(zd)[..., None]
                        store_means(group, index, 'mean_term', (terms * m).sum(dim=1))
                        store_means(group, index, 'mean_abs_term', (terms.abs() * m).sum(dim=1))
                    c = terms.sum(dim=-1)
                    rows_i, pos_i = torch.where(union)
                    token_parts.append((index, c[rows_i, pos_i].float().cpu().numpy(),
                                        rows_i.cpu().numpy(), pos_i.cpu().numpy()))
                if validate:
                    captured[index]['z'] = z.detach()[: self.validation_examples].clone()

            def keep(name, index=index):
                def hook(module, args, output):
                    note((name, index))
                    if validate:
                        captured[index][name] = output.detach()[: self.validation_examples].clone()
                return hook

            registrations += [(down, 'register_forward_pre_hook', pre_hook),
                              (mlp.gate_proj, 'register_forward_hook', keep('gate')),
                              (mlp.up_proj, 'register_forward_hook', keep('up'))]
            if hasattr(down, 'lora_A'):
                registrations += [(down.lora_A['default'], 'register_forward_hook', keep('lora_a')),
                                  (down.lora_B['default'], 'register_forward_hook', keep('lora_b')),
                                  (down.base_layer, 'register_forward_hook', keep('base_out')),
                                  (down, 'register_forward_hook', keep('down_out'))]

        if head_factors is not None and factors is None:
            raise CollectorFailure('head factors given for the untuned arm')
        head_slots = {}
        n_heads = self.model.config.num_attention_heads
        for index in sorted(heads or []):
            attn = self.layers[index].self_attn
            o_proj = attn.o_proj
            head_dim = getattr(attn, 'head_dim', self.model.config.hidden_size // n_heads)
            slot = head_slots.setdefault(index, {})

            def o_pre(module, args, index=index, slot=slot, head_dim=head_dim):
                note(('o', index))
                x = args[0]
                if x.shape[-1] != n_heads * head_dim or x.dtype != self.dtype:
                    raise CollectorFailure('o_proj input width/dtype differs from heads x head_dim', dict(layer=index))
                if not torch.isfinite(x).all():
                    raise CollectorFailure('nonfinite o_proj input', dict(layer=index))
                xd = x.detach().double()
                xh = xd.view(*xd.shape[:2], n_heads, head_dim)
                norms = xh.norm(dim=-1)
                for group in GROUPS:
                    m = masks[group].to(xd)[..., None]
                    store_means(group, index, 'head_mean_norm', (norms * m).sum(dim=1))
                if head_factors is not None:
                    a = torch.from_numpy(head_factors['layers'][index]['a']).to(x.device).double()
                    reads = (xh * a.view(n_heads, head_dim)).sum(dim=-1)
                    for group in GROUPS:
                        m = masks[group].to(xd)[..., None]
                        store_means(group, index, 'head_mean_term', (reads * m).sum(dim=1))
                        store_means(group, index, 'head_mean_abs_term', (reads.abs() * m).sum(dim=1))
                    if validate:
                        slot['x'] = x.detach()[: self.validation_examples].clone()
                        slot['head_sum'] = reads[: self.validation_examples].sum(dim=-1)

            registrations.append((o_proj, 'register_forward_pre_hook', o_pre))
            if head_factors is not None:
                if not hasattr(o_proj, 'lora_A'):
                    raise CollectorFailure('o_proj has no adapter for head read checks', dict(layer=index))

                def o_lora_a(module, args, output, slot=slot):
                    if validate:
                        slot['lora_a'] = output.detach()[: self.validation_examples].clone()
                registrations.append((o_proj.lora_A['default'], 'register_forward_hook', o_lora_a))

        context = self.model.disable_adapter() if factors is None else _null()
        with context, self._hooks(registrations):
            logits = self._forward(inputs)
        if not torch.isfinite(logits).all():
            raise CollectorFailure('nonfinite logits')
        layers = len(self.layers)
        if any(fired.get(('z', i)) != 1 for i in range(layers)):
            raise CollectorFailure('down_proj pre-hook did not fire exactly once per layer')
        routing = dict(lora_a_calls=sum(fired.get(('lora_a', i), 0) for i in range(layers)),
                       lora_b_calls=sum(fired.get(('lora_b', i), 0) for i in range(layers)))
        if factors is None and (routing['lora_a_calls'] or routing['lora_b_calls']):
            raise CollectorFailure('adapter branch ran while disabled', routing)
        if factors is not None and (routing['lora_a_calls'] != layers or routing['lora_b_calls'] != layers):
            raise CollectorFailure('adapter branch did not run once per layer', routing)
        if validate:
            validation = self._validate(captured, factors, masks, arrays, counts)
            union_valid = union[: self.validation_examples]
            bound = dict(atol=self.tolerances['identity_atol'], rtol=self.tolerances['identity_rtol'])
            for index, slot in sorted(head_slots.items()):
                if head_factors is None:
                    continue
                a = head_factors['layers'][index]['a']
                x = slot['x'][union_valid[: slot['x'].shape[0]]].double().cpu().numpy()
                for name, observed in [('head_read_sum', slot['head_sum']), ('o_proj_lora_a_read', slot['lora_a'])]:
                    rec = nm.check_read(observed[union_valid[: observed.shape[0]]].double().cpu().numpy(), a, x, **bound)
                    validation.append(dict(layer=index, check=name, **rec))
            failures = [r for r in validation if r['check'] in ('head_read_sum', 'o_proj_lora_a_read')
                        and r['status'] != 'pass']
            if failures:
                raise CollectorFailure('independent head read check failed', dict(failures=failures[:8]))
        if any(fired.get(('o', i)) != 1 for i in head_slots):
            raise CollectorFailure('o_proj pre-hook did not fire exactly once per selected layer')
        tokens = None
        if factors is not None:
            token_parts.sort(key=lambda part: part[0])
            first = token_parts[0]
            if any((p[2] != first[2]).any() or (p[3] != first[3]).any() for p in token_parts):
                raise CollectorFailure('token mapping differs across layers')
            group_code = np.where(masks['response_content'][first[2], first[3]].cpu().numpy(), 0,
                                  np.where(masks['reply_start'][first[2], first[3]].cpu().numpy(), 1, 2))
            # termination overlaps the last content token; that token keeps code 0 and is
            # recovered from the termination mask positions, which are stored in the rows.
            tokens = dict(ct=np.stack([p[1] for p in token_parts]).astype(np.float32),
                          token_rows=first[2].astype(np.int32), token_positions=first[3].astype(np.int32),
                          token_groups=group_code.astype(np.uint8))
        return dict(arrays=arrays, counts=counts, tokens=tokens, validation=validation, routing=routing,
                    logits=logits, example_ids=[r['id'] for r in rows],
                    padded_width=int(inputs['input_ids'].shape[1]))

    def _validate(self, captured, factors, masks, arrays, counts):
        records = []
        act = None
        for index, layer in enumerate(self.layers):
            got = captured[index]
            act = layer.mlp.act_fn
            z = got['z']
            rebuilt = act(got['gate']) * got['up']
            valid = (masks['response_content'] | masks['reply_start'] | masks['termination'])[: z.shape[0]]
            z_np, rebuilt_np = z[valid].double().cpu().numpy(), rebuilt[valid].double().cpu().numpy()
            rec = nm.check_reconstruction(z_np, rebuilt_np,
                                          rel_to_layer_max=self.tolerances['z_reconstruction_relative_to_layer_max'])
            records.append(dict(layer=index, check='z_reconstruction', **rec))
            for row in range(z.shape[0]):
                for group in GROUPS:
                    positions = masks[group][row].nonzero().flatten().tolist()
                    reported_mean = arrays[(group, index, 'mean_z')][row].astype(np.float64)
                    reported_abs = arrays[(group, index, 'mean_abs_z')][row].astype(np.float64)
                    n = counts[group][row]
                    red = nm.check_reduction(z[row].double().cpu().numpy(), positions,
                                             np.nan_to_num(reported_mean) * n, np.nan_to_num(reported_abs) * n, n)
                    records.append(dict(layer=index, check='reduction', group=group, row=row, **red))
            if factors is not None:
                saved = factors['layers'][index]
                bound = dict(atol=self.tolerances['identity_atol'], rtol=self.tolerances['identity_rtol'])
                rec = nm.check_read(got['lora_a'][valid].double().cpu().numpy(), saved['a'], z_np, **bound)
                records.append(dict(layer=index, check='lora_a_read', **rec))
                live_scale = float(layer.mlp.down_proj.scaling['default'])  # PEFT's own path, not the saved value
                branch = (got['lora_b'][valid].double() * live_scale).cpu().numpy()
                rec = nm.check_branch(branch, saved['a'], saved['b'], z_np, factors['scale'], **bound)
                records.append(dict(layer=index, check='adapter_branch', **rec))
                diff = (got['down_out'][valid].double() - got['base_out'][valid].double()).cpu().numpy()
                rec = nm.check_branch(diff, saved['a'], saved['b'], z_np, factors['scale'], **bound)
                records.append(dict(layer=index, check='output_minus_base_diagnostic', gate=False, **rec))
        failures = [r for r in records if r.get('gate', True) and r['status'] != 'pass']
        if failures:
            raise CollectorFailure('independent instrument check failed', dict(failures=failures[:8]))
        return records


@contextmanager
def _null():
    yield


# --- live adapter state ------------------------------------------------------------------------

def check_live_adapter(model, adapter_dir, *, expected_scale, expected_keys=None, adapter_name='default'):
    """Saved vs live factors, keys, shapes, dtype (bitwise), routing and scale; never loads anything."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(f'{adapter_dir}/adapter_model.safetensors')
    live = get_peft_model_state_dict(model, adapter_name=adapter_name, save_embedding_layers=False)
    reasons = []
    if expected_keys is not None and set(saved) != set(expected_keys):
        reasons.append('saved keys differ from the sealed inventory key list')
    missing, unexpected = sorted(set(saved) - set(live)), sorted(set(live) - set(saved))
    if missing or unexpected:
        reasons.append('saved/live key sets differ')
    mismatched, cast, zero_b = [], [], []
    for key in sorted(set(saved) & set(live)):
        s, l = saved[key], live[key].detach().cpu()
        if s.dtype != l.dtype:
            cast.append(key)
        elif s.shape != l.shape or not torch.equal(s, l):
            mismatched.append(key)
        if key.endswith('lora_B.weight') and not torch.any(s):
            zero_b.append(key[:-len('.lora_B.weight')])
    if cast:
        reasons.append('undeclared dtype cast between saved and live factors')
    if mismatched:
        reasons.append('live factors differ from saved factors')
    routing = []
    if getattr(model, 'active_adapters', [adapter_name]) != [adapter_name]:
        reasons.append('unexpected active adapter set')
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and adapter_name in module.lora_A:
            state = dict(module=name, merged=bool(module.merged), disabled=bool(module.disable_adapters),
                         active=list(module.active_adapters), scale=float(module.scaling[adapter_name]))
            if state['merged'] or state['disabled'] or state['active'] != [adapter_name] \
                    or state['scale'] != float(expected_scale):
                routing.append(state)
    if routing:
        reasons.append('module merged, disabled, inactive or with an unexpected scale')
    return dict(status='fail' if reasons else 'pass', reasons=reasons, missing_keys=missing,
                unexpected_keys=unexpected, mismatched_keys=mismatched, cast_keys=cast,
                zero_b_modules=zero_b, routing_problems=routing[:16], key_count=len(saved))


def adapter_effect(model, inputs, reference_logits=None):
    """Active-vs-disabled logits and disabled-vs-independent-base logits; classification only.

    The disabled-base pass bound is unresolved in the candidate plan, so no pass/fail is set.
    """
    import torch
    with torch.no_grad(), torch.autocast(next(model.parameters()).device.type, enabled=False):
        active = model(**inputs, use_cache=False).logits
        with model.disable_adapter():
            disabled = model(**inputs, use_cache=False).logits
    delta = float((active - disabled).abs().max())
    out = dict(active_minus_disabled_max_abs=delta,
               classification='changes-logits' if delta > 0 else 'no-logit-change-unresolved')
    if reference_logits is not None:
        out['disabled_minus_reference_max_abs'] = float((disabled - reference_logits).abs().max())
        out['disabled_reference_tolerance'] = None
    return out


def collect_cohort(collector, store, rows, *, arm, batch_size, factors=None, identity, validate=True,
                   heads=None, head_factors=None, key=None):
    """Collect frozen rows in order, one sealed block unit per batch; resume skips sealed units."""
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        unit_key = key if key is not None and len(rows) <= batch_size else \
            f'{arm}-rows-{start:06d}-{start + len(batch) - 1:06d}'
        unit_identity = dict(identity, arm=arm, rows=[r['id'] for r in batch])
        if store.state(unit_key) == 'sealed':
            results.append((unit_key, store.write_unit(unit_key, identity=unit_identity, example_ids=[], counts={},
                                                  blocks={})))
            continue
        out = collector.collect(batch, factors=factors, validate=validate, heads=heads, head_factors=head_factors)
        stats = BASE_STATS + (ADAPTER_STATS if factors is not None else []) + (
            ['head_mean_norm'] + (['head_mean_term', 'head_mean_abs_term'] if head_factors is not None else [])
            if heads else [])
        blocks = {k: v for k, v in out['arrays'].items() if k[2] in stats}
        meta = dict(routing=out['routing'], padded_width=out['padded_width'],
                    validation_failures=0, validation_checks=len(out['validation']),
                    adapter=None if factors is None else dict(
                        scale=factors['scale'],
                        b_norm={str(l): float(np.linalg.norm(f['b'])) for l, f in factors['layers'].items()}))
        results.append((unit_key, store.write_unit(unit_key, identity=unit_identity, example_ids=out['example_ids'],
                                              counts={g: v for g, v in out['counts'].items()}, blocks=blocks,
                                              tokens=out['tokens'], meta=meta)))
    return results
