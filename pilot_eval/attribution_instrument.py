"""Ticket 03: scalar-read attribution instrument (spec §Stage A+B), pinned ``peft==0.18.0``.

For every rank-1 adapter the scalar read c_t is the ``lora_A`` output ([batch, seq, 1]). PEFT computes the
branch inline as Δy = s · lora_B(lora_A(drop(x))), so removing the adapter (gate g: 1 -> 0) has first-order
effect −Σ_t ⟨∂M/∂Δy_t, Δy_t⟩ = −Σ_t c_t · ∂M/∂c_t; the gradient at c_t already passes through the real b and
scale. Only these scalar reads and their gradients are requested; no full-width branch output or gradient.

- **Graph.** All weights stay frozen (checked). The graph comes from ``requires_grad_(True)`` on the
  embedding output, an activation. Each metric has its own ``torch.autograd.grad`` call over the list of
  captured c_t tensors (fresh tensors; ``.grad`` is never used).
- **Pass 1.** One forward; the output head runs only at the read positions, in checkpointed position
  chunks (FP32 full-vocabulary logits are never materialized for the whole batch). Per example and metric:
  attr_i = −Σ_t c_t g_t over every unpadded position (prompt, header, response); also Σ_t |c_t| and the
  token count per adapter (write matching), signed Σ_t c_t per ``down_proj`` adapter (c̄), and per-token
  g_t for every ``down_proj`` adapter.
- **Pass 2.** Forward only; at each ``down_proj`` input z the pre-hook contracts attr_j = −a_j Σ_t z_{j,t} g_t
  for that layer and drops z (one layer's z at a time). Σ_j attr_j = attr_i is enforced per example.
- FP32 forward and backward, autocast off; products and sums in FP64.

``branch_oracle``, ``branch_gates`` and ``neuron_gate`` are an independent fixture path: they rebuild the
scaled branch explicitly with a gate and differentiate or perturb the gate.
"""
from contextlib import contextmanager, ExitStack

import numpy as np

from pilot_eval import attribution_metrics as am

PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
METRICS = ('trace', 'stop')
AVAILABLE_METRICS = METRICS + ('seqlogp',)


class InstrumentFailure(ValueError):
    pass


def _group(projection):
    return 'self_attn' if projection in PROJECTIONS[:4] else 'mlp'


def adapter_modules(model):
    """[(layer, projection, module)] in layer, then projection order (the inventory's 196 order)."""
    layers = model.get_base_model().model.layers
    out = []
    for index, layer in enumerate(layers):
        for projection in PROJECTIONS:
            module = getattr(getattr(layer, _group(projection)), projection)
            if not hasattr(module, 'lora_A'):
                raise InstrumentFailure(f'layer {index} {projection} has no LoRA adapter')
            out.append((index, projection, module))
    return out


@contextmanager
def full_fp64(model):
    """Make an FP64 Qwen2 really FP64 (diagnostic 4, 2026-11-16).

    transformers 4.57 Qwen2 casts to FP32 inside the eager attention softmax, every RMSNorm and the rotary embedding,
    even in an FP64 model; an "FP64" finite difference then still carries FP32 rounding from every block. Inside this
    context those three steps compute in the input's dtype when it is FP64 (inv_freq recomputed in FP64 for the
    default RoPE). FP32 inputs follow the original code paths exactly, so FP32 results are bit-identical."""
    import torch
    from transformers.models.qwen2 import modeling_qwen2 as mq
    original_attention = mq.eager_attention_forward

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.dtype != torch.float64:
            return original_attention(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        key_states = mq.repeat_kv(key, module.num_key_value_groups)
        value_states = mq.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float64)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        return torch.matmul(weights, value_states).transpose(1, 2).contiguous(), weights

    attention.__name__ = 'eager_attention_forward'
    patched = []

    def norm_forward(module, original):
        def forward(hidden_states):
            if hidden_states.dtype != torch.float64:
                return original(hidden_states)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            return module.weight * (hidden_states * torch.rsqrt(variance + module.variance_epsilon))
        return forward

    def rope_forward(module, original):
        def forward(x, position_ids):
            if x.dtype != torch.float64:
                return original(x, position_ids)
            if module.rope_type != 'default':
                raise ValueError(f'full_fp64 supports the default RoPE only, not {module.rope_type!r}')
            config = module.config
            dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
            inv = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=x.device) / dim))
            freqs = (inv[None, :, None].expand(position_ids.shape[0], -1, 1)
                     @ position_ids[:, None, :].to(torch.float64)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos() * module.attention_scaling, emb.sin() * module.attention_scaling
        return forward

    try:
        mq.eager_attention_forward = attention
        for module in model.modules():
            if isinstance(module, mq.Qwen2RMSNorm):
                patched.append((module, module.forward))
                module.forward = norm_forward(module, module.forward)
            elif isinstance(module, mq.Qwen2RotaryEmbedding):
                patched.append((module, module.forward))
                module.forward = rope_forward(module, module.forward)
        yield
    finally:
        mq.eager_attention_forward = original_attention
        for module, forward in patched:
            module.forward = forward


def contract(c, g, mask):
    """−Σ_t c_t g_t over the masked positions, FP64, per example (numpy)."""
    import torch
    mask = torch.as_tensor(mask).to(device=c.device, dtype=torch.float64)
    return (-(c[..., 0].double() * g[..., 0].double() * mask).sum(1)).cpu().numpy()


def response_mask(instrument, rows):
    """Mutation helper: only response-content (trace-target) positions."""
    import torch
    ids, _, _ = instrument.batch(rows)
    mask = torch.zeros(ids.shape, dtype=torch.float64, device=ids.device)
    for r, row in enumerate(rows):
        mask[r, row['masks']['trace_targets']] = 1
    return mask


def pad_rows(model, rows, pad_token_id):
    """Right-padded ``(input_ids, attention_mask, position_ids)`` on the model's device. Needs no PEFT wrapper,
    so checks such as ``c_attribution_checks.noop_check`` can batch rows for any model."""
    import torch
    if pad_token_id is None:
        raise InstrumentFailure('explicit pad token required')
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise InstrumentFailure('nonempty unique batch required')
    width = max(len(r['input_ids']) for r in rows)
    device = next(model.parameters()).device
    ids = torch.tensor([list(r['input_ids']) + [pad_token_id] * (width - len(r['input_ids'])) for r in rows],
                       device=device)
    attention = torch.tensor([[1] * len(r['input_ids']) + [0] * (width - len(r['input_ids'])) for r in rows],
                             device=device)
    positions = (attention.cumsum(1) - 1).clamp(min=0)
    return ids, attention, positions


class Instrument:
    def __init__(self, model, *, stop_ids, pad_token_id, identity_rtol, adapter='default', head_chunk=1024,
                 metric_order=METRICS, precision='fp32'):
        import torch
        self.torch, self.model, self.adapter = torch, model, adapter
        self.stop_ids = [int(s) for s in stop_ids]
        if pad_token_id is None:
            raise InstrumentFailure('explicit pad token required')
        if not isinstance(identity_rtol, float):
            raise InstrumentFailure('the neuron-sum identity tolerance must come from the launch record')
        if not metric_order or len(set(metric_order)) != len(metric_order) or not set(metric_order) <= set(AVAILABLE_METRICS):
            raise InstrumentFailure('metric order must name a nonempty set of known metrics once each')
        if precision not in ('fp32', 'fp64'):  # fp64: diagnostic 4 toward option A (user 2026-11-16)
            raise InstrumentFailure(f'precision must be fp32 or fp64, not {precision!r}')
        self.precision = precision
        self.dtype = torch.float32 if precision == 'fp32' else torch.float64
        self.pad, self.rtol, self.chunk, self.order = pad_token_id, identity_rtol, head_chunk, tuple(metric_order)
        self.body = model.get_base_model().model
        self.head = model.get_base_model().lm_head
        self.modules = adapter_modules(model)
        self.adapters = [(l, p) for l, p, _ in self.modules]
        self.down_index = {l: i for i, (l, p) in enumerate(self.adapters) if p == 'down_proj'}

    # --- checks ---------------------------------------------------------------------------

    def check_routing(self, *, untuned=False):
        problems = []
        if any(p.requires_grad for p in self.model.parameters()):
            raise InstrumentFailure('all model and adapter weights must stay frozen (requires_grad False)')
        if any(p.dtype != self.dtype for p in self.model.parameters()):
            raise InstrumentFailure(f'{self.precision.upper()} weights required for a {self.precision} instrument')
        if not untuned and list(self.model.active_adapters) != [self.adapter]:
            problems.append(dict(active=list(self.model.active_adapters)))
        for layer, projection, module in self.modules:
            dropout = module.lora_dropout[self.adapter] if self.adapter in module.lora_dropout else None
            state = dict(layer=layer, projection=projection, merged=bool(module.merged),
                         disabled=bool(module.disable_adapters), active=list(module.active_adapters),
                         present=self.adapter in module.lora_A, dropout=getattr(dropout, 'p', 0))
            if state['merged'] or not state['present'] or state['dropout'] not in (0, None) or \
                    (not untuned and (state['disabled'] or state['active'] != [self.adapter])):
                problems.append(state)
            elif state['present'] and module.lora_A[self.adapter].weight.shape[0] != 1:
                problems.append(dict(state, rank=module.lora_A[self.adapter].weight.shape[0]))
        if problems:
            raise InstrumentFailure(f'adapter routing check failed: {problems[:4]}')
        return dict(status='pass', adapters=len(self.modules))

    # --- batching ---------------------------------------------------------------------------

    def batch(self, rows):
        return pad_rows(self.model, rows, self.pad)

    @contextmanager
    def _capture(self, captured, calls):
        handles = []

        def grad_input(module, args, output):
            return output.requires_grad_(True)

        try:
            handles.append(self.body.embed_tokens.register_forward_hook(grad_input))
            for index, (_, _, module) in enumerate(self.modules):
                if self.adapter not in module.lora_A:
                    continue

                def hook(m, args, output, index=index):
                    if output.shape[-1] != 1:
                        raise InstrumentFailure('scalar read must be rank one')
                    calls[index] = calls.get(index, 0) + 1
                    captured[index] = output

                handles.append(module.lora_A[self.adapter].register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def hooked_logits(self, rows):
        torch = self.torch
        ids, attention, positions = self.batch(rows)
        with torch.enable_grad(), self._capture({}, {}):
            logits = self.model(input_ids=ids, attention_mask=attention, position_ids=positions,
                                use_cache=False).logits
        return logits.detach()

    # --- metrics through the bounded head --------------------------------------------------

    def _metrics(self, hidden, ids, rows):
        torch = self.torch
        from torch.utils.checkpoint import checkpoint
        values = {name: [] for name in self.order}
        reasons = []

        def terms(h, targets):
            return am.position_terms(self.head(h), targets, self.stop_ids)

        def read_positions(r, positions):
            index = torch.tensor(positions, device=hidden.device)
            parts = [checkpoint(terms, hidden[r, index[s:s + self.chunk]], ids[r, index[s:s + self.chunk] + 1],
                                use_reentrant=False) for s in range(0, len(positions), self.chunk)]
            return torch.cat(parts)

        for r, row in enumerate(rows):
            length = len(row['input_ids'])
            masks = row['masks']
            trace = am.logit_positions(masks['trace_targets'], length=length) if 'trace' in self.order else []
            stop = am.logit_positions(masks['stop_targets'], length=length) if 'stop' in self.order else []
            if 'trace' in self.order and not trace:
                raise InstrumentFailure(f"{row['id']}: empty trace mask")
            positions = sorted(set(trace) | set(stop))
            where = {p: k for k, p in enumerate(positions)}
            out = read_positions(r, positions) if positions else None
            if 'trace' in self.order:
                values['trace'].append(out[[where[p] for p in trace], 0].double().sum() / len(trace))
            if 'stop' in self.order:
                if stop:
                    values['stop'].append(out[[where[p] for p in stop], 1].double().sum() / len(stop))
                    reasons.append(None)
                else:
                    values['stop'].append(torch.tensor(float('nan'), dtype=torch.float64, device=hidden.device))
                    reasons.append(am.EMPTY_STOP)
            else:
                reasons.append(None)
            if 'seqlogp' in self.order:
                seq = am.logit_positions(masks['seq_targets'], length=length)
                if not seq:
                    raise InstrumentFailure(f"{row['id']}: empty sequence target mask")
                # Keep the legacy trace/stop head batches unchanged when this metric is added.
                values['seqlogp'].append(read_positions(r, seq)[:, 0].double().sum())
        return {k: torch.stack(v) for k, v in values.items()}, reasons

    # --- passes ------------------------------------------------------------------------------

    def _reduction(self, request, n):
        """Validate detached per-row coefficients before the model is evaluated."""
        if request is None:
            return None
        if self.order != ('seqlogp',):
            raise InstrumentFailure('question reduction requires seqlogp as the only metric')
        try:
            ids = list(request['question_ids'])
            coefficients = np.array(request['coefficients'], dtype=np.float64, copy=True)
            halves = np.array(request['half_coefficients'], dtype=np.float64, copy=True)
            unique = list(dict.fromkeys(ids))
        except (KeyError, TypeError, ValueError) as exc:
            raise InstrumentFailure('invalid neuron reduction request') from exc
        if len(ids) != n or coefficients.shape != (n,) or halves.shape != (n, 2):
            raise InstrumentFailure('neuron reduction coefficients must match the batch')
        if not np.isfinite(coefficients).all() or not np.isfinite(halves).all():
            raise InstrumentFailure('neuron reduction coefficients must be finite')
        return dict(ids=ids, unique=unique, coefficients=coefficients, halves=halves)

    def attribute(self, rows, *, raw=False, untuned=False, neuron_reduction=None, _corrupt_pass2=False):
        """Attribute selected metrics; optionally reduce seqlogp reads inside each layer.

        neuron_reduction supplies question_ids, per-row coefficients (R - LOO baseline) / n_q,
        and per-row half_coefficients for the two draw halves, with the half-specific baselines.
        These coefficients exclude the final 1/Q factor. Results include FP64 per-question
        sums and draw-half sums, plus per-response identity error, gross, and signed net.
        No per-response neuron matrix is retained by this reduced path.
        """
        torch = self.torch
        self.check_routing(untuned=untuned)
        ids, attention, positions = self.batch(rows)
        unpadded = attention.to(torch.float64)
        n, count = len(rows), len(self.modules)
        reduction = self._reduction(neuron_reduction, n)
        if reduction is not None and untuned:
            raise InstrumentFailure('question reduction requires an active adapter')
        captured, calls = {}, {}
        precision_scope = full_fp64(self.model) if self.precision == 'fp64' else ExitStack()
        precision_scope.__enter__()
        try:
            return self._attribute(rows, ids, attention, positions, unpadded, n, count, captured, calls,
                                   raw=raw, untuned=untuned, corrupt=_corrupt_pass2, reduction=reduction)
        finally:
            precision_scope.__exit__(None, None, None)

    def _attribute(self, rows, ids, attention, positions, unpadded, n, count, captured, calls, *, raw, untuned,
                   corrupt, reduction):
        torch = self.torch
        context = frozen_disable_adapter(self.model) if untuned else ExitStack()
        with context, torch.enable_grad(), torch.autocast(ids.device.type, enabled=False), \
                self._capture(captured, calls):
            hidden = self.body(input_ids=ids, attention_mask=attention, position_ids=positions,
                               use_cache=False).last_hidden_state
            metrics, reasons = self._metrics(hidden, ids, rows)
            if untuned:
                if calls:
                    raise InstrumentFailure('adapter branch ran while disabled')
            elif sorted(calls) != list(range(count)) or set(calls.values()) != {1}:
                raise InstrumentFailure(f'routing: lora_A did not run once per adapter ({len(calls)}/{count})')
            reads = [captured[i] for i in range(count)] if not untuned else []
            grads = {}
            for k, name in enumerate(self.order):
                value = metrics[name]
                defined = ~torch.isnan(value)
                if not reads or not bool(defined.any()):
                    grads[name] = None
                    continue
                grads[name] = torch.autograd.grad(value[defined].sum(), reads, retain_graph=k < len(self.order) - 1)
        attr = {}
        for name in self.order:
            if grads[name] is None:
                a = torch.zeros((n, count), dtype=torch.float64)
            else:
                a = torch.stack([-(reads[i][..., 0].detach().double() * grads[name][i][..., 0].double()
                                   * unpadded).sum(1) for i in range(count)], dim=1).cpu()
            a[torch.isnan(metrics[name]).cpu()] = float('nan')
            attr[name] = a.numpy()
        if reads:
            abs_c = torch.stack([(reads[i][..., 0].detach().double().abs() * unpadded).sum(1) for i in range(count)], 1)
            signed = torch.stack([(reads[i][..., 0].detach().double() * unpadded).sum(1)
                                  for i in sorted(self.down_index.values())], 1)
        else:
            abs_c = torch.zeros((n, count), dtype=torch.float64)
            signed = torch.zeros((n, len(self.down_index)), dtype=torch.float64)
        g_down = {name: {layer: (grads[name][i][..., 0].detach().double() if grads[name] is not None else None)
                         for layer, i in self.down_index.items()} for name in self.order}
        neuron = self._pass2(ids, attention, positions, g_down, attr, metrics, untuned=untuned,
                             corrupt=corrupt, reduction=reduction)
        term_size = neuron.pop('term_size')
        sum_abs_z = neuron.pop('sum_abs_z', None)
        out = dict(example_ids=[r['id'] for r in rows], adapters=list(self.adapters),
                   metrics={k: v.detach().cpu().numpy() for k, v in metrics.items()}, stop_reasons=reasons,
                   attr=attr, neuron=neuron, term_size=term_size, sum_abs_c=abs_c.cpu().numpy(),
                   token_count=attention.sum(1).cpu().numpy(), signed_sum_c=signed.cpu().numpy(),
                   routing=dict(lora_a_calls=sum(calls.values())), padded_width=int(ids.shape[1]))
        if reduction is not None:
            out['neuron_question_ids'] = reduction['unique']
            out['sum_abs_z'] = sum_abs_z
        if raw:
            out['raw'] = dict(c={i: reads[i].detach() for i in range(len(reads))},
                              g={name: ({i: grads[name][i].detach() for i in range(count)} if grads[name] else {})
                                 for name in self.order})
        return out

    def _pass2(self, ids, attention, positions, g_down, attr, metrics, *, untuned, corrupt, reduction):
        torch = self.torch
        width = self.model.config.intermediate_size
        n = ids.shape[0]
        unpadded = attention.to(torch.float64)
        result = {name: {} for name in self.order}
        result['term_size'] = {}
        if reduction is not None:
            result['sum_abs_z'] = {}
        if untuned:
            out = {name: {l: np.zeros((n, width)) for l in self.down_index} for name in self.order}
            out['term_size'] = {l: np.zeros((n, width)) for l in self.down_index}
            return out
        layers = self.body.layers
        fired = {}
        handles = []

        def pre(module, args, layer):
            fired[layer] = fired.get(layer, 0) + 1
            z = args[0].detach().double()
            a = module.lora_A[self.adapter].weight[0].detach().double()
            abs_z = (z.abs() * unpadded[..., None]).sum(1)
            term_size = (a.abs() * abs_z).cpu().numpy()
            result['term_size'][layer] = term_size.sum(0) if reduction is not None else term_size
            if reduction is not None:
                result['sum_abs_z'][layer] = abs_z.sum(0).cpu().numpy()
            for name in self.order:
                g = g_down[name][layer]
                if g is None:
                    if reduction is not None:
                        raise InstrumentFailure('cannot reduce undefined sequence attribution')
                    result[name][layer] = np.full((n, width), np.nan)
                    continue
                value = -(a * torch.einsum('btj,bt->bj', z, g * unpadded))
                value = (value * (1.01 if corrupt else 1.0)).cpu().numpy()
                if reduction is None:
                    result[name][layer] = value
                    continue
                net = value.sum(1)
                gross = np.abs(value).sum(1)
                error = net - attr[name][:, self.down_index[layer]]
                if not np.all(np.abs(error) <= self.rtol * gross):
                    raise InstrumentFailure(f'neuron-sum identity failed for {name} layer {layer}')
                by_question = []
                for question in reduction['unique']:
                    selected = np.asarray([qid == question for qid in reduction['ids']])
                    by_question.append((value[selected] * reduction['coefficients'][selected, None]).sum(0))
                result[name][layer] = dict(
                    by_question=np.stack(by_question),
                    half_sums=reduction['halves'].T @ value,
                    identity=dict(error=error, gross=gross, net=net))

        try:
            for layer in self.down_index:
                handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(
                    lambda m, args, layer=layer: pre(m, args, layer)))
            with torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
                self.body(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if any(fired.get(l) != 1 for l in self.down_index):
            raise InstrumentFailure('down_proj pre-hook did not fire once per layer in pass 2')
        if reduction is not None:
            return result
        for name in self.order:
            undefined = np.isnan(metrics[name].detach().cpu().numpy())
            for layer, i in self.down_index.items():
                result[name][layer][undefined] = np.nan
                total, scale = result[name][layer].sum(1), np.abs(result[name][layer]).sum(1)
                ok = undefined | (np.abs(total - attr[name][:, i]) <= self.rtol * scale)
                if not ok.all():
                    raise InstrumentFailure(f'neuron-sum identity failed for {name} layer {layer}: '
                                            f'{np.abs(total - attr[name][:, i])[~ok][:3]}')
        return result


@contextmanager
def frozen_disable_adapter(model):
    """``model.disable_adapter()`` that keeps every weight's ``requires_grad`` as it was.

    In peft 0.18.0 leaving ``disable_adapter()`` marks the active adapter's weights trainable again, even if they
    were frozen before (reproduced on the tiny Qwen2 fixture); the untuned reference must not unfreeze them."""
    states = [(p, p.requires_grad) for p in model.parameters()]
    try:
        with model.disable_adapter():
            yield
    finally:
        for p, state in states:
            p.requires_grad_(state)


# --- independent fixture path: explicit scaled-branch gates ----------------------------------------

@contextmanager
def branch_gates(model, gates, adapter='default'):
    """Rebuild y = base(x) + g · s · B(A(drop(x))) explicitly for the gated adapters (index -> [B,1,1] gate)."""
    modules = adapter_modules(model)
    saved = []
    try:
        for index, gate in gates.items():
            module = modules[index][2]

            def forward(x, *args, module=module, gate=gate, **kwargs):
                base = module.base_layer(x, *args, **kwargs)
                branch = module.lora_B[adapter](module.lora_A[adapter](module.lora_dropout[adapter](x)))
                return base + gate * branch * module.scaling[adapter]

            saved.append((module, module.forward))
            module.forward = forward
        yield
    finally:
        for module, original in saved:
            module.forward = original


@contextmanager
def neuron_gate(model, neuron, adapter='default'):
    """Gate neuron j's term in a ``down_proj`` read: c' = c − (1 − g) a_j z_j. ``neuron`` = (layer, j, gate)."""
    if neuron is None:
        yield
        return
    layer, j, gate = neuron
    module = model.get_base_model().model.layers[layer].mlp.down_proj
    original = module.forward

    def forward(x, *args, **kwargs):
        base = module.base_layer(x, *args, **kwargs)
        a = module.lora_A[adapter].weight[0, j]
        c = module.lora_A[adapter](module.lora_dropout[adapter](x)) - (1 - gate) * a * x[..., j:j + 1]
        return base + module.lora_B[adapter](c) * module.scaling[adapter]

    module.forward = forward
    try:
        yield
    finally:
        module.forward = original


def branch_oracle(model, rows, name, *, stop_ids, pad_token_id, adapter='default'):
    """−dM/dg for every adapter via explicit per-example gates on the scaled branch output (fixture oracle)."""
    import torch
    inst = Instrument(model, stop_ids=stop_ids, pad_token_id=pad_token_id, identity_rtol=1.0, adapter=adapter)
    ids, attention, positions = inst.batch(rows)
    gates = {i: torch.ones((len(rows), 1, 1), requires_grad=True, device=ids.device) for i in range(len(inst.modules))}
    masks = [r['masks'] for r in rows]
    with torch.enable_grad(), branch_gates(model, gates, adapter):
        logits = model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits
        if name == 'trace':
            value = am.trace_loglik(logits, ids, masks)
        elif name == 'stop':
            value = am.stop_pull(logits, masks, stop_ids=stop_ids)[0]
        elif name == 'seqlogp':
            value = am.seqlogp(logits, ids, masks)
        else:
            raise ValueError(f'unknown metric: {name}')
        defined = ~torch.isnan(value)
        grads = torch.autograd.grad(value[defined].sum(), [gates[i] for i in range(len(gates))])
    out = -torch.stack([g.reshape(-1).double() for g in grads], dim=1).cpu().numpy()
    out[~defined.cpu().numpy()] = np.nan
    return out


def unit_blocks(out):
    """BlockStore blocks/counts/results for one (arm, batch) unit (FP32 storage of FP64 results).

    Group ``examples`` (count 1) holds trace-metric and write arrays; group ``stop`` (count 0 for an empty
    stop mask, so its rows are NaN, never 0) holds stop-pull arrays. Layer 0 for adapter-wide arrays; the
    ``down_proj`` layer for neuron arrays.
    """
    n = len(out['example_ids'])
    stop_defined = [0 if r else 1 for r in out['stop_reasons']]
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    blocks = {('examples', 0, 'attr_trace'): f32(out['attr']['trace']),
              ('examples', 0, 'sum_abs_c'): f32(out['sum_abs_c']),
              ('examples', 0, 'token_count'): f32(np.asarray(out['token_count'])[:, None]),
              ('examples', 0, 'signed_sum_c'): f32(out['signed_sum_c']),
              ('examples', 0, 'trace_loglik'): f32(np.asarray(out['metrics']['trace'])[:, None]),
              ('stop', 0, 'attr_stop'): f32(out['attr']['stop']),
              ('stop', 0, 'stop_pull'): f32(np.asarray(out['metrics']['stop'])[:, None])}
    for layer, values in out['neuron']['trace'].items():
        blocks[('examples', layer, 'neuron_trace')] = f32(values)
    for layer, values in out['term_size'].items():
        blocks[('examples', layer, 'term_size')] = f32(values)
    for layer, values in out['neuron']['stop'].items():
        blocks[('stop', layer, 'neuron_stop')] = f32(values)
    counts = dict(examples=[1] * n, stop=stop_defined)
    results = [dict(id=i, stop_reason=r, trace_loglik=float(t), stop_pull=None if r else float(s))
               for i, r, t, s in zip(out['example_ids'], out['stop_reasons'], out['metrics']['trace'],
                                     out['metrics']['stop'])]
    return blocks, counts, results
