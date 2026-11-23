"""Masked read-change ("rewire") module for the pinned Qwen2 MLP.

Clean-room implementation from the documented method only (spec D16): the equations in Max Loeffler's
``owl-rewired-neurons`` README and ``RANK0.md`` at ``max@a6dc314``
(``a6dc314740acbf726dcac435b4e84eebbbde49d8``), and the hard-concrete / L0 gate of Louizos, Welling and
Kingma 2018, "Learning Sparse Neural Networks through L0 Regularization". No code from that repository is
copied or vendored; it carries no license.

Every MLP keeps its frozen base forward ``down(act(gate(x)) * up(x))`` and adds a per-neuron, gated change to
what each neuron *reads* (its gate and up pre-activations). The write (``down_proj``) and every base weight
stay frozen. For neuron ``i`` of a layer, with gate value ``z_i`` in [0, 1]:

* rank k >= 1 (rank 1 is ``k = 1``): ``Δpart = z ⊙ ((x Aᵀ) Bᵀ)`` with ``A`` [k, d] shared read directions and
  ``B`` [n, k] per-neuron coefficients. At rank 1 this is ``z ⊙ ((x·a) b)``.
* rank 0 (full rows, the intended method in ``RANK0.md``): ``Δpart = z ⊙ (x Wᵀ)`` with ``W`` [n, d].

Parts are gate and up (researcher decision). One gate value per neuron is shared by both parts.

Init (spec, D10): ``A ~ N(0, 1)/√d`` drawn from a CPU ``torch.Generator`` seeded ``seed + layer``, gate part
first, then up; ``B = 0`` and ``W = 0`` so the initial model equals the base model; ``logα = mask_init``.

The gate values ``z`` come from a pluggable ``gate`` callable (``log_alpha -> z``). The default is the
deterministic hard-concrete gate ``clip(σ(logα)(ζ−γ)+γ, 0, 1)``; the stochastic training gate with the
shared per-group noise cache is ticket 03 (``hard_concrete.py``), which installs itself through
``set_gate``.

The change is applied at every position the MLP sees. Under microbatch 1 there is no padding; under padded
batches the attention mask already keeps padded positions from reaching unpadded ones.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

PARTS = ('gate', 'up')
PARAMETER_TAG = 'rewire_'  # every new parameter's name contains this; base names are unchanged

# Hard-concrete constants (Louizos et al. 2018; values used at max@a6dc314).
BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1


def deterministic_gate(log_alpha):
    """``clip(σ(logα)(ζ−γ)+γ, 0, 1)``: the evaluation-time gate. Python-float constants only, so it never
    creates a tensor on another device."""
    return torch.clamp(torch.sigmoid(log_alpha) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def is_rewire_parameter(name):
    return PARAMETER_TAG in name


class RewireMLP(nn.Module):
    """Drop-in replacement for one ``Qwen2MLP``.

    The base submodules are re-registered under their original names (``gate_proj``, ``up_proj``,
    ``down_proj``, ``act_fn``), so base parameter names, the state dict of the base model and
    ``sft_backend.frozen_weight_hash`` (with rewire parameters excluded) are unchanged by installation.
    """

    def __init__(self, mlp, layer, *, rank, seed=42, mask_init=1.0, gate=None):
        super().__init__()
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f'rank must be a non-negative integer, got {rank!r}')
        for name in ('gate_proj', 'up_proj', 'down_proj', 'act_fn'):
            if not hasattr(mlp, name):
                raise ValueError(f'not a Qwen2 MLP: missing {name}')
        self.gate_proj, self.up_proj, self.down_proj, self.act_fn = (
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp.act_fn)
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            if getattr(module, 'bias', None) is not None:
                raise ValueError('the pinned Qwen2 MLP projections have no bias')
        weight = self.gate_proj.weight
        neurons, hidden = weight.shape
        if tuple(self.up_proj.weight.shape) != (neurons, hidden) or \
                tuple(self.down_proj.weight.shape) != (hidden, neurons):
            raise ValueError('gate, up and down projection shapes do not form a Qwen2 MLP')
        self.layer, self.rank, self.seed = layer, rank, seed
        self.hidden_size, self.intermediate_size = hidden, neurons
        self.enabled = True
        self.gate = gate if gate is not None else deterministic_gate
        device, dtype = weight.device, weight.dtype
        if rank == 0:
            for part in PARTS:
                self.register_parameter(f'rewire_W_{part}',
                                        nn.Parameter(torch.zeros(neurons, hidden, device=device, dtype=dtype)))
        else:
            generator = torch.Generator(device='cpu').manual_seed(seed + layer)
            for part in PARTS:  # fixed draw order: gate, then up
                directions = torch.randn(rank, hidden, generator=generator, dtype=torch.float32) / math.sqrt(hidden)
                self.register_parameter(f'rewire_A_{part}', nn.Parameter(directions.to(device=device, dtype=dtype)))
                self.register_parameter(f'rewire_B_{part}',
                                        nn.Parameter(torch.zeros(neurons, rank, device=device, dtype=dtype)))
        self.rewire_log_alpha = nn.Parameter(torch.full((neurons,), float(mask_init), device=device, dtype=dtype))

    # --- read changes --------------------------------------------------------------------------------

    def read_change(self, x, part):
        """Ungated change to ``part``'s pre-activation, shape ``x.shape[:-1] + (n,)``."""
        if self.rank == 0:
            return x @ getattr(self, f'rewire_W_{part}').T
        return (x @ getattr(self, f'rewire_A_{part}').T) @ getattr(self, f'rewire_B_{part}').T

    def gate_values(self):
        return self.gate(self.rewire_log_alpha)

    def pre_activations(self, x):
        """``(gate, up)`` pre-activations, each ``x.shape[:-1] + (n,)``, with the gated read change added when on."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.enabled:
            z = self.gate_values()
            gate = gate + z * self.read_change(x, 'gate')
            up = up + z * self.read_change(x, 'up')
        return gate, up

    def forward(self, x):
        gate, up = self.pre_activations(x)
        return self.down_proj(self.act_fn(gate) * up)

    def coefficients(self):
        """Per-part coefficient tensors that are zero exactly when the read change is zero (B or W)."""
        letter = 'W' if self.rank == 0 else 'B'
        return {part: getattr(self, f'rewire_{letter}_{part}') for part in PARTS}

    def extra_repr(self):
        return f'layer={self.layer}, rank={self.rank}, hidden={self.hidden_size}, neurons={self.intermediate_size}'


# --- model-level helpers -----------------------------------------------------------------------------

def _decoder_layers(model):
    for path in (('model', 'layers'), ('base_model', 'model', 'model', 'layers')):
        node = model
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        return node
    raise ValueError('cannot find the decoder layers of this model')


def install(model, *, rank, seed=42, mask_init=1.0, gate=None):
    """Replace every layer's MLP by a ``RewireMLP``, freeze every base parameter, and return the modules.

    Only the new ``rewire_`` parameters require grad afterwards. Refuses a model that is already rewired.
    """
    layers = _decoder_layers(model)
    if any(isinstance(layer.mlp, RewireMLP) for layer in layers):
        raise ValueError('model is already rewired')
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    modules = []
    for index, layer in enumerate(layers):
        layer.mlp = RewireMLP(layer.mlp, index, rank=rank, seed=seed, mask_init=mask_init, gate=gate)
        modules.append(layer.mlp)
    return modules


def modules(model):
    found = [layer.mlp for layer in _decoder_layers(model) if isinstance(layer.mlp, RewireMLP)]
    if not found:
        raise ValueError('model has no rewire modules')
    return found


def set_gate(model, gate):
    """Install one gate callable on every module (ticket 03 plugs the stochastic gate in here)."""
    for module in modules(model):
        module.gate = gate if gate is not None else deterministic_gate


def trainable_parameters(model):
    """``(name, parameter)`` for every rewire parameter, sorted by name."""
    return sorted((name, p) for name, p in model.named_parameters() if is_rewire_parameter(name))


@contextlib.contextmanager
def off(model):
    """Every module returns the exact base MLP inside the context; the previous state is restored after."""
    found = modules(model)
    previous = [m.enabled for m in found]
    for m in found:
        m.enabled = False
    try:
        yield model
    finally:
        for m, state in zip(found, previous):
            m.enabled = state


def zero_change(model, inputs):
    """Zero-change hook for ``sft_backend.HFTrainingEngine.preflight``: logits with the modules on equal the
    logits with them off, bit for bit, and every coefficient tensor is exactly zero."""
    with torch.no_grad():
        rewired = model(input_ids=inputs, use_cache=False).logits
        with off(model):
            base = model(input_ids=inputs, use_cache=False).logits
    zero = all(torch.count_nonzero(c).item() == 0 for m in modules(model) for c in m.coefficients().values())
    return bool(torch.equal(rewired, base)) and zero


def parameter_counts(hidden_size, intermediate_size, layers, rank):
    """Trainable read parameters and mask logits implied by the shapes (spec D13)."""
    per_part = intermediate_size * hidden_size if rank == 0 else rank * (hidden_size + intermediate_size)
    return {'read': layers * len(PARTS) * per_part, 'mask': layers * intermediate_size}
