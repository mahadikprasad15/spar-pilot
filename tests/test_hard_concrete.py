"""Sparse rewiring ticket 03: hard-concrete gate, shared mask cache (D14) and survivor export.

Covers spec tests 5, 6 and 10a (CPU, tiny fixtures, no model downloads).
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pilot_eval import attribution_units as au
from pilot_eval import hard_concrete as hc


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class _TinyMaskedMLP(torch.nn.Module):
    """Minimal rank-1 masked read-change MLP fixture for testing D14 shared-mask equivalence."""

    def __init__(self, d_in: int = 6, d_mid: int = 10, d_out: int = 5, seed: int = 7) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_base = torch.nn.Parameter(torch.randn(d_mid, d_in, generator=g) * 0.2, requires_grad=False)
        self.up_base = torch.nn.Parameter(torch.randn(d_mid, d_in, generator=g) * 0.2, requires_grad=False)
        self.down_base = torch.nn.Parameter(torch.randn(d_out, d_mid, generator=g) * 0.2, requires_grad=False)
        self.gate_a = torch.nn.Parameter(torch.randn(d_in, generator=g) / math.sqrt(d_in))
        self.gate_b = torch.nn.Parameter(torch.randn(d_mid, generator=g) * 0.15)
        self.up_a = torch.nn.Parameter(torch.randn(d_in, generator=g) / math.sqrt(d_in))
        self.up_b = torch.nn.Parameter(torch.randn(d_mid, generator=g) * 0.15)
        self.log_alpha = torch.nn.Parameter(torch.linspace(-1.5, 1.5, d_mid))

    def forward_with_z(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g = F.linear(x, self.gate_base) + z * ((x @ self.gate_a).unsqueeze(-1) * self.gate_b)
        u = F.linear(x, self.up_base) + z * ((x @ self.up_a).unsqueeze(-1) * self.up_b)
        return F.linear(F.silu(g) * u, self.down_base)

    def forward_cached(self, x: torch.Tensor, cache: hc.SharedMaskCache, layer: int = 0) -> torch.Tensor:
        z = cache.gate(layer, self.log_alpha, training=self.training)
        return self.forward_with_z(x, z)


class TestHardConcreteEquations(unittest.TestCase):
    """Spec test 5: hand-computed equations, Monte-Carlo check, and freeze threshold."""

    def test_hand_computed_values(self) -> None:
        beta, gamma, zeta = hc.BETA, hc.GAMMA, hc.ZETA
        vals = [-4.0, -2.5, -1.0, 0.0, 1.0, 3.0]
        la = torch.tensor(vals, dtype=torch.float32)

        # Deterministic gate
        z_det = hc.deterministic_gate(la)
        for i, v in enumerate(vals):
            expected = min(1.0, max(0.0, _sigmoid(v) * (zeta - gamma) + gamma))
            self.assertAlmostEqual(float(z_det[i].item()), expected, places=6)

        # Expected active probability
        probs = hc.expected_active_probs(la)
        offset = beta * math.log(-gamma / zeta)
        for i, v in enumerate(vals):
            expected_p = _sigmoid(v - offset)
            self.assertAlmostEqual(float(probs[i].item()), expected_p, places=6)
        self.assertAlmostEqual(float(hc.expected_active(la).item()), sum(_sigmoid(v - offset) for v in vals), places=5)

        # Stochastic gate from fixed u
        u_vals = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        u = torch.tensor(u_vals, dtype=torch.float32)
        z_stoc = hc.sample_gate_from_u(la, u)
        for i, (v, uv) in enumerate(zip(vals, u_vals)):
            s = _sigmoid((math.log(uv) - math.log(1.0 - uv) + v) / beta)
            expected_z = min(1.0, max(0.0, s * (zeta - gamma) + gamma))
            self.assertAlmostEqual(float(z_stoc[i].item()), expected_z, places=6)

    def test_expected_active_matches_monte_carlo(self) -> None:
        # Declared tolerance: 0.01 (> 4 SE for N=50,000 where max SE = 0.5/sqrt(50000) ~= 0.0022)
        la = torch.tensor([-3.0, -1.5, 0.0, 1.0, 2.5], dtype=torch.float32)
        analytic = hc.expected_active_probs(la)
        gen = torch.Generator().manual_seed(20261123)
        n_draws = 50_000
        la_rep = la.unsqueeze(0).expand(n_draws, -1)
        z = hc.sample_gate(la_rep, generator=gen)
        empirical = (z > 0.0).float().mean(dim=0)
        max_err = float((analytic - empirical).abs().max().item())
        self.assertLess(max_err, 0.01, f'Monte-Carlo error {max_err} exceeded 0.01')

    def test_freeze_threshold_at_minus_ln_11(self) -> None:
        thresh = -math.log(11.0)
        self.assertAlmostEqual(hc.FREEZE_LOG_ALPHA_THRESHOLD, thresh, places=10)
        la = torch.nn.Parameter(
            torch.tensor([thresh - 1e-2, thresh - 1e-4, thresh + 1e-4, thresh + 1e-2, -5.0, 1.0], dtype=torch.float32)
        )
        survivors = hc.freeze_log_alpha(la)
        self.assertEqual(survivors.tolist(), [False, False, True, True, False, True])
        self.assertFalse(la.requires_grad)
        z_after = hc.deterministic_gate(la)
        self.assertEqual(z_after.tolist(), [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        self.assertEqual(la.tolist(), [-20.0, -20.0, 20.0, 20.0, -20.0, 20.0])


class TestSurvivorExportAndSealing(unittest.TestCase):
    """Spec test 6: three counts, zero-based export, norms, and attribution_units sealing."""

    def test_counts_export_and_seal_pre_and_post_freeze(self) -> None:
        la0 = torch.tensor([-4.0, -2.5, -2.0, 0.5, 3.0], dtype=torch.float32)
        la1 = torch.tensor([-3.0, 1.0, 2.0, -0.1, -5.0], dtype=torch.float32)
        layers = [la0.clone(), la1.clone()]

        # Rank-1 factors for gauge-invariant norms
        a_g = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]  # norms 5.0, 2.0
        b_g = [torch.tensor([0.0, 1.0, -2.0, 0.5, 1.5]), torch.tensor([1.0, -1.0, 0.25, 0.0, 2.0])]
        ng = [hc.row_change_norms(rank=1, a=a, b=b, z=hc.deterministic_gate(la)) for a, b, la in zip(a_g, b_g, layers)]
        nu = [torch.zeros_like(la) for la in layers]

        # Pre-freeze export
        exp_pre = hc.export_survivors(layers, frozen=False, norms_gate_by_layer=ng, norms_up_by_layer=nu, step=960)
        self.assertFalse(exp_pre['frozen'])
        self.assertIsNone(exp_pre['counts']['total']['binary_survivors'])

        # Independent recomputation
        expected_total = float(hc.expected_active(la0).item() + hc.expected_active(la1).item())
        self.assertAlmostEqual(exp_pre['counts']['total']['expected_active'], expected_total, places=6)
        # Active indices: layer 0 -> {2, 3, 4}, layer 1 -> {1, 2, 3}
        self.assertEqual(exp_pre['counts']['total']['nonzero_z_det'], 6)
        self.assertEqual([(s['layer'], s['neuron']) for s in exp_pre['survivors']], [(0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3)])
        for s in exp_pre['survivors']:
            self.assertIsNone(s['binary'])
            la_val = float(layers[s['layer']][s['neuron']].item())
            self.assertAlmostEqual(s['log_alpha'], la_val, places=6)
            self.assertAlmostEqual(s['z_det'], float(hc.deterministic_gate(torch.tensor(la_val)).item()), places=6)
            self.assertAlmostEqual(s['norm_gate'], float(ng[s['layer']][s['neuron']].item()), places=6)
            self.assertEqual(s['norm_up'], 0.0)

        # Unfrozen tensors must be refused if frozen=True
        with self.assertRaises(ValueError):
            hc.count_gates(layers, frozen=True)

        # Post-freeze export + seal
        for la in layers:
            hc.freeze_log_alpha(la)
        ng_post = [hc.row_change_norms(rank=1, a=a, b=b, z=hc.deterministic_gate(la)) for a, b, la in zip(a_g, b_g, layers)]
        exp_post = hc.export_survivors(layers, frozen=True, norms_gate_by_layer=ng_post, norms_up_by_layer=nu, step=1024)
        self.assertTrue(exp_post['frozen'])
        self.assertEqual(exp_post['counts']['total']['binary_survivors'], 6)
        self.assertEqual(exp_post['counts']['total']['nonzero_z_det'], 6)
        self.assertEqual([pl['binary_survivors'] for pl in exp_post['counts']['per_layer']], [3, 3])
        self.assertTrue(all(s['binary'] is True and s['z_det'] == 1.0 for s in exp_post['survivors']))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'masks' / '1024'
            status = hc.seal_survivor_export(out_dir, exp_post, filename='1024.json')
            self.assertEqual(status, 'written')
            self.assertEqual(au.verify_run(out_dir), {'1024.json'})
            reloaded = json.loads((out_dir / '1024.json').read_text())
            self.assertEqual(reloaded['counts']['total']['binary_survivors'], 6)
            self.assertEqual([(s['layer'], s['neuron']) for s in reloaded['survivors']], [(0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3)])
            # Re-sealing identical data reuses; tampering fails verify_run
            self.assertEqual(hc.seal_survivor_export(out_dir, exp_post, filename='1024.json'), 'reused')
            (out_dir / '1024.json').write_text('{"tampered": true}\n')
            with self.assertRaises(ValueError):
                au.verify_run(out_dir)


class TestSharedMaskCacheD14(unittest.TestCase):
    """Spec test 10a: D14 shared-mask equivalence, mutation check, checkpointing, boundaries."""

    def _make_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(123)
        x = torch.randn(4, 3, 6, generator=g)
        y = torch.randint(0, 5, (4, 3), generator=g)
        return x, y

    def _run_batched(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
        model = _TinyMaskedMLP()
        z = hc.sample_gate_from_u(model.log_alpha, u)
        logits = model.forward_with_z(x, z)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum') / y.numel()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def _run_microbatch_1_shared(
        self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, *, use_checkpoint: bool = False
    ) -> dict[str, torch.Tensor]:
        model = _TinyMaskedMLP()
        cache = hc.SharedMaskCache(seed=1042, group_size=4)
        cache.begin_step(step=1, num_examples=4)
        cache.set_group_u({0: u})
        total_tokens = y.numel()
        for i in range(4):
            xi = x[i : i + 1]
            yi = y[i : i + 1]
            if use_checkpoint:
                logits = checkpoint(lambda inp: model.forward_cached(inp, cache, layer=0), xi, use_reentrant=False)
            else:
                logits = model.forward_cached(xi, cache, layer=0)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yi.reshape(-1), reduction='sum') / total_tokens
            loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def test_four_single_passes_equal_one_four_row_pass(self) -> None:
        x, y = self._make_batch()
        u = torch.empty(10).uniform_(hc.U_EPS, 1.0 - hc.U_EPS, generator=torch.Generator().manual_seed(99))
        batched_grads = self._run_batched(x, y, u)
        shared_grads = self._run_microbatch_1_shared(x, y, u, use_checkpoint=False)
        ckpt_grads = self._run_microbatch_1_shared(x, y, u, use_checkpoint=True)

        for name in batched_grads:
            self.assertTrue(
                torch.allclose(batched_grads[name], shared_grads[name], atol=1e-5, rtol=1e-5),
                f'gradient mismatch for {name}: max diff {(batched_grads[name] - shared_grads[name]).abs().max()}',
            )
            self.assertTrue(
                torch.allclose(shared_grads[name], ckpt_grads[name], atol=1e-6, rtol=1e-6),
                f'checkpoint recompute mismatch for {name}',
            )
        # Ensure log_alpha actually received a nonzero gradient
        self.assertGreater(float(shared_grads['log_alpha'].abs().sum().item()), 0.0)

    def test_mutation_redrawing_u_per_pass_fails_equivalence(self) -> None:
        x, y = self._make_batch()
        gen = torch.Generator().manual_seed(99)
        u0 = torch.empty(10).uniform_(hc.U_EPS, 1.0 - hc.U_EPS, generator=gen)
        batched_grads = self._run_batched(x, y, u0)

        # Mutated loop: draws a fresh u on every single example
        model = _TinyMaskedMLP()
        cache = hc.SharedMaskCache(seed=99, group_size=4)
        total_tokens = y.numel()
        for i in range(4):
            cache.draw_group({0: (10,)})
            logits = model.forward_cached(x[i : i + 1], cache, layer=0)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[i : i + 1].reshape(-1), reduction='sum') / total_tokens
            loss.backward()
        mutated_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        any_diff = any(not torch.allclose(batched_grads[n], mutated_grads[n], atol=1e-5, rtol=1e-5) for n in batched_grads)
        self.assertTrue(any_diff, 'redrawing u per pass should not match shared-u batched gradients')

    def test_group_boundaries_and_logging(self) -> None:
        cache = hc.SharedMaskCache(seed=1042, group_size=4)
        self.assertEqual(cache.begin_step(step=1, num_examples=8), [(0, 4), (4, 8)])
        self.assertEqual(cache.begin_step(step=2, num_examples=6), [(0, 4), (4, 6)])
        self.assertEqual(cache.begin_step(step=3, num_examples=1), [(0, 1)])
        logs = cache.step_logs
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0], {'step': 1, 'seed': 1042, 'group_size': 4, 'num_examples': 8, 'num_groups': 2, 'boundaries': [[0, 4], [4, 8]]})
        self.assertEqual(logs[1]['boundaries'], [[0, 4], [4, 6]])
        # Calling gate() without draw_group() raises
        with self.assertRaises(RuntimeError):
            cache.gate(0, torch.ones(4), training=True)
        # After freeze(), deterministic binary gate works without cached u
        cache.freeze()
        z = cache.gate(0, torch.tensor([20.0, -20.0]), training=True)
        self.assertEqual(z.tolist(), [1.0, 0.0])


class TestGaugeInvariantRowNorms(unittest.TestCase):
    """Gauge invariance and zero-signal safety of row_change_norms."""

    def test_rank1_rankk_rank0_invariance_and_zero_signal(self) -> None:
        g = torch.Generator().manual_seed(42)
        a = torch.randn(8, generator=g)
        b = torch.randn(12, generator=g)
        z = torch.rand(12, generator=g)

        n1 = hc.row_change_norms(rank=1, a=a, b=b, z=z)
        n1_scaled = hc.row_change_norms(rank=1, a=a * 3.7, b=b / 3.7, z=z)
        self.assertTrue(torch.allclose(n1, n1_scaled, atol=1e-6, rtol=1e-6))

        # Equivalent rank-0 full matrix w = b[:, None] * a[None, :]
        w = b.unsqueeze(1) * a.unsqueeze(0)
        n0 = hc.row_change_norms(rank=0, w=w, z=z)
        self.assertTrue(torch.allclose(n1, n0, atol=1e-6, rtol=1e-6))

        # Rank-k gauge invariance under (M A, B M^{-1})
        A = torch.randn(3, 8, generator=g)
        B = torch.randn(12, 3, generator=g)
        M = torch.tensor([[2.0, 0.5, -0.3], [0.0, 1.5, 0.7], [-0.4, 0.2, 1.2]])
        nk = hc.row_change_norms(rank=3, a=A, b=B, z=z)
        nk_gauge = hc.row_change_norms(rank=3, a=M @ A, b=B @ torch.linalg.inv(M), z=z)
        self.assertTrue(torch.allclose(nk, nk_gauge, atol=1e-5, rtol=1e-5))

        # Zero-signal at step 0
        zeros_r1 = hc.row_change_norms(rank=1, a=a, b=torch.zeros_like(b), z=z)
        zeros_r0 = hc.row_change_norms(rank=0, w=torch.zeros_like(w), z=z)
        self.assertTrue(torch.all(zeros_r1 == 0.0))
        self.assertTrue(torch.all(zeros_r0 == 0.0))
        self.assertTrue(torch.isfinite(zeros_r1).all())


if __name__ == '__main__':
    unittest.main()
