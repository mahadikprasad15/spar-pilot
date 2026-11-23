"""Sparse rewiring ticket 06: dynamics logs and settle diagnostics (spec tests 14 and 18).

CPU fixture tests, no model downloads.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_dynamics as rd


class TestStepDynamicsAndSnapshots(unittest.TestCase):
    """Spec test 14 (logging, snapshots, gauge invariance, churn) and zero-signal safety."""

    def test_logged_counts_match_recomputation_from_saved_snapshot(self) -> None:
        la = [
            torch.tensor([-4.0, -2.8, -1.5, 0.5, 2.5], dtype=torch.float32),
            torch.tensor([-3.5, 0.2, 1.2, -0.8, -5.0], dtype=torch.float32),
        ]
        a = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]
        b = [torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75]), torch.tensor([1.0, 0.5, -0.5, 1.5, 0.0])]
        ng = [hc.row_change_norms(rank=1, a=ai, b=bi, z=hc.deterministic_gate(li)) for ai, bi, li in zip(a, b, la)]
        nu = [hc.row_change_norms(rank=1, a=ai, b=bi * 0.5, z=hc.deterministic_gate(li)) for ai, bi, li in zip(a, b, la)]

        row = rd.step_dynamics_row(
            step=16,
            log_alpha_by_layer=la,
            norms_gate_by_layer=ng,
            norms_up_by_layer=nu,
            frozen=False,
            ce_loss=1.23,
            l0_loss=0.04,
            total_loss=1.27,
            lr_reads=3e-3,
            lr_masks=0.1,
            grad_norm_reads=0.8,
            grad_norm_masks=0.2,
            examples_seen=128,
        )

        with tempfile.TemporaryDirectory() as tmp:
            snap_path = rd.save_log_alpha_snapshot(Path(tmp) / 'masks/trace/16.npz', la, step=16)
            loaded = rd.load_log_alpha_snapshot(snap_path)
            self.assertEqual(loaded['step'], 16)
            self.assertEqual(loaded['layers'], [0, 1])
            self.assertEqual(loaded['log_alpha'].dtype, np.float16)

            recomputed = rd.recompute_counts_from_snapshot(loaded['log_alpha'], frozen=False)
            self.assertAlmostEqual(row['expected_active'], recomputed['total']['expected_active'], places=2)
            self.assertEqual(row['nonzero_z_det'], recomputed['total']['nonzero_z_det'])
            self.assertIsNone(row['binary_survivors'])
            self.assertEqual(row['per_layer_nonzero'], [pl['nonzero_z_det'] for pl in recomputed['per_layer']])

            # JSONL round-trip
            log_path = Path(tmp) / 'logs/steps.jsonl'
            rd.append_jsonl(log_path, row)
            rows = rd.read_jsonl(log_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['step'], 16)
            self.assertEqual(rows[0]['nonzero_z_det'], 6)

    def test_norm_summaries_are_gauge_invariant(self) -> None:
        la = [torch.tensor([-3.0, 0.5, 1.5, -0.5], dtype=torch.float32)]
        a = torch.tensor([1.2, -0.7, 2.3])
        b = torch.tensor([0.5, -1.5, 0.8, 2.0])
        z = hc.deterministic_gate(la[0])

        ng1 = [hc.row_change_norms(rank=1, a=a, b=b, z=z)]
        ng2 = [hc.row_change_norms(rank=1, a=a * 4.5, b=b / 4.5, z=z)]
        row1 = rd.step_dynamics_row(step=1, log_alpha_by_layer=la, norms_gate_by_layer=ng1, norms_up_by_layer=ng1)
        row2 = rd.step_dynamics_row(step=1, log_alpha_by_layer=la, norms_gate_by_layer=ng2, norms_up_by_layer=ng2)
        for group in ('gate_all', 'gate_survivors', 'up_all', 'up_survivors'):
            for key in ('sum', 'median', 'max'):
                self.assertAlmostEqual(row1['norms'][group][key], row2['norms'][group][key], places=6)

    def test_zero_signal_step_0_and_empty_survivors_are_finite(self) -> None:
        # All gates dead (z_det == 0) and all read changes zero
        la_dead = [torch.full((5,), -10.0, dtype=torch.float32)]
        zeros = [torch.zeros(5, dtype=torch.float32)]
        row = rd.step_dynamics_row(step=0, log_alpha_by_layer=la_dead, norms_gate_by_layer=zeros, norms_up_by_layer=zeros)
        self.assertEqual(row['nonzero_z_det'], 0)
        self.assertEqual(row['norms']['gate_survivors'], {'count': 0, 'sum': 0.0, 'median': 0.0, 'max': 0.0})
        self.assertEqual(row['norms']['up_survivors'], {'count': 0, 'sum': 0.0, 'median': 0.0, 'max': 0.0})
        # Empty vs empty churn has well-defined Jaccard = 1.0
        empty = np.zeros((2, 5), dtype=bool)
        churn = rd.mask_churn(empty, empty)
        self.assertEqual(churn['jaccard'], 1.0)
        self.assertEqual(churn['switched_on'], 0)
        self.assertEqual(churn['switched_off'], 0)

    def test_mask_churn_planted_pair(self) -> None:
        prev = np.array([[True, True, False, False], [True, False, False, False]])
        curr = np.array([[True, False, True, False], [True, False, True, False]])
        # intersection = 2 ((0,0), (1,0)), union = 5, switched_on = 2 ((0,2), (1,2)), switched_off = 1 ((0,1))
        churn = rd.mask_churn(prev, curr)
        self.assertEqual(churn['prev_count'], 3)
        self.assertEqual(churn['curr_count'], 4)
        self.assertEqual(churn['intersection'], 2)
        self.assertEqual(churn['union'], 5)
        self.assertEqual(churn['switched_on'], 2)
        self.assertEqual(churn['switched_off'], 1)
        self.assertAlmostEqual(churn['jaccard'], 2.0 / 5.0, places=10)


class TestSettleDiagnostics(unittest.TestCase):
    """Spec test 18: cadence (256, then every 64), exact values, and zero side effects on training."""

    def test_cadence_and_recomputation_on_planted_trace(self) -> None:
        tracker = rd.SettleTracker()
        # Planted trace: layers=2, width=100; stable after step 256
        base = torch.full((2, 100), -5.0, dtype=torch.float32)
        base[0, :50] = 2.0
        base[1, :50] = 2.0

        snapshots: dict[int, torch.Tensor] = {}
        check_steps: list[int] = []
        for step in range(0, 449):
            cur = base.clone()
            if step == 192:
                # 100 active
                pass
            elif step == 256:
                # Turn off 10 neurons -> Jaccard = 90/100 = 0.90 < 0.99
                cur[0, 40:50] = -5.0
            elif step >= 320:
                # Same 90 neurons -> Jaccard = 1.0, rel_change = 0.0 -> meets_reference = True
                cur[0, 40:50] = -5.0
            snapshots[step] = cur
            rec = tracker.observe(step, [cur[0], cur[1]])
            if rec is not None:
                check_steps.append(step)

        self.assertEqual(check_steps, [256, 320, 384, 448])
        recs = tracker.records
        # Check 256 vs 192
        self.assertEqual(recs[0]['prev_step'], 192)
        self.assertAlmostEqual(recs[0]['jaccard'], 0.90, places=6)
        self.assertEqual(recs[0]['switched_off'], 10)
        self.assertEqual(recs[0]['switched_on'], 0)
        self.assertFalse(recs[0]['meets_reference'])
        self.assertTrue(recs[0]['diagnostic_only'])
        # Independent expected_active_rel_change check
        e192 = float(hc.expected_active(snapshots[192]).item())
        e256 = float(hc.expected_active(snapshots[256]).item())
        self.assertAlmostEqual(recs[0]['expected_active_rel_change'], abs(e256 - e192) / e192, places=6)

        # Check 320 vs 256 (identical -> settled)
        self.assertEqual(recs[1]['prev_step'], 256)
        self.assertAlmostEqual(recs[1]['jaccard'], 1.0, places=6)
        self.assertAlmostEqual(recs[1]['expected_active_rel_change'], 0.0, places=6)
        self.assertTrue(recs[1]['meets_reference'])

    def test_training_trajectory_bit_identical_with_diagnostics_on_vs_off(self) -> None:
        def _run(enable_diagnostics: bool) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
            torch.manual_seed(77)
            w = torch.nn.Parameter(torch.randn(4, 3))
            la = torch.nn.Parameter(torch.linspace(-1.0, 2.0, 4))
            opt = torch.optim.AdamW([w, la], lr=0.05)
            tracker = rd.SettleTracker(start=4, interval=2) if enable_diagnostics else None
            losses: list[float] = []
            for step in range(1, 9):
                opt.zero_grad()
                x = torch.randn(2, 3)
                z = hc.sample_gate(la)
                loss = ((x @ w.T) * z).pow(2).mean() + 0.01 * hc.expected_active(la)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                if tracker is not None:
                    tracker.observe(step, [la])
            return w.detach().clone(), la.detach().clone(), losses

        w_off, la_off, l_off = _run(False)
        w_on, la_on, l_on = _run(True)
        self.assertTrue(torch.equal(w_off, w_on))
        self.assertTrue(torch.equal(la_off, la_on))
        self.assertEqual(l_off, l_on)


if __name__ == '__main__':
    unittest.main()
