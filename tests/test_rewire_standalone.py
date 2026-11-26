"""Sparse rewiring ticket 12 (D22): standalone analysis tests (spec test 16). CPU, NumPy only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import rewire_standalone as rs


class TestStandaloneAnalysis(unittest.TestCase):
    """Spec test 16: settle step, participation ratio, a-cosine invariance, flip tables."""

    def test_settle_step_on_planted_trace_with_wobble(self) -> None:
        # Diagnostics: passes at 256, wobbles at 320, then passes at 384 and 448 -> settle step = 384
        diag = [
            {'step': 256, 'jaccard': 0.995, 'expected_active_rel_change': 0.01},
            {'step': 320, 'jaccard': 0.950, 'expected_active_rel_change': 0.05},
            {'step': 384, 'jaccard': 0.992, 'expected_active_rel_change': 0.015},
            {'step': 448, 'jaccard': 0.998, 'expected_active_rel_change': 0.005},
        ]
        # Snapshots (2 layers x 200): final has layer 0 [0..99] (100 neurons)
        final_mask = np.zeros((2, 200), dtype=bool)
        final_mask[0, :100] = True
        final_survivors = [(0, i) for i in range(100)]

        snap_0 = np.zeros((2, 200), dtype=bool)
        snap_0[0, :50] = True  # Jaccard = 0.50
        snap_16 = final_mask.copy()
        snap_16[0, 99] = False  # Jaccard = 99/100 = 0.99
        snap_32 = final_mask.copy()  # Jaccard = 1.00

        res = rs.compute_settle_step(
            settle_records=diag,
            snapshot_masks={0: snap_0, 16: snap_16, 32: snap_32},
            final_survivors=final_survivors,
            freeze_step=960,
        )
        self.assertEqual(res['definition_status'], 'proposed')
        self.assertEqual(res['from_diagnostics'], 384)
        self.assertEqual(res['from_snapshots'], 16)

    def test_participation_ratio_planted_rank1_and_rank3(self) -> None:
        rng = np.random.default_rng(42)
        # Planted rank-1 matrix: u v^T
        u = rng.normal(size=(8, 1))
        v = rng.normal(size=(1, 16))
        mat_r1 = u @ v
        er1 = rs.effective_rank_of_rows(mat_r1)
        self.assertEqual(er1['definition_status'], 'proposed')
        self.assertAlmostEqual(er1['participation_ratio'], 1.0, places=6)
        self.assertEqual(er1['count_for_energy'], 1)

        # Planted rank-3 matrix with 3 equal singular values -> PR = 3.0, count_for_energy(0.90) = 3
        q_left, _ = np.linalg.qr(rng.normal(size=(10, 3)))
        q_right, _ = np.linalg.qr(rng.normal(size=(16, 3)))
        mat_r3 = q_left @ (2.5 * np.eye(3)) @ q_right.T
        er3 = rs.effective_rank_of_rows(mat_r3)
        self.assertAlmostEqual(er3['participation_ratio'], 3.0, places=6)
        self.assertEqual(er3['count_for_energy'], 3)

        # Empty and all-zero rows are safe
        self.assertEqual(rs.effective_rank_of_rows(np.zeros((0, 16)))['participation_ratio'], 0.0)
        self.assertEqual(rs.effective_rank_of_rows(np.zeros((5, 16)))['participation_ratio'], 0.0)

    def test_a_cosine_sign_free_and_gauge_invariant(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=32)
        g = rng.normal(size=32)
        c_base = rs.abs_cosine(a, g)
        c_neg = rs.abs_cosine(-a, g)
        c_scaled = rs.abs_cosine(7.25 * a, -0.3 * g)
        self.assertAlmostEqual(c_base, c_neg, places=10)
        self.assertAlmostEqual(c_base, c_scaled, places=10)
        self.assertEqual(rs.abs_cosine(np.zeros(32), g), 0.0)

        out = rs.rank1_shared_direction_cosine(
            rewire_a={'gate': {0: a}, 'up': {0: g}},
            gold_lora_a={'gate': {0: -3.0 * a}, 'up': {0: g}},
            random_draws=500,
            seed=20261125,
        )
        self.assertAlmostEqual(out['rows'][0]['abs_cosine'], 1.0, places=10)
        self.assertAlmostEqual(out['rows'][1]['abs_cosine'], 1.0, places=10)
        self.assertLess(out['random_reference']['mean'], 0.25)

    def test_flip_table_on_planted_responses(self) -> None:
        qids = [f'q{i}' for i in range(12)]
        # arm: 0..7 correct (8/12); ref: 0..4 and 10 correct (6/12)
        # both: 0..4 (5), fixed: 5,6,7 (3), broken: 10 (1), neither: 8,9,11 (3)
        arm = {q: {'reward': 1 if i < 8 else 0, 'invalid': False, 'capped': False, 'n_tokens': 100} for i, q in enumerate(qids)}
        ref = {q: {'reward': 1 if (i < 5 or i == 10) else 0} for i, q in enumerate(qids)}

        ft = rs.flip_table(arm, ref, qids, arm_label='sparse-r1', ref_label='gold-sft', bootstrap_draws=400, seed=20261125)
        self.assertEqual(ft['fixed'], 3)
        self.assertEqual(ft['broken'], 1)
        self.assertEqual(ft['both'], 5)
        self.assertEqual(ft['neither'], 3)
        self.assertAlmostEqual(ft['mcnemar_exact_p'], cg.mcnemar_exact(3, 1), places=10)
        self.assertAlmostEqual(ft['delta_pp']['estimate'], 100.0 * (8 - 6) / 12.0, places=6)

        beh = rs.summarize_behaviour_responses(arm, qids)
        self.assertAlmostEqual(beh['accuracy'], 8.0 / 12.0, places=6)
        self.assertEqual(beh['invalid'], 0)
        self.assertEqual(beh['capped'], 0)

    def test_stability_change_size_activation_shift_and_seal(self) -> None:
        m0 = np.array([[True, False, False], [False, True, False]])
        m1 = np.array([[False, True, False], [False, True, False]])
        m2 = np.array([[True, True, False], [False, True, False]])
        stab = rs.survivor_stability({0: m0, 8: m1, 16: m2}, [(0, 0), (0, 1), (1, 1)])
        by_pair = {(r['layer'], r['neuron']): r for r in stab['per_survivor']}
        self.assertEqual(by_pair[(0, 0)]['first_on_step'], 0)
        self.assertEqual(by_pair[(0, 0)]['stayed_on_from_step'], 16)
        self.assertEqual(by_pair[(1, 1)]['stayed_on_from_step'], 0)

        dist = rs.per_layer_distribution([(0, 0), (0, 1), (1, 1)], num_layers=3)
        self.assertEqual(dist['per_layer_counts'], [2, 1, 0])
        self.assertAlmostEqual(dist['uniform_reference'], 1.0, places=6)

        cs_out = rs.summarize_change_sizes(
            [{'layer': 0, 'neuron': 0, 'norm_gate': 2.0, 'norm_up': 1.0}, {'layer': 1, 'neuron': 1, 'norm_gate': 0.5, 'norm_up': 0.0}],
            num_layers=2,
        )
        self.assertAlmostEqual(cs_out['per_layer'][0]['gate_over_up'], 2.0, places=6)
        self.assertIsNone(cs_out['per_layer'][1]['gate_over_up'])

        act = rs.summarize_activation_shift(np.array([-2.0, 2.0]), np.array([0.1, -0.1]))
        self.assertAlmostEqual(act['survivors']['mean'], 0.0, places=6)
        self.assertAlmostEqual(act['survivors']['mean_abs'], 2.0, places=6)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'analysis'
            self.assertEqual(rs.seal_standalone_analysis(out_dir, {'selection': dist, 'activation_shift': act}), 'written')
            self.assertEqual(au.verify_run(out_dir), {'standalone_analysis.json'})
            reloaded = json.loads((out_dir / 'standalone_analysis.json').read_text())
            self.assertEqual(reloaded['schema'], rs.SCHEMA)


if __name__ == '__main__':
    unittest.main()
