"""Sparse rewiring ticket 11 (D21): Rank 1 vs Rank 0 comparison tests. CPU, NumPy only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import rewire_rank_comparison as rrc


class TestRankComparison(unittest.TestCase):
    def test_survivor_overlap_unique_and_layer_matched_null(self) -> None:
        shape = (4, 50)
        # Plant strong overlap in layer 0 (4 shared out of 5 each) and disjoint in layer 1
        r1 = {
            'counts': {'total': {'expected_active': 7.2, 'nonzero_z_det': 7, 'binary_survivors': 7}},
            'survivors': [{'layer': 0, 'neuron': i} for i in (0, 1, 2, 3, 4)] + [{'layer': 1, 'neuron': 10}, {'layer': 1, 'neuron': 11}],
        }
        r0 = {
            'counts': {'total': {'expected_active': 6.8, 'nonzero_z_det': 6, 'binary_survivors': 6}},
            'survivors': [{'layer': 0, 'neuron': i} for i in (0, 1, 2, 3, 9)] + [{'layer': 1, 'neuron': 20}],
        }
        res = rrc.compare_survivors(r1, r0, shape=shape, draws=400, seed=20261124)
        self.assertEqual(res['intersection_count'], 4)
        self.assertEqual(res['union_count'], 9)
        self.assertAlmostEqual(res['jaccard'], 4.0 / 9.0, places=10)
        self.assertEqual([(p['layer'], p['neuron']) for p in res['shared']], [(0, 0), (0, 1), (0, 2), (0, 3)])
        self.assertEqual([(p['layer'], p['neuron']) for p in res['unique_to_r1']], [(0, 4), (1, 10), (1, 11)])
        self.assertEqual([(p['layer'], p['neuron']) for p in res['unique_to_r0']], [(0, 9), (1, 20)])
        self.assertEqual(res['per_layer'][0]['intersection'], 4)
        self.assertEqual(res['per_layer'][1]['intersection'], 0)
        self.assertEqual(res['per_layer'][2]['jaccard'], 1.0)
        # Planted overlap of 4/5 in width 50 should have very small upper-tail p
        self.assertLess(res['layer_matched_null']['exact']['p_upper'], 1e-3)
        self.assertEqual(res['layer_matched_null']['observed'], 4)

    def test_three_way_c_sets_overlap(self) -> None:
        shape = (3, 40)
        r1 = [(0, 1), (0, 2), (0, 3), (1, 5)]
        r0 = [(0, 1), (0, 2), (0, 8), (2, 7)]
        c_sets = {
            'global64-reads': [(0, 1), (0, 2), (0, 9), (1, 5)],
            'layer64-reads': [(0, 1), (2, 7)],
        }
        res = rrc.compare_three_way_c_sets(r1, r0, c_sets, shape=shape, draws=200, seed=20261124)
        self.assertEqual(res['universe_size'], 120)
        g = res['sets']['global64-reads']
        self.assertEqual(g['r1_vs_c']['observed'], 3)
        self.assertEqual(g['r0_vs_c']['observed'], 2)
        self.assertEqual(g['three_way_vs_c']['observed'], 2)
        self.assertEqual(res['sets']['layer64-reads']['three_way_vs_c']['observed'], 1)

    def test_paired_accuracy_mcnemar_and_bootstrap(self) -> None:
        qids = [f'q{i}' for i in range(20)]
        # r1 gets 10/20 (0..9); r0 gets 14/20 (0..13) -> r0_only=4, r1_only=0, both=10, neither=6
        r1 = {q: {'reward': 1 if i < 10 else 0} for i, q in enumerate(qids)}
        r0 = {q: {'reward': 1 if i < 14 else 0} for i, q in enumerate(qids)}
        untuned = {q: {'reward': 1 if i < 8 else 0} for i, q in enumerate(qids)}
        gold = {q: {'reward': 1 if i < 12 else 0} for i, q in enumerate(qids)}

        res = rrc.compare_paired_accuracy(
            r1, r0, qids, references={'untuned': untuned, 'gold-sft': gold}, bootstrap_draws=500, seed=20261124
        )
        self.assertAlmostEqual(res['accuracy']['r1'], 0.50, places=6)
        self.assertAlmostEqual(res['accuracy']['r0'], 0.70, places=6)
        pair = res['r0_vs_r1']
        self.assertEqual(pair['both_correct'], 10)
        self.assertEqual(pair['r0_only'], 4)
        self.assertEqual(pair['r1_only'], 0)
        self.assertEqual(pair['neither'], 6)
        self.assertAlmostEqual(pair['delta_pp_b_minus_a']['estimate'], 20.0, places=6)
        self.assertAlmostEqual(pair['mcnemar_exact_p'], cg.mcnemar_exact(4, 0), places=10)
        self.assertGreater(pair['delta_pp_b_minus_a']['p2.5'], 0.0)
        self.assertIn('gold-sft', res['vs_references'])
        self.assertIn('untuned', res['vs_references'])

    def test_val_ce_hardening_and_seal_round_trip(self) -> None:
        shape = (2, 20)
        r1 = [(0, 1), (1, 2)]
        r0 = [(0, 1), (1, 3)]
        val_r1 = [{'step': 0, 'val_ce': 2.0}, {'step': 64, 'val_ce': 1.4}]
        val_r0 = [{'step': 0, 'val_ce': 2.0}, {'step': 64, 'val_ce': 1.3}]
        h_r1 = {'val_ce_before': 1.10, 'val_ce_after': 1.12, 'pre_nonzero': 5, 'post_binary': 4}
        h_r0 = {'val_ce_before': 1.05, 'val_ce_after': 1.06, 'pre_nonzero': 6, 'post_binary': 5}

        comp = rrc.compare_ranks(
            r1_export=r1,
            r0_export=r0,
            shape=shape,
            c_sets={'global64-reads': [(0, 1)]},
            val_ce=(val_r1, val_r0),
            hardening=(h_r1, h_r0),
            null_draws=100,
            bootstrap_draws=100,
        )
        self.assertAlmostEqual(comp['val_ce_and_hardening']['hardening']['r1']['delta_val_ce'], 0.02, places=6)
        self.assertAlmostEqual(comp['val_ce_and_hardening']['val_ce_trajectory'][1]['delta_r0_minus_r1'], -0.1, places=6)
        self.assertIn('lr-differs', comp['labels'])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'compare'
            self.assertEqual(rrc.seal_rank_comparison(out_dir, comp), 'written')
            self.assertEqual(au.verify_run(out_dir), {'rank_comparison.json'})
            reloaded = json.loads((out_dir / 'rank_comparison.json').read_text())
            self.assertEqual(reloaded['schema'], rrc.SCHEMA)


if __name__ == '__main__':
    unittest.main()
