"""Sparse rewiring ticket 11 (D21): Rank 1 vs Rank 0 comparison tests. CPU, NumPy only."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_selection as sel
from pilot_eval import rewire_attribution as ra
from pilot_eval import rewire_rank_comparison as rrc


def _final_export(pairs: list[tuple[int, int]], *, frozen: bool = True, schema: str = rrc.SURVIVOR_EXPORT_SCHEMA) -> dict:
    return {
        'schema': schema,
        'frozen': frozen,
        'counts': {'total': {'expected_active': float(len(pairs)), 'nonzero_z_det': len(pairs), 'binary_survivors': len(pairs) if frozen else None}},
        'survivors': [{'layer': l, 'neuron': n} for l, n in pairs],
    }


def _sealed_c_entry(root: Path, *, arm: str = 'online-grpo', seed: int = 0) -> dict:
    L, W = ra.SHAPE
    d = np.random.default_rng(seed).normal(size=(L, W))
    d[0, 5] = np.nan  # unmeasured neuron: must be excluded, never zero
    globals_ = sel.top_positive(np.where(np.isfinite(d), d, 0.0).ravel(), 64)[0]
    groups = {
        'global64-reads': {
            'kind': 'reads',
            'targets': globals_,
            'reads': sel._reads(globals_, np.repeat(np.arange(L), W), W),
        }
    }
    buf = io.BytesIO()
    np.savez(buf, dJdlambda=d, removal_J=-d)
    au.seal_run(root / 'c-attribution/aggregate' / arm, {'neurons.npz': buf.getvalue()})
    au.seal_run(
        root / sel.FREEZE_DIR / arm,
        {f'{arm}.json': {'schema': 'c-attribution-freeze-v1', 'arm': arm, 'groups': groups}},
    )
    entry_spec = ra.c_production_entries(arms=(arm,), groups=('global64-reads',))[0]
    return ra.load_entry(root, entry_spec)


class TestRankComparison(unittest.TestCase):
    def test_survivor_overlap_unique_layer_matched_null_and_final_export_guard(self) -> None:
        shape = (4, 50)
        r1 = _final_export([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 10), (1, 11)])
        r0 = _final_export([(0, 0), (0, 1), (0, 2), (0, 3), (0, 9), (1, 20)])

        # Pre-freeze (frozen=False) export must be refused
        with self.assertRaises(ValueError):
            rrc.compare_survivors(_final_export([(0, 0)], frozen=False), r0, shape=shape, draws=50)
        # Wrong schema must be refused
        with self.assertRaises(ValueError):
            rrc.compare_survivors(_final_export([(0, 0)], schema='wrong-schema'), r0, shape=shape, draws=50)

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
        self.assertLess(res['layer_matched_null']['exact']['p_upper'], 1e-3)
        self.assertEqual(res['layer_matched_null']['observed'], 4)

    def test_three_way_attribution_routed_through_rewire_attribution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entry = _sealed_c_entry(Path(tmp), arm='online-grpo', seed=3)
            g64 = entry['frozen']['global64-reads']
            W = ra.SHAPE[1]
            # Plant 2 shared neurons from global64-reads, 1 unmeasured neuron (0, 5), plus distinct neurons
            p0, p1, p2 = (g64[0] // W, g64[0] % W), (g64[1] // W, g64[1] % W), (g64[2] // W, g64[2] % W)
            r1 = _final_export([p0, p1, p2, (0, 5)])
            r0 = _final_export([p0, p1, (0, 5), (1, 1)])

            res = rrc.compare_three_way_manifest(r1, r0, [entry], draws=200, seed=ra.NULL_SEED)
            key = 'c-production-1/online-grpo/dJdlambda'
            block = res['entries'][key]
            # Unmeasured (0, 5) is excluded, never counted as 0
            self.assertEqual(block['r1']['survivors']['unmeasured'], 1)
            self.assertEqual(block['r1']['frozen']['global64-reads']['observed'], 3)
            self.assertEqual(block['r0']['frozen']['global64-reads']['observed'], 2)
            self.assertEqual(block['three_way']['frozen']['global64-reads']['observed'], 2)
            # Matches direct rewire_attribution.compare identically (same seed, same null)
            direct_r1 = ra.compare([p0, p1, p2, (0, 5)], entry, label='sparse-gold-sft-r1', draws=200, seed=ra.NULL_SEED)
            self.assertEqual(block['r1']['seed'], direct_r1['seed'])
            self.assertEqual(block['r1']['frozen']['global64-reads'], direct_r1['frozen']['global64-reads'])

    def test_paired_accuracy_mcnemar_bootstrap_and_verified_eval_loader(self) -> None:
        qids = [f'q{i}' for i in range(20)]
        r1 = {q: {'question_id': q, 'reward': 1 if i < 10 else 0} for i, q in enumerate(qids)}
        r0 = {q: {'question_id': q, 'reward': 1 if i < 14 else 0} for i, q in enumerate(qids)}
        untuned = {q: {'reward': 1 if i < 8 else 0} for i, q in enumerate(qids)}
        gold = {q: {'reward': 1 if i < 12 else 0} for i, q in enumerate(qids)}

        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / 'eval_r1'
            jsonl = ''.join(json.dumps(r1[q]) + '\n' for q in qids)
            au.seal_run(eval_dir, {'responses.jsonl': jsonl, 'config.json': {'arm': 'r1'}})
            loaded_r1, prov = rrc.load_verified_eval_responses(eval_dir)
            self.assertEqual(len(loaded_r1), 20)
            self.assertTrue(len(prov['responses_sha256']) == 64)

            res = rrc.compare_paired_accuracy(
                loaded_r1, r0, qids, references={'untuned': untuned, 'gold-sft': gold}, bootstrap_draws=500, seed=20261124
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

            # Tampered sealed eval unit must be refused
            (eval_dir / 'responses.jsonl').write_text('{"tampered": true}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                rrc.load_verified_eval_responses(eval_dir)

    def test_val_ce_hardening_and_seal_round_trip(self) -> None:
        shape = (2, 20)
        r1 = _final_export([(0, 1), (1, 2)])
        r0 = _final_export([(0, 1), (1, 3)])
        val_r1 = [{'step': 0, 'val_ce': 2.0}, {'step': 64, 'val_ce': 1.4}]
        val_r0 = [{'step': 0, 'val_ce': 2.0}, {'step': 64, 'val_ce': 1.3}]
        h_r1 = {'val_ce_before': 1.10, 'val_ce_after': 1.12, 'pre_nonzero': 5, 'post_binary': 4}
        h_r0 = {'val_ce_before': 1.05, 'val_ce_after': 1.06, 'pre_nonzero': 6, 'post_binary': 5}

        comp = rrc.compare_ranks(
            r1_export=r1,
            r0_export=r0,
            shape=shape,
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
