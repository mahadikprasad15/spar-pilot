"""Sparse rewiring ticket 13: training-dynamics figure tests (spec test 14 figure check)."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pilot_eval import hard_concrete as hc
from pilot_eval import rewire_dynamics as rd
from pilot_eval import rewire_dynamics_figure as rdf


def _fixture_arm(label: str, *, num_steps: int = 10, freeze_step: int = 8) -> dict:
    step_rows = []
    for step in range(num_steps):
        frozen = step >= freeze_step
        if frozen:
            la = [torch.tensor([20.0, -20.0, 20.0, -20.0]), torch.tensor([-20.0, 20.0, -20.0, -20.0])]
        else:
            shift = -0.3 * step
            la = [torch.tensor([1.0 + shift, -0.5 + shift, 1.5 + shift, -2.5]), torch.tensor([-1.0, 1.2 + shift, -2.5, -3.0])]
        scale = 0.05 * step
        ng = [torch.full((4,), scale), torch.full((4,), scale * 0.8)]
        nu = [torch.full((4,), scale * 0.5), torch.full((4,), scale * 0.4)]
        step_rows.append(
            rd.step_dynamics_row(
                step=step,
                log_alpha_by_layer=la,
                norms_gate_by_layer=ng,
                norms_up_by_layer=nu,
                frozen=frozen,
            )
        )
    return {
        'label': label,
        'step_rows': step_rows,
        'val_ce_rows': [{'step': 0, 'val_ce': 1.8}, {'step': 4, 'val_ce': 1.4}, {'step': 8, 'val_ce': 1.2}],
        'val_acc_rows': [{'step': 0, 'accuracy': 0.55}, {'step': 8, 'accuracy': 0.65}],
        'cohort150_rows': [{'step': 8, 'accuracy': 0.64}],
        'settle_rows': [{'step': 6, 'jaccard': 0.92, 'expected_active_rel_change': 0.05}, {'step': 8, 'jaccard': 0.995, 'expected_active_rel_change': 0.01}],
        'churn_rows': [{'step': 4, 'jaccard': 0.80, 'switched_on': 1, 'switched_off': 2}, {'step': 8, 'jaccard': 0.95, 'switched_on': 0, 'switched_off': 1}],
        'references': {'untuned_val_ce': 1.8, 'gold_sft_val_ce': 1.25, 'untuned_val_acc': 0.55, 'gold_sft_val_acc': 0.66},
    }


class TestDynamicsFigure(unittest.TestCase):
    def test_refuses_missing_duplicate_or_unsorted_steps(self) -> None:
        arm = _fixture_arm('sparse-gold-sft-r1')
        # Missing step 3
        arm_missing = dict(arm, step_rows=[r for r in arm['step_rows'] if r['step'] != 3])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                rdf.build_dynamics_figure(arm_missing, tmp, freeze_step=8)
            self.assertIn('missing steps', str(ctx.exception).lower())

            # Duplicate step
            arm_dup = dict(arm, step_rows=arm['step_rows'] + [arm['step_rows'][-1]])
            with self.assertRaises(ValueError):
                rdf.build_dynamics_figure(arm_dup, tmp, freeze_step=8)

            # Unsorted steps
            arm_unsorted = dict(arm, step_rows=list(reversed(arm['step_rows'])))
            with self.assertRaises(ValueError):
                rdf.build_dynamics_figure(arm_unsorted, tmp, freeze_step=8)

    def test_single_arm_and_side_by_side_outputs_and_provenance(self) -> None:
        r1 = _fixture_arm('sparse-gold-sft-r1')
        r0 = _fixture_arm('sparse-gold-sft-r0')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'dynamics'
            prov = rdf.build_dynamics_figure(r1, out, name='r1', freeze_step=8)
            for ext in ('png', 'svg', 'csv'):
                p = out / f'r1.{ext}'
                self.assertTrue(p.is_file() and p.stat().st_size > 0)
                self.assertEqual(prov['outputs'][ext]['sha256'], hashlib.sha256(p.read_bytes()).hexdigest())
            prov_disk = json.loads((out / 'r1_provenance.json').read_text())
            self.assertIn('co-movement is not evidence', prov_disk['caption'])
            self.assertEqual(prov_disk['freeze_step'], 8)

            # Side-by-side figure
            prov2 = rdf.build_side_by_side_dynamics_figure(r1, r0, out, name='both', freeze_step=8)
            self.assertEqual([a['label'] for a in prov2['arms']], ['sparse-gold-sft-r1', 'sparse-gold-sft-r0'])
            with (out / 'both.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertEqual({r['arm'] for r in rows}, {'sparse-gold-sft-r1', 'sparse-gold-sft-r0'})
            # P2.2: Panel-4 settle relative change & churn counts are persisted in plotted CSV
            r1_by_step = {int(r['step']): r for r in rows if r['arm'] == 'sparse-gold-sft-r1'}
            self.assertAlmostEqual(float(r1_by_step[6]['settle_expected_active_rel_change']), 0.05, places=6)
            self.assertAlmostEqual(float(r1_by_step[8]['settle_expected_active_rel_change']), 0.01, places=6)
            self.assertEqual(int(r1_by_step[4]['churn_switched_on']), 1)
            self.assertEqual(int(r1_by_step[4]['churn_switched_off']), 2)
            self.assertEqual(int(r1_by_step[8]['churn_switched_on']), 0)
            self.assertEqual(int(r1_by_step[8]['churn_switched_off']), 1)

    def test_p2_2_plan_binding_refuses_truncated_steps_and_missing_planned_series(self) -> None:
        arm = _fixture_arm('sparse-gold-sft-r1', num_steps=10, freeze_step=8)
        plan = {
            'settings': {'training': {'steps': 9, 'freeze_after_step': 8}},
            'validation_steps': [4, 8],
            'checkpoints': [0, 8],
            'cohort150_steps': [8],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'plan_bound'
            # Happy path: exact match with plan (including cohort150)
            prov = rdf.build_dynamics_figure(arm, out, name='ok', plan=plan, require_cohort150=True)
            self.assertEqual(prov['freeze_step'], 8)

            # Truncated step log (missing tail steps 6..9) is refused
            truncated = dict(arm, step_rows=arm['step_rows'][:6])
            with self.assertRaises(ValueError) as ctx:
                rdf.build_dynamics_figure(truncated, out, name='trunc', plan=plan)
            self.assertIn('step log', str(ctx.exception).lower())

            # Missing planned validation_ce step
            missing_val = dict(arm, val_ce_rows=[{'step': 0, 'val_ce': 1.8}, {'step': 4, 'val_ce': 1.4}])
            with self.assertRaises(ValueError) as ctx:
                rdf.build_dynamics_figure(missing_val, out, name='mval', plan=plan)
            self.assertIn('validation ce', str(ctx.exception).lower())

            # Missing planned val_acc checkpoint
            missing_acc = dict(arm, val_acc_rows=[{'step': 0, 'accuracy': 0.55}])
            with self.assertRaises(ValueError) as ctx:
                rdf.build_dynamics_figure(missing_acc, out, name='macc', plan=plan)
            self.assertIn('greedy accuracy', str(ctx.exception).lower())

            # Missing planned cohort150 step when require_cohort150=True
            missing_c150 = dict(arm, cohort150_rows=[])
            with self.assertRaises(ValueError) as ctx:
                rdf.build_dynamics_figure(missing_c150, out, name='mc150', plan=plan, require_cohort150=True)
            self.assertIn('cohort150', str(ctx.exception).lower())


if __name__ == '__main__':
    unittest.main()
