"""Rank-0 read-LR probe on CPU: the tiny Qwen2 fixture of ``test_rewire_launch``, two LRs, real training and eval."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilot_eval import rewire_bundle as rb
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_launch as rl
from pilot_eval import rewire_lr_probe as lp
from pilot_eval import sparse_rewiring_plan as sp
from tests.test_rewire_launch import TINY, _probes, _record, _world


class TestLrProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.dict(sp.RULES, TINY)]
        self.patches[0].start()
        self.w = _world(self.root)
        self.patches.append(mock.patch.object(rb, 'VALIDATION_IDS_SHA256', self.w['ids_sha256']))
        self.patches[-1].start()
        hf = self.root / 'hf'
        hf.mkdir()
        self.patches.append(mock.patch.dict(os.environ, {'HF_HOME': str(hf), 'HF_HUB_OFFLINE': '1'}))
        self.patches[-1].start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_probe_changes_only_the_read_lr_and_reports_each(self):
        record = _record(self.root, self.w)
        data = rl.load_inputs(record)
        base = data['arm_plans'][lp.ARM]
        cut = lp.probe_plan(base, 3e-5)
        self.assertEqual(cut['optimizer']['reads']['lr'], 3e-5)
        self.assertNotEqual(cut['plan_sha256'], base['plan_sha256'])
        self.assertEqual(cut['steps'], base['steps'])
        self.assertEqual({k: v for k, v in cut.items() if k not in ('optimizer', 'run_id', 'plan_sha256', 'probe_of')},
                         {k: v for k, v in base.items() if k not in ('optimizer', 'run_id', 'plan_sha256')})

        # The Pod's network volume refuses utime/chmod (2026-11-25): results must still reach it.
        self.patches.append(mock.patch('os.utime', side_effect=PermissionError(1, 'Operation not permitted')))
        self.patches[-1].start()
        self.patches.append(mock.patch('os.chmod', side_effect=PermissionError(1, 'Operation not permitted')))
        self.patches[-1].start()
        backend = rc.RewireBackend(record, self.root / 'volume', device='cpu', probes=_probes(self.w, record))
        out = self.root / 'out'
        res = lp.run_probe(record, backend, data, lrs=[3e-5, 1e-2], work=self.root / 'work-probe', out=out,
                           greedy_steps={record['checkpoints'][-1]}, code_commit='deadbeef', log=lambda _m: None)
        self.assertEqual(sorted(res['lrs']), ['0.01', '3e-05'])
        last = str(record['checkpoints'][-1])
        for row in res['lrs'].values():
            self.assertIn('accuracy', row['evals'][last])
            self.assertTrue((out / row['run_id'] / 'logs' / 'steps.jsonl').is_file())
            self.assertFalse((self.root / 'work-probe' / row['run_id'] / 'checkpoints').exists())
        self.assertNotEqual(res['lrs']['0.01']['evals'][last]['val_ce'], res['lrs']['3e-05']['evals'][last]['val_ce'])
        self.assertEqual(json.loads((out / 'lr-probe.json').read_text())['arm'], lp.ARM)


if __name__ == '__main__':
    unittest.main()
