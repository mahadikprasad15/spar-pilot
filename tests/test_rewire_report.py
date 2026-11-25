"""Ticket 18: report and catalog registration from a real (tiny, CPU) production chain's volume copy."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from pilot_eval import c_attribution_bundle as cb
from pilot_eval import catalog as cat
from pilot_eval import rewire_bundle as rb
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_launch as rl
from pilot_eval import rewire_report as rr
from pilot_eval import sparse_rewiring_plan as sp
from tests.test_rewire_launch import TINY, _probes, _record, _world

REPO = Path(__file__).resolve().parents[1]


class TestRewireReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = root = Path(cls.tmp.name)
        cls.patches = [mock.patch.dict(sp.RULES, TINY)]
        cls.patches[0].start()
        cls.w = w = _world(root)
        cohort = w['art'] / rb.ROLES['gold:evaluation-items']
        cls.patches += [mock.patch.object(rb, 'VALIDATION_IDS_SHA256', w['ids_sha256']),
                        mock.patch.dict(rb.EXPECTED, {'gold:evaluation-items': cb.sha256(cohort)})]
        hf = root / 'hf'
        hf.mkdir()
        cls.patches.append(mock.patch.dict(os.environ, {'HF_HOME': str(hf), 'HF_HUB_OFFLINE': '1'}))
        for p in cls.patches[1:]:
            p.start()
        record = dict(_record(root, w), mode='production')   # tiny shapes; validate() enforces 1,024 steps
        cls.record = record
        backend = rc.RewireBackend(record, root / 'volume', device='cpu', probes=_probes(w, record))
        (root / 'volume').mkdir()
        chain = rl.build(record, root / 'local', root / 'volume', backend=backend,
                         exporter=lambda _l, **_k: {'readback_verified': True, 'completeness': 'complete'},
                         runner=lambda cmd, **_k: subprocess.CompletedProcess(cmd, 0, '', ''),
                         volume_identity='vol-test-01', current_pod='pod-cpu-1')
        cls.summary = chain.run()
        cls.run_root = root / 'volume' / 'c-attribution' / rc.chain_run_id(record)   # what is on sl-0

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):
            p.stop()
        cls.tmp.cleanup()

    def test_report_from_the_volume_copy_then_register_and_verify(self):
        self.assertEqual(self.summary['state'], 'complete', self.summary.get('failures'))
        report = rr.build(self.run_root)
        final = str(self.record['training']['steps'])
        self.assertEqual(report['mode'], 'production')
        self.assertEqual(sorted(report['references']), ['gold-sft', 'untuned'])
        for arm, spec in self.record['arms'].items():
            a = report['arms'][arm]
            self.assertEqual(a['rank'], spec['rank'])
            self.assertEqual(sorted(a['checkpoints'], key=int), [str(s) for s in self.record['checkpoints']])
            self.assertEqual(list(a['cohort150']), [final])
            self.assertEqual(a['cohort150'][final]['questions'], rb.COHORT150_SIZE)
            self.assertEqual(a['survivors']['count'], sum(a['survivors']['per_layer']))
        self.assertIn('rank_comparison', report['analyses']['missing'])   # no analysis sealed in this fixture
        md = (self.run_root / rc.REWIRE_SUBTREE / 'report.md').read_text()
        self.assertIn('Cohort150 correct', md)
        self.assertIn('not available: rank_comparison', md)

        catalog = self.root / 'catalog'
        shutil.copytree(REPO / 'catalog', catalog)
        before = (catalog / 'cohorts.yaml').read_text()
        written = rr.register(self.run_root, catalog, location='sl-0',
                              prefix=f'c-attribution/{rc.chain_run_id(self.record)}')
        self.assertEqual(sorted(written['arms']), sorted(self.record['arms']))
        n_ckpt = len(self.record['checkpoints'])
        self.assertEqual(len(written['responses']), 2 + 2 * (n_ckpt + 1))
        after = (catalog / 'cohorts.yaml').read_text()
        it = iter(after.splitlines())
        self.assertTrue(all(line in it for line in before.splitlines()))   # every line, comments included, kept in order
        self.assertEqual(written['cohorts'], [rr.VAL_COHORT])
        self.assertEqual(len(written['prior_uses'][rr.TEST_COHORT]), 1)

        local = self.root / 'local-artifacts'
        src = self.w['art'] / rb.ROLES['validation:exclusions']
        (local / rr.VALIDATION_SOURCE).parent.mkdir(parents=True)
        shutil.copyfile(src, local / rr.VALIDATION_SOURCE)
        result = cat.verify(cat.load(catalog), roots={'sl-0': str(self.root / 'volume'), 'local': str(local)})
        mine = [c for c in result['checks'] if 'sparse-gold-sft' in c['entry'] or 'sparse-rewiring' in c['entry']
                or c['entry'] == f'cohort:{rr.VAL_COHORT}']
        self.assertTrue(mine)
        self.assertEqual([c for c in mine if c['status'] != 'pass'], [])
        self.assertEqual({c['entry'] for c in mine if c['entry'].startswith('arm:')},
                         {f'arm:{a}' for a in self.record['arms']})

        with self.assertRaises(rr.ReportError):   # never overwrites
            rr.register(self.run_root, catalog, location='sl-0')

    def test_tampered_eval_unit_is_refused(self):
        arm = next(iter(self.record['arms']))
        unit = (self.run_root / rc.REWIRE_SUBTREE / 'runs' / self.record['arms'][arm]['run_id'] / 'eval' /
                str(self.record['checkpoints'][-1]) / 'greedy-val')
        copy = self.root / 'tampered'
        shutil.copytree(self.run_root, copy)
        scores = copy / unit.relative_to(self.run_root) / 'scores.json'
        body = json.loads(scores.read_text())
        body['correct'] += 1
        scores.write_text(json.dumps(body))
        with self.assertRaises(ValueError):
            rr.build(copy)


if __name__ == '__main__':
    unittest.main()
