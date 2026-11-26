"""Sparse rewiring ticket 15: the rehearsal launcher and the real qualification stage, end to end on CPU.

A saved tiny Qwen2 and a fixture artifact root with every bundled role (gold training rows and saved order, gold's
plan config, a real tiny gold-SFT LoRA adapter, the validation set, tokenizer and generation config) run through
the real ``rewire_launch.build`` -> ``rewire_chain`` -> ``c_attribution_chain.Chain``: preflight, the real
``rewire_qualification.Qualification`` with automatic sign-off, the real gold references in a child process,
training and evaluation of both arms, teardown. Only CUDA availability and memory counters, disk usage, the volume
identity, the Hub and the Pod stop are stubbed. The rehearsal YAML is the real one, shortened to 3 steps and the
tiny model's shapes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = get_peft_model = None
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from pilot_eval import c_attribution_bundle as cb
from pilot_eval import catalog as cat
from pilot_eval import rewire_bundle as rb
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_launch as rl
from pilot_eval import rewire_qualification as rq
from pilot_eval import sparse_rewiring_plan as sp

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'
REHEARSAL_YAML = Path(__file__).resolve().parents[1] / '.scratch' / 'sparse-rewiring-gsm8k' / 'launch-record-rehearsal.yaml'
TINY = {'model.layers': 2, 'model.hidden': 16, 'model.intermediate': 32}
EOS = 2


def _item(index, prompt, answer):
    ids = prompt + answer + [EOS]
    return dict(id=f'gsm8k:train:{index}', source_index=index, input_ids=ids, prompt_tokens=len(prompt),
                labels=[-100] * len(prompt) + answer + [EOS], gold=f'#### {index % 10}')


def _world(root: Path):
    """Tiny model, artifact root with every bundle role, a sealed bundle; returns paths and the pins."""
    art = root / 'artifacts'
    mdir = art / cb.MODEL_DIR
    mdir.mkdir(parents=True)
    vocab = {'<pad>': 0, '<bos>': 1, '<|im_end|>': 2, **{f'tok{i}': i for i in range(3, 64)}}
    backend_tok = Tokenizer(WordLevel(vocab=vocab, unk_token='<pad>'))
    backend_tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend_tok, pad_token='<pad>', bos_token='<bos>', eos_token='<|im_end|>',
                            chat_template='{% for m in messages %}{{ m.content }}{% endfor %}').save_pretrained(mdir)
    for name in ('vocab.json', 'merges.txt'):  # roles the real bundle carries; unused by the fast tokenizer
        (mdir / name).write_text('{}' if name.endswith('json') else '#version: 0.2\n')
    (mdir / 'generation_config.json').write_text(json.dumps({'eos_token_id': EOS, 'do_sample': False}))

    model_dir = root / 'model'
    cfg = json.loads(FIXTURE_CONFIG.read_text())
    cfg.update(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=64, bos_token_id=1, eos_token_id=EOS,
               torch_dtype='float32')
    torch.manual_seed(19)
    base = Qwen2ForCausalLM(Qwen2Config(**cfg)).to(torch.float32)
    base.save_pretrained(model_dir)
    saved = json.loads((model_dir / 'config.json').read_text())
    saved.pop('pad_token_id', None)
    (model_dir / 'config.json').write_text(json.dumps(saved, indent=2))

    # a real (tiny) gold-SFT LoRA adapter with nonzero B
    adapter_dir = art / rb.GOLD_RUN / 'checkpoints' / 'checkpoint-64'
    peft_model = get_peft_model(Qwen2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32),
                                LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if 'lora_B' in name:
                p.normal_(0, 0.02)
    peft_model.save_pretrained(adapter_dir)

    rows = [_item(i, [5, 6, 7 + i % 20], [8 + i % 30, 9]) for i in range(16)]
    order = [r['id'] for r in reversed(rows)]
    val = [_item(100 + i, [5, 6, 30 + i], [10 + i, 11]) for i in range(4)]
    val[-1] = _item(103, [5, 6, 40, 41, 42], [12, 13, 14])  # the longest validation row
    files = {
        rb.ROLES['gold:sft-config']: {'training_items_path': rb.ROLES['gold:training-items']},
        rb.ROLES['gold:training-items']: rows,
        rb.ROLES['gold:evaluation-items']: [dict(id=f'gsm8k:test:{i}', source_index=i, prompt=f'tok5 tok6 tok{20 + i % 40}',
                                                 gold=f'#### {i % 10}') for i in range(rb.COHORT150_SIZE)],
        rb.ROLES['gold:order']: order,
        rb.ROLES['validation:items']: val,
        rb.ROLES['validation:exclusions']: {'ids': [v['id'] for v in val],
                                            'ids_sha256': cat.ids_digest(v['id'] for v in val)},
    }
    for rel, value in files.items():
        (art / rel).parent.mkdir(parents=True, exist_ok=True)
        (art / rel).write_text(json.dumps(value))
    pinned = cb.pinned_identities(model_dir)
    bundle = cb.build(art, rb.default_entries(), model_dir=model_dir, pinned=pinned)
    bundle_path = root / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle))
    return dict(art=art, model_dir=model_dir, bundle=bundle_path, pinned=pinned,
                ids_sha256=files[rb.ROLES['validation:exclusions']]['ids_sha256'])


def _record(root: Path, w, **overrides):
    raw = yaml.safe_load(REHEARSAL_YAML.read_text())
    raw.update(volume_identity='vol-test-01', export_repo='test-org/test-private')
    raw['model'].update(layers=2, hidden=16, intermediate=32)
    raw['data'].update(training_examples=16, gold_order_steps=2)
    raw['training'].update(steps=3, warmup_steps=1, freeze_after_step=2)
    raw['checkpoints'] = [0, 2, 3]
    raw['cadences'].update(validation_ce_every=2, gate_snapshot_every=2)
    raw['cadences']['settle'].update(first=2, every=2)
    raw['evaluation'].update(cap=4, eos_token_id=EOS, batch_size=2)
    raw['evaluation']['validation']['questions'] = 4
    raw['evaluation']['cohort150']['steps'] = [3]
    raw['qualification'].update(disposable_steps=2, greedy_questions=2, cached_vs_forced_rows=2)
    for dotted, value in overrides.items():
        node = raw
        *parents, key = dotted.split('.')
        for part in parents:
            node = node[part]
        node[key] = value
    draft = root / 'draft.yaml'
    draft.write_text(yaml.safe_dump(raw, sort_keys=False))
    work = root / 'work'
    work.mkdir(exist_ok=True)
    path = sp.finalize_draft(draft, root / 'launch_record.json', code_commit='deadbeef', pod_id='pod-cpu-1',
                             fill={'inputs.artifact_root': str(w['art']), 'inputs.bundle': str(w['bundle']),
                                   'inputs.model_dir': str(w['model_dir']), 'inputs.work_dir': str(work)})
    return sp.load_launch_record(path, code_commit='deadbeef')


def _probes(w, record, *, peak=1):
    return {
        'pinned_base_model': w['pinned'],
        'disk_usage': lambda _p: shutil._ntuple_diskusage(total=int(record['disk']['container_total_gb'] * 1.1e9),
                                                          used=int(1e9), free=int(record['disk']['container_total_gb'] * 1e9)),
        'cuda_available': True,
        'volume_identity': lambda: [],
        'private_repo': lambda _repo: [],
        'driver': lambda: 'cpu-test-driver',
        'reset_peak': lambda: None,
        'peak_memory': lambda: peak,
        'total_memory': lambda: 10 ** 9,
    }


@unittest.skipUnless(LoraConfig is not None, 'peft is not installed in this environment')
class TestRehearsalLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.dict(sp.RULES, TINY)]
        for p in self.patches:
            p.start()
        self.w = _world(self.root)
        self.patches.append(mock.patch.object(rb, 'VALIDATION_IDS_SHA256', self.w['ids_sha256']))
        self.patches[-1].start()
        hf = self.root / 'hf'
        hf.mkdir()
        self.patches.append(mock.patch.dict(os.environ, {'HF_HOME': str(hf), 'HF_HUB_OFFLINE': '1'}))
        self.patches[-1].start()
        self.exports, self.stops = [], []

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _chain(self, record, *, peak=1, stop_pod=True):
        def exporter(local_root, **kwargs):
            self.exports.append(kwargs)
            return {'readback_verified': True, 'completeness': 'complete'}

        def runner(cmd, **_kw):
            self.stops.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

        backend = rc.RewireBackend(record, self.root / 'volume', device='cpu', probes=_probes(self.w, record, peak=peak))
        return rl.build(record, self.root / 'local', self.root / 'volume', backend=backend, exporter=exporter,
                        runner=runner, volume_identity='vol-test-01', current_pod='pod-cpu-1', stop_pod=stop_pod)

    def test_rehearsal_runs_preflight_qualification_references_both_arms_and_stops(self):
        record = _record(self.root, self.w)
        (self.root / 'volume').mkdir()
        summary = self._chain(record).run()
        self.assertEqual(summary['state'], 'complete', summary.get('failures'))
        self.assertEqual(summary['signoff']['decision'], 'accept')
        checks = summary['signoff']['checks']
        for arm in rl.ARM_ORDER:
            for name in ('zero_update', 'memory', 'recompute', 'restart_reproduces', 'checkpoint_reload',
                         'cached_vs_forced'):
                self.assertTrue(checks[f'{arm}:{name}'], f'{arm}:{name}')
        self.assertTrue(checks['disk'] and checks['precision'])
        local = self.root / 'local'
        qual = json.loads((local / rc.QUALIFICATION_DIR / 'summary' / 'qualification.json').read_text())
        self.assertIn('estimates', qual['details']['sparse-gold-sft-r0'])
        self.assertEqual(qual['details']['sparse-gold-sft-r1']['disposable_run']['steps'], 2)
        self.assertFalse((Path(record['inputs']['work_dir']) / 'qualification-scratch' / 'sparse-gold-sft-r1' / 'a').exists())
        refs = json.loads((local / rc.REFERENCES_DIR / 'summary' / 'references.json').read_text())
        self.assertEqual(sorted(refs), ['gold-sft', 'untuned'])
        self.assertNotEqual(refs['gold-sft']['val_ce'], refs['untuned']['val_ce'])  # the adapter really loaded
        for arm in rl.ARM_ORDER:
            run_id = record['arms'][arm]['run_id']
            self.assertTrue((local / rc.REWIRE_SUBTREE / 'runs' / run_id / rc.ARM_EVIDENCE_DIR / 'complete.json').is_file())
            vol_arm = (self.root / 'volume' / 'c-attribution' / 'sparse-rewiring-rehearsal-pod-1'
                       / rc.REWIRE_SUBTREE / 'runs' / run_id)
            self.assertTrue((vol_arm / rc.ARM_EVIDENCE_DIR / 'complete.json').is_file())
            self.assertTrue((vol_arm / rc.ARM_EVIDENCE_DIR / 'logs' / 'steps.jsonl').is_file())
            for step in record['checkpoints']:
                self.assertTrue((local / rc.REWIRE_SUBTREE / 'runs' / run_id / 'eval' / str(step) / 'greedy-val' /
                                 'complete.json').is_file())
            self.assertFalse(any((local / rc.REWIRE_SUBTREE / 'runs' / run_id / 'eval').glob('*/greedy-cohort150')))
        self.assertTrue(self.exports and self.stops)
        self.assertTrue(str(self.root / 'volume' / 'c-attribution' / 'sparse-rewiring-rehearsal-pod-1') in
                        str(next((self.root / 'volume' / 'c-attribution').iterdir())))

    def test_failed_qualification_stops_before_any_scientific_stage(self):
        record = _record(self.root, self.w)
        (self.root / 'volume').mkdir()
        summary = self._chain(record, peak=10 ** 12).run()  # peak above memory_fraction x total
        self.assertEqual(summary['state'], 'stopped')
        self.assertIn('memory', summary['stop_reason'])
        self.assertNotIn('references', summary['stages'])
        self.assertFalse(any(k.startswith('train:') for k in summary['stages']))

    def test_production_scores_cohort150_at_the_final_step_only_and_keeps_the_pod(self):
        # The tiny record with its mode switched after sealing (a real production record is 1,024 steps, which
        # ``validate`` enforces); cohort150 is gold's frozen file, checked against ticket 01's hash.
        record = dict(_record(self.root, self.w), mode='production')
        bundle = json.loads(Path(record['inputs']['bundle']).read_text())
        cohort = rb.path_of(bundle, self.w['art'], 'gold:evaluation-items')
        with self.assertRaises(cb.BundleError):   # not ticket 01's bytes
            rl.load_inputs(record)
        self.patches.append(mock.patch.dict(rb.EXPECTED, {'gold:evaluation-items': cb.sha256(cohort)}))
        self.patches[-1].start()
        (self.root / 'volume').mkdir()
        summary = self._chain(record, stop_pod=False).run()
        self.assertEqual(summary['state'], 'complete', summary.get('failures'))
        final = str(record['training']['steps'])
        for arm in rl.ARM_ORDER:
            ev = self.root / 'local' / rc.REWIRE_SUBTREE / 'runs' / record['arms'][arm]['run_id'] / 'eval'
            self.assertEqual(sorted(p.parent.name for p in ev.glob('*/greedy-cohort150')), [final])
            scores = json.loads((ev / final / 'greedy-cohort150' / 'scores.json').read_text())
            self.assertTrue(scores['exploratory'])
        responses = (ev / final / 'greedy-cohort150' / 'responses.jsonl').read_text().splitlines()
        self.assertEqual(len(responses), rb.COHORT150_SIZE)
        self.assertEqual(self.stops, [])   # no --stop-pod: backed up, verified, Pod left running
        self.assertTrue(summary['stop']['kept'] and not summary['stop']['requested'])
        self.assertIn('runpodctl stop pod pod-cpu-1', summary['stop']['reason'])

    def test_cohort150_must_be_150_gold_test_items(self):
        record = _record(self.root, self.w)
        bundle = json.loads(Path(record['inputs']['bundle']).read_text())
        cohort = rb.path_of(bundle, self.w['art'], 'gold:evaluation-items')
        items = json.loads(cohort.read_text())
        cohort.write_text(json.dumps(items[:149]))
        with mock.patch.dict(rb.EXPECTED, {'gold:evaluation-items': cb.sha256(cohort)}):
            with self.assertRaises(cb.BundleError):
                rb.check_cohort150(bundle, self.w['art'])

    def test_disposable_plan_keeps_the_frozen_steps_under_its_own_name(self):
        record = _record(self.root, self.w)
        data = rl.load_inputs(record)
        plan = data['arm_plans']['sparse-gold-sft-r1']
        self.assertNotIn('read_lr', plan)  # PR 6 plans carry it in optimizer.reads.lr
        cut = rq.disposable_plan(plan, 2)
        self.assertEqual(cut['steps'], plan['steps'][:2])
        self.assertEqual(cut['checkpoints'], [0, 2])
        self.assertEqual(cut['run_id'], plan['run_id'] + '-qualification')
        self.assertEqual(cut['plan_sha256'], sp.plan_sha256(cut))
        self.assertNotEqual(cut['plan_sha256'], plan['plan_sha256'])
        with self.assertRaises(ValueError):
            rq.disposable_plan(plan, len(plan['steps']) + 1)

    def test_bundle_validation_set_must_be_ticket_04s(self):
        record = _record(self.root, self.w)
        bundle = json.loads(Path(record['inputs']['bundle']).read_text())
        with mock.patch.object(rb, 'VALIDATION_IDS_SHA256', '0' * 64):
            with self.assertRaises(cb.BundleError):
                rb.check_validation(bundle, self.w['art'], questions=4)
        with self.assertRaises(cb.BundleError):
            rb.check_validation(bundle, self.w['art'], questions=128)


class TestRecordFiles(unittest.TestCase):
    def test_rehearsal_and_production_drafts_have_no_unresolved_leaves(self):
        for name in ('launch-record-rehearsal.yaml', 'launch-record-draft.yaml'):
            self.assertEqual(sp.draft_problems(REHEARSAL_YAML.with_name(name)), [], name)

    def test_rehearsal_draft_differs_from_production_only_in_length_and_names(self):
        reh = yaml.safe_load(REHEARSAL_YAML.read_text())
        prod = yaml.safe_load(REHEARSAL_YAML.with_name('launch-record-draft.yaml').read_text())
        differing = {k for k in prod if prod[k] != reh[k]}
        self.assertEqual(differing, {'mode', 'label', 'arms', 'data', 'training', 'checkpoints', 'cadences',
                                     'evaluation'})
        for arm in prod['arms']:
            self.assertEqual({k: v for k, v in prod['arms'][arm].items() if k != 'run_id'},
                             {k: v for k, v in reh['arms'][arm].items() if k != 'run_id'})
        self.assertEqual({k: v for k, v in prod['data'].items() if k != 'gold_order_steps'},
                         {k: v for k, v in reh['data'].items() if k != 'gold_order_steps'})


if __name__ == '__main__':
    unittest.main()


class TestCommandLine(unittest.TestCase):
    """``paths`` and ``precheck`` as the launch script calls them, on a record sealed at this checkout's commit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        raw = yaml.safe_load(REHEARSAL_YAML.read_text())
        draft = self.root / 'draft.yaml'
        draft.write_text(yaml.safe_dump(raw, sort_keys=False))
        for name in ('artifacts', 'work'):
            (self.root / name).mkdir()
        from pilot_eval import attribution_chain as pc
        self.commit = pc._commit()
        self.record_path = sp.finalize_draft(
            draft, self.root / 'launch_record.json', code_commit=self.commit, pod_id='pod-cli-1',
            fill={'inputs.artifact_root': str(self.root / 'artifacts'), 'inputs.bundle': str(self.root / 'b.json'),
                  'inputs.model_dir': str(self.root / 'model'), 'inputs.work_dir': str(self.root / 'work')})

    def tearDown(self):
        self.tmp.cleanup()

    def _main(self, *args):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {'RUNPOD_POD_ID': 'pod-cli-1'}):
            code = rl.main([*args, '--record', str(self.record_path), '--local', str(self.root / 'local'),
                            '--volume', str(self.root / 'vol')])
        return code, out.getvalue()

    def test_paths_name_the_namespaced_run_and_both_stop_forms(self):
        code, out = self._main('paths')
        self.assertEqual(code, 0)
        paths = json.loads(out)
        self.assertEqual(paths['chain_dir'],
                         str(self.root / 'vol' / 'c-attribution' / 'sparse-rewiring-rehearsal-pod-1' / 'c-attribution' / 'chain'))
        self.assertEqual(paths['stop'], ['runpodctl stop pod pod-cli-1', 'runpodctl pod stop pod-cli-1'])

    def test_precheck_accepts_a_fresh_run_and_refuses_a_finished_one(self):
        from pilot_eval import attribution_chain as pc

        with mock.patch.object(pc, 'observed_volume_identity', return_value='sl-0'):
            code, out = self._main('precheck')
            self.assertEqual(code, 0, out)
            done = self.root / 'vol' / 'c-attribution' / 'sparse-rewiring-rehearsal-pod-1' / 'c-attribution' / 'chain' / 'done'
            done.parent.mkdir(parents=True)
            done.write_text('complete\n')
            code, _ = self._main('precheck')
            self.assertEqual(code, 3)

    def test_precheck_and_watchdog_copy_refuse_ordinary_local_directory_as_volume(self):
        from pilot_eval import attribution_units as au

        local_dir = self.root / 'local'
        vol_dir = self.root / 'vol'
        vol_dir.mkdir(parents=True, exist_ok=True)
        # Write a real sealed unit on local disk (same filesystem as vol_dir)
        au.seal_run(local_dir / 'c-attribution' / 'sparse-rewiring' / 'unit', {'a.json': {'ok': True}})

        # Real observed_volume_identity (unmocked): ordinary directory must be refused before arming watchdog
        code_pre, _ = self._main('precheck')
        self.assertEqual(code_pre, 3)

        # And watchdog-copy must also refuse and NOT copy anything into the ordinary local directory
        code_wd, _ = self._main('watchdog-copy')
        self.assertEqual(code_wd, 3)
        self.assertEqual(list(vol_dir.iterdir()), [])

        # Real launch script refuses ordinary local directory before arming watchdog
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'rewire_launch.sh'
        env = dict(os.environ, RUNPOD_POD_ID='pod-cli-1', PY=os.environ.get('PY', sys.executable))
        proc = subprocess.run(
            ['bash', str(script), str(self.record_path), str(local_dir), str(vol_dir)],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn('NO RUN STARTED', proc.stderr)
        self.assertEqual(list(vol_dir.iterdir()), [])

    def test_cli_run_returns_nonzero_on_stopped_or_failed_and_zero_on_complete(self):
        from pilot_eval import attribution_chain as pc

        def _stub_chain(state):
            class _C:
                def run(self_inner):
                    return {'state': state, 'backups': {'statement': 'ok'}, 'stop': {'requested': False}}
            return _C()

        with mock.patch.object(pc, 'observed_volume_identity', return_value='sl-0'):
            for state, expected_code in (('stopped', 1), ('failed', 1), ('complete', 0)):
                with mock.patch.object(rl, 'build', return_value=_stub_chain(state)):
                    code, out = self._main('run')
                    self.assertEqual(code, expected_code, f'state={state!r} returned {code} ({out})')
                    self.assertEqual(json.loads(out)['state'], state)

        runbook = (Path(__file__).resolve().parents[1] / '.scratch' / 'sparse-rewiring-gsm8k' / 'pod-runbook.md').read_text()
        self.assertIn('set -o pipefail', runbook)

    def test_launch_script_refuses_a_missing_record_without_starting(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'rewire_launch.sh'
        proc = subprocess.run(['bash', str(script), str(self.root / 'nope.json'), str(self.root / 'local'),
                               str(self.root / 'vol')], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn('NO RUN STARTED', proc.stderr)
