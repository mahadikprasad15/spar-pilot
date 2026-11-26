"""CPU unit & regression tests for Sampling SFT (k = 2) plan, paired activation shifts, HTML report, and Gold analysis.

Covers:
  - Gold v1 launch-record backward compatibility
  - K2 launch-record sealing, executable_sha256 binding, clean-tree git guard (P1 #2)
  - K2 sealed export directory verification & refusal tests (P1 #1)
  - K2-D1 headroom-scaled lambda & < 4 trigger, K2-D3 64-step/epoch batching across 16 epochs
  - Short-batch [3..8] training matching FP64 completion CE oracle
  - Two-layer paired-forward activation shift sign regression (P2 #6)
  - Comparative Gold-vs-K2 HTML report builder on 2-survivor fixture + Gold HTML regression (P2 #7)
  - Committed Gold rewire_analysis stale standalone rebuild on --gold-adapter & activation-shift (P2 #8)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'notebooks') not in sys.path:
    sys.path.insert(0, str(REPO / 'notebooks'))
if str(REPO / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO / 'scripts'))

from pilot_eval import attribution_units as au
from pilot_eval import rewire_activation_shift as ras
from pilot_eval import rewire_analysis as ran
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_training as rt
from pilot_eval import sparse_rewiring_k2_plan as k2p
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.attribution_chain import LaunchRefused
from pilot_eval.training import file_hash
import build_down_proj_report as bdr
import decode_down_proj as ddp
import extract_survivor_shifts as ess

K2_DRAFT = REPO / '.scratch/sparse-rewiring-sampling-k2/launch-record-k2-matched-draft.yaml'
K2_SCALED_DRAFT = REPO / '.scratch/sparse-rewiring-sampling-k2/launch-record-k2-scaled-draft.yaml'
GOLD_DRAFT = REPO / '.scratch/sparse-rewiring-gsm8k/launch-record-draft.yaml'
GOLD_CACHE = REPO / 'notebooks/cache/down_proj_decoded.pt'

FILL = {
    'inputs.artifact_root': '/w/inputs',
    'inputs.bundle': '/w/b.json',
    'inputs.model_dir': '/w/model',
    'inputs.work_dir': '/w',
    'inputs.val128_k2_targets': '/w/val128_k2.json',
}


def _signed_k2(draft_path: Path = K2_DRAFT, **overrides):
    rec = yaml.safe_load(draft_path.read_text())
    for k, text in rec['decisions'].items():
        rec['decisions'][k] = text.replace('[proposed]', '[user test]')
    for dotted, val in overrides.items():
        cur = rec
        keys = dotted.split('.')
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    return rec


def _synthetic_k2_ids(seed: int = 7):
    all_ids = [f'gsm8k:train:{i}' for i in sorted(random.Random(seed).sample(range(7473), 512))]
    gold_order = list(all_ids)
    random.Random(42).shuffle(gold_order)
    rng = random.Random(99)
    uncovered = set()
    for b_idx, n_drop in [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)]:
        slot = gold_order[b_idx * 8:(b_idx + 1) * 8]
        uncovered.update(slot[:n_drop])
    remaining_candidates = [eid for eid in gold_order[40:] if eid not in uncovered]
    uncovered.update(rng.sample(remaining_candidates, 35 - len(uncovered)))
    covered = [eid for eid in all_ids if eid not in uncovered]
    assert len(covered) == 477 and len(uncovered) == 35
    return covered, gold_order


class _ToyMLP(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, n, bias=False)
        self.up_proj = nn.Linear(d, n, bias=False)
        self.down_proj = nn.Linear(n, d, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _ToyLayer(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.mlp = _ToyMLP(d, n)

    def forward(self, x):
        return x + self.mlp(x)


class _ToyInner(nn.Module):
    def __init__(self, vocab: int, d: int, n: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_ToyLayer(d, n) for _ in range(layers)])


class _ToyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16, n: int = 24, layers: int = 2):
        super().__init__()
        self.model = _ToyInner(vocab, d, n, layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return type('Out', (), {'logits': logits})()


class _ToyTok:
    def decode(self, ids, **_kw):
        return ' '.join(f't{i}' for i in ids)


class TestSparseRewiringK2(unittest.TestCase):
    def test_v1_gold_draft_untouched(self):
        self.assertEqual(sp.draft_problems(GOLD_DRAFT), [])

    def test_k2_draft_refused_until_proposed_decisions_signed(self):
        probs = k2p.draft_problems(K2_DRAFT)
        self.assertEqual(sorted(p.split(' ')[0] for p in probs), ['decisions.K2-D4', 'decisions.K2-D5'])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LaunchRefused):
                k2p.finalize_draft(K2_DRAFT, Path(td) / 'r.json', code_commit='abc123', pod_id='pod1', fill=FILL)

    def test_k2_signed_draft_seals_detects_tamper_and_binds_executable_sha(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            draft = td_p / 'signed.yaml'
            draft.write_text(yaml.safe_dump(_signed_k2(), sort_keys=False))
            out = k2p.finalize_draft(draft, td_p / 'launch.json', code_commit='abc123', pod_id='pod1', fill=FILL)
            rec = k2p.load_launch_record(out, code_commit='abc123')
            self.assertEqual(rec['objective']['l0_lambda'], 6e-4)
            self.assertEqual(rec['executable_sha256'], k2p.executable_code_sha256())
            # Changed executable source digest must be refused
            with self.assertRaises(LaunchRefused):
                k2p.load_launch_record(out, code_commit='abc123', executable_sha256='0' * 64)
            # Tampered JSON must be refused
            tampered = json.loads(out.read_text())
            tampered['objective']['l0_lambda'] = 2e-4
            out.write_text(json.dumps(tampered))
            with self.assertRaises(LaunchRefused):
                k2p.load_launch_record(out, code_commit='abc123')

    def test_p1_2_clean_git_guard_refuses_untracked_or_modified_executable(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
            all_files = list(k2p.EXECUTABLE_SOURCES) + ['pilot_eval/sft_backend.py', 'pilot_eval/rewire_dynamics.py']
            for rel in all_files:
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f'# {rel}\n', encoding='utf-8')
            # Commit all except the launcher (untracked launcher -> must refuse!)
            tracked = [r for r in all_files if r != 'pilot_eval/rewire_k2_launch.py']
            subprocess.run(['git', 'add', *tracked], cwd=repo, check=True)
            subprocess.run(['git', 'commit', '-q', '-m', 'partial'], cwd=repo, check=True)
            with self.assertRaises(LaunchRefused):
                k2p.verified_code_identity(repo, require_clean=True)
            # Commit the launcher too -> clean tree succeeds, even with unrelated local cache file!
            subprocess.run(['git', 'add', 'pilot_eval/rewire_k2_launch.py'], cwd=repo, check=True)
            subprocess.run(['git', 'commit', '-q', '-m', 'complete'], cwd=repo, check=True)
            (repo / 'notebooks' / 'cache').mkdir(parents=True)
            (repo / 'notebooks' / 'cache' / 'local.pt').write_bytes(b'cache')
            ident = k2p.verified_code_identity(repo, require_clean=True)
            self.assertEqual(len(ident['code_commit']), 40)
            # Modify a shared runtime dependency (sft_backend.py) -> MUST refuse AND change executable_sha256!
            (repo / 'pilot_eval/sft_backend.py').write_text('# changed completion_loss denominator\n', encoding='utf-8')
            self.assertNotEqual(k2p.executable_code_sha256(repo), ident['executable_sha256'])
            with self.assertRaises(LaunchRefused):
                k2p.verified_code_identity(repo, require_clean=True)

    def test_p1_4_production_locks_preregistered_hyperparameters_and_exact_seeds(self):
        """Production sealing refuses off-spec mask LR, data_order_base, or replication seed (e.g. 999)."""
        bad_cases = [
            {'optimizer.masks.lr': 0.01},
            {'seeds.data_order_base': 999},
            {'arms.sparse-k2-sft-r1-matched-s43.seed': 999},
            {'hard_concrete.log_alpha_init': 2.5},
            {'optimizer.reads.clip_norm': 5.0},
        ]
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            for idx, overrides in enumerate(bad_cases):
                rec = _signed_k2(**overrides)
                draft = td_p / f'bad_prod_{idx}.yaml'
                draft.write_text(yaml.safe_dump(rec, sort_keys=False))
                with self.assertRaises(LaunchRefused, msg=f'should refuse {overrides}'):
                    k2p.finalize_draft(draft, td_p / f'bad_{idx}.json', code_commit='abc123', pod_id='pod1', fill=FILL)

    def test_k2_scaled_lambda_requires_routing_and_run_b_trigger(self):
        cal = k2p.compute_scaled_lambda(ce_untuned_k2=0.3600, ce_lora_k2=0.3100)
        expected = 6.0e-4 * (0.0500 / k2p.GOLD_VAL_CE_HEADROOM)
        self.assertAlmostEqual(cal['lambda_scaled'], expected, places=9)
        self.assertTrue(k2p.should_trigger_run_b(0))
        self.assertTrue(k2p.should_trigger_run_b(3))
        self.assertFalse(k2p.should_trigger_run_b(4))
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Scaled draft without routing.gate_receipt / headroom_summary must be refused
            bad_rec = _signed_k2(**{
                'label': 'production-k2-scaled',
                'objective.lambda_regime': 'scaled',
                'objective.l0_lambda': float(cal['lambda_scaled']),
            })
            bad_draft = td_p / 'bad_scaled.yaml'
            bad_draft.write_text(yaml.safe_dump(bad_rec, sort_keys=False))
            with self.assertRaises(LaunchRefused):
                k2p.finalize_draft(bad_draft, td_p / 'bad.json', code_commit='abc123', pod_id='pod1', fill=FILL)
            # With routing paths filled, schema validation succeeds
            good_rec = _signed_k2(K2_SCALED_DRAFT)
            good_draft = td_p / 'good_scaled.yaml'
            good_draft.write_text(yaml.safe_dump(good_rec, sort_keys=False))
            scaled_fill = dict(
                FILL,
                **{
                    'objective.l0_lambda': float(cal['lambda_scaled']),
                    'routing.gate_receipt': '/w/gate_receipt.json',
                    'routing.headroom_summary': '/w/references/summary',
                },
            )
            out = k2p.finalize_draft(good_draft, td_p / 'scaled.json', code_commit='abc123', pod_id='pod1',
                                     fill=scaled_fill)
            loaded = k2p.load_launch_record(out, code_commit='abc123')
            self.assertAlmostEqual(loaded['objective']['l0_lambda'], expected, places=9)

    def test_k2_batching_477_items_across_16_epochs_and_two_seeds(self):
        covered, gold_order = _synthetic_k2_ids()
        rec = _signed_k2()
        rec.update(code_commit='abc123', executable_sha256=k2p.executable_code_sha256(), pod_id='pod1')
        rec['inputs'].update({k.split('.')[1]: v for k, v in FILL.items()})
        k2p.validate(rec, code_commit='abc123')

        p42 = k2p.build_plan(rec, 'sparse-k2-sft-r1-matched-s42', covered, gold_order,
                             inputs_sha256={'k2_data_sha256': 'deadbeef'})
        p43 = k2p.build_plan(rec, 'sparse-k2-sft-r1-matched-s43', covered, gold_order,
                             inputs_sha256={'k2_data_sha256': 'deadbeef'})
        self.assertEqual(len(p42['steps']), 1024)
        ep1 = p42['steps'][:64]
        self.assertEqual(sum(len(s['ids']) for s in ep1), 477)
        self.assertEqual(len(ep1[0]['ids']), 3)
        self.assertEqual(ep1[0]['mask_groups'], [[0, 3]])
        for ep in range(2, 17):
            ep_steps = p42['steps'][(ep - 1) * 64:ep * 64]
            sizes = [len(s['ids']) for s in ep_steps]
            self.assertEqual(sizes.count(8), 29)
            self.assertEqual(sizes.count(7), 35)
            self.assertEqual(sorted(eid for s in ep_steps for eid in s['ids']), sorted(covered))
        self.assertEqual(p42['settings']['seeds']['mask_noise'], 1042)
        self.assertEqual(p43['settings']['seeds']['mask_noise'], 1043)
        with tempfile.TemporaryDirectory() as td:
            path = sp.write_plan(p42, Path(td) / 'plan.json')
            loaded = k2p.load_plan(path, rec, 'sparse-k2-sft-r1-matched-s42')
            self.assertEqual(loaded['plan_sha256'], p42['plan_sha256'])

    def test_short_batches_3_to_8_train_rewire_matches_fp64_oracle(self):
        torch.manual_seed(0)
        model = _ToyCausalLM(vocab=32, d=16, n=24, layers=2)
        rm.install(model, rank=1, seed=42, mask_init=1.0)
        rng = random.Random(123)
        rows = []
        for idx in range(33):
            seq_len = rng.randint(6, 11)
            inp = [rng.randint(1, 31) for _ in range(seq_len)]
            rows.append({'id': f'ex-{idx}', 'input_ids': inp, 'labels': [-100] * 3 + inp[3:]})
        sizes = [3, 4, 5, 6, 7, 8]
        step_batches, pos = [], 0
        for sz in sizes:
            step_batches.append([rows[i]['id'] for i in range(pos, pos + sz)])
            pos += sz

        total_tok, ce_sum_fp64 = 0, 0.0
        with torch.no_grad():
            rm.set_gate(model, None)
            for r in rows[:3]:
                x = torch.tensor([r['input_ids']], dtype=torch.long)
                logits = model(x).logits[0, :-1].double()
                tgt = torch.tensor(r['labels'][1:], dtype=torch.long)
                mask = tgt != -100
                ce_sum_fp64 += float(F.cross_entropy(logits[mask], tgt[mask], reduction='sum').item())
                total_tok += int(mask.sum().item())
        oracle_step1_ce = ce_sum_fp64 / total_tok

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / 'run'
            res = rt.train_rewire(
                model, rows, out_dir, step_batches=step_batches, rank=1, read_lr=3e-3, mask_lr=0.1,
                l0_lambda=6e-4, warmup_steps=2, freeze_step=4, mask_seed=1042, mask_group_size=4,
                checkpoint_steps=[0, 4, 6], snapshot_interval=2, settle_start=2, settle_interval=2,
                gradient_checkpointing=False, model_context_limit=64, pad_token_id=0, execution_kind='cpu-fixture',
            )
            steps_logged = [json.loads(l) for l in (out_dir / 'logs' / 'steps.jsonl').read_text().splitlines() if l.strip()]
            self.assertAlmostEqual(steps_logged[1]['ce_loss'], oracle_step1_ce, delta=1e-6)
            self.assertEqual(res['successful_steps'], 6)

    def test_p2_6_two_layer_paired_activation_shift_prevents_upstream_sign_flip(self):
        """Regression for P2 #6: upstream Layer 0 rewiring flips downstream Layer 1's sign if a_base uses x_rewired."""
        torch.manual_seed(0)
        model = _ToyCausalLM(vocab=16, d=4, n=4, layers=2)
        with torch.no_grad():
            model.model.embed_tokens.weight.zero_()
            model.model.embed_tokens.weight[:, 0] = 1.0  # input residual x0 = [1, 0, 0, 0]
            # Layer 0 base MLP = 0, down_proj[:, 0] = [0, 3, 0, 0]
            l0 = model.model.layers[0].mlp
            l0.gate_proj.weight.zero_()
            l0.up_proj.weight.zero_()
            l0.down_proj.weight.zero_()
            l0.down_proj.weight[1, 0] = 3.0
            # Layer 1 base MLP reads dim 1 strongly (+1.0) for neuron 0
            l1 = model.model.layers[1].mlp
            l1.gate_proj.weight.zero_()
            l1.up_proj.weight.zero_()
            l1.gate_proj.weight[0, 1] = 1.0
            l1.up_proj.weight[0, 1] = 1.0
            l1.down_proj.weight.zero_()

        mods = rm.install(model, rank=1, seed=42, mask_init=2.0)
        with torch.no_grad():
            # Layer 0 neuron 0 activates strongly (+2.0 * +2.0 -> pushes dim 1 by +3 * SiLU(2)*2 ≈ +10.6!)
            mods[0].rewire_log_alpha.fill_(-10.0)
            mods[0].rewire_log_alpha[0] = 10.0
            mods[0].rewire_A_gate.zero_(); mods[0].rewire_A_gate[0, 0] = 2.0
            mods[0].rewire_B_gate.zero_(); mods[0].rewire_B_gate[0, 0] = 1.0
            mods[0].rewire_A_up.zero_();   mods[0].rewire_A_up[0, 0] = 2.0
            mods[0].rewire_B_up.zero_();   mods[0].rewire_B_up[0, 0] = 1.0
            # Layer 1 neuron 0 has a modest NEGATIVE direct read delta (-0.2 on dim 1)
            # Single-pass (buggy) sees x_rewired[1] ≈ +10.6 and reports NEGATIVE delta;
            # True paired forward sees base x_base[1] = 0 -> a_base = 0, a_rew >> 0 -> POSITIVE delta!
            mods[1].rewire_log_alpha.fill_(-10.0)
            mods[1].rewire_log_alpha[0] = 10.0
            mods[1].rewire_A_gate.zero_(); mods[1].rewire_A_gate[0, 1] = -0.2
            mods[1].rewire_B_gate.zero_(); mods[1].rewire_B_gate[0, 0] = 1.0
            mods[1].rewire_A_up.zero_();   mods[1].rewire_A_up[0, 1] = -0.2
            mods[1].rewire_B_up.zero_();   mods[1].rewire_B_up[0, 0] = 1.0

        item = {'id': 'ex-0', 'input_ids': [1, 2, 3, 4], 'labels': [-100, 2, 3, 4]}
        surv = ess._extract_survivors(mods)
        enriched = ess.measure_activation_shifts(model, mods, surv, [item], _ToyTok())
        l1_rec = next(s for s in enriched if s['layer'] == 1 and s['idx'] == 0)
        # Paired forward must yield strong POSITIVE shift and active_pole == '+v'
        self.assertGreater(l1_rec['activation_shift']['mean_delta'], 50.0)
        self.assertEqual(l1_rec['activation_shift']['active_pole'], '+v')

        # Cross-check against committed Gold paired-forward measurement on identical positions
        sel = ras.select_neurons([{'layer': 1, 'neuron': 0}], layers=2, neurons=4, per_layer=1)
        gold_m = ras.measure(model, [{'input_ids': [1, 2, 3]}], sel, torch.device('cpu'))
        self.assertAlmostEqual(
            l1_rec['activation_shift']['mean_delta'], float(gold_m[1]['survivor_mean'][0]), places=4
        )

            # Unsealed checkpoint must be refused by _load_verified_final_checkpoint
        with tempfile.TemporaryDirectory() as td:
            bad_ckpt = Path(td) / 'unsealed'
            bad_ckpt.mkdir()
            with self.assertRaises(ValueError):
                ess._load_verified_final_checkpoint(_ToyCausalLM(vocab=16, d=4, n=4, layers=2),
                                                    ckpt_dir=bad_ckpt, rank=1)

    def test_p2_7_missing_nonfinal_lens_layer_and_rank_deficient_principal_angles(self):
        """Reject missing non-final lens layers and verify rank-revealing SVD principal angles on duplicate rows."""
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Final layer (27) uses identity cleanly; missing non-final layer (11) MUST raise ValueError
        lenses = {'rlens': {0: torch.eye(4)}}
        self.assertTrue(torch.equal(ddp.transport_vector(lenses, 'rlens', 27, v, final_layer=27), v))
        self.assertTrue(torch.equal(ddp.transport_vector(lenses, 'logit_lens', 11, v, final_layer=27), v))
        with self.assertRaises(ValueError):
            ddp.transport_vector(lenses, 'rlens', 11, v, final_layer=27)

        # Duplicate copies of two orthogonal vectors -> rank 1 vs rank 1 -> single angle of 90.0 deg!
        A = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        B = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        pa = bdr.compute_principal_angles(A, B)
        self.assertEqual(pa['rank_a'], 1)
        self.assertEqual(pa['rank_b'], 1)
        self.assertEqual(pa['angles_deg'], [90.0])
        self.assertEqual(pa['cosines'], [0.0])

    def test_p2_7_comparative_k2_html_report_for_two_survivors_and_gold_regression(self):
        """Exercise real HTML builder on a 2-survivor K2 fixture (no Gold rank fields) + Gold report regression."""
        id_to_str = ['####', '<<', '>>', '0', '1', '=', '$', '\n', '\n\n', '**', '###', '\\[',
                     'boxed', 'Step', 'Therefore', 'So', 'Answer', '<|im_end|>', '<|endoftext|>', 'cat']
        vocab_size, d_model = len(id_to_str), 8
        lenses = ['logit_lens', 'jlens', 'jpp', 'rlens']
        torch.manual_seed(5)

        def _make_cache(survivors, seed):
            torch.manual_seed(seed)
            n = len(survivors)
            logits = {ln: torch.randn(n, vocab_size).to(torch.float16) for ln in lenses}
            trans = {ln: torch.randn(n, d_model) for ln in lenses}
            topk = {
                ln: [
                    {
                        'pos': [(j, id_to_str[j], float(logits[ln][i, j])) for j in range(6)],
                        'neg': [(j, id_to_str[j], -float(logits[ln][i, j])) for j in range(6, 12)],
                        'std': 1.0,
                    }
                    for i in range(n)
                ]
                for ln in lenses
            }
            return {
                'model_id': ddp.QWEN_MODEL_ID,
                'model_revision': ddp.QWEN_MODEL_REVISION,
                'lens_repo': ddp.LENS_REPO,
                'lens_revision': 'local-verified-assets',
                'asset_sha256': {'model.safetensors': 'a' * 64, 'tokenizer.json': 'b' * 64},
                'survivors': survivors,
                'lens_names': lenses,
                'id_to_str': id_to_str,
                'raw_v': torch.randn(n, d_model),
                'transported_u': trans,
                'u_raw_norms': {ln: torch.ones(n) for ln in lenses},
                'logits_fp16': logits,
                'top_k_records': topk,
                'layer_norm_stats': {17: {'mean': 1.0}},
                'probe': ddp.compute_probe_zscores(logits, id_to_str),
            }

        k2_surv = [
            {'layer': 17, 'idx': 6230, 'label': 'L17 #6230', 'group': 'k2-s42', 'log_alpha': 1.8, 'z_det': 1.0, 'v_norm': 1.2},
            {'layer': 19, 'idx': 999,  'label': 'L19 #999',  'group': 'k2-s42', 'log_alpha': 1.3, 'z_det': 0.9, 'v_norm': 0.9},
        ]
        gold_surv = [
            {'layer': 17, 'idx': 6230, 'label': 'L17 #6230', 'group': 'shared'},
            {'layer': 11, 'idx': 8235, 'label': 'L11 #8235', 'group': 'shared'},
            {'layer': 27, 'idx': 6214, 'label': 'L27 #6214', 'group': 'r1_only'},
        ]
        shifts_payload = {
            'survivors': [
                {'layer': 17, 'idx': 6230, 'activation_shift': {'mean_delta': 0.42, 'mean_abs_delta': 0.55, 'frac_pos': 0.81, 'active_pole': '+v'}},
                {'layer': 19, 'idx': 999,  'activation_shift': {'mean_delta': -0.31, 'mean_abs_delta': 0.40, 'frac_pos': 0.22, 'active_pole': '-v'}},
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            k2_pt = td_p / 'k2.pt'
            gold_pt = td_p / 'gold.pt'
            shifts_js = td_p / 'shifts.json'
            out_html = td_p / 'k2_report.html'
            torch.save(_make_cache(k2_surv, 10), k2_pt)
            torch.save(_make_cache(gold_surv, 20), gold_pt)
            shifts_js.write_text(json.dumps(shifts_payload), encoding='utf-8')

            res = bdr.build_comparative_report(
                k2_pt,
                gold_cache_path=gold_pt,
                out_paths=[out_html],
                k2_shifts_path=shifts_js,
                experiment_meta={'lambda_regime': 'matched', 'l0_lambda': 6e-4, 'seed': 42},
            )
            text = out_html.read_text(encoding='utf-8')
            self.assertIn('L17 #6230', text)
            self.assertIn('L19 #999', text)
            self.assertIn('Principal Angles', text)
            self.assertIn('Activation Shift Sign', text)
            self.assertEqual(res['payload']['overlap']['shared_count'], 1)
            self.assertEqual(res['payload']['overlap']['shared_neurons'], ['L17 #6230'])
            self.assertEqual(len(res['payload']['principal_angles']['rlens_transported']['angles_deg']), 2)
            self.assertEqual(res['payload']['neurons'][0]['activation_shift']['active_pole'], '+v')
            self.assertEqual(res['payload']['neurons'][1]['activation_shift']['active_pole'], '-v')

            # Mismatched or missing revision/asset metadata between compared caches MUST be refused
            bad_gold = _make_cache(gold_surv, 20)
            bad_gold.pop('lens_revision')
            bad_gold_pt = td_p / 'bad_gold.pt'
            torch.save(bad_gold, bad_gold_pt)
            with self.assertRaises(ValueError):
                bdr.build_comparative_report(k2_pt, gold_cache_path=bad_gold_pt, out_paths=[td_p / 'bad.html'])

            # Existing Gold report regression (runs real build_gold_report into temp output)
            if GOLD_CACHE.is_file():
                gold_out = td_p / 'gold_report.html'
                written = bdr.build_gold_report(cache_path=GOLD_CACHE, out_paths=[gold_out])
                self.assertEqual(written, [gold_out])
                self.assertGreater(gold_out.stat().st_size, 100_000)
                gold_html = gold_out.read_text(encoding='utf-8')
                self.assertIn('correlational findings &amp; hypothesis', gold_html)
                self.assertIn('hypothesis, not yet causally tested', gold_html)
                self.assertNotIn('Strong confirmation of your Section 7 hypothesis', gold_html)

    def test_p2_9_comparative_report_against_actual_gold_cache(self):
        """Real comparative Gold-vs-K2 report succeeds against the existing legacy 25 MB Gold cache."""
        if not GOLD_CACHE.is_file():
            self.skipTest(f'Gold cache not present at {GOLD_CACHE}')
        gold_data = torch.load(str(GOLD_CACHE), map_location='cpu', weights_only=False)
        id_to_str = list(gold_data['id_to_str'])
        vocab_size = len(id_to_str)
        d_model = int(gold_data['transported_u']['rlens'].shape[1])
        lens_names = list(gold_data['lens_names'])

        g = torch.Generator().manual_seed(7)
        k2_surv = [
            {'layer': 17, 'idx': 6230, 'v_norm': 1.2, 'v_norm_pct': 78.1},
            {'layer': 19, 'idx': 999, 'v_norm': 0.9, 'v_norm_pct': 41.0},
        ]
        n = len(k2_surv)
        k2_cache = {
            'model_id': ddp.QWEN_MODEL_ID,
            'model_revision': ddp.QWEN_MODEL_REVISION,
            'lens_repo': ddp.LENS_REPO,
            'lens_revision': '4fdc576695c5b4f0b173ba00f48e589b3789c452',
            'lens_meta': gold_data['lens_meta'],
            'asset_sha256': {
                'model.safetensors': 'full-local-sha',
                'model.safetensors:header': 'hdr-sha',
                'tokenizer.json': 'tok-sha',
                'lenses': {'jlens': 'j1', 'jpp': 'j2', 'rlens': 'r1'},
            },
            'survivors': k2_surv,
            'lens_names': lens_names,
            'raw_v': torch.randn(n, d_model, generator=g),
            'transported_u': {ln: torch.randn(n, d_model, generator=g) for ln in lens_names},
            'u_raw_norms': {ln: torch.ones(n) for ln in lens_names},
            'logits_fp16': {ln: torch.randn(n, vocab_size, generator=g).to(torch.float16) for ln in lens_names},
            'id_to_str': id_to_str,
            'top_k_records': {
                ln: [
                    {'pos': [(0, id_to_str[0], 2.0)], 'neg': [(1, id_to_str[1], -2.0)], 'std': 1.0}
                    for _ in range(n)
                ]
                for ln in lens_names
            },
        }
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            k2_pt = td_p / 'k2_real_gold_compat.pt'
            out_html = td_p / 'k2_vs_real_gold.html'
            torch.save(k2_cache, k2_pt)

            res = bdr.build_comparative_report(
                k2_pt,
                gold_cache_path=GOLD_CACHE,
                out_paths=[out_html],
                experiment_meta={'lambda_regime': 'matched', 'l0_lambda': 6e-4, 'seed': 42},
            )
            self.assertTrue(out_html.is_file())
            self.assertEqual(res['payload']['overlap']['gold_count'], 18)
            self.assertEqual(res['payload']['overlap']['k2_count'], 2)
            self.assertEqual(res['payload']['overlap']['shared_count'], 1)
            self.assertEqual(res['payload']['overlap']['shared_neurons'], ['L17 #6230'])
            self.assertEqual(len(res['payload']['principal_angles']['rlens_transported']['angles_deg']), 2)

            # Mismatched lens_meta provenance against the real Gold cache MUST be refused
            bad_k2 = dict(k2_cache)
            bad_k2['lens_meta'] = {
                'rlens': {'provenance': {'source': 'bubskee/dolos-qwen25-workspace-lenses@deadbeef'}},
            }
            bad_k2_pt = td_p / 'bad_k2.pt'
            torch.save(bad_k2, bad_k2_pt)
            with self.assertRaises(ValueError):
                bdr.build_comparative_report(bad_k2_pt, gold_cache_path=GOLD_CACHE, out_paths=[td_p / 'bad.html'])

    def test_p2_10_local_vs_streamed_cache_compatibility_and_header_hash(self):
        """Pod-local and Mac HTTP-streamed decodes share model.safetensors:header and compare cleanly."""
        import hashlib
        import struct
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            st_path = td_p / 'model.safetensors'
            save_file({'w': torch.arange(6, dtype=torch.float32).reshape(2, 3)}, str(st_path))
            raw = st_path.read_bytes()
            hdr_len = struct.unpack('<Q', raw[:8])[0]
            expected_hdr_sha = hashlib.sha256(raw[8:8 + hdr_len]).hexdigest()
            self.assertEqual(ddp._safetensors_header_sha256(st_path), expected_hdr_sha)

            vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '####', '<|endoftext|>', '\n', '\n\n', ' ']
            lens_names = ['logit_lens', 'jlens', 'jpp', 'rlens']

            def _make(surv, asset_sha256):
                g = torch.Generator().manual_seed(11)
                n = len(surv)
                return {
                    'model_id': ddp.QWEN_MODEL_ID,
                    'model_revision': ddp.QWEN_MODEL_REVISION,
                    'lens_repo': ddp.LENS_REPO,
                    'lens_revision': '4fdc576695c5b4f0b173ba00f48e589b3789c452',
                    'asset_sha256': asset_sha256,
                    'survivors': surv,
                    'lens_names': lens_names,
                    'raw_v': torch.randn(n, 8, generator=g),
                    'transported_u': {ln: torch.randn(n, 8, generator=g) for ln in lens_names},
                    'u_raw_norms': {ln: torch.ones(n) for ln in lens_names},
                    'logits_fp16': {ln: torch.randn(n, len(vocab), generator=g).to(torch.float16) for ln in lens_names},
                    'id_to_str': vocab,
                    'top_k_records': {
                        ln: [{'pos': [(0, '0', 1.0)], 'neg': [(1, '1', -1.0)], 'std': 1.0} for _ in range(n)]
                        for ln in lens_names
                    },
                }

            local_assets = {
                'model.safetensors': 'full-file-sha256-pod',
                'model.safetensors:header': expected_hdr_sha,
                'tokenizer.json': 'tok-sha-same',
                'lenses': {'jlens': 'j1', 'jpp': 'j2', 'rlens': 'r1'},
            }
            streamed_assets = {
                'model.safetensors:header': expected_hdr_sha,
                'tokenizer.json': 'tok-sha-same',
                'lenses': {'jlens': 'j1', 'jpp': 'j2', 'rlens': 'r1'},
            }
            k2_pt = td_p / 'k2_local.pt'
            gold_pt = td_p / 'gold_streamed.pt'
            torch.save(_make([{'layer': 17, 'idx': 6230, 'v_norm': 1.0, 'v_norm_pct': 50.0}], local_assets), k2_pt)
            torch.save(_make([{'layer': 17, 'idx': 6230, 'v_norm': 1.0, 'v_norm_pct': 50.0}], streamed_assets), gold_pt)

            out_html = td_p / 'local_vs_streamed.html'
            res = bdr.build_comparative_report(k2_pt, gold_cache_path=gold_pt, out_paths=[out_html])
            self.assertEqual(res['payload']['overlap']['shared_count'], 1)

            # Different model header hash MUST be refused
            bad_streamed = dict(streamed_assets, **{'model.safetensors:header': '0' * 64})
            bad_gold_pt = td_p / 'gold_bad_hdr.pt'
            torch.save(_make([{'layer': 17, 'idx': 6230, 'v_norm': 1.0, 'v_norm_pct': 50.0}], bad_streamed), bad_gold_pt)
            with self.assertRaises(ValueError):
                bdr.build_comparative_report(k2_pt, gold_cache_path=bad_gold_pt, out_paths=[td_p / 'x.html'])

            # Different tokenizer hash MUST be refused
            bad_tok = dict(streamed_assets, **{'tokenizer.json': 'tok-different'})
            bad_tok_pt = td_p / 'gold_bad_tok.pt'
            torch.save(_make([{'layer': 17, 'idx': 6230, 'v_norm': 1.0, 'v_norm_pct': 50.0}], bad_tok), bad_tok_pt)
            with self.assertRaises(ValueError):
                bdr.build_comparative_report(k2_pt, gold_cache_path=bad_tok_pt, out_paths=[td_p / 'y.html'])

            # Different lens file hash MUST be refused
            bad_lens = dict(streamed_assets, lenses={'jlens': 'j1', 'jpp': 'j2', 'rlens': 'r-wrong'})
            bad_lens_pt = td_p / 'gold_bad_lens.pt'
            torch.save(_make([{'layer': 17, 'idx': 6230, 'v_norm': 1.0, 'v_norm_pct': 50.0}], bad_lens), bad_lens_pt)
            with self.assertRaises(ValueError):
                bdr.build_comparative_report(k2_pt, gold_cache_path=bad_lens_pt, out_paths=[td_p / 'z.html'])

    def test_p2_7_asymmetric_lens_orientation_and_immutable_revision_guard(self):
        """Asymmetric orientation check accepts J_true, rejects transposed J_true.T, and refuses mutable main."""
        # Known asymmetric mapping: e0 -> e1, e2 -> e3 (strictly non-symmetric!)
        J_true = torch.zeros((4, 4))
        J_true[1, 0] = 1.0
        J_true[3, 2] = 1.0
        X = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        Y = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        ok = ddp.verify_lens_orientation({'rlens': {11: J_true}}, probe_pairs=(X, Y), final_layer=27)
        self.assertTrue(ok['rlens']['verified'])
        self.assertGreater(ok['rlens']['mean_score_J_x'], ok['rlens']['mean_score_JT_x'])
        # Transposed matrix MUST be rejected!
        with self.assertRaises(ValueError):
            ddp.verify_lens_orientation({'rlens': {11: J_true.T}}, probe_pairs=(X, Y), final_layer=27)
        # Mutable 'main' revision MUST be refused for HF downloads!
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                ddp._download_hf(ddp.LENS_REPO, 'a.pt', Path(td) / 'a.pt', revision='main')

    def test_p2_8_rewire_analysis_rebuilds_standalone_when_gold_adapter_or_shift_arrives(self):
        """Regression for P2 #8: documented --gold-adapter & activation-shift sequence must rebuild standalone."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / 'c-attribution' / 'sparse-rewiring' / 'analysis'
            base.mkdir(parents=True)
            ga_dir = root / 'gold-adapter'
            ga_dir.mkdir()
            (ga_dir / 'adapter_model.safetensors').write_bytes(b'dummy-gold-adapter')

            fake_arm = {'arm': 'sparse-gold-sft-r1', 'rank': 1, 'export_path': str(root / 'exp')}
            fake_record = {'arms': {'sparse-gold-sft-r1': {'rank': 1}}}
            calls = []

            def _fake_standalone(arm, refs, *, gold_adapter=None, activation_shift=None):
                calls.append((gold_adapter is not None, activation_shift is not None))
                return {
                    'arm': arm['arm'],
                    'has_gold_cosine': gold_adapter is not None,
                    'has_activation_shift': activation_shift is not None,
                }

            with mock.patch('pilot_eval.rewire_report.paths', return_value={'analysis': base}), \
                 mock.patch('pilot_eval.rewire_report.launch_record', return_value=fake_record), \
                 mock.patch.object(ran, 'load_arm', return_value=fake_arm), \
                 mock.patch.object(ran, 'load_references', return_value={}), \
                 mock.patch.object(ran, 'standalone', side_effect=_fake_standalone):
                # Step 1: initial run without --gold-adapter or activation-shift
                d1 = ran.run(root, only=('standalone',))
                self.assertNotIn('kept', d1['standalone:sparse-gold-sft-r1'])
                s1 = json.loads((base / 'standalone-sparse-gold-sft-r1' / 'standalone_analysis.json').read_text())
                self.assertFalse(s1['has_gold_cosine'])

                # Step 2: identical re-run keeps sealed unit
                d2 = ran.run(root, only=('standalone',))
                self.assertIn('kept', d2['standalone:sparse-gold-sft-r1'])

                # Step 3: documented second call with --gold-adapter supersedes & rebuilds!
                d3 = ran.run(root, gold_adapter=ga_dir, only=('standalone',))
                self.assertNotIn('kept', d3['standalone:sparse-gold-sft-r1'])
                s3 = json.loads((base / 'standalone-sparse-gold-sft-r1' / 'standalone_analysis.json').read_text())
                self.assertTrue(s3['has_gold_cosine'])
                self.assertFalse(s3['has_activation_shift'])

                # Step 4: activation-shift arrives later -> supersedes & rebuilds again!
                au.seal_run(
                    base / 'activation-shift-sparse-gold-sft-r1',
                    {'activation_shift.json': {'per_neuron_mean_shift': {}, 'per_neuron_mean_abs_shift': {}, 'tokens': 10}},
                )
                d4 = ran.run(root, gold_adapter=ga_dir, only=('standalone',))
                self.assertNotIn('kept', d4['standalone:sparse-gold-sft-r1'])
                s4 = json.loads((base / 'standalone-sparse-gold-sft-r1' / 'standalone_analysis.json').read_text())
                self.assertTrue(s4['has_gold_cosine'])
                self.assertTrue(s4['has_activation_shift'])
                self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
