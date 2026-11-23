"""Sparse rewiring ticket 07: training loop tests (spec tests 7, 9, 10 and 19 + PR #15 review fixes).

CPU fixture tests on a tiny Qwen2ForCausalLM, no model downloads.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_dynamics as rd
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_training as rt
from pilot_eval import sft_backend as sb
from pilot_eval import sparse_rewiring_plan as sp

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'


def _tiny_qwen2(seed: int = 17, *, rank: int = 1) -> Qwen2ForCausalLM:
    cfg_dict = json.loads(FIXTURE_CONFIG.read_text(encoding='utf-8'))
    cfg_dict.update(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype='float32',
    )
    cfg = Qwen2Config(**cfg_dict)
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)
    model.config._attn_implementation = 'eager'
    rm.install(model, rank=rank, seed=42)
    return model


def _variable_length_rows() -> list[dict]:
    return [
        {'id': 'e0', 'input_ids': [5, 6, 7, 2], 'labels': [-100, -100, 7, 2]},
        {'id': 'e1', 'input_ids': [8, 9, 10, 11, 12, 13, 14, 15, 2], 'labels': [-100, -100, 10, 11, 12, 13, 14, 15, 2]},
        {'id': 'e2', 'input_ids': [3, 4, 2], 'labels': [-100, -100, 2]},
        {'id': 'e3', 'input_ids': [4, 5, 6, 7, 8, 2], 'labels': [-100, -100, 6, 7, 8, 2]},
        {'id': 'e4', 'input_ids': [9, 10, 11, 2], 'labels': [-100, -100, 11, 2]},
        {'id': 'e5', 'input_ids': [6, 7, 8, 9, 10, 11, 2], 'labels': [-100, -100, 8, 9, 10, 11, 2]},
        {'id': 'e6', 'input_ids': [7, 8, 2], 'labels': [-100, -100, 2]},
        {'id': 'e7', 'input_ids': [10, 11, 12, 13, 2], 'labels': [-100, -100, 12, 13, 2]},
    ]


def _tiny_plan(rank: int, read_lr: float, batches: list[list[str]], *, warmup: int, freeze: int, checkpoints: list[int]) -> dict:
    total = len(batches)
    steps = []
    for s, ids in enumerate(batches, start=1):
        steps.append({
            'step': s,
            'ids': ids,
            'lr_multiplier': sp.lr_multiplier(s, warmup=warmup, total=total),
            'masks': 'frozen-binary' if s > freeze else 'trainable',
            'mask_groups': [[0, 4], [4, len(ids)]] if len(ids) > 4 else [[0, len(ids)]],
        })
    plan = {
        'schema': sp.PLAN_SCHEMA,
        'arm': f'sparse-gold-sft-r{rank}',
        'run_id': f'tiny-r{rank}',
        'rank': rank,
        'read_lr': read_lr,
        'checkpoints': checkpoints,
        'settings': {
            'model': {'layers': 2, 'hidden': 16, 'intermediate': 32, 'parts': ['gate', 'up']},
            'training': {'steps': total, 'warmup_steps': warmup, 'freeze_after_step': freeze},
            'objective': {'l0_lambda': 1.0},
            'seeds': {'mask_noise': 1042},
            'data': {'examples_per_mask': 4},
            'storage': {
                'checkpoint_contents': ['read_factors', 'log_alpha', 'seal'],
                'optimizer_state': 'none',
                'interruption': 'restart-from-step-0',
            },
        },
        'steps': steps,
    }
    plan['checkpoint_bytes'] = rckpt.payload_bytes(plan)
    plan['plan_sha256'] = sp.plan_sha256(plan)
    return plan


class TestRewireTrainingLoop(unittest.TestCase):
    def test_lr_schedule_matches_frozen_plan_on_all_1024_steps(self) -> None:
        for step_1based in range(1, 1025):
            ours = rt.warmup_cosine_multiplier(step_1based - 1, warmup_steps=20, total_steps=1024)
            planned = sp.lr_multiplier(step_1based, warmup=20, total=1024)
            self.assertEqual(ours, planned, f'mismatch at step {step_1based}')
        self.assertEqual(rt.warmup_cosine_multiplier(1023, warmup_steps=20, total_steps=1024), 0.0)

    def test_spec_test_7_token_weighted_step_loss_vs_per_example_mean(self) -> None:
        rows = _variable_length_rows()
        batch_ids = [r['id'] for r in rows]

        ref_model = _tiny_qwen2(seed=17, rank=1)
        ref_model.eval()
        total_nll = 0.0
        total_tok = 0
        per_ex_means = []
        with torch.no_grad():
            for r in rows:
                inp = torch.tensor([r['input_ids']], dtype=torch.long)
                lab = torch.tensor([r['labels']], dtype=torch.long)
                logits = ref_model(input_ids=inp, use_cache=False).logits
                shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shifted_labels = lab[:, 1:].reshape(-1)
                nll = float(F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100, reduction='sum').item())
                n_sup = int((shifted_labels != -100).sum().item())
                total_nll += nll
                total_tok += n_sup
                per_ex_means.append(nll / n_sup)
        expected_token_weighted = total_nll / total_tok
        wrong_per_example_mean = sum(per_ex_means) / len(per_ex_means)
        self.assertGreater(abs(expected_token_weighted - wrong_per_example_mean), 1e-3)

        model = _tiny_qwen2(seed=17, rank=1)
        with tempfile.TemporaryDirectory() as tmp:
            res = rt.train_rewire(
                model,
                rows,
                tmp,
                step_batches=[batch_ids],
                rank=1,
                read_lr=3e-3,
                l0_lambda=6e-4,
                warmup_steps=1,
                freeze_step=None,
                checkpoint_steps=(0, 1),
                snapshot_interval=1,
                enable_settle_diagnostics=False,
                gradient_checkpointing=False,
                model_context_limit=64,
                pad_token_id=0,
                execution_kind='cpu-fixture',
            )
            step1_ce = res['final_step_record']['ce_loss']
            self.assertAlmostEqual(step1_ce, expected_token_weighted, places=5)
            self.assertNotAlmostEqual(step1_ce, wrong_per_example_mean, places=3)

    def test_spec_test_9_read_only_clipping_plan_schedule_freeze_and_pr16_sink(self) -> None:
        rows = _variable_length_rows()
        batches = [[r['id'] for r in rows] for _ in range(4)]
        plan = _tiny_plan(1, 3e-3, batches, warmup=2, freeze=2, checkpoints=[0, 2, 4])
        model = _tiny_qwen2(seed=23, rank=1)
        base_hash = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
        val_calls: list[int] = []

        def _val_stub(m: torch.nn.Module, step: int) -> dict:
            val_calls.append(step)
            return reval.evaluate_validation_ce(m, rows[:2])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sink = rckpt.CheckpointSink(out, plan, runtime_sha256='rt-1', code_commit='c0ffee')
            res = rt.train_rewire(
                model,
                rows,
                out,
                plan=plan,
                sink=sink,
                mask_lr=0.1,
                max_grad_norm=1e-4,
                snapshot_interval=2,
                enable_settle_diagnostics=False,
                gradient_checkpointing=False,
                model_context_limit=64,
                pad_token_id=0,
                expected_base_sha256=base_hash,
                execution_kind='real-cuda',
                val_ce_fn=_val_stub,
            )
            self.assertEqual(sink.missing(), [])
            # Verify PR #16 checkpoints exist and verify cleanly
            for step in (0, 2, 4):
                rckpt.verify_checkpoint(rckpt.checkpoint_dir(out, step), plan, runtime_sha256='rt-1', code_commit='c0ffee')

            step_rows = rd.read_jsonl(out / 'logs' / 'steps.jsonl')
            self.assertEqual([r['step'] for r in step_rows], [0, 1, 2, 3, 4])
            # D14 mask_group_log is persisted in every step row
            self.assertEqual(step_rows[1]['mask_group_log']['boundaries'], [[0, 4], [4, 8]])
            self.assertEqual(step_rows[1]['mask_group_log']['seed'], 1042)
            # Shared schedule across both groups matches plan['steps'][s-1]['lr_multiplier']
            for r in step_rows[1:]:
                p_mult = plan['steps'][r['step'] - 1]['lr_multiplier']
                self.assertAlmostEqual(r['lr_reads'], 3e-3 * p_mult, places=10)
                self.assertAlmostEqual(r['lr_masks'], 0.1 * p_mult, places=10)
            # Mask grads were NOT clipped at 1e-4
            self.assertGreater(step_rows[1]['grad_norm_masks'], 1e-2)
            # After freeze at step 2: steps 3 & 4 have l0_loss == 0, grad_norm_masks == 0, frozen == True
            self.assertTrue(res['frozen'])
            self.assertTrue(res['deterministic_algorithms'])
            self.assertTrue((out / 'hardening.json').is_file())
            self.assertEqual(step_rows[3]['l0_loss'], 0.0)
            self.assertEqual(step_rows[3]['grad_norm_masks'], 0.0)
            self.assertTrue(step_rows[3]['frozen'])
            self.assertIn(0, val_calls)
            self.assertIn(2, val_calls)
            self.assertIn(4, val_calls)

    def test_spec_test_10_checkpointed_vs_non_checkpointed_identical(self) -> None:
        rows = _variable_length_rows()
        batches = [[r['id'] for r in rows[:4]], [r['id'] for r in rows[4:]]]

        def _run(use_ckpt: bool) -> dict[str, torch.Tensor]:
            m = _tiny_qwen2(seed=31, rank=1)
            with tempfile.TemporaryDirectory() as tmp:
                rt.train_rewire(
                    m,
                    rows,
                    tmp,
                    step_batches=batches,
                    rank=1,
                    read_lr=3e-3,
                    l0_lambda=6e-4,
                    warmup_steps=1,
                    freeze_step=None,
                    mask_seed=1042,
                    checkpoint_steps=(0, 2),
                    snapshot_interval=2,
                    enable_settle_diagnostics=False,
                    gradient_checkpointing=use_ckpt,
                    model_context_limit=64,
                    pad_token_id=0,
                    execution_kind='cpu-fixture',
                )
            return {n: p.detach().clone() for n, p in rm.trainable_parameters(m)}

        p_no = _run(False)
        p_yes = _run(True)
        for name in p_no:
            self.assertTrue(
                torch.allclose(p_no[name], p_yes[name], atol=1e-6, rtol=1e-6),
                f'checkpointed vs non-checkpointed diverged for {name}',
            )

    def test_spec_test_19_carried_over_checks_and_rank0(self) -> None:
        rows = _variable_length_rows()
        batches = [[r['id'] for r in rows[:4]]]

        with tempfile.TemporaryDirectory() as tmp:
            # Red 1: non-FP32 weights refused
            m_half = _tiny_qwen2(seed=41, rank=1).to(dtype=torch.float16)
            with self.assertRaises(ValueError):
                rt.train_rewire(
                    m_half, rows, tmp, step_batches=batches, rank=1, read_lr=3e-3,
                    model_context_limit=64, pad_token_id=0, execution_kind='cpu-fixture',
                )

            # Red 2: context overflow refused
            m_ok = _tiny_qwen2(seed=41, rank=1)
            with self.assertRaises(ValueError):
                rt.train_rewire(
                    m_ok, rows, tmp, step_batches=batches, rank=1, read_lr=3e-3,
                    model_context_limit=4, pad_token_id=0, execution_kind='cpu-fixture',
                )

            # Red 3: missing expected_base_sha256 on real-cuda path refused (lesson 2026-11-20)
            with self.assertRaises(ValueError):
                rt.train_rewire(
                    m_ok, rows, tmp, step_batches=batches, rank=1, read_lr=3e-3,
                    model_context_limit=64, pad_token_id=0, expected_base_sha256=None, execution_kind='real-cuda',
                )

            # Red 4: wrong expected_base_sha256 refused
            with self.assertRaises(ValueError):
                rt.train_rewire(
                    m_ok, rows, tmp, step_batches=batches, rank=1, read_lr=3e-3,
                    model_context_limit=64, pad_token_id=0, expected_base_sha256='0' * 64, execution_kind='real-cuda',
                )

            # Red 5: nonzero step-0 read coefficient refused
            m_dirty = _tiny_qwen2(seed=41, rank=1)
            with torch.no_grad():
                rm.modules(m_dirty)[0].rewire_B_gate.fill_(1.0)
            with self.assertRaises(ValueError):
                rt.train_rewire(
                    m_dirty, rows, tmp, step_batches=batches, rank=1, read_lr=3e-3,
                    model_context_limit=64, pad_token_id=0, execution_kind='cpu-fixture',
                )

            # Green: rank 0 + use_reentrant=False + foreach=False verified via spy
            m_r0 = _tiny_qwen2(seed=41, rank=0)
            base_hash = sb.frozen_weight_hash(m_r0, trained=rm.is_rewire_parameter)
            orig_adamw = torch.optim.AdamW
            seen_foreach: list[bool | None] = []
            seen_reentrant: list[bool | None] = []

            def _spy_adamw(*args, **kwargs):
                seen_foreach.append(kwargs.get('foreach'))
                return orig_adamw(*args, **kwargs)

            orig_gc = m_r0.gradient_checkpointing_enable

            def _spy_gc(*args, **kwargs):
                seen_reentrant.append((kwargs.get('gradient_checkpointing_kwargs') or {}).get('use_reentrant'))
                return orig_gc(*args, **kwargs)

            with mock.patch.object(torch.optim, 'AdamW', side_effect=_spy_adamw), mock.patch.object(
                m_r0, 'gradient_checkpointing_enable', side_effect=_spy_gc
            ):
                res = rt.train_rewire(
                    m_r0,
                    rows,
                    Path(tmp) / 'r0',
                    step_batches=batches,
                    rank=0,
                    read_lr=3e-4,
                    warmup_steps=1,
                    freeze_step=None,
                    expected_base_sha256=base_hash,
                    execution_kind='real-cuda',
                    checkpoint_steps=(0, 1),
                    snapshot_interval=1,
                    enable_settle_diagnostics=False,
                    gradient_checkpointing=True,
                    model_context_limit=64,
                    pad_token_id=0,
                )
            self.assertEqual(seen_foreach, [False])
            self.assertEqual(seen_reentrant, [False])
            self.assertEqual(res['base_sha256_before'], base_hash)
            self.assertEqual(res['base_sha256_after'], base_hash)


if __name__ == '__main__':
    unittest.main()
