"""Sparse rewiring ticket 09: evaluation wiring tests (checkpoint 0 identity via PR #16, spec test 22, D12)."""
from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from pilot_eval import attribution_units as au
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import sparse_rewiring_plan as sp

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'


def _tiny_qwen2(seed: int = 11) -> Qwen2ForCausalLM:
    """Build a tiny FP32 eager Qwen2ForCausalLM from the real config fixture (no pad_token_id)."""
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
    assert cfg.pad_token_id is None
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)
    model.config._attn_implementation = 'eager'
    return model


def _tiny_plan(rank: int) -> dict:
    """Minimal valid PR #6 plan matching our tiny Qwen2 shape (2 layers, hidden 16, intermediate 32)."""
    plan = {
        'schema': sp.PLAN_SCHEMA,
        'arm': f'sparse-gold-sft-r{rank}',
        'run_id': f'tiny-r{rank}',
        'rank': rank,
        'checkpoints': [0],
        'settings': {
            'model': {'layers': 2, 'hidden': 16, 'intermediate': 32, 'parts': ['gate', 'up']},
            'storage': {
                'checkpoint_contents': ['read_factors', 'log_alpha', 'seal'],
                'optimizer_state': 'none',
                'interruption': 'restart-from-step-0',
            },
        },
        'steps': [],
    }
    plan['checkpoint_bytes'] = rckpt.payload_bytes(plan)
    plan['plan_sha256'] = sp.plan_sha256(plan)
    return plan


def _child_pid_helper(payload: dict) -> dict:
    return {
        'pid': os.getpid(),
        'echo': payload['value'],
        'offline': os.environ.get('HF_HUB_OFFLINE'),
        'cublas': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'runtime_sha256': 'rt-child-ok',
    }


class TestRewireEvaluation(unittest.TestCase):
    def test_spec_test_22_pad_and_eos_from_tokenizer_rejects_model_config_pad(self) -> None:
        tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
        tiny_cfg = _tiny_qwen2().config
        self.assertIsNone(tiny_cfg.pad_token_id)
        ids = reval.resolve_token_ids(tok, model_config=tiny_cfg)
        self.assertEqual(ids, {'pad_token_id': 0, 'eos_token_id': 2})

        bad_cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, pad_token_id=0)
        with self.assertRaises(ValueError):
            reval.resolve_token_ids(tok, model_config=bad_cfg)

        with self.assertRaises(ValueError):
            reval.resolve_token_ids(types.SimpleNamespace(pad_token_id=None, eos_token_id=2), model_config=tiny_cfg)

    def test_checkpoint_0_equals_untuned_ce_and_greedy_outputs_via_pr16_loader(self) -> None:
        base_model = _tiny_qwen2(seed=21)
        val_items = [
            {'input_ids': [5, 6, 7, 8, 9, 2], 'labels': [-100, -100, 7, 8, 9, 2]},
            {'input_ids': [3, 4, 10, 11, 2], 'labels': [-100, -100, 10, 11, 2]},
        ]
        questions = {
            'q0': {'prompt_ids': [5, 6, 7], 'gold': '42'},
            'q1': {'prompt_ids': [3, 4, 10, 11], 'gold': '7'},
        }
        decode = lambda ids: f'#### {sum(ids) % 10}'

        untuned_ce = reval.evaluate_validation_ce(base_model, val_items)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            untuned_greedy = reval.evaluate_greedy(
                base_model,
                questions,
                ['q0', 'q1'],
                tmp_path / 'eval_untuned',
                arm='untuned',
                step=0,
                cohort='gsm8k-train-val128',
                decode=decode,
                pad_token_id=0,
                eos_token_id=2,
                cap=6,
                batch_size=2,
            )

            for rank in (1, 0):
                plan = _tiny_plan(rank)
                m_src = _tiny_qwen2(seed=21)
                rm.install(m_src, rank=rank, seed=42)
                run_dir = tmp_path / f'run_r{rank}'
                rckpt.save_checkpoint(run_dir, 0, m_src, plan, runtime_sha256='rt-def', code_commit='c0ffee')

                m_loaded = _tiny_qwen2(seed=21)
                seal = reval.load_checkpoint_for_eval(
                    m_loaded,
                    rckpt.checkpoint_dir(run_dir, 0),
                    plan,
                    runtime_sha256='rt-def',
                    code_commit='c0ffee',
                )
                self.assertEqual(seal['step'], 0)
                self.assertEqual(seal['rank'], rank)

                loaded_ce = reval.evaluate_validation_ce(m_loaded, val_items)
                self.assertEqual(loaded_ce['val_ce'], untuned_ce['val_ce'])

                loaded_greedy = reval.evaluate_greedy(
                    m_loaded,
                    questions,
                    ['q0', 'q1'],
                    tmp_path / f'eval_r{rank}_0',
                    arm=f'sparse-gold-sft-r{rank}',
                    step=0,
                    cohort='gsm8k-train-val128',
                    decode=decode,
                    pad_token_id=0,
                    eos_token_id=2,
                    cap=6,
                    batch_size=2,
                    checkpoint_meta=seal,
                )
                for qid in ('q0', 'q1'):
                    self.assertEqual(
                        loaded_greedy['responses'][qid]['generated_token_ids'],
                        untuned_greedy['responses'][qid]['generated_token_ids'],
                    )
                self.assertEqual(au.verify_run(tmp_path / f'eval_r{rank}_0'), {'config.json', 'responses.jsonl', 'scores.json'})

    def test_cohort150_restricted_to_64_and_final_and_prior_use_sealed(self) -> None:
        model = _tiny_qwen2(seed=33)
        questions = {'q0': {'prompt_ids': [5, 6, 7], 'gold': '42'}}
        decode = lambda ids: '#### 42'
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for disallowed in (0, 16, 128, 960):
                with self.assertRaises(ValueError):
                    reval.evaluate_greedy(
                        model,
                        questions,
                        ['q0'],
                        tmp_path / f'bad_{disallowed}',
                        arm='sparse-gold-sft-r1',
                        step=disallowed,
                        cohort=reval.COHORT150_NAME,
                        decode=decode,
                        pad_token_id=0,
                        eos_token_id=2,
                        cap=4,
                    )
            out = reval.evaluate_greedy(
                model,
                questions,
                ['q0'],
                tmp_path / 'ok_64',
                arm='sparse-gold-sft-r1',
                step=64,
                cohort=reval.COHORT150_NAME,
                decode=decode,
                pad_token_id=0,
                eos_token_id=2,
                cap=4,
            )
            self.assertTrue(out['scores']['exploratory'])
            self.assertEqual(out['prior_use']['cohort'], reval.COHORT150_NAME)
            self.assertEqual(out['prior_use']['step'], 64)
            (tmp_path / 'ok_64' / 'responses.jsonl').write_text('{"tampered": true}\n')
            with self.assertRaises(ValueError):
                reval.prior_use_from_sealed_eval(tmp_path / 'ok_64')

    def test_unbound_checkpoint_refused_and_child_process_checks_env_and_runtime(self) -> None:
        plan = _tiny_plan(1)
        model = _tiny_qwen2(seed=44)
        rm.install(model, rank=1, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            rckpt.save_checkpoint(run_dir, 0, model, plan, runtime_sha256='r1', code_commit='c1')
            ckpt_dir = rckpt.checkpoint_dir(run_dir, 0)
            # Refuses mismatched runtime or empty/unknown code_commit
            with self.assertRaises(ValueError):
                reval.load_checkpoint_for_eval(_tiny_qwen2(seed=44), ckpt_dir, plan, runtime_sha256='wrong', code_commit='c1')
            with self.assertRaises(ValueError):
                reval.load_checkpoint_for_eval(_tiny_qwen2(seed=44), ckpt_dir, plan, runtime_sha256='', code_commit='c1')
            with self.assertRaises(ValueError):
                reval.load_checkpoint_for_eval(_tiny_qwen2(seed=44), ckpt_dir, plan, runtime_sha256='r1', code_commit='unknown')

        require_ieee = hasattr(torch.backends.cuda.matmul, 'fp32_precision')
        res = reval.run_in_child_process(
            'tests.test_rewire_evaluation',
            '_child_pid_helper',
            {'value': 'ok'},
            expected_runtime_sha256='rt-child-ok',
            require_ieee=require_ieee,
        )
        self.assertEqual(res['echo'], 'ok')
        self.assertNotEqual(res['pid'], os.getpid())
        self.assertEqual(res['offline'], '1')
        self.assertEqual(res['cublas'], ':4096:8')
        # Mismatched runtime_sha256 from child is refused
        with self.assertRaises(RuntimeError):
            reval.run_in_child_process(
                'tests.test_rewire_evaluation',
                '_child_pid_helper',
                {'value': 'ok'},
                expected_runtime_sha256='rt-different',
                require_ieee=require_ieee,
            )


if __name__ == '__main__':
    unittest.main()
