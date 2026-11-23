"""Sparse rewiring ticket 14: Pod chain integration and real-path CPU test (spec tests 20 and 24).

Runs the real path on a saved tiny Qwen2ForCausalLM + saved local tokenizer + real launch-record
schema, stubbing ONLY CUDA availability, disk usage, volume identity, Hub exporter, and Pod stop.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_chain as cc
from pilot_eval import neuron_audit_collector as nac
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_evaluation as reval
from pilot_eval import sparse_rewiring_plan as sp


@contextlib.contextmanager
def _ieee_backends_if_torch_lt_29():
    """On PyTorch >= 2.9, use the real torch.backends.*.fp32_precision untouched; on older local torch,
    provide an in-memory backend table so set_ieee_precision / precision_readback still execute their real code."""
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        yield
        return
    fake_backends = {path: types.SimpleNamespace(fp32_precision='tf32') for path in nac.PRECISION_BACKENDS}
    with mock.patch.object(nac, '_backend', side_effect=lambda path: fake_backends[path]):
        yield

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'
DRAFT_YAML = Path(__file__).resolve().parents[1] / '.scratch' / 'sparse-rewiring-gsm8k' / 'launch-record-draft.yaml'


def _save_tiny_qwen2_and_tokenizer(root: Path) -> tuple[Path, Path, dict[str, str]]:
    """Save a real tiny FP32 Qwen2ForCausalLM (no pad_token_id in config) and local tokenizer to disk."""
    model_dir = root / 'model'
    tok_dir = root / 'artifacts' / 'tokenizer'
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Real local tokenizer with pad_token_id=0, bos_token_id=1, eos_token_id=2
    vocab = {'<pad>': 0, '<bos>': 1, '<|im_end|>': 2}
    for i in range(3, 64):
        vocab[f'tok{i}'] = i
    backend_tok = Tokenizer(WordLevel(vocab=vocab, unk_token='<pad>'))
    backend_tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok,
        pad_token='<pad>',
        bos_token='<bos>',
        eos_token='<|im_end|>',
        chat_template='{% for m in messages %}{{ m.content }}{% endfor %}',
    )
    hf_tok.save_pretrained(tok_dir)

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
    torch.manual_seed(19)
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)
    model.config._attn_implementation = 'eager'
    model.save_pretrained(model_dir)
    # Verify saved config.json has no pad_token_id (spec test 22)
    saved_cfg = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    saved_cfg.pop('pad_token_id', None)
    (model_dir / 'config.json').write_text(json.dumps(saved_cfg, indent=2), encoding='utf-8')

    gen_cfg = root / 'artifacts' / 'generation_config.json'
    gen_cfg.write_text(json.dumps({'eos_token_id': 2, 'do_sample': False}), encoding='utf-8')

    bundle = {
        'files': [
            {'role': 'tokenizer:tokenizer.json', 'path': 'tokenizer/tokenizer.json'},
            {'role': 'generation-config', 'path': 'generation_config.json'},
        ]
    }
    bundle_path = root / 'artifacts' / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')

    pinned = rc.hash_model_downloads(model_dir)['files']
    return model_dir, bundle_path, pinned


def _make_record(root: Path, model_dir: Path, bundle_path: Path) -> dict:
    """Build a rehearsal launch record from the real draft YAML with tiny 3-step settings."""
    record = yaml.safe_load(DRAFT_YAML.read_text(encoding='utf-8'))
    for key, text in record['decisions'].items():
        record['decisions'][key] = text.replace('[proposed]', '[user test]')
    record['mode'] = 'rehearsal'
    record['label'] = 'rewire-cpu-chain-test'
    record['run_id'] = 'rewire-cpu-chain-test'
    record['code_commit'] = 'deadbeef'
    record['pod_id'] = 'pod-cpu-1'
    record['volume_identity'] = 'vol-test-01'
    record['export_repo'] = 'test-org/test-private'
    record['training']['steps'] = 3
    record['training']['warmup_steps'] = 1
    record['training']['freeze_after_step'] = 2
    record['checkpoints'] = [0, 2, 3]
    record['cadences']['gate_snapshot_every'] = 2
    record['cadences']['validation_ce_every'] = 2
    record['cadences']['settle']['first'] = 2
    record['cadences']['settle']['every'] = 2
    record['evaluation']['cap'] = 4
    record['evaluation']['eos_token_id'] = 2
    record['evaluation']['batch_size'] = 2
    record['evaluation']['cohort150']['steps'] = [3]
    record['inputs'].update(
        artifact_root=str(root / 'artifacts'),
        bundle=str(bundle_path),
        model_dir=str(model_dir),
        work_dir=str(root / 'work'),
    )
    (root / 'work').mkdir(parents=True, exist_ok=True)
    return record


class TestRewirePodChain(unittest.TestCase):
    """Spec tests 20 (real-path CPU chain) and 24 (no legacy allow_tf32; IEEE readback required)."""

    def test_spec_test_20_real_path_cpu_chain_and_relaunch_keep_completed_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned = _save_tiny_qwen2_and_tokenizer(root)
            record = _make_record(root, model_dir, bundle_path)
            volume = root / 'volume'
            volume.mkdir()
            hf_cache = root / 'hf_cache'
            hf_cache.mkdir()

            probes = {
                'pinned_base_model': pinned,
                'disk_usage': lambda _p: shutil._ntuple_diskusage(
                    total=int(record['disk']['container_total_gb'] * 1.1e9),
                    used=int(10e9),
                    free=int(record['disk']['min_free_gb'] * 1.5e9),
                ),
                'cuda_available': True,
                'volume_identity': lambda: [],
                'private_repo': lambda _repo: [],
                'driver': lambda: 'cpu-test-driver',
            }

            rows = [
                {'id': f'e{i}', 'input_ids': [5, 6, 7 + i, 8 + i, 2], 'labels': [-100, -100, 7 + i, 8 + i, 2]}
                for i in range(8)
            ]
            step_batches = [[f'e{i}' for i in range(8)] for _ in range(3)]
            arm_plans = {
                'sparse-gold-sft-r1': {'rank': 1, 'step_batches': step_batches, 'checkpoint_steps': [0, 2, 3]},
                'sparse-gold-sft-r0': {'rank': 0, 'step_batches': step_batches, 'checkpoint_steps': [0, 2, 3]},
            }
            val_items = rows[:2]
            val_questions = {'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}, 'v1': {'prompt_ids': [5, 6, 8], 'gold': '7'}}
            c150_questions = {'c0': {'prompt_ids': [5, 6, 9], 'gold': '12'}}

            exported_calls: list[dict] = []

            def _stub_exporter(local_root, **kwargs):
                exported_calls.append({'local_root': str(local_root), **kwargs})
                return {'readback_verified': True, 'completeness': 'complete'}

            stop_cmds: list[Any] = []

            def _stub_runner(cmd, **kwargs):
                stop_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

            env_patch = {'HF_HOME': str(hf_cache), 'HF_HUB_OFFLINE': '1'}
            with _ieee_backends_if_torch_lt_29(), mock.patch.dict(os.environ, env_patch, clear=False):
                # Attempt 1: r1 trains & evaluates cleanly; simulate a mid-run crash during r0 training
                backend = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                chain1 = rc.build_rewire_chain(
                    record=record,
                    backend=backend,
                    local_root=root / 'work',
                    volume=volume,
                    training_rows=rows,
                    arm_plans=arm_plans,
                    val_items=val_items,
                    val_questions=val_questions,
                    val_question_ids=['v0', 'v1'],
                    cohort150_questions=c150_questions,
                    cohort150_question_ids=['c0'],
                    arm_order=['sparse-gold-sft-r1', 'sparse-gold-sft-r0'],
                    gradient_checkpointing=False,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )

                def _crash_r0(_ctx):
                    partial = root / 'work' / rc.REWIRE_SUBTREE / 'runs' / record['arms']['sparse-gold-sft-r0']['run_id']
                    partial.mkdir(parents=True, exist_ok=True)
                    (partial / 'partial.tmp').write_text('interrupted', encoding='utf-8')
                    raise RuntimeError('simulated pod crash during r0')

                chain1.stages['train:sparse-gold-sft-r0'] = _crash_r0
                summary1 = chain1.run()
                self.assertEqual(summary1['state'], 'failed')
                self.assertEqual(summary1['stages']['train:sparse-gold-sft-r1']['status'], 'complete')
                self.assertEqual(summary1['stages']['eval:sparse-gold-sft-r1']['status'], 'complete')

                # Attempt 2 on a wiped local container disk restoring from volume:
                # preflight reruns, completed r1 stages are verified & kept (resumed=True), r0 restarts from step 0!
                shutil.rmtree(root / 'work')
                (root / 'work').mkdir()
                backend2 = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                chain2 = rc.build_rewire_chain(
                    record=record,
                    backend=backend2,
                    local_root=root / 'work',
                    volume=volume,
                    training_rows=rows,
                    arm_plans=arm_plans,
                    val_items=val_items,
                    val_questions=val_questions,
                    val_question_ids=['v0', 'v1'],
                    cohort150_questions=c150_questions,
                    cohort150_question_ids=['c0'],
                    arm_order=['sparse-gold-sft-r1', 'sparse-gold-sft-r0'],
                    gradient_checkpointing=False,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )
                summary2 = chain2.run()

                self.assertEqual(summary2['state'], 'complete', f"relaunch failed: {summary2.get('failures')}")
                self.assertGreaterEqual(len(exported_calls), 1)
                self.assertGreaterEqual(len(stop_cmds), 1)
                self.assertFalse(summary2['stages']['preflight'].get('resumed', False))
                self.assertTrue(summary2['stages']['train:sparse-gold-sft-r1']['resumed'])
                self.assertTrue(summary2['stages']['eval:sparse-gold-sft-r1']['resumed'])
                self.assertFalse(summary2['stages']['train:sparse-gold-sft-r0'].get('resumed', False))
                self.assertFalse(summary2['stages']['eval:sparse-gold-sft-r0'].get('resumed', False))

                # Verify preflight hashed every model download
                pf = json.loads((root / 'work' / rc.PREFLIGHT_DIR / 'preflight.json').read_text(encoding='utf-8'))
                self.assertEqual(pf['model_downloads'], pinned)
                self.assertEqual(au.verify_run(root / 'work' / rc.PREFLIGHT_DIR), {'preflight.json'})

                # Verify both arms' checkpoints and sealed eval units
                for arm in ('sparse-gold-sft-r1', 'sparse-gold-sft-r0'):
                    run_id = record['arms'][arm]['run_id']
                    arm_dir = root / 'work' / rc.REWIRE_SUBTREE / 'runs' / run_id
                    for step in (0, 2, 3):
                        self.assertEqual(
                            au.verify_run(arm_dir / 'checkpoints' / f'checkpoint-{step}'),
                            {reval.CHECKPOINT_WEIGHTS_FILE, reval.CHECKPOINT_META_FILE},
                        )
                    self.assertTrue((arm_dir / 'eval' / '3' / 'greedy-cohort150' / 'complete.json').is_file())
                    self.assertFalse((arm_dir / 'eval' / '0' / 'greedy-cohort150').exists())

    def test_spec_test_24_no_legacy_tf32_setter_and_ieee_readback_required(self) -> None:
        # 1. AST guard: none of the new sparse-rewiring modules accesses or assigns allow_tf32 in code
        pkg = Path(__file__).resolve().parents[1] / 'pilot_eval'
        for name in ('hard_concrete.py', 'rewire_dynamics.py', 'rewire_evaluation.py', 'rewire_training.py', 'rewire_chain.py'):
            tree = ast.parse((pkg / name).read_text(encoding='utf-8'), filename=name)
            bad = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == 'allow_tf32']
            self.assertEqual(bad, [], f'{name} must never access or assign legacy allow_tf32 (lines {bad})')

        # 2. Runtime guard: preflight refuses if any precision backend does not read 'ieee'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned = _save_tiny_qwen2_and_tokenizer(root)
            record = _make_record(root, model_dir, bundle_path)
            volume = root / 'volume'
            volume.mkdir()
            hf_cache = root / 'hf_cache'
            hf_cache.mkdir()
            probes = {
                'pinned_base_model': pinned,
                'disk_usage': lambda _p: shutil._ntuple_diskusage(total=int(500e9), used=int(10e9), free=int(400e9)),
                'cuda_available': True,
                'volume_identity': lambda: [],
                'private_repo': lambda _repo: [],
                'driver': lambda: 'cpu-test-driver',
            }
            with _ieee_backends_if_torch_lt_29(), mock.patch.dict(
                os.environ, {'HF_HOME': str(hf_cache), 'HF_HUB_OFFLINE': '1'}, clear=False
            ):
                backend = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                self.assertEqual(backend.precision_problems(), [])
                with mock.patch.object(
                    backend, 'precision_readback', return_value={'cuda_matmul': 'tf32', 'matmul_precision': 'highest'}
                ):
                    probs = backend.preflight()
                    self.assertTrue(any('IEEE FP32 precision not in effect' in p for p in probs))


if __name__ == '__main__':
    unittest.main()
