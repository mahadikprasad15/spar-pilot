"""Sparse rewiring ticket 14: Pod chain integration and real-path CPU test (spec tests 20 and 24).

Uses the real ``sparse_rewiring_plan.finalize_draft`` / ``load_launch_record`` / ``build_plan``,
real ``c_attribution_bundle.check_base_model``, real ``rewire_checkpoint.run_arm`` (PR #16),
and real ``c_attribution_chain.Chain`` with qualification sign-off on a saved tiny Qwen2ForCausalLM.
Stubs ONLY CUDA availability, disk usage, volume identity, Hub exporter, and Pod stop.
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
from pilot_eval import c_attribution_bundle as cabundle
from pilot_eval import c_attribution_chain as cc
from pilot_eval import neuron_audit_collector as nac
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import sft_backend as sb
from pilot_eval import sparse_rewiring_plan as sp

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'
DRAFT_YAML = Path(__file__).resolve().parents[1] / '.scratch' / 'sparse-rewiring-gsm8k' / 'launch-record-draft.yaml'


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


def _save_tiny_qwen2_and_tokenizer(root: Path) -> tuple[Path, Path, dict[str, Any], str]:
    """Save a real tiny FP32 Qwen2ForCausalLM (no pad_token_id in config), local tokenizer, and bundle."""
    model_dir = root / 'model'
    tok_dir = root / 'artifacts' / 'tokenizer'
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

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
    base_sha = sb.frozen_weight_hash(model, trained=rm.is_rewire_parameter)
    model.save_pretrained(model_dir)
    # Remove pad_token_id if transformers wrote it (spec test 22)
    saved_cfg = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    saved_cfg.pop('pad_token_id', None)
    (model_dir / 'config.json').write_text(json.dumps(saved_cfg, indent=2), encoding='utf-8')
    # Ensure generation_config.json exists in model_dir for cabundle.MODEL_FILES
    (model_dir / 'generation_config.json').write_text(json.dumps({'eos_token_id': 2, 'do_sample': False}), encoding='utf-8')

    gen_cfg = root / 'artifacts' / 'generation_config.json'
    gen_cfg.write_text(json.dumps({'eos_token_id': 2, 'do_sample': False}), encoding='utf-8')

    file_hashes = {name: cabundle.sha256(model_dir / name) for name in cabundle.MODEL_FILES}
    pinned = cabundle.pinned_identities(model_dir)
    bundle = {
        'base_model': {'mode': 'required', 'file_sha256': file_hashes, 'pinned': pinned},
        'files': [
            {'role': 'tokenizer:tokenizer.json', 'path': 'tokenizer/tokenizer.json'},
            {'role': 'generation-config', 'path': 'generation_config.json'},
        ],
    }
    bundle_path = root / 'artifacts' / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    return model_dir, bundle_path, pinned, base_sha


def _build_validated_record_and_plans(
    root: Path, model_dir: Path, bundle_path: Path, rows: list[dict]
) -> tuple[dict, dict[str, dict]]:
    """Load the draft YAML through ``sparse_rewiring_plan.finalize_draft`` + ``load_launch_record`` (typecheck + validate)."""
    raw = yaml.safe_load(DRAFT_YAML.read_text(encoding='utf-8'))
    for key, text in raw['decisions'].items():
        raw['decisions'][key] = text.replace('[proposed]', '[user test]')
    raw['mode'] = 'rehearsal'
    raw['label'] = 'rehearsal-rewire-cpu-chain'
    raw['volume_identity'] = 'vol-test-01'
    raw['export_repo'] = 'test-org/test-private'
    raw['training']['steps'] = 3
    raw['training']['warmup_steps'] = 1
    raw['training']['freeze_after_step'] = 2
    raw['data']['gold_order_steps'] = 3
    raw['checkpoints'] = [0, 2, 3]
    raw['cadences']['gate_snapshot_every'] = 2
    raw['cadences']['validation_ce_every'] = 2
    raw['cadences']['settle']['first'] = 2
    raw['cadences']['settle']['every'] = 2
    raw['evaluation']['cap'] = 4
    raw['evaluation']['eos_token_id'] = 2
    raw['evaluation']['batch_size'] = 2
    raw['evaluation']['cohort150']['steps'] = [3]
    raw['qualification']['disposable_steps'] = 2
    raw['qualification']['greedy_questions'] = 2

    draft_path = root / 'draft.yaml'
    draft_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding='utf-8')
    work_dir = root / 'work'
    work_dir.mkdir(parents=True, exist_ok=True)
    record_path = root / 'launch_record.json'
    sp.finalize_draft(
        draft_path,
        record_path,
        code_commit='deadbeef',
        pod_id='pod-cpu-1',
        fill={
            'inputs.artifact_root': str(root / 'artifacts'),
            'inputs.bundle': str(bundle_path),
            'inputs.model_dir': str(model_dir),
            'inputs.work_dir': str(work_dir),
        },
    )
    record = sp.load_launch_record(record_path, code_commit='deadbeef')

    # Build real PR #6 arm plans adjusted to our tiny Qwen2 shape (2 layers, hidden 16, intermediate 32)
    step_ids = [[r['id'] for r in rows] for _ in range(3)]
    arm_plans: dict[str, dict] = {}
    for arm, cfg in record['arms'].items():
        steps = [
            {
                'step': s,
                'ids': step_ids[s - 1],
                'lr_multiplier': sp.lr_multiplier(s, warmup=1, total=3),
                'masks': 'frozen-binary' if s > 2 else 'trainable',
                'mask_groups': [[0, 4], [4, 8]],
            }
            for s in range(1, 4)
        ]
        plan = {
            'schema': sp.PLAN_SCHEMA,
            'arm': arm,
            'run_id': cfg['run_id'],
            'rank': int(cfg['rank']),
            'read_lr': float(cfg['read_lr']),
            'checkpoints': list(record['checkpoints']),
            'settings': {
                'model': {'layers': 2, 'hidden': 16, 'intermediate': 32, 'parts': ['gate', 'up']},
                'training': dict(record['training']),
                'objective': dict(record['objective']),
                'seeds': dict(record['seeds']),
                'data': dict(record['data']),
                'storage': dict(record['storage']),
            },
            'steps': steps,
        }
        plan['checkpoint_bytes'] = rckpt.payload_bytes(plan)
        plan['plan_sha256'] = sp.plan_sha256(plan)
        arm_plans[arm] = plan
    return record, arm_plans


class TestRewirePodChain(unittest.TestCase):
    """Spec tests 20 (real-path CPU chain with PR #6 + PR #16 + sign-off) and 24 (precision)."""

    def test_non_rehearsal_chain_without_qualification_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, _, _ = _save_tiny_qwen2_and_tokenizer(root)
            rows = [{'id': f'e{i}', 'input_ids': [5, 6, 7, 2], 'labels': [-100, -100, 7, 2]} for i in range(8)]
            record, _ = _build_validated_record_and_plans(root, model_dir, bundle_path, rows)
            prod_record = dict(record, mode='production', run_id='prod-no-qual')
            chain = cc.Chain(
                prod_record,
                local_root=root / 'work',
                volume=root / 'vol',
                stages={'preflight': lambda _ctx: {}},
                volume_identity='vol-test-01',
                current_pod='pod-cpu-1',
                stage_order=['preflight'],
            )
            with self.assertRaises(RuntimeError):
                chain._check_signoff({'completed': {}})

    def test_spec_test_20_real_path_cpu_chain_with_signoff_and_pr16_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, base_sha = _save_tiny_qwen2_and_tokenizer(root)
            rows = [
                {'id': f'e{i}', 'input_ids': [5, 6, 7 + i, 8 + i, 2], 'labels': [-100, -100, 7 + i, 8 + i, 2]}
                for i in range(8)
            ]
            record, arm_plans = _build_validated_record_and_plans(root, model_dir, bundle_path, rows)
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
                'reset_peak': lambda: None,
                'peak_memory': lambda: 0,
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

            def _qual_ok(_ctx):
                return {'checks': {'noop': True, 'processor': True, 'cached_vs_forced': True}, 'arms': list(record['arms'])}

            def _refs_stub(ctx):
                au.seal_run(ctx.chain.local_root / f'{rc.REFERENCES_DIR}/summary', {'references.json': {'untuned': {'val_ce': 1.5}}})
                return {'outputs': [f'{rc.REFERENCES_DIR}/summary']}

            env_patch = {'HF_HOME': str(hf_cache), 'HF_HUB_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            with _ieee_backends_if_torch_lt_29(), mock.patch.dict(os.environ, env_patch, clear=False):
                # Attempt 1: qualification signs off, references + r1 complete, simulated crash mid-r0
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
                    expected_base_sha256=base_sha,
                    execution_kind='real-cuda',
                    model_context_limit=64,
                    gradient_checkpointing=False,
                    qualification_fn=_qual_ok,
                    references_fn=_refs_stub,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )

                r0_run_id = arm_plans['sparse-gold-sft-r0']['run_id']

                def _crash_r0(_ctx):
                    partial = root / 'work' / rc.REWIRE_SUBTREE / 'runs' / r0_run_id
                    partial.mkdir(parents=True, exist_ok=True)
                    (partial / 'partial.safetensors').write_bytes(b'partial')
                    raise RuntimeError('simulated pod crash during r0')

                chain1.stages['train:sparse-gold-sft-r0'] = _crash_r0
                summary1 = chain1.run()
                self.assertEqual(summary1['state'], 'failed')
                self.assertEqual(summary1['stages'].get('preflight', {}).get('status'), 'complete', f"failures: {summary1.get('failures')}")
                self.assertEqual(summary1['signoff']['decision'], 'accept')
                self.assertIn('train:sparse-gold-sft-r1', summary1['stages'], f"failures: {summary1.get('failures')}")
                self.assertEqual(summary1['stages']['train:sparse-gold-sft-r1']['status'], 'complete')
                self.assertEqual(summary1['stages']['eval:sparse-gold-sft-r1']['status'], 'complete')

                # Attempt 2: wipe local disk and restore from volume; r1 is verified & kept, r0 restarts via PR #16
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
                    expected_base_sha256=base_sha,
                    execution_kind='real-cuda',
                    model_context_limit=64,
                    gradient_checkpointing=False,
                    qualification_fn=_qual_ok,
                    references_fn=_refs_stub,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )
                summary2 = chain2.run()

                self.assertEqual(summary2['state'], 'complete', f"relaunch failed: {summary2.get('failures')}")
                self.assertGreaterEqual(len(exported_calls), 1)
                self.assertGreaterEqual(len(stop_cmds), 1)
                self.assertFalse(summary2['stages']['preflight'].get('resumed', False))
                self.assertTrue(summary2['stages']['qualification']['resumed'])
                self.assertTrue(summary2['stages']['references']['resumed'])
                self.assertTrue(summary2['stages']['train:sparse-gold-sft-r1']['resumed'])
                self.assertTrue(summary2['stages']['eval:sparse-gold-sft-r1']['resumed'])
                self.assertFalse(summary2['stages']['train:sparse-gold-sft-r0'].get('resumed', False))

                # Verify both arms' PR #16 checkpoints and sealed eval units
                for arm in ('sparse-gold-sft-r1', 'sparse-gold-sft-r0'):
                    plan = arm_plans[arm]
                    arm_dir = root / 'work' / rc.REWIRE_SUBTREE / 'runs' / plan['run_id']
                    for step in (0, 2, 3):
                        rckpt.verify_checkpoint(rckpt.checkpoint_dir(arm_dir, step), plan, code_commit='deadbeef')
                    self.assertTrue((arm_dir / 'eval' / '3' / 'greedy-cohort150' / 'complete.json').is_file())
                    self.assertFalse((arm_dir / 'eval' / '0' / 'greedy-cohort150').exists())

    def test_spec_test_24_no_legacy_tf32_setter_and_ieee_readback_required(self) -> None:
        pkg = Path(__file__).resolve().parents[1] / 'pilot_eval'
        for name in ('hard_concrete.py', 'rewire_dynamics.py', 'rewire_evaluation.py', 'rewire_training.py', 'rewire_chain.py'):
            tree = ast.parse((pkg / name).read_text(encoding='utf-8'), filename=name)
            bad = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == 'allow_tf32']
            self.assertEqual(bad, [], f'{name} must never access or assign legacy allow_tf32 (lines {bad})')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, _ = _save_tiny_qwen2_and_tokenizer(root)
            rows = [{'id': f'e{i}', 'input_ids': [5, 6, 7, 2], 'labels': [-100, -100, 7, 2]} for i in range(8)]
            record, _ = _build_validated_record_and_plans(root, model_dir, bundle_path, rows)
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
                'reset_peak': lambda: None,
                'peak_memory': lambda: 0,
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
