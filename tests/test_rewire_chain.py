"""Sparse rewiring ticket 14: Pod chain integration and P1 regression tests (spec tests 20 and 24).

Exercises real component boundaries:
- Real ``sparse_rewiring_plan.finalize_draft`` / ``load_launch_record`` / ``build_plan`` (P1.2, P1.6)
- Real ``RUNPOD_POD_ID`` lookup for missing & mismatching Pod IDs (P1.1)
- Real ``qualified_identity`` handoff, attempt-scoped preflight receipts, and row-content hash binding (P1.3)
- Real clean-directory recovery restoring ``training-complete.json``, logs, ``hardening.json``, and
  ``masks/trace/*.npz`` to build the dynamics figure and pass ``verify_arm`` (P1.4)
- Real crash AFTER ``CheckpointSink`` saves & seals ``checkpoint-0``, verifying ``copy_sealed`` returns
  ``verified=True`` with zero broken seals in the backup tree (P1.5)
- Real cohort150 refusal in rehearsal and 150-ID requirement in production (P1.7)
- Spec test 24: zero ``allow_tf32`` in AST + IEEE readback refusal.
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

from pilot_eval import attribution_chain as ac
from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_bundle as cabundle
from pilot_eval import c_attribution_chain as cc
from pilot_eval import neuron_audit_collector as nac
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_dynamics as rd
from pilot_eval import rewire_dynamics_figure as rdf
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_module as rm
from pilot_eval import rewire_standalone as rs
from pilot_eval import rewire_training as rt
from pilot_eval import sft_backend as sb
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.workflow import _hash

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'
DRAFT_YAML = Path(__file__).resolve().parents[1] / '.scratch' / 'sparse-rewiring-gsm8k' / 'launch-record-draft.yaml'


@contextlib.contextmanager
def _ieee_backends_if_torch_lt_29():
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        yield
        return
    fake_backends = {path: types.SimpleNamespace(fp32_precision='tf32') for path in nac.PRECISION_BACKENDS}
    with mock.patch.object(nac, '_backend', side_effect=lambda path: fake_backends[path]):
        yield


def _save_tiny_qwen2_and_tokenizer(root: Path) -> tuple[Path, Path, dict[str, Any], str]:
    model_dir = root / 'model'
    tok_dir = root / 'artifacts' / 'tokenizer'
    gold_adapter_dir = root / 'artifacts' / 'adapters' / 'gold-sft-step64'
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    gold_adapter_dir.mkdir(parents=True, exist_ok=True)

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
    saved_cfg = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    saved_cfg.pop('pad_token_id', None)
    (model_dir / 'config.json').write_text(json.dumps(saved_cfg, indent=2), encoding='utf-8')
    (model_dir / 'generation_config.json').write_text(json.dumps({'eos_token_id': 2, 'do_sample': False}), encoding='utf-8')

    gen_cfg = root / 'artifacts' / 'generation_config.json'
    gen_cfg.write_text(json.dumps({'eos_token_id': 2, 'do_sample': False}), encoding='utf-8')
    # Placeholder gold-sft adapter in sealed bundle for P1.6 adapter_dir resolution check
    (gold_adapter_dir / 'adapter_model.safetensors').write_bytes(b'stub-safetensors')

    file_hashes = {name: cabundle.sha256(model_dir / name) for name in cabundle.MODEL_FILES}
    pinned = cabundle.pinned_identities(model_dir)
    bundle = {
        'base_model': {'mode': 'required', 'file_sha256': file_hashes, 'pinned': pinned},
        'files': [
            {'role': 'tokenizer:tokenizer.json', 'path': 'tokenizer/tokenizer.json'},
            {'role': 'generation-config', 'path': 'generation_config.json'},
            {'role': 'adapter:gold-sft-step64:adapter_model.safetensors', 'path': 'adapters/gold-sft-step64/adapter_model.safetensors'},
        ],
    }
    bundle_path = root / 'artifacts' / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    return model_dir, bundle_path, pinned, base_sha


def _build_validated_record_and_canonical_plans(
    root: Path, model_dir: Path, bundle_path: Path, rows: list[dict], *, mode: str = 'rehearsal'
) -> tuple[dict, dict[str, dict]]:
    """Build launch record via ``finalize_draft`` + ``load_launch_record`` and plans via REAL ``sp.build_plan`` (P1.2)."""
    raw = yaml.safe_load(DRAFT_YAML.read_text(encoding='utf-8'))
    for key, text in raw['decisions'].items():
        raw['decisions'][key] = text.replace('[proposed]', '[user test]')
    raw['mode'] = mode
    raw['label'] = 'rehearsal-rewire-cpu-chain' if mode == 'rehearsal' else 'prod-rewire-cpu-chain'
    raw['volume_identity'] = 'vol-test-01'
    raw['export_repo'] = 'test-org/test-private'
    raw['training']['steps'] = 3
    raw['training']['warmup_steps'] = 1
    raw['training']['freeze_after_step'] = 2
    raw['data']['training_examples'] = len(rows)
    raw['data']['examples_per_step'] = 8
    raw['data']['gold_order_steps'] = min(len(rows) // 8, 3)
    raw['checkpoints'] = [0, 2, 3]
    raw['cadences']['gate_snapshot_every'] = 2
    raw['cadences']['validation_ce_every'] = 2
    raw['cadences']['settle']['first'] = 2
    raw['cadences']['settle']['every'] = 2
    raw['evaluation']['cap'] = 4
    raw['evaluation']['eos_token_id'] = 2
    raw['evaluation']['batch_size'] = 2
    raw['evaluation']['cohort150']['steps'] = [3]

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

    # Build canonical plans through the REAL producer sp.build_plan (adjusted to tiny model shape)
    tiny_record = dict(record, model=dict(record['model'], layers=2, hidden=16, intermediate=32))
    training_ids = [r['id'] for r in rows]
    gold_order = list(training_ids)
    inputs_sha = dict(training_items_sha256=_hash(rows), gold_order_sha256=_hash(gold_order))
    arm_plans = {
        arm: sp.build_plan(tiny_record, arm, training_ids, gold_order, inputs_sha256=inputs_sha)
        for arm in record['arms']
    }
    # Assert canonical plans do NOT have a fake top-level read_lr (they use optimizer.reads.lr)
    for p in arm_plans.values():
        assert 'read_lr' not in p and 'lr' in p['optimizer']['reads']
    return record, arm_plans


def _make_rows(n: int = 24) -> list[dict]:
    return [
        {'id': f'e{i}', 'input_ids': [5, 6, 7 + (i % 10), 8 + (i % 10), 2], 'labels': [-100, -100, 7 + (i % 10), 8 + (i % 10), 2]}
        for i in range(n)
    ]


class TestRewirePodChainP1Regressions(unittest.TestCase):
    """P1.1–P1.7 regression tests plus spec tests 20 and 24."""

    def test_p1_1_missing_and_mismatching_pod_id_refused_with_no_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, base_sha = _save_tiny_qwen2_and_tokenizer(root)
            rows = _make_rows(24)
            record, arm_plans = _build_validated_record_and_canonical_plans(root, model_dir, bundle_path, rows)
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
            stop_cmds: list[Any] = []

            def _stub_runner(cmd, **kwargs):
                stop_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

            for idx, bad_env_pod in enumerate(('', 'wrong-pod-999')):
                work_i = root / f'work_bad_{idx}'
                work_i.mkdir()
                env = {'HF_HOME': str(hf_cache), 'HF_HUB_OFFLINE': '1', 'RUNPOD_POD_ID': bad_env_pod}
                with _ieee_backends_if_torch_lt_29(), mock.patch.dict(os.environ, env, clear=False):
                    backend = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                    # P1.6 check: validated record resolves gold-sft adapter dir from inputs.sources
                    self.assertTrue(backend.adapter_dir('gold-sft').is_dir())
                    chain = rc.build_rewire_chain(
                        record=record,
                        backend=backend,
                        local_root=work_i,
                        volume=volume,
                        training_rows=rows,
                        arm_plans=arm_plans,
                        val_items=rows[:2],
                        val_questions={'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        val_question_ids=['v0'],
                        expected_base_sha256=base_sha,
                        execution_kind='real-cuda',
                        model_context_limit=64,
                        gradient_checkpointing=True,
                        exporter=lambda *a, **kw: {'readback_verified': True, 'completeness': 'complete'},
                        runner=_stub_runner,
                    )
                    summary = chain.run()
                    self.assertEqual(summary['state'], 'failed')
                    self.assertEqual(stop_cmds, [], f'must never issue stop on bad RUNPOD_POD_ID={bad_env_pod!r}')

    def test_p1_3_mutated_row_tokens_refused_under_same_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, _, base_sha = _save_tiny_qwen2_and_tokenizer(root)
            rows = _make_rows(24)
            _, arm_plans = _build_validated_record_and_canonical_plans(root, model_dir, bundle_path, rows)
            plan = arm_plans['sparse-gold-sft-r1']

            # Mutate a completion token while keeping the exact same row IDs
            mutated_rows = [dict(r) for r in rows]
            mutated_rows[0] = dict(mutated_rows[0], input_ids=[5, 6, 9, 9, 2], labels=[-100, -100, 9, 9, 2])
            model = Qwen2ForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager')
            rm.install(model, rank=1, seed=42)
            with self.assertRaises(ValueError) as ctx:
                rt.train_rewire(
                    model,
                    mutated_rows,
                    root / 'out',
                    plan=plan,
                    model_context_limit=64,
                    pad_token_id=0,
                    expected_base_sha256=base_sha,
                    execution_kind='real-cuda',
                )
            self.assertIn('training_items_sha256', str(ctx.exception))

    def test_p1_7_cohort150_forbidden_in_rehearsal_and_required_in_production(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, base_sha = _save_tiny_qwen2_and_tokenizer(root)
            rows = _make_rows(24)
            record, arm_plans = _build_validated_record_and_canonical_plans(root, model_dir, bundle_path, rows, mode='rehearsal')
            volume = root / 'volume'
            volume.mkdir()
            probes = {'pinned_base_model': pinned}

            with _ieee_backends_if_torch_lt_29():
                backend = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                # Rehearsal refuses cohort150 inputs
                with self.assertRaises(ValueError):
                    rc.build_rewire_stages(
                        record=record,
                        backend=backend,
                        training_rows=rows,
                        arm_plans=arm_plans,
                        val_items=rows[:2],
                        val_questions={'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        val_question_ids=['v0'],
                        cohort150_questions={'c0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        cohort150_question_ids=['c0'],
                    )
                # Production refuses missing cohort150 or != 150 unique IDs
                prod_record = dict(record, mode='production')
                with self.assertRaises(ValueError):
                    rc.build_rewire_stages(
                        record=prod_record,
                        backend=backend,
                        training_rows=rows,
                        arm_plans=arm_plans,
                        val_items=rows[:2],
                        val_questions={'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        val_question_ids=['v0'],
                    )
                with self.assertRaises(ValueError):
                    rc.build_rewire_stages(
                        record=prod_record,
                        backend=backend,
                        training_rows=rows,
                        arm_plans=arm_plans,
                        val_items=rows[:2],
                        val_questions={'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        val_question_ids=['v0'],
                        cohort150_questions={'c0': {'prompt_ids': [5, 6, 7], 'gold': '42'}},
                        cohort150_question_ids=['c0'],
                    )

    def test_p1_2_3_4_5_real_path_chain_crash_after_checkpoint_and_clean_recovery(self) -> None:
        """Full real-path CPU chain:
        - Real canonical sp.build_plan (P1.2)
        - Real RUNPOD_POD_ID env check (P1.1)
        - Real qualified_identity & attempt-scoped preflight with driver change (P1.3)
        - Real crash AFTER CheckpointSink writes & seals checkpoint-0 on r0 (P1.5)
        - Clean-directory restore via restore_from_volume + restore_arm_evidence -> verify_arm + dynamics figure (P1.4)
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, base_sha = _save_tiny_qwen2_and_tokenizer(root)
            rows = _make_rows(24)
            record, arm_plans = _build_validated_record_and_canonical_plans(root, model_dir, bundle_path, rows)
            volume = root / 'volume'
            volume.mkdir()
            hf_cache = root / 'hf_cache'
            hf_cache.mkdir()

            driver_name = {'v': 'driver-pod-A'}
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
                'driver': lambda: driver_name['v'],
                'reset_peak': lambda: None,
                'peak_memory': lambda: 0,
            }
            val_items = rows[:2]
            val_questions = {'v0': {'prompt_ids': [5, 6, 7], 'gold': '42'}, 'v1': {'prompt_ids': [5, 6, 8], 'gold': '7'}}

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

            env_patch = {
                'HF_HOME': str(hf_cache),
                'HF_HUB_OFFLINE': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'RUNPOD_POD_ID': 'pod-cpu-1',
            }
            with _ieee_backends_if_torch_lt_29(), mock.patch.dict(os.environ, env_patch, clear=False):
                # Attempt 1: r1 finishes; r0 crashes AFTER CheckpointSink writes & seals checkpoint-0!
                backend1 = rc.RewireBackend(record, volume, device='cpu', probes=probes)
                chain1 = rc.build_rewire_chain(
                    record=record,
                    backend=backend1,
                    local_root=root / 'work',
                    volume=volume,
                    training_rows=rows,
                    arm_plans=arm_plans,
                    val_items=val_items,
                    val_questions=val_questions,
                    val_question_ids=['v0', 'v1'],
                    arm_order=['sparse-gold-sft-r1', 'sparse-gold-sft-r0'],
                    expected_base_sha256=base_sha,
                    execution_kind='real-cuda',
                    model_context_limit=64,
                    gradient_checkpointing=True,
                    qualification_fn=_qual_ok,
                    references_fn=_refs_stub,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )

                r0_plan = arm_plans['sparse-gold-sft-r0']
                r0_rel = rc._arm_root(r0_plan['run_id'])

                def _crash_r0_after_real_ckpt0(ctx):
                    arm_dir = ctx.chain.local_root / r0_rel
                    m = backend1.load_rewired_model(rank=0, seed=42)
                    sink = rckpt.CheckpointSink(arm_dir, r0_plan, runtime_sha256=_hash(backend1.runtime()), code_commit='deadbeef')
                    sink(0, m)  # Real sealed checkpoint-0 written!
                    raise RuntimeError('simulated crash after real checkpoint-0')

                chain1.stages['train:sparse-gold-sft-r0'] = _crash_r0_after_real_ckpt0
                summary1 = chain1.run()
                self.assertEqual(summary1['state'], 'failed')
                self.assertEqual(summary1['signoff']['decision'], 'accept')
                # P1.3: qualified_identity is persisted and non-None!
                vol_state = json.loads((cc.run_root(volume, record['label']) / cc.STATE_FILE).read_text(encoding='utf-8'))
                self.assertIsNotNone(vol_state['qualified_identity'])
                self.assertEqual(vol_state['qualified_identity']['rows_sha256'], rt.rows_content_sha256(rows))

                # Attempt 2 on a replacement Pod (different host driver) with wiped local disk:
                driver_name['v'] = 'driver-pod-B'
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
                    arm_order=['sparse-gold-sft-r1', 'sparse-gold-sft-r0'],
                    expected_base_sha256=base_sha,
                    execution_kind='real-cuda',
                    model_context_limit=64,
                    gradient_checkpointing=True,
                    qualification_fn=_qual_ok,
                    references_fn=_refs_stub,
                    exporter=_stub_exporter,
                    runner=_stub_runner,
                )
                summary2 = chain2.run()
                self.assertEqual(summary2['state'], 'complete', f"relaunch failed: {summary2.get('failures')}")
                # P1.5: backup copy must be verified=True with zero broken seals!
                self.assertTrue(summary2['backups']['volume_copy']['verified'], summary2['backups']['volume_copy'])
                self.assertEqual(summary2['backups']['volume_copy']['coverage'], 'complete')
                self.assertTrue(summary2['stages']['train:sparse-gold-sft-r1']['resumed'])
                self.assertTrue(summary2['stages']['eval:sparse-gold-sft-r1']['resumed'])

                # P1.4: clean-directory restore from volume alone recovers everything needed to verify_arm & build dynamics figure!
                clean_restore = root / 'clean_restore'
                clean_restore.mkdir()
                ac.restore_from_volume(cc.run_root(volume, record['label']), clean_restore, subtree=cc.SUBTREE)
                for arm in ('sparse-gold-sft-r1', 'sparse-gold-sft-r0'):
                    plan = arm_plans[arm]
                    restored_arm = clean_restore / rc._arm_root(plan['run_id'])
                    rc.restore_arm_evidence(restored_arm)
                    rckpt.verify_arm(restored_arm, plan, code_commit='deadbeef')
                    self.assertTrue((restored_arm / 'hardening.json').is_file())
                    step_rows = rd.read_jsonl(restored_arm / 'logs' / 'steps.jsonl')
                    val_rows = rd.read_jsonl(restored_arm / 'logs' / 'validation.jsonl')
                    self.assertEqual([r['step'] for r in step_rows], [0, 1, 2, 3])
                    self.assertEqual([r['step'] for r in val_rows], list(plan['validation_steps']))
                    self.assertTrue((restored_arm / 'masks' / 'trace' / f"{plan['gate_snapshot_steps'][0]}.npz").is_file())
                    # P2.2: load_arm_artifacts reads & verifies the clean-restored arm end-to-end
                    loaded = rs.load_arm_artifacts(restored_arm, plan, code_commit='deadbeef')
                    self.assertEqual(loaded['arm'], arm)
                    self.assertEqual(loaded['rank'], plan['rank'])
                    self.assertTrue(loaded['final_export']['frozen'])
                    self.assertFalse(loaded['final_export']['survivors_omitted'])
                    self.assertIn(plan['gate_snapshot_steps'][0], loaded['snapshots'])
                    self.assertEqual([r['step'] for r in loaded['step_rows']], [0, 1, 2, 3])
                    self.assertEqual([r['step'] for r in loaded['val_ce_rows']], list(plan['validation_steps']))
                    self.assertEqual(loaded['hardening']['freeze_step'], 2)
                    self.assertIsNotNone(loaded['hardening']['post_binary'])
                    self.assertIsNotNone(loaded['hardening']['val_ce_before'])
                    self.assertIsNotNone(loaded['hardening']['val_ce_after'])
                    # Build plan-bound dynamics figure directly from the clean-restored logs
                    val_acc_rows = [{'step': s, 'accuracy': 0.5} for s in plan['checkpoints']]
                    prov = rdf.build_dynamics_figure(
                        {
                            'label': arm,
                            'step_rows': loaded['step_rows'],
                            'val_ce_rows': loaded['val_ce_rows'],
                            'val_acc_rows': val_acc_rows,
                        },
                        restored_arm / 'dynamics',
                        name=arm,
                        plan=plan,
                    )
                    self.assertTrue((restored_arm / 'dynamics' / f'{arm}.png').is_file())
                    self.assertEqual(prov['freeze_step'], 2)

    def test_spec_test_24_no_legacy_tf32_setter_and_ieee_readback_required(self) -> None:
        pkg = Path(__file__).resolve().parents[1] / 'pilot_eval'
        for name in ('hard_concrete.py', 'rewire_dynamics.py', 'rewire_evaluation.py', 'rewire_training.py', 'rewire_chain.py'):
            tree = ast.parse((pkg / name).read_text(encoding='utf-8'), filename=name)
            bad = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == 'allow_tf32']
            self.assertEqual(bad, [], f'{name} must never access or assign legacy allow_tf32 (lines {bad})')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, bundle_path, pinned, _ = _save_tiny_qwen2_and_tokenizer(root)
            rows = _make_rows(24)
            record, _ = _build_validated_record_and_canonical_plans(root, model_dir, bundle_path, rows)
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
