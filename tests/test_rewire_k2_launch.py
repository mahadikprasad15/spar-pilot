"""End-to-end offline CPU tests for Sampling SFT (k = 2) pod chain, strict input/gate enforcement, and recipe.

Covers all P1 & P2 contracts from ``.scratch/sparse-rewiring-sampling-k2/review-implementation.md``:
  - P1 #1: Sealed training export (refuses raw JSONL, missing seal entries, coverage mismatch),
           pinned adapter weights & config SHA-256 verification and plan binding, strict val128-k2
           seal, recipe, prompt-masking, completion-label identity, and single-EOS enforcement.
  - P1 #3: Production Run A defaults to seed 42 ONLY; check-run-b refuses unsealed/partial step-64
           checkpoints; exact 3-vs-4 survivor routing boundary; Run B refuses missing/tampered gate,
           mismatched inputs, or arbitrary --l0-lambda differing from sealed k2_headroom.json.
  - P1 #4: Stage 1 val128-k2 generator uses 4 total attempts with retry seeds [0, 1001, 1002, 1003]
           (spy-engine case succeeding only on the 4th attempt!) and HFK2Engine sets top_k=0.
  - P2 #5: Production reference bar records val128 CE, val128 greedy accuracy + tokens_mean, AND
           cohort150 greedy accuracy + tokens_mean for both untuned and dense k=2 references.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = get_peft_model = None
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO / 'scripts'))

from pilot_eval import attribution_units as au
from pilot_eval import c_attribution_bundle as cb
from pilot_eval import catalog as cat
from pilot_eval import neuron_audit_collector as nac
from pilot_eval import rewire_bundle as rb
from pilot_eval import rewire_chain as rc
from pilot_eval import rewire_checkpoint as rckpt
from pilot_eval import rewire_evaluation as reval
from pilot_eval import rewire_k2_launch as k2l
from pilot_eval import sampling_sft_sampler as sampler
from pilot_eval import sparse_rewiring_k2_plan as k2p
from pilot_eval.attribution_chain import LaunchRefused
from pilot_eval.training import file_hash
import extract_survivor_shifts as ess
import generate_val128_k2_targets as valgen

FIXTURE_CONFIG = Path(__file__).resolve().parent / 'fixtures' / 'rewire' / 'config.json'
K2_DIR = REPO / '.scratch' / 'sparse-rewiring-sampling-k2'
REHEARSAL_YAML = K2_DIR / 'launch-record-k2-rehearsal-draft.yaml'
MATCHED_YAML = K2_DIR / 'launch-record-k2-matched-draft.yaml'
SCALED_YAML = K2_DIR / 'launch-record-k2-scaled-draft.yaml'
TINY = {'model.layers': 2, 'model.hidden': 16, 'model.intermediate': 32}
EOS = 2


def _item(index: int, prompt: list[int], answer: list[int], *, gold: str | None = None):
    ids = prompt + answer + [EOS]
    return dict(
        id=f'gsm8k:train:{index}',
        source_index=index,
        input_ids=ids,
        prompt_tokens=len(prompt),
        labels=[-100] * len(prompt) + answer + [EOS],
        gold=gold if gold is not None else f'#### {index % 10}',
        question=f'What is {index % 10}?',
    )


class _SpySamplerEngine:
    """Mock sampling_sft_sampler engine testing the established 4-attempt retry policy [0, 1001, 1002, 1003].

    - pid 0: fails attempts 0, 1, 2 (wrong answer), succeeds ONLY on 4th attempt (retry=3, data_seed=1003)!
    - pid 1, 2: succeed immediately on initial attempt (retry=0, data_seed=0).
    - pid 3: fails all 4 attempts -> recorded in uncovered_ids.
    """

    def __init__(self, tok_map: dict[str, int], *, eos: int = EOS):
        self.tok = tok_map
        self.eos = eos
        self.seen_seeds_by_pid: dict[int, list[int]] = {}

    def clear_cache(self) -> None:
        return None

    def generate(self, requests):
        outs = []
        for req in requests:
            pid = int(req['problem_id'])
            seed = int(req['seed'])
            self.seen_seeds_by_pid.setdefault(pid, []).append(seed)
            # Determine which retry_data_seed produced this request_seed
            s3 = sampler.request_seed(1003, pid, int(req['block']), int(req['step']))
            is_attempt3 = (seed == s3)
            if pid == 3 or (pid == 0 and not is_attempt3):
                seq = [self.tok['####'], self.tok['9'], self.eos]
            else:
                ans = str((100 + pid) % 10)
                seq = [self.tok['####'], self.tok[ans], self.eos]
            max_t = int(req['max_tokens'])
            if len(seq) <= max_t:
                outs.append({'token_ids': seq, 'finish_reason': 'stop', 'stop_reason': self.eos})
            else:
                outs.append({'token_ids': seq[:max_t], 'finish_reason': 'length', 'stop_reason': None})
        return outs

    def score(self, requests):
        outs = []
        for req in requests:
            ids = list(req['response_ids'])
            start = len(req['prompt_ids'])
            val = -0.25 if int(req['step']) >= 0 else -0.40
            outs.append({
                'token_ids': ids,
                'positions': list(range(start, start + len(ids))),
                'logprobs': [val] * len(ids),
            })
        return outs


def _world(root: Path):
    """Tiny Qwen2 model + artifact root with sealed Gold bundle, sealed k=2 export, k=2 adapter, and val128-k2."""
    art = root / 'artifacts'
    mdir = art / cb.MODEL_DIR
    mdir.mkdir(parents=True)

    vocab = {
        '<pad>': 0, '<bos>': 1, '<|im_end|>': 2, '####': 3,
        **{str(d): 4 + d for d in range(10)},
        **{f'tok{i}': i for i in range(14, 64)},
    }
    backend_tok = Tokenizer(WordLevel(vocab=vocab, unk_token='<pad>'))
    backend_tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok,
        pad_token='<pad>',
        bos_token='<bos>',
        eos_token='<|im_end|>',
        chat_template='{% for m in messages %}{{ m.content }}{% endfor %}',
    )
    tok.save_pretrained(mdir)
    for name in ('vocab.json', 'merges.txt'):
        (mdir / name).write_text('{}' if name.endswith('json') else '#version: 0.2\n')
    (mdir / 'generation_config.json').write_text(json.dumps({'eos_token_id': EOS, 'do_sample': False, 'top_k': 20}))

    model_dir = root / 'model'
    cfg = json.loads(FIXTURE_CONFIG.read_text())
    cfg.update(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=EOS, torch_dtype='float32',
    )
    torch.manual_seed(19)
    base = Qwen2ForCausalLM(Qwen2Config(**cfg)).to(torch.float32)
    base.save_pretrained(model_dir)
    saved = json.loads((model_dir / 'config.json').read_text())
    saved.pop('pad_token_id', None)
    (model_dir / 'config.json').write_text(json.dumps(saved, indent=2))

    k2_adapter_rel = 'runs/sampling-sft/sampling-sft-v1/training/l4-night-v1-train-k2/checkpoints/checkpoint-64'
    for rel_dir, seed in ((rb.GOLD_RUN + '/checkpoints/checkpoint-64', 7), (k2_adapter_rel, 11)):
        adapter_dir = art / rel_dir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        if LoraConfig is not None and get_peft_model is not None:
            peft_model = get_peft_model(
                Qwen2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32),
                LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0),
            )
            with torch.no_grad():
                for name, p in peft_model.named_parameters():
                    if 'lora_B' in name:
                        p.normal_(0, 0.05)
            peft_model.save_pretrained(adapter_dir)
        else:
            save_file({'lora_A.weight': torch.randn(2, 16), 'lora_B.weight': torch.randn(16, 2) * 0.05},
                      str(adapter_dir / 'adapter_model.safetensors'))
            (adapter_dir / 'adapter_config.json').write_text(json.dumps({
                'peft_type': 'LORA', 'r': 2, 'lora_alpha': 4, 'target_modules': ['q_proj', 'v_proj'],
            }))

    # 16 total Gold slots, 14 covered k=2 rows (2 uncovered slots -> batch sizes 6 and 8!)
    gold_rows = [_item(i, [15, 16, 17 + i % 20], [18 + i % 30, 19]) for i in range(16)]
    gold_by_id = {r['id']: r for r in gold_rows}
    gold_order = [r['id'] for r in reversed(gold_rows)]
    uncovered_list = [gold_order[0], gold_order[1]]
    uncovered_set = set(uncovered_list)

    # Build full sealed sampling_sft_training_data export: training.jsonl + batches.json + coverage.json + export.complete.json
    k2_rows = []
    batches = []
    for step_idx in range(2):
        slot_ids = gold_order[step_idx * 8:(step_idx + 1) * 8]
        slots = []
        nonmissing = 0
        for s_pos, sid in enumerate(slot_ids):
            miss = sid in uncovered_set
            slots.append({'source_id': sid, 'missing': miss})
            if not miss:
                nonmissing += 1
                gr = gold_by_id[sid]
                row = _item(gr['source_index'], gr['input_ids'][:gr['prompt_tokens']], [22 + gr['source_index'] % 25, 23, 24])
                row.update(step=step_idx + 1, slot=s_pos)
                k2_rows.append(row)
        batches.append({'step': step_idx + 1, 'nonmissing_count': nonmissing, 'slots': slots})

    k2_export_dir = art / 'runs/sampling-sft/sampling-sft-v1/training-data/l4-training-data-v1'
    k2_arm_dir = k2_export_dir / 'arms' / 'k2'
    k2_arm_dir.mkdir(parents=True, exist_ok=True)
    (k2_arm_dir / 'training.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in k2_rows), encoding='utf-8')
    (k2_arm_dir / 'batches.json').write_text(json.dumps(batches, indent=2), encoding='utf-8')
    (k2_arm_dir / 'coverage.json').write_text(json.dumps({'uncovered_ids': uncovered_list, 'coverage_decision': 'ok'}), encoding='utf-8')
    export_seal_path = k2_export_dir / 'export.complete.json'
    export_seal_path.write_text(json.dumps({
        'files': {
            rel: file_hash(k2_export_dir / rel)
            for rel in k2p.K2_EXPORT_FILES_REQUIRED
        },
    }, indent=2), encoding='utf-8')

    # 4 validation questions
    val = [_item(100 + i, [15, 16, 30 + i], [3, 4 + ((100 + i) % 10)]) for i in range(4)]
    val[-1] = _item(103, [15, 16, 40, 41, 42], [3, 7, 19], gold='#### 3')

    files = {
        rb.ROLES['gold:sft-config']: {'training_items_path': rb.ROLES['gold:training-items']},
        rb.ROLES['gold:training-items']: gold_rows,
        rb.ROLES['gold:evaluation-items']: [
            dict(id=f'gsm8k:test:{i}', source_index=i, prompt=f'tok15 tok16 tok{20 + i % 40}', gold=f'#### {i % 10}')
            for i in range(rb.COHORT150_SIZE)
        ],
        rb.ROLES['gold:order']: gold_order,
        rb.ROLES['validation:items']: val,
        rb.ROLES['validation:exclusions']: {
            'ids': [v['id'] for v in val],
            'ids_sha256': cat.ids_digest(v['id'] for v in val),
        },
    }
    for rel, value in files.items():
        (art / rel).parent.mkdir(parents=True, exist_ok=True)
        (art / rel).write_text(json.dumps(value))

    pinned = cb.pinned_identities(model_dir)
    bundle = cb.build(art, rb.default_entries(), model_dir=model_dir, pinned=pinned)
    bundle_path = root / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle))

    # Generate sealed val128-k2-targets.json WITH verified provenance, full IEEE precision readback & max_blocks=32!
    full_ieee_readback = {**{path: 'ieee' for path in nac.PRECISION_BACKENDS}, 'matmul_precision': 'highest'}
    provenance = valgen.collect_provenance(
        model_dir=model_dir,
        tokenizer_dir=mdir,
        val_items_path=art / rb.ROLES['validation:items'],
        bundle_path=bundle_path,
        pinned_base_model=pinned,
        engine_name='spy-cpu-fixture',
        precision_readback=full_ieee_readback,
    )
    spy_engine = _SpySamplerEngine(vocab, eos=EOS)
    val_k2_payload = valgen.generate_val128_k2_targets(
        val,
        engine=spy_engine,
        tokenizer=tok,
        eos_token_id=EOS,
        stop_token_ids=[EOS],
        k=2,
        max_retries=3,
        data_seed=0,
        block_limit=32,
        provenance=provenance,
    )
    val_k2_path = root / 'val128-k2-targets.json'
    val_k2_path.write_text(json.dumps(val_k2_payload, indent=2), encoding='utf-8')

    k2_adapter_dir = art / k2_adapter_rel
    return dict(
        art=art,
        mdir=mdir,
        model_dir=model_dir,
        tok=tok,
        vocab=vocab,
        val=val,
        provenance=provenance,
        bundle=bundle_path,
        val_k2_path=val_k2_path,
        val_k2_payload=val_k2_payload,
        k2_export_dir=k2_export_dir,
        k2_export_seal_sha=file_hash(export_seal_path),
        k2_adapter_weights_sha=file_hash(k2_adapter_dir / 'adapter_model.safetensors'),
        k2_adapter_config_sha=file_hash(k2_adapter_dir / 'adapter_config.json'),
        pinned=pinned,
        ids_sha256=files[rb.ROLES['validation:exclusions']]['ids_sha256'],
        spy_engine=spy_engine,
    )


def _record(root: Path, w, *, draft_path: Path = REHEARSAL_YAML, **overrides):
    raw = yaml.safe_load(draft_path.read_text())
    for k, text in raw['decisions'].items():
        raw['decisions'][k] = text.replace('[proposed]', '[user test]')
    raw.update(volume_identity='vol-test-01', export_repo='test-org/test-private')
    raw['model'].update(layers=2, hidden=16, intermediate=32)
    raw['data'].update(
        total_slots=16, training_examples=14, uncovered_slots=2,
        steps_per_epoch=2, examples_per_step_nominal=8, gold_order_steps=2,
    )
    raw['training'].update(steps=3, warmup_steps=1, freeze_after_step=2)
    raw['checkpoints'] = [0, 2, 3]
    raw['cadences'].update(validation_ce_every=2, gate_snapshot_every=2)
    raw['cadences']['settle'].update(first=2, every=2)
    raw['evaluation'].update(cap=4, eos_token_id=EOS, batch_size=2)
    raw['evaluation']['validation']['questions'] = 4
    raw['evaluation']['validation']['generator_engine'] = 'spy-cpu-fixture'
    raw['evaluation']['cohort150']['steps'] = [3]
    raw['qualification'].update(disposable_steps=2, greedy_questions=2, cached_vs_forced_rows=2)
    for dotted, value in overrides.items():
        node = raw
        *parents, key = dotted.split('.')
        for part in parents:
            node = node[part]
        node[key] = value
    draft = root / f'draft_{raw["label"]}.yaml'
    draft.write_text(yaml.safe_dump(raw, sort_keys=False))
    work = root / 'work'
    work.mkdir(exist_ok=True)
    fill = {
        'inputs.artifact_root': str(w['art']),
        'inputs.bundle': str(w['bundle']),
        'inputs.model_dir': str(w['model_dir']),
        'inputs.work_dir': str(work),
        'inputs.val128_k2_targets': str(w['val_k2_path']),
    }
    path = k2p.finalize_draft(draft, root / f'launch_{raw["label"]}.json', code_commit='deadbeef',
                              pod_id='pod-cpu-1', fill=fill)
    return k2p.load_launch_record(path, code_commit='deadbeef')


def _probes(w, record, *, peak=1):
    return {
        'pinned_base_model': w['pinned'],
        'disk_usage': lambda _p: shutil._ntuple_diskusage(
            total=int(record['disk']['container_total_gb'] * 1.1e9),
            used=int(1e9),
            free=int(record['disk']['container_total_gb'] * 1e9),
        ),
        'cuda_available': True,
        'volume_identity': lambda: [],
        'private_repo': lambda _repo: [],
        'driver': lambda: 'cpu-test-driver',
        'reset_peak': lambda: None,
        'peak_memory': lambda: peak,
        'total_memory': lambda: 10 ** 9,
    }


def _rewrite_final_survivor_count(arm_dir: Path, target_survivors: int) -> None:
    """Adjust final checkpoint log_alpha in a completed arm and re-seal it cleanly for 3-vs-4 gate boundary tests."""
    plan = json.loads((arm_dir / 'plan.json').read_text())
    arm_seal = json.loads((arm_dir / rckpt.ARM_SEAL).read_text())
    final_step = int(plan['steps'][-1]['step'])
    ckpt_dir = rckpt.checkpoint_dir(arm_dir, final_step)
    state_path = ckpt_dir / rckpt.STATE_FILE
    tensors = load_file(str(state_path))
    remaining = int(target_survivors)
    for key in sorted(tensors):
        if key.endswith('rewire_log_alpha'):
            la = torch.full_like(tensors[key], -10.0)
            take = min(remaining, la.numel())
            if take > 0:
                la[:take] = 10.0
                remaining -= take
            tensors[key] = la
    save_file(tensors, str(state_path))
    # Update checkpoint seal, arm seal, and arm_evidence seal to reflect the legitimate fixture state
    seal_path = ckpt_dir / rckpt.SEAL_FILE
    cseal = json.loads(seal_path.read_text())
    cseal['files'][rckpt.STATE_FILE] = file_hash(state_path)
    seal_path.write_text(json.dumps(cseal, indent=2, sort_keys=True) + '\n')
    arm_seal['checkpoints'][str(final_step)] = file_hash(seal_path)
    (arm_dir / rckpt.ARM_SEAL).write_text(json.dumps(arm_seal, indent=2, sort_keys=True) + '\n')
    ev_dir = arm_dir / rc.ARM_EVIDENCE_DIR
    if ev_dir.exists():
        shutil.rmtree(ev_dir)
    rc.seal_arm_evidence(arm_dir, plan)


class TestRewireK2Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.dict(k2p.RULES, TINY)]
        if not hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            fake_backends = {path: types.SimpleNamespace(fp32_precision='tf32') for path in nac.PRECISION_BACKENDS}
            self.patches.append(mock.patch.object(nac, '_backend', side_effect=lambda path: fake_backends[path]))
        for p in self.patches:
            p.start()
        self.w = _world(self.root)
        self.patches += [
            mock.patch.object(rb, 'VALIDATION_IDS_SHA256', self.w['ids_sha256']),
            mock.patch.object(k2p, 'K2_EXPORT_COMPLETE_SHA256', self.w['k2_export_seal_sha']),
            mock.patch.object(k2p, 'K2_ADAPTER_WEIGHTS_SHA256', self.w['k2_adapter_weights_sha']),
            mock.patch.object(k2p, 'K2_ADAPTER_CONFIG_SHA256', self.w['k2_adapter_config_sha']),
            mock.patch.dict(os.environ, {'HF_HOME': str(self.root / 'hf'), 'HF_HUB_OFFLINE': '1'}),
        ]
        (self.root / 'hf').mkdir()
        for p in self.patches[len(self.patches) - 5:]:
            p.start()
        self.exports, self.stops = [], []

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def _make_references_fn(self, record, backend):
        if LoraConfig is not None:
            return None

        def _no_peft_references(ctx):
            data = k2l.load_k2_inputs(record)
            c150, c150_ids = (None, None)
            if data['cohort150_items'] is not None:
                c150, c150_ids = k2l.rl.cohort150_questions(data['cohort150_items'], backend.encode)
            out_dir = ctx.chain.local_root / rc.REFERENCES_DIR
            model = backend.load_base_model()
            refs, prior_uses, subpaths = {}, [], []
            try:
                ce_untuned = float(reval.evaluate_validation_ce(model, data['val_items'])['val_ce'])
                for ref_arm, ref_step, ce_val in (('untuned', 0, ce_untuned), ('sampling-sft-k2', 64, max(ce_untuned - 0.05, 1e-4))):
                    vrel = f'{ref_arm}-step-{ref_step}/greedy-val'
                    gv = reval.evaluate_greedy(
                        model, data['val_questions'], data['val_question_ids'], out_dir / vrel,
                        arm=ref_arm, step=ref_step, cohort='gsm8k-train-val128', decode=backend.decode,
                        pad_token_id=backend.pad_token_id, eos_token_id=backend.eos_token_id,
                        cap=int(record['evaluation']['cap']), batch_size=int(record['evaluation']['batch_size']),
                        final_step=int(record['training']['steps']), mode=record['mode'],
                    )
                    subpaths.append(vrel)
                    prior_uses.append(gv['prior_use'])
                    entry = {
                        'step': ref_step, 'val_ce_k2': ce_val, 'val_ce': ce_val,
                        'val_k2_items': len(data['val_items']),
                        'val_accuracy': gv['scores']['accuracy'],
                        'val_tokens_mean': gv['scores']['tokens_mean'],
                        'val_tokens_median': gv['scores']['tokens_median'],
                        'eval_dir': gv['directory'],
                    }
                    if c150 is not None and c150_ids is not None:
                        crel = f'{ref_arm}-step-{ref_step}/greedy-cohort150'
                        gc = reval.evaluate_greedy(
                            model, c150, c150_ids, out_dir / crel,
                            arm=ref_arm, step=ref_step, cohort=reval.COHORT150_NAME, decode=backend.decode,
                            pad_token_id=backend.pad_token_id, eos_token_id=backend.eos_token_id,
                            cap=int(record['evaluation']['cap']), batch_size=int(record['evaluation']['batch_size']),
                            final_step=ref_step, mode=record['mode'],
                        )
                        subpaths.append(crel)
                        prior_uses.append(gc['prior_use'])
                        entry.update(
                            cohort150_accuracy=gc['scores']['accuracy'],
                            cohort150_tokens_mean=gc['scores']['tokens_mean'],
                            cohort150_tokens_median=gc['scores']['tokens_median'],
                            cohort150_eval_dir=gc['directory'],
                        )
                    refs[ref_arm] = entry
            finally:
                model = None
                backend.release()
            headroom = k2p.compute_scaled_lambda(refs['untuned']['val_ce_k2'], refs['sampling-sft-k2']['val_ce_k2'])
            headroom.update(
                val_k2_items=len(data['val_items']),
                val_greedy_questions=len(data['val_question_ids']),
                val_accuracy_untuned=refs['untuned']['val_accuracy'],
                val_accuracy_lora_k2=refs['sampling-sft-k2']['val_accuracy'],
                val_tokens_mean_untuned=refs['untuned']['val_tokens_mean'],
                val_tokens_mean_lora_k2=refs['sampling-sft-k2']['val_tokens_mean'],
                cohort150_accuracy_untuned=refs['untuned'].get('cohort150_accuracy'),
                cohort150_accuracy_lora_k2=refs['sampling-sft-k2'].get('cohort150_accuracy'),
                cohort150_tokens_mean_untuned=refs['untuned'].get('cohort150_tokens_mean'),
                cohort150_tokens_mean_lora_k2=refs['sampling-sft-k2'].get('cohort150_tokens_mean'),
                adapter_dir=str(data['adapter_dir']),
                adapter_sha256=data['adapter_hashes'],
                inputs_sha256=data['shared_inputs_sha'],
            )
            au.seal_run(
                out_dir / 'summary',
                {'references.json': refs, 'k2_headroom.json': headroom, 'prior_uses.json': prior_uses},
            )
            return {'outputs': [f'{rc.REFERENCES_DIR}/summary', *(f'{rc.REFERENCES_DIR}/{s}' for s in subpaths)],
                    'references': refs, 'k2_headroom': headroom}
        return _no_peft_references

    def _chain(self, record, *, local_name='local', only_arm=None, gate_receipt=None, matched_s42_dir=None,
               peak=1, stop_pod=True):
        def exporter(local_root, **kwargs):
            self.exports.append(kwargs)
            return {'readback_verified': True, 'completeness': 'complete'}

        def runner(cmd, **_kw):
            self.stops.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout='ok', stderr='')

        backend = rc.RewireBackend(
            record, self.root / 'volume', device='cpu', probes=_probes(self.w, record, peak=peak)
        )
        return k2l.build(
            record,
            self.root / local_name,
            self.root / 'volume',
            only_arm=only_arm,
            gate_receipt=gate_receipt,
            matched_s42_dir=matched_s42_dir,
            backend=backend,
            references_fn=self._make_references_fn(record, backend),
            exporter=exporter,
            runner=runner,
            volume_identity='vol-test-01',
            current_pod='pod-cpu-1',
            stop_pod=stop_pod,
        )

    def test_p1_4_val128_k2_generator_four_attempts_seeds_and_hf_top_k_zero(self):
        payload = self.w['val_k2_payload']
        self.assertEqual(payload['schema'], k2l.VAL128_K2_SCHEMA)
        self.assertEqual(payload['recipe']['retry_data_seeds'], [0, 1001, 1002, 1003])
        self.assertEqual(payload['recipe']['total_attempts_max'], 4)
        self.assertEqual(payload['recipe']['top_k'], 0)
        self.assertEqual(payload['covered_count'], 3)
        self.assertEqual(payload['uncovered_ids'], ['gsm8k:train:103'])
        # pid 0 failed attempts 0, 1, 2 and succeeded ONLY on 4th attempt (retry=3, seed=1003)!
        row0 = next(r for r in payload['items'] if r['id'] == 'gsm8k:train:100')
        self.assertEqual(row0['retry'], 3)
        self.assertEqual(row0['attempt_seed'], 1003)
        # Verify HFK2Engine resolved generation kwargs explicitly disable top_k (top_k=0)
        model = Qwen2ForCausalLM.from_pretrained(self.w['model_dir'], torch_dtype=torch.float32)
        eng = valgen.HFK2Engine(model, stops=[EOS], pad_token_id=0, batch_size=2)
        kw = eng.resolved_generation_kwargs(max_new_tokens=64)
        self.assertEqual(kw['top_k'], 0)
        self.assertEqual(kw['temperature'], 0.6)
        self.assertEqual(kw['top_p'], 1.0)
        self.assertEqual(kw['repetition_penalty'], 1.1)
        self.assertIsNone(kw['min_p'])
        # Verify batched left-padded HFK2Engine matches batch_size=1 on variable-length prompts & per-row seeds
        eng1 = valgen.HFK2Engine(model, stops=[EOS], pad_token_id=0, batch_size=1)
        reqs = [
            {'prompt_ids': [1, 3, 4], 'max_tokens': 5, 'seed': 101},
            {'prompt_ids': [1, 5, 6, 7, 8], 'max_tokens': 4, 'seed': 202},
        ]
        self.assertEqual(eng.generate(reqs), eng1.generate(reqs))
        # Evidence preserved: decoded text, full attempts log, and terminal disposition for uncovered ID
        self.assertIn('text', row0)
        self.assertGreaterEqual(len(payload['attempts']), 6)
        self.assertIn('gsm8k:train:103', payload['uncovered_dispositions'])
        self.assertEqual(payload['uncovered_dispositions']['gsm8k:train:103']['attempts_count'], 4)

        # P2 Stage 1 interruption & resume produces bit-for-bit identical targets and evidence!
        prog_path = self.root / 'resume_test.progress.json'
        paused = valgen.generate_val128_k2_targets(
            self.w['val'],
            engine=_SpySamplerEngine(self.w['vocab'], eos=EOS),
            tokenizer=self.w['tok'],
            eos_token_id=EOS,
            stop_token_ids=[EOS],
            k=2, max_retries=3, data_seed=0, block_limit=32,
            provenance=self.w['provenance'],
            progress_path=prog_path,
            stop_after_blocks=1,
        )
        # Note: all mock items finish in 1 block on retry 0; still verify multi-retry resume via progress file
        self.assertTrue(prog_path.is_file())
        # Force a mid-run interruption right after retry 0 completes by truncating completed_retries to [0]
        prog_data = json.loads(prog_path.read_text())
        if prog_data['completed_retries'] == [0, 1, 2, 3]:
            # Simulate interruption after retry 0
            prog_data['completed_retries'] = [0]
            prog_data['selected_by_id'] = {k: v for k, v in prog_data['selected_by_id'].items() if v['retry'] == 0}
            prog_data['attempts_log'] = [a for a in prog_data['attempts_log'] if a['retry'] == 0]
            prog_data['active_retry'] = None
            prog_data['active_resume'] = None
            prog_path.write_text(json.dumps(prog_data))
        resumed = valgen.generate_val128_k2_targets(
            self.w['val'],
            engine=_SpySamplerEngine(self.w['vocab'], eos=EOS),
            tokenizer=self.w['tok'],
            eos_token_id=EOS,
            stop_token_ids=[EOS],
            k=2, max_retries=3, data_seed=0, block_limit=32,
            provenance=self.w['provenance'],
            progress_path=prog_path,
        )
        self.assertEqual(resumed['items_sha256'], payload['items_sha256'])
        self.assertEqual(resumed['items'], payload['items'])
        self.assertEqual(resumed['attempts'], payload['attempts'])
        self.assertEqual(resumed['uncovered_dispositions'], payload['uncovered_dispositions'])

    def test_p1_stage1_default_pin_flow_cli_and_no_bundle_and_base_weight_binding(self):
        """Regression for review-ab186ff: real default pin flow (with/without bundle, CLI main) & base-weight export check."""
        val_path = self.w['art'] / rb.ROLES['validation:items']
        with mock.patch.object(cb, 'PINNED_BASE_MODEL', self.w['pinned']):
            # 1. Default pinned_base_model=None WITH bundle succeeds (no TypeError)
            p_with = valgen.collect_provenance(
                model_dir=self.w['model_dir'], tokenizer_dir=self.w['mdir'],
                val_items_path=val_path, bundle_path=self.w['bundle'],
                precision_readback={**{p: 'ieee' for p in nac.PRECISION_BACKENDS}, 'matmul_precision': 'highest'},
            )
            self.assertEqual(sorted(p_with['model_file_sha256']), sorted(cb.MODEL_FILES))
            # 2. Default pinned_base_model=None WITHOUT bundle succeeds (no KeyError) and verifies pins
            p_no = valgen.collect_provenance(
                model_dir=self.w['model_dir'], tokenizer_dir=self.w['mdir'],
                val_items_path=val_path, bundle_path=None,
            )
            self.assertEqual(p_no['model_file_sha256'], p_with['model_file_sha256'])
            # 3. Exercise real CLI valgen.main() default pin flow end-to-end!
            cli_out = self.root / 'cli_val128_k2.json'
            with mock.patch.object(valgen, 'HFK2Engine', return_value=_SpySamplerEngine(self.w['vocab'], eos=EOS)):
                rc_code = valgen.main([
                    '--val-items', str(val_path),
                    '--model-dir', str(self.w['model_dir']),
                    '--tokenizer-dir', str(self.w['mdir']),
                    '--bundle', str(self.w['bundle']),
                    '--out', str(cli_out),
                    '--device', 'cpu',
                    '--skip-id-hash-check',
                ])
            self.assertEqual(rc_code, 0)
            cli_payload = json.loads(cli_out.read_text())
            self.assertEqual(cli_payload['provenance']['engine'], 'hf-fp32-ieee')
            self.assertEqual(cli_payload['provenance']['model_file_sha256'], p_with['model_file_sha256'])
            # 4. P2 #1: extract_survivor_shifts.verify_base_model_and_tokenizer refuses modified weights w/ same config!
            ok_base = ess.verify_base_model_and_tokenizer(self.w['model_dir'], self.w['mdir'], self.w['bundle'])
            self.assertEqual(ok_base['model_file_sha256'], p_with['model_file_sha256'])
            mut_model = self.root / 'mutated_model'
            shutil.copytree(self.w['model_dir'], mut_model)
            # Corrupt model.safetensors bytes while leaving config.json untouched
            st_file = mut_model / 'model.safetensors'
            st_file.write_bytes(st_file.read_bytes() + b'\x00')
            with self.assertRaises((cb.BundleError, ValueError)):
                ess.verify_base_model_and_tokenizer(mut_model, self.w['mdir'], self.w['bundle'])
            # Without bundle, collect_provenance ALSO refuses mutated model weights (never self-certifies!)
            with self.assertRaises((cb.BundleError, ValueError)):
                valgen.collect_provenance(
                    model_dir=mut_model, tokenizer_dir=self.w['mdir'],
                    val_items_path=val_path, bundle_path=None,
                )

    def test_p1_1_strict_input_refusals_export_adapter_and_malformed_val_targets(self):
        rec = _record(self.root, self.w)
        # 1. Direct JSONL instead of sealed export dir -> refused
        bad_rec = json.loads(json.dumps(rec))
        bad_rec['inputs']['k2_training_data'] = str(self.w['k2_export_dir'] / 'arms/k2/training.jsonl')
        with self.assertRaises(LaunchRefused):
            k2l.load_k2_inputs(bad_rec)
        # 2. Export missing required sealed entry (batches.json omitted from seal) -> refused
        bad_exp = self.root / 'bad_export'
        shutil.copytree(self.w['k2_export_dir'], bad_exp)
        s = json.loads((bad_exp / 'export.complete.json').read_text())
        del s['files']['arms/k2/batches.json']
        (bad_exp / 'export.complete.json').write_text(json.dumps(s))
        bad_rec = json.loads(json.dumps(rec))
        bad_rec['inputs']['k2_training_data'] = str(bad_exp)
        with self.assertRaises(LaunchRefused):
            k2l.load_k2_inputs(bad_rec)
        # 3. Unsealed raw list val128-k2 targets -> refused
        raw_val = self.root / 'raw_list.json'
        raw_val.write_text(json.dumps(self.w['val_k2_payload']['items']))
        bad_rec = json.loads(json.dumps(rec))
        bad_rec['inputs']['val128_k2_targets'] = str(raw_val)
        with self.assertRaises(LaunchRefused):
            k2l.load_k2_inputs(bad_rec)
        # 4. Malformed labels (scoring prompt tokens, substituted completion label, missing EOS) -> refused
        malformed = json.loads(self.w['val_k2_path'].read_text())
        malformed['items'][0]['input_ids'] = malformed['items'][0]['input_ids'][:3] + [13]
        malformed['items'][0]['labels'] = [15, 16, 30, 999]   # prompt unmasked + substituted label + no EOS!
        malformed['items_sha256'] = k2p._hash(malformed['items'])
        mal_path = self.root / 'malformed_val.json'
        mal_path.write_text(json.dumps(malformed))
        bad_rec = json.loads(json.dumps(rec))
        bad_rec['inputs']['val128_k2_targets'] = str(mal_path)
        with self.assertRaises(LaunchRefused):
            k2l.load_k2_inputs(bad_rec)
        # 4b. Provenance & full recipe refusals: proposer, budget, stops, engine, executable digest, IEEE precision!
        for label, mutator in (
            ('wrong_model', lambda d: d['provenance']['model_file_sha256'].update({'config.json': '0' * 64})),
            ('block_limit_1', lambda d: d['recipe'].update({'max_blocks': 1})),
            ('wrong_stops', lambda d: d['recipe'].update({'stop_token_ids': [EOS, 999]})),
            ('wrong_accept_rule', lambda d: d['recipe'].update({'accept_rule': 'always-accept'})),
            ('unqualified_engine', lambda d: d['provenance'].update({'engine': 'unqualified-engine'})),
            ('arbitrary_exec_sha', lambda d: d['provenance'].update({'executable_sha256': '0' * 64})),
            ('non_ieee_matmul', lambda d: d['provenance']['precision_readback'].update({'matmul_precision': 'medium'})),
            ('missing_prec_backend', lambda d: d['provenance']['precision_readback'].pop(nac.PRECISION_BACKENDS[0])),
        ):
            mut = json.loads(self.w['val_k2_path'].read_text())
            mutator(mut)
            mp = self.root / f'bad_val_{label}.json'
            mp.write_text(json.dumps(mut))
            bad_rec = json.loads(json.dumps(rec))
            bad_rec['inputs']['val128_k2_targets'] = str(mp)
            with self.assertRaises(LaunchRefused, msg=f'should refuse {label}'):
                k2l.load_k2_inputs(bad_rec)
        # 5. Wrong adapter weights OR wrong adapter config in production -> refused; both hashes bound in plan!
        prod_rec = dict(rec, mode='production')
        cohort = self.w['art'] / rb.ROLES['gold:evaluation-items']
        with mock.patch.dict(rb.EXPECTED, {'gold:evaluation-items': cb.sha256(cohort)}):
            good = k2l.load_k2_inputs(prod_rec)
            first_plan = next(iter(good['arm_plans'].values()))
            self.assertEqual(first_plan['inputs']['k2_lora_adapter_weights_sha256'], self.w['k2_adapter_weights_sha'])
            self.assertEqual(first_plan['inputs']['k2_lora_adapter_config_sha256'], self.w['k2_adapter_config_sha'])
            with mock.patch.object(k2p, 'K2_ADAPTER_WEIGHTS_SHA256', '0' * 64):
                with self.assertRaises(LaunchRefused):
                    k2l.load_k2_inputs(prod_rec)
            with mock.patch.object(k2p, 'K2_ADAPTER_CONFIG_SHA256', '0' * 64):
                with self.assertRaises(LaunchRefused):
                    k2l.load_k2_inputs(prod_rec)

    def test_k2_rehearsal_runs_both_seeds_with_distinct_checkpoints(self):
        record = _record(self.root, self.w)
        (self.root / 'volume').mkdir()
        summary = self._chain(record).run()
        self.assertEqual(summary['state'], 'complete', summary.get('failures'))
        self.assertEqual(summary['signoff']['decision'], 'accept')
        local = self.root / 'local'
        arms = list(record['arms'].keys())
        final = record['training']['steps']
        states = {}
        for arm in arms:
            arm_dir = local / rc.REWIRE_SUBTREE / 'runs' / record['arms'][arm]['run_id']
            self.assertFalse(any((arm_dir / 'eval').glob('*/greedy-cohort150')))
            states[arm] = load_file(str(rckpt.checkpoint_dir(arm_dir, final) / rckpt.STATE_FILE))
        self.assertTrue(any(not torch.equal(states[arms[0]][k], states[arms[1]][k]) for k in states[arms[0]]))

    def test_p1_3_and_p2_5_production_default_s42_gate_boundary_3_vs_4_and_run_b_routing(self):
        cohort = self.w['art'] / rb.ROLES['gold:evaluation-items']
        self.patches.append(mock.patch.dict(rb.EXPECTED, {'gold:evaluation-items': cb.sha256(cohort)}))
        self.patches[-1].start()
        (self.root / 'volume').mkdir()

        # 1. Production Run A default (only_arm=None) MUST schedule seed 42 ONLY (never seed 43!)
        run_a = dict(_record(self.root, self.w, **{'label': 'rehearsal-k2-a'}), mode='production')
        s42_arm = 'sparse-k2-sft-r1-rehearsal-s42'
        s43_arm = 'sparse-k2-sft-r1-rehearsal-s43'
        summary_a = self._chain(run_a, local_name='local_a', stop_pod=False).run()
        self.assertEqual(summary_a['state'], 'complete', summary_a.get('failures'))
        self.assertIn(f'train:{s42_arm}', summary_a['stages'])
        self.assertNotIn(f'train:{s43_arm}', summary_a['stages'])
        self.assertEqual(self.stops, [])

        # P2 #5: Production reference summary MUST include val128 + cohort150 accuracy & tokens_mean for BOTH refs
        ref_summary_dir = self.root / 'local_a' / rc.REFERENCES_DIR / 'summary'
        refs = json.loads((ref_summary_dir / 'references.json').read_text())
        headroom = json.loads((ref_summary_dir / 'k2_headroom.json').read_text())
        for ref_name in ('untuned', 'sampling-sft-k2'):
            self.assertIn('val_tokens_mean', refs[ref_name])
            self.assertIn('cohort150_accuracy', refs[ref_name])
            self.assertIn('cohort150_tokens_mean', refs[ref_name])
        self.assertIsNotNone(headroom['cohort150_accuracy_untuned'])
        self.assertIsNotNone(headroom['cohort150_tokens_mean_lora_k2'])

        # 2. Unsealed partial checkpoint-64 directory MUST be refused by check_run_b_trigger
        partial_dir = self.root / 'partial_arm'
        (partial_dir / 'checkpoints' / 'checkpoint-64').mkdir(parents=True)
        save_file({'model.layers.0.mlp.rewire_log_alpha': torch.full((32,), -10.0)},
                  str(partial_dir / 'checkpoints' / 'checkpoint-64' / rckpt.STATE_FILE))
        with self.assertRaises(LaunchRefused):
            k2l.check_run_b_trigger(partial_dir)

        s42_dir = self.root / 'local_a' / rc.REWIRE_SUBTREE / 'runs' / run_a['arms'][s42_arm]['run_id']
        # Running Run A seed 43 without a verified gate must be refused
        with self.assertRaises(LaunchRefused):
            self._chain(run_a, local_name='local_a43_nogate', only_arm=s43_arm)

        # 3. Exact 4-survivor boundary (>= 4): allows Run A seed 43, REFUSES Run B!
        _rewrite_final_survivor_count(s42_dir, 4)
        receipt4_path = self.root / 'gate_4.json'
        gate4 = k2l.check_run_b_trigger(s42_dir, headroom_dir=ref_summary_dir, write_receipt=receipt4_path)
        self.assertEqual(gate4['survivor_count'], 4)
        self.assertFalse(gate4['trigger_run_b'])
        # Run A seed 43 now authorized!
        chain_s43 = self._chain(run_a, local_name='local_a43', only_arm=s43_arm, gate_receipt=receipt4_path, stop_pod=False)
        self.assertIn(f'train:{s43_arm}', chain_s43.stages)
        # Run B refused at 4 survivors!
        run_b_4 = dict(
            _record(
                self.root, self.w,
                **{
                    'label': 'rehearsal-k2-b4',
                    'objective.lambda_regime': 'scaled',
                    'objective.l0_lambda': float(headroom['lambda_scaled']),
                    'routing.gate_receipt': str(receipt4_path),
                    'routing.headroom_summary': str(ref_summary_dir),
                },
            ),
            mode='production',
        )
        with self.assertRaises(LaunchRefused):
            self._chain(run_b_4, local_name='local_b4')

        # 4. Exact 3-survivor boundary (< 4): REFUSES Run A seed 43, ALLOWS Run B ONLY at sealed lambda_scaled!
        _rewrite_final_survivor_count(s42_dir, 3)
        receipt3_path = self.root / 'gate_3.json'
        # Exercise direct CLI check-run-b entrypoint too
        self.assertEqual(
            k2l.main(['check-run-b', '--arm-dir', str(s42_dir), '--headroom-dir', str(ref_summary_dir),
                      '--out', str(receipt3_path)]),
            0,
        )
        gate3 = json.loads(receipt3_path.read_text())
        self.assertEqual(gate3['survivor_count'], 3)
        self.assertTrue(gate3['trigger_run_b'])
        # Direct seal_k2_draft refuses mismatched --l0-lambda against sealed headroom!
        scaled_draft_raw = yaml.safe_load(SCALED_YAML.read_text())
        for k, text in scaled_draft_raw['decisions'].items():
            scaled_draft_raw['decisions'][k] = text.replace('[proposed]', '[user test]')
        scaled_draft_raw.update(volume_identity='vol-test-01', export_repo='test-org/test-private')
        scaled_draft_raw['model'].update(layers=2, hidden=16, intermediate=32)
        scaled_draft_raw['data'].update(
            total_slots=16, training_examples=14, uncovered_slots=2,
            steps_per_epoch=2, examples_per_step_nominal=8, gold_order_steps=2,
        )
        scaled_draft_raw['training'].update(steps=3, warmup_steps=1, freeze_after_step=2)
        scaled_draft_raw['checkpoints'] = [0, 2, 3]
        scaled_draft_raw['cadences'].update(validation_ce_every=2, gate_snapshot_every=2)
        scaled_draft_raw['cadences']['settle'].update(first=2, every=2)
        scaled_draft_raw['evaluation'].update(cap=4, eos_token_id=EOS, batch_size=2)
        scaled_draft_raw['evaluation']['validation']['questions'] = 4
        scaled_draft_raw['evaluation']['validation']['generator_engine'] = 'spy-cpu-fixture'
        scaled_draft_raw['evaluation']['cohort150']['steps'] = [3]
        scaled_draft_raw['qualification'].update(disposable_steps=2, greedy_questions=2, cached_vs_forced_rows=2)
        scaled_draft_path = self.root / 'scaled_cli_draft.yaml'
        scaled_draft_path.write_text(yaml.safe_dump(scaled_draft_raw, sort_keys=False))
        fake_ident = {'code_commit': 'deadbeef', 'executable_sha256': k2p.executable_code_sha256()}
        with self.assertRaises(LaunchRefused):
            k2l.seal_k2_draft(
                scaled_draft_path, self.root / 'bad_sealed_b.json',
                artifact_root=str(self.w['art']), bundle=str(self.w['bundle']),
                model_dir=str(self.w['model_dir']), work_dir=str(self.root / 'work'),
                val128_k2_targets=str(self.w['val_k2_path']), pod_id='pod-cpu-1',
                l0_lambda=float(headroom['lambda_scaled']) * 2.0,
                gate_receipt=str(receipt3_path), headroom_summary=str(ref_summary_dir),
                code_identity=fake_ident,
            )
        # Run A seed 43 refused at 3 survivors!
        with self.assertRaises(LaunchRefused):
            self._chain(run_a, local_name='local_a43_blocked', only_arm=s43_arm, gate_receipt=receipt3_path)
        # Run B with arbitrary wrong lambda refused!
        run_b_wrong_lam = dict(
            _record(
                self.root, self.w,
                **{
                    'label': 'rehearsal-k2-bwrong',
                    'objective.lambda_regime': 'scaled',
                    'objective.l0_lambda': float(headroom['lambda_scaled']) * 1.5,
                    'routing.gate_receipt': str(receipt3_path),
                    'routing.headroom_summary': str(ref_summary_dir),
                },
            ),
            mode='production',
        )
        with self.assertRaises(LaunchRefused):
            self._chain(run_b_wrong_lam, local_name='local_bwrong')
        # 5. Completed-arm activation shift exporter (P2 #6): real --arm-dir succeeds & rejects step-0!
        fresh_base = Qwen2ForCausalLM.from_pretrained(self.w['model_dir'], torch_dtype=torch.float32)
        mods_loaded, prov_loaded = ess._load_verified_final_checkpoint(
            fresh_base, arm_dir=s42_dir, val_items_path=self.w['val_k2_path']
        )
        self.assertTrue(prov_loaded['verified'])
        self.assertEqual(prov_loaded['final_step'], run_a['training']['steps'])
        surv_loaded = ess._extract_survivors(mods_loaded)
        self.assertEqual(len(surv_loaded), 3)
        shifts_loaded = ess.measure_activation_shifts(
            fresh_base, mods_loaded, surv_loaded, self.w['val_k2_payload']['items'], self.w['tok']
        )
        self.assertEqual(len(shifts_loaded), 3)
        # Intermediate step-0 checkpoint MUST be rejected!
        with self.assertRaises(ValueError):
            ess._load_verified_final_checkpoint(
                Qwen2ForCausalLM.from_pretrained(self.w['model_dir'], torch_dtype=torch.float32),
                ckpt_dir=s42_dir / 'checkpoints' / 'checkpoint-0',
            )

        # 6. Wrong completed plan (l0_lambda=0.003, real seals, same 6 input digests) MUST be refused!
        wrong_plan_rec = dict(_record(self.root, self.w, **{'label': 'rehearsal-k2-wronglam', 'objective.l0_lambda': 3.0e-3}),
                              mode='production')
        summary_wrong = self._chain(wrong_plan_rec, local_name='local_wronglam', stop_pod=False).run()
        self.assertEqual(summary_wrong['state'], 'complete', summary_wrong.get('failures'))
        wrong_s42_dir = self.root / 'local_wronglam' / rc.REWIRE_SUBTREE / 'runs' / wrong_plan_rec['arms'][s42_arm]['run_id']
        _rewrite_final_survivor_count(wrong_s42_dir, 4)
        with self.assertRaises(LaunchRefused):
            self._chain(run_a, local_name='local_a43_wrongplan', only_arm=s43_arm, matched_s42_dir=wrong_s42_dir)

        # 7. Substituted headroom seal (same 6 inputs, different sealed lambda) MUST be refused by Run B!
        alt_headroom_dir = self.root / 'alt_headroom_summary'
        alt_h = json.loads((ref_summary_dir / 'k2_headroom.json').read_text())
        alt_h['ce_lora_k2'] = alt_h['ce_untuned_k2'] - 0.08
        alt_h['lambda_scaled'] = k2p.compute_scaled_lambda(alt_h['ce_untuned_k2'], alt_h['ce_lora_k2'])['lambda_scaled']
        au.seal_run(alt_headroom_dir, {'k2_headroom.json': alt_h, 'references.json': refs, 'prior_uses.json': []})
        run_b_sub_h = dict(
            _record(
                self.root, self.w,
                **{
                    'label': 'rehearsal-k2-bsubh',
                    'objective.lambda_regime': 'scaled',
                    'objective.l0_lambda': float(alt_h['lambda_scaled']),
                    'routing.gate_receipt': str(receipt3_path),
                    'routing.headroom_summary': str(alt_headroom_dir),
                },
            ),
            mode='production',
        )
        with self.assertRaises(LaunchRefused):
            self._chain(run_b_sub_h, local_name='local_bsubh')

        # 8. Authentic volume restore (arm & headroom moved to a clean restored path) STILL authorizes Run B!
        restored_arm = self.root / 'restored_vol' / 'arm_s42'
        restored_headroom = self.root / 'restored_vol' / 'headroom_summary'
        shutil.copytree(s42_dir, restored_arm)
        shutil.copytree(ref_summary_dir, restored_headroom)
        # Remove loose training-complete.json & plan.json in restored_arm to prove arm_evidence restore works!
        (restored_arm / rckpt.ARM_SEAL).unlink()
        (restored_arm / 'plan.json').unlink()
        run_b_restored = dict(
            _record(
                self.root, self.w,
                **{
                    'label': 'rehearsal-k2-brestored',
                    'objective.lambda_regime': 'scaled',
                    'objective.l0_lambda': float(headroom['lambda_scaled']),
                    'routing.gate_receipt': str(receipt3_path),
                    'routing.headroom_summary': str(restored_headroom),
                },
            ),
            mode='production',
        )
        chain_b_restored = self._chain(
            run_b_restored, local_name='local_brestored', matched_s42_dir=restored_arm, stop_pod=False
        )
        self.assertIn(f'train:{s42_arm}', chain_b_restored.stages)
        self.assertIn(f'train:{s43_arm}', chain_b_restored.stages)

        # Run B with exact sealed lambda_scaled authorized and builds both seeds!
        run_b_ok = dict(
            _record(
                self.root, self.w,
                **{
                    'label': 'rehearsal-k2-bok',
                    'objective.lambda_regime': 'scaled',
                    'objective.l0_lambda': float(headroom['lambda_scaled']),
                    'routing.gate_receipt': str(receipt3_path),
                    'routing.headroom_summary': str(ref_summary_dir),
                },
            ),
            mode='production',
        )
        chain_b = self._chain(run_b_ok, local_name='local_bok', stop_pod=False)
        self.assertIn(f'train:{s42_arm}', chain_b.stages)
        self.assertIn(f'train:{s43_arm}', chain_b.stages)


class TestK2DraftYAMLs(unittest.TestCase):
    def test_all_three_k2_drafts_match_schema(self):
        for draft in (REHEARSAL_YAML, MATCHED_YAML, SCALED_YAML):
            self.assertEqual(k2p.draft_problems(draft), [], draft.name)


if __name__ == '__main__':
    unittest.main()
