"""Ticket 11 with 06b-09: the real chain stages end to end on CPU (two-arm rehearsal, tiny random Qwen2 + PEFT).

Only the untuned *trace* generator is scripted (a random model never solves GSM8K); attribution, freeze,
Stage C, edited-checkpoint generation, the report and teardown all run the real code.
"""
import json
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import attribution_ablation as ab  # noqa: E402
from pilot_eval import attribution_chain as ch  # noqa: E402
from pilot_eval import attribution_stages as st  # noqa: E402
from pilot_eval import attribution_traces as at  # noqa: E402
from pilot_eval import attribution_units as au  # noqa: E402
from tests.test_attribution_instrument import PROJ  # noqa: E402

EOS, PAD = 31, 0
TOK = {i: str(i) for i in range(1, 10)}
TOK.update({10: '\n', 11: '#### ', 12: 'x ', EOS: '<|im_end|>', PAD: '<pad>'})


def decode(ids):
    return ''.join(TOK.get(i, 'y') for i in ids if i not in (EOS, PAD))


def make(seed, arm_seed=None):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                                        attn_implementation='eager'))
    if arm_seed is None:
        for p in base.parameters():
            p.requires_grad_(False)
        return base.eval()
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=2, lora_dropout=0, target_modules=PROJ))
    generator = torch.Generator().manual_seed(arm_seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_' in name:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.8)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


class Scripted:
    """Untuned trace generator: a fixed, mostly correct answer per prompt (length varies with the prompt)."""

    def generate(self, prompts):
        out = []
        for p in prompts:
            k = 1 + p[-1] % 3
            answer = 7 if p[0] % 5 else 6  # some wrong answers exercise rejection
            content = [12] * k + [10, 11, answer]
            out.append(dict(generated_ids=content + [EOS], stop_reason='eos', text=decode(content)))
        return out


class TinyBackend:
    def __init__(self, tmp, *, fail_stage_c_after=None):
        self.tmp, self.models, self.fail_after, self.calls = tmp, {}, fail_stage_c_after, 0

    def runtime(self):
        return dict(model='tiny-qwen2', torch=torch.__version__)

    def decoder(self):
        return dict(eos_token_id=EOS, pad_token_id=PAD)

    def token_text(self, i):
        return TOK.get(i, 'y')

    def decode(self, ids):
        return decode(ids)

    def decoding(self):
        return dict(mode='greedy', max_new_tokens=6, precision='fp32')

    def sources(self):
        return dict(provenance={'gold-sft': dict(status='available'), 'online-grpo': dict(status='partial')},
                    provenance_statement='no overlap with available sources')

    def unresolved_arms(self):
        return []

    def prompt_rows(self, split):
        offset, n = dict(calibration=(1, 4), discovery=(40, 8), heldout=(80, 8))[split]
        return [dict(id=f'{split}:{i}', split=split, source_ref=i,
                     prompt_ids=[(offset + i) % 29 + 1, 3, 4, (i * 7) % 29 + 1][:3 + i % 2] + [5])
                for i in range(n)]

    def gold(self, split):
        if split in ('rehearsal', 'cohort'):
            return {r['id']: '#### 7' for r in (self.rehearsal_items() if split == 'rehearsal' else self.cohort_items())}
        return {r['id']: 'w\n#### 7' for r in self.prompt_rows(split)}

    def untuned_generator(self, size):
        return Scripted()

    def load(self, arm):
        if arm not in self.models:
            self.models[arm] = make(0, arm_seed={'gold-sft': 11, 'online-grpo': 12}[arm])
        return self.models[arm]

    def preflight(self):
        return []

    def device_memory(self):
        return 10**9

    def reset_peak(self):
        pass

    def peak_memory(self):
        return 10**8

    def is_oom(self, exc):
        return False

    def rehearsal_items(self):
        return [dict(id=f'r{i}', prompt_ids=[i + 1, 2, 3][: 1 + i % 3]) for i in range(4)]

    def cohort_items(self):
        return [dict(id=f'gsm8k:test:{i}', prompt_ids=[i + 3, 4, 5, 6][: 2 + i % 3]) for i in range(5)]

    def stage_d_generator(self, arm, adapters, size, edit_dir):
        from peft import PeftModel
        if arm is None:
            return at.HFGenerator(make(0), pad_token_id=PAD, eos_token_id=EOS, max_new_tokens=6, decode=decode), None
        source = self.tmp / 'adapters' / arm
        if not source.exists():
            self.load(arm).save_pretrained(source)
        edit = None
        directory = source
        if adapters:
            from pilot_eval.attribution_instrument import adapter_modules
            names = [adapter_modules(self.load(arm))[i][:2] for i in adapters]
            edit = ab.edit_checkpoint(source, edit_dir, names)
            directory = edit_dir
        model = PeftModel.from_pretrained(make(0), directory).eval()
        return at.HFGenerator(model, pad_token_id=PAD, eos_token_id=EOS, max_new_tokens=6, decode=decode), edit


def launch(tmp_path):
    rec = dict(
        schema=ch.LAUNCH_SCHEMA, run_id='rehearsal-cpu-1', mode='rehearsal', rehearsal_label='rehearsal-cpu-2arm',
        code_commit='abc', pod_id='pod-1', pod_stop_commands=['echo stop {id}'], expected_arms=['gold-sft', 'online-grpo'],
        volume_identity='vol', export_repo='owner/private', hourly_cost_usd=1.0,
        caps_seconds=dict(total=3600, preflight=600, qualification=600, base_traces=600, attribution_and_freeze=600,
                          stage_c=600, stage_d=600, report=600),
        tolerances=dict(finite_difference=dict(epsilon=1e-2, median_rel_err_max=0.05, max_rel_err_max=0.20),
                        neuron_sum_identity=1e-4, batch_invariance=1e-4, resolution_floor='10x batch invariance'),
        minimums=dict(discovery=3, heldout=3),
        selection=dict(top_group=10, top_singles=5, tolerance=0.2, seed=20000, random_seeds=[1, 2, 3], read_size=4),
        statistics=dict(resamples=200, seed=1234, alpha=0.05, margin=0.25, sparse_lower=0.7, dense_upper=0.3,
                        gate_singles_min=4, null_orders=100, null_seed=4),
        generation_batch_rule=dict(candidates=[1, 2, 4], items=4, max_flips=1, max_stop_changes=1),
        token_budget_sweep=[32, 64], watchdog_margin_seconds=60, marker_stale_seconds=600,
        timeouts_seconds=dict(copy=60, export_attempt=60, stop_attempt=10, writer_join=5),
        qualification=dict(calibration_minimum=2, fd_arms=['gold-sft'], fd_rows=2, fd_adapters=2, fd_neurons=2,
                           head_chunk=4, memory_fraction=0.8),
        inputs=dict(note='fixture backend'))
    path = tmp_path / 'launch_record.json'
    ch.seal_launch_record(rec, path)
    return ch.load_launch_record(path, code_commit='abc')


def chain(tmp_path, record, backend, stops):
    def exporter(local, **kw):
        return dict(revision='a' * 40, completeness='complete', readback_verified=True)

    def runner(cmd, **kw):
        stops.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '', '')

    stages = st.build(record, tmp_path / 'local', tmp_path / 'volume', backend=backend, code_sha256='c' * 64)
    return ch.Chain(record, local_root=tmp_path / 'local', volume_root=tmp_path / 'volume', stages=stages,
                    exporter=exporter, runner=runner, volume_identity='vol')


def test_rehearsal_chain_runs_every_real_stage_end_to_end(tmp_path):
    record, backend, stops = launch(tmp_path), TinyBackend(tmp_path), []
    summary = chain(tmp_path, record, backend, stops).run()
    assert summary['state'] == 'complete', json.dumps(summary, default=str)[:3000]
    assert summary['signoff']['by'] == ch.AUTO_REVIEWER
    local = tmp_path / 'local'
    for directory in [st.QUAL_DIR, ab.FREEZE_DIR, st.REPORT_DIR, 'attribution/traces/discovery/rows',
                      'attribution/stage_d/untuned/reference', 'attribution/stage_d/gold-sft/full']:
        au.verify_run(local / directory)
    freeze = json.loads((local / ab.FREEZE_DIR / 'freeze.json').read_text())
    assert set(freeze['arms']) == {'gold-sft', 'online-grpo'} and freeze['sources'][0]['split'] == 'discovery'
    report = json.loads((local / st.REPORT_DIR / 'report.json').read_text())
    assert report['primary1']['outcome'] in ('supported', 'contradicted', 'inconclusive')
    assert summary['backups']['volume_copy']['verified'] is True and stops == [['echo', 'stop', 'pod-1']]
    # the volume holds the sealed units and the freeze
    assert au.Units(tmp_path / 'volume', plan_sha256='x', code_sha256='x', runtime_sha256='x').state(
        au.unit_key('ab', 'gold-sft', batch=0)) == 'sealed'
    au.verify_run(tmp_path / 'volume' / ab.FREEZE_DIR)
    qual = json.loads((local / st.QUAL_DIR / 'qualification.json').read_text())
    assert all(qual['checks'].values()) and qual['generation_batch'] in (1, 2, 4)


def test_interrupted_stage_c_resumes_and_skips_sealed_units(tmp_path):
    record, stops = launch(tmp_path), []
    backend = TinyBackend(tmp_path)
    original = ab.StageC.evaluate
    calls = {'n': 0}

    def flaky(self, rows, condition):
        calls['n'] += 1
        if calls['n'] == 5:
            raise RuntimeError('simulated crash in Stage C')
        return original(self, rows, condition)

    ab.StageC.evaluate = flaky
    try:
        first = chain(tmp_path, record, backend, stops).run()
    finally:
        ab.StageC.evaluate = original
    assert first['state'] == 'failed' and 'simulated crash' in first['failures'][0]['error']
    sealed_before = {k for k, v in au.Units(tmp_path / 'local', plan_sha256='x', code_sha256='x',
                                            runtime_sha256='x').store.status().items() if v == 'sealed'}
    seen = []

    def counting(self, rows, condition):
        seen.append(condition.get('kind'))
        return original(self, rows, condition)

    ab.StageC.evaluate = counting
    try:
        second = chain(tmp_path, record, TinyBackend(tmp_path), stops).run()
    finally:
        ab.StageC.evaluate = original
    assert second['state'] == 'complete' and second['stages']['attribution_and_freeze'].get('resumed') is True
    assert any(k.startswith('c-gold-sft') for k in sealed_before)
    assert len(stops) == 2  # each run ended with a stop request
