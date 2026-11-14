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
    def __init__(self, tmp, *, fail_stage_c_after=None, strict_rehearsal=False):
        self.tmp, self.models, self.fail_after, self.calls = tmp, {}, fail_stage_c_after, 0
        self.strict = strict_rehearsal  # a rehearsal must never read discovery, confirmation or the cohort
        self.probes = []

    def _guard(self, what):
        if self.strict:
            raise AssertionError(f'rehearsal read a blind input: {what}')

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
        if split != 'calibration':
            self._guard(split)
        offset, n = dict(calibration=(1, 8), discovery=(40, 8), heldout=(80, 8))[split]
        return [dict(id=f'{split}:{i}', split=split, source_ref=i,
                     prompt_ids=[(offset + i) % 29 + 1, 3, 4, (i * 7) % 29 + 1][:3 + i % 2] + [5])
                for i in range(n)]

    def gold(self, split):
        if split not in ('calibration', 'rehearsal'):
            self._guard(f'gold:{split}')
        if split in ('rehearsal', 'cohort'):
            return {r['id']: '#### 7' for r in (self.rehearsal_items() if split == 'rehearsal' else self.cohort_items())}
        return {r['id']: 'w\n#### 7' for r in self.prompt_rows(split)}

    def gold_cohort(self):
        return self.gold('cohort')

    def untuned_generator(self, size):
        return Scripted()

    def load(self, arm):
        if arm not in self.models:
            self.models[arm] = make(0, arm_seed=11 + PRODUCTION_ARMS.index(arm))
        return self.models[arm]

    def preflight(self):
        return []

    def parameter_count(self):
        return sum(p.numel() for p in make(0).parameters())

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
        self._guard('cohort')
        return [dict(id=f'gsm8k:test:{i}', prompt_ids=[i + 3, 4, 5, 6][: 2 + i % 3]) for i in range(5)]

    def stage_d_generator(self, arm, adapters, size, edit_dir, *, probe_ids):
        self.probes.append(list(probe_ids))
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


def init_rows(tmp_path):
    path = tmp_path / 'init_rows.jsonl'
    path.write_text(''.join(json.dumps(dict(arm=a, layer=l, module='down_proj', cos_a=0.9, a_drift=0.1)) + '\n'
                            for a in ['gold-sft', 'online-grpo'] for l in (0, 1)))
    return path


def pilot_file(tmp_path, arm):
    path = tmp_path / f'pilot-{arm}.jsonl'
    path.write_text(''.join(json.dumps(dict(id=f'gsm8k:test:{i}', text='x\n#### 7' if i % 2 else 'x\n#### 3',
                                            stop_reason='eos')) + '\n' for i in range(5)))
    return path


PRODUCTION_ARMS = ['gold-sft', 'online-grpo', 'offline-grpo', 'sampling-sft-k0', 'sampling-sft-k2', 'sampling-sft-k5']


def launch(tmp_path, **overrides):
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
        inputs=dict(note='fixture backend', init_check_rows=str(init_rows(tmp_path))),
        compute=dict(fp32_peak_flops=1e9),
        stage_d=dict(early_stop_rule=dict(kind='fraction-of-untuned-same-item', ratio=0.5),
                     pilot_results={'gold-sft': dict(path=str(pilot_file(tmp_path, 'gold-sft')), filters={}),
                                    'online-grpo': dict(path=str(pilot_file(tmp_path, 'online-grpo')), filters={})}))
    rec.update(overrides)
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


def test_rehearsal_chain_uses_only_visible_inputs(tmp_path):
    """Rehearsal: calibration rows stand in for discovery and confirmation, the 20 rehearsal items for the cohort.

    The strict backend raises if anything asks for discovery, held-out or cohort data."""
    record, backend, stops = launch(tmp_path), TinyBackend(tmp_path, strict_rehearsal=True), []
    summary = chain(tmp_path, record, backend, stops).run()
    assert summary['state'] == 'complete', json.dumps(summary, default=str)[:3000]
    local = tmp_path / 'local'
    assert not (local / 'attribution/traces/discovery').exists() and not (local / 'attribution/traces/heldout').exists()
    freeze = json.loads((local / ab.FREEZE_DIR / 'freeze.json').read_text())
    assert freeze['rehearsal'] is True and freeze['sources'][0]['split'] == 'calibration'
    responses = st.stage_d_batches(local, 'gold-sft', 'full')
    assert {r['id'] for r in responses} == {'r0', 'r1', 'r2', 'r3'}
    report = json.loads((local / st.REPORT_DIR / 'report.json').read_text())
    assert report['stage_d']['pilot_comparison'] == dict(available=False, reason='rehearsal: no pilot items')
    assert report['rehearsal'] is True
    md = (local / st.REPORT_DIR / 'report.md').read_text()
    assert md.startswith('# REHEARSAL') and 'not scientific results' in md
    assert stops == [['echo', 'stop', 'pod-1']]


def test_production_freeze_refuses_calibration_inputs_and_rehearsal_freeze_is_refused_in_production(tmp_path):
    from tests.test_attribution_ablation import SETTINGS, discovery_inputs
    with pytest.raises(ValueError, match='discovery'):
        ab.build_freeze({'gold-sft': discovery_inputs()}, sources=[dict(split='calibration')], analysis={}, **SETTINGS)
    rehearsal = ab.build_freeze({'gold-sft': discovery_inputs()}, sources=[dict(split='calibration')], analysis={},
                                rehearsal=True, **SETTINGS)
    path = ab.seal_freeze(tmp_path, rehearsal)
    assert ab.conditions(path, 'gold-sft', 'trace', rehearsal=True)
    with pytest.raises(ValueError, match='rehearsal'):
        ab.conditions(path, 'gold-sft', 'trace')


def test_production_chain_runs_every_real_stage_end_to_end(tmp_path):
    record = launch(tmp_path, run_id='production-cpu-1', mode='production', expected_arms=PRODUCTION_ARMS)
    backend, stops = TinyBackend(tmp_path), []
    summary = chain(tmp_path, record, backend, stops).run()
    assert summary['state'] == 'complete', json.dumps(summary, default=str)[:3000]
    assert summary['signoff']['by'] == ch.AUTO_REVIEWER
    local = tmp_path / 'local'
    for directory in [st.QUAL_DIR, ab.FREEZE_DIR, st.REPORT_DIR, 'attribution/traces/discovery/rows',
                      'attribution/traces/heldout/rows', 'attribution/stage_d/untuned/reference/config',
                      'attribution/stage_d/gold-sft/full/b00000']:
        au.verify_run(local / directory)
    freeze = json.loads((local / ab.FREEZE_DIR / 'freeze.json').read_text())
    assert set(freeze['arms']) == set(PRODUCTION_ARMS) and freeze['sources'][0]['split'] == 'discovery'
    assert freeze.get('rehearsal') is False
    report = json.loads((local / st.REPORT_DIR / 'report.json').read_text())
    assert report['primary1']['outcome'] in ('supported', 'contradicted', 'inconclusive')
    assert report['rehearsal'] is False
    assert summary['backups']['volume_copy']['verified'] is True and stops == [['echo', 'stop', 'pod-1']]
    assert au.Units(tmp_path / 'volume', plan_sha256='x', code_sha256='x', runtime_sha256='x').state(
        au.unit_key('ab', 'gold-sft', batch=0)) == 'sealed'
    au.verify_run(tmp_path / 'volume' / ab.FREEZE_DIR)
    compute = report['compute']
    assert {'qualification', 'attribution_and_freeze', 'stage_c', 'stage_d'} <= set(compute)
    attribution = compute['attribution_and_freeze']
    assert attribution['tokens'] > 0 and 0 <= attribution['padding_fraction'] < 1 and attribution['profiled_units'] == 1
    assert attribution['nvml']['available'] is False and attribution['mfu']['available'] is True
    conc = report['descriptive']['concentration']
    assert set(conc) == set(PRODUCTION_ARMS) and 'neurons_to_50' in conc['gold-sft']['trace']
    assert report['descriptive']['resolution_floor']['rule'] == '10x batch invariance'
    assert report['descriptive']['a_vs_a0']['gold-sft']['trace']['cos_a'] == 0.9
    sd = report['stage_d']
    assert sd['early_stop_reference']['rule']['ratio'] == 0.5 and 'untuned|reference' in sd['outcomes']
    assert set(sd['pilot_comparison']) == {'gold-sft', 'online-grpo'} and sd['pilot_comparison']['gold-sft']['n'] == 5
    md = (local / st.REPORT_DIR / 'report.md').read_text()
    assert '## Stage D' in md and 'early-stop rate' in md and '## Concentration' in md and 'REHEARSAL' not in md
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


# --- review round 4 (5): recovery onto an empty local disk; Stage D per batch -------------------------

def _stale_marker(tmp_path):
    import socket
    (tmp_path / 'volume' / ch.MARKER_FILE).write_text(json.dumps(dict(
        run_id='rehearsal-cpu-1', host=socket.gethostname(), pid=999999, state='running', heartbeat_unix=0)))


def test_r4_5_resume_onto_an_empty_local_disk_restores_sealed_work_from_the_volume(tmp_path):
    import shutil
    record, stops = launch(tmp_path), []
    original = ab.StageC.evaluate

    def crash(self, rows, condition):
        raise RuntimeError('simulated Pod loss in Stage C')

    ab.StageC.evaluate = crash
    try:
        assert chain(tmp_path, record, TinyBackend(tmp_path, strict_rehearsal=True), stops).run()['state'] == 'failed'
    finally:
        ab.StageC.evaluate = original
    shutil.rmtree(tmp_path / 'local')  # a fresh Pod: nothing local survives
    _stale_marker(tmp_path)
    summary = chain(tmp_path, record, TinyBackend(tmp_path, strict_rehearsal=True), stops).run()
    assert summary['state'] == 'complete', json.dumps(summary, default=str)[:2000]
    assert summary['stages']['qualification'] == dict(status='complete', resumed=True)
    assert summary['restored_from_volume']['units'] and summary['restored_from_volume']['runs']


def test_r4_5_stage_d_is_sealed_per_batch_and_an_interrupted_condition_resumes(tmp_path):
    record, stops = launch(tmp_path), []
    original_generate, original_stage_d = at.generate, st.Stages.stage_d
    seen = dict(in_stage_d=False, calls=0, crash_at=None)

    def stage_d(self, ctx):
        seen['in_stage_d'] = True
        try:
            return original_stage_d(self, ctx)
        finally:
            seen['in_stage_d'] = False

    def generate(generator, rows, batches):
        if seen['in_stage_d']:
            seen['calls'] += 1
            if seen['calls'] == seen['crash_at']:
                raise RuntimeError('interrupted inside Stage D')
        return original_generate(generator, rows, batches)

    at.generate, st.Stages.stage_d = generate, stage_d
    try:
        seen['crash_at'] = 2  # the first Stage D batch is sealed, the second is interrupted
        first = chain(tmp_path, record, TinyBackend(tmp_path, strict_rehearsal=True), stops).run()
        assert first['state'] == 'failed' and first['failures'][0]['error'].endswith('interrupted inside Stage D')
        sealed = list((tmp_path / 'local' / st.STAGE_D_DIR).glob('*/*/b*/complete.json'))
        assert len(sealed) == 1  # exactly the batch generated before the interruption
        seen.update(calls=0, crash_at=None)
        _stale_marker(tmp_path)
        second = chain(tmp_path, record, TinyBackend(tmp_path, strict_rehearsal=True), stops).run()
    finally:
        at.generate, st.Stages.stage_d = original_generate, original_stage_d
    assert second['state'] == 'complete', json.dumps(second, default=str)[:2000]
    total = 0
    for config in sorted((tmp_path / 'local' / st.STAGE_D_DIR).glob('*/*/config/config.json')):
        total += len(json.loads(config.read_text())['batches'])
        arm, condition = config.parent.parent.parent.name, config.parent.parent.name
        assert st.stage_d_batches(tmp_path / 'local', arm, condition) is not None
    assert total > 1 and seen['calls'] == total - 1  # the batch sealed before the crash was not regenerated


def test_r4_5_edited_checkpoint_is_reused_or_rebuilt_after_an_interruption(tmp_path):
    from tests.test_attribution_instrument import tiny
    model = tiny(seed=2)
    model.save_pretrained(tmp_path / 'adapter')
    names = [(0, 'down_proj')]
    record = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited', names)
    again = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited', names)  # retry after success: reused
    assert again['edited_sha256'] == record['edited_sha256'] and again['reused'] is True
    (tmp_path / 'edited' / 'edit_record.json').unlink()  # interrupted before its record was written
    rebuilt = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited', names)
    assert rebuilt['edited_sha256'] == record['edited_sha256'] and rebuilt['reused'] is False
    other = ab.edit_checkpoint(tmp_path / 'adapter', tmp_path / 'edited', [(1, 'down_proj')])  # different edit
    assert other['zeroed_keys'] != record['zeroed_keys']


def test_r4_1_finite_differences_put_gates_on_the_input_device_and_move_metrics_to_cpu(monkeypatch):
    """CPU cannot show a CUDA device mismatch, so this fails on the two patterns that do: a gate built without
    the input's device, and a metric converted with .numpy() before .cpu(). The real check is 06a on a GPU."""
    import torch
    from pilot_eval import attribution_instrument as ai
    from pilot_eval import attribution_metrics as am
    from tests.test_attribution_instrument import rows as fixture_rows, tiny
    model = tiny()
    inst = ai.Instrument(model, stop_ids=[31], pad_token_id=0, identity_rtol=1e-4)
    real_full = torch.full

    def full(*args, **kwargs):
        assert 'device' in kwargs, 'gate created without an explicit device'
        return real_full(*args, **kwargs)

    class CudaLike:
        def __init__(self, tensor):
            self.tensor = tensor

        def detach(self):
            return self

        def cpu(self):
            return self.tensor.detach().cpu()

        def numpy(self):
            raise TypeError("can't convert cuda tensor to numpy; use .cpu() first")

    real_trace, real_stop = am.trace_loglik, am.stop_pull
    monkeypatch.setattr(torch, 'full', full)
    monkeypatch.setattr(am, 'trace_loglik', lambda *a, **k: CudaLike(real_trace(*a, **k)))
    monkeypatch.setattr(am, 'stop_pull', lambda *a, **k: (CudaLike(real_stop(*a, **k)[0]), real_stop(*a, **k)[1]))
    out = st.finite_difference_check(model, inst, fixture_rows()[:3], [31], epsilon=1e-2, adapters=1, neurons=1)
    assert np.isfinite(out['median_rel_err']) and out['checked']
