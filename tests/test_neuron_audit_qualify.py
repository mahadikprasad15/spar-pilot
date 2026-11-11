"""Ticket 03b runner on a tiny CPU Qwen/PEFT world: stages, gates, flags, fallbacks, resume."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import neuron_audit_collector as nc  # noqa: E402
from pilot_eval import neuron_audit_plan as plan_mod  # noqa: E402
from pilot_eval import neuron_audit_qualify as q  # noqa: E402
from pilot_eval.sft_backend import frozen_weight_hash  # noqa: E402

REAL_PLAN = Path(__file__).resolve().parents[1] / '.scratch/joint-neuron-adapter-audit/frozen_plan_v1.yaml'
PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
VOCAB = {1: '<|im_start|>', 2: 'assistant', 3: '\n', 4: '<|im_end|>', 5: 'user'}


class FakeTokenizer:
    eos_token_id = 4

    def token_text(self, token_id):
        return VOCAB.get(token_id, f' w{token_id}')

    def identity(self):
        return dict(kind='fake')


def base_model(dtype='float32'):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                                         pad_token_id=0, attn_implementation='eager'))
    return model.to(getattr(torch, dtype)).eval()


class FakeBackend:
    """CPU stand-in for TransformersBackend; memory use is simulated from the largest padded batch."""

    def __init__(self, adapter_dirs, reference_arm, fail_fp64_on=()):
        self.adapter_dirs, self.reference_arm, self.fail_fp64_on = adapter_dirs, reference_arm, set(fail_fp64_on)
        self.max_padded = 0

    def tokenizer(self):
        return FakeTokenizer()

    def fresh_base(self, dtype, device=None):
        return base_model(dtype)

    def load(self, arm, dtype, device=None):
        from peft import PeftModel
        if dtype == 'float64' and (device or 'gpu') in self.fail_fp64_on:
            raise RuntimeError('CUDA out of memory (simulated)')
        adapter = self.adapter_dirs[self.reference_arm if arm == 'untuned' else arm]
        return PeftModel.from_pretrained(base_model(dtype), adapter, torch_device='cpu').to(getattr(torch, dtype)).eval()

    def release(self, model):
        pass

    def device_info(self):
        return dict(name='fake-cpu', total_memory=10_000_000)

    def reset_peak(self):
        self.max_padded = 0

    def peak_memory(self):
        return self.max_padded * 20_000  # bytes per padded token (simulated)

    def synchronize(self):
        pass

    @staticmethod
    def is_oom(exc):
        return 'out of memory' in str(exc)


def row(i, n_content, group, split):
    prompt = [1, 5, 3, 6 + i % 20, 7, 4, 3]
    header = [1, 2, 3]
    content = [8 + (i + k) % 20 for k in range(n_content)]
    ids = prompt + header + content + [4, 3]
    r = len(prompt) + 2
    content_pos = list(range(r + 1, r + 1 + n_content))
    e = content_pos[-1] + 1
    return dict(id=f'{group}:{split}:{i}', group=group, split=split, input_ids=ids,
                masks=dict(response_content=content_pos, reply_start=[r], termination=[e - 1, e]))


@pytest.fixture
def world(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model
    root = tmp_path / 'artifacts'
    adapter_dirs, arms = {}, {}
    for seed, arm in enumerate(['gold-sft', 'offline-grpo'], start=1):
        model = get_peft_model(base_model(), LoraConfig(r=1, lora_alpha=2, lora_dropout=0, target_modules=PROJ))
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_' in name:
                    p.normal_(0, 0.5)
        directory = root / 'adapters' / arm
        model.save_pretrained(directory)
        adapter_dirs[arm] = str(directory)
        keys_rel = f'inv/keys/{arm}/step-64.json'
        from safetensors.numpy import load_file
        keys = sorted(load_file(directory / 'adapter_model.safetensors'))
        (root / keys_rel).parent.mkdir(parents=True, exist_ok=True)
        (root / keys_rel).write_text(json.dumps(dict(keys=keys)))
        arms[arm] = dict(checkpoints={'64': dict(config=dict(scale=2.0), key_list_path=keys_rel, files={
            'adapter_model.safetensors': dict(sha256=hashlib.sha256(
                (directory / 'adapter_model.safetensors').read_bytes()).hexdigest())})})
    base_hash = frozen_weight_hash(FakeBackend(adapter_dirs, 'gold-sft').load('gold-sft', 'float32'))
    inv = root / 'inv'
    (inv / 'inventory.json').write_text(json.dumps(dict(
        audit_step=64, base=dict(expected_base_parameter_sha256=base_hash), arms=dict(untuned=dict(), **arms))))
    seq = root / 'seq'
    seq.mkdir()
    rows = [row(i, 2 + i % 5, 'math' if i % 2 else 'general', 'calibration') for i in range(14)]
    rows += [row(100 + i, 2 + i % 6, 'math' if i % 2 else 'general', 'discovery' if i < 8 else 'heldout')
             for i in range(12)]
    (seq / 'sequences.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (seq / 'manifest.json').write_text(json.dumps(dict(sequences_sha256=sha(seq / 'sequences.jsonl'))))
    (seq / 'review').mkdir()
    (seq / 'review/background_sample.jsonl').write_text('{}\n')
    (seq / 'review/review.json').write_text(json.dumps(dict(
        decision='accept', sample_sha256=sha(seq / 'review/background_sample.jsonl'))))
    (seq / 'complete.json').write_text(json.dumps(dict(files={n: sha(seq / n) for n in ['sequences.jsonl',
                                                                                          'manifest.json']})))
    (inv / 'complete.json').write_text(json.dumps(dict(inventory_sha256=sha(inv / 'inventory.json'))))
    value = yaml.safe_load(REAL_PLAN.read_text())
    value.update(status='approved', approval=dict(approved_by='t', approved_on='2026-11-11', notes=None),
                 arms=['untuned', 'gold-sft', 'offline-grpo'])
    value['inputs'].update(inventory_run='inv', sequences_run='seq')
    value['runtime']['code_commit'] = 'c' * 40
    value['operations']['qualification_max_gpu_hours'] = 1
    value['preflight'].update(expected_sys_executable='x', expected_sys_prefix='y')
    value['calibration_allocation'].update(fp64_reference=dict(math=2, general=2), adr0008_calibrate=4,
                                           adr0008_validate=4)
    value['benchmark'].update(examples=[4, 14], token_budgets_padded=[48, 96, 200, 400])
    plan_path = tmp_path / 'plan.yaml'
    plan_path.write_text(yaml.safe_dump(value))
    plan = plan_mod.load(plan_path, '03b')
    volume = tmp_path / 'volume'
    volume.mkdir()
    config = dict(artifact_root=str(root), output_root=str(tmp_path / 'out'), run_id='q1', volume_dir=str(volume),
                  local_temp=str(tmp_path), adapter_dirs=adapter_dirs, reference_arm='gold-sft')
    backend = FakeBackend(adapter_dirs, 'gold-sft')
    original = nc.NeuronCollector.collect

    def observed(self, rows, **kw):
        backend.max_padded = max(backend.max_padded, len(rows) * max(len(r['input_ids']) for r in rows))
        return original(self, rows, **kw)

    monkeypatch.setattr(nc.NeuronCollector, 'collect', observed)
    preflight = lambda: dict(local_disk=dict(free_bytes=10 ** 12, total_bytes=10 ** 12),
                             hf_online=dict(usage=dict(observed_private_bytes=0)),
                             precision_device=dict(device='fake-cpu'))
    return dict(plan=plan, config=config, backend=backend, preflight=preflight, root=root, plan_value=value,
                tmp=tmp_path)


def qualification(world, **backend_kw):
    backend = world['backend'] if not backend_kw else FakeBackend(world['config']['adapter_dirs'], 'gold-sft',
                                                                  **backend_kw)
    return q.Qualification(world['plan'], world['config'], backend, preflight=world['preflight'])


def test_full_cpu_qualification_seals_every_stage_and_freezes_batches(world):
    summary = qualification(world).run()
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    assert summary['status'] == 'qualified' and summary['arms_qualified'] == ['gold-sft', 'offline-grpo', 'untuned']
    assert all((run_dir / 'stages' / s / 'complete.json').is_file() for s in q.STAGES)
    assert summary['ready_for_04'] is False and len(summary['blockers_for_04']) == 2
    bench = json.loads((run_dir / 'stages/benchmark/result.json').read_text())
    # simulated memory: 20 kB per padded token, limit 8 MB -> budgets above 400 tokens would not fit
    assert bench['selected_budget'] in (200, 400) and bench['vram_limit_bytes'] == 8_000_000
    assert bench['phase_1b']['scope'] in ('all', 'subset', 'omit')
    manifest = json.loads((run_dir / 'stages/batch_freeze/batch_manifest.json').read_text())
    assert manifest['examples'] == 12 and manifest['budget_padded_tokens'] == bench['selected_budget']
    assert all(len(b['ids']) * b['padded_width'] <= manifest['budget_padded_tokens'] for b in manifest['batches'])
    agree = json.loads((run_dir / 'stages/batch_agreement/result.json').read_text())
    assert max(agree['logits_max_abs'].values()) <= 1e-3 and agree['requires_signoff'] is True
    floor = json.loads((run_dir / 'stages/fp64_floor/result.json').read_text())['floors']['gold-sft']
    assert floor['method'] == 'fp64-gpu' and set(floor['layers']) == {'0', '1'}
    arms = json.loads((run_dir / 'stages/arms/result.json').read_text())
    assert arms['details']['gold-sft']['live_adapter']['status'] == 'pass'
    assert arms['details']['gold-sft']['disabled_vs_fresh_base'] == 0.0
    assert json.loads((run_dir / 'meta/status.json').read_text())['state'] == 'complete'


def test_resume_reuses_sealed_stages(world):
    first = qualification(world).run()
    second = qualification(world).run()
    assert first == second
    log = (Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1/logs/progress.jsonl').read_text()
    assert log.count('"reused"') == len(q.STAGES)


def test_stop_after_then_resume(world):
    partial = qualification(world).run(stop_after='arms')
    assert set(partial) == {'preflight', 'masks', 'arms'}
    assert qualification(world).run()['status'] == 'qualified'


def test_mask_gate_stops_with_evidence(world):
    seq = world['root'] / 'seq/sequences.jsonl'
    rows = [json.loads(l) for l in seq.read_text().splitlines()]
    rows[0]['masks']['reply_start'] = [rows[0]['masks']['reply_start'][0] + 1]  # shifted by one token
    seq.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(q.QualificationFailure, match='decoded-mask'):
        qualification(world).run()
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    assert list((run_dir / 'stages/masks').glob('failed-*.json'))
    assert json.loads((run_dir / 'meta/status.json').read_text())['state'] == 'failed'


def test_changed_adapter_fails_that_arm_and_labels_study_partial(world):
    path = Path(world['config']['adapter_dirs']['offline-grpo']) / 'adapter_model.safetensors'
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    summary = qualification(world).run()
    assert summary['status'] == 'partial' and summary['arms_failed'] == ['offline-grpo']
    assert 'offline-grpo' in summary['partial_label']


def test_logit_batching_gate_stops(world, monkeypatch):
    original = nc.NeuronCollector.collect

    def noisy(self, rows, **kw):
        out = original(self, rows, **kw)
        if len(rows) > 1:
            out['logits'] = out['logits'] + 0.01
        return out

    monkeypatch.setattr(nc.NeuronCollector, 'collect', noisy)
    with pytest.raises(q.QualificationFailure, match='ADR 0014'):
        qualification(world).run()


def test_fp64_fallback_chain_and_proxy_floor(world):
    floors = qualification(world, fail_fp64_on={'gpu'}).run()['fp64_floor_methods']
    assert set(floors.values()) == {'fp64-cpu-3-examples'}


def test_proxy_floor_when_every_fp64_attempt_fails(world):
    floors = qualification(world, fail_fp64_on={'gpu', 'cpu'}).run()['fp64_floor_methods']
    assert set(floors.values()) == {'proxy floor'}


def test_signoff_recorded_once(world):
    qualification(world).run()
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    record = q.signoff(run_dir, reviewer='r', decision='accept', notes='flags reviewed')
    assert record['qualification_sha256'] and isinstance(record['flags'], list)
    with pytest.raises(ValueError, match='already'):
        q.signoff(run_dir, reviewer='r', decision='reject')


# --- pure decision math ---------------------------------------------------------------------

def test_pack_respects_budget_and_is_deterministic():
    rows = [dict(id=f'r{i}', input_ids=[1] * (5 + i % 7)) for i in range(30)]
    a, b = q.pack(rows, 40), q.pack(list(reversed(rows)), 40)
    assert [[r['id'] for r in x] for x in a] == [[r['id'] for r in x] for x in b]
    assert all(len(x) * max(len(r['input_ids']) for r in x) <= 40 for x in a)
    assert sorted(r['id'] for x in a for r in x) == sorted(r['id'] for r in rows)
    with pytest.raises(ValueError, match='longer'):
        q.pack([dict(id='x', input_ids=[1] * 50)], 40)


def test_mask_problems_detect_each_corruption():
    tok = FakeTokenizer()
    good = row(1, 4, 'math', 'discovery')
    assert q.mask_problems(good, tok.token_text, 4) == []
    for corrupt in [dict(reply_start=[good['masks']['reply_start'][0] + 1]),
                    dict(response_content=good['masks']['response_content'][1:]),
                    dict(termination=[t + 1 for t in good['masks']['termination']])]:
        bad = json.loads(json.dumps(good))
        bad['masks'].update(corrupt)
        assert q.mask_problems(bad, tok.token_text, 4)


def test_normalized_discrepancy_and_flags():
    single = np.array([1.0, -1.0, 1.0, -1.0])
    assert q.normalized_discrepancy(single, single) == 0.0
    assert q.normalized_discrepancy(single, single + 1e-3) == pytest.approx(1e-3)
    plan = yaml.safe_load(REAL_PLAN.read_text())['review_flags']
    rec = lambda **kw: dict(dict(arm='a', batch='b0', example='e', padded=False, layer=5, group='response_content',
                                 stat='mean_z', value=0.0), **kw)
    assert q.review_flags([rec(layer=1, value=1e-5)], plan) == []
    assert q.review_flags([rec(layer=1, value=2e-4)], plan)[0]['flag'] == 'early_layer'
    assert q.review_flags([rec(layer=9, value=2e-4)], plan) == []  # only layers 0-2
    padding = [rec(example='u', value=0.0), rec(example='p1', padded=True, value=5e-6),
               rec(example='p2', padded=True, value=5e-6)]
    assert [f['flag'] for f in q.review_flags(padding, plan)] == ['padding']
    padding[0]['value'] = 1e-6
    assert q.review_flags(padding, plan) == []


def test_floor_zero_is_unresolved_never_spacing():
    d = np.ones(100)
    assert q.floor_from(d, d) == dict(p99=None, status='unresolved', median=0.0, max=0.0)
    assert q.floor_from(d, d + 1e-9)['status'] == 'measured'


def test_scope_rule_all_subset_omit():
    fits = lambda layers: dict(time_ratio=1.2, projected_bytes=10)
    small = lambda total: total <= 60
    assert q.choose_scope(fits, all_layers=list(range(28)), subset_layers=list(range(10, 18)), max_ratio=1.5,
                          fits_storage=small)['scope'] == 'all'
    subset_only = lambda layers: dict(time_ratio=2.0 if len(layers) == 28 else 1.4, projected_bytes=10)
    assert q.choose_scope(subset_only, all_layers=list(range(28)), subset_layers=list(range(10, 18)), max_ratio=1.5,
                          fits_storage=small)['scope'] == 'subset'
    too_big = lambda layers: dict(time_ratio=1.0, projected_bytes=99)
    result = q.choose_scope(too_big, all_layers=list(range(28)), subset_layers=list(range(10, 18)), max_ratio=1.5,
                            fits_storage=small)
    assert result['scope'] == 'omit' and len(result['tried']) == 2
    assert q.choose_scope(fits, all_layers=[0, 1], subset_layers=list(range(10, 18)), max_ratio=1.0,
                          fits_storage=small)['tried'][1]['available'] is False


def test_projected_bytes_matches_spec_illustration():
    # Spec: E=700, L=28, J=8960, A=7, G=3, two FP32 stats -> about 27.5 GiB of dense base payload.
    dense_only = q.projected_bytes(arms_trained=0, arms_total=7, examples=700, layers=28, width=8960, groups=3,
                                   union_tokens=0)
    assert dense_only / 2 ** 30 == pytest.approx(27.5, abs=0.1)


def test_summary_binds_inputs_runtime_and_runs_the_bf16_diagnostic(world):
    summary = qualification(world).run()
    assert summary['runtime_fingerprint']['device'] == 'fake-cpu' and summary['runtime_fingerprint']['torch']
    seq_sha = hashlib.sha256((world['root'] / 'seq/sequences.jsonl').read_bytes()).hexdigest()
    assert summary['inputs']['sequences_sha256'] == seq_sha and set(summary['inputs']['adapters']) == {'gold-sft',
                                                                                                         'offline-grpo'}
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    bf16 = json.loads((run_dir / 'stages/bf16_diagnostic/result.json').read_text())
    assert bf16['label'].startswith('DIAGNOSTIC ONLY') and bf16['arms']['gold-sft']['status'] == 'measured'
    assert set(bf16['arms']['gold-sft']['layers']) == {'0', '1'}


def test_sealed_stages_are_never_reused_on_a_different_runtime(world, monkeypatch):
    qualification(world).run(stop_after='masks')
    monkeypatch.setattr(FakeBackend, 'device_info', lambda self: dict(name='other-gpu', total_memory=10_000_000))
    with pytest.raises(q.QualificationFailure, match='different runtime'):
        qualification(world).run()


def test_qualification_ceiling_is_enforced_cumulatively(world):
    clock = dict(t=0.0)

    def now():
        clock['t'] += 500.0
        return clock['t']

    run = lambda: q.Qualification(world['plan'], world['config'], world['backend'], preflight=world['preflight'],  # noqa: E731
                                  now=now).run()
    with pytest.raises(q.QualificationFailure, match='time ceiling'):
        run()  # plan ceiling 1 h; every tick advances 500 s
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    assert json.loads((run_dir / 'meta/clock.json').read_text())['seconds'] > 3600
    clock['t'] = 0.0
    with pytest.raises(q.QualificationFailure, match='time ceiling'):
        run()  # a restart inherits the elapsed time instead of starting over


def test_phase_1b_out_of_memory_keeps_the_1a_measurement(world, monkeypatch):
    original = q.Qualification._time_run

    def oom_for_heads(self, arm, batches, *, heads=None):  # only the benchmark's 1B timing runs out of memory
        if heads:
            raise RuntimeError('CUDA out of memory (simulated, 1B hooks)')
        return original(self, arm, batches, heads=heads)

    monkeypatch.setattr(q.Qualification, '_time_run', oom_for_heads)
    qual = qualification(world)
    results = qual.run(stop_after='benchmark')
    bench = results['benchmark']
    assert bench['phase_1b']['scope'] == 'omit' and bench['projection']['hooks'] == '1A'
    assert bench['selected_budget'] in (200, 400) and any('oom_1B_all_layers' in p for p in bench['probes'])


def test_four_hour_ceiling_survives_an_interruption(world, monkeypatch):
    value = dict(world['plan_value'])
    value['operations'] = dict(value['operations'], qualification_max_gpu_hours=4)  # the frozen plan's value
    path = world['tmp'] / 'plan-4h.yaml'
    path.write_text(yaml.safe_dump(value))
    plan = plan_mod.load(path, '03b')
    assert plan['operations']['qualification_max_gpu_hours'] == 4
    clock = dict(t=0.0, step=1.0)

    def now():
        clock['t'] += clock['step']
        return clock['t']

    def make():
        return q.Qualification(plan, world['config'], world['backend'], preflight=world['preflight'], now=now)

    original = q.Qualification.stage_benchmark

    def killed(self, *a, **k):
        clock['t'] += 3 * 3600  # three hours pass inside the stage, then the worker dies
        raise RuntimeError('worker killed (simulated)')

    monkeypatch.setattr(q.Qualification, 'stage_benchmark', killed)
    with pytest.raises(RuntimeError, match='killed'):
        make().run()
    monkeypatch.setattr(q.Qualification, 'stage_benchmark', original)
    run_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    before = json.loads((run_dir / 'meta/clock.json').read_text())['seconds']
    assert before < 4 * 3600  # with the fake clock only checkpoints record time; on the Pod a 60 s heartbeat does too
    clock.update(t=0.0, step=1800.0)  # new process: its own clock restarts at zero
    with pytest.raises(q.QualificationFailure, match='time ceiling') as info:
        make().run()
    assert info.value.evidence['cap_seconds'] == 4 * 3600
    assert info.value.evidence['elapsed_seconds'] > 4 * 3600


def test_clock_heartbeat_records_time_between_checkpoints(world, monkeypatch):
    import threading
    started = threading.Event()
    release = threading.Event()

    def slow(self, *a, **k):
        started.set()
        release.wait(5)
        raise RuntimeError('worker killed (simulated)')

    monkeypatch.setattr(q.Qualification, 'stage_benchmark', slow)
    config = dict(world['config'], clock_interval_seconds=0.05)
    clock = dict(t=0.0)

    def now():
        clock['t'] += 100.0
        return clock['t']

    qual = q.Qualification(world['plan'], config, world['backend'], preflight=world['preflight'], now=now)
    runner = threading.Thread(target=lambda: pytest.raises(RuntimeError, qual.run))
    runner.start()
    started.wait(30)
    path = Path(config['output_root']) / q.RUN_SUBDIR / 'q1/meta/clock.json'
    first = json.loads(path.read_text())['seconds']
    import time as _time
    _time.sleep(0.3)
    assert json.loads(path.read_text())['seconds'] > first  # advanced while no checkpoint ran
    release.set()
    runner.join(10)


def test_production_stages_are_not_reused_across_a_code_change(world):
    first = dict(world['config'], code_head_override='a' * 40)
    q.Qualification(world['plan'], first, world['backend'], preflight=world['preflight']).run(stop_after='masks')
    changed = dict(world['config'], code_head_override='b' * 40)
    with pytest.raises(q.QualificationFailure, match='identity or integrity differs'):
        q.Qualification(world['plan'], changed, world['backend'], preflight=world['preflight']).run()
