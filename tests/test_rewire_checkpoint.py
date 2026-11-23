"""Ticket 08: checkpoints and restart. Spec tests 11 and 21, on CPU with a tiny Qwen2 and no download.

The training loop is ticket 07. ``fixture_train`` below is a stand-in with the same contract (it calls
``sink(step, model)`` at step 0 and after every optimizer step, and hardens the masks after the freeze step),
so restart and determinism are tested through the real checkpoint path without the real loop.
"""
import copy
import json
import math
import random
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('safetensors')

from pilot_eval import rewire_checkpoint as ck  # noqa: E402
from pilot_eval import rewire_module as rw  # noqa: E402
from pilot_eval import sparse_rewiring_plan as sp  # noqa: E402
from pilot_eval.neuron_audit_collector import set_ieee_precision  # noqa: E402
from pilot_eval.training import file_hash  # noqa: E402

LAYERS, HIDDEN, INTERMEDIATE, VOCAB = 2, 8, 12, 32
STEPS, FREEZE = 12, 10          # stand-ins for 1,024 and 960
POINTS = [0, 1, 2, 4, 8, 10, 12]
RUNTIME, COMMIT = 'r' * 64, 'c0ffee'
DRAFT = Path(__file__).resolve().parents[1] / '.scratch/sparse-rewiring-gsm8k/launch-record-draft.yaml'


def tiny_plan(rank=1, **changes):
    plan = dict(
        schema=sp.PLAN_SCHEMA, arm=f'sparse-gold-sft-r{rank}', run_id=f'r{rank}', rank=rank,
        settings=dict(model=dict(layers=LAYERS, hidden=HIDDEN, intermediate=INTERMEDIATE),
                      storage=dict(checkpoint_contents=['read_factors', 'log_alpha', 'seal'],
                                   optimizer_state='none', interruption='restart-from-step-0',
                                   gate_snapshot_dtype='float16'),
                      seeds=dict(data_order_base=42, read_init_base=42),
                      training=dict(steps=STEPS, freeze_after_step=FREEZE)),
        checkpoints=list(POINTS),
        checkpoint_bytes=sp.checkpoint_bytes(rank, layers=LAYERS, hidden=HIDDEN, intermediate=INTERMEDIATE))
    plan.update(changes)
    plan['plan_sha256'] = sp.plan_sha256(plan)
    return plan


def model_factory(plan):
    def make():
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
            attn_implementation='eager', use_cache=False))
        rw.install(model, rank=plan['rank'], seed=plan['settings']['seeds']['read_init_base'])
        return model
    return make


def fixture_train(run_dir, *, crash_at=None, stop_after=None, skip=None):
    """Stand-in for ticket 07's loop: AdamW (foreach off) on the rewire parameters, CE plus the L0 term, masks
    hardened to +/-20 after the freeze step, one JSON line per step."""
    def train(model, plan, sink):
        set_ieee_precision()
        torch.use_deterministic_algorithms(True)
        model.train()
        params = [p for _, p in rw.trainable_parameters(model)]
        optimizer = torch.optim.AdamW(params, lr=3e-2, foreach=False)
        generator = torch.Generator().manual_seed(plan['settings']['seeds']['data_order_base'])
        data = torch.randint(1, VOCAB, (STEPS, 2, 9), generator=generator)
        log = Path(run_dir) / 'logs' / 'steps.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        sink(0, model)
        for step in range(1, STEPS + 1):
            if step == crash_at:
                raise RuntimeError(f'simulated crash at step {step}')
            if step == FREEZE + 1:
                with torch.no_grad():
                    for m in rw.modules(model):
                        on = m.rewire_log_alpha > -math.log(11.0)
                        m.rewire_log_alpha.copy_(torch.where(on, 20.0, -20.0))
                        m.rewire_log_alpha.requires_grad_(False)
            out = model(input_ids=data[step - 1], labels=data[step - 1], use_cache=False)
            l0 = sum(torch.sigmoid(m.rewire_log_alpha - (2 / 3) * math.log(0.1 / 1.1)).sum()
                     for m in rw.modules(model))
            loss = out.loss + 6e-4 * l0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with log.open('a') as stream:
                stream.write(json.dumps(dict(step=step, loss=out.loss.item(), l0=l0.item())) + '\n')
            if step == skip:
                continue
            sink(step, model)
            if step == stop_after:
                return
    return train


def run(run_dir, plan, **kw):
    return ck.run_arm(run_dir, plan, model_factory=model_factory(plan), train=fixture_train(run_dir, **kw),
                      runtime_sha256=RUNTIME, code_commit=COMMIT, files=['logs/steps.jsonl'])


def hashes(run_dir):
    return {s: file_hash(ck.checkpoint_dir(run_dir, s) / ck.STATE_FILE) for s in POINTS}


def never_train(model, plan, sink):
    raise AssertionError('a completed, verified arm must not be retrained')


# --- spec test 11: restart, not resume ------------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 0])
def test_two_runs_from_step_0_are_bit_identical(tmp_path, rank):
    plan = tiny_plan(rank)
    a, b = tmp_path / 'a' / 'r', tmp_path / 'b' / 'r'
    assert run(a, plan)['action'] == 'trained'
    assert run(b, plan)['action'] == 'trained'
    assert hashes(a) == hashes(b)
    assert (a / 'logs/steps.jsonl').read_bytes() == (b / 'logs/steps.jsonl').read_bytes()
    assert (a / ck.ARM_SEAL).read_bytes() == (b / ck.ARM_SEAL).read_bytes()
    _, at8 = ck.load_tensors(ck.checkpoint_dir(a, 8), plan)
    _, at_freeze = ck.load_tensors(ck.checkpoint_dir(a, FREEZE), plan)
    _, final = ck.load_tensors(ck.checkpoint_dir(a, STEPS), plan)
    alpha = [n for n in final if n.endswith('log_alpha')]
    # The run moved, 960's stand-in holds continuous logα, and the hardening shows from the next checkpoint on.
    assert any(torch.count_nonzero(at8[n]).item() for n in at8 if 'rewire_B_' in n or 'rewire_W_' in n)
    assert not all(set(at_freeze[n].abs().unique().tolist()) <= {20.0} for n in alpha)
    assert all(set(final[n].abs().unique().tolist()) <= {20.0} for n in alpha)


def test_crash_discards_the_partial_arm_and_restarts_from_step_0(tmp_path):
    plan = tiny_plan(1)
    reference = tmp_path / 'ref' / 'r'
    run(reference, plan)
    arm = tmp_path / 'pod' / 'r'
    with pytest.raises(RuntimeError, match='simulated crash'):
        run(arm, plan, crash_at=6)
    assert sorted(p.name for p in (arm / 'checkpoints').iterdir()) == [
        'checkpoint-0', 'checkpoint-1', 'checkpoint-2', 'checkpoint-4']
    assert not (arm / ck.ARM_SEAL).exists()

    result = run(arm, plan)
    assert result['action'] == 'restarted'
    assert hashes(arm) == hashes(reference)
    assert (arm / 'logs/steps.jsonl').read_bytes() == (reference / 'logs/steps.jsonl').read_bytes()
    evidence = Path(result['evidence'])
    assert evidence.name.startswith('r.interrupted-') and evidence.parent == arm.parent
    assert not list(evidence.rglob('*.safetensors'))      # payloads dropped (D24 disk budget)
    assert sorted(p.name for p in (evidence / 'checkpoints').iterdir()) == [
        'checkpoint-0', 'checkpoint-1', 'checkpoint-2', 'checkpoint-4']
    discarded = json.loads((evidence / 'discarded.json').read_text())
    assert [r['path'] for r in discarded['removed_payloads']] == [
        f'checkpoints/checkpoint-{s}/{ck.STATE_FILE}' for s in (0, 1, 2, 4)]
    events = [json.loads(line) for line in (arm / 'logs/events.jsonl').read_text().splitlines()]
    assert [(e['kind'], e['discarded_checkpoints']) for e in events] == [('restart-from-step-0', [0, 1, 2, 4])]
    ck.verify_arm(arm, plan)


def test_a_half_written_checkpoint_is_an_interrupted_attempt(tmp_path):
    plan = tiny_plan(1)
    arm = tmp_path / 'r'
    with pytest.raises(RuntimeError):
        run(arm, plan, crash_at=3)
    (arm / 'checkpoints' / 'checkpoint-4.tmp').mkdir()       # killed inside save_checkpoint
    (arm / 'checkpoints' / 'checkpoint-4.tmp' / ck.STATE_FILE).write_bytes(b'partial')
    assert run(arm, plan)['action'] == 'restarted'
    assert not (arm / 'checkpoints' / 'checkpoint-4.tmp').exists()
    ck.verify_arm(arm, plan)


def test_a_completed_arm_restored_from_the_volume_is_verified_and_kept(tmp_path):
    plan = tiny_plan(0)
    trained = tmp_path / 'pod' / 'r'
    run(trained, plan)
    restored = tmp_path / 'fresh-pod' / 'r'
    shutil.copytree(trained, restored)
    before = hashes(restored)
    result = ck.run_arm(restored, plan, model_factory=model_factory(plan), train=never_train,
                        runtime_sha256='another-runtime', code_commit='another-commit')
    assert result['action'] == 'kept'
    assert result['seal']['runtime_sha256'] == RUNTIME     # the arm keeps the runtime it was trained under
    assert hashes(restored) == before


@pytest.mark.parametrize('damage', ['payload', 'missing-checkpoint', 'log', 'seal-plan', 'extra-file'])
def test_a_completed_arm_that_fails_verification_is_refused_not_retrained(tmp_path, damage):
    plan = tiny_plan(1)
    arm = tmp_path / 'r'
    run(arm, plan)
    if damage == 'payload':
        path = ck.checkpoint_dir(arm, 8) / ck.STATE_FILE
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))
    elif damage == 'missing-checkpoint':
        shutil.rmtree(ck.checkpoint_dir(arm, FREEZE))
    elif damage == 'log':
        with (arm / 'logs/steps.jsonl').open('a') as stream:
            stream.write('{}\n')
    elif damage == 'seal-plan':
        seal = json.loads((arm / ck.ARM_SEAL).read_text())
        seal['checkpoints'].pop(str(STEPS))
        (arm / ck.ARM_SEAL).write_text(json.dumps(seal))
    else:
        (ck.checkpoint_dir(arm, 8) / 'optimizer.pt').write_bytes(b'state')
    with pytest.raises(ck.CheckpointRefused):
        ck.run_arm(arm, plan, model_factory=model_factory(plan), train=never_train,
                   runtime_sha256=RUNTIME, code_commit=COMMIT)
    assert (arm / ck.ARM_SEAL).exists()          # nothing moved or deleted


def test_a_completed_arm_under_another_plan_is_refused(tmp_path):
    arm = tmp_path / 'r'
    run(arm, tiny_plan(1))
    other = tiny_plan(1, checkpoints=[0, 1, 2, 4, 8, 10, 11, 12])
    with pytest.raises(ck.CheckpointRefused, match='another plan'):
        ck.prepare_arm(arm, other)


def test_an_arm_that_skips_or_stops_early_is_not_sealed(tmp_path):
    plan = tiny_plan(1)
    with pytest.raises(ck.CheckpointRefused, match='reported step 9, expected 8'):
        run(tmp_path / 'skip', plan, skip=8)
    with pytest.raises(ck.CheckpointRefused, match=r'without planned checkpoints \[10, 12\]'):
        run(tmp_path / 'short', plan, stop_after=9)
    assert not (tmp_path / 'skip' / ck.ARM_SEAL).exists()
    assert not (tmp_path / 'short' / ck.ARM_SEAL).exists()


# --- spec test 21: storage policy (D24) -----------------------------------------------------------

@pytest.mark.parametrize('rank', [1, 0])
def test_a_checkpoint_holds_only_factors_log_alpha_and_its_seal(tmp_path, rank):
    plan = tiny_plan(rank)
    arm = tmp_path / 'r'
    run(arm, plan)
    for step in POINTS:
        folder = ck.checkpoint_dir(arm, step)
        assert sorted(p.name for p in folder.iterdir()) == [ck.SEAL_FILE, ck.STATE_FILE]
        seal = ck.verify_checkpoint(folder, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
        assert seal['recoverable'] is False and seal['config_sha256'] == plan['plan_sha256']
        assert seal['step'] == step and seal['code_commit'] == COMMIT and seal['runtime_sha256'] == RUNTIME
        _, tensors = ck.load_tensors(folder, plan)
        letters = {'W'} if rank == 0 else {'A', 'B'}
        assert {n.rsplit('.', 1)[1] for n in tensors} == {
            f'rewire_{x}_{p}' for x in letters for p in ('gate', 'up')} | {'rewire_log_alpha'}
        assert all(t.dtype == torch.float32 for t in tensors.values())
        assert sum(t.numel() * 4 for t in tensors.values()) == plan['checkpoint_bytes'] == seal['payload_bytes']
        assert (folder / ck.STATE_FILE).stat().st_size - plan['checkpoint_bytes'] < 16_384   # header only
    (ck.checkpoint_dir(arm, 4) / 'optimizer.pt').write_bytes(b'')
    with pytest.raises(ck.CheckpointRefused, match='D24'):
        ck.verify_checkpoint(ck.checkpoint_dir(arm, 4), plan)


def _real_plans():
    import yaml
    record = yaml.safe_load(DRAFT.read_text())
    for key, text in record['decisions'].items():
        record['decisions'][key] = text.replace('[proposed]', '[user test]')
    record.update(code_commit='c0ffee', pod_id='pod1')
    record['inputs'].update(artifact_root='/w/inputs', bundle='/w/b.json', model_dir='/w/model', work_dir='/w')
    ids = [f'gsm8k:train:{i}' for i in sorted(random.Random(7).sample(range(7473), 512))]
    order = list(range(512))
    random.Random(42).shuffle(order)
    gold = [ids[i] for i in order]
    sp.validate(record, code_commit='c0ffee')
    return record, {arm: sp.build_plan(record, arm, ids, gold, inputs_sha256={}) for arm in sp.ARMS}


def test_planned_disk_total_is_computed_from_shapes_and_checked_by_the_record():
    record, plans = _real_plans()
    totals = {arm: ck.storage(plan) for arm, plan in plans.items()}
    r0, r1 = totals['sparse-gold-sft-r0'], totals['sparse-gold-sft-r1']
    assert r0['checkpoints'] == r1['checkpoints'] == 11
    # Rank 0: 28 layers x 2 parts x 8960 x 1536 FP32 reads + 28 x 8960 logα, 11 times: about 34 GB (D24).
    assert r0['checkpoint_bytes'] == (28 * 2 * 8960 * 1536 + 28 * 8960) * 4
    assert 33.5e9 < r0['total_bytes'] < 34.5e9
    assert r1['checkpoint_bytes'] == (28 * 2 * (1536 + 8960) + 28 * 8960) * 4
    # The record's disk block must cover what these checkpoints write (plus gate snapshots).
    assert sp.planned_disk_bytes(record) >= r0['total_bytes'] + r1['total_bytes']
    small = copy.deepcopy(record)
    small['disk']['container_total_gb'] = (r0['total_bytes'] + r1['total_bytes']) / 1e9
    from pilot_eval.attribution_chain import LaunchRefused
    with pytest.raises(LaunchRefused, match='D24'):
        sp.validate(small, code_commit='c0ffee')
    # The real names match what install() gives a real-shaped model, without allocating one.
    names = ck.expected_tensors(plans['sparse-gold-sft-r1'])
    assert len(names) == 28 * 5 and 'model.layers.27.mlp.rewire_B_up' in names


def test_storage_refuses_a_plan_whose_size_or_policy_is_not_d24():
    plan = tiny_plan(1)
    bad = dict(plan, checkpoint_bytes=plan['checkpoint_bytes'] + 4)
    bad['plan_sha256'] = sp.plan_sha256(bad)
    with pytest.raises(ck.CheckpointRefused, match='checkpoint_bytes'):
        ck.storage(bad)
    settings = copy.deepcopy(plan['settings'])
    settings['storage']['optimizer_state'] = 'adam'
    with pytest.raises(ck.CheckpointRefused, match='D24'):
        ck.storage(dict(plan, settings=settings))


# --- writer and loader ----------------------------------------------------------------------------

def test_checkpoint_0_must_be_the_untuned_model(tmp_path):
    plan = tiny_plan(1)
    model = model_factory(plan)()
    with torch.no_grad():
        rw.modules(model)[0].rewire_B_up[3, 0] = 1e-3
    with pytest.raises(ck.CheckpointRefused, match='untuned'):
        ck.save_checkpoint(tmp_path, 0, model, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
    assert not ck.checkpoint_dir(tmp_path, 0).exists()


def test_writer_refuses_unplanned_steps_overwrites_and_a_model_of_another_rank(tmp_path):
    plan = tiny_plan(1)
    model = model_factory(plan)()
    with pytest.raises(ck.CheckpointRefused, match='not a planned'):
        ck.save_checkpoint(tmp_path, 3, model, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
    ck.save_checkpoint(tmp_path, 0, model, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
    with pytest.raises(ck.CheckpointRefused, match='already exists'):
        ck.save_checkpoint(tmp_path, 0, model, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
    with pytest.raises(ck.CheckpointRefused, match='differ from the plan'):
        ck.save_checkpoint(tmp_path / 'x', 0, model_factory(tiny_plan(0))(), plan, runtime_sha256=RUNTIME,
                           code_commit=COMMIT)


def test_loader_restores_every_rewire_parameter_and_refuses_mismatches(tmp_path):
    plan = tiny_plan(1)
    arm = tmp_path / 'r'
    run(arm, plan)
    folder = ck.checkpoint_dir(arm, 8)
    model = model_factory(plan)()
    base = {n: p.clone() for n, p in model.named_parameters() if not rw.is_rewire_parameter(n)}
    ck.load_into(model, folder, plan, runtime_sha256=RUNTIME, code_commit=COMMIT)
    _, saved = ck.load_tensors(folder, plan)
    assert all(torch.equal(p, saved[n]) for n, p in rw.trainable_parameters(model))
    assert all(torch.equal(p, base[n]) for n, p in model.named_parameters() if n in base)
    with pytest.raises(ck.CheckpointRefused, match='runtime'):
        ck.load_into(model, folder, plan, runtime_sha256='x' * 64)
    with pytest.raises(ck.CheckpointRefused, match='code commit'):
        ck.load_into(model, folder, plan, code_commit='deadbeef')
    with pytest.raises(ck.CheckpointRefused, match='identity'):
        ck.load_into(model, folder, tiny_plan(1, run_id='other'))
    with pytest.raises(ck.CheckpointRefused, match='own hash'):
        ck.load_into(model, folder, dict(plan, rank=0))
