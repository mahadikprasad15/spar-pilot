import pytest

from pilot_eval.offline_grpo_engine import pool_window


def _rows():
    return [dict(id=f'q{i}', prompt='prompt', gold='2', input_ids=[1, 2], draws=[
        dict(group_id=f'q{i}', draw_id=f'q{i}:draw:{j}', draw_index=j,
             text='#### 2', token_ids=[3], generated_token_ids=[3, 4],
             completion_mask=[1, 1], stop_reason='eos', capped=False, reward_v3=1)
        for j in range(8)]) for i in range(8)]


def test_pool_window_preserves_order_and_checks_rewards():
    rows = _rows()
    prompts, outputs = pool_window(rows)
    assert prompts == [[1, 2]] * 8
    assert len(outputs) == 64
    assert outputs[0]['generated_token_ids'] == [3, 4]
    rows[0]['draws'][0]['reward_v3'] = 0
    with pytest.raises(ValueError, match='reward'):
        pool_window(rows)


def test_pool_window_rejects_mask_and_draw_identity():
    rows = _rows()
    rows[0]['draws'][0]['completion_mask'] = [1, 0]
    with pytest.raises(ValueError, match='mask'):
        pool_window(rows)
    rows = _rows()
    rows[0]['draws'][1]['draw_id'] = rows[0]['draws'][0]['draw_id']
    with pytest.raises(ValueError, match='draw'):
        pool_window(rows)


def test_checkpoint_zero_hashes_are_checked_before_loading(tmp_path):
    from pilot_eval.offline_grpo_engine import verify_initial_adapter
    from pilot_eval.training import file_hash
    weights = tmp_path / 'adapter_model.safetensors'
    config = tmp_path / 'adapter_config.json'
    weights.write_bytes(b'weights')
    config.write_text('{}')
    verify_initial_adapter(tmp_path, file_hash(weights), file_hash(config))
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint-0'):
        verify_initial_adapter(tmp_path, '0' * 64, file_hash(config))


def test_run_requires_prepared_training_plan(tmp_path):
    from pilot_eval.offline_grpo_training import run
    with pytest.raises((FileNotFoundError, ValueError)):
        run(tmp_path / 'missing.json', tmp_path)


def test_interrupted_run_replays_only_after_checkpoint(tmp_path, monkeypatch):
    import json
    from pilot_eval import offline_grpo_training as training
    groups = [[f'q{s}-{i}' for i in range(8)] for s in range(64)]
    windows = [[{'id': q} for q in group] for group in groups]
    config = {'optimizer_groups': groups, 'settings': {}, 'weights_sha256': 'x',
              'adapter_config_sha256': 'x', 'runtime': {'fixture': True},
              'monitor_policy': {k: {'threshold': 2, 'action': 'flag'} for k in
                  ('cap_fraction', 'dead_group_fraction', 'length_ratio_change_8_steps')},
              'length_change_definition': 'step-lag-8'}
    directory = tmp_path / 'run'; directory.mkdir()
    (directory / 'config.json').write_text(json.dumps(config))
    monkeypatch.setattr(training, '_load', lambda *args: (config, directory, {}, {}, windows, tmp_path))

    class Engine:
        def __init__(self, fail): self.step = 0; self.fail = fail
        def runtime(self): return {'fixture': True}
        def check_integrity(self): return {'base_unchanged': True}
        def close(self): pass
        def train_window(self, rows, step):
            if self.fail and step == 10: raise RuntimeError('interrupt')
            assert step == self.step + 1
            self.step = step
            draws = [dict(draw_id=f"{r['id']}:draw:{d}", step=step, reward=d%2,
                          strict={'correct': bool(d%2)}, flexible={'correct': bool(d%2)},
                          capped=False, token_count=1, advantage=1.) for r in rows for d in range(8)]
            evidence = {'logratio': self.diagnostic_sink('logratio', step, {'fixture': True})}
            if step == 1:
                evidence['window1_push'] = self.diagnostic_sink('window1-push', step, {'passed': True, 'fixture': True})
            return draws, {'loss': 0.0, 'diagnostic_evidence': evidence}
        def save_checkpoint(self, path, step):
            path.mkdir(parents=True)
            for f in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                      'scheduler.pt', 'rng_state.pth', 'prior-gradient.pt'):
                (path/f).write_text('fixture')
            (path/'trainer_state.json').write_text(json.dumps({'global_step': step}))
        def restore_checkpoint(self, path): self.step = json.loads((path/'trainer_state.json').read_text())['global_step']
    class Dependencies:
        def __init__(self, fail): self.fail = fail
        def load_training(self, *args): return Engine(self.fail)

    with pytest.raises(RuntimeError, match='interrupt'):
        training.run(tmp_path/'unused', tmp_path, dependencies=Dependencies(True))
    assert json.loads((directory/'meta/status.json').read_text())['state'] == 'failed'
    result = training.run(tmp_path/'unused', tmp_path, dependencies=Dependencies(False))
    assert result['optimizer_steps'] == 64
    records = [json.loads(x) for x in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len(records) == len({r['draw_id'] for r in records}) == 4096
    assert sorted(p.name for p in (directory/'checkpoints').glob('checkpoint-*')) == [f'checkpoint-{i}' for i in (0, 16, 32, 64, 8)]
    assert training.run(tmp_path/'unused', tmp_path, dependencies=Dependencies(True)) == result
    diagnostic = next((directory/'attempts').glob('*/diagnostics/001-logratio.json'))
    saved = diagnostic.read_bytes(); diagnostic.write_text('{}')
    with pytest.raises(ValueError, match='diagnostic evidence changed'):
        training.run(tmp_path/'unused', tmp_path, dependencies=Dependencies(True))
    diagnostic.write_bytes(saved)
    (directory/'results/responses.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='changed'):
        training.run(tmp_path/'unused', tmp_path, dependencies=Dependencies(True))


def test_pinned_trl_online_and_offline_update_parity(tmp_path):
    import copy
    from test_grpo_algorithm import tiny, torch_stack
    torch = torch_stack()
    from pilot_eval.grpo_algorithm import GRPOWindow
    from pilot_eval.grpo_train_engine import GRPOTrainingEngine
    from pilot_eval.offline_grpo_engine import OfflineGRPOTrainingEngine
    model, tokenizer = tiny()
    def engine(cls, model, path):
        obj = cls.__new__(cls)
        obj.torch = torch; obj.model = model; obj.plan = {'optimizer': {'max_grad_norm': 1.0}}
        obj.settings = {}; obj.global_step = 0; obj.previous_gradient = None; obj._sync = lambda: None
        obj.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
        obj.scheduler = torch.optim.lr_scheduler.LambdaLR(obj.optimizer, lambda _: 1.0)
        obj.window = GRPOWindow(model, tokenizer, output_dir=path, microbatch_groups=1, completion_limit=4)
        return obj
    offline = engine(OfflineGRPOTrainingEngine, copy.deepcopy(model), tmp_path/'offline')
    online = engine(GRPOTrainingEngine, model, tmp_path/'online')
    rows = _rows()
    for i, row in enumerate(rows):
        row['input_ids'] = [0]; row['prompt_tokens'] = 1
        for j, draw in enumerate(row['draws']):
            draw.update(text='#### 2' if i and j % 2 else '#### 3', token_ids=[j%2] * (1+j%2),
                        generated_token_ids=[j%2] * (1+j%2), completion_mask=[1] * (1+j%2),
                        stop_reason='cap', capped=True, reward_v3=0)
            if i and j % 2:
                draw.update(stop_reason='eos', capped=False, reward_v3=1)
    online._sample = lambda _: [{k:d[k] for k in ('text','token_ids','generated_token_ids','stop_reason')} for r in rows for d in r['draws']]
    a, _ = online.train_window(rows, 1)
    b, _ = offline.train_window(rows, 1)
    assert [r['reward'] for r in a] == [r['reward'] for r in b]
    assert online.window.advantages == offline.window.advantages
    torch.testing.assert_close(online.previous_gradient, offline.previous_gradient, atol=1e-6, rtol=1e-4)
    for (name, p), (other_name, q) in zip(online.model.named_parameters(), offline.model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-4)
