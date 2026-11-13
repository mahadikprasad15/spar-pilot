"""Tiny Torch oracle; run directly in an environment with Torch and NumPy."""
import copy
import torch
from pilot_eval.grpo_algorithm import GRPOWindow
from pilot_eval.grpo_train_engine import GRPOTrainingEngine
from pilot_eval.offline_grpo_engine import OfflineGRPOTrainingEngine

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    @property
    def device(self):
        return self.lora_weight.device

class Loss:
    def compute_loss(self, model, batch):
        ids = batch['completion_ids'].float()
        mask = batch['completion_mask']
        scores = model.lora_weight[0] * ids + model.lora_weight[1] * ids.square()
        return -(scores * mask * batch['advantages'][:, None]).sum() / batch['num_items_in_batch']

def engine(cls):
    e = cls.__new__(cls)
    e.torch = torch; e.model = Model(); e.plan = {'optimizer': {'max_grad_norm': 1.0}}
    e.settings = {}; e.global_step = 0; e.previous_gradient = None; e._sync = lambda: None
    e.optimizer = torch.optim.AdamW(e.model.parameters(), lr=1e-4, weight_decay=0)
    e.scheduler = torch.optim.lr_scheduler.LambdaLR(e.optimizer, lambda _: 1.)
    w = GRPOWindow.__new__(GRPOWindow)
    w.model = e.model; w.microbatch = 16; w.trainer = Loss(); w.pad_id = 0
    w.parameter_names = ['lora_weight']; e.window = w
    return e

def run():
    rows = []
    for i in range(8):
        draws = []
        for j in range(8):
            cap = i == 1 and j == 0
            correct = i != 0 and j % 2 == 0
            tok = [1 + j % 3] * (1 + j % 3)
            draws.append(dict(group_id=f'q{i}', draw_id=f'q{i}:draw:{j}', draw_index=j,
                text='#### 2' if correct else '#### 3', token_ids=tok,
                generated_token_ids=tok + ([] if cap else [4]),
                completion_mask=[1] * (len(tok) + (0 if cap else 1)),
                stop_reason='cap' if cap else 'eos', capped=cap,
                reward_v3=int(correct and not cap)))
        rows.append(dict(id=f'q{i}', prompt='p', gold='2', input_ids=[2, 3], draws=draws))
    online, offline = engine(GRPOTrainingEngine), engine(OfflineGRPOTrainingEngine)
    online._sample = lambda _: [{k:d[k] for k in ('text','token_ids','generated_token_ids','stop_reason')} for r in rows for d in r['draws']]
    online_rows = [dict(r, prompt_tokens=len(r['input_ids'])) for r in rows]
    a, _ = online.train_window(online_rows, 1)
    b, _ = offline.train_window(copy.deepcopy(rows), 1)
    assert [r['reward'] for r in a] == [r['reward'] for r in b]
    assert online.window.advantages == offline.window.advantages
    assert torch.equal(online.previous_gradient, offline.previous_gradient)
    assert torch.equal(online.model.lora_weight, offline.model.lora_weight)
    assert online.previous_gradient.norm() > 0
    denominator = sum(len(d['generated_token_ids']) for r in rows for d in r['draws'])
    assert denominator > 64
    # Independent analytic gradient for the mocked token loss; criterion declared
    # before comparison: absolute tolerance 1e-6, no relative tolerance.
    expected = [0.0, 0.0]
    for row in rows:
        rewards = [d['reward_v3'] for d in row['draws']]
        mean = sum(rewards) / 8
        sample_sd = (sum((r - mean) ** 2 for r in rewards) / 7) ** .5
        for draw, reward in zip(row['draws'], rewards):
            advantage = (reward - mean) / (sample_sd + 1e-4)
            expected[0] -= advantage * sum(draw['generated_token_ids']) / denominator
            expected[1] -= advantage * sum(t*t for t in draw['generated_token_ids']) / denominator
    assert torch.allclose(online.previous_gradient, torch.tensor(expected, dtype=torch.float64), atol=1e-6, rtol=0)

if __name__ == '__main__':
    run()
