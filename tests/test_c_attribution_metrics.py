import math

import pytest
import torch

from pilot_eval import attribution_metrics as am


def test_seqlogp_sums_generated_tokens_including_final_eos_and_ignores_padding():
    # Three-token vocabulary. Each prediction row is already a log-probability distribution.
    probs = torch.tensor([
        [[0.2, 0.3, 0.5], [0.5, 0.25, 0.25], [0.1, 0.7, 0.2], [0.6, 0.3, 0.1]],
        [[0.4, 0.1, 0.5], [0.2, 0.6, 0.2], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05]],
    ])
    logits = probs.log().requires_grad_(True)
    ids = torch.tensor([[0, 2, 1, 2], [1, 2, 0, 0]])
    masks = [dict(length=4, seq_targets=[1, 2, 3]), dict(length=2, seq_targets=[1])]
    values = am.seqlogp(logits, ids, masks)
    assert values.dtype == torch.float64
    assert values.tolist() == pytest.approx([math.log(0.5) + math.log(0.25) + math.log(0.2), math.log(0.5)], abs=1e-6)
    values.sum().backward()
    assert logits.grad[0, :3].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() == 0
    assert logits.grad[1, 1:].abs().sum() == 0


def test_seqlogp_rejects_empty_duplicate_and_out_of_range_targets():
    logits = torch.zeros((1, 3, 2))
    ids = torch.zeros((1, 3), dtype=torch.long)
    for targets in ([], [0], [1, 1], [3]):
        with pytest.raises(ValueError):
            am.seqlogp(logits, ids, [dict(seq_targets=targets)])


def test_seqlogp_capped_row_includes_all_1024_tokens():
    logits = torch.zeros((1, 1025, 2))
    ids = torch.zeros((1, 1025), dtype=torch.long)
    value = am.seqlogp(logits, ids, [dict(seq_targets=list(range(1, 1025)))])
    assert value.item() == pytest.approx(-1024 * math.log(2), abs=1e-4)
