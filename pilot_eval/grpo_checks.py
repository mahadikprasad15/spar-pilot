"""Independent offline calculations used by the public GRPO control workflow."""
import math


def validate_algorithm(task, directory):
    import torch
    from pilot_eval.grpo_controls import tiny_model
    from pilot_eval.grpo_algorithm import GRPOWindow, group_advantages, gradient_consistency

    model, tokenizer = tiny_model(task, 42)
    completions = [[i % 2] * (1 + i % 3) for i in range(64)]
    rewards = [i % 2 for i in range(64)]
    window = GRPOWindow(model, tokenizer, output_dir=directory / 'micro', microbatch_groups=1)
    loss = window.backward([[0]] * 8, completions, rewards, capped=[False] * 64)
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    if not any(torch.count_nonzero(g) for g in gradients.values()):
        raise ValueError('independent gradient comparison requires a nonzero witness')
    model.zero_grad(set_to_none=True)
    scale = .5 / (math.sqrt(2 / 7) + .0001)
    terms = []
    for i, tokens in enumerate(completions):
        ids = torch.tensor([[0] + tokens])
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1]
        logp = logits.log_softmax(-1).gather(-1, torch.tensor(tokens).reshape(1, -1, 1)).squeeze(-1)
        terms.append(-(logp - logp.detach()).exp().sum() * (scale if rewards[i] else -scale))
    reference = sum(terms) / sum(map(len, completions))
    reference.backward()
    if abs(loss - float(reference.detach())) > 1e-6:
        raise ValueError('independent whole-window loss mismatch')
    independent_error = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(gradients[n], p.grad, atol=1e-6, rtol=1e-4)
            independent_error = max(independent_error, float((gradients[n] - p.grad).abs().max()))
    model.zero_grad(set_to_none=True)
    whole = GRPOWindow(model, tokenizer, output_dir=directory / 'whole', microbatch_groups=8)
    whole.backward([[0]] * 8, completions, rewards, capped=[False] * 64)
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(gradients[n], p.grad, atol=1e-6, rtol=1e-4)
    previous = whole.flat_gradient()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    whole.backward([[0]] * 8, completions, [1] * 64, capped=[True] * 64)
    current = whole.flat_gradient()
    if torch.count_nonzero(current) or whole.dead_group_fraction != 1:
        raise ValueError('capped/dead groups must give zero policy gradient')
    if gradient_consistency(previous, current)['reason'] != 'zero_gradient':
        raise ValueError('zero-gradient consistency must be undefined')
    optimizer.step()
    if not any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError('Adam momentum witness did not move parameters')
    expected = group_advantages([0] * 4 + [1] * 4 + [1] * 8)
    if max(abs(x + scale) for x in expected[:4]) > 1e-12 or expected[8:] != [0] * 8:
        raise ValueError('group standardization mismatch')
    return dict(passed=True, independent_gradient_max_absolute_error=independent_error,
                gradient_atol=1e-6, gradient_rtol=1e-4,
                checks=['independent_loss_and_gradients', 'unequal_lengths_padding',
                        'microbatch_equivalence', 'group_sample_std', 'cap_and_dead_groups',
                        'adam_momentum', 'undefined_zero_gradient_cosine'],
                resolved_trainer_args=whole.resolved_args)
