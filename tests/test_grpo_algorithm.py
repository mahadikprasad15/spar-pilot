"""Real offline model checks at the public training-window boundary."""
import math
import pytest


def test_group_advantages_use_sample_std_and_dead_groups_are_zero():
    from pilot_eval.grpo_algorithm import group_advantages
    advantages = group_advantages([0, 0, 0, 0, 1, 1, 1, 1] + [1] * 8)
    expected = .5 / (math.sqrt(2 / 7) + .0001)
    assert advantages[:4] == pytest.approx([-expected] * 4)
    assert advantages[4:8] == pytest.approx([expected] * 4)
    assert advantages[8:] == [0] * 8
    assert sum(advantages[:8]) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        group_advantages([1] * 7)


def torch_stack():
    try:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from trl import GRPOTrainer
    except (ImportError, OSError) as exc:
        pytest.skip(f'optional real-model dependencies unavailable: {exc}')
    return torch


def tiny():
    torch = torch_stack()
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=2, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0, tie_word_embeddings=False,
        pad_token_id=None, bos_token_id=0, eos_token_id=None, attn_implementation='eager'))
    model = get_peft_model(model, LoraConfig(r=1, lora_alpha=1, lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'))
    token = Tokenizer(WordLevel({'bad': 0, 'good': 1}, unk_token='bad'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, pad_token='bad', unk_token='bad')
    tokenizer.padding_side = 'left'
    return model, tokenizer


def test_real_trl_loss_matches_independent_token_window_and_microbatch_gradients(tmp_path):
    torch = torch_stack()
    torch.set_num_threads(1)
    from pilot_eval.grpo_algorithm import GRPOWindow
    model, tokenizer = tiny()
    prompts = [[0]] * 8
    completions = [[d % 2] * (1 + d % 3) for d in range(64)]
    rewards = [float(d % 2) for d in range(64)]
    window = GRPOWindow(model, tokenizer, output_dir=tmp_path, microbatch_groups=1)
    loss = window.backward(prompts, completions, rewards, capped=[False] * 64)
    actual = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any(torch.count_nonzero(g) for g in actual.values()), 'gradient comparison needs a nonzero witness'
    model.zero_grad(set_to_none=True)
    # Independent direct per-sequence differentiable reference, no window helpers.
    terms = []
    scale = .5 / (math.sqrt(2 / 7) + .0001)
    for i, tokens in enumerate(completions):
        ids = torch.tensor([[0] + tokens])
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1]
        logp = logits.log_softmax(-1).gather(-1, torch.tensor(tokens).reshape(1, -1, 1)).squeeze(-1)
        advantage = scale if rewards[i] else -scale
        terms.append(-(logp - logp.detach()).exp().sum() * advantage)
    reference = sum(terms) / sum(map(len, completions))
    reference.backward()
    assert loss == pytest.approx(float(reference.detach()), abs=1e-6)
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(actual[n], p.grad, atol=1e-6, rtol=1e-4)
    model.zero_grad(set_to_none=True)
    whole = GRPOWindow(model, tokenizer, output_dir=tmp_path / 'whole', microbatch_groups=8)
    whole.backward(prompts, completions, rewards, capped=[False] * 64)
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(actual[n], p.grad, atol=1e-6, rtol=1e-4)
    assert whole.resolved_args['generation_batch_size'] == 64
    assert whole.resolved_args['num_generations'] == 8
    assert whole.resolved_args['loss_type'] == 'dapo'


def test_dead_groups_have_zero_policy_gradient_but_adam_momentum_can_move(tmp_path):
    torch = torch_stack()
    from pilot_eval.grpo_algorithm import GRPOWindow, gradient_consistency
    model, tokenizer = tiny()
    window = GRPOWindow(model, tokenizer, output_dir=tmp_path, microbatch_groups=8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0)
    completions = [[d % 2] for d in range(64)]
    window.backward([[0]] * 8, completions, [d % 2 for d in range(64)], capped=[False] * 64)
    previous = window.flat_gradient()
    assert previous.norm() > 0
    assert gradient_consistency(None, previous)['reason'] == 'first_step'
    assert gradient_consistency(previous, previous)['cosine'] == pytest.approx(1)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before = {n:p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
    window.backward([[0]] * 8, completions, [1] * 64, capped=[True] * 64)
    assert window.advantages == [0] * 64 and window.dead_group_fraction == 1
    current = window.flat_gradient()
    assert torch.count_nonzero(current) == 0
    assert gradient_consistency(previous, current)['reason'] == 'zero_gradient'
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n,p in model.named_parameters() if p.requires_grad)
