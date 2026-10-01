import json

import pytest


def test_real_cpu_trainer_masks_accumulates_and_preserves_frozen_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    pytest.importorskip('peft', exc_type=ImportError)
    pytest.importorskip('trl', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from pilot_eval.sft_backend import HFTrainingEngine
    from pilot_eval.training import verified_checkpoints
    from test_sft_prepare import TrainingData, source_plan
    from pilot_eval.sft import prepare_sft, load_sft

    torch.set_num_threads(1)
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'tiny', dependencies=TrainingData())
    config, rows, _ = load_sft(path, tmp_path)
    # A tiny real Qwen architecture, built from config, never downloaded.
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=4,
        intermediate_size=8, num_hidden_layers=28, num_attention_heads=1,
        num_key_value_heads=1, max_position_embeddings=32, eos_token_id=3, pad_token_id=0))
    base_config = model.config
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    for row in rows:
        row.update(input_ids=[1, 2, 3], labels=[-100, 2, 3], prompt_tokens=1)
    config['max_length'] = 3
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'[PAD]': 0, 'question': 1, 'answer': 2, '[EOS]': 3, '[UNK]': 4}, unk_token='[UNK]')),
        eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]')
    engine = HFTrainingEngine(config, rows, model=model, tokenizer=tokenizer)
    directory = tmp_path / 'tiny-real-run'
    directory.mkdir()
    evidence = engine.preflight(directory)
    assert evidence['target_count'] == 196
    assert evidence['zero_write']
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 9:
            raise RuntimeError('simulated optimizer interruption')
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupted_step)
    with pytest.raises(RuntimeError, match='optimizer interruption'):
        engine.train(directory, None)
    assert sorted(verified_checkpoints(directory)) == [0, 8]
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    engine.close()
    fresh = Qwen2ForCausalLM(base_config)
    fresh.load_state_dict(initial_weights)
    engine = HFTrainingEngine(config, rows, model=fresh, tokenizer=tokenizer)
    result = engine.train(directory, directory / 'checkpoints/checkpoint-8')
    assert result['successful_steps'] == 64
    assert result['example_exposures'] == 512
    assert result['base_sha256_before'] == result['base_sha256_after']
    assert sorted(verified_checkpoints(directory)) == [0, 8, 16, 32, 64]
    logs = [json.loads(line) for line in (directory / 'logs/steps.jsonl').read_text().splitlines()]
    assert len(logs) == 64
    assert len({item for log in logs for item in log['example_ids']}) == 512
    assert all(log['learning_rate'] == 1e-4 for log in logs)
