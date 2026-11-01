"""Production endpoint loop with the real HFTrainingEngine/PEFT on a tiny 28-layer Qwen.

T16 qualifies SFTTrainer; the sampled arms train through sampling_sft_training_backend.train.
This runs that exact loop (PEFT adapter load/save, initialization identity, explicit slots,
denominator, clipping, AdamW, checkpoint seals) on CPU. Needs the pinned trainer packages
(run with the Pod's training venv); skipped elsewhere. Not a GPU or real-model qualification.
"""
import json
import math

import pytest


def test_production_loop_matches_independent_loss_with_real_peft(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('peft');pytest.importorskip('transformers')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from pilot_eval.sft import optimizer_settings,adapter_settings
    from pilot_eval.sft_backend import HFTrainingEngine,frozen_weight_hash
    from pilot_eval.sampling_sft_readiness_backend import rows as tiny_rows,oracle_loss
    from pilot_eval import sampling_sft_training_backend as backend
    torch.manual_seed(42)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=4,intermediate_size=8,num_hidden_layers=28,
        num_attention_heads=1,num_key_value_heads=1,max_position_embeddings=32,eos_token_id=3,pad_token_id=0,
        attention_dropout=0.0,attn_implementation='eager'))
    gold=dict(seed=42,adapter=adapter_settings(),optimizer=optimizer_settings())
    rows=[]
    for step in range(1,65): # Varying 3-8 covered slots per original step, as in the sampled arms.
        for i,row in enumerate(tiny_rows(3+step%6)):rows.append({**row,'id':f's{step}:{i}'})
    by_step={};[by_step.setdefault(r['id'].split(':')[0],[]).append(r) for r in rows]
    batches=[dict(step=s,slots=[dict(source_id=r['id'],missing=False) for r in by_step[f's{s}']]+
        [dict(source_id=f'missing{s}',missing=True)]*(8-len(by_step[f's{s}'])),nonmissing_count=len(by_step[f's{s}'])) for s in range(1,65)]
    engine=HFTrainingEngine(gold,rows,model=model,tokenizer=None)
    initial=tmp_path/'checkpoint-0';engine.model.save_pretrained(str(initial),safe_serialization=True)
    first=by_step['s1'];expected,_=oracle_loss(engine.model,first)
    folder=tmp_path/'run';(folder/'results').mkdir(parents=True)
    config=dict(k=0,continuation=None,checkpoint_steps=[0,8,16,32,64],capture_recovery=True,
        expected_base_sha256=frozen_weight_hash(engine.model))
    runtime=dict(execution_kind='cpu-fixture',model_context_limit=32)
    result=backend.train(engine,rows,batches,gold,config,folder,tmp_path,initial,runtime)
    logs=[json.loads(l) for l in (folder/'logs/steps.jsonl').read_text().splitlines()]
    assert result['successful_steps']==64 and result['base_sha256_before']==result['base_sha256_after']
    assert abs(logs[0]['loss']-expected)<=1e-6+1e-6*abs(expected)
    assert [l['example_ids'] for l in logs]==[[s['source_id'] for s in b['slots'] if not s['missing']] for b in batches]
    assert all(math.isfinite(l['grad_norm']) and l['learning_rate']==gold['optimizer']['learning_rate'] for l in logs)
    assert json.loads((folder/'results/initialization.json').read_text())['b_exact_zero'] is True
    for step in [0,8,16,32,64]:
        assert (folder/f'checkpoints/checkpoint-{step}/complete.json').is_file()
    assert (folder/'checkpoints/checkpoint-0/adapter_model.safetensors').read_bytes()==(initial/'adapter_model.safetensors').read_bytes()
    assert (folder/'checkpoints/checkpoint-64/adapter_model.safetensors').read_bytes()!=(initial/'adapter_model.safetensors').read_bytes()
