"""Supplemental measurement contract: independent small distributions and masks."""
import numpy as np
import pytest


def test_next_target_views_and_full_vocabulary_kl_have_known_answers():
    from pilot_eval.token_measurement_math import prediction_positions, full_vocabulary_kl, summarize_kl
    rows = [dict(id='one', input_ids=[9, 4, 5, 6], attention_mask=[1]*4,
                 masks={'question':[False,True,False,False],
                        'solution':[False,False,True,True], 'user':[False]*4}),
            dict(id='two', input_ids=[9, 7], attention_mask=[1,1],
                 masks={'question':[False]*2,'solution':[False]*2,'user':[False,True]})]
    positions = prediction_positions(rows)
    assert positions == [dict(example=0,context=0,target=1,view='question',token_id=4),
                         dict(example=0,context=1,target=2,view='solution',token_id=5),
                         dict(example=0,context=2,target=3,view='solution',token_id=6),
                         dict(example=1,context=0,target=1,view='user',token_id=7)]
    p, q = np.log([[.8,.2],[.5,.5]]), np.log([[.5,.5],[.8,.2]])
    values = full_vocabulary_kl(p,q)
    np.testing.assert_allclose(values, [.19274475702175753,.22314355131420976],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(full_vocabulary_kl(p,p),[0.,0.])
    # Different lengths make the two weighting definitions observably distinct.
    summary=summarize_kl([dict(example=0,view='solution')]*2+[dict(example=1,view='solution')],
                         np.array([0.,0.,.6]),['one','two'])
    cell=next(r for r in summary if r['view']=='solution')
    assert cell['token_mean']==pytest.approx(.2)
    assert cell['example_mean']==pytest.approx(.3)
    assert cell['tokens']==3 and cell['examples']==2


def tiny_instrument(tmp_path):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from pilot_eval.activation_engine import ActivationEngine
    from pilot_eval.activation_prepare import PROJECTIONS
    from pilot_eval.sft_backend import frozen_weight_hash
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,
        intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=32,pad_token_id=0,attn_implementation='eager')),
        LoraConfig(r=1,lora_alpha=1,lora_dropout=0,target_modules=PROJECTIONS))
    base=frozen_weight_hash(model)
    model.save_pretrained(tmp_path/'zero',save_embedding_layers=False)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.fill_(.03)
    model.save_pretrained(tmp_path/'trained',save_embedding_layers=False)
    engine=ActivationEngine(model=model,expected_layers=2,expected_base_hash=base)
    rows=[dict(id='one',input_ids=[1,2,3,4],attention_mask=[1]*4,
               masks={'question':[False,True,False,False],'solution':[False,False,True,True],'user':[False]*4}),
          dict(id='two',input_ids=[1,5,6],attention_mask=[1]*3,
               masks={'question':[False]*3,'solution':[False]*3,'user':[False,True,True]})]
    return torch,engine,rows


def test_real_supplement_reconstructs_branches_and_matches_independent_full_logits(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    from pilot_eval.token_measurement_math import full_vocabulary_kl
    torch,instrument,rows=tiny_instrument(tmp_path)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1_000_000)
    # A normal full model call is allowed only in this tiny independent oracle.
    tokens=torch.tensor([[1,2,3,4],[1,5,6,0]])
    attention=torch.tensor([[1,1,1,1],[1,1,1,0]])
    with torch.no_grad():
        with instrument.model.disable_adapter():
            q=instrument.model(input_ids=tokens,attention_mask=attention).logits.numpy()
        p=instrument.model(input_ids=tokens,attention_mask=attention).logits.numpy()
    seen={};handles=[]
    for layer,projection,module in instrument.modules:
        def observe(m,args,output,key=(layer,projection)):
            seen[key]=(args[0].detach().numpy().copy(),output.detach().numpy().copy(),
                       m.weight.detach().numpy().copy())
        handles.append(module.lora_A['default'].register_forward_hook(observe))
    try:
        with engine.integrity_scope():
            result=engine.measure(tmp_path/'trained',rows,step=64)
        positions=result['metadata']['predictions']
        expected=full_vocabulary_kl(np.array([p[r['example'],r['context']] for r in positions]),
                                    np.array([q[r['example'],r['context']] for r in positions]))
        from pilot_eval.activation_profile import AGREEMENT
        # Full-head and chunked-head FP32 matmuls follow the existing policy.
        np.testing.assert_allclose(result['arrays']['kl'],expected,**AGREEMENT)
        assert result['arrays']['coefficients'].shape==(5,14)
        for mi,module in enumerate(result['metadata']['modules']):
            x,c,a=seen[module['layer'],module['projection']]
            for ci,position in enumerate(result['metadata']['coefficient_positions']):
                wanted=np.dot(a.astype(np.float64)[0],x[position['example'],position['position']].astype(np.float64))
                assert result['arrays']['coefficients'][ci,mi]==pytest.approx(wanted,abs=1e-6,rel=1e-5)
        assert result['validation']['rank1_passed'] and result['validation']['reference_invariant']
        assert result['validation']['max_logit_contexts']<=2
        with engine.integrity_scope():
            zero=engine.measure(tmp_path/'zero',rows,step=0)
        assert np.count_nonzero(zero['arrays']['kl'])==0
        assert np.count_nonzero(zero['arrays']['coefficients'])>0
        assert zero['validation']['zero_contribution']
    finally:
        for handle in handles:handle.remove()
        engine.close()


def test_supplement_rejects_workspace_before_output_head_allocation(tmp_path):
    from pilot_eval.token_measurement_engine import TokenMeasurementEngine
    _,instrument,rows=tiny_instrument(tmp_path)
    engine=TokenMeasurementEngine(instrument,context_chunk=2,workspace_bytes=1)
    try:
        with pytest.raises(ValueError,match='workspace'):
            engine.measure(tmp_path/'trained',rows,step=64)
    finally:engine.close()
