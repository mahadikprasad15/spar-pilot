"""Independent qualification-layout mapping checks."""

def test_each_distinct_merged_mask_gets_its_own_unhalved_diagnostic():
    from pilot_eval.stage0_conditional import diagnostic_layouts
    variants={
        'V6-sft-unhalved':dict(mode='unhalved',source_arm='sft',adapter_disable_modules=['a','b','c'],merged_modules={x:{'path':x} for x in ['a','b','c']}),
        'V6-sft-full':dict(mode='full',source_arm='sft',adapter_disable_modules=['a','b','c']),
        'V6-sft-matched':dict(mode='matched',source_arm='sft',adapter_disable_modules=['a','b']),
        'V6c-sft-matched':dict(mode='control',source_arm='sft',adapter_disable_modules=['a','b']),
        'V6-sft-ratio':dict(mode='ratio',source_arm='sft',adapter_disable_modules=['b']),
    }
    for row in variants.values():row['status']='ready-for-qualification'
    result=diagnostic_layouts(dict(variants=variants),['sft'])
    assert list(result)==['sft','sft-mask-1','sft-mask-2']
    assert [row['adapter_disable_modules'] for row in result.values()]==[['a','b','c'],['a','b'],['b']]
    assert all(row['mode']=='unhalved' for row in result.values())
    assert set(result['sft-mask-1']['merged_modules'])=={'a','b'}
