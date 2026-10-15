"""Independent token and numerical cases at the public vocabulary boundary."""
import pytest


def test_normalized_token_counts_merge_variants_and_exclude_added_and_partial_bytes():
    from pilot_eval.trace_vocabulary import TokenInventory
    inventory = TokenInventory({0:' total',1:'total',2:' Total',3:'####',4:'\ufffd',5:' ',6:'<special>',7:'Ａ'}, added_ids=[6])
    assert inventory.count([0,1,2,3,4,5,6,7]) == {'total':3,'####':1,'a':1}
    assert inventory.normalize(0) == 'total'
    assert inventory.normalize(4) is None
    assert inventory.normalize(6) is None


def test_log_odds_matches_independent_literals_and_filters_rare_nonpositive_strings():
    from pilot_eval.trace_vocabulary import build_reference, log_odds
    # Independently evaluated values from the stated equations; no implementation-derived fixture.
    rare = log_odds(2,0,61000,500000,1000)
    common = log_odds(300,150,61000,500000,1000)
    assert rare == pytest.approx(0.5023558891671656, abs=1e-9)
    assert common == pytest.approx(27.907134536543953, abs=1e-9)
    assert common > rare
    assert 3/1 > 301/151  # Add-one ratio ranks the rare example higher; normalization is common.
    ref = build_reference({'math':300,'rare':2,'####':50,'background':60648},
                          {'math':150,'####':10,'background':499840})
    assert 'math' in ref['strings'] and '####' in ref['strings']
    assert 'rare' not in ref['strings'] and 'background' not in ref['strings']
    assert ref['format_number'] == ['####']
    assert ref['word'] == ['math']
    assert ref['short_reference'] is True


def test_reference_freeze_is_reused_and_detects_edited_bytes_and_changed_parents(tmp_path):
    from pilot_eval.trace_vocabulary import freeze_references, load_references
    parent = tmp_path/'counts.json';parent.write_text('source counts')
    references = {'ref_questions':{'strings':['total']},'ref_gold':{'strings':['####']},'ref_grpo':{'strings':['answer']}}
    path = freeze_references(tmp_path/'references', references,
        parameters={'alpha0':1000}, input_files={'counts':parent})
    assert load_references(path)['references'] == references
    assert freeze_references(tmp_path/'references', references,
        parameters={'alpha0':1000}, input_files={'counts':parent}) == path
    parent.write_text('changed counts')
    with pytest.raises(ValueError, match='parent'):
        load_references(path)
    parent.write_text('source counts')
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='freeze.*hash'):
        load_references(path)


def test_background_preparation_is_bounded_exact_and_resumes_sealed_scan_chunks(tmp_path):
    from pilot_eval.trace_vocabulary import prepare_background
    docs=[{'text':v} for v in ['a'*80,'a'*80,'b'*80,'c'*80,'x'*10,'d'*80]]
    class Tokenizer:
        def __init__(self): self.seen=[]
        def encode(self,text,add_special_tokens=False):
            assert add_special_tokens is False
            self.seen.append(text)
            return [ord(c) for c in text]
    def interrupted():
        yield from docs[:3]
        raise RuntimeError('fixture disconnect')
    tok=Tokenizer()
    kwargs=dict(measurement_indices=[2],scan_limit=6,token_target=100,seed=0,chunk_size=2,
                tokenizer_identity={'fixture':'character'})
    with pytest.raises(RuntimeError,match='disconnect'):
        prepare_background(tmp_path/'background',interrupted(),tok,**kwargs)
    resumed=Tokenizer()
    result=prepare_background(tmp_path/'background',iter(docs),resumed,**kwargs)
    assert result['selected_indices']==[0,5]
    assert result['token_count']==100
    assert [r['retained_tokens'] for r in result['documents']]==[80,20]
    assert resumed.seen==['b'*80,'c'*80,'x'*10,'d'*80]
    assert result['documents'][1]['token_ids']==[100]*20
    assert prepare_background(tmp_path/'background',None,None,**kwargs)==result
    (tmp_path/'background/chunks/000000.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        prepare_background(tmp_path/'background',None,None,**kwargs)


def test_native_inventory_uses_real_tokenizer_decoding_without_model_imports(tmp_path):
    import json
    from tokenizers import Tokenizer,models,pre_tokenizers,AddedToken
    from pilot_eval.trace_vocabulary import TokenInventory
    engine=Tokenizer(models.WordLevel({'[UNK]':0,'total':1,'Ａ':2},unk_token='[UNK]'))
    engine.pre_tokenizer=pre_tokenizers.Whitespace()
    engine.add_special_tokens([AddedToken('[UNK]',special=True)])
    p=tmp_path/'tokenizer.json';engine.save(str(p))
    inventory=TokenInventory.from_file(p)
    assert inventory.count(inventory.encode('total total Ａ'))=={'total':2,'a':1}
    assert inventory.normalize(0) is None


def test_human_reference_review_binds_freeze_and_top30_before_production_scoring(tmp_path):
    from pilot_eval.trace_vocabulary import freeze_references, load_references, approve_references
    path=freeze_references(tmp_path/'refs',{'ref_questions':{},'ref_gold':{},'ref_grpo':{}},parameters={},input_files={})
    (path.parent/'top30.md').write_text('Reference preview fixture.')
    with pytest.raises(ValueError,match='review'):
        load_references(path,require_review=True)
    approve_references(path,reviewer='Prasad',notes='Read the preview; fixture references only.')
    assert load_references(path,require_review=True)['protocol']=='pilot5-overlap-v1.1'
    (path.parent/'top30.md').write_text('Changed preview.')
    with pytest.raises(ValueError,match='review'):
        load_references(path,require_review=True)


def test_background_scan_checks_replayed_measurement_identity(tmp_path):
    from pilot_eval.trace_vocabulary import prepare_background
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return list(range(len(text)))
    with pytest.raises(ValueError,match='measurement.*identity'):
        prepare_background(tmp_path/'background',iter([{'id':'doc','text':'a'*80}]),Tokenizer(),
            measurement_indices=[0],measurement_sources=[{'source_index':0,'source_id':'doc','full_document_sha256':'wrong'}],
            scan_limit=1,token_target=1,chunk_size=1,seed=0,tokenizer_identity={'fixture':True})
