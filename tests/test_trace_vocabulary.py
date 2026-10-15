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
