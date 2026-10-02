import pytest

@pytest.mark.parametrize('text,gold', [
    ('2 + 2/2 = 3 bolts\n#### 3', '3'),
    ('4*20=<<4*20=80>>80\n2*80=<<2*80=160>>160\n#### 260', '260'),
    ('Reasoning uses 9 and 3.\n#### 3.0', '3'),
    ('Final answer: It takes 3 bolts.', '3'),
])
def test_v3_accepts_strict_gold_style_and_preserves_prose(text,gold):
    from pilot_eval.scoring import score_gsm8k_flexible_v3
    result=score_gsm8k_flexible_v3(text,'#### '+gold)
    assert result['correct'] is True
    assert result['scorer_version']=='gsm8k-flexible-v3'

@pytest.mark.parametrize('text', ['#### 12\n#### 72', 'Final answer: 12\n#### 72',
                                 'Final answer: 36 hours in 4 weeks.', '#### 1/0'])
def test_v3_rejects_conflicts_and_malformed_answers(text):
    from pilot_eval.scoring import score_gsm8k_flexible_v3
    assert score_gsm8k_flexible_v3(text,'#### 72')['status']=='invalid'

def test_v3_extraction_has_no_gold_access_and_capped_remains_invalid():
    from pilot_eval.scoring import extract_gsm8k_flexible_v3,score_gsm8k_flexible_v3
    text='Reasoning: 2+1=3\n#### 3'
    right=score_gsm8k_flexible_v3(text,'#### 3')
    wrong=score_gsm8k_flexible_v3(text,'#### 4')
    assert {k:v for k,v in right.items() if k!='correct'}==extract_gsm8k_flexible_v3(text)
    assert {k:v for k,v in wrong.items() if k!='correct'}==extract_gsm8k_flexible_v3(text)
    assert wrong['correct'] is False
    assert score_gsm8k_flexible_v3(text,'#### 3',True)['status']=='invalid'
