"""Ticket 05: implementation, policy and numerical checks (CPU)."""
import copy
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from pilot_eval import attribution_instrument as ai  # noqa: E402
from pilot_eval import c_attribution as ca  # noqa: E402
from pilot_eval import c_attribution_checks as cc  # noqa: E402
from test_attribution_instrument import PROJ, instrument, tiny  # noqa: E402

FIXTURES = Path(__file__).parent / 'fixtures' / 'c_attribution_checks'
PINNED = Path(os.environ.get('PILOT_ARTIFACTS', '/Users/prasadmahadik/spar/pilot1/artifacts')) / \
    'inputs/models/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
SETTINGS = {'evaluation_batch_size': 8, 'generation_groups': 4, 'completion_limit': 1024, 'top_p': 1.0, 'top_k': 0}


def _rows(q=2, n=4, base=7):
    rows, rewards = [], {}
    for i in range(q):
        rewards[f'q{i}'] = [int((i + d) % 2 == 0) for d in range(n)]
        for d in range(n):
            size = base + (d + 2 * i) % 4
            ids = [(5 * i + 3 * d + 7 * k) % 29 + 1 for k in range(size)]
            rows.append({'id': f'q{i}:{d}', 'question_id': f'q{i}', 'draw_index': d,
                         'input_ids': ids, 'masks': {'seq_targets': list(range(4, size)), 'length': size}})
    return rows, rewards


def _weights(rows, rewards):
    w = ca.loo_weights(rewards)
    return np.array([w['weights'][r['question_id']][r['draw_index']] for r in rows])


# --- logits processors ---------------------------------------------------------------------------

def _tokenizer():
    transformers = pytest.importorskip('transformers')
    for source in (PINNED, 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'):
        try:
            tok = transformers.AutoTokenizer.from_pretrained(str(source), local_files_only=True)
            tok.padding_side = 'left'  # as backend.load_hf_backend sets it
            return tok
        except Exception:
            continue
    pytest.skip('no local Qwen2.5 tokenizer')


def _qwen_vocab_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                        attn_implementation='eager')).eval()


def test_fixture_is_the_pinned_generation_config():
    assert cc.json.loads((FIXTURES / 'generation_config.json').read_text())['repetition_penalty'] == 1.1
    if (PINNED / 'generation_config.json').exists():
        assert (PINNED / 'generation_config.json').read_bytes() == (FIXTURES / 'generation_config.json').read_bytes()


def test_explicit_sampled_arguments_resolve_to_identity_processors_on_both_bases():
    tok = _tokenizer()
    model = _qwen_vocab_model()
    result = cc.processor_check(model, tok, settings=SETTINGS, generation_config_path=FIXTURES / 'generation_config.json')
    assert result['passed'], result
    pinned = result['bases'][0]
    # The pinned file's 0.7 / 20 / 0.8 / 1.1 defaults are overridden by _generate's explicit arguments.
    assert pinned['resolved_generation_config'] == {'do_sample': True, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0,
                                                    'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0,
                                                    'num_beams': 1}
    assert pinned['arguments']['repetition_penalty'] == 1.0


@pytest.mark.parametrize('override', [{'repetition_penalty': 1.1}, {'temperature': 0.7}, {'top_k': 20}])
def test_processor_check_fails_when_a_modifying_argument_is_injected(override):
    tok = _tokenizer()
    model = _qwen_vocab_model()
    result = cc.resolve_processors(model, tok, settings=SETTINGS, base='pinned-file',
                                   generation_config_path=FIXTURES / 'generation_config.json', overrides=override)
    assert not result['passed'] and result['processors']


# --- surrogate finite differences ----------------------------------------------------------------

def _fp64(model):
    reference = copy.deepcopy(model).double()
    return reference


def test_surrogate_finite_differences_agree_on_the_tiny_model():
    model = tiny()
    rows, rewards = _rows()
    inst = instrument(model, metric_order=('seqlogp',))
    result = cc.surrogate_finite_difference(inst, _fp64(model), rows, _weights(rows, rewards), epsilon=1e-3,
                                            adapters=5, neurons=3, median_tol=0.05, max_tol=0.20)
    assert result['passed'], result['items']
    assert len(result['items']) == 8 and result['median_rel_err'] < 1e-3
    assert result['floors']['fp32_ulp_M_over_eps'] > result['floors']['fp64_ulp_M_over_eps']


@pytest.mark.parametrize('mutation', ['sign-flip', 'permuted'])
def test_surrogate_finite_difference_fails_for_mutated_weights(mutation):
    model = tiny()
    rows, rewards = _rows()
    inst = instrument(model, metric_order=('seqlogp',))
    w = _weights(rows, rewards)
    reference = _fp64(model)
    original = inst.attribute

    def mutated(batch, **kw):
        out = original(batch, **kw)
        attr = out['attr']['seqlogp']
        neuron = out['neuron']['seqlogp']
        if mutation == 'sign-flip':
            out['attr']['seqlogp'] = -attr
            out['neuron']['seqlogp'] = {l: -v for l, v in neuron.items()}
        else:  # weights applied to the wrong rows: attributions shifted by one row
            out['attr']['seqlogp'] = np.roll(attr, 1, axis=0)
            out['neuron']['seqlogp'] = {l: np.roll(v, 1, axis=0) for l, v in neuron.items()}
        return out

    inst.attribute = mutated
    result = cc.surrogate_finite_difference(inst, reference, rows, w, epsilon=1e-3, median_tol=0.05, max_tol=0.20)
    assert not result['passed']


def test_surrogate_reference_must_be_fp64():
    model = tiny()
    rows, rewards = _rows()
    with pytest.raises(cc.CheckFailure):
        cc.surrogate_finite_difference(instrument(model, metric_order=('seqlogp',)), model, rows,
                                       _weights(rows, rewards), epsilon=1e-3)


# --- cached against teacher-forced ---------------------------------------------------------------

def test_cached_and_forced_agree_on_the_tiny_model():
    model = tiny()
    rows, _ = _rows(q=2, n=3)
    result = cc.cached_vs_forced(model, rows, [4] * len(rows), pad_token_id=0)
    assert result['per_token_abs']['max'] < 1e-5, result
    assert result['per_row_abs_M']['max'] < 1e-4


def test_cached_and_forced_agree_on_a_capped_length_sequence():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1100,
                                        attn_implementation='eager'))
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=2, lora_dropout=0, target_modules=PROJ)).eval()
    prompt = [3, 5, 7, 9, 11]
    capped = prompt + [(k * 7) % 30 + 1 for k in range(1024)]
    short = prompt[:3] + [4, 8, 31]
    rows = [{'id': 'capped', 'input_ids': capped, 'masks': {'seq_targets': list(range(5, len(capped)))}},
            {'id': 'short', 'input_ids': short, 'masks': {'seq_targets': list(range(3, len(short)))}}]
    result = cc.cached_vs_forced(model, rows, [5, 3], pad_token_id=0)
    assert result['generated_lengths'] == [1024, 3]
    assert result['per_token_abs']['max'] < 1e-4, result


# --- batch invariance, no-op, replay, numerics -------------------------------------------------------

def test_seqlogp_batch_invariance_on_the_tiny_model():
    model = tiny()
    rows, _ = _rows()
    inst = instrument(model, metric_order=('seqlogp',))
    result = cc.batch_invariance(inst, rows, [30, 60], tolerance=1e-4)
    assert result['passed'], result
    assert result['max_abs'] >= 0


def test_noop_passes_for_an_empty_context_and_fails_when_a_weight_is_left_modified():
    model = tiny()
    rows, _ = _rows(q=1, n=2)
    assert cc.noop_check(model, rows, cc.empty_context)['passed']

    @contextmanager
    def leaky():
        module = model.get_base_model().model.layers[0].mlp.down_proj
        with torch.no_grad():
            module.lora_A['default'].weight[0, 0] = 0.0  # an edit that is never restored
        yield

    result = cc.noop_check(model, rows, leaky)
    assert not result['passed'] and not result['restored']


def test_replay_report_finds_the_first_divergence():
    out = cc.replay_report({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2]}, {'a': [1, 2, 3], 'b': [1, 9, 3], 'c': [1, 2, 5]})
    assert out['rows']['a'] == {'identical': True, 'first_divergence': None}
    assert out['rows']['b']['first_divergence'] == 1 and out['rows']['c']['first_divergence'] == 2
    assert out['identical_rows'] == 1 and out['review'] and 'passed' not in out


def test_numerics_record_keeps_historical_tolerances(tmp_path):
    record = cc.numerics_record(batch_invariance={'max_abs': 1e-7, 'max_fraction': 1e-6, 'budgets': [30]},
                                gross_net_ratios=[0.1, 0.5, np.nan, 1.0],
                                identity={'passed': True})
    assert record['historical_tolerances'] == cc.PRODUCTION_1
    assert record['gross_net_ratio']['median'] == pytest.approx(0.5)
    cc.write_numerics(tmp_path / 'numerics.json', record)
    assert cc.json.loads((tmp_path / 'numerics.json').read_text())['batch_invariance']['max_fraction'] == 1e-6
