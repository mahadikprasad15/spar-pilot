import json
import tarfile

import pytest

from pilot_eval import sampling_stage0_cut_agreement as cut_agreement
from pilot_eval.sampling_stage0_cuts import IntegrityFailure, collect, variant_catalogue


def items(n=150):
    return [dict(id=f'q{i}', prompt=f'p{i}', gold='#### 1') for i in range(n)]


def sources(arms=('a', 'b')):
    variants = variant_catalogue(arms)
    return dict(variants=variants, variant_order=list(variants), decoding=dict(eos_token_id=2, pad_token_id=0, max_new_tokens=8))


class Engine:
    def __init__(self, numeric_error=None, diverge=(), integrity_break=None):
        self.inputs = {}; self.calls = []; self.numeric_error = numeric_error or {}; self.diverge = set(diverge)
        self.integrity_break = integrity_break
        self.tokenizer = type('T', (), {'eos_token_id': 2, 'encode': lambda s, t, add_special_tokens=False: [5]})()
    def prepare_inputs(self, rows):
        self.inputs.update({r['id']: [1] for r in rows})
        return [dict(id=r['id'], input_ids=[1], attention_mask=[1]) for r in rows]
    def check_integrity(self, full=False): return dict(base_unchanged=True, adapters_unchanged=True)
    def generate(self, rows, variant, seed):
        self.calls.append((variant, len(rows)))
        if variant == self.integrity_break: raise IntegrityFailure('adapter payload changed after switching')
        token = 9 if len(rows) > 1 and variant in self.diverge else 1
        return [dict(text='#### 1', token_ids=[token], generated_token_ids=[token, 2], stop_reason='eos') for _ in rows]
    def compare(self, rows, variant, tokens):
        error = self.numeric_error.get(variant, 1e-5)
        return dict(max_absolute_error=error, examples=[dict(id=r['id'], errors=[error]) for r in rows])
    def divergence(self, rows, variant, index, reference, responses): return dict(near_tie=False, first_divergence=0)


def run_agreement(tmp_path, engine):
    src = sources(); cohort = items()
    config = cut_agreement.configuration(src, cohort)
    return config, cut_agreement.run(config, cohort, tmp_path, {'device': 'NVIDIA A100', 'session_id': 's'}, engine)


def test_original_protocol_scope_numeric_for_all_and_replay_for_two_output_cuts(tmp_path):
    engine = Engine()
    config, result = run_agreement(tmp_path, engine)
    assert config['replay_variants'] == ['a-output', 'b-output'] and config['sizes'] == [16, 6]
    assert sum(1 for v, n in engine.calls if n == 1) == 32 # fresh singleton references for the two replayed cuts
    assert set(result['batch_sizes'].values()) == {16} and result['batch_gate'] == 1e-3
    assert cut_agreement.verify(config, tmp_path)['batch_sizes'] == result['batch_sizes']


def test_numeric_failure_is_per_variant_and_replay_failure_covers_its_arm(tmp_path):
    engine = Engine(numeric_error={'a-input': 2e-3}, diverge={'b-output'})
    _, result = run_agreement(tmp_path, engine)
    sizes = result['batch_sizes']
    assert sizes['a-input'] == 1 and sizes['a-output'] == 16 and sizes['a-original'] == 16
    assert all(sizes[v] == 1 for v in sizes if v.startswith('b-')) # an arm's replay failure is not extrapolated
    assert result['replays']['b']['passed'] is False and result['replays']['a']['passed'] is True


def test_agreement_tampering_is_detected(tmp_path):
    config, _ = run_agreement(tmp_path, Engine())
    path = tmp_path/'results/agreement.json'
    path.write_text(path.read_text().replace('16', '1', 1))
    with pytest.raises(ValueError, match='changed'):
        cut_agreement.verify(config, tmp_path)


def test_batched_variants_generate_whole_groups_and_singleton_variants_do_not(tmp_path):
    src = sources(('a',)); engine = Engine()
    config = dict(src, seed=42, scorer='gsm8k-flexible-v3', adapter_files={},
                  batch_sizes={v: (1 if v == 'a-input' else 16) for v in src['variants']})
    result = collect(config, items(22), tmp_path, {'device': 'x'}, engine)
    assert result['state'] == 'complete' and result['variants']['a-input']['inference_batch_size'] == 1
    assert [n for v, n in engine.calls if v == 'a-output'] == [16, 6]
    assert [n for v, n in engine.calls if v == 'a-input'] == [1]*22
    assert len(list((tmp_path/'units').iterdir())) == 8*2 # one sealed unit per variant per group


def test_integrity_failure_while_switching_is_persisted(tmp_path):
    src = sources(('a',)); config = dict(src, seed=42, scorer='gsm8k-flexible-v3', adapter_files={})
    with pytest.raises(IntegrityFailure):
        collect(config, items(2), tmp_path, {'device': 'x'}, Engine(integrity_break='a-input'))
    assert (tmp_path/'meta/integrity-failure.json').exists()
    with pytest.raises(ValueError, match='Prior integrity'):
        collect(config, items(2), tmp_path, {'device': 'x'}, Engine())


def test_export_leaves_out_reconstructible_bases(tmp_path):
    from pilot_eval.sampling_stage0_cut_workflow import export
    src = sources(('a',)); config = dict(src, seed=42, scorer='gsm8k-flexible-v3', adapter_files={})
    collect(config, items(2), tmp_path, {'device': 'x'}, Engine())
    (tmp_path/'build/units/m').mkdir(parents=True)
    (tmp_path/'build/units/m/bases.npz').write_bytes(b'large'); (tmp_path/'build/units/m/factors.npz').write_bytes(b'small')
    store = {}
    class Hub:
        def info(self, repo): return type('Info', (), {'private': True})()
        def publish(self, repo, files): store.update({k: p.read_bytes() for k, p in files.items()}); return 'a'*40
        def fetch(self, repo, name, revision):
            p = tmp_path.parent/'remote'; p.write_bytes(store[name]); return p
    receipt = export(tmp_path, tmp_path, repo='private/repo', client=Hub())
    (tmp_path.parent/'bundle.tar.gz').write_bytes(store[receipt['archive']])
    names = tarfile.open(tmp_path.parent/'bundle.tar.gz').getnames()
    assert 'build/units/m/factors.npz' in names and not any(n.endswith('bases.npz') for n in names)


def test_consistency_compares_fresh_originals_with_yesterdays_behaviour(tmp_path):
    from pilot_eval.sampling_stage0_cut_workflow import consistency
    root = tmp_path; run = root/'runs/stage0/n/principal-cuts'; run.mkdir(parents=True)
    behaviour = root/'runs/behaviour'; (behaviour/'results').mkdir(parents=True); (behaviour/'meta').mkdir()
    (behaviour/'meta/runtime.json').write_text(json.dumps({'device': 'A100'}))
    old = [dict(variant='a', item_id=f'q{i}', generated_token_ids=[1, 2], flexible=dict(correct=True)) for i in range(3)]
    (behaviour/'results/responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old))
    (root/'behaviour.json').write_text(json.dumps(dict(run_path='runs/behaviour', batch_sizes={'a': 16})))
    config = dict(run_path='runs/stage0/n/principal-cuts', behaviour_plan='behaviour.json',
                  variants={'a-original': dict(source_arm='a', operation='original')}, batch_sizes={'a-original': 16})
    rows = [dict(variant='a-original', item_id=f'q{i}', generated_token_ids=[1, 2] if i else [3, 2], flexible=dict(correct=bool(i)))
            for i in range(3)]
    report = consistency(config, rows, run)['arms']['a']
    assert report['compared'] == 3 and report['exact_token_matches'] == 2 and report['differing_ids'] == ['q0']


def test_exception_inside_agreement_integrity_check_is_persisted(tmp_path):
    class Broken(Engine):
        def check_integrity(self, full=False): raise RuntimeError('guard crashed')
    with pytest.raises(RuntimeError, match='guard crashed'):
        run_agreement(tmp_path, Broken())
    assert (tmp_path/'meta/integrity-failure.json').exists()
    with pytest.raises(ValueError, match='Prior integrity'):
        run_agreement(tmp_path, Engine())


def test_agreement_for_other_sources_is_rejected_before_any_model_load(tmp_path):
    from pilot_eval.sampling_stage0_cut_workflow import evaluation_plan
    src = sources(); cohort = items()
    config = cut_agreement.configuration(src, cohort)
    cut_agreement.run(config, cohort, tmp_path/'batch-agreement', {'device': 'NVIDIA A100'}, Engine())
    changed = dict(src, decoding={**src['decoding'], 'max_new_tokens': 9})
    with pytest.raises(ValueError, match='different cut sources'):
        evaluation_plan(changed, tmp_path, cohort)
    assert evaluation_plan(src, tmp_path, cohort)['batch_sizes'] == {v: 16 for v in src['variants']}
