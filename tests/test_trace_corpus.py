"""Public raw-corpus workflow with controlled pinned source/tokenizer boundaries."""
import hashlib
import json
from pathlib import Path

import pytest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CharacterTokenizer:
    all_special_ids = [0]
    bos_token_id = 0
    eos_token_id = None
    metadata = {'backend': 'independent-character-fixture', 'add_bos_token': False}
    files = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) + 10 for char in text]

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text),
                'offset_mapping': [(index, index + 1) for index in range(len(text))]}


class Sources:
    def __init__(self, root):
        self.tokenizer = CharacterTokenizer()
        self.rows = []
        for index in range(150):
            text = self.document(index)['text']
            self.rows.append(dict(id=f'fineweb:train:{index}', corpus='fineweb',
                source_index=index, source_id=f'doc-{index}', source_url=None,
                source_split='train', dataset_revision='web-pin', content=text[:128],
                content_sha256=digest(text[:128]), full_document_sha256=digest(text),
                prefix_token_ids=self.tokenizer.encode(text)[:128]))
        self.config = dict(protocol_version='pilot3-write-v1', model='Qwen/Qwen2.5-1.5B-Instruct',
            model_revision='model-pin', tokenizer_revision='tokenizer-pin', context_limit=32768,
            fineweb=dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT', revision='web-pin',
                pool_limit=2000, count=150, content_token_limit=128, source_indices=list(range(150))),
            items_path='inputs/legacy.json', items_sha256=digest(self.rows))
        (root / 'inputs').mkdir()
        (root / self.config['items_path']).write_text(json.dumps(self.rows))
        self.path = root / 'activation.prepared.json'
        self.path.write_text(json.dumps(self.config))
        self.stream_calls = 0

    @staticmethod
    def document(index):
        return {'id': f'doc-{index}', 'text': f'{index:04d} ' + 'x' * 160}

    def load_source(self, path, root):
        # External source archive boundary; no training or checkpoint recreation.
        return self.config, json.loads((root / self.config['items_path']).read_text())

    def load_tokenizer(self, model, revision):
        assert (model, revision) == ('Qwen/Qwen2.5-1.5B-Instruct', 'tokenizer-pin')
        return self.tokenizer

    def stream_fineweb(self, config, revision):
        assert (config, revision) == ('sample-10BT', 'web-pin')
        self.stream_calls += 1
        for index in range(2100):
            yield self.document(index)


def review():
    return dict(schema_version=1, procedure='same-pool-disjoint-random-v1',
        seed=42, reviewed=True, notes='Fixture procedure approved before selection.',
        prior_exposure='unknown; stratum does not imply unseen data')


def test_prepare_freezes_raw_inputs_and_restores_without_network(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus, load_corpus
    deps = Sources(tmp_path)
    plan_path = prepare_corpus(deps.path, tmp_path, 'raw-v1', review=review(), dependencies=deps)
    plan, rows = load_corpus(plan_path, tmp_path)
    assert len(rows) == 200
    assert [row['id'] for row in rows[:150]] == [f'fineweb:train:{i}' for i in range(150)]
    assert all(row['stratum'] == 'legacy-150' for row in rows[:150])
    assert all(row['stratum'] == 'new-50' for row in rows[150:])
    assert len({row['id'] for row in rows}) == 200
    assert all(len(row['content_positions']) == 128 for row in rows)
    first = rows[0]
    assert first['input_ids'][:5] == [58, 58, 58, 58, 42]
    assert first['offsets'][:5] == [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    assert first['content_positions'] == list(range(128))
    assert first['position_ids'] == list(range(128))
    assert first['attention_mask'] == [1] * 128
    assert plan['tokenizer']['bos_token_id'] == 0
    assert first['position_zero_kind'] == 'content'
    assert plan['form'] == 'raw' and plan['status'] == 'prepared'
    assert deps.stream_calls == 1

    class Offline(Sources):
        def load_tokenizer(self, *args):
            raise AssertionError('completed corpus must not retokenize')
        def stream_fineweb(self, *args):
            raise AssertionError('completed corpus must not resample or stream')
        def load_source(self, *args):
            raise AssertionError('completed corpus verifies its saved input parent fingerprints')
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert prepare_corpus(deps.path, tmp_path, 'raw-v1', review=review(),
                          dependencies=Offline.__new__(Offline)) == plan_path
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_unreviewed_selection_stops_before_source_or_dataset_access(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus
    deps = Sources(tmp_path)
    class NoAccess(Sources):
        def load_source(self, *args):
            raise AssertionError('review must gate source access')
    settings = review()
    settings['reviewed'] = False
    with pytest.raises(ValueError, match='reviewed'):
        prepare_corpus(deps.path, tmp_path, 'unreviewed', review=settings,
                       dependencies=NoAccess.__new__(NoAccess))
    assert not (tmp_path / 'plans/unreviewed').exists()


def test_missing_sealed_input_is_rejected_even_if_completion_marker_omits_it(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus, load_corpus
    deps = Sources(tmp_path)
    path = prepare_corpus(deps.path, tmp_path, 'sealed', review=review(), dependencies=deps)
    config = json.loads(path.read_text())
    marker_path = path.parent / 'complete.json'
    marker = json.loads(marker_path.read_text())
    del marker['files'][config['pool_path']]
    marker_path.write_text(json.dumps(marker))
    (tmp_path / config['pool_path']).unlink()
    with pytest.raises(ValueError, match='omits required'):
        load_corpus(path, tmp_path)


def test_legacy_mapping_failure_is_saved_and_never_replaced(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus
    deps = Sources(tmp_path)
    deps.rows[0]['prefix_token_ids'][0] += 1
    deps.config['items_sha256'] = digest(deps.rows)
    (tmp_path / deps.config['items_path']).write_text(json.dumps(deps.rows))
    deps.path.write_text(json.dumps(deps.config))
    with pytest.raises(ValueError, match='Legacy raw-token mapping changed: fineweb:train:0'):
        prepare_corpus(deps.path, tmp_path, 'bad-map', review=review(), dependencies=deps)
    directory = tmp_path / 'runs/pilot-5/bad-map/corpus'
    status = json.loads((directory / 'meta/status.json').read_text())
    assert status['state'] == 'failed' and status['completed'] == 0
    assert 'fineweb:train:0' in (directory / 'logs/errors.jsonl').read_text()
    assert not (tmp_path / 'plans/bad-map/complete.json').exists()


def test_real_tokenizer_file_keeps_bos_configuration_separate_from_raw_inputs(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from pilot_eval.trace_corpus import PinnedRawTokenizer, prepare_corpus, load_corpus
    engine = Tokenizer(models.WordLevel({'[UNK]': 0, '<s>': 1, 'word': 2}, unk_token='[UNK]'))
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    engine.add_special_tokens(['<s>'])
    engine.post_processor = processors.TemplateProcessing(single='<s> $A', special_tokens=[('<s>', 1)])
    tokenizer_path = tmp_path / 'tokenizer.json'
    engine.save(str(tokenizer_path))
    config_path = tmp_path / 'tokenizer_config.json'
    config_path.write_text(json.dumps({'bos_token': '<s>', 'add_bos_token': True,
                                     'eos_token': None, 'pad_token': None}))
    tokenizer = PinnedRawTokenizer(tokenizer_path, config_path)
    assert tokenizer.encode('word word') == [2, 2]
    assert tokenizer('word word', add_special_tokens=False, return_offsets_mapping=True) == {
        'input_ids': [2, 2], 'offset_mapping': [(0, 4), (5, 9)]}
    assert tokenizer.bos_token_id == 1
    assert tokenizer.metadata['configured_add_bos_token'] is True
    assert tokenizer.metadata['raw_add_special_tokens'] is False


def test_tokenizer_files_are_retained_and_sealed_with_corpus(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus, load_corpus
    deps = Sources(tmp_path)
    tokenizer_file = tmp_path / 'fixture-tokenizer.json'
    tokenizer_file.write_text('{"fixture": true}')
    deps.tokenizer.files = {'tokenizer.json': tokenizer_file}
    path = prepare_corpus(deps.path, tmp_path, 'retained', review=review(), dependencies=deps)
    config, _ = load_corpus(path, tmp_path)
    retained = tmp_path / config['tokenizer_files']['tokenizer.json']['path']
    assert retained.read_bytes() == tokenizer_file.read_bytes()
    retained.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_corpus(path, tmp_path)


def test_self_consistent_legacy_text_must_still_be_the_pinned_document_prefix(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus
    deps = Sources(tmp_path)
    deps.rows[0]['content'] = 'z' * 128
    deps.rows[0]['content_sha256'] = digest('z' * 128)
    deps.rows[0]['prefix_token_ids'] = deps.tokenizer.encode('z' * 128)
    deps.config['items_sha256'] = digest(deps.rows)
    (tmp_path / deps.config['items_path']).write_text(json.dumps(deps.rows))
    deps.path.write_text(json.dumps(deps.config))
    with pytest.raises(ValueError, match='Legacy raw-token mapping changed'):
        prepare_corpus(deps.path, tmp_path, 'wrong-prefix', review=review(), dependencies=deps)


def test_notebook_commands_prepare_and_verify_corpus_in_fresh_cpu_process(tmp_path):
    import subprocess
    import sys
    from pilot_eval.cli import main
    deps = Sources(tmp_path)
    review_path = tmp_path / 'review.json'
    review_path.write_text(json.dumps(review()))
    args = ['traces-corpus-prepare', '--source-config', str(deps.path), '--review', str(review_path),
            '--name', 'cli-corpus', '--output-root', str(tmp_path)]
    assert main(args, dependencies=deps) == 0
    path = tmp_path / 'plans/cli-corpus/traces.corpus.json'
    subprocess.run([sys.executable, '-m', 'pilot_eval', 'traces-corpus-verify',
                    '--config', str(path), '--output-root', str(tmp_path)], check=True)
    assert main(args, dependencies=deps) == 0
    assert deps.stream_calls == 1


def test_fresh_notebook_restores_selected_corpus_without_tokenizer_access(tmp_path):
    from pilot_eval.trace_corpus import prepare_corpus
    from pilot_eval.trace_session import TraceSession
    deps = Sources(tmp_path)
    corpus = prepare_corpus(deps.path, tmp_path, 'raw', review=review(), dependencies=deps)
    session = TraceSession(tmp_path, tmp_path, 'experiment')
    session.select_corpus(corpus)
    fresh = TraceSession(tmp_path, tmp_path, 'experiment').restore()
    assert fresh['corpus']['path'] == str(corpus)
    assert fresh['corpus']['counts'] == {'legacy-150': 150, 'new-50': 50}
    assert fresh['corpus']['state'] == 'saved; verify inputs before capture'
    corpus.write_text('{}')
    with pytest.raises(ValueError, match='Selected corpus config changed'):
        session.restore()
