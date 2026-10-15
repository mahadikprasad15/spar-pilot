"""Public prereveal review workflow; CPU only and no learned readouts."""
import pytest


def test_exposure_history_is_durable_and_cannot_be_relabelled(tmp_path):
    from pilot_eval.trace_review import record_exposure, inspect_review
    event = dict(schema_version=1, id='first-look', reviewer='Prasad',
        experiment='pilot-4', output_kind='scalar-results',
        description='Saw behavioral accuracy plots; token-list history unknown.',
        observed_at='unknown', unknown_history='Earlier token-list exposure not established.')
    record_exposure(tmp_path, 'review-test', event)
    restored = inspect_review(tmp_path, 'review-test')
    assert restored['state'] == 'awaiting-review'
    assert restored['reveal_allowed'] is False
    assert restored['exposures'] == [event]
    assert restored['learned_lists_exposed'] is False
    record_exposure(tmp_path, 'review-test', event)
    assert len(inspect_review(tmp_path, 'review-test')['exposures']) == 1
    with pytest.raises(ValueError, match='different'):
        record_exposure(tmp_path, 'review-test', dict(event, output_kind='learned-lens-token-lists'))
    learned = dict(event, id='second-look', experiment='pilot-5',
        output_kind='learned-lens-token-lists', description='Opened adapted preview before freeze.')
    record_exposure(tmp_path, 'review-test', learned)
    assert inspect_review(tmp_path, 'review-test')['learned_lists_exposed'] is True
