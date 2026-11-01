from scripts.sampling_coverage_audit import batch_histogram


def test_batch_histogram_uses_saved_membership_and_reports_empty():
    assert batch_histogram([["c", "a"], ["b", "d"]], {"a", "c", "d"}) == {0: 1, 1: 1}
