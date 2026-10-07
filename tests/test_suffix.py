"""engine/spec/suffix.py: the longest earlier occurrence of a history's suffix and what followed it."""
from engine.spec.suffix import MAX_MATCH, SuffixIndex


def test_longest_match_and_continuation():
    s = SuffixIndex([1, 2, 3, 4, 5, 6, 7, 9, 9, 2, 3, 4])
    assert s.draft(4) == (3, [5, 6, 7, 9])
    s.extend([5, 6])  # the suffix 2 3 4 5 6 now matches 5 tokens
    assert s.draft(4) == (5, [7, 9, 9, 2])


def test_prefers_longer_over_newer():
    s = SuffixIndex([8, 1, 2, 3, 50, 0, 1, 2, 3, 60, 8, 1, 2, 3])  # 8 1 2 3 occurred first, 0 1 2 3 later
    assert s.draft(2) == (4, [50, 0])


def test_no_match_and_short_history():
    assert SuffixIndex([1, 2]).draft(4) == (0, [])
    assert SuffixIndex([1, 2, 3]).draft(4) == (0, [])
    assert SuffixIndex([1, 2, 3, 4, 5]).draft(4) == (0, [])


def test_runs_and_match_cap():
    assert SuffixIndex([7, 7, 7, 7]).draft(4) == (3, [7])
    s = SuffixIndex(list(range(100)) + list(range(100)))
    assert s.draft(3) == (MAX_MATCH, [0, 1, 2])
