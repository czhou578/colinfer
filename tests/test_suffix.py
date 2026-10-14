"""engine/spec/suffix.py: the longest earlier occurrence of a history's suffix and what followed it, and the index
against a plain reference (the lists-of-positions index it replaced): the same drafts after every token, built in
one go or in pieces, with image ids past the vocabulary."""
import random

from engine.spec.suffix import MAX_CANDS, MAX_MATCH, NGRAM, SuffixIndex


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


class Reference:
    """Every position of each NGRAM-token key, newest MAX_CANDS examined."""

    def __init__(self, tokens):
        self.toks, self.ends = [], {}
        self.extend(tokens)

    def extend(self, tokens):
        h = self.toks
        for t in tokens:
            h.append(t)
            if len(h) >= NGRAM:
                self.ends.setdefault(tuple(h[-NGRAM:]), []).append(len(h) - 1)

    def draft(self, k):
        h = self.toks
        i = len(h) - 1
        best, at = 0, -1
        for c in reversed(self.ends.get(tuple(h[-NGRAM:]), [])[-MAX_CANDS - 1:]):
            if c == i:
                continue
            n = NGRAM
            while n < MAX_MATCH and c - n >= 0 and h[c - n] == h[i - n]:
                n += 1
            if n > best:
                best, at = n, c
        return (best, h[at + 1: at + 1 + k]) if at >= 0 else (0, [])


def stream(rng, n):
    """Repetitive tokens: a small vocabulary, copied spans, and image ids past the vocabulary."""
    out = []
    while len(out) < n:
        r = rng.random()
        if r < 0.3 and len(out) > 40:
            a = rng.randrange(len(out) - 20)
            out += out[a:a + rng.randrange(3, 40)]
        elif r < 0.35:
            out += [2 ** 48 + 248320 + rng.randrange(4) for _ in range(rng.randrange(1, 6))]
        else:
            out += [rng.randrange(12) for _ in range(rng.randrange(1, 8))]
    return out[:n]


def test_the_index_drafts_like_the_reference():
    rng = random.Random(0)
    for trial in range(6):
        toks = stream(rng, 1500)
        ref, one, pieces = Reference([]), SuffixIndex(), SuffixIndex()
        at = 0
        for i in range(len(toks)):
            ref.extend([toks[i]])
            one.extend([toks[i]])
            if i + 1 - at >= rng.randrange(1, 64) or i == len(toks) - 1:  # extended in pieces, as the scheduler does per chunk
                pieces.extend(toks[at:i + 1])
                at = i + 1
            else:
                continue
            for k in (3, 15):
                want = ref.draft(k)
                assert one.draft(k) == want == pieces.draft(k), (trial, i, k)
        assert SuffixIndex(toks).draft(15) == ref.draft(15)
    assert SuffixIndex([1, 2]).draft(3) == (0, []) and SuffixIndex([]).draft(3) == (0, [])
