"""Suffix-match drafts (prompt lookup / suffix decoding): when the last tokens of a slot's history occurred earlier in
that history, in the prompt or in the reply so far, the tokens that followed them then are a draft that costs no GPU
work. Code edits, quoted text and structured output repeat long spans that the MTP drafter, one token at a time,
predicts less reliably. Drafts never change outputs: the speculative cycle accepts only what the target itself emits.

The index maps every NGRAM-token sequence of the history to the positions where it ends and grows as tokens are
appended; a lookup extends the newest occurrences of the last NGRAM tokens backwards to the longest match.
"""
MIN_MATCH = 8    # the server's default --suffix-drafts: shorter repeats are left to the MTP drafter
NGRAM = 3        # tokens of the lookup key (the shortest match found)
MAX_MATCH = 32   # match lengths are measured up to this
MAX_CANDS = 16   # occurrences of the key examined, newest first


class SuffixIndex:
    def __init__(self, tokens: list[int]):
        self.toks: list[int] = []
        self.ends: dict[tuple, list[int]] = {}
        self.extend(tokens)

    def extend(self, tokens: list[int]):
        h = self.toks
        for t in tokens:
            h.append(t)
            if len(h) >= NGRAM:
                self.ends.setdefault(tuple(h[-NGRAM:]), []).append(len(h) - 1)

    def draft(self, k: int) -> tuple[int, list[int]]:
        """(match length, up to k tokens that followed the longest earlier occurrence of the history's suffix)."""
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
