"""Suffix-match drafts (prompt lookup / suffix decoding). The history of a slot is its prompt and the reply so far.
When the last tokens of the history occurred earlier in it, the tokens that followed them then are a draft that costs
no GPU work. Code edits, quoted text and structured output repeat long spans. The MTP drafter, one token at a time,
predicts these spans less reliably. Drafts never change the outputs: the speculative cycle accepts only what the
target itself emits.

The index maps each NGRAM-token key of the history to the newest position where it ends, and each position to the
previous one with the same key (a chain, newest first). It grows with each appended token: the scheduler extends it
with each prefill chunk while the GPU runs the chunk, and with each emitted token, and keeps it with the slot while
it stays a prefix of the slot's history. A lookup walks the chain of the last NGRAM tokens and extends each occurrence
backwards to the longest match.
"""
from array import array

MIN_MATCH = 8    # the server's suffix-draft minimum: shorter repeats are left to the MTP drafter
NGRAM = 3        # tokens of the lookup key (the shortest match found)
MAX_MATCH = 32   # match lengths are measured up to this
MAX_CANDS = 16   # occurrences of the key examined, newest first


class SuffixIndex:
    def __init__(self, tokens=()):
        self.toks: list[int] = []
        self.head: dict[tuple, int] = {}  # the last NGRAM tokens -> the newest position where they end
        self.prev = array("i")          # position p (from NGRAM - 1, so index p - NGRAM + 1) -> the previous end of the same key, or -1
        self.extend(tokens)

    def extend(self, tokens):
        h, head, prev = self.toks, self.head, self.prev
        i = len(h)
        for t in tokens:
            h.append(t)
            if i >= NGRAM - 1:
                key = (h[i - 2], h[i - 1], t)
                prev.append(head.get(key, -1))
                head[key] = i
            i += 1

    def draft(self, k: int) -> tuple[int, list[int]]:
        """(match length, up to k tokens that followed the longest earlier occurrence of the history's suffix)."""
        h, prev = self.toks, self.prev
        i = len(h) - 1
        if i < NGRAM - 1:
            return 0, []
        best, at, seen = 0, -1, 0
        c = prev[i - NGRAM + 1]  # the occurrence before the one that ends the history
        while c >= 0 and seen < MAX_CANDS:
            n = NGRAM
            while n < MAX_MATCH and c - n >= 0 and h[c - n] == h[i - n]:
                n += 1
            if n > best:
                best, at = n, c
            seen += 1
            c = prev[c - NGRAM + 1]
        return (best, h[at + 1: at + 1 + k]) if at >= 0 else (0, [])
