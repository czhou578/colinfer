"""N-gram / prompt-lookup drafter (PLAN.md 4.5 item 1): propose the tokens that followed the most recent
earlier occurrence of the current suffix. Host-side, O(1) per step via an incremental index."""
from __future__ import annotations


class NgramDrafter:
    def __init__(self, max_n: int = 3, min_n: int = 2):
        self.max_n, self.min_n = max_n, min_n
        self.tokens: list[int] = []
        self.index: dict[tuple, int] = {}  # n-gram -> position right after its latest occurrence

    def reset(self, tokens):
        self.tokens, self.index = [], {}
        self.extend(tokens)

    def extend(self, new):
        for t in new:
            self.tokens.append(t)
            L = len(self.tokens)
            # index n-grams ending at L-2 (so the latest occurrence of the current suffix is never itself)
            for n in range(self.min_n, self.max_n + 1):
                if L - 1 >= n:
                    self.index[tuple(self.tokens[L - 1 - n:L - 1])] = L - 1

    def propose(self, k: int) -> list[int]:
        """Up to k draft tokens continuing self.tokens (empty if no n-gram match)."""
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(self.tokens) < n:
                continue
            pos = self.index.get(tuple(self.tokens[-n:]))
            if pos is not None:
                return self.tokens[pos:pos + k]
        return []
