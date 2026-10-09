"""In-graph sampler (PLAN.md 4.3 item 7): greedy / temperature / min-p / top-k / top-p per slot.

All parameters are in per-slot device buffers, so the decode CUDA graph can capture the sampler. The host rewrites the
buffers only when the request of a slot changes.

The position keys the randomness. The sampler draws the token at absolute position t of a request by inverse CDF from
the processed distribution. It uses one uniform u = Philox(seed, counter t) (csrc/sampling.cu). Thus the tokens of a
request depend only on (seed, prompt), never on the other requests in the batch. Speculative decoding reproduces plain
sampling token for token (engine/spec/accept.py).

Order (as in vLLM / HF): temperature -> min-p (relative to the top token) -> top-k -> top-p -> sample.
temperature == 0 selects the argmax for that slot, whatever the other parameters are.
"""
from __future__ import annotations

import math

import torch


class SamplerParams:
    def __init__(self, batch: int, vocab: int, device):
        self.vocab = vocab
        self.temperature = torch.zeros(batch, dtype=torch.float32, device=device)
        self.top_k = torch.full((batch,), vocab, dtype=torch.int32, device=device)
        self.top_p = torch.ones(batch, dtype=torch.float32, device=device)
        self.log_min_p = torch.full((batch,), -math.inf, dtype=torch.float32, device=device)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=device)

    def view(self, lo: int, hi: int) -> "SamplerParams":
        """Parameters of slots [lo, hi), sharing this object's device buffers."""
        v = SamplerParams.__new__(SamplerParams)
        v.vocab = self.vocab
        for name in ("temperature", "top_k", "top_p", "log_min_p", "seed"):
            setattr(v, name, getattr(self, name)[lo:hi])
        return v

    def set(self, slot: int, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, seed: int = 0):
        """top_k <= 0 disables top-k, and so does a top_k of at least the vocabulary (the buffer is int32); min_p <= 0
        disables min-p."""
        self.temperature[slot] = float(temperature)
        self.top_k[slot] = min(int(top_k), self.vocab) if top_k > 0 else self.vocab
        self.top_p[slot] = float(top_p)
        self.log_min_p[slot] = math.log(min_p) if min_p > 0 else -math.inf
        self.seed[slot] = int(seed)


def sample(logits: torch.Tensor, p: SamplerParams, pos: torch.Tensor) -> torch.Tensor:
    """logits [B, V] fp32 -> token ids [B] int64. pos [B]: the position of the token being sampled (device).
    Graph-capturable (no host syncs)."""
    from engine.spec.accept import draw
    return draw(logits[:, None], p, pos)[:, 0]
