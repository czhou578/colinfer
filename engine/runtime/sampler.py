"""In-graph sampler (PLAN.md 4.3 item 7): greedy / temperature / min-p / top-k / top-p per slot.

All parameters live in per-slot device buffers so the sampler is captured inside the decode CUDA
graph; the host only rewrites the buffers when a slot's request changes. Randomness is Philox
(FlashInfer's rejection sampler) keyed by a per-slot seed and an offset that the graph advances
on the device every step.

Order (as in vLLM / HF): temperature -> min-p (relative to the top token) -> top-k -> top-p -> sample.
temperature == 0 selects the argmax for that slot regardless of the other parameters.
"""
from __future__ import annotations

import math

import torch

OFFSET_STRIDE = 1 << 20  # Philox offset consumed per step; far more than one rejection loop draws


class SamplerParams:
    def __init__(self, batch: int, vocab: int, device):
        self.vocab = vocab
        self.temperature = torch.zeros(batch, dtype=torch.float32, device=device)
        self.top_k = torch.full((batch,), vocab, dtype=torch.int32, device=device)
        self.top_p = torch.ones(batch, dtype=torch.float32, device=device)
        self.log_min_p = torch.full((batch,), -math.inf, dtype=torch.float32, device=device)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=device)
        self.offset = torch.zeros(batch, dtype=torch.int64, device=device)

    def set(self, slot: int, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, seed: int = 0):
        """top_k <= 0 disables top-k; min_p <= 0 disables min-p."""
        self.temperature[slot] = float(temperature)
        self.top_k[slot] = int(top_k) if top_k > 0 else self.vocab
        self.top_p[slot] = float(top_p)
        self.log_min_p[slot] = math.log(min_p) if min_p > 0 else -math.inf
        self.seed[slot] = int(seed)
        self.offset[slot] = 0


def sample(logits: torch.Tensor, p: SamplerParams) -> torch.Tensor:
    """logits [B, V] fp32 -> token ids [B] int64. Graph-capturable (no host syncs)."""
    import flashinfer.sampling as fs
    greedy = logits.argmax(-1)
    scaled = logits / p.temperature.clamp_min(1e-6)[:, None]
    thr = scaled.amax(-1, keepdim=True) + p.log_min_p[:, None]
    scaled = scaled.masked_fill(scaled < thr, float("-inf"))
    # One call per slot: FlashInfer 0.7.0.post1 accepts per-row seed/offset arrays but row 0's values
    # perturb every row's draw, so a slot's output would depend on its neighbours. Slots are <= 4.
    drawn = torch.cat([fs.top_k_top_p_sampling_from_logits(scaled[b:b + 1], p.top_k[b:b + 1], p.top_p[b:b + 1],
                                                           seed=p.seed[b:b + 1], offset=p.offset[b:b + 1])
                       for b in range(logits.shape[0])])
    p.offset += OFFSET_STRIDE
    return torch.where(p.temperature > 0, drawn.long(), greedy)
