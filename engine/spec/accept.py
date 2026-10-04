"""Distribution-preserving acceptance for speculative sampling (PLAN.md 4.5, Phase 4 week 14).

The MTP drafter proposes deterministically (its argmax), i.e. a point-mass proposal q = delta(d).
Speculative sampling (Leviathan et al. 2023; Chen et al. 2023) with such a proposal reduces to:
    accept d_i with probability p_i(d_i); on the first rejection emit a token drawn from p_i with d_i
    removed (renormalised), i.e. norm(max(0, p_i - q_i)); if every draft is accepted, emit a bonus token
    drawn from p_k.
p_i is the target's processed distribution at row i (temperature -> min-p -> top-k -> top-p, exactly
what plain sampling draws from), so the emitted sequence has exactly the plain-sampling distribution.
Everything runs on the device (graph-capturable).
"""
from __future__ import annotations

import math

import torch


def processed_probs(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
                    log_min_p: torch.Tensor) -> torch.Tensor:
    """logits [R, V] fp32 (rows of one request) -> renormalised probabilities [R, V] after the filters.
    temperature / top_k / top_p / log_min_p: 1-element device tensors (the slot's parameters)."""
    import flashinfer.sampling as fs
    z = logits / temperature.clamp_min(1e-6)
    z = z.masked_fill(z < z.amax(-1, keepdim=True) + log_min_p, float("-inf"))
    p = torch.softmax(z, -1)
    R = p.shape[0]
    p = fs.top_k_renorm_probs(p, top_k.expand(R).contiguous())
    return fs.top_p_renorm_probs(p, top_p.expand(R).contiguous())


def accept_sample(P: torch.Tensor, drafts: torch.Tensor, u: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor):
    """P [k+1, V] target probabilities, drafts [k] (int64), u [k] uniforms in [0, 1).
    Returns (n [1] int32 = accepted inputs incl. the first, i.e. 1 + accepted drafts; next token [1] int64)."""
    import flashinfer.sampling as fs
    k = drafts.numel()
    rows = torch.arange(k, device=P.device)
    pd = P[rows, drafts]                                   # p_i(d_i)
    acc = (u < pd).int()
    n = (1 + acc.cumprod(0).sum()).int().view(1)           # 1..k+1
    resid = P[:k].clone()
    resid.scatter_(1, drafts[:, None], 0.0)  # (no CPU scalar tensor: graph-capturable)
    s = resid.sum(-1, keepdim=True)
    resid = torch.where(s > 0, resid / s.clamp_min(1e-30), P[:k])  # p_i was a point mass on d_i: never rejected anyway
    cand = torch.cat([resid, P[k:]], 0)                     # row i < k: correction after rejecting d_i; row k: bonus
    c = fs.sampling_from_probs(cand, seed=seed, offset=offset)
    nxt = c.long().gather(0, (n.long() - 1))
    return n, nxt
