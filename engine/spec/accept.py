"""Sampling that speculative decoding reproduces exactly (PLAN.md 4.5, Phase 4 week 14 / Phase 5).

Every token is drawn by inverse CDF from the target's processed distribution p_t (temperature -> min-p -> top-k ->
top-p, exactly what plain sampling uses) with a uniform keyed by (seed, position t). Verification samples the
target at every verified row with the uniforms of those rows' positions and accepts the leading drafts that equal
those samples; the first mismatch is replaced by the target's sample, and if every draft matches the last row's
sample is the bonus token. The emitted sequence is therefore exactly the sequence plain sampling would produce
(same seed, same logits), whatever the draft length, the batch width or the drafter.

With a deterministic drafter (the MTP argmax), the chance of accepting draft d is P(x_t = d) = p_t(d), the same
acceptance rate as speculative rejection sampling (Leviathan et al. 2023) with a point-mass proposal, whose
correction distribution normalize(p_t with d removed) is exactly the law of x_t given x_t != d.
Everything runs on the device (graph-capturable).
"""
from __future__ import annotations

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


def inverse_cdf(P: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """P [R, V] probabilities, u [R] uniforms in (0, 1] -> [R] int64: the first index whose CDF reaches u
    (never a zero-probability token)."""
    c = P.cumsum(-1)
    return (c < (u * c[:, -1])[:, None]).sum(-1).clamp_max(P.shape[-1] - 1)


def draw(logits: torch.Tensor, p, pos: torch.Tensor) -> torch.Tensor:
    """logits [B, R, V] fp32: row r of slot b predicts the token at position pos[b] + r.
    p: SamplerParams for the B slots. Returns the sampled tokens [B, R] int64 (argmax for slots with temperature 0)."""
    from engine.kernels import ops
    B, R, _ = logits.shape
    u = torch.empty(B, R, device=logits.device)
    ops().philox_uniform(p.seed, pos.long().contiguous(), u)
    greedy = logits.argmax(-1)
    rows = [inverse_cdf(processed_probs(logits[b], p.temperature[b:b + 1], p.top_k[b:b + 1], p.top_p[b:b + 1], p.log_min_p[b:b + 1]), u[b])
            for b in range(B)]
    return torch.where(p.temperature[:, None] > 0, torch.stack(rows), greedy)
