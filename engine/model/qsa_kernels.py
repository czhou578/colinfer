"""Qwen Sparse Attention (QSA) on the kernels of csrc/qsa.cu, for qwen4_exp (docs/history/qwen4exp_plan.md section 5.3).

The indexer of an attention layer scores 4-token blocks of the context for each new row; the row then attends to its top
512 blocks and to the tokens of its incomplete block (the tail). The state of a slot and layer: the raw indexer keys
ik bf16 [Lmax, 128] (no norm, no RoPE) and the block keys ikb bf16 [Lmax / 4, 128]. One step of T rows per slot:

    q_idx = qsa_prologue(idx, ...)        q_layernorm + RoPE of the 4 query heads; raw keys -> ik
    qsa_block_keys(ik, ikb, ...)          the blocks that the new rows complete -> ikb
    scores = qsa_score(q_idx, ikb, ...)   fp32 [B, 16, Lmax / 4]
    sel, nsel = qsa_topk(scores, ...)     the top 512 blocks of each row, ascending
    out = qsa_attn(q, k, v, sel, nsel, pos_t, ...)

The functions here allocate the outputs and call ops(). The torch references (qsa_select_reference and its helpers)
follow transformers' Qwen4ExpTextQSAIndexer; the tests compare the kernels with them."""
from __future__ import annotations

import math

import torch

from engine.kernels import ops

INDEX_HEADS, INDEX_DIM, RATIO, TOPK_BLOCKS, MAX_ROWS = 4, 128, 4, 512, 16


# ------------------------------------------------------------------------------------------------------------ kernels
def qsa_prologue(idx, qn_w, inv_freq, pos_t, ik, eps, T: int, active=None, rope_delta=None, rope_pos=None, mrope=(11, 10)):
    """idx: bf16 [B, T, 640] rows (a strided column slice is fine). Returns q_idx bf16 [B, T, 4, 128]; writes the raw keys
    to ik [B, Lmax, 128] at pos_t + t (active slots)."""
    B = pos_t.numel()
    q = torch.empty(B, T, INDEX_HEADS, INDEX_DIM, device=ik.device, dtype=torch.bfloat16)
    ops().qsa_prologue(idx, qn_w, inv_freq, pos_t, ik, q, eps, active, rope_delta, rope_pos, *mrope)
    return q


def qsa_block_keys(ik, ikb, kn_w, inv_freq, pos_t, T: int, eps, active=None, rope_delta=None, rope_pos=None, mrope=(11, 10)):
    """Writes ikb[b, j] for the blocks that the rows [pos_t[b], pos_t[b] + T) complete. rope_pos: the slot's int32
    [B, Lmax, 3] record of rotary positions (else 4 j + rope_delta, else 4 j)."""
    ops().qsa_block_keys(ik, ikb, kn_w, inv_freq, pos_t, T, eps, active, rope_delta, rope_pos, *mrope)


def qsa_score(q_idx, ikb, pos_t, scores=None):
    """fp32 scores [B, 16, NBL] (rows >= T and blocks no row sees are not written)."""
    B, NBL = ikb.shape[0], ikb.shape[1]
    if scores is None:
        scores = torch.empty(B, MAX_ROWS, NBL, device=ikb.device, dtype=torch.float32)
    ops().qsa_score(q_idx, ikb, pos_t, scores)
    return scores


def qsa_topk(scores, pos_t, T: int, sel=None, nsel=None):
    """sel int32 [B, S, 512] (ascending, -1 past the count), nsel int32 [B, S]."""
    B, S = scores.shape[:2]
    if sel is None:
        sel = torch.empty(B, S, TOPK_BLOCKS, device=scores.device, dtype=torch.int32)
    if nsel is None:
        nsel = torch.empty(B, S, device=scores.device, dtype=torch.int32)
    ops().qsa_topk(scores, pos_t, T, sel, nsel)
    return sel, nsel


def qsa_attn(q, k, v, sel, nsel, pos_t, scale, gate=None, out=None):
    """q bf16 [B, 24, T, 256], k / v e4m3 [B, 2, Lmax, 256]. With gate (the q_proj output [B, T, 24, 512]): out
    [B, T, 24 * 256] = attn * sigmoid(gate); else out has the q layout."""
    if out is None:
        B, H, T, D = q.shape
        out = torch.empty((B, T, H * D) if gate is not None else q.shape, device=q.device, dtype=torch.bfloat16)
    ops().qsa_attn(q, k, v, sel, nsel, pos_t, out, scale, gate)
    return out


# ------------------------------------------------------------------------------------------------------------ references
def zc_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm: fp32 math, (1 + w), cast back to the input dtype."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * (1.0 + w.float())).type_as(x)


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor, mrope=(11, 10)) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos / sin [N, R] at positions [N] (or the (t, h, w) positions [N, 3] of interleaved multimodal RoPE)."""
    if positions.dim() == 2:
        i = torch.arange(inv_freq.numel(), device=positions.device)
        axis = torch.where((i % 3 == 1) & (i < 3 * mrope[0]), 1, torch.where((i % 3 == 2) & (i < 3 * mrope[1]), 2, 0))
        freqs = positions[:, axis].float() * inv_freq[None, :]
    else:
        freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, hf: bool = False) -> torch.Tensor:
    """RoPE on the first R = cos.shape[-1] dims of x [..., N, D] (cos / sin [N, R]). hf: transformers' rounding (bf16 cos /
    sin, every op in bf16); else the kernels' (fp32 cos / sin and math, one rounding)."""
    r = cos.shape[-1]
    dt = x.dtype
    if hf:
        cos, sin, xr = cos.to(dt), sin.to(dt), x[..., :r]
    else:
        xr = x[..., :r].float()
    rot = torch.cat((-xr[..., r // 2:], xr[..., :r // 2]), dim=-1)
    y = (xr * cos + rot * sin).to(dt)
    return torch.cat((y, x[..., r:]), dim=-1)


def block_keys_reference(ik: torch.Tensor, kn_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float, hf: bool = False) -> torch.Tensor:
    """The block keys of the nb = len(ik) // 4 complete blocks of one slot: ik bf16 [L, 128] raw keys, cos / sin [nb, R] at
    the block starts. pooled = bf16(fp32 mean of 4), k_layernorm, RoPE."""
    nb = ik.shape[0] // RATIO
    pooled = ik[:nb * RATIO].view(nb, RATIO, -1).float().mean(dim=1).to(ik.dtype)
    return apply_rope(zc_rmsnorm(pooled, kn_w, eps), cos, sin, hf)


def score_reference(q_idx: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """transformers' block scores for one row: q_idx [4, 128], keys [nb, 128] -> fp32 [nb]."""
    s = torch.matmul(q_idx.float(), keys.float().transpose(-1, -2)).transpose(-1, -2)
    return torch.relu(s).sum(dim=-1) / math.sqrt(INDEX_DIM)


def select_blocks(scores: torch.Tensor, k: int = TOPK_BLOCKS) -> torch.Tensor:
    """The blocks of the top min(k, n) scores, ties to the lower index (a stable descending sort), ascending."""
    if scores.numel() <= k:
        return torch.arange(scores.numel(), device=scores.device)
    order = torch.sort(scores, descending=True, stable=True).indices
    return order[:k].sort().values


def qsa_select_reference(q_idx: torch.Tensor, ik: torch.Tensor, positions: torch.Tensor, kn_w: torch.Tensor, inv_freq: torch.Tensor,
                         eps: float, rope_pos: torch.Tensor | None = None, hf: bool = False, mrope=(11, 10)):
    """The indexer's selection for the rows of one slot, ported from transformers' Qwen4ExpTextQSAIndexer.forward.

    q_idx [T, 4, 128]: the normed, rotated query heads (the prologue output). ik [L, 128]: the raw keys of the slot,
    valid at least through max(positions). positions [T]: the cache position p of each row. rope_pos (optional) [L, 3]:
    the rotary positions of the cache rows (else the cache position). Returns (mask bool [T, L]: the tokens each row
    attends to, scores: a list of the T fp32 score vectors [nb_t])."""
    T, L = q_idx.shape[0], ik.shape[0]
    mask = torch.zeros(T, L, dtype=torch.bool, device=ik.device)
    out_scores = []
    for t in range(T):
        p = int(positions[t])
        nb = (p + 1) // RATIO
        if nb > 0:
            starts = torch.arange(nb, device=ik.device) * RATIO
            cos, sin = rope_cos_sin(rope_pos[starts] if rope_pos is not None else starts, inv_freq, mrope)
            keys = block_keys_reference(ik[:nb * RATIO], kn_w, cos, sin, eps, hf)
            s = score_reference(q_idx[t], keys)
            blocks = select_blocks(s)
            tok = (blocks[:, None] * RATIO + torch.arange(RATIO, device=ik.device)).flatten()
            mask[t, tok] = True
        else:
            s = torch.zeros(0, device=ik.device)
        mask[t, nb * RATIO:p + 1] = True  # the tail
        out_scores.append(s)
    return mask, out_scores


def selection_mask(sel: torch.Tensor, nsel: torch.Tensor, positions: torch.Tensor, L: int) -> torch.Tensor:
    """bool [T, L]: the tokens of the kernel's selection (sel [T, 512], nsel [T]) plus the tail of each row."""
    T = positions.numel()
    mask = torch.zeros(T, L, dtype=torch.bool, device=sel.device)
    for t in range(T):
        n, p = int(nsel[t]), int(positions[t])
        blocks = sel[t, :n].long()
        mask[t, (blocks[:, None] * RATIO + torch.arange(RATIO, device=sel.device)).flatten()] = True
        mask[t, (p + 1) // RATIO * RATIO:p + 1] = True
    return mask


# ------------------------------------------------------------------------------------------------------------ self-test
def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def qsa_selftest(o, g: torch.Generator) -> dict[str, float]:
    """Startup self-test of the QSA kernels (engine/selftest.py style): relative errors against torch on a small problem
    of one slot at position 2500 (625 complete blocks: the top-k path). o: ops(); g: a CUDA generator."""
    dev = "cuda"
    res = {}
    L, pos, eps = 2560, 2500, 1e-6
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
    ik = torch.randn(1, L, INDEX_DIM, device=dev, generator=g).bfloat16()
    kn_w = (torch.randn(INDEX_DIM, device=dev, generator=g) * 0.2).bfloat16()
    q_idx = torch.randn(1, 1, INDEX_HEADS, INDEX_DIM, device=dev, generator=g).bfloat16()
    pos_t = torch.tensor([0], dtype=torch.int32, device=dev)
    ikb = torch.zeros(1, L // RATIO, INDEX_DIM, device=dev, dtype=torch.bfloat16)
    o.qsa_block_keys(ik, ikb, kn_w, inv, pos_t, pos + 1, eps)
    nb = (pos + 1) // RATIO
    cos, sin = rope_cos_sin(torch.arange(nb, device=dev) * RATIO, inv)
    want = block_keys_reference(ik[0, :nb * RATIO], kn_w, cos, sin, eps)
    res["qsa_block_keys"] = _rel(ikb[0, :nb], want)
    pos_t.fill_(pos)
    scores = torch.empty(1, MAX_ROWS, L // RATIO, device=dev)
    o.qsa_score(q_idx, ikb, pos_t, scores)
    res["qsa_score"] = _rel(scores[0, 0, :nb], score_reference(q_idx[0, 0], ikb[0, :nb]))
    sel = torch.empty(1, MAX_ROWS, TOPK_BLOCKS, dtype=torch.int32, device=dev)
    nsel = torch.empty(1, MAX_ROWS, dtype=torch.int32, device=dev)
    o.qsa_topk(scores, pos_t, 1, sel, nsel)
    res["qsa_topk"] = float(int(nsel[0, 0]) != TOPK_BLOCKS or not torch.equal(sel[0, 0].long(), select_blocks(scores[0, 0, :nb])))
    # sparse attention of the selection against SDPA with the explicit mask
    D = o.HEAD_DIM
    q = torch.randn(1, 24, 1, D, device=dev, generator=g).bfloat16()
    k = torch.randn(1, 2, L, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    v = torch.randn(1, 2, L, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    out = torch.empty_like(q)
    o.qsa_attn(q, k, v, sel, nsel, pos_t, out, D ** -0.5)
    m = selection_mask(sel[0], nsel[0], pos_t + 0, L)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m[None, None], enable_gqa=True)
    res["qsa_attn"] = _rel(out, ref)
    return res
