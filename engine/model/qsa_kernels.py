"""Qwen Sparse Attention (QSA) on the kernels of csrc/qsa.cu, for qwen4_exp (docs/history/qwen4exp_plan.md section 5.3).

The indexer of an attention layer scores 4-token blocks of the context for each new row; the row then attends to its top
512 blocks and to the tokens of its incomplete block (the tail). The state of a slot and layer: the raw indexer keys
ik bf16 [Lmax, 128] (no norm, no RoPE) and the block keys ikb bf16 [Lmax / 4, 128]. One step of T rows per slot:

    q_idx = qsa_prologue(idx, ...)        q_layernorm + RoPE of the 4 query heads; raw keys -> ik
    qsa_block_keys(ik, ikb, ...)          the blocks that the new rows complete -> ikb
    scores = qsa_score(q_idx, ikb, ...)   fp32 [B, 16, Lmax / 4]
    sel, nsel = qsa_topk(scores, ...)     the top 512 blocks of each row, ascending
    out = qsa_attn(q, k, v, sel, nsel, pos_t, ...)

A prefill chunk of one slot (T rows, csrc/qsa_prefill.cu) runs prefill_prologue, prefill_block_keys, prefill_select
and prefill_attend (tests/test_qsa_prefill.py).

The functions here allocate the outputs and call ops(). The torch references (qsa_select_reference and its helpers)
follow transformers' Qwen4ExpTextQSAIndexer; tests/test_qsa.py compares the kernels with them. qsa_selftest() is an
entry of the startup self-test (engine/selftest.py)."""
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


# ------------------------------------------------------------------------------------------------------------ prefill
# A prefill chunk of one slot: T rows at the positions pos0 .. pos0 + T - 1 (T up to 8192), csrc/qsa_prefill.cu.
# The selection: qsa_score_rows (the computation of qsa_score, all rows in one launch) and qsa_topk over a group of rows,
# so a row selects exactly the blocks that decode selects for it. The attention has four paths (prefill_attend):
#   "union"   qsa_union + qsa_union_attn: per group of 4 rows, the union of their blocks; the 12 heads of a KV head and
#             the 4 rows share the K / V reads. The default.
#   "causal"  every row sees every token (pos0 + T <= DENSE_LEN): FlashInfer FA2, causal, over a bf16 copy of the prefix
#   "mask"    FlashInfer FA2 with the selection as a packed custom mask: dense cost over the whole prefix
#   "list"    qsa_attn (the decode kernel) over groups of rows: the bits of a row equal its decode bits; 24 stripe
#             partials per row make it slow
DENSE_LEN = TOPK_BLOCKS * RATIO + RATIO - 1  # 2051: a row at position p sees every token iff p + 1 <= 2051
SELECT_GROUP = 512  # rows per score buffer (fp32 [512, L / 4]: 134 MB at 262k tokens)
LIST_GROUP = 256  # rows per qsa_attn launch of the "list" path (its fp32 partials: 151 MB)
PREFILL_PATH = "union"  # the default path of prefill_attend. On the real checkpoint and WikiText (bench/prefill_chunk_bench.py --part real), a
# 4096-row chunk: a 4-row group attends to 3108 keys (union) at the prefix 8k and 3695 at 64k (one row: 2048 + its tail); union 12.8 / 15.0 ms,
# mask 21.1 / 107.5 ms per layer


def _row_pos(pos0: int, T: int, step: int, device) -> torch.Tensor:
    """int32 [ceil(T / step)]: pos0, pos0 + step, ... (the pos_t of each launch; no host synchronization)."""
    return torch.arange(0, T, step, dtype=torch.int32, device=device) + pos0


def _pos_tensor(pos0: int, pos_t, device) -> torch.Tensor:
    return pos_t if pos_t is not None else torch.full((1,), pos0, dtype=torch.int32, device=device)


def prefill_prologue(qp, kp, vp, idx, q_norm_w, k_norm_w, iq_norm_w, inv_freq, kc, vc, ik, rope_rec, pos0: int, eps, rope_pos=None, rope_delta=None,
                     mrope=(11, 10), pos_t=None):
    """The prologues of an attention layer over the T rows of a chunk of one slot at positions pos0 .. pos0 + T - 1.

    qp [T, 24 * 512], kp / vp [T, 2 * 256], idx [T, 640]: bf16 column views of the stacked q | k | v | index projection.
    kc / vc e4m3 [1, 2, Lmax, 256], ik bf16 [1, Lmax, 128], rope_rec int32 [1, Lmax, 3] (the slot's record of rotary
    positions). rope_pos: int32 [T, 3] (images), else every axis is the cache position + rope_delta (int32 [1] on the
    device, or None for 0). pos_t: int32 [1] holding pos0 on the device (the state's; else a new one).
    Writes K / V, the raw indexer keys and the rope record of the rows. Returns (q bf16 [1, 24, T, 256], q_idx bf16
    [1, T, 4, 128])."""
    T, dev = qp.shape[0], qp.device
    if rope_pos is None:
        p = torch.arange(pos0, pos0 + T, dtype=torch.int32, device=dev)
        if rope_delta is not None:
            p = p + rope_delta.view(1)
        rope_pos = p[:, None].expand(T, 3).contiguous()
    rope_rec[0, pos0:pos0 + T] = rope_pos
    pos_t = _pos_tensor(pos0, pos_t, dev)
    q = torch.empty(1, qp.shape[1] // (2 * ops().HEAD_DIM), T, ops().HEAD_DIM, device=dev, dtype=torch.bfloat16)
    ops().attn_prologue(qp, kp, vp, q_norm_w, k_norm_w, inv_freq, pos_t, kc, vc, q, eps, None, None, rope_pos, *mrope)
    q_idx = qsa_prologue(idx, iq_norm_w, inv_freq, pos_t, ik, eps, T, rope_pos=rope_pos, mrope=mrope)
    return q, q_idx


def prefill_block_keys(ik, ikb, kn_w, inv_freq, rope_rec, pos0: int, T: int, eps, mrope=(11, 10), pos_t=None):
    """The block keys of every block that the chunk's rows complete (4 j + 3 in [pos0, pos0 + T)), at the rotary
    positions of the record rope_rec int32 [1, Lmax, 3]. ik bf16 [1, Lmax, 128], ikb bf16 [1, Lmax / 4, 128]."""
    qsa_block_keys(ik, ikb, kn_w, inv_freq, _pos_tensor(pos0, pos_t, ik.device), T, eps, rope_pos=rope_rec, mrope=mrope)


def prefill_select(q_idx, ikb, pos0: int, T: int, group: int = SELECT_GROUP, sel=None, nsel=None):
    """The selection of the T rows of a chunk: q_idx bf16 [1, T, 4, 128] (or [T, 4, 128]), ikb bf16 [1, NBL, 128] with
    the blocks through (pos0 + T) / 4 written. Returns (sel int32 [T, 512] ascending, -1 past the count; nsel int32 [T]).
    `group` rows at a time: qsa_score_rows writes their scores to one fp32 buffer [group, nbl] (nbl: the blocks the chunk
    can see, rounded up to 4), then one qsa_topk launch selects for all of them."""
    q_idx = q_idx.reshape(T, INDEX_HEADS, INDEX_DIM)
    dev = ikb.device
    group = max(MAX_ROWS, group // MAX_ROWS * MAX_ROWS)
    nbl = min(ikb.shape[1], -(-((pos0 + T) // RATIO) // 4) * 4)
    if sel is None:
        sel = torch.empty(T, TOPK_BLOCKS, dtype=torch.int32, device=dev)
    if nsel is None:
        nsel = torch.empty(T, dtype=torch.int32, device=dev)
    if nbl == 0:  # no row has a complete block
        sel.fill_(-1)
        nsel.zero_()
        return sel, nsel
    kb = ikb[0, :nbl]  # contiguous: the score kernel takes the row length from it
    pos = _row_pos(pos0, T, group, dev)
    scores = torch.empty(min(group, T), nbl, device=dev, dtype=torch.float32)
    k = ops()
    for i, g0 in enumerate(range(0, T, group)):
        n = min(group, T - g0)
        sc, p = scores[:n], pos[i:i + 1]
        k.qsa_score_rows(q_idx[g0:g0 + n], kb, p, sc)
        k.qsa_topk(sc[None], p, n, sel[None, g0:g0 + n], nsel[None, g0:g0 + n])
    return sel, nsel


def prefill_union(sel, nsel, pos0: int, T: int):
    """(ul int32 [ceil(T / 4), cap], uc int32 [ceil(T / 4)]): per group of 4 rows, the union of their selected and tail
    blocks, ascending, as (block << 4) | (the mask of the rows that attend to it)."""
    dev, ng = sel.device, -(-T // 4)
    ul = torch.empty(ng, ops().qsa_union_cap(), dtype=torch.int32, device=dev)
    uc = torch.empty(ng, dtype=torch.int32, device=dev)
    ops().qsa_union(sel, nsel, torch.full((1,), pos0, dtype=torch.int32, device=dev), ul, uc, (pos0 + T) // RATIO + 1)
    return ul, uc


def packed_selection_mask(sel, nsel, pos0: int, T: int, Lp: int) -> torch.Tensor:
    """The selection of T rows (sel [T, 512], nsel [T]) plus each row's tail as FlashInfer's packed custom mask over Lp
    keys (Lp % 8 == 0): the bits of the row-major bool [T, Lp], little bit order, uint8 [T * Lp / 8]. A 4-token block
    is one nibble; built with torch ops on the device."""
    assert Lp % 8 == 0
    dev, nb4 = sel.device, Lp // RATIO
    nib = torch.zeros(T, nb4 + 1, dtype=torch.uint8, device=dev)
    valid = torch.arange(TOPK_BLOCKS, device=dev)[None, :] < nsel[:, None]
    nib.scatter_(1, torch.where(valid, sel.long(), nb4), 15)  # the entries past the count go to the dropped column nb4
    p1 = torch.arange(pos0 + 1, pos0 + T + 1, device=dev)
    nib.scatter_(1, (p1 // RATIO)[:, None], ((1 << (p1 % RATIO)) - 1).to(torch.uint8)[:, None])  # the tail: (p + 1) % 4 tokens
    return (nib[:, 0:nb4:2] | (nib[:, 1:nb4:2] << 4)).reshape(-1)


def _gated(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """o bf16 [T, H * D] times sigmoid of the gate half of each q row (gate: the q rows [T, H * 2 D]), one rounding."""
    T, HD = o.shape
    D = ops().HEAD_DIM
    g = gate.reshape(T, HD // D, 2 * D)[..., D:]
    return (o.view(T, -1, D).float() * torch.sigmoid(g.float())).to(torch.bfloat16).view(T, HD)


def prefill_attend(q, kc, vc, sel, nsel, pos0: int, scale: float, gate, path: str | None = None) -> torch.Tensor:
    """The attention of the T rows of a chunk of one slot: q bf16 [1, 24, T, 256] (the prologue output), kc / vc e4m3
    [1, 2, Lmax, 256] with the chunk's rows written, sel / nsel the chunk's selection (prefill_select; unused on the
    causal path), gate: the q rows bf16 [T, 24 * 512] (q | gate per head; a column view is fine). Returns
    o bf16 [T, 24 * 256] = attention * sigmoid(gate). path: "union" (the default), "causal", "mask" or "list" (see
    above)."""
    import flashinfer
    T, H, D = q.shape[2], q.shape[1], q.shape[3]
    L = pos0 + T
    path = path or PREFILL_PATH
    if path == "union":
        ul, uc = prefill_union(sel, nsel, pos0, T)
        out = torch.empty(T, H * D, device=q.device, dtype=torch.bfloat16)
        ops().qsa_union_attn(q, kc, vc, torch.full((1,), pos0, dtype=torch.int32, device=q.device), ul, uc, gate, out, scale)
        return out
    if path == "list":
        out = torch.empty(T, H * D, device=q.device, dtype=torch.bfloat16)
        pos = _row_pos(pos0, T, LIST_GROUP, q.device)
        for i, t0 in enumerate(range(0, T, LIST_GROUP)):
            t1 = min(T, t0 + LIST_GROUP)
            qsa_attn(q[:, :, t0:t1].contiguous(), kc, vc, sel[None, t0:t1], nsel[None, t0:t1], pos[i:i + 1], scale,
                     gate[t0:t1].contiguous(), out[t0:t1].view(1, t1 - t0, H * D))
        return out
    if path == "causal":
        if L > DENSE_LEN:
            raise ValueError(f"the causal path needs pos0 + T <= {DENSE_LEN}, got {L}")
        kk, vv = kc[0, :, :L].to(torch.bfloat16), vc[0, :, :L].to(torch.bfloat16)
        o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), kk, vv, causal=True, kv_layout="HND", sm_scale=scale)
        return _gated(o.reshape(T, -1), gate)
    if path != "mask":
        raise ValueError(f"unknown attention path {path!r}")
    Lp = -(-L // 8) * 8
    kk = torch.empty(kc.shape[1], Lp, D, device=q.device, dtype=torch.bfloat16)
    vv = torch.empty_like(kk)
    kk[:, :L], vv[:, :L] = kc[0, :, :L], vc[0, :, :L]
    kk[:, L:], vv[:, L:] = 0, 0  # masked keys, but a NaN in V would still reach P V
    mask = packed_selection_mask(sel, nsel, pos0, T, Lp)
    o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), kk, vv, packed_custom_mask=mask, kv_layout="HND", sm_scale=scale)
    return _gated(o.reshape(T, -1), gate)


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
    """Startup self-test of the QSA kernels (an entry of engine/selftest.py): relative errors against torch on a small
    problem of one slot at position 2500 (625 complete blocks: the top-k path). o: ops(); g: a CUDA generator."""
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
