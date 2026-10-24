"""Qwen Sparse Attention kernels (csrc/qsa.cu) against torch references ported from transformers' Qwen4ExpTextQSAIndexer
(engine/model/qsa_kernels.py). The tests check:
- the indexer prologue (q_layernorm + RoPE, the raw key write) and the block keys (mean of 4, k_layernorm, RoPE)
- the block scores, and that the score of a row does not depend on T
- the top-k: the same set as a stable sort / torch.topk, ties (many zero scores, -0.0) to the lower index
- the whole selection against the reference indexer
- the sparse attention against SDPA with the explicit mask of the selection, a dense list against attn_decode (bit for
  bit), and T = 1 against T = 16 per row (bit for bit)
Run: uv run pytest tests/test_qsa.py -q"""
import pytest
import torch
import torch.nn.functional as F

from engine.model import qsa_kernels as qk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

EPS, D, HQ, HKV = 1e-6, 256, 24, 2
INV = None


def inv_freq():
    global INV
    if INV is None:
        INV = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    return INV


def ulps(a, b):
    """the bf16 ulp distance of each element (both bf16)"""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7fff), ia)
    ib = torch.where(ib < 0, -(ib & 0x7fff), ib)
    return (ia - ib).abs()


def assert_bf16_close(got, want, max_ulp=1, frac=0.01):
    """at most max_ulp bf16 ulps everywhere, and a mismatch on at most frac of the elements (rsqrt and the reduction
    order of the norm may round differently from torch)"""
    u = ulps(got, want)
    assert u.max().item() <= max_ulp, u.max().item()
    assert (u > 0).float().mean().item() <= frac, (u > 0).float().mean().item()


# ------------------------------------------------------------------------------------------------------------ prologue
@pytest.mark.parametrize("T", [1, 5, 16])
@pytest.mark.parametrize("mrope", [False, True])
def test_prologue(T, mrope):
    torch.manual_seed(T + 100 * mrope)
    B, Lmax = 2, 1024
    proj = torch.randn(B, T, 13952, device="cuda").bfloat16() * 2
    idx = proj[..., 13312:]  # the index columns of the stacked q | k | v | index projection: a strided view
    qn_w = (torch.randn(128, device="cuda") * 0.3).bfloat16()
    pos_t = torch.tensor([7, 500], dtype=torch.int32, device="cuda")
    active = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    ik = torch.zeros(B, Lmax, 128, device="cuda").bfloat16()
    rope_pos = None
    if mrope:  # an image: the (t, h, w) positions differ from the cache rows
        rope_pos = torch.randint(0, 400, (B * T, 3), dtype=torch.int32, device="cuda")
    q = qk.qsa_prologue(idx, qn_w, inv_freq(), pos_t, ik, EPS, T, active=active, rope_pos=rope_pos)
    q2 = qk.qsa_prologue(idx.contiguous(), qn_w, inv_freq(), pos_t, ik.clone(), EPS, T, active=active, rope_pos=rope_pos)
    assert torch.equal(q, q2)  # strided and contiguous rows give the same bits
    for b in range(B):
        p = torch.arange(int(pos_t[b]), int(pos_t[b]) + T, device="cuda")
        cos, sin = qk.rope_cos_sin(rope_pos.view(B, T, 3)[b].long() if mrope else p, inv_freq())
        qr = idx[b, :, :512].reshape(T, 4, 128)
        want = qk.apply_rope(qk.zc_rmsnorm(qr, qn_w, EPS).transpose(0, 1), cos, sin).transpose(0, 1)
        assert_bf16_close(q[b], want)
        # transformers rounds each op of the RoPE in bf16: within a few bf16 ulps of the magnitude of that
        hf = qk.apply_rope(qk.zc_rmsnorm(qr, qn_w, EPS).transpose(0, 1), cos, sin, hf=True).transpose(0, 1)
        torch.testing.assert_close(q[b].float(), hf.float(), rtol=2 ** -6, atol=2 ** -6)
    # the raw key of each row lands at pos_t + t of the active slot only
    assert torch.equal(ik[0, 7:7 + T], idx[0, :, 512:])
    assert not ik[0, :7].any() and not ik[0, 7 + T:].any() and not ik[1].any()


# ------------------------------------------------------------------------------------------------------------ block keys
@pytest.mark.parametrize("pos,T", [(3, 1), (5, 1), (7, 1), (98, 8), (0, 1000), (0, 1), (1001, 16)])
def test_block_keys(pos, T):
    torch.manual_seed(pos * 7 + T)
    B, Lmax = 2, 1056
    ik = torch.randn(B, Lmax, 128, device="cuda").bfloat16()
    kn_w = (torch.randn(128, device="cuda") * 0.3).bfloat16()
    sentinel = torch.full((B, Lmax // 4, 128), 7.0, device="cuda").bfloat16()
    ikb = sentinel.clone()
    pos_t = torch.tensor([pos, pos + 1], dtype=torch.int32, device="cuda")
    qk.qsa_block_keys(ik, ikb, kn_w, inv_freq(), pos_t, T, EPS)
    for b in range(B):
        p0 = pos + b
        lo, hi = p0 // 4, (p0 + T) // 4  # the blocks with 4 j + 3 in [p0, p0 + T)
        written = torch.zeros(Lmax // 4, dtype=torch.bool, device="cuda")
        written[lo:hi] = True
        assert torch.equal(ikb[b, ~written], sentinel[b, ~written])
        if hi > lo:
            starts = torch.arange(lo, hi, device="cuda") * 4
            cos, sin = qk.rope_cos_sin(starts, inv_freq())
            want = qk.block_keys_reference(ik[b, 4 * lo:4 * hi], kn_w, cos, sin, EPS)
            assert_bf16_close(ikb[b, lo:hi], want)


def test_block_keys_rope_positions():
    """rope_pos (the slot's [Lmax, 3] record) and rope_delta pick the rotary position of token 4 j"""
    torch.manual_seed(3)
    B, Lmax, T = 1, 256, 256
    ik = torch.randn(B, Lmax, 128, device="cuda").bfloat16()
    kn_w = (torch.randn(128, device="cuda") * 0.3).bfloat16()
    pos_t = torch.zeros(B, dtype=torch.int32, device="cuda")
    rp = torch.randint(0, 300, (B, Lmax, 3), dtype=torch.int32, device="cuda")
    ikb = torch.zeros(B, Lmax // 4, 128, device="cuda").bfloat16()
    qk.qsa_block_keys(ik, ikb, kn_w, inv_freq(), pos_t, T, EPS, rope_pos=rp)
    cos, sin = qk.rope_cos_sin(rp[0, ::4].long(), inv_freq())
    assert_bf16_close(ikb[0], qk.block_keys_reference(ik[0], kn_w, cos, sin, EPS))
    delta = torch.tensor([-17], dtype=torch.int32, device="cuda")
    qk.qsa_block_keys(ik, ikb, kn_w, inv_freq(), pos_t, T, EPS, rope_delta=delta)
    cos, sin = qk.rope_cos_sin(torch.arange(0, Lmax, 4, device="cuda") - 17, inv_freq())
    assert_bf16_close(ikb[0], qk.block_keys_reference(ik[0], kn_w, cos, sin, EPS))


# ------------------------------------------------------------------------------------------------------------ scores
def _score_inputs(B, T, NBL, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, 4, 128, device="cuda").bfloat16()
    ikb = torch.randn(B, NBL, 128, device="cuda").bfloat16()
    return q, ikb


@pytest.mark.parametrize("T,pos", [(1, 2), (1, 2050), (3, 777), (8, 4093), (16, 10000), (5, 0)])
def test_score_matches_reference(T, pos):
    B, NBL = 2, 4096
    q, ikb = _score_inputs(B, T, NBL, T * 10 + pos)
    pos_t = torch.tensor([pos, pos + 13], dtype=torch.int32, device="cuda")
    scores = qk.qsa_score(q, ikb, pos_t)
    for b in range(B):
        for t in range(T):
            nb = (int(pos_t[b]) + t + 1) // 4
            want = qk.score_reference(q[b, t], ikb[b, :nb])
            torch.testing.assert_close(scores[b, t, :nb], want, rtol=1e-5, atol=1e-5)
            assert not (scores[b, t, :nb].view(torch.int32) == torch.tensor(-0.0).view(torch.int32).item()).any()  # no -0.0
            hi = min((int(pos_t[b]) + T) // 4, NBL)
            assert torch.isneginf(scores[b, t, nb:hi]).all()


def test_score_row_independent_of_T():
    """Row t of a 16-row launch == the one-row launch at position pos + t, bit for bit."""
    B, T, NBL, pos = 2, 16, 2048, 5000
    q, ikb = _score_inputs(B, T, NBL, 5)
    pos_t = torch.tensor([pos, pos + 1], dtype=torch.int32, device="cuda")
    wide = qk.qsa_score(q, ikb, pos_t)
    for t in range(T):
        one = qk.qsa_score(q[:, t:t + 1].contiguous(), ikb, pos_t + t)
        for b in range(B):
            nb = (int(pos_t[b]) + t + 1) // 4
            assert torch.equal(one[b, 0, :nb], wide[b, t, :nb]), t


# ------------------------------------------------------------------------------------------------------------ top-k
def _tied_scores(nb, seed):
    """scores with many exact ties: 70% zeros (some -0.0), the rest from a few distinct values"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randint(1, 40, (nb,), device="cuda", generator=g).float() * 0.25
    z = torch.rand(nb, device="cuda", generator=g) < 0.7
    s[z] = 0.0
    neg = z & (torch.rand(nb, device="cuda", generator=g) < 0.5)
    s[neg] = -0.0
    return s


@pytest.mark.parametrize("nb", [1, 300, 511, 512, 513, 600, 2000, 65536])
@pytest.mark.parametrize("kind", ["tied", "random", "zeros"])
def test_topk_matches_torch(nb, kind):
    NBL = 65536
    B, S, T = 1, 16, 3
    scores = torch.full((B, S, NBL), float("nan"), device="cuda")  # past nb: never read
    pos = 4 * nb - 1  # row 0 has exactly nb complete blocks; rows 1, 2 the same (positions 4 nb, 4 nb + 1)
    for t in range(T):
        if kind == "tied":
            scores[0, t, :nb] = _tied_scores(nb, nb + t)
        elif kind == "random":
            scores[0, t, :nb] = torch.rand(nb, device="cuda")
        else:
            scores[0, t, :nb] = 0.0
    pos_t = torch.tensor([pos], dtype=torch.int32, device="cuda")
    sel, nsel = qk.qsa_topk(scores, pos_t, T)
    for t in range(T):
        k = min(nb, 512)
        assert int(nsel[0, t]) == k
        want = qk.select_blocks(scores[0, t, :nb])
        assert torch.equal(sel[0, t, :k].long(), want), t
        assert (sel[0, t, k:] == -1).all()
        if nb > 512:  # torch.topk on CUDA keeps the lowest index at the boundary too, once -0.0 is +0.0 (x + 0.0)
            tk = torch.topk(scores[0, t, :nb] + 0.0, 512).indices.sort().values
            assert torch.equal(tk, want)


# ------------------------------------------------------------------------------------------------------------ whole selection
@pytest.mark.parametrize("positions", [[2, 3, 4], [2050, 2051, 2052], [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007], [8190]])
def test_selection_matches_reference_indexer(positions):
    """prologue -> block keys -> score -> top-k against the reference indexer on the raw keys (the kernels' RoPE
    rounding). A block may differ only if its reference score is within 1e-4 of the reference threshold."""
    from engine.kernels import ops
    torch.manual_seed(positions[0])
    T, pos, Lmax = len(positions), positions[0], 8192 + 64
    idx = torch.randn(1, T, 640, device="cuda").bfloat16()
    qn_w = (torch.randn(128, device="cuda") * 0.3).bfloat16()
    kn_w = (torch.randn(128, device="cuda") * 0.3).bfloat16()
    ik = torch.randn(1, Lmax, 128, device="cuda").bfloat16()
    ikb = torch.zeros(1, Lmax // 4, 128, device="cuda").bfloat16()
    pos_t = torch.tensor([pos], dtype=torch.int32, device="cuda")
    # the history: block keys of the blocks complete before pos (a restore rebuild), then the step of T rows
    hist = torch.zeros(1, dtype=torch.int32, device="cuda")
    if pos > 0:
        ops().qsa_block_keys(ik, ikb, kn_w, inv_freq(), hist, pos, EPS)
    q = qk.qsa_prologue(idx, qn_w, inv_freq(), pos_t, ik, EPS, T)
    qk.qsa_block_keys(ik, ikb, kn_w, inv_freq(), pos_t, T, EPS)
    scores = qk.qsa_score(q, ikb, pos_t)
    sel, nsel = qk.qsa_topk(scores, pos_t, T)
    pos_list = torch.tensor(positions, device="cuda")
    got = qk.selection_mask(sel[0], nsel[0], pos_list, Lmax)
    want, ref_scores = qk.qsa_select_reference(q[0], ik[0], pos_list, kn_w, inv_freq(), EPS)
    for t in range(T):
        diff = (got[t] ^ want[t]).nonzero().flatten()
        if diff.numel():
            s = ref_scores[t]
            thr = s.sort(descending=True).values[511]
            blocks = (diff // 4).unique()
            assert ((s[blocks] - thr).abs() <= 1e-4 * max(1.0, thr.item())).all(), (t, blocks)


# ------------------------------------------------------------------------------------------------------------ attention
def _kv(B, Lmax, seed):
    torch.manual_seed(seed)
    k = (torch.randn(B, HKV, Lmax, D, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = torch.randn(B, HKV, Lmax, D, device="cuda").to(torch.float8_e4m3fn)
    return k, v


def _random_selection(B, S, positions, seed, full=False):
    """sel / nsel for rows at the given positions [B, T]: random block subsets (all 512 when full), ascending"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    sel = torch.full((B, S, 512), -1, dtype=torch.int32, device="cuda")
    nsel = torch.zeros(B, S, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t, p in enumerate(positions[b]):
            nb = (p + 1) // 4
            n = min(nb, 512) if full else int(torch.randint(0, min(nb, 512) + 1, (1,), device="cuda", generator=g))
            blocks = torch.randperm(nb, device="cuda", generator=g)[:n].sort().values
            sel[b, t, :n] = blocks.int()
            nsel[b, t] = n
    return sel, nsel


@pytest.mark.parametrize("B,T,pos,full", [(1, 1, [0], False), (1, 1, [2], False), (2, 1, [9000, 300], False), (1, 4, [5000], True),
                                          (2, 8, [3, 2047], False), (1, 16, [7000], False), (1, 1, [4095], True)])
def test_attn_matches_sdpa_with_mask(B, T, pos, full):
    torch.manual_seed(T + sum(pos))
    Lmax = 9216
    k, v = _kv(B, Lmax, sum(pos))
    positions = [[p + t for t in range(T)] for p in pos]
    sel, nsel = _random_selection(B, 16, positions, sum(pos) + T, full)
    q = torch.randn(B, HQ, T, D, device="cuda").bfloat16()
    gate = torch.randn(B, T, HQ, 2 * D, device="cuda").bfloat16()
    pos_t = torch.tensor(pos, dtype=torch.int32, device="cuda")
    out = qk.qsa_attn(q, k, v, sel, nsel, pos_t, D ** -0.5, gate)
    for b in range(B):
        mask = qk.selection_mask(sel[b], nsel[b], torch.tensor(positions[b], device="cuda"), Lmax)
        ref = F.scaled_dot_product_attention(q[b:b + 1].float(), k[b:b + 1].float(), v[b:b + 1].float(), attn_mask=mask[None, None],
                                             enable_gqa=True)  # [1, HQ, T, D]
        ref = (ref[0].transpose(0, 1) * torch.sigmoid(gate[b, :, :, D:].float())).reshape(T, HQ * D)
        err = (out[b].float() - ref).abs().max().item()
        assert err <= 1e-2 * max(1.0, ref.abs().max().item()), (b, err)


@pytest.mark.parametrize("T,pos", [(1, 0), (1, 2050), (4, 1000), (16, 2035), (8, 31)])
def test_dense_list_equals_attn_decode(T, pos):
    """Rows with p + 1 <= 2051 select every block: the result equals attn_decode (G = 12, the same stripes) bit for bit."""
    from engine.kernels import ops
    B, Lmax = 2, 4096
    k, v = _kv(B, Lmax, T + pos)
    q = torch.randn(B, HQ, T, D, device="cuda").bfloat16()
    gate = torch.randn(B, T, HQ, 2 * D, device="cuda").bfloat16()
    pos_t = torch.tensor([pos, pos + 1 if pos + T + 1 <= 2051 else pos - 1], dtype=torch.int32, device="cuda")
    positions = [[int(pos_t[b]) + t for t in range(T)] for b in range(B)]
    sel, nsel = _random_selection(B, 16, positions, 0, full=True)
    got = qk.qsa_attn(q, k, v, sel, nsel, pos_t, D ** -0.5, gate)
    want = torch.empty_like(got)
    ops().attn_decode(q, k, v, pos_t + T, want, D ** -0.5, gate)
    assert torch.equal(got, want)


def test_attn_row_bits_independent_of_T():
    """Row t of a 16-row launch == the one-row launch of that row (its position and its list), bit for bit."""
    B, T, Lmax = 2, 16, 12288
    k, v = _kv(B, Lmax, 77)
    pos = [10000, 1500]
    positions = [[p + t for t in range(T)] for p in pos]
    sel, nsel = _random_selection(B, 16, positions, 78)
    q = torch.randn(B, HQ, T, D, device="cuda").bfloat16()
    gate = torch.randn(B, T, HQ, 2 * D, device="cuda").bfloat16()
    pos_t = torch.tensor(pos, dtype=torch.int32, device="cuda")
    wide = qk.qsa_attn(q, k, v, sel, nsel, pos_t, D ** -0.5, gate)
    for t in range(T):
        s1 = torch.full_like(sel, -1)
        n1 = torch.zeros_like(nsel)
        s1[:, 0], n1[:, 0] = sel[:, t], nsel[:, t]
        one = qk.qsa_attn(q[:, :, t:t + 1].contiguous(), k, v, s1, n1, pos_t + t, D ** -0.5, gate[:, t:t + 1].contiguous())
        assert torch.equal(one[:, 0], wide[:, t]), t


def test_selftest():
    from engine.kernels import ops
    res = qk.qsa_selftest(ops(), torch.Generator(device="cuda").manual_seed(1))
    assert all(v < 5e-3 for v in res.values()), res
