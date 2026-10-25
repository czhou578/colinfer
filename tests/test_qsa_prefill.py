"""The sparse attention of a prefill chunk (csrc/qsa_prefill.cu, engine/model/qsa_kernels.py prefill_*). The tests check:
- qsa_score_rows gives the bits of qsa_score (the decode kernel) for every row, at any chunk length and offset
- prefill_select selects for each row exactly the blocks of the decode path (qsa_score + qsa_topk of that row alone)
- the group unions (qsa_union) against Python sets
- prefill_prologue / prefill_block_keys: the K / V, raw keys, rope record and block keys of a chunk
- prefill_attend on every path against SDPA with the reference indexer's mask (qsa_select_reference), from rows with
  no complete block through the 2051 boundary to long context, with and without a cached prefix
Run: uv run pytest tests/test_qsa_prefill.py -q"""
import pytest
import torch
import torch.nn.functional as F

from engine.model import qsa_kernels as qk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

EPS, D, HQ, HKV = 1e-6, 256, 24, 2


def inv_freq():
    return 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))


def ops():
    from engine.kernels import ops as o
    return o()


# ------------------------------------------------------------------------------------------------------------ scores
@pytest.mark.parametrize("pos0,T", [(0, 5), (3, 16), (100, 17), (2040, 300), (9000, 1000)])
def test_score_rows_bits_equal_decode(pos0, T):
    torch.manual_seed(pos0 + T)
    Lmax = 12288
    q = torch.randn(T, 4, 128, device="cuda").bfloat16()
    ikb = torch.randn(1, Lmax // 4, 128, device="cuda").bfloat16()
    nbl = (pos0 + T) // 4
    scores = torch.full((T, Lmax // 4), float("nan"), device="cuda")
    ops().qsa_score_rows(q, ikb, torch.tensor([pos0], dtype=torch.int32, device="cuda"), scores)
    for t0 in range(0, T, 16):  # the decode kernel on the full ikb, 16 rows a launch
        m = min(16, T - t0)
        want = qk.qsa_score(q[None, t0:t0 + m].contiguous(), ikb, torch.tensor([pos0 + t0], dtype=torch.int32, device="cuda"))
        for t in range(m):
            nb = (pos0 + t0 + t + 1) // 4
            assert torch.equal(scores[t0 + t, :nb], want[0, t, :nb]), (t0, t)
            assert torch.isneginf(scores[t0 + t, nb:min(nbl, (pos0 + t0 + m) // 4)]).all()


def _decode_select(q_idx, ikb, pos):
    """the decode path for one row at position pos: qsa_score + qsa_topk with T = 1 on the slot's full block keys"""
    p = torch.tensor([pos], dtype=torch.int32, device="cuda")
    sc = qk.qsa_score(q_idx[None, None].contiguous(), ikb, p)
    sel, nsel = qk.qsa_topk(sc, p, 1)
    return sel[0, 0], int(nsel[0, 0])


@pytest.mark.parametrize("pos0,T,group", [(0, 40, 512), (2030, 64, 512), (5000, 600, 256), (2048 * 4 - 7, 70, 32)])
def test_select_equals_decode(pos0, T, group):
    torch.manual_seed(pos0 + T)
    Lmax = 16384
    q = torch.randn(T, 4, 128, device="cuda").bfloat16()
    ikb = torch.randn(1, Lmax // 4, 128, device="cuda").bfloat16()
    ikb[0, ::7] = 0  # zero keys: tied (zero) scores
    sel, nsel = qk.prefill_select(q, ikb, pos0, T, group=group)
    rows = sorted(set(range(0, T, max(1, T // 40))) | {T - 1})
    for t in rows:
        s1, n1 = _decode_select(q[t], ikb, pos0 + t)
        assert int(nsel[t]) == n1 == min(512, (pos0 + t + 1) // 4), t
        assert torch.equal(sel[t], s1), t


def test_union_matches_sets():
    torch.manual_seed(1)
    pos0, T, Lmax = 4093, 37, 8192
    q = torch.randn(T, 4, 128, device="cuda").bfloat16()
    ikb = torch.randn(1, Lmax // 4, 128, device="cuda").bfloat16()
    sel, nsel = qk.prefill_select(q, ikb, pos0, T)
    ul, uc = qk.prefill_union(sel, nsel, pos0, T)
    for gi in range((T + 3) // 4):
        want = {}
        for tt in range(min(4, T - 4 * gi)):
            t = 4 * gi + tt
            p = pos0 + t
            blocks = set(sel[t, :int(nsel[t])].tolist())
            if (p + 1) % 4:
                blocks.add((p + 1) // 4)
            for b in blocks:
                want[b] = want.get(b, 0) | (1 << tt)
        got = ul[gi, :int(uc[gi])].tolist()
        assert [e >> 4 for e in got] == sorted(want), gi
        assert all((e & 15) == want[e >> 4] for e in got), gi


# ------------------------------------------------------------------------------------------------------------ the chunk
class Chunk:
    """A slot with a random cached prefix of pos0 rows and a chunk of T rows through the prologues and block keys."""

    def __init__(self, pos0, T, Lmax, seed, rope_pos=False):
        g = torch.Generator(device="cuda").manual_seed(seed)
        rnd = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=g) * sc).bfloat16()  # noqa: E731
        self.pos0, self.T, self.Lmax = pos0, T, Lmax
        self.kc = (torch.randn(1, HKV, Lmax, D, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        self.vc = torch.randn(1, HKV, Lmax, D, device="cuda", generator=g).to(torch.float8_e4m3fn)
        self.ik = rnd(1, Lmax, 128)
        self.ikb = torch.zeros(1, Lmax // 4, 128, device="cuda", dtype=torch.bfloat16)
        self.rec = torch.zeros(1, Lmax, 3, dtype=torch.int32, device="cuda")
        self.rec[0, :pos0] = torch.arange(pos0, dtype=torch.int32, device="cuda")[:, None]
        self.qn, self.kn, self.iqn, self.ikn = rnd(256, sc=0.3), rnd(256, sc=0.3), rnd(128, sc=0.3), rnd(128, sc=0.3)
        if pos0:  # the block keys of the cached prefix (a restore rebuild)
            qk.qsa_block_keys(self.ik, self.ikb, self.ikn, inv_freq(), torch.zeros(1, dtype=torch.int32, device="cuda"), pos0, EPS,
                              rope_pos=self.rec)
        self.proj = rnd(T, 13952, sc=2.0)  # the stacked q | k | v | index projection of the chunk
        qp, kp, vp, idx = self.proj.split([12288, 512, 512, 640], dim=-1)
        self.qp, self.idx = qp, idx
        self.rope_pos = None
        if rope_pos:  # image rows: (t, h, w) positions apart from the cache positions
            self.rope_pos = torch.randint(0, pos0 + T, (T, 3), dtype=torch.int32, device="cuda", generator=g)
        self.q, self.q_idx = qk.prefill_prologue(qp, kp, vp, idx, self.qn, self.kn, self.iqn, inv_freq(), self.kc, self.vc, self.ik, self.rec, pos0,
                                                 EPS, rope_pos=self.rope_pos)
        qk.prefill_block_keys(self.ik, self.ikb, self.ikn, inv_freq(), self.rec, pos0, T, EPS)


def test_prologue_and_block_keys():
    """prefill_prologue writes what attn_prologue and qsa_prologue write for the chunk, and the rope record; the block
    keys of the chunk's blocks (one of them begun in the prefix) equal the reference on the record's positions."""
    pos0, T, Lmax = 1001, 70, 2048
    for rope_pos in (False, True):
        c = Chunk(pos0, T, Lmax, 5, rope_pos)
        want_rec = c.rope_pos if rope_pos else torch.arange(pos0, pos0 + T, dtype=torch.int32, device="cuda")[:, None].expand(T, 3)
        assert torch.equal(c.rec[0, pos0:pos0 + T], want_rec)
        assert torch.equal(c.ik[0, pos0:pos0 + T], c.idx[:, 512:])
        # the same ops by hand on a fresh cache
        kc, vc = torch.zeros_like(c.kc), torch.zeros_like(c.vc)
        qp, kp, vp, idx = c.proj.split([12288, 512, 512, 640], dim=-1)
        q = torch.empty(1, HQ, T, D, device="cuda", dtype=torch.bfloat16)
        pos_t = torch.tensor([pos0], dtype=torch.int32, device="cuda")
        rp = want_rec.contiguous()
        ops().attn_prologue(qp, kp, vp, c.qn, c.kn, inv_freq(), pos_t, kc, vc, q, EPS, None, None, rp)
        assert torch.equal(q, c.q)
        assert torch.equal(kc[:, :, pos0:pos0 + T].view(torch.uint8), c.kc[:, :, pos0:pos0 + T].view(torch.uint8))
        assert torch.equal(qk.qsa_prologue(idx, c.iqn, inv_freq(), pos_t, c.ik.clone(), EPS, T, rope_pos=rp), c.q_idx)
        lo, hi = pos0 // 4, (pos0 + T) // 4  # the blocks with 4 j + 3 in [pos0, pos0 + T)
        cos, sin = qk.rope_cos_sin(c.rec[0, 4 * lo:4 * hi:4].long(), inv_freq())
        want = qk.block_keys_reference(c.ik[0, 4 * lo:4 * hi], c.ikn, cos, sin, EPS)
        u = (c.ikb[0, lo:hi].float() - want.float()).abs() / want.float().abs().clamp_min(1e-3)
        assert u.max().item() < 2 ** -7
        assert not c.ikb[0, hi:].any()


def _check_attend(c: Chunk, rows, paths):
    """prefill_attend on each path against SDPA with the reference indexer's mask for the given rows of the chunk."""
    pos0, T = c.pos0, c.T
    L = pos0 + T
    sel, nsel = qk.prefill_select(c.q_idx, c.ikb, pos0, T)
    rows_t = torch.tensor(rows, device="cuda")
    positions = rows_t + pos0
    got_mask = qk.selection_mask(sel[rows_t], nsel[rows_t], positions, L)
    ref_mask, ref_scores = qk.qsa_select_reference(c.q_idx[0, rows_t], c.ik[0, :L], positions, c.ikn, inv_freq(), EPS, rope_pos=c.rec[0, :L].long())
    for i in range(len(rows)):  # the kernel's selection may differ from the reference only at a near tie
        diff = (got_mask[i] ^ ref_mask[i]).nonzero().flatten()
        if diff.numel():
            s = ref_scores[i]
            thr = s.sort(descending=True).values[511]
            assert ((s[(diff // 4).unique()] - thr).abs() <= 1e-4 * max(1.0, thr.item())).all(), rows[i]
    q = c.q[:, :, rows_t].float()
    ref = F.scaled_dot_product_attention(q, c.kc[:, :, :L].float(), c.vc[:, :, :L].float(), attn_mask=ref_mask[None, None], enable_gqa=True)
    gate = c.qp.reshape(T, HQ, 2 * D)[rows_t, :, D:].float()
    ref = (ref[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(len(rows), -1)
    tol = 1e-2 * max(1.0, ref.abs().max().item())
    outs = {}
    for path in paths:
        o = qk.prefill_attend(c.q, c.kc, c.vc, sel, nsel, pos0, D ** -0.5, c.qp, path)
        assert o.shape == (T, HQ * D) and o.dtype == torch.bfloat16
        err = (o[rows_t].float() - ref).abs().max().item()
        assert err <= tol, (path, err)
        outs[path] = o
    if "list" in outs and "union" in outs:  # the same f16 core: close to each other as well
        assert (outs["list"].float() - outs["union"].float()).abs().max().item() <= tol
    return outs


@pytest.mark.parametrize("pos0,T", [(0, 3), (0, 61), (2040, 32), (0, 2051), (0, 2052), (1, 2051), (3000, 300), (11000, 97)])
def test_attend_matches_sdpa_with_reference_mask(pos0, T):
    Lmax = 12288
    c = Chunk(pos0, T, Lmax, pos0 * 3 + T)
    rows = sorted(set(range(0, T, max(1, T // 24))) | {0, T - 1} | {t for t in (2049 - pos0, 2050 - pos0, 2051 - pos0) if 0 <= t < T})
    paths = ["union", "mask", "list"] + (["causal"] if pos0 + T <= qk.DENSE_LEN else [])
    _check_attend(c, rows, paths)


def test_attend_image_rope_positions():
    """rows with explicit (t, h, w) rotary positions: the block keys rotate at the recorded positions"""
    c = Chunk(2500, 200, 4096, 9, rope_pos=True)
    _check_attend(c, list(range(0, 200, 9)), ["union"])


def test_causal_path_refuses_sparse_rows():
    c = Chunk(2000, 100, 4096, 3)
    sel, nsel = qk.prefill_select(c.q_idx, c.ikb, 2000, 100)
    with pytest.raises(ValueError):
        qk.prefill_attend(c.q, c.kc, c.vc, sel, nsel, 2000, D ** -0.5, c.qp, "causal")
