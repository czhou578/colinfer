"""The decode MoE kernels (csrc/moe.cu) and KernelMoE (engine/model/moe_kernels.py). The tests check:
- the routing against torch (a stable sort of the fp32 softmax), with ties and -0.0
- each kernel against a torch reference, and the full layer against the per-expert fp32 loop of transformers
- that each output row is bit-identical for any set of rows in the launch, with all, no and some experts shared
- the skip flag, a CUDA graph replayed with new routing, the shape checks, and one layer at the real shape
Run: uv run pytest tests/test_moe.py -q"""
import os

import pytest
import torch

from engine.model.moe_kernels import KernelMoE, moe_reference, random_moe_weights, route_reference
from engine.weights.loader import dequant_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

TOPK = 10


def ops():
    from engine.kernels import ops as o
    return o()


def rel(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def ref_args(wts):
    return {k: v for k, v in wts.items() if k != "in_scale"}


def route(logits, E, stats=None):
    M = logits.shape[0]
    ids = torch.empty(M, TOPK, dtype=torch.int32, device="cuda")
    w = torch.empty(M, TOPK, device="cuda")
    sg = torch.empty(M, device="cuda")
    ops().moe_route(logits, E, ids, w, sg, stats)
    return ids, w, sg


def experts(moe, x, ids, w, sg, shared):
    """gate_up -> down -> combine with given routing (fresh outputs)."""
    M = x.shape[0]
    h = torch.empty(M * TOPK, moe.I, dtype=torch.bfloat16, device="cuda")
    yp = torch.empty(M * TOPK, moe.K, device="cuda")
    y = torch.empty(M, moe.K, dtype=torch.bfloat16, device="cuda")
    ops().moe_gate_up(x, ids, moe.w_gu, moe.sf_gu, moe.gs, h)
    ops().moe_down(h, ids, moe.w_dn, moe.sf_dn, moe.gs, yp)
    ops().moe_combine(yp, w, sg, shared, y)
    return h, yp, y


@pytest.fixture(scope="module")
def small():
    """E = 16, K = 1024, I = 256 (Is = 256): heavy expert sharing."""
    wts = random_moe_weights(16, 1024, 256, generator=gen(1))
    return wts, KernelMoE(**wts)


@pytest.fixture(scope="module")
def wide():
    """E = 512, K = 512, I = 128: room for rows with no expert in common (M <= 48)."""
    wts = random_moe_weights(512, 512, 128, Is=256, generator=gen(2))
    return wts, KernelMoE(**wts, max_rows=48)


# ------------------------------------------------------------------------------------------------------------ routing
@pytest.mark.parametrize("E,NL", [(512, 520), (16, 24), (64, 65)])
def test_route_matches_torch(E, NL):
    g = gen(E)
    M = 37
    logits = (torch.randn(M, NL, device="cuda", generator=g) * 2).bfloat16()
    logits[0, :E] = 0.0
    logits[0, 1::3] = -0.0  # -0.0 ties +0.0
    logits[1, :E] = torch.randint(-2, 3, (E,), device="cuda", generator=g).bfloat16()  # many ties at the boundary
    ids, w, sg = route(logits, E)
    rid, rw = route_reference(logits[:, :E])
    assert torch.equal(ids.long(), rid)
    ulp = rw.abs() * 2.0 ** -7
    assert ((w - rw).abs() <= ulp).all() and (w == rw).float().mean() > 0.95
    assert torch.equal(w, w.bfloat16().float())
    assert torch.equal(sg, torch.sigmoid(logits[:, E]).float())
    # with ties, the selected set is torch.topk's set (the lowest indices at the boundary). Row 0 is left out: torch.topk
    # orders -0.0 below +0.0, but the softmax (and so the reference sort) ties them.
    tk = torch.topk(logits[1:, :E].float(), TOPK, dim=-1).indices
    assert torch.equal(ids[1:].long().sort(-1).values, tk.sort(-1).values)


def test_route_stats():
    E = 512
    logits = torch.randn(8, E + 8, device="cuda", generator=gen(5)).bfloat16()
    stats = torch.zeros(2, dtype=torch.int32, device="cuda")
    ids, _, _ = route(logits, E, stats)
    route(logits, E, stats)
    assert stats.tolist() == [2 * ids.unique().numel(), 2]


# ------------------------------------------------------------------------------------------------------------ kernels
def test_each_kernel_against_torch(small):
    wts, moe = small
    E, K, I = moe.E, moe.K, moe.I
    M = 5
    x = torch.randn(M, K, device="cuda", generator=gen(6)).bfloat16()
    ids = torch.stack([torch.randperm(E, device="cuda", generator=gen(8 + m))[:TOPK] for m in range(M)]).int()
    w = torch.rand(M, TOPK, device="cuda", generator=gen(9)).bfloat16().float()
    sg = torch.rand(M, device="cuda", generator=gen(10)).bfloat16().float()
    shared = torch.randn(M, K, device="cuda", generator=gen(11)).bfloat16()
    h, yp, y = experts(moe, x, ids, w, sg, shared)
    for p in range(M * TOPK):
        e = int(ids.view(-1)[p])
        wg = dequant_nvfp4(wts["w_gu"][e, :I], wts["sf_gu"][e, :I], wts["gs"][e, 0], torch.float32)
        wu = dequant_nvfp4(wts["w_gu"][e, I:], wts["sf_gu"][e, I:], wts["gs"][e, 1], torch.float32)
        wd = dequant_nvfp4(wts["w_dn"][e], wts["sf_dn"][e], wts["gs"][e, 2], torch.float32)
        xr = x[p // TOPK].float()
        assert rel(h[p], torch.nn.functional.silu(xr @ wg.T) * (xr @ wu.T)) < 5e-3
        assert rel(yp[p], h[p].float() @ wd.T) < 1e-5  # exact operands: only the fp32 order differs
    acc = torch.zeros(M, K, device="cuda")
    for k in range(TOPK):  # the combine's order, op for op (the build has no fma contraction)
        acc = acc + w[:, k:k + 1] * yp.view(M, TOPK, K)[:, k]
    assert torch.equal(y, (acc + sg[:, None] * shared.float()).bfloat16())
    _, _, y0 = experts(moe, x, ids, w, sg, None)
    assert torch.equal(y0, acc.bfloat16())


@pytest.mark.parametrize("M", [1, 7, 16])
def test_forward_against_reference(small, M):
    wts, moe = small
    x = torch.randn(M, moe.K, device="cuda", generator=gen(20 + M)).bfloat16()
    y = moe(x)
    logits = moe.last_routing(M)[0].clone()
    full = torch.cat([wts["router_w"], wts["shared_gate_w"]]).float()
    assert rel(logits[:, :moe.E + 1], x.float() @ full.T) < 3e-3
    assert rel(y, moe_reference(x, **ref_args(wts), logits=logits)) < 5e-3


def test_forward_wide_against_reference(wide):
    wts, moe = wide
    x = torch.randn(48, moe.K, device="cuda", generator=gen(30)).bfloat16()
    y = moe(x)
    assert rel(y, moe_reference(x, **ref_args(wts), logits=moe.last_routing(48)[0].clone())) < 5e-3


# ------------------------------------------------------------------------------------------------------------ row identity
def forced_ids(pattern, M, E, g):
    if pattern == "all":  # every row has the same 10 experts (in a different order per row)
        base = torch.randperm(E, device="cuda", generator=g)[:TOPK]
        return torch.stack([base[torch.randperm(TOPK, device="cuda", generator=g)] for _ in range(M)]).int()
    if pattern == "none":  # no expert in common
        return torch.randperm(E, device="cuda", generator=g)[:M * TOPK].view(M, TOPK).int()
    pool = torch.randperm(E, device="cuda", generator=g)[:20]  # partial overlap: 10 of 20 experts per row
    return torch.stack([pool[torch.randperm(20, device="cuda", generator=g)[:TOPK]] for _ in range(M)]).int()


@pytest.mark.parametrize("M", [1, 7, 8, 16, 48])
@pytest.mark.parametrize("pattern", ["all", "none", "partial"])
def test_row_identity_forced_routing(wide, M, pattern):
    _, moe = wide
    g = gen(M * 7 + len(pattern))
    x = torch.randn(M, moe.K, device="cuda", generator=g).bfloat16()
    ids = forced_ids(pattern, M, moe.E, g)
    w = torch.rand(M, TOPK, device="cuda", generator=g).bfloat16().float()
    sg = torch.rand(M, device="cuda", generator=g).bfloat16().float()
    shared = torch.randn(M, moe.K, device="cuda", generator=g).bfloat16()
    h, yp, y = experts(moe, x, ids, w, sg, shared)
    for m in range(M):  # each row alone
        h1, yp1, y1 = experts(moe, x[m:m + 1].contiguous(), ids[m:m + 1].contiguous(), w[m:m + 1].contiguous(), sg[m:m + 1].contiguous(),
                              shared[m:m + 1].contiguous())
        assert torch.equal(h1, h[m * TOPK:(m + 1) * TOPK]) and torch.equal(yp1, yp[m * TOPK:(m + 1) * TOPK])
        assert torch.equal(y1[0], y[m])
    gc = torch.Generator().manual_seed(M)
    for _ in range(3):  # random subsets in random order
        sub = torch.randperm(M, generator=gc)[:int(torch.randint(1, M + 1, (1,), generator=gc))].cuda()
        _, _, ys = experts(moe, x[sub].contiguous(), ids[sub].contiguous(), w[sub].contiguous(), sg[sub].contiguous(), shared[sub].contiguous())
        assert torch.equal(ys, y[sub])


@pytest.mark.parametrize("M", [1, 7, 8, 16])
def test_row_identity_forward(small, M):
    _, moe = small
    g = gen(40 + M)
    x = torch.randn(M, moe.K, device="cuda", generator=g).bfloat16()
    x[M // 2] = x[0]  # two rows with the same experts
    y = moe(x)
    for m in range(M):
        assert torch.equal(moe(x[m:m + 1])[0], y[m])
    sub = torch.randperm(M, device="cuda", generator=g)[:max(1, M // 2)]
    assert torch.equal(moe(x[sub]), y[sub])


def test_rows_beyond_max_rows(small):
    _, moe = small
    x = torch.randn(21, moe.K, device="cuda", generator=gen(50)).bfloat16()
    y = moe(x)
    assert torch.equal(moe(x[16:]), y[16:]) and torch.equal(moe(x[3:4]), y[3:4])


# ------------------------------------------------------------------------------------------------------------ skip, graph, shapes
def test_skip_flag_zero_output(small):
    _, moe = small
    x = torch.randn(4, moe.K, device="cuda", generator=gen(60)).bfloat16()
    ref = moe(x)
    flag = torch.ones(1, dtype=torch.int32, device="cuda")
    ops().skinny_skip(flag)
    try:
        y = moe(x)
        assert torch.equal(y, torch.zeros_like(y))
        flag.zero_()  # cleared at run time: the launches made while it was set read the device int
        assert torch.equal(moe(x), ref)
    finally:
        ops().skinny_skip(None)


@pytest.mark.skipif(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1", reason="graph capture needs the caching allocator (memcheck runs)")
def test_cuda_graph_replay_with_new_routing(small):
    from engine.model.fast import capture
    _, moe = small
    xs = torch.randn(8, moe.K, device="cuda", generator=gen(70)).bfloat16()
    graph, out = capture(lambda: moe(xs))
    routings = []
    for seed in (71, 72, 73):
        xn = torch.randn(8, moe.K, device="cuda", generator=gen(seed)).bfloat16()
        xs.copy_(xn)
        graph.replay()
        routings.append(moe.last_routing(8)[1].clone())
        assert torch.equal(out, moe(xn))
        assert torch.equal(routings[-1], moe.last_routing(8)[1])
    assert not torch.equal(routings[0], routings[1])  # the replays routed differently


def test_rejects_bad_shapes(small):
    _, moe = small
    K, I, E = moe.K, moe.I, moe.E
    x = torch.randn(2, K, device="cuda").bfloat16()
    ids = torch.zeros(2, TOPK, dtype=torch.int32, device="cuda")
    h = torch.empty(2 * TOPK, I, dtype=torch.bfloat16, device="cuda")
    yp = torch.empty(2 * TOPK, K, device="cuda")
    o = ops()
    with pytest.raises(RuntimeError):  # M > 48
        o.moe_gate_up(torch.randn(49, K, device="cuda").bfloat16(), torch.zeros(49, TOPK, dtype=torch.int32, device="cuda"), moe.w_gu,
                      moe.sf_gu, moe.gs, torch.empty(490, I, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(RuntimeError):  # ids int64
        o.moe_gate_up(x, ids.long(), moe.w_gu, moe.sf_gu, moe.gs, h)
    with pytest.raises(RuntimeError):  # K % 512
        o.moe_gate_up(x[:, :768].contiguous(), ids, moe.w_gu[:, :, :384].contiguous(), moe.sf_gu, moe.gs, h)
    with pytest.raises(RuntimeError):  # h shape
        o.moe_gate_up(x, ids, moe.w_gu, moe.sf_gu, moe.gs, h[:10])
    with pytest.raises(RuntimeError):  # row-major (unswizzled) scale shape is accepted only with the right byte count
        o.moe_gate_up(x, ids, moe.w_gu, moe.sf_gu[:, :100], moe.gs, h)
    with pytest.raises(RuntimeError):  # gs [E, 3]
        o.moe_down(h, ids, moe.w_dn, moe.sf_dn, moe.gs[:, :2].contiguous(), yp)
    with pytest.raises(RuntimeError):  # I = 192 has no down kernel
        o.moe_down(torch.empty(20, 192, dtype=torch.bfloat16, device="cuda"), ids, torch.empty(E, K, 96, dtype=torch.uint8, device="cuda"),
                   torch.empty(E, K * 12, dtype=torch.uint8, device="cuda"), moe.gs, yp)
    with pytest.raises(RuntimeError):  # E > 512
        route(torch.randn(2, 600, device="cuda").bfloat16(), 513)
    with pytest.raises(RuntimeError):  # NL >= E + 1
        route(torch.randn(2, 16, device="cuda").bfloat16(), 16)
    with pytest.raises(RuntimeError):  # w shape in the combine
        o.moe_combine(yp, torch.empty(2, 9, device="cuda"), torch.empty(2, device="cuda"), None, x.clone())


# ------------------------------------------------------------------------------------------------------------ real shape
@pytest.fixture(scope="module")
def real():
    """One layer at the real shape: E = 512, K = 2560, I = 640 (1.4 GB of experts)."""
    wts = random_moe_weights(512, 2560, 640, generator=gen(100))
    moe = KernelMoE(**wts)
    yield wts, moe
    del moe, wts
    torch.cuda.empty_cache()


def test_real_shape(real):
    wts, moe = real
    x = torch.randn(8, 2560, device="cuda", generator=gen(101)).bfloat16()
    y = moe(x)
    logits = moe.last_routing(8)[0].clone()
    assert rel(y, moe_reference(x, **ref_args(wts), logits=logits)) < 5e-3
    for m in (0, 5):
        assert torch.equal(moe(x[m:m + 1])[0], y[m])


def test_selftest_entries():
    from engine.model.moe_kernels import moe_selftest, skinny_bf16_selftest
    g = torch.Generator(device="cuda").manual_seed(1234)
    res = {**skinny_bf16_selftest(ops(), g), **moe_selftest(ops(), g)}
    assert all(v < 5e-3 for v in res.values()), res
