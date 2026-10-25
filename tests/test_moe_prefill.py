"""The prefill MoE (csrc/moe_prefill.cu, KernelMoE.prefill): the pairs sorted by expert through two CUTLASS SM120 grouped
NVFP4 GEMMs. The tests check, with group sizes 0, 1, 127, 128, 129, 300 and random:
- the gathered and quantized rows and their per-group swizzled scales, byte for byte against ops().nvfp4_quant
- each grouped GEMM against an emulation in torch (quant_emul's activation quantization, the loader's dequantization)
- the SwiGLU quantization against the emulation, and the combine op for op
- the full prefill layer against the emulation, and that two runs are equal
Run: uv run pytest tests/test_moe_prefill.py -q"""
import pytest
import torch

from engine.model.moe_kernels import KernelMoE, prefill_plan, random_moe_weights
from engine.weights.loader import dequant_nvfp4
from engine.weights.quant_emul import fake_quant_nvfp4_unscaled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

TOPK = 10
SIZES = [0, 1, 127, 128, 129, 300]  # group sizes; the other experts of the layer take the rest of the pairs


def ops():
    from engine.kernels import ops as o
    return o()


def rel(out, ref):
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def ids_with_sizes(T, E, sizes, seed):
    """ids int32 [T, 10]: experts 0.. have the given pair counts, the others share the rest (each <= T). Each token has 10
    distinct experts: expert runs of length <= T, laid along the tokens cyclically, never meet themselves in a token."""
    gc = torch.Generator().manual_seed(seed)
    P = T * TOPK
    rest = P - sum(sizes)
    k = E - len(sizes)
    counts = [rest // k + (1 if i < rest % k else 0) for i in range(k)]
    assert max(counts) <= T and max(sizes) <= T
    seq = torch.repeat_interleave(torch.arange(E), torch.tensor(sizes + counts))
    ids = torch.empty(T, TOPK, dtype=torch.long)
    slot = torch.zeros(T, dtype=torch.long)
    for j, e in enumerate(seq.tolist()):
        t = j % T
        ids[t, slot[t]] = e
        slot[t] += 1
    perm = torch.argsort(torch.rand(E, generator=gc))  # experts in random order, so that the sizes are not in id order
    ids = perm[ids]
    for t in range(T):  # random slot order within each token
        ids[t] = ids[t, torch.randperm(TOPK, generator=gc)]
    return ids.int().cuda()


def unswizzle(sf, R, K):
    """swizzled scale bytes (CUTLASS 128x4 layout) -> [R, K/16] row-major"""
    r = torch.arange(R, device=sf.device)[:, None]
    kb = torch.arange(K // 16, device=sf.device)[None, :]
    kb4 = (K // 16 + 3) // 4
    idx = ((r // 128) * kb4 + kb // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + kb % 4
    return sf.view(-1)[idx]


def pieces(moe, x, ids, w, sg, shared):
    """The prefill steps with given routing: (plan, a, sfa, d1, a2, sfa2, d2, y)."""
    o, E, K, I = ops(), moe.E, moe.K, moe.I
    T = x.shape[0]
    P = T * TOPK
    order, off, sfo, inv = prefill_plan(ids, E)
    rows = o.moe_sf_rows(P, E)
    a = torch.empty(P, K // 2, dtype=torch.uint8, device="cuda")
    sfa = torch.zeros(rows * K // 16, dtype=torch.uint8, device="cuda")
    o.moe_gather_quant(x, ids, order, off, sfo, moe.in_scale, a, sfa)
    d1 = torch.empty(P, 2 * I, dtype=torch.bfloat16, device="cuda")
    o.moe_grouped_gemm(0, a, sfa, moe.w_gu, moe.sf_gu, d1, off, sfo, moe.in_scale, moe.gs)
    a2 = torch.empty(P, I // 2, dtype=torch.uint8, device="cuda")
    sfa2 = torch.zeros(rows * I // 16, dtype=torch.uint8, device="cuda")
    o.moe_swiglu_quant(d1, ids, order, off, sfo, moe.in_scale, a2, sfa2)
    d2 = torch.empty(P, K, dtype=torch.bfloat16, device="cuda")
    o.moe_grouped_gemm(1, a2, sfa2, moe.w_dn, moe.sf_dn, d2, off, sfo, moe.in_scale, moe.gs)
    y = torch.empty(T, K, dtype=torch.bfloat16, device="cuda")
    o.moe_combine_prefill(d2, inv, w, sg, shared, y)
    return (order, off, sfo, inv), a, sfa, d1, a2, sfa2, d2, y


def weights(wts, e, I):
    wgu = dequant_nvfp4(wts["w_gu"][e], wts["sf_gu"][e], torch.tensor(1.0), torch.float32)  # unscaled: alpha after the product
    wd = dequant_nvfp4(wts["w_dn"][e], wts["sf_dn"][e], torch.tensor(1.0), torch.float32)
    return wgu, wd


def routing(T, g):
    w = torch.rand(T, TOPK, device="cuda", generator=g).bfloat16().float()
    sg = torch.rand(T, device="cuda", generator=g).bfloat16().float()
    return w, sg


@pytest.fixture(scope="module")
def layer():
    """E = 16, K = 1024, I = 256"""
    wts = random_moe_weights(16, 1024, 256, generator=gen(1))
    return wts, KernelMoE(**wts)


@pytest.mark.parametrize("seed", [0, 1])
def test_groups_against_emulation(layer, seed):
    wts, moe = layer
    E, K, I = moe.E, moe.K, moe.I
    T = 300
    g = gen(10 + seed)
    x = torch.randn(T, K, device="cuda", generator=g).bfloat16()
    sizes = SIZES if seed == 0 else [int(v) for v in torch.randint(0, T + 1, (6,), generator=torch.Generator().manual_seed(seed))]
    ids = ids_with_sizes(T, E, sizes, seed)
    w, sg = routing(T, g)
    shared = torch.randn(T, K, device="cuda", generator=g).bfloat16()
    (order, off, sfo, inv), a, sfa, d1, a2, sfa2, d2, y = pieces(moe, x, ids, w, sg, shared)
    counts = (off[1:] - off[:-1]).tolist()
    assert all(counts.count(c) >= sizes.count(c) for c in sizes) and sum(counts) == T * TOPK  # the group sizes under test occur
    inS = wts["in_scale"]
    agree, total = 0, 0
    for e in range(E):
        lo, hi, m = off[e].item(), off[e + 1].item(), counts[e]
        if m == 0:
            continue
        toks = order[lo:hi].long() // TOPK
        xe = x[toks].contiguous()
        # the gathered rows and scales, byte for byte against the dense quantizer
        q_ref = torch.empty(m, K // 2, dtype=torch.uint8, device="cuda")
        sf_ref = torch.empty(ops().nvfp4_sf_size(m, K), dtype=torch.uint8, device="cuda")
        ops().nvfp4_quant(xe, inS[e, 0].item(), q_ref, sf_ref)
        assert torch.equal(a[lo:hi], q_ref)
        base = sfo[e].item() * K // 16
        assert torch.equal(unswizzle(sfa[base:], m, K), unswizzle(sf_ref, m, K))
        # gate | up: alpha * (fake-quantized x) . W^T
        wgu, wd = weights(wts, e, I)
        alpha = inS[e, 0] * wts["gs"][e, 0]
        ref1 = fake_quant_nvfp4_unscaled(xe, inS[e, 0].item()).float() @ wgu.T * alpha
        assert rel(d1[lo:hi], ref1) < 1e-2, (e, m)
        # silu(g) * u quantized: the emulation on the kernel's d1
        g1, u1 = d1[lo:hi, :I].float(), d1[lo:hi, I:].float()
        h = torch.nn.functional.silu(g1) * u1
        hq_ref = fake_quant_nvfp4_unscaled(h, inS[e, 2].item())
        hq = dequant_nvfp4(a2[lo:hi], unswizzle(sfa2[sfo[e].item() * I // 16:], m, I).view(torch.float8_e4m3fn), torch.tensor(1.0), torch.float32)
        agree += (hq == hq_ref).sum().item()
        total += hq.numel()
        # down: alpha * (the kernel's quantized h) . W_down^T
        ref2 = hq @ wd.T * (inS[e, 2] * wts["gs"][e, 2])
        assert rel(d2[lo:hi], ref2) < 1e-2, (e, m)
    assert agree / total > 0.97, agree / total
    # the combine, op for op in slot order
    acc = torch.zeros(T, K, device="cuda")
    rows = d2[inv.long()].view(T, TOPK, K).float()
    for k in range(TOPK):
        acc = acc + w[:, k:k + 1] * rows[:, k]
    assert torch.equal(y, (acc + sg[:, None] * shared.float()).bfloat16())


def test_full_prefill_against_emulation_and_deterministic(layer):
    wts, moe = layer
    E, K, I = moe.E, moe.K, moe.I
    T = 257
    x = torch.randn(T, K, device="cuda", generator=gen(20)).bfloat16()
    y = moe.prefill(x)
    assert torch.equal(moe.prefill(x), y)
    # the emulation of the whole layer: the same routing (the router product and moe_route), then per expert
    logits = x @ moe.router.T
    ids = torch.empty(T, TOPK, dtype=torch.int32, device="cuda")
    w = torch.empty(T, TOPK, device="cuda")
    sg = torch.empty(T, device="cuda")
    ops().moe_route(logits, E, ids, w, sg)
    inS = wts["in_scale"]
    out = torch.zeros(T, K, device="cuda")
    for e in ids.unique().tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        wgu, wd = weights(wts, e, I)
        d1 = (fake_quant_nvfp4_unscaled(x[rows], inS[e, 0].item()).float() @ wgu.T * (inS[e, 0] * wts["gs"][e, 0])).bfloat16().float()
        h = torch.nn.functional.silu(d1[:, :I]) * d1[:, I:]
        d2 = (fake_quant_nvfp4_unscaled(h, inS[e, 2].item()) @ wd.T * (inS[e, 2] * wts["gs"][e, 2])).bfloat16().float()
        out.index_add_(0, rows, d2 * w[rows, slots, None])
    shared = (torch.nn.functional.silu(x @ moe.sh_gate.T) * (x @ moe.sh_up.T)) @ moe.sh_down.T
    assert rel(y, out + sg[:, None] * shared.float()) < 2e-2
    # and close to the decode path (bf16 activations): the NVFP4 activation quantization is the difference
    assert rel(y, moe(x)) < 5e-2


def test_rejects_bad_shapes(layer):
    _, moe = layer
    o = ops()
    K, I, E = moe.K, moe.I, moe.E
    P = 20
    a = torch.empty(P, K // 2, dtype=torch.uint8, device="cuda")
    sfa = torch.empty(o.moe_sf_rows(P, E) * K // 16, dtype=torch.uint8, device="cuda")
    off = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    d1 = torch.empty(P, 2 * I, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError):  # which
        o.moe_grouped_gemm(2, a, sfa, moe.w_gu, moe.sf_gu, d1, off, off, moe.in_scale, moe.gs)
    with pytest.raises(RuntimeError):  # scale buffer too small
        o.moe_grouped_gemm(0, a, sfa[:100], moe.w_gu, moe.sf_gu, d1, off, off, moe.in_scale, moe.gs)
    with pytest.raises(RuntimeError):  # d shape
        o.moe_grouped_gemm(0, a, sfa, moe.w_gu, moe.sf_gu, d1[:, :I], off, off, moe.in_scale, moe.gs)
    with pytest.raises(RuntimeError):  # off [E + 1]
        o.moe_grouped_gemm(0, a, sfa, moe.w_gu, moe.sf_gu, d1, off[:E], off, moe.in_scale, moe.gs)


@pytest.fixture(scope="module")
def real():
    """One layer at the real shape: E = 512, K = 2560, I = 640 (1.4 GB of experts)."""
    wts = random_moe_weights(512, 2560, 640, generator=gen(100))
    moe = KernelMoE(**wts)
    yield wts, moe
    del moe, wts
    torch.cuda.empty_cache()


def test_real_shape_prefill(real):
    wts, moe = real
    x = torch.randn(512, 2560, device="cuda", generator=gen(101)).bfloat16()
    y = moe.prefill(x)
    assert torch.equal(moe.prefill(x), y)
    assert rel(y, moe(x)) < 5e-2
