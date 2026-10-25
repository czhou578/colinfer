"""Hyper-connection kernels (csrc/hc.cu, the grouped form of csrc/norm.cu) and KernelGatedResidual
(engine/model/hc_kernels.py) against gated_residual_reference (transformers' Qwen4ExpTextGatedResidual op for op, fp32 math,
bf16 rounding points, exactly rounded matmuls) and against the transformers module itself in bf16 and in fp32. Rows are
bit-identical for any M (decode = verify). Run: uv run pytest tests/test_hc.py -q"""
import pytest
import torch

from engine.model import hc_kernels as hk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

N, H, R, EPS = 10240, 2560, 320, 1e-6


def ulps(a, b):
    """the bf16 ulp distance of each element (both bf16)"""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7fff), ia)
    ib = torch.where(ib < 0, -(ib & 0x7fff), ib)
    return (ia - ib).abs()


def weights(seed, inj=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s, sc=1.0: (torch.randn(*s, device="cuda", generator=g) * sc).bfloat16()  # noqa: E731
    return dict(norm_w=rnd(N, sc=0.2), w_down=rnd(R, N, sc=0.01), w_inj=rnd(4, N, sc=0.01) if inj else None, w_up=rnd(N, R, sc=0.05))


def x4_rows(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, N, device="cuda", generator=g)
    return (x * torch.rand(M, 4, 1, device="cuda", generator=g).repeat_interleave(H, 1).view(M, N) * 4).bfloat16()  # streams of different scale


def norm_reference(x4, w):
    M = x4.shape[0]
    g = x4.float().view(M, 4, H)
    return (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + EPS)).view(M, -1).mul(1.0 + w.float()).bfloat16()


def test_hc_norm():
    """grouped zero-centered RMSNorm: <= 1 ulp (the sum of squares is reduced in another order than torch's mean)"""
    from engine.kernels import ops
    w = weights(1)["norm_w"]
    x4 = x4_rows(16, 1)
    n = torch.empty_like(x4)
    ops().hc_norm(x4, w, EPS, n)
    u = ulps(n, norm_reference(x4, w))
    assert u.max().item() <= 1 and (u > 0).float().mean().item() < 1e-3, ((u > 0).sum().item(), u.max().item())
    # groups = 1 is the plain rmsnorm, bit for bit
    ref = torch.empty_like(x4)
    ops().rmsnorm(x4, w, EPS, ref)
    ops().hc_norm(x4, w, EPS, n, groups=1)
    assert torch.equal(n, ref)


def assert_y_close(y, y_ref, p, frac):
    """y equals the reference except where u (the K = 320 product: fp32 accumulation in another order than the exact
    reference) rounds the other way. Then one p_s moves by 1 ulp, and y (the mean of 4 terms, which can cancel) moves by at
    most 1 ulp of its largest term: |y - y_ref| <= 2^-8 max_s |p_s| (bf16 has 8 significant bits)."""
    bad = y != y_ref
    assert bad.float().mean().item() <= frac, bad.float().mean().item()
    bound = p.abs().amax(1) * 2.0 ** -8
    assert ((y.float() - y_ref.float()).abs() <= bound).all()


@pytest.mark.parametrize("M", [1, 3, 8, 16])
def test_hc_mix_matches_reference(M):
    """hc_mix from the same n, d and e: inj bit for bit; y bit for bit except where u rounds the other way (assert_y_close)"""
    from engine.kernels import ops
    W = weights(M)
    x4 = x4_rows(M, M)
    parts = {}
    y_ref, inj_ref = hk.gated_residual_reference(x4, W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS, parts)
    n = parts["n"]
    de = (n.double() @ torch.cat([W["w_down"], W["w_inj"]]).double().T).bfloat16()  # [M, 324], exactly rounded
    y = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    inj = torch.empty(M, 4, device="cuda", dtype=torch.bfloat16)
    ops().hc_mix(de[:, :R], W["w_up"], n, y, de[:, R:], inj)
    assert torch.equal(inj, inj_ref)
    assert_y_close(y, y_ref, parts["p"], 2e-3)


def test_hc_post_bit_exact():
    from engine.kernels import ops
    g = torch.Generator(device="cuda").manual_seed(5)
    for M in (1, 7, 16):
        x4 = torch.randn(M, N, device="cuda", generator=g).bfloat16()
        o = torch.randn(M, H, device="cuda", generator=g).bfloat16()
        inj = (torch.rand(M, 4, device="cuda", generator=g) * 2).bfloat16()
        want = hk.post_reference(o, inj, x4)
        ops().hc_post(o, inj, x4)
        assert torch.equal(x4, want)


def test_post_matches_transformers_expression():
    g = torch.Generator(device="cuda").manual_seed(6)
    M = 5
    x4 = torch.randn(M, N, device="cuda", generator=g).bfloat16()
    o = torch.randn(M, H, device="cuda", generator=g).bfloat16()
    inj = (torch.rand(M, 4, device="cuda", generator=g) * 2).bfloat16()
    injection = o.unsqueeze(-2) * inj.unsqueeze(-1)  # transformers' decoder layer, in bf16
    assert torch.equal(x4 + injection.flatten(-2), hk.post_reference(o, inj, x4))


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("inj", [True, False])
def test_kernel_gated_residual_matches_reference(inj, seed):
    """the whole module (hc_norm, bf16_gemv down | inject, hc_mix) against gated_residual_reference: inj within 1 ulp, y as
    in assert_y_close (hc_norm and the products may also round the other way)"""
    W = weights(seed, inj)
    gr = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)
    x4 = x4_rows(16, seed)
    got = gr(x4)
    parts = {}
    want = hk.gated_residual_reference(x4, W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS, parts)
    if inj:
        (y, i), (y_ref, i_ref) = got, want
        assert ulps(i, i_ref).max().item() <= 1
    else:
        y, y_ref = got, want
    assert_y_close(y, y_ref, parts["p"], 1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_transformers_module(dtype):
    """KernelGatedResidual against transformers' Qwen4ExpTextGatedResidual run in bf16 (cuBLAS matmuls round the split-K
    partials in bf16, so a few ulps) and in fp32 (no bf16 rounding points: a relative error of a few bf16 ulps)."""
    from transformers.models.qwen4_exp import configuration_qwen4_exp as c
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    W = weights(21)
    cfg = c.Qwen4ExpTextConfig(hidden_size=H, hc_count=4, hc_lowrank=R, rms_norm_eps=EPS)
    mod = m.Qwen4ExpTextGatedResidual(cfg).cuda()
    with torch.no_grad():
        mod.hc_norm.weight.copy_(W["norm_w"].float())
        mod.input_mix_weight_down.weight.copy_(W["w_down"].float())
        mod.input_mix_weight_up.weight.copy_(W["w_up"].float())
        mod.block_inject_weight.weight.copy_(W["w_inj"].float())
    mod = mod.to(dtype)
    x4 = x4_rows(8, 21)
    with torch.no_grad():
        y_hf, _, inj_hf = mod(x4.to(dtype))
    y, inj = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)(x4)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    assert rel(y, y_hf) < 1e-2 and rel(inj, inj_hf) < 1e-2, (rel(y, y_hf), rel(inj, inj_hf))


def test_rows_bit_identical_across_M():
    """Row i of a 16-row call == the same row in a call of M rows, for M = 1 .. 16 (decode = verify)."""
    W = weights(31)
    gr = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)
    x4 = x4_rows(16, 31)
    o = torch.randn(16, H, device="cuda").bfloat16()
    y16, inj16 = gr(x4)
    post16 = gr.post(o, inj16, x4.clone())
    for M in range(1, 17):
        for lo in sorted({0, 16 - M}):
            y, inj = gr(x4[lo:lo + M].contiguous())
            assert torch.equal(y, y16[lo:lo + M]) and torch.equal(inj, inj16[lo:lo + M]), (M, lo)
            post = gr.post(o[lo:lo + M].contiguous(), inj, x4[lo:lo + M].clone())
            assert torch.equal(post, post16[lo:lo + M]), (M, lo)


def test_selftest():
    from engine.kernels import ops
    res = hk.hc_selftest(ops(), torch.Generator(device="cuda").manual_seed(1))
    assert all(v < 5e-3 for v in res.values()), res


# ------------------------------------------------------------------------------------------------------------ decode copies
def int8_rows(w):
    """The INT8 W_up copy of tools/requant_qwen4exp.py: (q int8, s fp32 [N]), W_up ~ q * s per row."""
    from tools.requant_qwen4exp import quantize_group
    t, _ = quantize_group({"m": w.float()}, "int8")
    return t["m.qweight"].cuda(), t["m.weight_scale"].cuda()


def copies(W, inj=True):
    """The KernelGatedResidual of W with its decode copies attached as attach_decode_copies4 attaches them: INT6 down |
    inject (+ 4 zero rows: 328 rows) and the INT8 W_up."""
    from engine.model.fast_qwen4exp import _copy_of
    from tools.requant_qwen4exp import quantize_group
    gr = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)
    ws = {"down": W["w_down"].float()} | ({"inj": W["w_inj"].float()} if inj else {})
    t, _ = quantize_group(ws, "int6")
    dec = _copy_of({k: v.cuda() for k, v in t.items()}, list(ws), pad_to=-(-gr.w_di.shape[0] // 8) * 8)
    q, s = int8_rows(W["w_up"])
    gr.set_decode_copies(dec=dec, w_up_q=q, w_up_s=s)
    return gr


@pytest.mark.parametrize("M", [1, 3, 8, 16])
def test_hc_mix_int8_matches_dequantized_reference(M):
    """The INT8 mode of hc_mix from the same n, d and e against the reference on the dequantized W_up (q * s, exactly
    rounded products): inj bit for bit; y as in assert_y_close (u = bf16(s * sum a q): one rounding, fp32 sum)."""
    from engine.kernels import ops
    W = weights(40 + M)
    q, s = int8_rows(W["w_up"])
    w_deq = q.double() * s.double()[:, None]
    x4 = x4_rows(M, M)
    n = norm_reference(x4, W["norm_w"])
    de = (n.double() @ torch.cat([W["w_down"], W["w_inj"]]).double().T).bfloat16()
    a = hk._bf(torch.nn.functional.silu(hk._bf(de[:, :R].float() / 4).float()))
    u = hk._bf(a.double() @ w_deq.T)
    p = hk._bf(hk._bf(torch.sigmoid(u.float())).float() * n.float()).float().view(M, 4, H)
    y_ref = hk._bf((((p[:, 0] + p[:, 1]) + p[:, 2]) + p[:, 3]) / 4)
    inj_ref = hk._bf(2 * hk._bf(torch.sigmoid(hk._bf(de[:, R:].float() / 4).float())).float())
    y = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    inj = torch.empty(M, 4, device="cuda", dtype=torch.bfloat16)
    ops().hc_mix(de[:, :R], q, n, y, de[:, R:], inj, s)
    assert torch.equal(inj, inj_ref)
    assert_y_close(y, y_ref, p, 2e-3)


def test_copies_against_bf16_path():
    """The gated residual with its copies (INT6 down | inject, INT8 W_up) against the BF16 path: y within 1% and inj
    within 0.5% relative (measured 0.32% and 0.34%; the INT6 weight error is about 2.2%, and the sigmoids damp it), and
    the final mixer's form (no inject) the same."""
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()  # noqa: E731
    for inj in (True, False):
        W = weights(50, inj)
        x4 = x4_rows(16, 50)
        want = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)(x4)
        got = copies(W, inj)(x4)
        if inj:
            (y, i), (y_ref, i_ref) = got, want
            assert rel(i, i_ref) < 5e-3, rel(i, i_ref)
        else:
            y, y_ref = got, want
        assert rel(y, y_ref) < 1e-2, rel(y, y_ref)
        print(f"\n[hc copies] inject {inj}: y rel {rel(y, y_ref):.2e}" + (f", inj rel {rel(i, i_ref):.2e}" if inj else ""))


@pytest.mark.parametrize("inj", [True, False])
def test_copies_rows_bit_identical_across_M(inj):
    """With the copies: row i of a 16-row call == the same row in a call of M rows, M = 1 .. 16 (decode = verify)."""
    W = weights(61, inj)
    gr = copies(W, inj)
    assert gr.dec.out_features == (328 if inj else 320)
    x4 = x4_rows(16, 61)
    full = gr(x4)
    for M in range(1, 17):
        for lo in sorted({0, 16 - M}):
            got = gr(x4[lo:lo + M].contiguous())
            if inj:
                assert torch.equal(got[0], full[0][lo:lo + M]) and torch.equal(got[1], full[1][lo:lo + M]), (M, lo)
            else:
                assert torch.equal(got, full[lo:lo + M]), (M, lo)


def test_set_decode_copies_refuses_wrong_shapes():
    W = weights(70)
    gr = hk.KernelGatedResidual(W["norm_w"], W["w_down"], W["w_inj"], W["w_up"], EPS)
    q, s = int8_rows(W["w_up"])
    with pytest.raises(ValueError, match="INT8 W_up"):
        gr.set_decode_copies(w_up_q=q[:, :160].contiguous(), w_up_s=s)
    with pytest.raises(ValueError, match="INT8 W_up"):
        gr.set_decode_copies(w_up_q=q.float(), w_up_s=s)
