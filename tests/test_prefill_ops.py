"""Fused prefill glue (csrc/prefill_ops.cu) against PyTorch. Run: uv run pytest tests/test_prefill_ops.py -q"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.weights.quant_emul import fake_quant_nvfp4_unscaled  # noqa: E402
from tests.test_gemm_nvfp4 import unswizzle  # noqa: E402
from engine.weights.loader import E2M1_LUT  # noqa: E402


def ops():
    from engine.kernels import ops as o
    return o()


def test_fp8_quant():
    x = (torch.randn(300, 5120, device="cuda") * 3).bfloat16()
    x[0, 0] = 1e5
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device="cuda")
    ops().fp8_quant(x, 0.02, out)
    assert torch.equal(out.float(), (x.float() / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn).float())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_silu_mul_quant(seed):
    torch.manual_seed(seed)
    M, I = 257, 17408
    gu = (torch.randn(M, 2 * I, device="cuda") * 2).bfloat16()
    h = F.silu(gu[:, :I]) * gu[:, I:]
    s = float(h.float().abs().max()) / (6 * 448) * 0.7
    q = torch.empty(M, I // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(ops().nvfp4_sf_size(M, I), dtype=torch.uint8, device="cuda")
    ops().silu_mul_quant(gu, s, q, sf)
    lut = E2M1_LUT.cuda()
    got = torch.stack([lut[(q & 15).long()], lut[(q >> 4).long()]], -1).reshape(M, I) * unswizzle(sf, M, I).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    assert torch.equal(got, fake_quant_nvfp4_unscaled(h, s).float())


@pytest.mark.parametrize("T", [1, 2, 3, 100])
def test_causal_conv_silu(T):
    C = 10240
    x = torch.randn(T, C, device="cuda").bfloat16()
    st = torch.randn(C, 3, device="cuda").bfloat16()
    w = torch.randn(C, 1, 4, device="cuda").bfloat16()
    outs = [torch.empty(T, n, device="cuda").bfloat16() for n in (2048, 2048, 6144)]
    ops().causal_conv_silu(x, st, w, outs)
    xin = torch.cat([st, x.t()], -1)
    ref = F.silu(F.conv1d(xin[None], w, None, padding=0, groups=C)[0]).t()
    assert torch.equal(torch.cat(outs, -1), ref.contiguous())
    # strided input rows (a column slice of a wider GEMM output)
    wide = torch.randn(T, C + 6144, device="cuda").bfloat16()
    ops().causal_conv_silu(wide[:, :C], st, w, outs)
    ref2 = F.silu(F.conv1d(torch.cat([st, wide[:, :C].t()], -1)[None], w, None, padding=0, groups=C)[0]).t()
    assert torch.equal(torch.cat(outs, -1), ref2.contiguous())


def test_gated_rmsnorm():
    from engine.model.qwen35 import RMSNormGated
    n = RMSNormGated(128, 1e-6).cuda().bfloat16()
    with torch.no_grad():
        n.weight.normal_(1, 0.2)
    o = torch.randn(4096, 128, device="cuda").bfloat16()
    z = torch.randn(4096, 128, device="cuda").bfloat16()
    out = torch.empty_like(o)
    ops().gated_rmsnorm(o, z, n.weight.data, 1e-6, out)
    assert torch.equal(out, n(o, z))


@pytest.mark.parametrize("with_y", [False, True])
def test_add_rmsnorm_outputs(with_y):
    from engine.model.qwen35 import RMSNorm
    torch.manual_seed(7)
    M, K = 300, 5120
    n = RMSNorm(K, 1e-6).cuda().bfloat16()
    with torch.no_grad():
        n.weight.normal_(0, 0.2)
    x = torch.randn(M, K, device="cuda").bfloat16()
    y = torch.randn(M, K, device="cuda").bfloat16() if with_y else None
    xs = x + y if with_y else x
    ref_n = n(xs)
    x_out = torch.empty_like(x)
    n_out = torch.empty_like(x)
    q4 = torch.empty(M, K // 2, dtype=torch.uint8, device="cuda")
    sf4 = torch.empty(ops().nvfp4_sf_size(M, K), dtype=torch.uint8, device="cuda")
    q8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    s4, s8 = 0.004, 0.02
    ops().add_rmsnorm(x, y, n.weight.data, 1e-6, x_out=x_out if with_y else None, n_out=n_out, q4=q4, sf4=sf4, in_scale4=s4, q8=q8, in_scale8=s8)
    if with_y:
        assert torch.equal(x_out, xs)
    # reduction order differs from torch.mean: rare 1-ulp differences in bf16, nothing more
    diff = (n_out.float() - ref_n.float()).abs()
    ulp = ref_n.float().abs().clamp_min(1e-30) * 2 ** -7
    assert bool((diff <= ulp).all()) and (diff > 0).float().mean().item() < 1e-3
    ref_n = n_out  # the quantized outputs must match the kernel's own normed values exactly
    assert torch.equal(q8.float(), (ref_n.float() / s8).clamp(-448, 448).to(torch.float8_e4m3fn).float())
    lut = E2M1_LUT.cuda()
    got = torch.stack([lut[(q4 & 15).long()], lut[(q4 >> 4).long()]], -1).reshape(M, K) * unswizzle(sf4, M, K).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    assert torch.equal(got, fake_quant_nvfp4_unscaled(ref_n, s4).float())


def test_gate_fp8():
    T, H, D = 77, 24, 256
    o = torch.randn(T, H * D, device="cuda").bfloat16()
    qp = torch.randn(T, H * 2 * D + 2048, device="cuda").bfloat16()[:, : H * 2 * D]  # strided rows
    out = torch.empty(T, H * D, dtype=torch.float8_e4m3fn, device="cuda")
    ops().gate_fp8(o, qp, D, 0.05, out)
    gate = qp.reshape(T, H, 2 * D)[:, :, D:].reshape(T, -1)
    ref = ((o * torch.sigmoid(gate)) .float() / 0.05).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(out.float(), ref.float())


@pytest.mark.parametrize("T", [1, 5, 300])
def test_causal_conv_silu_l2norm(T):
    """The fused per-head L2 norm of q and k matches FLA's l2norm applied to the plain conv outputs."""
    from fla.modules.l2norm import l2norm_fwd

    from engine.kernels import ops
    torch.manual_seed(T)
    c1, c2, C = 256, 512, 896
    x = torch.randn(T, C, device="cuda").bfloat16()
    st = torch.randn(C, 3, device="cuda").bfloat16()
    w = torch.randn(C, 4, device="cuda").bfloat16() * 0.5
    plain = [torch.empty(T, n, device="cuda", dtype=torch.bfloat16) for n in (c1, c2 - c1, C - c2)]
    fused = [torch.empty_like(t) for t in plain]
    ops().causal_conv_silu(x, st, w, plain)
    ops().causal_conv_silu(x, st, w, fused, 1e-6)
    for p, f in zip(plain[:2], fused[:2]):
        ref = l2norm_fwd(p.view(T, -1, 128), eps=1e-6)[0].view(T, -1).float()
        assert (f.float() - ref).abs().max().item() < 1e-2 * ref.abs().max().item()
    assert torch.equal(plain[2], fused[2])
