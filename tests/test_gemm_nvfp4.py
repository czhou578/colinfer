"""Prefill NVFP4 GEMM (csrc/gemm_nvfp4.cu): activation quantizer vs the bit-exact emulation, and the
CUTLASS GEMM vs an fp32 reference on dequantized operands. Run: uv run pytest tests/test_gemm_nvfp4.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.weights.loader import E2M1_LUT, dequant_nvfp4  # noqa: E402
from engine.weights.quant_emul import fake_quant_nvfp4_unscaled  # noqa: E402


def ops():
    from engine.kernels import ops as o
    return o()


def unswizzle(sf, R, K):
    """swizzled CUTLASS 128x4 layout -> row-major [R, K/16]"""
    KB = K // 16
    kb4 = (KB + 3) // 4
    r = torch.arange(R, device=sf.device)[:, None]
    kb = torch.arange(KB, device=sf.device)[None, :]
    off = ((r // 128) * kb4 + kb // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + kb % 4
    return sf.view(-1)[off]


def quant(x, s):
    M, K = x.shape
    q = torch.empty(M, K // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(ops().nvfp4_sf_size(M, K), dtype=torch.uint8, device="cuda")
    ops().nvfp4_quant(x, s, q, sf)
    return q, sf


@pytest.mark.parametrize("M,K", [(1, 5120), (300, 5120), (2048, 17408)])
def test_quant_matches_emulation(M, K):
    torch.manual_seed(M)
    x = torch.randn(M, K, device="cuda")
    x[torch.rand_like(x) < 0.001] *= 40
    x = x.bfloat16()
    s = float(x.float().abs().max()) / (6 * 448) * 0.6
    q, sf = quant(x, s)
    sfr = unswizzle(sf, M, K).view(torch.float8_e4m3fn).float()
    lut = E2M1_LUT.cuda()
    got = torch.stack([lut[(q & 15).long()], lut[(q >> 4).long()]], -1).reshape(M, K) * sfr.repeat_interleave(16, 1)
    assert torch.equal(got, fake_quant_nvfp4_unscaled(x, s).float())


@pytest.mark.parametrize("M,N,K", [(2048, 17408, 5120), (2048, 5120, 17408), (129, 5120, 5120), (17, 1024, 5120)])
@pytest.mark.parametrize("tile", [0, 1])
def test_gemm_matches_reference(M, N, K, tile):
    torch.manual_seed(M + N + tile)
    x = torch.randn(M, K, device="cuda").bfloat16()
    s_in = float(x.float().abs().max()) / (6 * 448)
    a, sfa = quant(x, s_in)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    wsf = (torch.rand(N, K // 16, device="cuda") * 2 + 0.25).to(torch.float8_e4m3fn)
    w_scale = 0.0013
    sfb = torch.empty(ops().nvfp4_sf_size(N, K), dtype=torch.uint8, device="cuda")
    ops().nvfp4_swizzle_sf(wsf.view(torch.uint8), sfb, K)
    res = torch.randn(M, N, device="cuda").bfloat16()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ops().nvfp4_gemm(a, sfa, w, sfb, s_in * w_scale, res, out, tile)
    ref = (fake_quant_nvfp4_unscaled(x, s_in).float() * s_in) @ dequant_nvfp4(w, wsf, torch.tensor(w_scale), torch.float32).T + res.float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3, rel


def test_startup_selftest_passes():
    from engine.selftest import run_selftest
    res = run_selftest()
    assert len(res) >= 9 and all(v < 5e-3 for v in res.values())


@pytest.mark.parametrize("M", [1, 37, 256, 2048])
def test_gemm_swiglu_fused(M):
    """Up GEMM with silu(gate) * acc and NVFP4 quantization in the epilogue == NVFP4 of silu(gate) * up computed from the
    two plain GEMMs (up to rounding ties: the fused path does not round up and the product to bf16 first)."""
    from engine.kernels import ops
    torch.manual_seed(M)
    K, I = 1024, 2048

    def q4(x, s):
        q = torch.empty(x.shape[0], x.shape[1] // 2, dtype=torch.uint8, device="cuda")
        sf = torch.empty(ops().nvfp4_sf_size(x.shape[0], x.shape[1]), dtype=torch.uint8, device="cuda")
        ops().nvfp4_quant(x, s, q, sf)
        return q, sf
    x = torch.randn(M, K, device="cuda").bfloat16()
    wg, wu = (torch.randn(I, K, device="cuda") * 0.05).bfloat16(), (torch.randn(I, K, device="cuda") * 0.05).bfloat16()
    sx, sw, sh = x.float().abs().max().item() / 2688, 0.25 / 2688, 0.02
    xq, xsf = q4(x, sx)
    (gq, gsf), (uq, usf) = q4(wg, sw), q4(wu, sw)
    tile = 0 if M >= 1536 else 1
    gate, up = (torch.empty(M, I, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    ops().nvfp4_gemm(xq, xsf, gq, gsf, sx * sw, None, gate, tile)
    ops().nvfp4_gemm(xq, xsf, uq, usf, sx * sw, None, up, tile)
    hq = torch.empty(M, I // 2, dtype=torch.uint8, device="cuda")
    hsf = torch.empty(ops().nvfp4_sf_size(M, I), dtype=torch.uint8, device="cuda")
    ops().nvfp4_gemm_swiglu(xq, xsf, uq, usf, sx * sw, gate, hq, hsf, torch.tensor([1.0 / sh], device="cuda"), tile)
    lut = E2M1_LUT.cuda()
    got = torch.stack([lut[(hq & 15).long()], lut[(hq >> 4).long()]], -1).reshape(M, I)
    got = got * unswizzle(hsf, M, I).view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    want = fake_quant_nvfp4_unscaled((torch.nn.functional.silu(gate.float()) * up.float()).bfloat16(), sh).float()
    assert (got == want).float().mean().item() > 0.97
