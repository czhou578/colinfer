"""Correctness of the decode GEMV kernels (csrc/gemv.cu) against a PyTorch fp32 reference built
from the same dequantization code the Phase 1 loader uses. Run: uv run pytest tests/test_gemv.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.weights.loader import dequant_nvfp4  # noqa: E402

# (N, K) of every NVFP4 / FP8 linear in Qwen3.8-27B (lm_head trimmed to keep the test fast)
NVFP4_SHAPES = [(17408, 5120), (5120, 17408), (8192, 5120)]
FP8_SHAPES = [(10240, 5120), (6144, 5120), (5120, 6144), (12288, 5120), (1024, 5120)]


def ops():
    from engine.kernels import ops as o
    return o()


def rand_nvfp4(N, K, g=0.37):
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    sf = (torch.rand(N, K // 16, device="cuda") * 3 + 0.25).to(torch.float8_e4m3fn)
    return w, sf, g


def rand_x(M, K):
    return torch.randn(M, K, device="cuda").bfloat16()


def check(out, ref, name):
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 2e-3, f"{name}: relative error {rel:.2e}"


@pytest.mark.parametrize("N,K", NVFP4_SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 4])
def test_nvfp4_gemv(N, K, M):
    torch.manual_seed(N + M)
    w, sf, g = rand_nvfp4(N, K)
    x = rand_x(M, K)
    ref = x.float() @ dequant_nvfp4(w, sf, torch.tensor(g), torch.float32).T
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    ops().nvfp4_gemv(x, w, sf, g, None, out)
    check(out, ref, "fp32 out")
    res = torch.randn(M, N, device="cuda").bfloat16()
    out16 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ops().nvfp4_gemv(x, w, sf, g, res, out16)
    check(out16, ref + res.float(), "bf16 out + residual")


@pytest.mark.parametrize("M", [1, 4])
def test_nvfp4_swiglu(M):
    N, K = 17408, 5120
    torch.manual_seed(M)
    wg, sg, gg = rand_nvfp4(N, K, 0.21)
    wu, su, gu = rand_nvfp4(N, K, 0.33)
    x = rand_x(M, K)
    g = x.float() @ dequant_nvfp4(wg, sg, torch.tensor(gg), torch.float32).T
    u = x.float() @ dequant_nvfp4(wu, su, torch.tensor(gu), torch.float32).T
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ops().nvfp4_swiglu(x, wg, sg, gg, wu, su, gu, out)
    check(out, torch.nn.functional.silu(g) * u, "swiglu")


@pytest.mark.parametrize("N,K", FP8_SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 4])
def test_fp8_gemv(N, K, M):
    torch.manual_seed(N * 7 + M)
    w = (torch.randn(N, K, device="cuda") * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = 0.0021
    x = rand_x(M, K)
    ref = x.float() @ (w.float() * s).T
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, N, device="cuda").bfloat16()
    ops().fp8_gemv(x, w, s, res, out)
    check(out, ref + res.float(), "fp8 + residual")


def test_rejects_bad_shapes():
    w, sf, g = rand_nvfp4(64, 5120)
    with pytest.raises(RuntimeError):
        ops().nvfp4_gemv(rand_x(9, 5120), w, sf, g, None, torch.empty(9, 64, device="cuda"))  # M <= 8
    with pytest.raises(RuntimeError):
        ops().nvfp4_gemv(rand_x(1, 4096), w, sf, g, None, torch.empty(1, 64, device="cuda"))


def test_bf16_gemv_and_rmsnorm():
    torch.manual_seed(5)
    w = torch.randn(96, 5120, device="cuda").bfloat16()
    x = rand_x(2, 5120)
    out = torch.empty(2, 96, device="cuda", dtype=torch.bfloat16)
    ops().bf16_gemv(x, w, out)
    check(out, x.float() @ w.float().T, "bf16 gemv")
    from engine.model.qwen35 import RMSNorm
    n = RMSNorm(5120, 1e-6).cuda().bfloat16()
    with torch.no_grad():
        n.weight.normal_(0, 0.2)
    y = torch.empty_like(x)
    ops().rmsnorm(x, n.weight.data, 1e-6, y)
    assert torch.equal(y, n(x)), (y.float() - n(x).float()).abs().max()
