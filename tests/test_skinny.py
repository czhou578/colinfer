"""The tensor-core skinny GEMM (csrc/skinny.cu) on NVFP4, FP8 and SwiGLU weights against an fp32 reference built from the
loader's dequantization, every output row bit-identical for any number of rows (what keeps plain decode, verify and the
drafter consistent), the draft early-exit skip flag, and the small decode ops (bf16 GEMV, RMSNorm).
INT6 / INT5: tests/test_skinny_int.py. Run: uv run pytest tests/test_skinny.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")

from engine.weights.loader import dequant_nvfp4  # noqa: E402


def ops():
    from engine.kernels import ops as o
    return o()


def rand_nvfp4(N, K):
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    sf = (torch.rand(N, K // 16, device="cuda") * 3 + 0.25).to(torch.float8_e4m3fn)
    return w, sf


def rel(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("N,K", [(17408, 5120), (5120, 17408), (2056, 1024)])
def test_nvfp4_accuracy_and_row_invariance(N, K):
    torch.manual_seed(N + K)
    w, sf = rand_nvfp4(N, K)
    x = torch.randn(16, K, device="cuda").bfloat16()
    res = torch.randn(16, N, device="cuda").bfloat16()
    ref = x.float() @ dequant_nvfp4(w, sf, torch.tensor(0.37), torch.float32).T
    full = torch.empty(16, N, device="cuda", dtype=torch.float32)
    ops().skinny_nvfp4(x, w, sf, 0.37, None, full)
    assert rel(full, ref) < 1e-5
    for M in (1, 3, 8):  # the first M rows of a 16-row launch, bit for bit
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        ops().skinny_nvfp4(x[:M].contiguous(), w, sf, 0.37, None, out)
        assert torch.equal(out, full[:M])
    out = torch.empty(16, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_nvfp4(x, w, sf, 0.37, res, out)
    assert rel(out, ref + res.float()) < 3e-3


@pytest.mark.parametrize("M", [1, 4, 16])
def test_fp8_row_scales(M):
    torch.manual_seed(M)
    N, K = 1032, 5120
    w = (torch.randn(N, K, device="cuda") * 30).to(torch.float8_e4m3fn)
    rs = torch.rand(N, device="cuda") * 0.01 + 0.001
    x = torch.randn(M, K, device="cuda").bfloat16()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_fp8(x, w, 1.0, None, out, rs)
    assert rel(out, x.float() @ (w.float() * rs[:, None]).T) < 3e-3
    out1 = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_fp8(x[-1:].contiguous(), w, 1.0, None, out1, rs)
    assert torch.equal(out1, out[-1:])


def test_swiglu():
    torch.manual_seed(1)
    N, K = 2048, 5120
    (wg, sg), (wu, su) = rand_nvfp4(N, K), rand_nvfp4(N, K)
    x = (torch.randn(8, K, device="cuda") * 0.05).bfloat16()
    out = torch.empty(8, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_swiglu(x, wg, sg, 0.3, wu, su, 0.2, out)
    g = x.float() @ dequant_nvfp4(wg, sg, torch.tensor(0.3), torch.float32).T
    u = x.float() @ dequant_nvfp4(wu, su, torch.tensor(0.2), torch.float32).T
    assert rel(out, torch.nn.functional.silu(g) * u) < 3e-3


def test_skip_flag_zeroes_output_without_reading():
    torch.manual_seed(2)
    N, K = 1024, 5120
    w, sf = rand_nvfp4(N, K)
    x = torch.randn(4, K, device="cuda").bfloat16()
    flag = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.full((4, N), 7.0, device="cuda", dtype=torch.bfloat16)
    ops().skinny_skip(flag)
    try:
        ops().skinny_nvfp4(x, w, sf, 1.0, None, out)
    finally:
        ops().skinny_skip(None)
    assert torch.equal(out, torch.zeros_like(out))
    flag.zero_()  # flag clear at run time: a normal GEMM (launches made while set read the flag on the device)
    ops().skinny_skip(flag)
    try:
        ops().skinny_nvfp4(x, w, sf, 1.0, None, out)
    finally:
        ops().skinny_skip(None)
    assert rel(out, x.float() @ dequant_nvfp4(w, sf, torch.tensor(1.0), torch.float32).T) < 3e-3


def test_rejects_bad_shapes():
    w, sf = rand_nvfp4(64, 4096)
    with pytest.raises(RuntimeError):
        ops().skinny_nvfp4(torch.randn(17, 4096, device="cuda").bfloat16(), w, sf, 1.0, None, torch.empty(17, 64, device="cuda"))  # M <= 16
    with pytest.raises(RuntimeError):
        ops().skinny_nvfp4(torch.randn(1, 2048, device="cuda").bfloat16(), w, sf, 1.0, None, torch.empty(1, 64, device="cuda"))


def test_bf16_gemv_and_rmsnorm():
    torch.manual_seed(5)
    w = torch.randn(96, 5120, device="cuda").bfloat16()
    x = torch.randn(2, 5120, device="cuda").bfloat16()
    out = torch.empty(2, 96, device="cuda", dtype=torch.bfloat16)
    ops().bf16_gemv(x, w, out)
    assert rel(out, x.float() @ w.float().T) < 2e-3
    from engine.model.qwen35 import RMSNorm
    n = RMSNorm(5120, 1e-6).cuda().bfloat16()
    with torch.no_grad():
        n.weight.normal_(0, 0.2)
    y = torch.empty_like(x)
    ops().rmsnorm(x, n.weight.data, 1e-6, y)
    assert torch.equal(y, n(x)), (y.float() - n(x).float()).abs().max()
