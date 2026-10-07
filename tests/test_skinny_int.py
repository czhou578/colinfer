"""INT6 / INT5 skinny GEMM (csrc/skinny.cu) against x @ (q * s)^T, with the one bf16 rounding of q * s that the kernel
does. Each output row must be bit-identical for M = 1 and 16. Run: uv run pytest tests/test_skinny_int.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.mark.parametrize("bits", [6, 5])
@pytest.mark.parametrize("N,K", [(256, 512), (1032, 1024), (5120, 6144)])
@pytest.mark.parametrize("M", [1, 3, 16])
def test_skinny_int(bits, N, K, M):
    from engine.kernels import ops
    from engine.weights.quantize import pack5, pack6, quantize_int
    torch.manual_seed(bits * 1000 + N + M)
    W = torch.randn(N, K, device="cuda") * 0.02
    gs = float(W.abs().max()) / (448 * (2 ** (bits - 1) - 1))
    codes, sf = quantize_int(W, gs, bits)
    lo, hi = (pack6 if bits == 6 else pack5)(codes)
    q = codes.float() - 2 ** (bits - 1)
    wq = (q.view(N, -1, 16) * sf.float()[..., None]).view(N, K).bfloat16().float()
    x = torch.randn(M, K, device="cuda").bfloat16()
    res = torch.randn(M, N, device="cuda").bfloat16()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_int(x, lo, hi, sf, gs, res, out)
    ref = (x.float() @ wq.t()) * gs + res.float()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 5e-3
    o1 = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)
    ops().skinny_int(x[:1].contiguous(), lo, hi, sf, gs, res[:1].contiguous(), o1)
    assert torch.equal(o1[0], out[0])
