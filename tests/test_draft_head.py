"""Low-rank draft head pieces (engine/spec/mtp.py, docs/phase6_progress.md section 19)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def test_rescore_nvfp4_matches_dequantized_rows():
    from requant_nvfp4 import quantize

    from engine.kernels import ops
    from engine.weights.loader import dequant_nvfp4
    torch.manual_seed(0)
    V, K, B, C = 3000, 5120, 3, 64
    w = torch.randn(V, K, device="cuda") * 0.02
    gs = float(w.abs().max()) / 2688
    p, sf = quantize(w, gs)
    W = dequant_nvfp4(p, sf, torch.tensor(gs), out_dtype=torch.float32)
    x = torch.randn(B, K, device="cuda").bfloat16()
    cand = torch.randint(0, V, (B, C), device="cuda")
    out = ops().rescore_nvfp4(x, p, sf, gs, cand)
    ref = torch.stack([W[cand[b]] @ x[b].float() for b in range(B)])
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
