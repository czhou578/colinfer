"""tools/awq_nvfp4.py: alpha = 0 is round-to-nearest; on inputs with outlier channels AWQ lowers the output error the
inputs see. Run: uv run pytest tests/test_awq.py -q"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def setup(K=512, N=256):
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda") * 0.02
    mag = torch.ones(K, device="cuda")
    mag[torch.randperm(K)[:8]] = 30.0  # a few outlier activation channels
    x = torch.randn(4096, K, device="cuda") * mag
    return {"a": w}, x.t() @ x / x.shape[0]


def test_alpha0_is_rtn():
    from awq_nvfp4 import out_error, quant_group
    from requant_nvfp4 import quantize
    from engine.weights.loader import dequant_nvfp4
    ws, H = setup()
    res, err = quant_group(ws, H, 0.0)
    packed, sf, gs, s = res["a"]
    p2, sf2 = quantize(ws["a"], gs)
    assert torch.equal(packed, p2) and torch.equal(sf.view(torch.uint8), sf2.view(torch.uint8))
    assert torch.allclose(s, torch.ones_like(s))
    e, t = out_error(ws["a"], dequant_nvfp4(p2, sf2, torch.tensor(gs, device="cuda"), torch.float32), H)
    assert abs(err - (e / t) ** 0.5) < 1e-6


def test_awq_beats_rtn_on_outlier_channels():
    from awq_nvfp4 import quant_group
    ws, H = setup()
    _, rtn = quant_group(ws, H, 0.0)
    _, awq = quant_group(ws, H, 0.5)
    assert awq < 0.8 * rtn, (rtn, awq)
