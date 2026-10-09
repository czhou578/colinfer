"""Fake NVFP4 activation quantization in plain PyTorch: the reference for the NVFP4 kernels (startup self-test, kernel
tests). Per 16-element block: SF = e4m3(amax / 6 / input_scale) and x_q = e2m1_rn(x / (SF * input_scale)); the dequant
is x_q * SF * input_scale. This matches scaled_fp4_quant / fp4_quantize with the global scale 1 / input_scale.

The result is the unscaled operand x_q * SF, which is exact in BF16; the per-tensor scale applies after the matmul.
"""
from __future__ import annotations

import torch

from engine.weights.quantize import E2M1_MIDPOINTS, E2M1_VALUES

_TIES_UP = (0.75, 1.75, 3.5)  # round-half-to-even goes to the upper grid point at these midpoints


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Round |a| (fp32, >= 0) to the e2m1 grid, nearest, ties to even, saturating at 6."""
    mids = torch.tensor(E2M1_MIDPOINTS, device=a.device, dtype=a.dtype)
    grid = torch.tensor(E2M1_VALUES, device=a.device, dtype=a.dtype)
    idx = torch.bucketize(a, mids)  # ties go to the lower point
    for t in _TIES_UP:
        idx = idx + (a == t).to(idx.dtype)
    return grid[idx.clamp_(max=7)]


def fake_quant_nvfp4_unscaled(x: torch.Tensor, input_scale: float) -> torch.Tensor:
    """Returns x_q * SF (without the input_scale factor) in x.dtype; exact in BF16."""
    shp = x.shape
    xf = x.float().reshape(-1, shp[-1] // 16, 16)
    amax = xf.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0 / input_scale).to(torch.float8_e4m3fn).float()
    out_scale = torch.where(sf != 0, 1.0 / (sf * input_scale), torch.zeros_like(sf))
    q = e2m1_round((xf * out_scale).abs()) * torch.sign(xf)
    return (q * sf).reshape(shp).to(x.dtype)
