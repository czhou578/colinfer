"""Fake-quantization that reproduces how W4A4/W8A8 kernels (vLLM 0.25 + FlashInfer) execute the
mixed-precision ModelOpt checkpoint, in plain PyTorch.

Three independently switchable effects (names used by --emulate):
  act_nvfp4    NVFP4 activations for NVFP4 linears (MLP gate/up/down, lm_head): per 16-element block,
               SF = e4m3(amax / 6 / input_scale), x_q = e2m1_rn(x / (SF * input_scale)); dequant
               x_q * SF * input_scale. Matches scaled_fp4_quant / fp4_quantize with global scale
               1 / input_scale.
  act_fp8      FP8 e4m3 static per-tensor activations for FP8 linears (attention, GDN projections):
               x_q = e4m3(clamp(x / input_scale, +-448)).
  fp8_requant  vLLM fuses q/k/v and in_proj_qkv + in_proj_z; shards with different FP8 weight scales
               are re-rounded onto the largest scale (requantize_with_max_scale).

NVFP4 and FP8 layers are computed in factored form like the real kernels: the unscaled operands
(e2m1 * block scale, or e4m3 values) are exact in BF16; the per-tensor scales are applied after
the matmul.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EFFECTS = ("act_nvfp4", "act_fp8", "fp8_requant")
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIES_UP = (0.75, 1.75, 3.5)  # round-half-to-even goes to the upper grid point at these midpoints


def parse_effects(spec: str | None) -> set[str]:
    if not spec:
        return set()
    s = set(x.strip() for x in spec.split(",") if x.strip())
    if "all" in s:
        return set(EFFECTS)
    bad = s - set(EFFECTS)
    if bad:
        raise ValueError(f"unknown emulation effects {bad}; choose from {EFFECTS} or 'all'")
    return s


def e2m1_round(a: torch.Tensor) -> torch.Tensor:
    """Round |a| (fp32, >= 0) to the e2m1 grid, nearest, ties to even, saturating at 6."""
    mids = torch.tensor(_E2M1_MIDS, device=a.device, dtype=a.dtype)
    grid = torch.tensor(_E2M1_GRID, device=a.device, dtype=a.dtype)
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


def fake_quant_fp8_unscaled(x: torch.Tensor, input_scale: float) -> torch.Tensor:
    """Returns e4m3(clamp(x / input_scale)) as x.dtype (exact)."""
    return (x.float() / input_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)


class QuantLinear(nn.Module):
    """Linear whose weight holds the unscaled quantized values (exact in BF16)."""

    def __init__(self, weight: torch.Tensor, kind: str, w_scale: float, in_scale: float, quant_act: bool):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.kind, self.w_scale, self.in_scale, self.quant_act = kind, float(w_scale), float(in_scale), quant_act

    def forward(self, x):
        if self.quant_act:
            xq = fake_quant_nvfp4_unscaled(x, self.in_scale) if self.kind == "nvfp4" else fake_quant_fp8_unscaled(x, self.in_scale)
            alpha = self.in_scale * self.w_scale
        else:
            xq, alpha = x, self.w_scale
        return (F.linear(xq, self.weight).float() * alpha).to(x.dtype)

    def extra_repr(self):
        return f"{self.kind}, {tuple(self.weight.shape)}, quant_act={self.quant_act}"


FP8_FUSED_GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"))


def requant_fused_fp8(meta: dict, sd: dict) -> int:
    """Re-round fused FP8 shards onto the group's max weight scale, as vLLM does. Returns #shards changed."""
    changed = 0
    layers = sorted({k.split(".")[1] for k in meta if k.startswith("layers.")}, key=int)
    for L in layers:
        for group in FP8_FUSED_GROUPS:
            names = [f"layers.{L}.{m}" for m in group]
            if not all(n in meta and meta[n]["kind"] == "fp8" for n in names):
                continue
            smax = max(meta[n]["w_scale"] for n in names)
            for n in names:
                s = meta[n]["w_scale"]
                if s != smax:
                    w = sd[n + ".weight"].float() * (s / smax)
                    sd[n + ".weight"] = w.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(sd[n + ".weight"].dtype)
                    meta[n]["w_scale"] = smax
                    changed += 1
    return changed


def install(model: nn.Module, meta: dict, effects: set[str]) -> dict:
    """Replace the nn.Linear modules named in meta with QuantLinear. Returns counts."""
    counts = dict(nvfp4=0, fp8=0)
    for name, m in meta.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child)
        quant_act = ("act_nvfp4" in effects) if m["kind"] == "nvfp4" else ("act_fp8" in effects)
        setattr(parent, child, QuantLinear(old.weight.data, m["kind"], m["w_scale"], m["in_scale"], quant_act))
        counts[m["kind"]] += 1
    return counts
