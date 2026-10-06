"""Weight quantizers shared by the engine and the tools: NVFP4 and block-scaled INT6 / INT5.

NVFP4 (ModelOpt's format, what the checkpoint's MLP and lm_head use): e2m1 values, one e4m3 scale per 16 weights, an
fp32 global scale. `quantize` picks each block's scale among amax/6, amax/5.5, ..., amax/4 (rounded to e4m3) by
squared error after e2m1 rounding. The engine uses it for the MTP drafter's decode copies and the low-rank draft head
(engine/spec/mtp.py).

INT6 / INT5 (the decode copies of the checkpoint's FP8 attention / GDN projections, tools/int6_requant.py): signed
q in [-(2^(b-1) - 1), 2^(b-1) - 1], w = q * block scale (e4m3, per 16 weights) * global scale, the block scale chosen
among amax/qmax * {1, .95, .9, .85, .8}. Stored as codes c = q + 2^(b-1) split into two planes that csrc/skinny.cu
streams: the low 4 bits exactly like NVFP4's nibbles, and the high 2 (INT6) or 1 (INT5) bits as a second plane.
"""
from __future__ import annotations

import os

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
NVFP4_DIVISORS = (6.0, 5.5, 5.0, 4.5, 4.0)
INT_DIVISORS = (1.0, 0.95, 0.9, 0.85, 0.8)

REQUANT_DIR = os.path.expanduser("~/.cache/colinfer/requant")


def nvfp4_global_scale(w: torch.Tensor) -> float:
    """The global scale that maps the tensor's largest weight onto e4m3 max x e2m1 max."""
    return float(w.abs().max()) / (448.0 * 6.0)


def quantize(w: torch.Tensor, gs: float, rows: int = 2048):
    """NVFP4: w [N, K] (fp32, cuda), global scale gs -> (packed uint8 [N, K/2] (element 2i in the low nibble),
    block scales e4m3 [N, K/16])."""
    N, K = w.shape
    grid, mids = E2M1_GRID.to(w.device), E2M1_MIDS.to(w.device)
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].reshape(-1, K // 16, 16) / gs         # blocks in units of the global scale
        amax = b.abs().amax(-1, keepdim=True)
        best_err = best_s = best_code = None
        for d in NVFP4_DIVISORS:
            s = (amax / d).clamp(max=448.0).to(torch.float8_e4m3fn).float()
            s = torch.where(s == 0, torch.ones_like(s), s)         # all-zero block: any scale
            y = (b / s).clamp(-6.0, 6.0)
            idx = torch.bucketize(y.abs(), mids)                    # nearest e2m1 magnitude
            q = grid[idx] * torch.sign(y)
            err = ((q * s - b) ** 2).sum(-1, keepdim=True)
            code = idx.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
            if best_err is None:
                best_err, best_s, best_code = err, s, code
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s, best_s)
                best_code = torch.where(better, code, best_code)
        code = best_code.view(-1, K)
        packed[r0:r0 + rows] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[r0:r0 + rows] = best_s.view(-1, K // 16).to(torch.float8_e4m3fn)
    return packed, scales


def quantize_int(w: torch.Tensor, gs: float, bits: int = 6, rows: int = 2048):
    """INT6 / INT5: w [N, K] fp32 cuda -> (codes uint8 [N, K] = q + 2^(bits-1), block scales e4m3 [N, K/16])."""
    qmax = 2 ** (bits - 1) - 1
    N, K = w.shape
    codes = torch.empty(N, K, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].reshape(-1, K // 16, 16) / gs
        amax = b.abs().amax(-1, keepdim=True)
        best = None
        for d in INT_DIVISORS:
            s = (amax * d / qmax).clamp(max=448.0).to(torch.float8_e4m3fn).float()
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = torch.round(b / s).clamp(-qmax, qmax)
            err = ((q * s - b) ** 2).sum(-1, keepdim=True)
            if best is None:
                best = [err, s, q]
            else:
                m = err < best[0]
                best = [torch.where(m, err, best[0]), torch.where(m, s, best[1]), torch.where(m, q, best[2])]
        codes[r0:r0 + rows] = (best[2] + 2 ** (bits - 1)).to(torch.uint8).view(-1, K)
        scales[r0:r0 + rows] = best[1].view(-1, K // 16).to(torch.float8_e4m3fn)
    return codes, scales


def dequant_int(codes: torch.Tensor, scales: torch.Tensor, gs: float, bits: int = 6) -> torch.Tensor:
    q = codes.float() - 2 ** (bits - 1)
    return (q.view(q.shape[0], -1, 16) * scales.float()[..., None] * gs).view(q.shape)


def pack6(codes: torch.Tensor):
    """6-bit codes uint8 [N, K] -> (lo [N, K/2]: the low nibbles, code 2i in the low half of byte i;
    hi [N, K/4]: the high 2 bits, code 4i + j in bits 2j .. 2j+1 of byte i)."""
    c = codes.to(torch.int32)
    lo, hi = c & 15, c >> 4
    nib = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
    two = (hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)).to(torch.uint8)
    return nib.contiguous(), two.contiguous()


def pack5(codes: torch.Tensor):
    """5-bit codes uint8 [N, K] -> (lo [N, K/2] nibbles, hi [N, K/8]: the high bit of code 8i + j in bit j of byte i)."""
    c = codes.to(torch.int32)
    lo, hi = c & 15, c >> 4
    nib = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
    bits = sum(hi[:, j::8] << j for j in range(8)).to(torch.uint8)
    return nib.contiguous(), bits.contiguous()


def unpack5(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    N = lo.shape[0]
    lo, hi = lo.to(torch.int32), hi.to(torch.int32)
    lo_codes = torch.stack([lo & 15, lo >> 4], -1).view(N, -1)
    hi_bits = torch.stack([(hi >> j) & 1 for j in range(8)], -1).view(N, -1)
    return (lo_codes | (hi_bits << 4)).to(torch.uint8)


def unpack6(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    N = lo.shape[0]
    lo, hi = lo.to(torch.int32), hi.to(torch.int32)
    lo_codes = torch.stack([lo & 15, lo >> 4], -1).view(N, -1)
    hi_bits = torch.stack([(hi >> (2 * j)) & 3 for j in range(4)], -1).view(N, -1)
    return (lo_codes | (hi_bits << 4)).to(torch.uint8)
