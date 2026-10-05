#!/usr/bin/env python3
"""INT6 (or INT5) re-quantization of the checkpoint's FP8 projections (docs/phase6_progress.md section 9).

NVFP4 for the Gated DeltaNet projections costs too much quality (Python-code perplexity +0.4-0.6% per projection type,
round-to-nearest; AWQ and GPTQ do not fix it). A 6-bit integer with an e4m3 scale per 16 weights and an fp32 global
scale (6.5 bits per weight against FP8's 8) has a *lower* weight error than the checkpoint's FP8 (2.24% vs 2.67%
relative to the BF16 originals): signed q in [-31, 31], w = q * s_block * s_global, the block scale chosen among
amax/31 * {1, .95, .9, .85, .8} by squared error. Stacked projections (q/k/v; in_proj_qkv + in_proj_z) share the
global scale.

  --simulate: write dequantized BF16 weights (<module>.weight_deq) for tests/perplexity.py --override
  default:    the decode format (attach_requant in engine/model/fast.py, csrc/skinny.cu INT6), codes c = q + 32:
              <module>.qweight_lo uint8 [N, K/2]: low 4 bits, code 2i in the low nibble of byte i (NVFP4's layout)
              <module>.qweight_hi uint8 [N, K/4]: high 2 bits, code 4i + j in bits 2j .. 2j+1 of byte i
                                  (INT5: [N, K/8], the high bit of code 8i + j in bit j of byte i)
              .weight_scale e4m3 [N, K/16], .weight_scale_2 fp32 (the global scale); codes c = q + 16 for INT5

   uv run python tools/int6_requant.py [--bits 6] [--filter linear_attn] [--simulate]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.weights.loader import resolve  # noqa: E402
from requant_nvfp4 import default_out  # noqa: E402

DIVISORS = (1.0, 0.95, 0.9, 0.85, 0.8)


def quantize_int(w: torch.Tensor, gs: float, bits: int = 6, rows: int = 2048):
    """w [N, K] fp32 cuda -> (codes uint8 [N, K] = q + 2^(bits-1), block scales e4m3 [N, K/16])."""
    qmax = 2 ** (bits - 1) - 1
    N, K = w.shape
    codes = torch.empty(N, K, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].view(-1, K // 16, 16) / gs
        amax = b.abs().amax(-1, keepdim=True)
        best = None
        for d in DIVISORS:
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
    """codes uint8 [N, K] (6-bit) -> (lo [N, K/2] nibbles, hi [N, K/4] 2-bit fields)."""
    c = codes.to(torch.int32)
    lo, hi = c & 15, c >> 4
    nib = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
    two = (hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)).to(torch.uint8)
    return nib.contiguous(), two.contiguous()


def pack5(codes: torch.Tensor):
    """codes uint8 [N, K] (5-bit) -> (lo [N, K/2] nibbles, hi [N, K/8]: code 8i + j in bit j of byte i)."""
    c = codes.to(torch.int32)
    lo, hi = c & 15, c >> 4
    nib = (lo[:, 0::2] | (lo[:, 1::2] << 4)).to(torch.uint8)
    bits = sum(hi[:, j::8] << j for j in range(8)).to(torch.uint8)
    return nib.contiguous(), bits.contiguous()


def unpack5(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    N = lo.shape[0]
    lo, hi = lo.to(torch.int32), hi.to(torch.int32)
    l = torch.stack([lo & 15, lo >> 4], -1).view(N, -1)
    h = torch.stack([(hi >> j) & 1 for j in range(8)], -1).view(N, -1)
    return (l | (h << 4)).to(torch.uint8)


def unpack6(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    N = lo.shape[0]
    lo, hi = lo.to(torch.int32), hi.to(torch.int32)
    l = torch.stack([lo & 15, lo >> 4], -1).view(N, -1)
    h = torch.stack([(hi >> (2 * j)) & 3 for j in range(4)], -1).view(N, -1)
    return (l | (h << 4)).to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--bits", type=int, default=6, choices=(5, 6))
    ap.add_argument("--filter", default=None, help="regex over module names (default: all 208 FP8 linears)")
    ap.add_argument("--simulate", action="store_true", help="write dequantized BF16 weights for tests/perplexity.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p4, p16 = resolve(a.nvfp4), resolve(a.bf16)
    tag = f"int{a.bits}" + ("_sim" if a.simulate else "") + (f"_{re.sub(r'[^a-z_]', '', a.filter)}" if a.filter else "")
    out = a.out or default_out(p4).replace("attn_gdn_nvfp4", f"attn_gdn_{tag}")
    wm4 = json.load(open(os.path.join(p4, "model.safetensors.index.json")))["weight_map"]
    wm16 = json.load(open(os.path.join(p16, "model.safetensors.index.json")))["weight_map"]
    fp8 = []
    for k, f in wm4.items():
        if k.endswith(".weight") and not k.startswith(("mtp.", "model.visual.")) and (not a.filter or re.search(a.filter, k)):
            with safe_open(os.path.join(p4, f), "pt") as s:
                if str(s.get_slice(k).get_dtype()) == "F8_E4M3":
                    fp8.append(k)
    groups: dict[str, list[str]] = {}
    for k in fp8:
        parent, leaf = k[: -len(".weight")].rsplit(".", 1)
        key = parent + (":qkv" if leaf in ("q_proj", "k_proj", "v_proj") else ":qkvz" if leaf in ("in_proj_qkv", "in_proj_z") else ":" + leaf)
        groups.setdefault(key, []).append(k)

    def load(name):
        with safe_open(os.path.join(p16, wm16[name]), "pt", device="cuda") as s:
            return s.get_tensor(name).float()
    t0 = time.time()
    tensors, rel = {}, []
    for gi, (key, names) in enumerate(sorted(groups.items())):
        ws = {n: load(n) for n in names}
        gs = max(float(w.abs().max()) for w in ws.values()) / (448.0 * (2 ** (a.bits - 1) - 1))
        for n, w in ws.items():
            codes, sf = quantize_int(w, gs, a.bits)
            deq = dequant_int(codes, sf, gs, a.bits)
            rel.append(float((deq - w).norm() / w.norm()))
            base = n[: -len(".weight")]
            if a.simulate:
                tensors[base + ".weight_deq"] = deq.to(torch.bfloat16).cpu()
            else:
                lo, hi = (pack6 if a.bits == 6 else pack5)(codes)
                tensors[base + ".qweight_lo"], tensors[base + ".qweight_hi"] = lo.cpu(), hi.cpu()
                tensors[base + ".weight_scale"] = sf.cpu()
                tensors[base + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
        if gi % 16 == 0:
            print(f"[int{a.bits}] {gi + 1}/{len(groups)} groups ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"source": a.bf16, "format": f"int{a.bits}_block16_e4m3", "for": a.nvfp4})
    print(f"[int{a.bits}] {len(fp8)} linears -> {out} in {time.time() - t0:.0f}s; relative weight error mean {sum(rel) / len(rel):.4f}")


if __name__ == "__main__":
    main()
