#!/usr/bin/env python3
"""INT6 (or INT5) re-quantization of the checkpoint's FP8 projections (docs/phase6_progress.md section 9).

NVFP4 for the Gated DeltaNet projections costs too much quality (Python-code perplexity +0.4-0.6% per projection type,
round-to-nearest; AWQ and GPTQ do not fix it). A 6-bit integer with an e4m3 scale per 16 weights and an fp32 global
scale (6.5 bits per weight against FP8's 8) has a *lower* weight error than the checkpoint's FP8 (2.24% vs 2.67%
relative to the BF16 originals): signed q in [-31, 31], w = q * s_block * s_global, the block scale chosen among
amax/31 * {1, .95, .9, .85, .8} by squared error. Stacked projections (q/k/v; in_proj_qkv + in_proj_z) share the
global scale.

  --simulate: write dequantized BF16 weights (<module>.weight_deq) for tests/perplexity.py --override
  default:    the decode format (attach_decode_copies in engine/model/fast.py, csrc/skinny.cu), codes c = q + 32:
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
from engine.weights.loader import resolve  # noqa: E402
from engine.weights.quantize import REQUANT_DIR, dequant_int, pack5, pack6, quantize_int  # noqa: E402


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
    out = a.out or os.path.join(REQUANT_DIR, os.path.basename(p4), f"attn_gdn_{tag}.safetensors")
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
