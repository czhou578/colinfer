#!/usr/bin/env python3
"""Low-bit weight formats for the MLP, simulated (docs/phase6_progress.md section 11).

The MLP is 17.1B of the 27B parameters and ~9.6 GB of the bytes decode reads per token (NVFP4: 4.5 bits per weight).
Every bit per weight saved is ~2.1 GB per token, ~13% faster decode. This tool quantizes the BF16 originals to a
scalar codebook with an e4m3 scale per block (plus an fp32 global scale), round-to-nearest with the block scale searched
by squared error, and either reports weight errors (--report) or writes dequantized weights (<module>.weight_deq) for
tests/perplexity.py --override.

Codebooks (levels before scaling):
  e2m1   NVFP4's {0, .5, 1, 1.5, 2, 3, 4, 6} (x sign)          4 bits
  int4   -7 .. 7                                                4 bits
  int3   -3 .. 3                                                3 bits (7 levels)
  int3h  +-0.5, +-1.5, +-2.5, +-3.5                             3 bits (8 levels, no zero)
  nf3    8 normal quantiles                                     3 bits
  int2h  +-0.5, +-1.5                                           2 bits
Bits per weight = code bits + 8 / block.

   uv run python tools/lowbit_sim.py --report
   uv run python tools/lowbit_sim.py --fmt int3h --block 16 [--filter 'mlp\\.(gate|up)_proj'] [--layers 0-63]
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


def codebook(fmt: str) -> torch.Tensor:
    if fmt == "e2m1":
        v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        v = [-x for x in v[1:]] + v
    elif fmt == "int4":
        v = list(range(-7, 8))
    elif fmt == "int3":
        v = list(range(-3, 4))
    elif fmt == "int3h":
        v = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    elif fmt == "nf3":
        n = torch.distributions.Normal(0, 1)
        p = torch.linspace(0.5 / 8, 1 - 0.5 / 8, 8)
        q = n.icdf(p)
        v = (q / q.abs().max()).tolist()
    elif fmt == "int2h":
        v = [-1.5, -0.5, 0.5, 1.5]
    else:
        raise ValueError(fmt)
    return torch.tensor(sorted(v), dtype=torch.float32)


def code_bits(fmt: str) -> int:
    return {"e2m1": 4, "int4": 4, "int3": 3, "int3h": 3, "nf3": 3, "int2h": 2}[fmt]


def quantize(w: torch.Tensor, fmt: str, block: int, rows: int = 1024) -> torch.Tensor:
    """w [N, K] fp32 cuda -> dequantized fp32 [N, K]."""
    cb = codebook(fmt).to(w.device)
    mx = float(cb.abs().max())
    mids = (cb[1:] + cb[:-1]) / 2
    N, K = w.shape
    gs = float(w.abs().max()) / (448.0 * mx)
    out = torch.empty_like(w)
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].view(-1, K // block, block) / gs
        amax = b.abs().amax(-1, keepdim=True)
        best_e = best_q = None
        for d in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            s = (amax * d / mx).clamp(max=448.0).to(torch.float8_e4m3fn).float()
            s = torch.where(s == 0, torch.ones_like(s), s)
            y = (b / s).clamp(-mx, mx)
            q = cb[torch.bucketize(y, mids)] * s
            e = ((q - b) ** 2).sum(-1, keepdim=True)
            if best_e is None:
                best_e, best_q = e, q
            else:
                m = e < best_e
                best_e, best_q = torch.where(m, e, best_e), torch.where(m, q, best_q)
        out[r0:r0 + rows] = (best_q * gs).view(-1, K)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--nvfp4", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--fmt", default="int3h")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--filter", default=r"mlp\.(gate|up|down)_proj", help="regex over module names")
    ap.add_argument("--layers", default="0-63", help="lo-hi")
    ap.add_argument("--report", action="store_true", help="weight errors of every format on a few layers, nothing written")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p16, p4 = resolve(a.bf16), resolve(a.nvfp4)
    wm16 = json.load(open(os.path.join(p16, "model.safetensors.index.json")))["weight_map"]
    wm4 = json.load(open(os.path.join(p4, "model.safetensors.index.json")))["weight_map"]
    P = "model.language_model.layers."

    def load16(n):
        with safe_open(os.path.join(p16, wm16[n]), "pt", device="cuda") as f:
            return f.get_tensor(n).float()
    if a.report:
        from engine.weights.loader import dequant_nvfp4

        def load4(n):
            with safe_open(os.path.join(p4, wm4[n]), "pt", device="cuda") as f:
                return f.get_tensor(n)
        fmts = [("ckpt nvfp4", None, None), ("e2m1", "e2m1", 16), ("int4", "int4", 16), ("int3", "int3", 16), ("int3h", "int3h", 16),
                ("nf3", "nf3", 16), ("int3h b32", "int3h", 32), ("int3h b8", "int3h", 8), ("int2h", "int2h", 16)]
        print("bits/w: " + "  ".join(f"{n} {code_bits(f) + 8 / b:.2f}" if f else f"{n} 4.50" for n, f, b in fmts))
        for L in (2, 31, 60):
            for m in ("gate_proj", "up_proj", "down_proj"):
                n = f"{P}{L}.mlp.{m}.weight"
                w = load16(n)
                r = {}
                for name, f, b in fmts:
                    if f is None:
                        base = n[: -len(".weight")]
                        deq = dequant_nvfp4(load4(n), load4(base + ".weight_scale"), load4(base + ".weight_scale_2"), torch.float32)
                    else:
                        deq = quantize(w, f, b)
                    r[name] = float((deq - w).norm() / w.norm())
                print(f"L{L} {m:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
        return
    lo, hi = (int(x) for x in a.layers.split("-"))
    names = [n for n in wm16 if n.startswith(P) and n.endswith(".weight") and re.search(a.filter, n)
             and lo <= int(n[len(P):].split(".")[0]) <= hi]
    tag = f"mlp_{a.fmt}_b{a.block}" + ("" if a.filter == r"mlp\.(gate|up|down)_proj" else "_" + re.sub(r"[^a-z]", "", a.filter)) + \
          ("" if a.layers == "0-63" else f"_L{lo}-{hi}")
    out = a.out or default_out(p4).replace("attn_gdn_nvfp4", tag + "_sim")
    t0, tensors, rel = time.time(), {}, []
    for i, n in enumerate(sorted(names)):
        w = load16(n)
        deq = quantize(w, a.fmt, a.block)
        rel.append(float((deq - w).norm() / w.norm()))
        tensors[n[: -len(".weight")] + ".weight_deq"] = deq.to(torch.bfloat16).cpu()
        if i % 24 == 0:
            print(f"[lowbit] {i + 1}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"format": f"{a.fmt}_block{a.block}_e4m3", "source": a.bf16})
    print(f"[lowbit] {len(names)} linears -> {out}; relative weight error mean {sum(rel) / len(rel):.4f}")


if __name__ == "__main__":
    main()
