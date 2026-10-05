#!/usr/bin/env python3
"""Activation-aware NVFP4 re-quantization of the checkpoint's FP8 attention / GDN projections (PLAN.md Phase 6,
docs/phase6_progress.md section 3).

Round-to-nearest NVFP4 for these 208 linears costs +2.3% Python-code perplexity, spread evenly over the projection
types; GPTQ cut the error the calibration inputs see but over-fit them (WikiText got worse). AWQ (Lin et al. 2023)
instead scales input channel j of a linear by s_j before quantizing and divides the activation by s_j at run time:
y = (W diag(s)) (x / s). Channels with large activations get finer quantization steps relative to their weights.
The weights themselves are not fitted to the calibration data, which keeps the method robust to other text.

Per group of linears sharing one input (q/k/v; in_proj_qkv + in_proj_z; o_proj; out_proj):

  s = d^(alpha / 2) / normalizer, d = E[x_j^2] from the calibration Hessian H = E[x^T x]
  alpha in a grid from 0 (round-to-nearest) to 1, chosen by the output error the calibration inputs see,
  sum over the group of tr(D H D^T) / tr(W H W^T), D = W_eff - W, W_eff = dequant(quant(W diag(s))) diag(1/s)

--fp8-keep F: the fraction of groups with the largest output error after AWQ is left out of the file, so decode keeps
the checkpoint's FP8 weights for them (mixed format).

Output: the requant_nvfp4.py format plus <module>.input_scale_awq (fp32 [K], the s above; the decode path divides
activations by it), read by tests/perplexity.py --override and engine/model/fast.py attach_requant.

   uv run python tools/awq_nvfp4.py [--seqs 128] [--fp8-keep 0.25]
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.weights.loader import PREFIX, dequant_nvfp4, resolve  # noqa: E402
from gptq_nvfp4 import calib_seqs, hessians  # noqa: E402
from requant_nvfp4 import default_out, quantize  # noqa: E402

ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def out_error(W, Weff, H):
    D = Weff - W
    return float(((D @ H) * D).sum()), float(((W @ H) * W).sum())


def quant_group(ws: dict, H: torch.Tensor, alpha: float):
    """ws: name -> W [N, K] fp32 (one input); -> (results name -> (packed, sf, gs, s), relative output error)."""
    d = H.diagonal().clamp_min(1e-12)
    s = d.pow(alpha / 2)
    s = s / (s.max() * s.min()).sqrt()  # AWQ's normalization: centered in log space
    s = s.clamp(1e-4, 1e4)
    scaled = {n: w * s[None, :] for n, w in ws.items()}
    gs = max(float(w.abs().max()) for w in scaled.values()) / (448.0 * 6.0)
    res, num, den = {}, 0.0, 0.0
    for n, w in scaled.items():
        packed, sf = quantize(w, gs)
        weff = dequant_nvfp4(packed, sf, torch.tensor(gs, device="cuda"), torch.float32) / s[None, :]
        e, t = out_error(ws[n], weff, H)
        num, den = num + e, den + t
        res[n] = (packed, sf, gs, s)
    return res, (num / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--seqs", type=int, default=128, help="calibration sequences (half WikiText train, half code)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--hessians", default=None, help="Hessian cache (default: next to the output)")
    ap.add_argument("--fp8-keep", type=float, default=0.0, help="fraction of groups (largest error after AWQ) left in FP8")
    ap.add_argument("--groups", default=None, help="regex: only groups matching it go into the file (e.g. 'self_attn'); the rest "
                    "keep decoding FP8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p4, p16 = resolve(a.nvfp4), resolve(a.bf16)
    tag = "_attn" if a.groups == "self_attn" else (f"_keep{a.fp8_keep:g}" if a.fp8_keep else "")
    out = a.out or default_out(p4).replace("attn_gdn_nvfp4", "attn_gdn_nvfp4_awq" + tag)
    cache = a.hessians or os.path.join(os.path.dirname(default_out(p4)), f"hessians_s{a.seqs}_c{a.ctx}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p4)
    H, targets = hessians(p4, calib_seqs(tok, a.seqs, a.ctx, random.Random(0)), cache)
    wm16 = json.load(open(os.path.join(p16, "model.safetensors.index.json")))["weight_map"]

    def load16(name):
        with safe_open(os.path.join(p16, wm16[name]), "pt", device="cuda") as f:
            return f.get_tensor(name).float()
    groups = {}
    for lin, hk in targets.items():
        parent, leaf = lin.rsplit(".", 1)
        gk = parent + (":qkv" if leaf in ("q_proj", "k_proj", "v_proj") else ":qkvz" if leaf in ("in_proj_qkv", "in_proj_z") else ":" + leaf)
        groups.setdefault(gk, []).append(lin)
    chosen = {}  # group -> (alpha, rtn error, awq error, results)
    for gi, (gk, lins) in enumerate(sorted(groups.items())):
        ws = {lin: load16(PREFIX + lin + ".weight") for lin in lins}
        Hg = H[targets[lins[0]]]
        best = None
        rtn = None
        for alpha in ALPHAS:
            res, err = quant_group(ws, Hg, alpha)
            if alpha == 0.0:
                rtn = err
            if best is None or err < best[1]:
                best = (alpha, err, {n: tuple(x.cpu() if torch.is_tensor(x) else x for x in r) for n, r in res.items()})
        chosen[gk] = (best[0], rtn, best[1], best[2])
        if gi % 8 == 0:
            print(f"[awq] {gi + 1}/{len(groups)} {gk}: alpha {best[0]:.1f}, output error RTN {rtn:.4f} -> AWQ {best[1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del ws
    errs = sorted(chosen.items(), key=lambda kv: -kv[1][2])
    keep = {gk for gk, _ in errs[: round(a.fp8_keep * len(errs))]}
    if a.groups:
        import re
        keep |= {gk for gk in chosen if not re.search(a.groups, gk)}
    tensors = {}
    for gk, (alpha, rtn, err, res) in chosen.items():
        if gk in keep:
            continue
        for lin, (packed, sf, gs, s) in res.items():
            name = PREFIX + lin
            tensors[name + ".weight"], tensors[name + ".weight_scale"] = packed, sf
            tensors[name + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
            tensors[name + ".input_scale_awq"] = s.float()
    save_file(tensors, out, metadata={"source": a.bf16, "format": "nvfp4", "method": "awq", "for": a.nvfp4,
                                      "fp8_keep": ",".join(sorted(keep))})
    n_rtn = sum(v[1] for v in chosen.values()) / len(chosen)
    n_awq = sum(v[2] for v in chosen.values()) / len(chosen)
    alphas = [v[0] for v in chosen.values()]
    print(f"[awq] {len(chosen) - len(keep)}/{len(chosen)} groups -> {out} in {time.time() - t0:.0f}s; mean output error "
          f"RTN {n_rtn:.4f} -> AWQ {n_awq:.4f}; alpha mean {sum(alphas) / len(alphas):.2f}; {len(keep)} groups kept in FP8")
    for kind in ("self_attn:qkv", "self_attn:o_proj", "linear_attn:qkvz", "linear_attn:out_proj"):
        v = [c for g, c in chosen.items() if g.endswith(kind)]
        print(f"[awq]   {kind:22s} RTN {sum(x[1] for x in v) / len(v):.4f} AWQ {sum(x[2] for x in v) / len(v):.4f} "
              f"alpha {sum(x[0] for x in v) / len(v):.2f}")


if __name__ == "__main__":
    main()
