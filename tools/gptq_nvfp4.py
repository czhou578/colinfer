#!/usr/bin/env python3
"""GPTQ re-quantization of the checkpoint's FP8 attention / GDN projections to NVFP4 (PLAN.md Phase 6).

tools/requant_nvfp4.py rounds each weight to the nearest NVFP4 value: WikiText perplexity +0.13%, but Python-code
perplexity +2.3% against the shipped FP8 weights, spread evenly over the four projection types. GPTQ (Frantar et
al. 2022) quantizes a linear column by column and pushes each column's rounding error onto the columns not yet
quantized, weighted by the inverse Hessian of the layer's inputs (H = X^T X over calibration data), so the error that
remains is the error the inputs cannot see.

  1. Calibration: the reference model (nvidia/Qwen3.8-27B-NVFP4 dequantized to BF16) runs over WikiText-103 *train*
     and Python sources of the installed packages (disjoint from the evaluation corpora: WikiText test, the Python
     standard library); forward hooks accumulate H for the input of every target linear (q/k/v share one, as do
     in_proj_qkv / in_proj_z).
  2. GPTQ on the BF16 originals (Qwen/Qwen3.8-27B), in blocks of 128 columns; NVFP4 block scales (16 columns) are
     chosen when the quantizer reaches a block, from the error-updated weights (squared-error search over
     amax/6 .. amax/4, as requant_nvfp4.py); global scale shared by stacked projections.
  3. Same output format and path as requant_nvfp4.py (attach_requant / perplexity.py --override read it).

   uv run python tools/gptq_nvfp4.py [--seqs 64]
"""
import argparse
import glob
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
from engine.weights.loader import PREFIX, dequant_nvfp4, load_model, resolve  # noqa: E402
from requant_nvfp4 import DIVISORS, GRID, MIDS, default_out  # noqa: E402


def calib_texts(rng: random.Random):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/train-*.parquet")))
    wiki = "\n\n".join(pq.read_table(files[0]).column("text").to_pylist()[:200000])
    import transformers
    src = sorted(glob.glob(os.path.join(os.path.dirname(transformers.__file__), "models", "*", "modeling_*.py")))
    rng.shuffle(src)
    code = "\n\n".join(open(f, errors="replace").read() for f in src[:400])
    return wiki, code


def quant_block(w, gs):
    """w [N, 16] (fp32), global scale -> the block scale per row [N, 1] (e4m3 values as fp32) with the least squared
    error after e2m1 rounding, among amax/6 .. amax/4."""
    grid, mids = GRID.to(w.device), MIDS.to(w.device)
    b = w / gs
    amax = b.abs().amax(-1, keepdim=True)
    best_e = best_s = None
    for d in DIVISORS:
        s = (amax / d).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        s = torch.where(s == 0, torch.ones_like(s), s)
        y = (b / s).clamp(-6.0, 6.0)
        q = grid[torch.bucketize(y.abs(), mids)] * torch.sign(y)
        e = ((q * s - b) ** 2).sum(-1, keepdim=True)
        if best_e is None:
            best_e, best_s = e, s
        else:
            better = e < best_e
            best_e, best_s = torch.where(better, e, best_e), torch.where(better, s, best_s)
    return best_s


def round_with_scale(w, s, gs):
    """w [N, j] columns of one block, s [N, 1] block scale, gs -> (dequantized [N, j], codes uint8 [N, j])."""
    grid, mids = GRID.to(w.device), MIDS.to(w.device)
    y = (w / (s * gs)).clamp(-6.0, 6.0)
    idx = torch.bucketize(y.abs(), mids)
    q = grid[idx] * torch.sign(y)
    code = idx.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    return q * s * gs, code


def gptq(W, H, gs, blocksize=128, damp=0.01):
    """W [N, K] fp32 (cuda), H [K, K] fp32 -> packed [N, K/2] uint8, scales e4m3 [N, K/16]."""
    W = W.clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    scales = torch.empty(N, K // 16, dtype=torch.float32, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for g0 in range(0, i2 - i1, 16):
            s = quant_block(W1[:, g0:g0 + 16], gs)       # block scale from the error-updated weights
            scales[:, (i1 + g0) // 16] = s[:, 0]
            for i in range(g0, g0 + 16):
                w = W1[:, i]
                d = Hinv1[i, i]
                qd, c = round_with_scale(w[:, None], s, gs)
                codes[:, i1 + i] = c[:, 0]
                err = (w - qd[:, 0]) / d
                W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
                Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales.to(torch.float8_e4m3fn)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--seqs", type=int, default=64, help="calibration sequences (half WikiText train, half code)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--damp", type=float, default=0.01, help="Hessian damping, fraction of mean(diag H): larger -> closer to round-to-nearest")
    ap.add_argument("--hessians", default=None, help="cache file for the calibration Hessians (computed once, reused)")
    ap.add_argument("--check", type=int, default=0, help="only compare GPTQ with round-to-nearest on this many linears")
    a = ap.parse_args()
    p4, p16 = resolve(a.nvfp4), resolve(a.bf16)
    out = a.out or default_out(p4).replace("attn_gdn_nvfp4", f"attn_gdn_nvfp4_gptq_d{a.damp:g}")
    t0 = time.time()
    # ---- 1. calibration Hessians
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p4)
    rng = random.Random(0)
    wiki, code = calib_texts(rng)
    seqs = []
    for text, n in ((wiki, a.seqs // 2), (code, a.seqs - a.seqs // 2)):
        ids = tok(text, return_tensors="pt").input_ids[0]
        starts = rng.sample(range(0, ids.numel() - a.ctx), n)
        seqs += [ids[s:s + a.ctx] for s in starts]
    H, cnt = {}, {}
    cached = a.hessians and os.path.exists(a.hessians)
    model = load_model(p4) if not cached else None

    def hook(name):
        def fn(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if name not in H:
                H[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                cnt[name] = 0
            H[name].addmm_(x.t(), x)
            cnt[name] += x.shape[0]
        return fn
    targets = {}  # linear name -> Hessian key (shared input)
    from engine.model.qwen35 import Qwen35Config
    cfg = Qwen35Config.from_checkpoint(p4)
    for i, btype in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        if btype == "full_attention":
            for n in ("q_proj", "k_proj", "v_proj"):
                targets[p + "self_attn." + n] = p + "attn_in"
            targets[p + "self_attn.o_proj"] = p + "attn_out"
        else:
            targets[p + "linear_attn.in_proj_qkv"] = p + "gdn_in"
            targets[p + "linear_attn.in_proj_z"] = p + "gdn_in"
            targets[p + "linear_attn.out_proj"] = p + "gdn_out"
    for i, layer in enumerate(model.layers if model is not None else []):
        p = f"layers.{i}."
        if layer.block_type == "full_attention":
            at = layer.self_attn
            at.q_proj.register_forward_pre_hook(hook(p + "attn_in"))
            at.o_proj.register_forward_pre_hook(hook(p + "attn_out"))
            for n in ("q_proj", "k_proj", "v_proj"):
                targets[p + "self_attn." + n] = p + "attn_in"
            targets[p + "self_attn.o_proj"] = p + "attn_out"
        else:
            g = layer.linear_attn
            g.in_proj_qkv.register_forward_pre_hook(hook(p + "gdn_in"))
            g.out_proj.register_forward_pre_hook(hook(p + "gdn_out"))
            targets[p + "linear_attn.in_proj_qkv"] = p + "gdn_in"
            targets[p + "linear_attn.in_proj_z"] = p + "gdn_in"
            targets[p + "linear_attn.out_proj"] = p + "gdn_out"
    for j, s in enumerate(seqs if model is not None else []):
        state = model.new_state(1, a.ctx)
        model(s.view(1, -1).cuda(), state)
        if j % 8 == 7:
            print(f"[gptq] calibration {j + 1}/{len(seqs)} sequences ({time.time() - t0:.0f}s)", flush=True)
    if cached:
        H = {k: v.cuda() for k, v in torch.load(a.hessians).items()}
        print(f"[gptq] Hessians from {a.hessians}")
    else:
        for k in H:
            H[k] /= cnt[k]
        del model
        torch.cuda.empty_cache()
        if a.hessians:
            torch.save({k: v.cpu() for k, v in H.items()}, a.hessians)
            print(f"[gptq] Hessians saved to {a.hessians}")
    # ---- 2. GPTQ per linear
    wm16 = json.load(open(os.path.join(p16, "model.safetensors.index.json")))["weight_map"]

    def load16(name):
        with safe_open(os.path.join(p16, wm16[name]), "pt", device="cuda") as f:
            return f.get_tensor(name).float()
    groups = {}
    for lin, hk in targets.items():
        parent, leaf = lin.rsplit(".", 1)
        gk = parent + (":qkv" if leaf in ("q_proj", "k_proj", "v_proj") else ":qkvz" if leaf in ("in_proj_qkv", "in_proj_z") else ":" + leaf)
        groups.setdefault(gk, []).append(lin)
    if a.check:
        rtn = default_out(p4)
        with safe_open(rtn, "pt", device="cuda") as f:
            for gk, lins in list(sorted(groups.items()))[: a.check]:
                ws = {lin: load16(PREFIX + lin + ".weight") for lin in lins}
                gs = max(float(w.abs().max()) for w in ws.values()) / (448.0 * 6.0)
                for lin, w in ws.items():
                    Hl = H[targets[lin]]
                    n = PREFIX + lin
                    r = dequant_nvfp4(f.get_tensor(n + ".weight"), f.get_tensor(n + ".weight_scale"), f.get_tensor(n + ".weight_scale_2"), torch.float32)
                    packed, sf = gptq(w, Hl, gs)
                    g = dequant_nvfp4(packed, sf, torch.tensor(gs, device="cuda"), torch.float32)
                    err = lambda d: float(torch.sqrt(((d @ Hl) * d).sum() / ((w @ Hl) * w).sum()))  # noqa: E731
                    print(f"[check] {lin:40s} output error: RTN {err(r - w):.4f}  GPTQ {err(g - w):.4f}   weight error RTN "
                          f"{float((r - w).norm() / w.norm()):.4f} GPTQ {float((g - w).norm() / w.norm()):.4f}", flush=True)
        return
    tensors, rel = {}, []
    for gi, (gk, lins) in enumerate(sorted(groups.items())):
        ws = {lin: load16(PREFIX + lin + ".weight") for lin in lins}
        gs = max(float(w.abs().max()) for w in ws.values()) / (448.0 * 6.0)
        for lin, w in ws.items():
            Hl = H[targets[lin]]
            packed, sf = gptq(w, Hl, gs, damp=a.damp)
            name = PREFIX + lin
            tensors[name + ".weight"], tensors[name + ".weight_scale"] = packed.cpu(), sf.cpu()
            tensors[name + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
            deq = dequant_nvfp4(packed, sf, torch.tensor(gs, device="cuda"), torch.float32)
            d = deq - w  # output error the calibration inputs see, relative: tr(D H D^T) / tr(W H W^T)
            rel.append(float(torch.sqrt(((d @ Hl) * d).sum() / ((w @ Hl) * w).sum())))
        if gi % 16 == 0:
            print(f"[gptq] {gi + 1}/{len(groups)} groups ({time.time() - t0:.0f}s), output error {sum(rel) / len(rel):.4f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"source": a.bf16, "format": "nvfp4", "method": "gptq", "for": a.nvfp4})
    print(f"[gptq] {len(targets)} linears -> {out} in {time.time() - t0:.0f}s; mean relative output error {sum(rel) / len(rel):.4f}")


if __name__ == "__main__":
    main()
