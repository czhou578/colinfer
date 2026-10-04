#!/usr/bin/env python3
"""Re-quantize the checkpoint's FP8 attention / GDN projections to NVFP4 from the BF16 originals (PLAN.md Phase 6,
docs/baseline.md section 3: 7.2 GB of FP8 -> ~4 GB, per-token weight stream 17.6 -> ~14.5 GB).

nvidia/Qwen3.8-27B-NVFP4 keeps 208 linears in FP8 (attention q/k/v/o, GDN in_proj_qkv / in_proj_z / out_proj).
This tool quantizes the same tensors of Qwen/Qwen3.8-27B (BF16, so the FP4 rounding is not stacked on the FP8 one)
to the ModelOpt NVFP4 format: e2m1 values, an e4m3 scale per 16-element block, an fp32 global scale.

* Block scale: for every block the scale is chosen among amax/6, amax/5.5, ..., amax/4 (rounded to e4m3) by
  squared error after e2m1 rounding; smaller divisors trade clipping of the block maximum for finer steps.
* Global scale: shared by the projections that the decode path stacks into one launch (q/k/v; in_proj_qkv +
  in_proj_z), amax over the group / (448 * 6).

Output: ~/.cache/colinfer/requant/<NVFP4 snapshot>/attn_gdn_nvfp4.safetensors with the checkpoint's tensor names
(<module>.weight uint8 [N, K/2], .weight_scale e4m3 [N, K/16], .weight_scale_2 fp32), loaded by
engine/model/fast.py (decode) and tests/perplexity.py --override.

   uv run python tools/requant_nvfp4.py
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.weights.loader import dequant_nvfp4, resolve  # noqa: E402

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
DIVISORS = (6.0, 5.5, 5.0, 4.5, 4.0)


def default_out(nvfp4_path: str) -> str:
    return os.path.join(os.path.expanduser("~/.cache/colinfer/requant"), os.path.basename(nvfp4_path), "attn_gdn_nvfp4.safetensors")


def quantize(w: torch.Tensor, gs: float, rows: int = 2048):
    """w [N, K] (fp32, cuda), global scale gs -> (packed uint8 [N, K/2], block scales e4m3 [N, K/16])."""
    N, K = w.shape
    grid, mids = GRID.to(w.device), MIDS.to(w.device)
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, N, rows):
        b = w[r0:r0 + rows].view(-1, K // 16, 16) / gs           # blocks in units of the global scale
        amax = b.abs().amax(-1, keepdim=True)
        best_err, best_s, best_code = None, None, None
        for d in DIVISORS:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--bf16", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p4, p16 = resolve(a.nvfp4), resolve(a.bf16)
    out = a.out or default_out(p4)
    wm4 = json.load(open(os.path.join(p4, "model.safetensors.index.json")))["weight_map"]
    wm16 = json.load(open(os.path.join(p16, "model.safetensors.index.json")))["weight_map"]
    fp8 = []
    for k, f in wm4.items():
        if k.endswith(".weight") and not k.startswith(("mtp.", "model.visual.")):
            with safe_open(os.path.join(p4, f), "pt") as s:
                if str(s.get_slice(k).get_dtype()) == "F8_E4M3":
                    fp8.append(k)
    # groups sharing a global scale (stacked into one decode launch)
    groups: dict[str, list[str]] = {}
    for k in fp8:
        mod = k[: -len(".weight")]
        parent, leaf = mod.rsplit(".", 1)
        key = parent + (":qkv" if leaf in ("q_proj", "k_proj", "v_proj") else ":qkvz" if leaf in ("in_proj_qkv", "in_proj_z") else ":" + leaf)
        groups.setdefault(key, []).append(k)

    def load(path, wm, name):
        with safe_open(os.path.join(path, wm[name]), "pt", device="cuda") as s:
            return s.get_tensor(name)
    t0 = time.time()
    tensors, rel4, rel8 = {}, [], []
    for gi, (key, names) in enumerate(sorted(groups.items())):
        ws = {n: load(p16, wm16, n).float() for n in names}
        gs = max(float(w.abs().max()) for w in ws.values()) / (448.0 * 6.0)
        for n, w in ws.items():
            packed, sf = quantize(w, gs)
            base = n[: -len(".weight")]
            tensors[n], tensors[base + ".weight_scale"] = packed.cpu(), sf.cpu()
            tensors[base + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
            deq = dequant_nvfp4(packed, sf, torch.tensor(gs, device="cuda"), torch.float32)
            w8 = load(p4, wm4, n).float() * load(p4, wm4, base + ".weight_scale").float()
            nrm = w.norm()
            rel4.append(float((deq - w).norm() / nrm))
            rel8.append(float((w8 - w).norm() / nrm))
        if gi % 16 == 0:
            print(f"[requant] {gi + 1}/{len(groups)} groups  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata={"source": a.bf16, "format": "nvfp4", "for": a.nvfp4})
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"[requant] {len(fp8)} linears -> {out} ({nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    print(f"[requant] relative weight error vs BF16: NVFP4 mean {sum(rel4) / len(rel4):.4f} max {max(rel4):.4f}; "
          f"checkpoint FP8 mean {sum(rel8) / len(rel8):.4f} max {max(rel8):.4f}")


if __name__ == "__main__":
    main()
