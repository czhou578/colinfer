#!/usr/bin/env python3
"""Requantized decode copies of the BF16 matrices of Qwen3.8-Flash-Next (qwen4_exp), docs/history/qwen4exp_plan.md phase B.

The exact path streams about 10.2 GB per decode token: 8.6 GB of it are the BF16 matrices outside the routed experts.
This tool writes smaller copies of them for the decode path (engine/model/fast_qwen4exp.py attach_decode_copies4). The
prefill keeps the BF16 weights (cuBLAS). The formats are those of the 27B's copies (engine/weights/quantize.py,
tools/int6_requant.py): INT6 / INT5 with an e4m3 scale per 16 weights and an fp32 global scale, NVFP4 with the same
scales, and one new format, INT8 with one fp32 scale per row.

The format of each class: the decision (engine/runtime/prepared.py DECODE4_FORMATS) and the measurements on the real
checkpoint. The perplexity gate is 0.5% against the exact decode path (2.3274), over the 8 golden WikiText windows of
2048 tokens (tests/golden_qwen4exp.py --decode-copies). The ratios below come from the same windows through verify of 16
rows and commit, which gives the decode path's bits. "held out": the next 8 windows (exact path 4.9309). BF16 GB:
the bytes per decode token of the exact path.

| class   | tensors (count)                                 | kernel               | BF16 GB | format: ratio (weight error)            | decision |
|---------|-------------------------------------------------|----------------------|---------|-----------------------------------------|----------|
| gdn_in  | in_proj_qkv + in_proj_z [16384, 2560] (36)      | skinny_int           | 3.02    | INT6 1.0103 (2.4%)                      | BF16     |
| gdn_out | out_proj [2560, 6144] (36)                      | skinny_int           | 1.13    | INT6 0.9992 (2.4%)                      | BF16 (*) |
|  both   | gdn_in and gdn_out                              |                      | 4.15    | INT5 1.0112 (4.6%); INT6 1.0088         |          |
| attn    | q + k + v + index_qk [13952, 2560], o_proj (12) | skinny_int           | 1.23    | INT6 1.0033 (2.4%)                      | INT6     |
| lm_head | [248320, 2560]                                  | skinny_int / _nvfp4  | 1.27    | NVFP4 1.0089 (8.5%); INT6 0.9998 (2.3%) | INT6     |
| hc      | down + inject [324, 10240] (97; mixer 320 rows) | skinny_int, 328 rows | 0.64    | INT6 1.0032 (2.4%)                      | BF16 (*) |
| hc_up   | input_mix_weight_up [10240, 320] (97)           | hc_mix INT8 mode     | 0.64    | INT8 per row 1.0012 (1.1%)              | INT8     |
| shared  | shared_expert gate + up [1280, 2560] (48)       | skinny_int, silu*up  | 0.31    | INT6 1.0111 (2.5%)                      | BF16     |

(*) alone within the gate, but not with the others. Combinations (golden windows; held out):
- all classes (GDN INT5, lm_head NVFP4): 1.0218. All INT6 / INT8 except shared: 1.0105.
- lm_head INT6 + hc_up INT8: 1.0008. + attn INT6 (the decision): 1.0014; held out 1.0043. + hc INT6 instead of attn: 1.0029.
- lm_head + hc_up + attn + hc: 1.0077. lm_head + hc_up + attn + gdn_out: 1.0064; held out 1.0113. All five: 1.0174.
The paired standard error of a ratio (per-token NLL differences over 16376 tokens) is about 0.0025, half the gate: the
gate resolves the large terms (GDN in_proj, the shared expert, the NVFP4 lm_head) but not the small ones, and terms
that are each within the noise add up past the gate. The routed experts (top 10 of 512) amplify a small change of a
layer's input into a different expert choice; the lm_head (no routing after it) stays neutral even at INT6.

The decision passes the gate on the decode path: 2.3307 against 2.3274 (1.0014), and the golden prompts pass (two
diverge at near-ties: the INT6 lm_head writes bf16 logits, so the top two can tie exactly). A decode step streams 8.15
GB of weights instead of 9.95 GB (the BF16 matrices 6.82 instead of 8.62; the experts 1.33) and takes 39.4 ms instead of
46.6 ms at 1k context (the verify of 8 rows 56.1 instead of 63.1 ms). The copies add 1.35 GB on the GPU. Kept
in BF16 besides the classes above: the shared experts' down_proj [2560, 640] (K = 640: the block-scaled skinny formats
need K % 512 == 0), the routers, the PLE projections (one layer) and the embedding (one row per token). The INT8 W_up
mode streams half the bytes: hc_mix takes 19.8 us instead of 33.4 us at M = 1 (97 calls per token).

Each stacked group (qkv | z; q | k | v | index; down | inject; gate | up) shares one global scale, so that the decode
path stays one launch per group. The block scale of INT6 / INT5 is chosen among amax / qmax * {1, .95, .9, .85, .8} by the
squared error; the NVFP4 block scale among amax / {6, 5.5, 5, 4.5, 4} (engine/weights/quantize.py). INT8: q = round(w / s)
with s = amax(row) / 127.

The files (one per class, in the prepared directory; engine/runtime/prepared.py), the names as tools/int6_requant.py's:
  decode4_<class>_<format>.safetensors, with per module (the checkpoint's name without .weight):
    INT6 / INT5: <module>.qweight_lo uint8 [N, K/2], .qweight_hi uint8 [N, K/4] (INT5: [N, K/8]), .weight_scale e4m3
                 [N, K/16], .weight_scale_2 fp32 (the group's global scale)
    NVFP4:       <module>.weight uint8 [N, K/2] (e2m1, element 2i in the low nibble), .weight_scale e4m3 [N, K/16],
                 .weight_scale_2 fp32
    INT8:        <module>.qweight int8 [N, K], .weight_scale fp32 [N]

The tool reads the BF16 tensors from the checkpoint's shards (no model load) and quantizes on the GPU, one group at a
time (about 30 s for all classes; up to 14 GB of GPU memory for the INT6 lm_head).

CAUTION: an over-commit of the unified memory powers the machine off. Run it under the GPU lock, with no model loaded:

   flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tools/requant_qwen4exp.py [--model PATH] [--out DIR]
         [--formats gdn_in=int6,lm_head=nvfp4] [--classes attn,lm_head]

tools/prepare.py runs it for a qwen4_exp checkpoint and writes the manifest.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import torch

from engine.runtime.prepared import DECODE4_CLASSES as CLASSES
from engine.runtime.prepared import DECODE4_FORMATS as FORMATS
from engine.runtime.prepared import decode4_file as file_name
from engine.weights.quantize import dequant_int, int_global_scale, nvfp4_global_scale, pack5, pack6, quantize, quantize_int

PREFIX = "model.language_model."  # engine.weights.loader.PREFIX
CHOICES = {"gdn_in": ("int5", "int6", "bf16"), "gdn_out": ("int5", "int6", "bf16"), "attn": ("int6", "int5", "bf16"),
           "lm_head": ("int6", "nvfp4", "bf16"),
           "hc": ("int6", "bf16"), "hc_up": ("int8", "bf16"), "shared": ("int6", "bf16")}


def parse_formats(spec: str | None, base: dict[str, str] | None = None) -> dict[str, str]:
    """"gdn=int6,lm_head=bf16" over the base formats (default FORMATS)."""
    out = dict(base or FORMATS)
    for item in filter(None, (spec or "").split(",")):
        cls, _, fmt = item.partition("=")
        if cls not in CHOICES or fmt not in CHOICES[cls]:
            raise ValueError(f"--formats {item!r}: a class of {CLASSES} with one of its formats {CHOICES.get(cls)}")
        out[cls] = fmt
    return out


def plan(names) -> dict[str, list[tuple[str, ...]]]:
    """The stacked groups of each class (module names without .weight, in the order of the decode stack), from the
    checkpoint's tensor names. A group appears only when the checkpoint has all its modules."""
    have = {n[: -len(".weight")] for n in names if n.endswith(".weight")}
    layers = sorted({int(m.group(1)) for n in have if (m := re.match(re.escape(PREFIX) + r"layers\.(\d+)\.", n))})
    out: dict[str, list[tuple[str, ...]]] = {c: [] for c in CLASSES}

    def add(cls, *mods):
        if all(m in have for m in mods):
            out[cls].append(mods)
    for i in layers:
        p = f"{PREFIX}layers.{i}."
        add("gdn_in", p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z")
        add("gdn_out", p + "linear_attn.out_proj")
        add("attn", *(p + "self_attn." + m for m in ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")))
        add("attn", p + "self_attn.o_proj")
        for h in ("attn_hyper_connection.", "mlp_hyper_connection."):
            add("hc", p + h + "input_mix_weight_down", p + h + "block_inject_weight")
            add("hc_up", p + h + "input_mix_weight_up")
        add("shared", p + "mlp.shared_expert.gate_proj", p + "mlp.shared_expert.up_proj")
    mixer = PREFIX + "hyper_connection_mixer."
    add("hc", mixer + "input_mix_weight_down")
    add("hc_up", mixer + "input_mix_weight_up")
    add("lm_head", "lm_head")
    return out


def _rel(deq: torch.Tensor, w: torch.Tensor) -> float:
    return float((deq - w).norm() / w.norm().clamp_min(1e-30))


def quantize_group(ws: dict[str, torch.Tensor], fmt: str) -> tuple[dict[str, torch.Tensor], float]:
    """The copy of one stacked group: ws {module: fp32 [N, K] on the GPU} -> ({file tensor name: CPU tensor}, the
    relative weight error of the group (||deq - w|| / ||w|| over all its modules))."""
    out: dict[str, torch.Tensor] = {}
    err2 = norm2 = 0.0
    if fmt in ("int6", "int5"):
        bits = int(fmt[-1])
        gs = int_global_scale(max(float(w.abs().max()) for w in ws.values()), bits)
        for m, w in ws.items():
            codes, sf = quantize_int(w, gs, bits)
            err2 += float((dequant_int(codes, sf, gs, bits) - w).pow(2).sum())
            lo, hi = (pack6 if bits == 6 else pack5)(codes)
            out[m + ".qweight_lo"], out[m + ".qweight_hi"], out[m + ".weight_scale"] = lo.cpu(), hi.cpu(), sf.cpu()
            out[m + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
            norm2 += float(w.pow(2).sum())
    elif fmt == "nvfp4":
        from engine.weights.loader import dequant_nvfp4
        gs = max(nvfp4_global_scale(w) for w in ws.values())
        for m, w in ws.items():
            packed, sf = quantize(w, gs)
            for r0 in range(0, w.shape[0], 8192):
                deq = dequant_nvfp4(packed[r0:r0 + 8192], sf[r0:r0 + 8192], torch.tensor(gs), torch.float32)
                err2 += float((deq - w[r0:r0 + 8192]).pow(2).sum())
            out[m + ".weight"], out[m + ".weight_scale"] = packed.cpu(), sf.cpu()
            out[m + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
            norm2 += float(w.pow(2).sum())
    elif fmt == "int8":
        for m, w in ws.items():
            s = w.abs().amax(1) / 127
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = torch.round(w / s[:, None]).clamp(-127, 127)
            err2 += float((q * s[:, None] - w).pow(2).sum())
            out[m + ".qweight"], out[m + ".weight_scale"] = q.to(torch.int8).cpu(), s.float().cpu()
            norm2 += float(w.pow(2).sum())
    else:
        raise ValueError(f"no decode copy format {fmt!r}")
    return out, (err2 / max(norm2, 1e-30)) ** 0.5


def dequantize(t: dict[str, torch.Tensor], module: str) -> torch.Tensor:
    """The fp32 weights [N, K] that the copy of `module` in the file tensors t stands for (the inverse of quantize_group;
    the tests compare the decode path with the exact path on these weights)."""
    if module + ".qweight" in t:  # INT8
        return t[module + ".qweight"].float() * t[module + ".weight_scale"].float()[:, None]
    gs = float(t[module + ".weight_scale_2"])
    if module + ".qweight_lo" in t:
        lo, hi = t[module + ".qweight_lo"].int(), t[module + ".qweight_hi"].int()
        N, K = lo.shape[0], lo.shape[1] * 2
        bits = 6 if hi.shape[1] * 4 == K else 5
        w = 2 if bits == 6 else 1
        low = torch.stack([lo & 15, lo >> 4], -1).view(N, K)
        high = torch.stack([(hi >> (w * j)) & (2 ** w - 1) for j in range(8 // w)], -1).view(N, K)
        return dequant_int((low | (high << 4)).to(torch.uint8), t[module + ".weight_scale"], gs, bits)
    from engine.weights.loader import dequant_nvfp4
    return dequant_nvfp4(t[module + ".weight"], t[module + ".weight_scale"], torch.tensor(gs), torch.float32)


def make_copies(names, load, formats: dict[str, str], classes=CLASSES, log=print) -> dict[str, tuple[str, dict[str, torch.Tensor], float]]:
    """{class: (format, the file's tensors, the mean relative weight error of its groups)} for the classes in `classes`
    whose format is not bf16. names: the checkpoint's tensor names; load(name) -> the BF16 tensor (any device)."""
    groups = plan(names)
    out = {}
    for cls in classes:
        fmt = formats[cls]
        if fmt == "bf16" or not groups[cls]:
            continue
        t0, tensors, errs = time.time(), {}, []
        for mods in groups[cls]:
            ws = {m: load(m + ".weight").to("cuda", torch.float32) for m in mods}
            got, err = quantize_group(ws, fmt)
            tensors.update(got)
            errs.append(err)
            del ws
        out[cls] = (fmt, tensors, sum(errs) / len(errs))
        log(f"[requant4] {cls}: {len(groups[cls])} groups -> {fmt} in {time.time() - t0:.1f}s, relative weight error mean {out[cls][2]:.4f}, "
            f"max {max(errs):.4f}")
    return out


def write_copies(out_dir: str, copies: dict, source: str = "") -> list[str]:
    """Writes one file per class into out_dir (a temporary name, then a rename); returns the file names."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for cls, (fmt, tensors, err) in copies.items():
        name = file_name(cls, fmt)
        path = os.path.join(out_dir, name)
        save_file(tensors, path + ".tmp", metadata={"source": source, "class": cls, "format": fmt, "weight_error": f"{err:.5f}"})
        os.replace(path + ".tmp", path)
        names.append(name)
    return names


def checkpoint_loader(path: str):
    """(the tensor names, load(name) -> the BF16 tensor on the GPU) of a checkpoint directory's safetensors shards."""
    from safetensors import safe_open

    from engine.weights.loader import weight_map
    wm = weight_map(path)
    handles: dict[str, object] = {}

    def load(name):
        f = wm[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(path, f), "pt", device="cuda")
        return handles[f].get_tensor(name)
    return list(wm), load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the requantized decode copies of a qwen4_exp checkpoint's BF16 matrices")
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4", help="an HF repo id in the local cache, or a directory")
    ap.add_argument("--out", default=None, help="the directory of the files (default: the checkpoint's prepared directory)")
    ap.add_argument("--formats", default=None, help=f"class=format overrides of {FORMATS} (bf16: no copy)")
    ap.add_argument("--classes", default=None, help="a comma list of the classes to write (default: all with a format)")
    a = ap.parse_args(argv)
    from engine.runtime.prepared import prepared_dir
    from engine.weights.loader import resolve
    path = resolve(a.model)
    out = os.path.expanduser(a.out) if a.out else prepared_dir(path)
    formats = parse_formats(a.formats)
    classes = tuple(a.classes.split(",")) if a.classes else CLASSES
    bad = [c for c in classes if c not in CLASSES]
    if bad:
        ap.error(f"--classes: {bad} not in {CLASSES}")
    t0 = time.time()
    names, load = checkpoint_loader(path)
    copies = make_copies(names, load, formats, classes)
    files = write_copies(out, copies, source=os.path.basename(path))
    for f in files:
        print(f"[requant4] {os.path.join(out, f)}: {os.path.getsize(os.path.join(out, f)) / 1e9:.3f} GB")
    print(f"[requant4] {len(files)} files in {time.time() - t0:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
