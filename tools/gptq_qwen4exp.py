#!/usr/bin/env python3
"""Calibrated (GPTQ) decode copies of the GDN projections and the shared experts of Qwen3.8-Flash-Next (qwen4_exp).

tools/requant_qwen4exp.py rounds each weight to the nearest code. For the GDN projections and the shared experts that
breaks the perplexity gate (INT6 in_proj alone: ratio 1.0103; the shared gate | up: 1.0111). GPTQ (Frantar et al. 2022)
quantizes the columns of a matrix one at a time and moves the error of each column onto the columns not yet quantized,
weighted by the inverse of the input second moment H = X^T X of the layer. So it minimizes the error of the layer's
output on the calibration inputs, not the error of the weights.

The tool:
1. loads the decode model (engine/model/fast_qwen4exp.py) and prefills --windows windows of --length tokens of text
   (WikiText train paragraphs, Python functions, TinyStories, as tools/drafter_data.py draws them; not the WikiText
   test text of the gate), each from a fresh state, in chunks of 4096 tokens;
2. collects H of each input during the prefill: the input of the stacked in_proj_qkv | in_proj_z and of out_proj of each
   GDN layer (a torch function mode sees the prefill's matmuls with those weights), and the input of the MoE block,
   which the shared expert's gate | up read (a wrapper of KernelMoE.prefill);
3. quantizes each stacked group with GPTQ in the engine's INT6 / INT5 layout (engine/weights/quantize.py: an e4m3 scale
   per 16 weights of a row, chosen among amax / qmax * INT_DIVISORS, and one fp32 global scale per stacked group), in
   blocks of 128 columns with a damping of 1% of the mean diagonal. By default in act order: the columns by descending
   diagonal of H, with the group scales chosen from the weights before GPTQ. --no-act-order: column order, each group's
   scales chosen from the weights that GPTQ has updated so far. On the gate (tools/requant_qwen4exp.py), the act order
   gives 1.0029 and the column order 1.0058;
4. writes the files of tools/requant_qwen4exp.py with the format names gptq6 / gptq5 (decode4_<class>_gptq6.safetensors,
   the tensor names of INT6 / INT5), so attach_decode_copies4 and the gate of tests/golden_qwen4exp.py --decode-copies
   read them unchanged, and a calibrated file never takes the name of a round-to-nearest one.

For each class it prints the relative output error on the calibration inputs, tr(E H E^T) / tr(W H W^T) with E the
weight error, for GPTQ and for round to nearest.

CAUTION: an over-commit of the unified memory powers the machine off. The tool loads the model (78.5 GB) and holds the
second moments on the GPU (7.6 GB). Stop the server, and run it under the GPU lock:

   flock -w 14400 ~/.cache/colinfer/gpu-model.lock uv run python tools/gptq_qwen4exp.py --out DIR [--formats gdn_in=gptq6,gdn_out=gptq5]
         [--windows 48] [--length 2048] [--no-act-order]

tools/prepare.py runs it for the classes whose format (engine/runtime/prepared.py DECODE4_FORMATS) is gptq6 or gptq5.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch
from torch.overrides import TorchFunctionMode

from engine.weights.quantize import E4M3_MAX, INT_DIVISORS, dequant_int, int_global_scale, pack5, pack6, quantize_int

CLASSES = ("gdn_in", "gdn_out", "shared")
DEFAULT_FORMATS = {"gdn_in": "gptq6", "gdn_out": "gptq6", "shared": "gptq6"}  # the file formats: INT6 / INT5 by GPTQ
BITS = {"gptq6": 6, "gptq5": 5}
BLOCK = 128    # the columns of one GPTQ block (a multiple of the 16-column scale group)
DAMP = 0.01    # the damping of H: this fraction of its mean diagonal on the diagonal


def block_scale(b: torch.Tensor, qmax: int) -> torch.Tensor:
    """The e4m3 scale (as fp32) of each row of b [N, 16] (in units of the global scale): quantize_int's choice among
    amax / qmax * INT_DIVISORS by the squared error of round to nearest."""
    amax = b.abs().amax(-1, keepdim=True)
    best_err, best_s = None, None
    for d in INT_DIVISORS:
        s = (amax * d / qmax).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()
        s = torch.where(s == 0, torch.ones_like(s), s)
        err = ((torch.round(b / s).clamp(-qmax, qmax) * s - b) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s = err, s
        else:
            m = err < best_err
            best_err, best_s = torch.where(m, err, best_err), torch.where(m, s, best_s)
    return best_s


def gptq(W: torch.Tensor, H: torch.Tensor, gs: float, bits: int, block: int = BLOCK, damp: float = DAMP, act_order: bool = False):
    """GPTQ of W fp32 [N, K] (cuda) with the input second moment H fp32 [K, K] and the global scale gs -> (codes uint8
    [N, K] = q + 2^(bits-1), block scales e4m3 [N, K/16]): quantize_int's layout. Each 16-column group gets its scales
    when GPTQ reaches it, from the weights updated by the error of the columns before it. act_order: the columns in the
    order of descending H diagonal (the inputs with the most energy first), with static group scales (chosen from the
    weights before GPTQ, since a group's columns are then spread over the whole order)."""
    qmax = 2 ** (bits - 1) - 1
    N, K = W.shape
    if K % 16 or block % 16:
        raise ValueError("K and the block must be multiples of 16")
    W = W.float() / gs
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    static = None
    perm = torch.arange(K, device=W.device)
    if act_order:
        static = block_scale(W.view(N * K // 16, 16), qmax).view(N, K // 16)
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    scales = static.clone() if static is not None else torch.empty(N, K // 16, dtype=torch.float32, device=W.device)
    group = perm // 16  # the scale group of each column in processing order
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if static is not None:
                s = static[:, group[i1 + i]]
            elif i % 16 == 0:
                s = block_scale(W1[:, i:i + 16], qmax)[:, 0]
                scales[:, (i1 + i) // 16] = s
            w = W1[:, i]
            q = torch.round(w / s).clamp(-qmax, qmax)
            codes[:, perm[i1 + i]] = (q + 2 ** (bits - 1)).to(torch.uint8)
            e = (w - q * s) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return codes, scales.to(torch.float8_e4m3fn)


def output_error(W: torch.Tensor, deq: torch.Tensor, H: torch.Tensor) -> float:
    """tr(E H E^T) / tr(W H W^T), E = deq - W: the relative error of the layer's output on the calibration inputs."""
    E = deq - W
    return float(((E @ H) * E).sum() / ((W @ H) * W).sum().clamp_min(1e-30))


def quantize_group(ws: dict[str, torch.Tensor], H: torch.Tensor, fmt: str, act_order: bool = False) -> tuple[dict[str, torch.Tensor], float, float]:
    """The GPTQ copy of one stacked group: ws {module: fp32 [N, K] cuda} (one input, so one H) -> ({file tensor name: CPU
    tensor} as tools/requant_qwen4exp.py writes them, the output error of GPTQ, the output error of round to nearest)."""
    bits = BITS[fmt]
    gs = int_global_scale(max(float(w.abs().max()) for w in ws.values()), bits)
    W = torch.cat(list(ws.values()))
    codes, sf = gptq(W, H, gs, bits, act_order=act_order)
    err = output_error(W, dequant_int(codes, sf, gs, bits), H)
    rc, rs = quantize_int(W, gs, bits)
    err_rtn = output_error(W, dequant_int(rc, rs, gs, bits), H)
    out, r0 = {}, 0
    for m, w in ws.items():
        n = w.shape[0]
        lo, hi = (pack6 if bits == 6 else pack5)(codes[r0:r0 + n])
        out[m + ".qweight_lo"], out[m + ".qweight_hi"], out[m + ".weight_scale"] = lo.cpu(), hi.cpu(), sf[r0:r0 + n].cpu()
        out[m + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
        r0 += n
    return out, err, err_rtn


# ------------------------------------------------------------------------------------------------------------ calibration
def calib_windows(tok, n: int, length: int, seed: int) -> list[list[int]]:
    """n windows of `length` tokens: half WikiText train paragraphs, a quarter Python functions, a quarter TinyStories
    starts, each window the texts of one corpus in a random order, joined by blank lines."""
    from tools.drafter_data import code_snippets, story_starts, wiki_material
    rng = random.Random(seed)
    corpora = [wiki_material(random.Random(rng.getrandbits(64)))[1], code_snippets(random.Random(rng.getrandbits(64))),
               story_starts(random.Random(rng.getrandbits(64)))]
    out = []
    for i in range(n):
        texts = corpora[0 if i % 4 < 2 else 1 + i % 2]
        ids: list[int] = []
        while len(ids) < length:
            ids += tok(rng.choice(texts) + "\n\n", add_special_tokens=False)["input_ids"]
        out.append(ids[:length])
    return out


MATMULS = (torch.Tensor.__matmul__, torch._C.TensorBase.matmul, torch.matmul)  # x @ w reaches the mode as TensorBase.matmul


class Moments(TorchFunctionMode):
    """Adds x^T x of the input rows of each matmul x @ w^T whose w is a registered weight (by its data pointer) to
    H[key], in fp32."""

    def __init__(self, keys: dict[int, str], H: dict[str, torch.Tensor], rows: dict[str, int]):
        super().__init__()
        self.keys, self.H, self.rows = keys, H, rows

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in MATMULS and len(args) == 2 and isinstance(args[1], torch.Tensor):
            key = self.keys.get(args[1].data_ptr())
            if key is not None:
                add(self.H, self.rows, key, args[0])
        return func(*args, **(kwargs or {}))


def add(H: dict, rows: dict, key: str, x: torch.Tensor):
    x = x.reshape(-1, x.shape[-1]).float()
    if key not in H:
        H[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        rows[key] = 0
    H[key].addmm_(x.T, x)
    rows[key] += x.shape[0]


def collect(model, windows: list[list[int]], classes, log=print) -> tuple[dict, dict]:
    """(H {key: the fp32 second moment, summed over the rows}, {key: (the class, {module: weight [N, K]})}) of the
    groups of `classes`, from the prefill of each window from a fresh state."""
    from engine.model import prefill_qwen4exp as pf
    from engine.weights.loader import PREFIX
    keys, groups, H, rows = {}, {}, {}, {}
    for i, layer in enumerate(model.layers):
        p = f"{PREFIX}layers.{i}."
        if layer.block_type == "linear_attention":
            g = layer.linear_attn
            if "gdn_in" in classes:
                n_qkv = g.qkvz.sizes[0]
                keys[g.qkvz.w.data_ptr()] = k = f"gdn_in.{i}"
                groups[k] = ("gdn_in", {p + "linear_attn.in_proj_qkv": g.qkvz.w[:n_qkv], p + "linear_attn.in_proj_z": g.qkvz.w[n_qkv:]})
            if "gdn_out" in classes:
                keys[g.out_proj.w.data_ptr()] = k = f"gdn_out.{i}"
                groups[k] = ("gdn_out", {p + "linear_attn.out_proj": g.out_proj.w})
        if "shared" in classes:
            moe, k = layer.mlp, f"shared.{i}"
            groups[k] = ("shared", {p + "mlp.shared_expert.gate_proj": moe.sh_gate, p + "mlp.shared_expert.up_proj": moe.sh_up})

            def wrapped(x, _orig=moe.prefill, _k=k):
                add(H, rows, _k, x)
                return _orig(x)
            moe.prefill = wrapped
    st = model.new_state(1, -(-max(len(w) for w in windows) // 16) * 16)
    t0 = time.time()
    with torch.inference_mode(), Moments(keys, H, rows):
        for j, ids in enumerate(windows):
            st.reset()
            st.pos = 0
            pf.prefill(model, torch.tensor([ids], device="cuda"), st, logits=False)
            if (j + 1) % 8 == 0:
                log(f"[gptq4] calibration: {j + 1} of {len(windows)} windows in {time.time() - t0:.0f}s")
    for layer in model.layers:
        layer.mlp.__dict__.pop("prefill", None)
    missing = sorted(set(groups) - set(H))
    if missing:
        raise RuntimeError(f"no calibration inputs for {missing[:4]}: the prefill did not run those matmuls")
    return {k: H[k] / rows[k] for k in H}, groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--out", required=True, help="the directory of the copies (a prepared directory, or one to test with the gate first)")
    ap.add_argument("--formats", default=None, help="class=format for gdn_in, gdn_out, shared (gptq6 or gptq5; default gptq6 each)")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-act-order", dest="act_order", action="store_false",
                    help="GPTQ in column order with the group scales chosen as it goes (default: act order, the columns by descending H "
                         "diagonal with static group scales)")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from tools.requant_qwen4exp import write_copies
    formats = dict(DEFAULT_FORMATS)
    for kv in (a.formats or "").split(","):
        if kv:
            c, f = kv.split("=")
            if c not in CLASSES or f not in BITS:
                raise SystemExit(f"--formats: {kv!r} (classes {CLASSES}, formats {tuple(BITS)})")
            formats[c] = f
    classes = [c for c in a.classes.split(",") if c]
    path, model = load_fast_qwen4exp(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    windows = calib_windows(tok, a.windows, a.length, a.seed)
    H, groups = collect(model, windows, classes)
    copies = {}
    for cls in classes:
        t0, tensors, errs, rtn = time.time(), {}, [], []
        for k in [k for k in groups if groups[k][0] == cls]:
            ws = {m: w.float() for m, w in groups[k][1].items()}
            got, e, er = quantize_group(ws, H.pop(k), formats[cls], a.act_order)
            tensors.update(got)
            errs.append(e)
            rtn.append(er)
            del ws
        mean = sum(errs) / len(errs)
        copies[cls] = (formats[cls], tensors, mean)
        print(f"[gptq4] {cls}: {len(errs)} groups -> {formats[cls]} in {time.time() - t0:.0f}s; relative output error on the calibration "
              f"inputs: GPTQ mean {mean:.5f} (max {max(errs):.5f}), round to nearest {sum(rtn) / len(rtn):.5f}", flush=True)
    src = f"tools/gptq_qwen4exp.py {os.path.basename(path)} windows={a.windows}x{a.length} seed={a.seed} act_order={a.act_order}"
    for name in write_copies(a.out, copies, source=src, error="output_error"):
        print(f"[gptq4] wrote {os.path.join(a.out, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
