#!/usr/bin/env python3
"""tensor_inventory.py -- what is inside a safetensors checkpoint, without loading any weights.

Reads only the safetensors headers. Collapses layer indices so a 64-layer model prints one row
per distinct parameter, reports dtype / shape / count / bytes per row, totals per category
(backbone, lm_head, embed, mtp, vision), and pairs every quantized weight with its scale
tensors so the quantization scheme (block size, scale dtype, which modules) is explicit.

Usage:
  uv run python tools/tensor_inventory.py nvidia/Qwen3.8-27B-NVFP4          # repo id (must be in the HF cache)
  uv run python tools/tensor_inventory.py /path/to/snapshot [--full]         # local dir; --full lists every tensor
"""
import argparse
import collections
import json
import os
import re
import struct
import sys


def resolve(path_or_repo):
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, local_files_only=True)


def read_header(fn):
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


LAYER_RE = re.compile(r"\.(\d+)\.")


def collapse(name):
    return LAYER_RE.sub(".{L}.", name)


def category(name):
    n = name.lower()
    if "visual" in n or "vision" in n:
        return "vision"
    if "embed_tokens" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    if "mtp" in n:
        return "mtp"
    return "backbone"


def fmt_bytes(b):
    return f"{b / 1e9:8.3f} GB" if b >= 1e8 else f"{b / 1e6:8.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--full", action="store_true", help="list every tensor, not the collapsed view")
    args = ap.parse_args()
    d = resolve(args.ckpt)
    print(f"checkpoint: {d}")

    for cfg in ("config.json", "hf_quant_config.json"):
        p = os.path.join(d, cfg)
        if os.path.exists(p):
            c = json.load(open(p))
            if cfg == "config.json":
                keep = {k: v for k, v in c.items() if k in (
                    "architectures", "model_type", "torch_dtype", "dtype", "transformers_version", "quantization_config")}
                tc = c.get("text_config", c)
                for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                          "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings", "layer_types",
                          "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                          "linear_value_head_dim", "linear_conv_kernel_dim", "full_attention_interval",
                          "mtp_num_hidden_layers", "num_nextn_predict_layers"):
                    if k in tc:
                        v = tc[k]
                        if k == "layer_types" and isinstance(v, list):
                            v = f"{len(v)} layers: " + ", ".join(f"{t}x{n}" for t, n in collections.Counter(v).items())
                        keep[k] = v
                print(f"\n[{cfg}]")
                for k, v in keep.items():
                    print(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}"[:300])
            else:
                print(f"\n[{cfg}]")
                q = c.get("quantization", c)
                for k, v in q.items():
                    s = json.dumps(v)
                    print(f"  {k}: {s[:200]}{' ...' if len(s) > 200 else ''}")

    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    tensors = {}  # name -> (dtype, shape, bytes, file)
    for f in files:
        for name, info in read_header(os.path.join(d, f)).items():
            b0, b1 = info["data_offsets"]
            tensors[name] = (info["dtype"], tuple(info["shape"]), b1 - b0, f)
    print(f"\n{len(files)} safetensors files, {len(tensors)} tensors, {fmt_bytes(sum(t[2] for t in tensors.values()))} total")

    if args.full:
        for name in sorted(tensors):
            dt, sh, nb, f = tensors[name]
            print(f"  {name:70s} {dt:8s} {str(list(sh)):22s} {fmt_bytes(nb)}")
        return

    # ---- collapsed view ----
    groups = collections.OrderedDict()
    for name in sorted(tensors, key=lambda n: (collapse(n), n)):
        g = groups.setdefault(collapse(name), dict(count=0, bytes=0, dtypes=set(), shapes=set(), first=name))
        dt, sh, nb, _ = tensors[name]
        g["count"] += 1
        g["bytes"] += nb
        g["dtypes"].add(dt)
        g["shapes"].add(sh)
    print(f"\n{'parameter (layer index collapsed)':72s} {'n':>4s} {'dtype':8s} {'shape':26s} {'bytes'}")
    for key, g in groups.items():
        shapes = sorted(g["shapes"])
        sh = str(list(shapes[0])) + (f" (+{len(shapes) - 1} more)" if len(shapes) > 1 else "")
        print(f"  {key:70s} {g['count']:4d} {'/'.join(sorted(g['dtypes'])):8s} {sh:26s} {fmt_bytes(g['bytes'])}")

    # ---- category totals ----
    cat = collections.Counter()
    for name, (dt, sh, nb, _) in tensors.items():
        cat[category(name)] += nb
    print("\nbytes by category")
    for k in ("backbone", "lm_head", "embed", "mtp", "vision"):
        if cat[k]:
            print(f"  {k:10s} {fmt_bytes(cat[k])}")
    per_token = cat["backbone"] + cat["lm_head"]
    print(f"  {'per-token':10s} {fmt_bytes(per_token)}   (backbone + lm_head: what a decode step must stream)")

    # ---- quantization pairing ----
    print("\nquantized weights and their scales (collapsed)")
    scale_suffixes = ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "input_scale_2",
                      "k_scale", "v_scale", "q_scale", "output_scale", "activation_scale", "weight_global_scale")
    quant = collections.OrderedDict()
    for key, g in groups.items():
        if key.endswith(".weight") and (("U8" in g["dtypes"]) or any(d.startswith("F8") for d in g["dtypes"]) or ("I8" in g["dtypes"])):
            quant[key[: -len(".weight")]] = g
    if not quant:
        print("  none (no U8 / F8 / I8 weight tensors)")
    for mod, g in quant.items():
        wsh = sorted(g["shapes"])[0]
        wdt = "/".join(sorted(g["dtypes"]))
        parts = [f"weight {wdt} {list(wsh)}"]
        for suf in scale_suffixes:
            sg = groups.get(f"{mod}.{suf}")
            if sg:
                ssh = sorted(sg["shapes"])[0]
                parts.append(f"{suf} {'/'.join(sorted(sg['dtypes']))} {list(ssh)}")
                if suf == "weight_scale" and len(wsh) == 2 and len(ssh) == 2 and ssh[1]:
                    elems_per_row = wsh[1] * (2 if wdt == "U8" else 1)
                    parts.append(f"-> block {elems_per_row // ssh[1]} along K" if elems_per_row % ssh[1] == 0 else "-> irregular block")
        print(f"  {mod}\n      " + "\n      ".join(parts))

    unq = [k for k, g in groups.items() if k.endswith(".weight") and k[: -len(".weight")] not in quant
           and not any(k.endswith(f".{s}") for s in scale_suffixes) and category(k) != "vision"]
    print("\nnot quantized (weights kept in " + "/".join(sorted({d for k in unq for d in groups[k]["dtypes"]})) + "):")
    for k in unq:
        print(f"  {k}  {fmt_bytes(groups[k]['bytes'])}")


if __name__ == "__main__":
    main()
