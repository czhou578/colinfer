"""Checkpoint loading for the Phase 1 reference model.

Loads a Qwen3.5-family safetensors checkpoint into Qwen35ForCausalLM on the GPU, dequantizing
FP8 (128x128 block scales) and NVFP4 (e2m1 + e4m3 block-16 scales + fp32 global scale) to BF16
on the fly. Slow and simple on purpose: this is how the quantized weights get validated before
any custom kernel exists (PLAN.md Phase 1).

Usage:
    model = load_model("Qwen/Qwen3.8-27B")            # repo id in the HF cache, or a local dir
"""
from __future__ import annotations

import json
import os
import time

import torch
from safetensors import safe_open

from engine.model.qwen35 import Qwen35Config, Qwen35ForCausalLM
from engine.weights import quant_emul

PREFIX = "model.language_model."
SKIP_PREFIXES = ("model.visual.", "mtp.")
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".weight_scale_inv", ".input_scale")


def resolve(path_or_repo: str) -> str:
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, local_files_only=True)


E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_nvfp4(packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """packed U8 [N, K/2] (element 2i in the low nibble, 2i+1 in the high nibble),
    block_scale e4m3 [N, K/16], global_scale f32 scalar -> [N, K] out_dtype."""
    n, half = packed.shape
    lut = E2M1_LUT.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    w = torch.stack((lo, hi), dim=-1).reshape(n, half * 2)  # interleave: 2i from low, 2i+1 from high
    scale = block_scale.to(torch.float32).repeat_interleave(16, dim=1)[:, : half * 2] * global_scale.to(torch.float32)
    return (w * scale).to(out_dtype)


def dequant_fp8_block(w8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128, out_dtype=torch.bfloat16) -> torch.Tensor:
    """w8 e4m3 [N, K], scale_inv [ceil(N/128), ceil(K/128)] -> [N, K] out_dtype."""
    n, k = w8.shape
    s = scale_inv.to(torch.float32).repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=1)[:, :k]
    return (w8.to(torch.float32) * s).to(out_dtype)


def dequant_fp8_tensor(w8: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    return (w8.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)


def load_state_dict(path: str, device="cuda", dtype=torch.bfloat16, verbose=True, factored=False) -> tuple[dict, dict, dict]:
    """Returns (state_dict with model-local names, info dict, quant meta). All weights in `dtype` on `device`.

    factored=False: quantized weights are fully dequantized (weight-only reference path).
    factored=True:  quantized weights keep only their exact unscaled values (e2m1 * block scale, or
    e4m3); per-tensor weight and input scales go to quant meta {module: {kind, w_scale, in_scale}}
    for engine.weights.quant_emul.QuantLinear."""
    idx_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        files = sorted(set(json.load(open(idx_file))["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    raw: dict[str, torch.Tensor] = {}
    t0 = time.time()
    total = 0
    for f in files:
        with safe_open(os.path.join(path, f), framework="pt", device=device) as sf:
            for name in sf.keys():
                if name.startswith(SKIP_PREFIXES):
                    continue
                t = sf.get_tensor(name)
                total += t.numel() * t.element_size()
                raw[name] = t
    if verbose:
        print(f"[loader] read {total / 1e9:.1f} GB from {len(files)} shards in {time.time() - t0:.1f}s")

    sd: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    info = dict(nvfp4=0, fp8_block=0, fp8_tensor=0, passthrough=0)

    def scalar(n, default=1.0):
        return float(raw[n].float().max()) if n in raw else default
    for name, t in raw.items():
        if name.endswith(SCALE_SUFFIXES):
            continue
        local = name[len(PREFIX):] if name.startswith(PREFIX) else name
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base is not None and t.dtype == torch.uint8 and (base + ".weight_scale") in raw:
            gs = raw.get(base + ".weight_scale_2", torch.ones((), device=t.device))
            if factored:
                sd[local] = dequant_nvfp4(t, raw[base + ".weight_scale"], torch.ones((), device=t.device), dtype)
                meta[local[: -len(".weight")]] = dict(kind="nvfp4", w_scale=float(gs.float()), in_scale=scalar(base + ".input_scale"))
            else:
                sd[local] = dequant_nvfp4(t, raw[base + ".weight_scale"], gs, dtype)
            info["nvfp4"] += 1
        elif base is not None and t.dtype == torch.float8_e4m3fn and (base + ".weight_scale_inv") in raw:
            sd[local] = dequant_fp8_block(t, raw[base + ".weight_scale_inv"], 128, dtype)
            info["fp8_block"] += 1
        elif base is not None and t.dtype == torch.float8_e4m3fn and (base + ".weight_scale") in raw:
            if factored:
                sd[local] = t.to(dtype)
                meta[local[: -len(".weight")]] = dict(kind="fp8", w_scale=scalar(base + ".weight_scale"), in_scale=scalar(base + ".input_scale"))
            else:
                sd[local] = dequant_fp8_tensor(t, raw[base + ".weight_scale"], dtype)
            info["fp8_tensor"] += 1
        else:
            sd[local] = t.to(dtype) if t.is_floating_point() else t
            info["passthrough"] += 1
    del raw
    if verbose:
        print(f"[loader] tensors: {info}  ({time.time() - t0:.1f}s)")
    return sd, info, meta


def load_model(path_or_repo: str, device="cuda", dtype=torch.bfloat16, verbose=True, emulate: str | None = None) -> Qwen35ForCausalLM:
    """emulate: comma list of quant_emul.EFFECTS (or 'all') to reproduce W4A4/W8A8 kernel numerics."""
    effects = quant_emul.parse_effects(emulate)
    path = resolve(path_or_repo)
    cfg = Qwen35Config.from_checkpoint(path)
    with torch.device("meta"):
        model = Qwen35ForCausalLM(cfg)
    sd, _, meta = load_state_dict(path, device, dtype, verbose, factored=bool(effects))
    if "fp8_requant" in effects:
        n = quant_emul.requant_fused_fp8(meta, sd)
        if verbose:
            print(f"[loader] fp8_requant: re-rounded {n} fused FP8 shards onto the group max scale")
    expected = set(model.state_dict().keys())
    missing = expected - sd.keys()
    unexpected = sd.keys() - expected
    if cfg.tie_word_embeddings and "lm_head.weight" in missing:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]
        missing.discard("lm_head.weight")
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(sd, strict=True, assign=True)
    if effects:
        counts = quant_emul.install(model, meta, effects)
        if verbose:
            print(f"[loader] emulating {sorted(effects)} on {counts}")
    model.eval()
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"[loader] {os.path.basename(path)}: {n / 1e9:.2f} B params on {device} as {dtype}")
    return model
