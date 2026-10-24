"""Synthetic checkpoints in the layout of nvidia/Qwen3.8-Flash-Next-NVFP4, for the tests of the n-gram table reader
(tests/test_ple.py) and the loader (tests/test_qwen4exp_loader.py).

write_safetensors writes the tensors in the given file order (safetensors' own writer sorts them). The real
n-gram file does not store its shards in index order, and the tests need that case.

write_tiny_checkpoint writes a tiny reference model (tests/test_qwen4exp_small.py) as the nvidia checkpoint stores the
real one: per-expert NVFP4 tensors (weight, weight_scale, weight_scale_2, input_scale), everything else in BF16, a
layer whose experts span two files, the n-gram table as FP8 shards (out of order) with one bf16 weight_scale in
model-fp8-mtp-ple.safetensors, mtp.* and model.visual.* decoys, model.safetensors.index.json and config.json.
"""
from __future__ import annotations

import json
import os
import struct

import torch

from engine.weights.direct import DTYPES

NAMES = {v: k for k, v in DTYPES.items()}
P = "model.language_model."


def write_safetensors(path: str, items: list[tuple[str, torch.Tensor]], metadata: dict | None = None, pad: int = 0):
    """Writes the tensors in the order of items. pad: zero bytes before each tensor's data (moves the offsets)."""
    hdr, off, blobs = {}, 0, []
    for name, t in items:
        t = t.detach().cpu().contiguous()
        b = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        blobs.append(b"\0" * pad + b)
        off += pad
        hdr[name] = {"dtype": NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    if metadata:
        hdr["__metadata__"] = metadata
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def ngram_shards(table: torch.Tensor, n_shards: int, order: list[int], prefix: str) -> list[tuple[str, torch.Tensor]]:
    """The FP8 table [rows, d] as n_shards shards of rows / n_shards, in the file order `order`."""
    rows = table.shape[0] // n_shards
    return [(f"{prefix}.shard_{i}.weight", table[i * rows:(i + 1) * rows]) for i in order]


def tiny_table_fp8(rows: int, dim: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A random FP8 table [rows, dim] and its bf16 scale [1]."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, dim, generator=g) * 64).to(torch.float8_e4m3fn), torch.tensor([0.0078125], dtype=torch.bfloat16)


def write_tiny_checkpoint(path: str, model, n_ngram_shards: int = 4, shard_order=(2, 0, 3, 1)) -> dict:
    """Writes the tiny reference model (dense experts, fp32) as an nvidia-layout checkpoint into the directory path.
    Returns {"experts": {layer: the NVFP4 Experts written}, "table": fp8 table, "scale": bf16 scale, "split_layer": the
    layer whose experts span two files}."""
    from engine.model.qwen4exp import nvfp4_experts
    os.makedirs(path, exist_ok=True)
    cfg = model.cfg
    sd = {k: v for k, v in model.state_dict().items()}
    E, L = cfg.num_experts, cfg.num_hidden_layers
    files: dict[str, list[tuple[str, torch.Tensor]]] = {f"model-0000{i}-of-00003.safetensors": [] for i in (1, 2, 3)}
    f1, f2, f3 = files
    ple_prefix = None
    experts = {}
    split_layer = L // 2
    for k, v in sd.items():
        if ".mlp.experts." in k:
            continue
        name = "lm_head.weight" if k == "lm_head.weight" else P + k
        if ".ngram_embedding.weight" in k:
            ple_prefix = P + k[: -len(".weight")]
            continue
        if v.is_floating_point():
            v = v.to(torch.bfloat16)
        li = int(k.split(".")[1]) if k.startswith("layers.") else -1
        files[f1 if 0 <= li < split_layer else f3 if li < 0 else f2].append((name, v))
    for li in range(L):
        dense = model.layers[li].mlp.experts
        ins = torch.tensor([[0.01 + 0.001 * li, 0.01 + 0.001 * li, 0.02 + 0.001 * li]]).expand(E, 3)  # one value per layer and projection
        q = nvfp4_experts(dense, ins)
        experts[li] = q
        I = q.inter
        for e in range(E):
            target = f1 if li < split_layer or (li == split_layer and e < E // 2) else f2  # the split layer's experts span two files
            for j, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                w, sf = (q.w_gu[e, :I], q.sf_gu[e, :I]) if j == 0 else (q.w_gu[e, I:], q.sf_gu[e, I:]) if j == 1 else (q.w_dn[e], q.sf_dn[e])
                base = f"{P}layers.{li}.mlp.experts.{e}.{proj}"
                files[target] += [(base + ".weight", w), (base + ".weight_scale", sf), (base + ".weight_scale_2", q.gs[e, j].clone()),
                                  (base + ".input_scale", q.in_scale[e, j].clone())]
    rows = sd[ple_prefix[len(P):] + ".weight"].shape[0]
    table, scale = tiny_table_fp8(rows, cfg.ngram_dim)
    files[f3] += [("model.visual.blocks.0.attn.qkv.weight", torch.randn(12, 4).to(torch.bfloat16)),
                  ("mtp.fc_hidden.weight", torch.randn(cfg.hidden_size, cfg.hidden_size).to(torch.bfloat16))]
    files["model-fp8-mtp-ple.safetensors"] = (
        [("mtp.layers.0.mlp.experts.0.down_proj.weight", torch.randn(cfg.hidden_size, 32).to(torch.float8_e4m3fn)),
         ("mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv", torch.ones(1, 1, dtype=torch.bfloat16)),
         (ple_prefix + ".weight_scale", scale)]
        + ngram_shards(table, n_ngram_shards, list(shard_order), ple_prefix))
    wm = {}
    for fname, items in files.items():
        write_safetensors(os.path.join(path, fname), items, {"format": "pt"}, pad=0)
        wm.update(dict.fromkeys((n for n, _ in items), fname))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(nvidia_config(cfg, n_ngram_shards), f)
    return {"experts": experts, "table": table, "scale": scale, "split_layer": split_layer, "ple_prefix": ple_prefix}


def nvidia_config(cfg, n_ngram_shards: int) -> dict:
    """config.json in the nvidia layout (a text_config inside the multimodal config) for a Qwen4ExpConfig."""
    t = {k: getattr(cfg, k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "rms_norm_eps", "vocab_size",
        "max_position_embeddings", "moe_intermediate_size", "shared_expert_intermediate_size", "num_experts", "num_experts_per_tok",
        "norm_topk_prob", "hidden_act", "output_gate_type", "hc_count", "hc_lowrank", "ple_layer_ids", "ple_embed_dim",
        "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "seed",
        "eos_token_id", "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio")}
    t |= {"model_type": "qwen4_exp_text", "layer_types": list(cfg.layer_types), "split_ngram_parts": n_ngram_shards,
          "tie_word_embeddings": False, "bos_token_id": cfg.eos_token_id,
          "rope_parameters": {"rope_type": "default", "rope_theta": cfg.rope_theta, "partial_rotary_factor": cfg.partial_rotary_factor,
                              "mrope_section": list(cfg.mrope_section), "mrope_interleaved": cfg.mrope_interleaved}}
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "image_token_id": cfg.image_token_id,
            "text_config": t, "tie_word_embeddings": False}
