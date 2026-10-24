"""Loads nvidia/Qwen3.8-Flash-Next-NVFP4 into the reference module tree (engine/model/qwen4exp.py) with stacked experts.

The checkpoint stores each routed expert as its own tensors: per projection (gate_proj, up_proj, down_proj) an NVFP4
weight uint8 [N, K / 2], its block scales weight_scale e4m3 [N, K / 16], a global scale weight_scale_2 f32 and an
input_scale f32. The kernels want one stacked tensor per layer. The loader preallocates the stacked tensors of every
layer (ExpertWeights with storage "nvfp4": w_gu, sf_gu, w_dn, sf_dn on the device, gs and in_scale [E, 3]) and reads
each expert's bytes straight into its slot (engine.weights.direct.read_tensors with into=). Thus no expert is copied a
second time, and the GPU holds no transient bytes. The gate rows come first in w_gu / sf_gu, then the up rows. The
block scales stay row-major: the kernel module swizzles its own layout. A layer's experts may span two files; the
loader reads by name, so this needs nothing special.

Everything else is BF16 and loads as is. The loader skips the vision tower (model.visual.*), the MTP block (mtp.*) and
the n-gram shards by name. It checks the PLE buffers of the checkpoint (layer_multipliers, ngram_heads_vocab_sizes,
ngram_heads_offsets) against the values that engine/weights/ple.py computes, and attaches an NgramTable over the
n-gram file to each PLE layer: the 51 GB table stays on disk.

Usage:
    model = load_qwen4exp("nvidia/Qwen3.8-Flash-Next-NVFP4")   # repo id in the HF cache, or a local dir
"""
from __future__ import annotations

import os
import time

import torch

from engine.model.qwen4exp import Qwen4ExpConfig, Qwen4ExpForCausalLM
from engine.weights.direct import read_tensors, release_buffers
from engine.weights.loader import PREFIX, SKIP_PREFIXES, resolve, weight_map
from engine.weights.ple import NgramTable, ngram_params

PROJS = ("gate_proj", "up_proj", "down_proj")  # the column of each projection in gs / in_scale


def skipped(name: str) -> bool:
    """The tensors the loader never reads: the vision tower, the MTP block and the n-gram table (read per row)."""
    return name.startswith(SKIP_PREFIXES) or ".ple_embedding.ngram_embedding." in name


def expert_destinations(model: Qwen4ExpForCausalLM, gs: torch.Tensor, ins: torch.Tensor) -> dict[str, torch.Tensor]:
    """{checkpoint name: its slot} for every routed expert tensor: the stacked device tensors of each layer, and the
    CPU tensors gs / ins [layers, E, 3] for the scalars."""
    into = {}
    for li, layer in enumerate(model.layers):
        ex = layer.mlp.experts
        I = ex.inter
        for e in range(ex.num_experts):
            b = f"{PREFIX}layers.{li}.mlp.experts.{e}."
            parts = {"gate_proj": (ex.w_gu[e, :I], ex.sf_gu[e, :I]), "up_proj": (ex.w_gu[e, I:], ex.sf_gu[e, I:]),
                     "down_proj": (ex.w_dn[e], ex.sf_dn[e])}
            for j, p in enumerate(PROJS):
                w, sf = parts[p]
                into |= {b + p + ".weight": w, b + p + ".weight_scale": sf, b + p + ".weight_scale_2": gs[li, e, j],
                         b + p + ".input_scale": ins[li, e, j]}
    return into


def check_ngram_buffers(cfg: Qwen4ExpConfig, plain: dict[str, torch.Tensor]):
    """Raises if the checkpoint's hash buffers differ from the computed ones (the n-gram ids would be wrong)."""
    for li, j in cfg.ple_layers.items():
        p = ngram_params(cfg, j)
        b = f"layers.{li}.ple.ple_embedding."
        for name, want in (("layer_multipliers", p.multipliers), ("ngram_heads_vocab_sizes", p.sizes), ("ngram_heads_offsets", p.offsets)):
            got = plain.get(b + name)
            if got is None or got.tolist() != list(want):
                raise ValueError(f"{b + name}: the checkpoint has {None if got is None else got.tolist()}, the config gives {list(want)}")


def load_qwen4exp(path_or_repo: str, device="cuda", log=print) -> Qwen4ExpForCausalLM:
    """The reference model (BF16, NVFP4 routed experts stacked per layer, the n-gram table on disk) on `device`."""
    t0 = time.time()
    path = resolve(path_or_repo)
    cfg = Qwen4ExpConfig.from_checkpoint(path)
    wm = weight_map(path)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(cfg, experts="nvfp4")
    L, E, H, I = cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
    for layer in model.layers:
        ex = layer.mlp.experts
        ex.w_gu = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=device)
        ex.sf_gu = torch.empty(E, 2 * I, H // 16, dtype=torch.float8_e4m3fn, device=device)
        ex.w_dn = torch.empty(E, H, I // 2, dtype=torch.uint8, device=device)
        ex.sf_dn = torch.empty(E, H, I // 16, dtype=torch.float8_e4m3fn, device=device)
    gs = torch.zeros(L, E, 3)
    ins = torch.zeros(L, E, 3)
    into = expert_destinations(model, gs, ins)
    files: dict[str, list[str]] = {}
    for name, f in wm.items():
        if not skipped(name):
            files.setdefault(f, []).append(name)
    missing = sorted(set(into) - set(wm))
    if missing:
        raise RuntimeError(f"missing expert tensors: {missing[:5]}")
    plain: dict[str, torch.Tensor] = {}
    try:
        for f in sorted(files):
            got = read_tensors(os.path.join(path, f), files[f], device, into=into)
            plain.update({(n[len(PREFIX):] if n.startswith(PREFIX) else n): t for n, t in got.items()})
    finally:
        release_buffers()
    for li, layer in enumerate(model.layers):
        layer.mlp.experts.gs = gs[li].to(device)
        layer.mlp.experts.in_scale = ins[li].to(device)
    check_ngram_buffers(cfg, plain)
    expected = set(model.state_dict())
    experts = {n for n in expected if ".mlp.experts." in n}
    tables = {n for n in expected if n.endswith(".ngram_embedding.weight")}
    unexpected = set(plain) - expected
    absent = expected - set(plain) - experts - tables
    if unexpected or absent:
        raise RuntimeError(f"state dict mismatch: missing={sorted(absent)[:5]} unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(plain, strict=False, assign=True)
    for li in cfg.ple_layers:
        emb = model.layers[li].ple.ple_embedding
        prefix = f"{PREFIX}layers.{li}.ple.ple_embedding.ngram_embedding"
        f = wm.get(prefix + ".shard_0.weight")
        if f is None:
            raise RuntimeError(f"no n-gram table {prefix}.shard_<i>.weight in the checkpoint")
        table = NgramTable(os.path.join(path, f), prefix)
        if (table.n_rows, table.dim) != (emb.params.padded_rows, cfg.ngram_dim):
            raise ValueError(f"{prefix}: {table.n_rows} rows x {table.dim}, the config gives {emb.params.padded_rows} x {cfg.ngram_dim}")
        emb.table, emb.ngram_embedding = table, None
    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [n for n, b in model.named_buffers() if b.is_meta]
    if leftover:
        raise RuntimeError(f"tensors left on meta: {leftover[:5]}")
    model.eval()
    shared = bool(torch.equal(gs[..., 0], gs[..., 1]))
    per_layer = bool((ins == ins[:, :1]).all())
    nbytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    log(f"[qwen4exp] {os.path.basename(os.path.normpath(path))}: {nbytes / 1e9:.1f} GB on {device} in {time.time() - t0:.1f}s; "
        f"gate/up global scales equal: {shared}; one input scale per layer and projection: {per_layer}")
    return model
