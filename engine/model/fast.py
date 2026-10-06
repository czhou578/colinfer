"""Decode-path model: the reference architecture (engine/model/qwen35.py) with every op on the engine's CUDA kernels and
the positions on the device, so a decode step or a whole speculative cycle is one CUDA graph.

Weights (nvidia/Qwen3.8-27B-NVFP4, kept quantized on the GPU, ~20 GB):
  MLP + lm_head        NVFP4 (Nvfp4Linear): streamed by the tensor-core skinny GEMM (csrc/skinny.cu) at decode, read by
                       the CUTLASS NVFP4 GEMM at prefill (engine/model/prefill.py).
  attention / GDN      FP8 (Fp8Linear, StackedFp8Linear): prefill multiplies the FP8 weights (W8A8). Decode streams an
  projections          INT6 (attention) / INT5 (GDN) copy when its files exist (attach_decode_copies, tools/int6_requant.py:
                       ~9.5% faster decode, perplexity within 0.25% of the checkpoint), else the FP8 weights themselves.
  everything else      BF16 (embeddings, norms, the GDN conv and b / a gates).

Bit identity. Every decode linear runs on the skinny GEMM, whose rows are bit-identical whatever the number of rows
(1..16) in the launch; attention and the GDN recurrence likewise compute a row the same way at any width (csrc/
attn_decode.cu, csrc/gdn_step.cu). So plain decode, the speculative verify of k+1 rows and any batch of slots produce the
same bits for the same token, and speculation can never change an output.

The KV cache is fp8 (e4m3, unit scale, saturating): half of bf16's bytes, and decode attention reads it directly.

Entry points: load_fast_model(path) -> to_fast(model) -> [attach_decode_copies(model, path)] -> DecodeGraph / MtpCycle
(engine/spec/mtp.py) / prefill(). FastQwen35.forward is one decode step (T = 1 per slot); verify / commit are the
speculative halves (T = k + 1 rows, then the accepted prefix).
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
from safetensors import safe_open

from engine.kernels import ops
from engine.model.qwen35 import Attention, GatedDeltaNet, ModelState, Qwen35Config, Qwen35ForCausalLM
from engine.weights.loader import PREFIX, SCALE_SUFFIXES, SKIP_PREFIXES, resolve
from engine.weights.quantize import REQUANT_DIR

MAX_ROWS = 16  # rows per weight pass of the skinny GEMM: a verify of width * (k + 1) rows must fit
GEMV_ROWS = 8  # rows per bf16 GEMV launch (the GDN b / a gates)


# ------------------------------------------------------------------------------------------------------------ linears
def _rows(fn, x2, r2, out):
    """Run a skinny GEMM over any number of rows, MAX_ROWS at a time."""
    for i in range(0, x2.shape[0], MAX_ROWS):
        fn(x2[i:i + MAX_ROWS], None if r2 is None else r2[i:i + MAX_ROWS], out[i:i + MAX_ROWS])


class Nvfp4Linear(nn.Module):
    """NVFP4 weights: packed e2m1 w [N, K/2], e4m3 block scales sf [N, K/16], fp32 global scale; in_scale: the static
    activation scale prefill quantizes with. out_fp32: logits."""

    def __init__(self, w: torch.Tensor, sf: torch.Tensor, gscale: float, out_fp32: bool = False, in_scale: float = 1.0):
        super().__init__()
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("sf", sf, persistent=False)
        self.gscale, self.out_fp32, self.in_scale = float(gscale), out_fp32, float(in_scale)
        self.out_features, self.in_features = w.shape[0], w.shape[1] * 2

    def rows(self, x2, r2, out):
        _rows(lambda x, r, o: ops().skinny_nvfp4(x, self.w, self.sf, self.gscale, r, o), x2, r2, out)

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.float32 if self.out_fp32 else torch.bfloat16)
        r2 = residual.reshape(-1, self.out_features).contiguous() if residual is not None else None
        self.rows(x2, r2, out)
        return out.view(*shp[:-1], self.out_features)


class IntLinear(nn.Module):
    """INT6 / INT5 decode copy of an FP8 linear (engine/weights/quantize.py): low-nibble plane wlo [N, K/2], high-bit plane
    whi [N, K/4] (INT6) or [N, K/8] (INT5), e4m3 block scales sf [N, K/16], fp32 global scale."""

    def __init__(self, wlo: torch.Tensor, whi: torch.Tensor, sf: torch.Tensor, gscale: float):
        super().__init__()
        self.register_buffer("wlo", wlo, persistent=False)
        self.register_buffer("whi", whi, persistent=False)
        self.register_buffer("sf", sf, persistent=False)
        self.gscale = float(gscale)
        self.out_features, self.in_features = wlo.shape[0], wlo.shape[1] * 2

    def rows(self, x2, r2, out):
        _rows(lambda x, r, o: ops().skinny_int(x, self.wlo, self.whi, self.sf, self.gscale, r, o), x2, r2, out)


class Fp8Linear(nn.Module):
    """FP8 weights w [N, K] with a per-tensor scale (prefill's W8A8 GEMM reads them). dec: an optional IntLinear copy
    that decode streams instead (attach_decode_copies)."""

    def __init__(self, w: torch.Tensor, scale: float, in_scale: float = 1.0):
        super().__init__()
        self.register_buffer("w", w, persistent=False)
        self.scale, self.in_scale = float(scale), float(in_scale)
        self.out_features, self.in_features = w.shape
        self.dec = None

    def rows(self, x2, r2, out):
        if self.dec is not None:
            self.dec.rows(x2, r2, out)
        else:
            _rows(lambda x, r, o: ops().skinny_fp8(x, self.w, self.scale, r, o), x2, r2, out)

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
        r2 = residual.reshape(-1, self.out_features).contiguous() if residual is not None else None
        self.rows(x2, r2, out)
        return out.view(*shp[:-1], self.out_features)


class StackedFp8Linear(nn.Module):
    """FP8 linears that read the same input (attention q / k / v; GDN in_proj_qkv / in_proj_z) stacked into one weight
    pass, with per-row scales so each part keeps its checkpoint scale exactly. forward returns one output per part
    (column views of one [rows, sum(sizes)] tensor). The parts stay usable (prefill) as views of the stacked weight."""

    def __init__(self, parts: list):
        super().__init__()
        self.register_buffer("w", torch.cat([p.w for p in parts]).contiguous(), persistent=False)
        self.register_buffer("rs", torch.cat([torch.full((p.out_features,), p.scale, dtype=torch.float32, device=p.w.device)
                                              for p in parts]), persistent=False)
        self.sizes = [p.out_features for p in parts]
        self.in_features = parts[0].in_features
        if len({p.in_scale for p in parts}) != 1:
            raise ValueError("stacked FP8 parts must share the input scale")
        self.in_scale = parts[0].in_scale
        self.dec = None
        off = 0
        for p in parts:
            p.w = self.w[off:off + p.out_features]
            off += p.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], sum(self.sizes), device=x.device, dtype=torch.bfloat16)
        if self.dec is not None:
            self.dec.rows(x2, None, out)
        else:
            _rows(lambda x_, r, o: ops().skinny_fp8(x_, self.w, 1.0, r, o, self.rs), x2, None, out)
        return [t.reshape(*shp[:-1], -1) for t in out.split(self.sizes, dim=-1)]


class LinearGroup(nn.Module):
    """Separate linears on the same input, called as one (KernelAttention.qkv where the parts are not stacked: the MTP
    drafter, whose parts keep their own NVFP4 global scales). forward returns one output per part."""

    def __init__(self, parts: list):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, x):
        return [p(x) for p in self.parts]


class SwiGLUMLP(nn.Module):
    """silu(gate) * up in one fused weight pass (skinny_swiglu), then down with the residual added in its epilogue."""

    def __init__(self, gate: Nvfp4Linear, up: Nvfp4Linear, down: Nvfp4Linear):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.gate.in_features).contiguous()
        h = torch.empty(x2.shape[0], self.gate.out_features, device=x.device, dtype=torch.bfloat16)
        g, u = self.gate, self.up
        _rows(lambda x_, r, o: ops().skinny_swiglu(x_, g.w, g.sf, g.gscale, u.w, u.sf, u.gscale, o), x2, None, h)
        return self.down(h.view(*shp[:-1], -1), residual)


def load_fast_model(path_or_repo: str, device="cuda", verbose=True) -> Qwen35ForCausalLM:
    """The reference module tree with the checkpoint's NVFP4 / FP8 linears swapped for kernel modules (weights stay
    quantized). Call to_fast() on the result."""
    path = resolve(path_or_repo)
    cfg = Qwen35Config.from_checkpoint(path)
    t0 = time.time()
    raw: dict[str, torch.Tensor] = {}
    files = sorted(set(json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"].values()))
    for f in files:
        with safe_open(os.path.join(path, f), framework="pt", device=device) as sf:
            for n in sf.keys():
                if not n.startswith(SKIP_PREFIXES):
                    raw[n] = sf.get_tensor(n)
    with torch.device("meta"):
        model = Qwen35ForCausalLM(cfg)
    quant: dict[str, nn.Module] = {}
    plain: dict[str, torch.Tensor] = {}
    for name, t in raw.items():
        if name.endswith(SCALE_SUFFIXES):
            continue
        local = name[len(PREFIX):] if name.startswith(PREFIX) else name
        base = name[: -len(".weight")]
        mod = local[: -len(".weight")] if local.endswith(".weight") else None
        insc = float(raw[base + ".input_scale"].float()) if base + ".input_scale" in raw else 1.0
        if t.dtype == torch.uint8 and base + ".weight_scale" in raw:
            quant[mod] = Nvfp4Linear(t, raw[base + ".weight_scale"], float(raw[base + ".weight_scale_2"].float()), out_fp32=(mod == "lm_head"),
                                     in_scale=insc)
        elif t.dtype == torch.float8_e4m3fn and base + ".weight_scale" in raw:
            quant[mod] = Fp8Linear(t, float(raw[base + ".weight_scale"].float()), in_scale=insc)
        elif t.dtype == torch.float8_e4m3fn:
            raise NotImplementedError(f"{name}: block-scaled FP8 checkpoints are not supported")
        else:
            plain[local] = t.to(torch.bfloat16) if t.is_floating_point() else t
    missing = set(model.state_dict()) - set(plain) - {m + ".weight" for m in quant}
    if missing:
        raise RuntimeError(f"missing tensors: {sorted(missing)[:5]}")
    model.load_state_dict(plain, strict=False, assign=True)
    for mod, q in quant.items():
        parent, _, child = mod.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, q)
    for layer in model.layers:
        layer.mlp = SwiGLUMLP(layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [n for n, b in model.named_buffers() if b.is_meta]
    if leftover:
        raise RuntimeError(f"tensors left on meta: {leftover[:5]}")
    del raw, plain
    torch.cuda.empty_cache()
    model.eval()
    if verbose:
        n4 = sum(isinstance(m, Nvfp4Linear) for m in model.modules())
        n8 = sum(isinstance(m, Fp8Linear) for m in model.modules())
        print(f"[fast] {os.path.basename(path)}: {n4} NVFP4 + {n8} FP8 kernel linears, "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU, loaded in {time.time() - t0:.0f}s")
    return model


# ------------------------------------------------------------------------------------------------------------ state
class FastState(ModelState):
    """ModelState with an fp8 KV cache and the device-side per-slot positions `pos_t` (int32 [B]) that graph replays
    advance. active (int32 [B]): decode updates only slots with active == 1 (idle or prefilling slots inside a batched
    step are masked off). view(lo, hi) shares the tensors of slots [lo, hi)."""

    def __init__(self, cfg, batch, max_seq_len, device, dtype=torch.bfloat16):
        super().__init__(cfg, batch, max_seq_len, device, dtype, kv_dtype=torch.float8_e4m3fn)
        self.pos_t = torch.zeros(batch, dtype=torch.int32, device=device)
        self.active = torch.ones(batch, dtype=torch.int32, device=device)

    def reset(self):
        super().reset()
        self.pos_t.zero_()

    def view(self, lo: int, hi: int) -> "FastState":
        v = FastState.__new__(FastState)
        v.cfg, v.max_seq_len, v.pos = self.cfg, self.max_seq_len, self.pos
        v.conv = {i: t[lo:hi] for i, t in self.conv.items()}
        v.rec = {i: t[lo:hi] for i, t in self.rec.items()}
        v.k = {i: t[lo:hi] for i, t in self.k.items()}
        v.v = {i: t[lo:hi] for i, t in self.v.items()}
        v.pos_t, v.active = self.pos_t[lo:hi], self.active[lo:hi]
        return v


# ------------------------------------------------------------------------------------------------------------ layers
class KernelAttention(Attention):
    """Gated GQA attention on T new rows per slot (1 for decode, k + 1 for verify): one stacked q / k / v pass, the fused
    prologue (q / k norm, partial RoPE, fp8 KV write at the device positions), multi-row tensor-core attention with the
    output gate fused, o_proj with the residual."""

    def forward(self, x, cos, sin, state: FastState, layer_idx: int, residual=None):
        B, T, _ = x.shape
        qp, kp, vp = self.qkv(x)
        kc, vc = state.k[layer_idx], state.v[layer_idx]
        qp = qp.reshape(B, T, -1)
        q = torch.empty(B, self.num_heads, T, self.head_dim, device=x.device, dtype=torch.bfloat16)
        ops().attn_prologue(qp, kp, vp, self.q_norm.weight, self.k_norm.weight, self.inv_freq, state.pos_t, kc, vc, q, self.q_norm.eps,
                            state.active)
        attn = torch.empty(B, T, self.num_heads * self.head_dim, device=x.device, dtype=torch.bfloat16)
        ops().attn_decode(q, kc, vc, state.pos_t + T, attn, self.head_dim ** -0.5, qp.contiguous())
        return self.o_proj(attn, residual)


class KernelRMSNorm(nn.Module):
    """Drop-in for qwen35.RMSNorm on [..., K] bf16 rows (same parameter, same rounding)."""

    def __init__(self, norm):
        super().__init__()
        self.weight, self.eps = norm.weight, norm.eps

    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(x2)
        ops().rmsnorm(x2, self.weight, self.eps, out)
        return out.view(x.shape)


_SIDE: dict = {}


def side_stream(device) -> torch.cuda.Stream:
    """A second stream for work that can overlap the main one (a parallel branch when captured into a graph)."""
    if device not in _SIDE:
        _SIDE[device] = torch.cuda.Stream(device)
    return _SIDE[device]


class KernelGDN(GatedDeltaNet):
    """Gated DeltaNet on T new tokens per slot (csrc/gdn_step.cu).

    Plain decode (T = 1): the conv and delta rule run on the slot's state and advance it (slots with active == 0 keep
    theirs). Speculative verify (state.spec): outputs for all T tokens, state untouched, the inputs kept for commit(),
    which advances the state by the accepted count. The mixed / z / b / a tensors stay column views of the projection
    outputs (the kernels take row strides), and the tiny b / a GEMV runs on a side stream beside the qkv / z weight
    pass."""

    def forward(self, x, state: FastState, layer_idx: int, residual=None):
        B, T, _ = x.shape
        ba = torch.empty(B * T, self.w_ba.shape[0], device=x.device, dtype=torch.bfloat16)
        x2 = x.reshape(B * T, -1)
        main, side = torch.cuda.current_stream(), side_stream(x.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for i in range(0, B * T, GEMV_ROWS):
                ops().bf16_gemv(x2[i:i + GEMV_ROWS], self.w_ba, ba[i:i + GEMV_ROWS])
        mixed, z = self.qkvz(x)
        main.wait_stream(side)
        mixed, z = mixed.reshape(B, T, -1), z.reshape(B, T, -1)
        ba = ba.view(B, T, -1)
        b, a = ba[..., : self.num_v_heads], ba[..., self.num_v_heads:]
        conv, rec = state.conv[layer_idx], state.rec[layer_idx]
        qkv = torch.empty(B, T, mixed.shape[-1], device=x.device, dtype=torch.bfloat16)
        ops().gdn_conv(mixed, conv, self.conv1d.weight, qkv)
        o = torch.empty(B, T, z.shape[-1], device=x.device, dtype=torch.bfloat16)
        if getattr(state, "spec", False):
            ops().gdn_delta(qkv, z, b, a, self.A_log, self.dt_bias, self.norm.weight, rec, o, self.num_k_heads, self.norm.eps)
            self._spec = (mixed, qkv, z, b, a)
        else:
            assert T == 1, "multi-token prompts go through engine/model/prefill.py"
            ops().gdn_conv_commit(mixed, conv, state.active)
            ops().gdn_delta(qkv, z, b, a, self.A_log, self.dt_bias, self.norm.weight, rec, o, self.num_k_heads, self.norm.eps, state.active)
        return self.out_proj(o.view(B, T, -1), residual)

    def commit(self, state: FastState, layer_idx: int, n: torch.Tensor):
        """Advance the conv / recurrent state by the first n[b] tokens of the last verify."""
        mixed, qkv, z, b, a = self._spec
        ops().gdn_conv_commit(mixed, state.conv[layer_idx], n)
        ops().gdn_delta(qkv, z, b, a, self.A_log, self.dt_bias, self.norm.weight, state.rec[layer_idx], None, self.num_k_heads, self.norm.eps, n)


def fast_layer_forward(self, x, cos, sin, state):
    """DecoderLayer.forward with kernel norms and the residual adds fused into the output GEMMs."""
    h = self.input_layernorm(x)
    if self.block_type == "linear_attention":
        x = self.linear_attn(h, state, self.layer_idx, residual=x)
    else:
        x = self.self_attn(h, cos, sin, state, self.layer_idx, residual=x)
    return self.mlp(self.post_attention_layernorm(x), residual=x)


# ------------------------------------------------------------------------------------------------------------ model
class FastQwen35(Qwen35ForCausalLM):
    def new_state(self, batch: int, max_seq_len: int) -> FastState:
        return FastState(self.cfg, batch, max_seq_len, self.embed_tokens.weight.device)

    def forward(self, input_ids: torch.Tensor, state: FastState) -> torch.Tensor:
        """One decode step: input_ids [B, 1] at positions state.pos_t -> fp32 logits [B, vocab]; pos_t += active."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, None, None, state)
        logits = self.lm_head(self.norm(x)).float()[:, -1]
        state.pos_t += state.active
        return logits

    def verify(self, input_ids: torch.Tensor, state: FastState):
        """Speculative verify of T rows per slot [y, d1..dk]: (fp32 logits [B, T, V], post-norm hidden [B, T, H]). KV is
        written for all T rows; the GDN state and the positions are left for commit()."""
        state.spec = True
        try:
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x, None, None, state)
            h = self.norm(x)
            logits = self.lm_head(h).float()
        finally:
            state.spec = False
        return logits, h

    def commit(self, state: FastState, n: torch.Tensor):
        """Accept the first n[b] (int32, device) of the T verified rows per slot: GDN state forward, positions += n."""
        for i, layer in enumerate(self.layers):
            if layer.block_type == "linear_attention":
                layer.linear_attn.commit(state, i, n)
        state.pos_t += n


def to_fast(model: Qwen35ForCausalLM) -> FastQwen35:
    """Switch a load_fast_model() model onto the kernel decode path: kernel attention / GDN / norms, stacked projections."""
    model.__class__ = FastQwen35
    d = model.cfg.rotary_dim
    inv_freq = 1.0 / (model.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=model.embed_tokens.weight.device) / d))
    for layer in model.layers:
        if layer.block_type == "full_attention":
            a = layer.self_attn
            a.__class__ = KernelAttention
            a.inv_freq = inv_freq
            a.qkv = StackedFp8Linear([a.q_proj, a.k_proj, a.v_proj])  # one weight pass for q, k, v
        else:
            g = layer.linear_attn
            g.__class__ = KernelGDN
            g.qkvz = StackedFp8Linear([g.in_proj_qkv, g.in_proj_z])
            g.w_ba = torch.cat([g.in_proj_b.weight, g.in_proj_a.weight]).contiguous()
        layer.input_layernorm = KernelRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = KernelRMSNorm(layer.post_attention_layernorm)
        layer.forward = fast_layer_forward.__get__(layer)
    model.norm = KernelRMSNorm(model.norm)
    return model


# ------------------------------------------------------------------------------------------------------------ decode copies
def decode_copies_paths(path_or_repo: str) -> list[str]:
    """INT6 attention + INT5 GDN decode copies of the checkpoint's FP8 projections (tools/int6_requant.py --bits 6 --filter
    self_attn; --bits 5 --filter linear_attn). WikiText perplexity -0.21%, Python code +0.23% against the FP8 weights."""
    d = os.path.join(REQUANT_DIR, os.path.basename(resolve(path_or_repo)))
    return [os.path.join(d, f) for f in ("attn_gdn_int6_self_attn.safetensors", "attn_gdn_int5_linear_attn.safetensors")]


def attach_decode_copies(model: FastQwen35, files) -> int:
    """Point the decode path of the FP8 attention / GDN projections at INT6 / INT5 copies (prefill keeps the FP8 weights).
    files: paths of tools/int6_requant.py outputs (each may cover some of the projections). Stacked projections share a
    global scale in the file, so they stay one launch. Call after to_fast and before capturing graphs. Returns the number
    of linears attached."""
    n = 0
    for file in files:
        with safe_open(file, framework="pt", device=str(model.embed_tokens.weight.device)) as f:
            keys = set(f.keys())

            def copy_of(names):
                if not all(PREFIX + m + ".qweight_lo" in keys for m in names):
                    return None
                gs = {float(f.get_tensor(PREFIX + m + ".weight_scale_2")) for m in names}
                assert len(gs) == 1, names
                cat = lambda suffix: torch.cat([f.get_tensor(PREFIX + m + suffix) for m in names]).contiguous()  # noqa: E731
                return IntLinear(cat(".qweight_lo"), cat(".qweight_hi"), cat(".weight_scale"), gs.pop())
            for i, layer in enumerate(model.layers):
                p = f"layers.{i}."
                if layer.block_type == "full_attention":
                    a = layer.self_attn
                    parts = ((a.qkv, [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"]), (a.o_proj, [p + "self_attn.o_proj"]))
                else:
                    g = layer.linear_attn
                    parts = ((g.qkvz, [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"]), (g.out_proj, [p + "linear_attn.out_proj"]))
                for mod, names in parts:
                    d = copy_of(names)
                    if d is not None:
                        mod.dec = d
                        n += len(names)
    return n


# ------------------------------------------------------------------------------------------------------------ graph
class DecodeGraph:
    """One CUDA graph for a plain decode step (T = 1 per slot): embed -> 64 layers -> lm_head -> sampler.

    Capture runs on a fresh state and resets it afterwards (warm-up executes the step for real). Per step the host writes
    the input tokens into a static buffer, replays, and reads back the sampled ids (logits in self.logits)."""

    def __init__(self, model: FastQwen35, state: FastState, params=None):
        """params: optional SamplerParams (B slots) shared with other graphs; default: a private one."""
        from engine.model.prefill import prepare_prefill
        from engine.runtime.sampler import SamplerParams, sample
        prepare_prefill(model)  # re-points weights (stacking); must happen before the graph records addresses
        self.model, self.state = model, state
        B = state.pos_t.shape[0]
        self.params = params if params is not None else SamplerParams(B, model.cfg.vocab_size, state.pos_t.device)
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=state.pos_t.device)
        state.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                sample(model(self.tok, state), self.params, state.pos_t)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.logits = model(self.tok, state)
            self.next = sample(self.logits, self.params, state.pos_t)  # pos_t now holds the predicted token's position
        state.reset()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B] long on the device. Returns sampled ids [B] (device; greedy for slots with temperature 0)."""
        self.tok.copy_(tokens.view(-1, 1))
        self.graph.replay()
        return self.next
