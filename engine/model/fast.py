"""Decode-path model (PLAN.md Phase 2): the Phase 1 architecture with every NVFP4 / FP8 linear
running on the weight-streaming GEMV kernels (csrc/gemv.cu, W4A16 / W8A16, activations bf16).
Weights stay quantized on the GPU (~22 GB). Everything that is not a quantized linear (norms,
GDN conv / delta rule, attention, rotary) is still the Phase 1 PyTorch code and is replaced
kernel by kernel in later steps.

Linears take any number of rows and issue the GEMV in chunks of 4, so prefill works (slowly:
every 4 prompt tokens stream all weights once) until Phase 3 brings a real prefill path.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
from safetensors import safe_open

from engine.kernels import ops
from engine.model.qwen35 import Qwen35Config, Qwen35ForCausalLM
from engine.weights.loader import PREFIX, SCALE_SUFFIXES, SKIP_PREFIXES, resolve

MAX_M = 4


class Nvfp4Linear(nn.Module):
    def __init__(self, w: torch.Tensor, sf: torch.Tensor, gscale: float, out_fp32: bool = False, in_scale: float = 1.0):
        super().__init__()
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("sf", sf, persistent=False)
        self.gscale, self.out_fp32, self.in_scale = float(gscale), out_fp32, float(in_scale)
        self.out_features, self.in_features = w.shape[0], w.shape[1] * 2

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.float32 if self.out_fp32 else torch.bfloat16)
        r2 = residual.reshape(-1, self.out_features).contiguous() if residual is not None else None
        for i in range(0, x2.shape[0], MAX_M):
            ops().nvfp4_gemv(x2[i:i + MAX_M], self.w, self.sf, self.gscale, None if r2 is None else r2[i:i + MAX_M], out[i:i + MAX_M])
        return out.view(*shp[:-1], self.out_features)


class Fp8Linear(nn.Module):
    def __init__(self, w: torch.Tensor, scale: float, in_scale: float = 1.0):
        super().__init__()
        self.register_buffer("w", w, persistent=False)
        self.scale, self.in_scale = float(scale), float(in_scale)
        self.out_features, self.in_features = w.shape

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
        r2 = residual.reshape(-1, self.out_features).contiguous() if residual is not None else None
        for i in range(0, x2.shape[0], MAX_M):
            ops().fp8_gemv(x2[i:i + MAX_M], self.w, self.scale, None if r2 is None else r2[i:i + MAX_M], out[i:i + MAX_M])
        return out.view(*shp[:-1], self.out_features)


class StackedFp8Linear(nn.Module):
    """Several FP8 linears that read the same input, stacked into one GEMV launch with per-row
    scales (each keeps its own checkpoint scale exactly). forward returns one output per part."""

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
        off = 0
        for p in parts:  # the parts keep working (prefill path) as views into the stacked weight
            p.w = self.w[off:off + p.out_features]
            off += p.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], sum(self.sizes), device=x.device, dtype=torch.bfloat16)
        for i in range(0, x2.shape[0], MAX_M):
            ops().fp8_gemv(x2[i:i + MAX_M], self.w, 1.0, None, out[i:i + MAX_M], self.rs)
        return [t.reshape(*shp[:-1], -1) for t in out.split(self.sizes, dim=-1)]


class SwiGLUMLP(nn.Module):
    """silu(gate) * up in one fused GEMV launch, then down."""

    def __init__(self, gate: Nvfp4Linear, up: Nvfp4Linear, down: Nvfp4Linear):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down

    def forward(self, x, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, self.gate.in_features).contiguous()
        h = torch.empty(x2.shape[0], self.gate.out_features, device=x.device, dtype=torch.bfloat16)
        for i in range(0, x2.shape[0], MAX_M):
            ops().nvfp4_swiglu(x2[i:i + MAX_M], self.gate.w, self.gate.sf, self.gate.gscale, self.up.w, self.up.sf, self.up.gscale, h[i:i + MAX_M])
        return self.down(h.view(*shp[:-1], -1), residual)


def load_fast_model(path_or_repo: str, device="cuda", verbose=True) -> Qwen35ForCausalLM:
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
        if t.dtype == torch.uint8 and base + ".weight_scale" in raw:
            insc = float(raw[base + ".input_scale"].float()) if base + ".input_scale" in raw else 1.0
            quant[mod] = Nvfp4Linear(t, raw[base + ".weight_scale"], float(raw[base + ".weight_scale_2"].float()), out_fp32=(mod == "lm_head"),
                                     in_scale=insc)
        elif t.dtype == torch.float8_e4m3fn and base + ".weight_scale" in raw:
            insc = float(raw[base + ".input_scale"].float()) if base + ".input_scale" in raw else 1.0
            quant[mod] = Fp8Linear(t, float(raw[base + ".weight_scale"].float()), in_scale=insc)
        elif t.dtype == torch.float8_e4m3fn:
            raise NotImplementedError(f"{name}: block-scaled FP8 is not supported on the decode path yet")
        else:
            plain[local] = t.to(torch.bfloat16) if t.is_floating_point() else t
    # load the non-quantized parameters, then swap the quantized linears in
    missing = set(model.state_dict()) - set(plain) - {m + ".weight" for m in quant}
    if missing:
        raise RuntimeError(f"missing tensors: {sorted(missing)[:5]}")
    model.load_state_dict(plain, strict=False, assign=True)
    for mod, q in quant.items():
        parent, _, child = mod.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, q)
    for layer in model.layers:
        if isinstance(layer.mlp.gate_proj, Nvfp4Linear):
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


# ----------------------------------------------------------------------------------------------
# Graph-capturable decode path: positions live on the device, attention reads seq_len on the GPU.
# ----------------------------------------------------------------------------------------------
from engine.model.qwen35 import Attention, GatedDeltaNet, ModelState, apply_rotary  # noqa: E402

ATTN_SPLITS = 32


class FastState(ModelState):
    """ModelState plus the device-side position (`pos_t`, int32 [B]) that graph replays advance.
    kv_fp8: store the attention KV cache as e4m3 with unit scale (saturating), halving its traffic."""

    def __init__(self, cfg, batch, max_seq_len, device, dtype=torch.bfloat16, kv_fp8: bool = False):
        super().__init__(cfg, batch, max_seq_len, device, dtype)
        self.kv_fp8 = kv_fp8
        if kv_fp8:
            for d in (self.k, self.v):
                for i in d:
                    d[i] = torch.zeros_like(d[i], dtype=torch.float8_e4m3fn)
        self.pos_t = torch.zeros(batch, dtype=torch.int32, device=device)
        self.arange = torch.arange(16, device=device)

    def reset(self):
        super().reset()
        self.pos_t.zero_()


class KernelAttention(Attention):
    """Gated GQA attention: KV written at device positions, split-KV decode kernel (any T <= 16)."""

    def forward(self, x, cos, sin, state: FastState, layer_idx: int, residual=None):
        B, T, _ = x.shape
        if hasattr(self, "qkv"):
            qp, kp, vp = self.qkv(x)
        else:
            qp, kp, vp = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        kc, vc = state.k[layer_idx], state.v[layer_idx]
        qp = qp.reshape(B, T, -1)
        q = torch.empty(B, self.num_heads, T, self.head_dim, device=x.device, dtype=torch.bfloat16)
        ops().attn_prologue(qp, kp, vp, self.q_norm.weight, self.k_norm.weight, self.inv_freq, state.pos_t,
                            kc, vc, q, self.q_norm.eps)
        attn = torch.empty(B, T, self.num_heads * self.head_dim, device=x.device, dtype=torch.bfloat16)
        ops().attn_decode(q, kc, vc, state.pos_t + T, attn, ATTN_SPLITS, self.head_dim ** -0.5, qp.contiguous())
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


_SIDE = {}


def _side_stream(device):
    if device not in _SIDE:
        _SIDE[device] = torch.cuda.Stream(device)
    return _SIDE[device]


class KernelGDN(GatedDeltaNet):
    """Single-token steps use the fused conv + delta-rule kernels (csrc/gdn_step.cu); multi-token
    prefill chunks fall back to the Phase 1 chunked PyTorch path."""

    def forward(self, x, state, layer_idx: int, residual=None):
        B, T, _ = x.shape
        if T != 1 or state is None:
            out = super().forward(x, state, layer_idx)
            return out if residual is None else residual + out
        if not hasattr(self, "w_ba"):  # in_proj_b and in_proj_a stacked: one tiny GEMV launch
            self.w_ba = torch.cat([self.in_proj_b.weight, self.in_proj_a.weight]).contiguous()
        # the tiny b/a GEMV runs on a side stream, overlapped with the big qkv/z weight stream
        x2 = x.view(B, -1)
        ba = torch.empty(B, self.w_ba.shape[0], device=x.device, dtype=torch.bfloat16)
        main = torch.cuda.current_stream()
        side = _side_stream(x.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            ops().bf16_gemv(x2, self.w_ba, ba)
        if hasattr(self, "qkvz"):
            mixed, z = self.qkvz(x)
            mixed, z = mixed.reshape(B, -1).contiguous(), z.reshape(B, -1).contiguous()
        else:
            mixed = self.in_proj_qkv(x).view(B, -1)
            z = self.in_proj_z(x).view(B, -1)
        main.wait_stream(side)
        b, a = ba[:, : self.num_v_heads].contiguous(), ba[:, self.num_v_heads:].contiguous()
        qkv = torch.empty_like(mixed)
        ops().gdn_conv(mixed, state.conv[layer_idx], self.conv1d.weight, qkv)
        o = torch.empty_like(z)
        ops().gdn_delta(qkv, z, b, a, self.A_log, self.dt_bias, self.norm.weight, state.rec[layer_idx], o,
                        self.num_k_heads, self.norm.eps)
        return self.out_proj(o.view(B, 1, -1), residual)


def fast_layer_forward(self, x, cos, sin, state):
    """DecoderLayer.forward with kernel norms and the residual adds fused into the output GEMVs."""
    h = self.input_layernorm(x)
    if self.block_type == "linear_attention":
        x = self.linear_attn(h, state, self.layer_idx, residual=x)
    else:
        x = self.self_attn(h, cos, sin, state, self.layer_idx, residual=x)
    return self.mlp(self.post_attention_layernorm(x), residual=x)


class FastQwen35(Qwen35ForCausalLM):
    kv_fp8 = False

    def new_state(self, batch: int, max_seq_len: int) -> FastState:
        p = self.embed_tokens.weight
        return FastState(self.cfg, batch, max_seq_len, p.device, p.dtype, kv_fp8=self.kv_fp8)

    def forward(self, input_ids: torch.Tensor, state: FastState, last_only: bool = False) -> torch.Tensor:
        B, T = input_ids.shape
        if state.pos + T > state.max_seq_len:
            raise ValueError(f"sequence length {state.pos + T} exceeds state max_seq_len {state.max_seq_len}")
        x = self.embed_tokens(input_ids)
        cos = sin = None  # RoPE is applied inside the fused attention prologue
        for layer in self.layers:
            x = layer(x, cos, sin, state)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x).float()
        state.pos_t += T
        state.pos += T
        return logits


def to_fast(model: Qwen35ForCausalLM, kv_fp8: bool = False) -> FastQwen35:
    """Switch a kernel-linear model (load_fast_model) onto the device-position decode path."""
    model.__class__ = FastQwen35
    model.kv_fp8 = kv_fp8
    for layer in model.layers:
        if layer.block_type == "full_attention":
            a = layer.self_attn
            a.__class__ = KernelAttention
            d = model.cfg.rotary_dim
            a.inv_freq = 1.0 / (model.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=model.embed_tokens.weight.device) / d))
            if isinstance(a.q_proj, Fp8Linear):  # one launch for q, k, v (the parts are dropped, memory reused)
                a.qkv = StackedFp8Linear([a.q_proj, a.k_proj, a.v_proj])
        else:
            g = layer.linear_attn
            g.__class__ = KernelGDN
            if isinstance(g.in_proj_qkv, Fp8Linear):
                g.qkvz = StackedFp8Linear([g.in_proj_qkv, g.in_proj_z])
                g.w_ba = torch.cat([g.in_proj_b.weight, g.in_proj_a.weight]).contiguous()
        layer.input_layernorm = KernelRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = KernelRMSNorm(layer.post_attention_layernorm)
        layer.forward = fast_layer_forward.__get__(layer)
    model.norm = KernelRMSNorm(model.norm)
    return model


class DecodeGraph:
    """One CUDA graph for a full decode step (T=1 per slot): embed -> 64 layers -> lm_head -> sampler.

    Capture runs on a fresh state and resets it afterwards (warm-up executes the step for real).
    Per step the host writes the input tokens into a static buffer, replays, and reads back the
    argmax ids (and optionally the logits)."""

    def __init__(self, model: FastQwen35, state: FastState):
        from engine.model.prefill import prepare_prefill
        from engine.runtime.sampler import SamplerParams, sample
        prepare_prefill(model)  # re-points weights (stacking); must happen before the graph records addresses
        self.model, self.state = model, state
        B = state.pos_t.shape[0]
        self.params = SamplerParams(B, model.cfg.vocab_size, state.pos_t.device)
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=state.pos_t.device)
        state.reset()
        state.pos = 1  # record the decode (has-previous-state) branches
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                sample(model(self.tok, state, last_only=True)[:, -1], self.params)
                state.pos = 1
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.logits = model(self.tok, state, last_only=True)[:, -1]
            self.next = sample(self.logits, self.params)
        state.reset()
        self.params.offset.zero_()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B] long on the device. Returns sampled ids [B] (device; greedy for slots with
        temperature 0, see self.params); logits in self.logits."""
        if self.state.pos + 1 > self.state.max_seq_len:
            raise ValueError("state full")
        self.tok.copy_(tokens.view(-1, 1))
        self.graph.replay()
        self.state.pos += 1
        return self.next
