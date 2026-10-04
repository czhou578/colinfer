"""Text-only Qwen3.5 / 3.6 / 3.8 family in plain PyTorch (PLAN.md Phase 1).

Mirrors transformers' modeling_qwen3_5.py op for op (same dtypes, same op order) so that greedy
generation is token-exact against the HF reference. Nothing here is fast; it is the correctness
reference every later kernel is tested against.

Architecture (Qwen3.8-27B): 64 layers, 48 Gated DeltaNet (linear attention) + 16 gated full
attention (GQA 24/4, head_dim 256, partial RoPE on the first 64 dims, theta 1e7), SwiGLU MLP,
zero-centered RMSNorm (y = norm(x) * (1 + w)), untied lm_head.

State is explicit (ModelState): per GDN layer a conv window [B, C, K-1] (bf16) and a recurrent
state [B, Hv, dk, dv] (fp32); per attention layer a static K/V cache [B, Hkv, max_len, D] (bf16).
Module names match the HF text model with the `model.language_model.` prefix removed, so the
loader is a prefix strip.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen35Config:
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    layer_types: list[str] = field(default_factory=list)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    @classmethod
    def from_checkpoint(cls, path: str) -> "Qwen35Config":
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("text_config", c)
        rp = t.get("rope_parameters", {})
        return cls(
            hidden_size=t["hidden_size"],
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_hidden_layers"],
            layer_types=list(t["layer_types"]),
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"],
            partial_rotary_factor=rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)),
            rope_theta=rp.get("rope_theta", t.get("rope_theta", 1e7)),
            linear_num_key_heads=t["linear_num_key_heads"],
            linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"],
            linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
            rms_norm_eps=t["rms_norm_eps"],
            vocab_size=t["vocab_size"],
            max_position_embeddings=t.get("max_position_embeddings", 262144),
            tie_word_embeddings=t.get("tie_word_embeddings", c.get("tie_word_embeddings", False)),
        )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


# ----------------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------------
class ModelState:
    """Explicit per-sequence state for one slot (batch of B identical-length sequences)."""

    def __init__(self, cfg: Qwen35Config, batch: int, max_seq_len: int, device, dtype=torch.bfloat16):
        self.cfg = cfg
        self.pos = 0
        self.max_seq_len = max_seq_len
        conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        self.conv: dict[int, torch.Tensor] = {}
        self.rec: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        for i, t in enumerate(cfg.layer_types):
            if t == "linear_attention":
                self.conv[i] = torch.zeros(batch, conv_dim, cfg.linear_conv_kernel_dim - 1, device=device, dtype=dtype)
                self.rec[i] = torch.zeros(batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
                                          cfg.linear_value_head_dim, device=device, dtype=torch.float32)
            else:
                self.k[i] = torch.zeros(batch, cfg.num_key_value_heads, max_seq_len, cfg.head_dim, device=device, dtype=dtype)
                self.v[i] = torch.zeros_like(self.k[i])

    def reset(self):
        self.pos = 0
        for d in (self.conv, self.rec, self.k, self.v):
            for t in d.values():
                t.zero_()


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Zero-centered RMSNorm: y = x / rms(x) * (1 + w). Computed in fp32, cast back."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class RMSNormGated(nn.Module):
    """GDN output norm: rmsnorm(x) * w, then * silu(gate). Same dtype dance as HF."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dt)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(dt)


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, initial_state, chunk_size: int = 64):
    """Chunked gated delta rule, identical math to transformers' torch_chunk_gated_delta_rule.
    query/key [B,T,H,dk], value [B,T,H,dv], g/beta [B,T,H]. Returns (out [B,T,H,dv] in input dtype,
    final state [B,H,dk,dv] fp32)."""
    dt_in = query.dtype
    B, T, _, dk = key.shape
    H, dv = value.shape[-2:]
    q, k, v, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                            for x in (query, key, value, beta, g)]
    q = l2norm(q, -1, 1e-6)
    k = l2norm(k, -1, 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    n_chunks = (T + pad) // chunk_size
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, decayed_k_beta, upper=False, unitriangular=True)
    if initial_state is None:
        state = torch.zeros(B, H, dk, dv, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    out = torch.zeros_like(new_values)
    q = q * cum_decay.exp().unsqueeze(-1)
    k = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(n_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter = q[:, :, i] @ state
        out[:, :, i] = inter + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, dv)[:, :, :T]
    return out.transpose(1, 2).to(dt_in, memory_format=torch.contiguous_format), state


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state):
    """One-token-at-a-time gated delta rule, identical to transformers' torch_recurrent_gated_delta_rule."""
    dt_in = query.dtype
    B, T, _, dk = key.shape
    H, dv = value.shape[-2:]
    q, k, v, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                            for x in (query, key, value, beta, g)]
    q = l2norm(q, -1, 1e-6)
    k = l2norm(k, -1, 1e-6)
    q = q / (q.shape[-1] ** 0.5)
    state = torch.zeros(B, H, dk, dv, dtype=v.dtype, device=v.device) if initial_state is None else initial_state.to(v)
    out = torch.zeros_like(v)
    for i in range(T):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        state = state * decay[:, :, i].exp()[..., None, None]
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(dt_in), state


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, bias=False, padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.num_v_heads, bias=False)

    def forward(self, x, state: ModelState | None, layer_idx: int):
        B, T, _ = x.shape
        K = self.conv_kernel_size
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # [B, C, T]
        z = self.in_proj_z(x).reshape(B, T, -1, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        w = self.conv1d.weight.squeeze(1)  # [C, K]
        has_prev = state is not None and state.pos > 0
        if has_prev and T == 1:
            conv_state = state.conv[layer_idx]
            xin = torch.cat([conv_state, mixed], dim=-1).to(w.dtype)
            conv_state.copy_(xin[:, :, -(K - 1):])
            out = F.conv1d(xin, w.unsqueeze(1), None, padding=0, groups=self.conv_dim)
            mixed = F.silu(out).to(mixed.dtype)
        else:
            if has_prev:
                xin = torch.cat([state.conv[layer_idx], mixed], dim=-1).to(w.dtype)
                out = F.conv1d(xin, w.unsqueeze(1), None, padding=0, groups=self.conv_dim)[:, :, -T:]
            else:
                xin = mixed.to(w.dtype)
                out = F.conv1d(xin, w.unsqueeze(1), None, padding=K - 1, groups=self.conv_dim)[:, :, :T]
            if state is not None:
                tail = xin[:, :, -(K - 1):]
                if tail.shape[-1] < K - 1:
                    tail = F.pad(tail, (K - 1 - tail.shape[-1], 0))
                state.conv[layer_idx].copy_(tail)
            mixed = F.silu(out).to(mixed.dtype)
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, -1, self.head_k_dim)
        k = k.reshape(B, T, -1, self.head_k_dim)
        v = v.reshape(B, T, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        init = state.rec[layer_idx] if has_prev else None
        if has_prev and T == 1:
            core, new_state = recurrent_gated_delta_rule(q, k, v, g, beta, init)
        else:
            core, new_state = chunk_gated_delta_rule(q, k, v, g, beta, init)
        if state is not None:
            state.rec[layer_idx].copy_(new_state)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(B, T, -1))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    """q/k [B,H,T,D], cos/sin [B,T,R] with R <= D; rotates only the first R dims."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    r = cos.shape[-1]
    q_rot, q_pass = q[..., :r], q[..., r:]
    k_rot, k_pass = k[..., :r], k[..., r:]
    q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, state: ModelState | None, layer_idx: int):
        B, T, _ = x.shape
        q, gate = torch.chunk(self.q_proj(x).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = self.q_norm(q.reshape(B, T, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        if state is not None:
            pos = state.pos
            if pos + T > state.max_seq_len:
                raise ValueError(f"sequence length {pos + T} exceeds state max_seq_len {state.max_seq_len}")
            state.k[layer_idx][:, :, pos:pos + T] = k
            state.v[layer_idx][:, :, pos:pos + T] = v
            k = state.k[layer_idx][:, :, :pos + T]
            v = state.v[layer_idx][:, :, :pos + T]
            if T == 1:
                attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            elif pos == 0:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                mask = torch.ones(T, pos + T, dtype=torch.bool, device=q.device).tril(pos)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(B, T, -1).contiguous()
        attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn)


class MLP(nn.Module):
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = cfg.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg)
        elif self.block_type == "full_attention":
            self.self_attn = Attention(cfg)
        else:
            raise ValueError(self.block_type)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, state):
        h = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            h = self.linear_attn(h, state, self.layer_idx)
        else:
            h = self.self_attn(h, cos, sin, state, self.layer_idx)
        x = x + h
        h = self.post_attention_layernorm(x)
        return x + self.mlp(h)


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen35Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._inv_freq = None

    def rotary(self, positions: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin [1, T, rotary_dim] in the model dtype, computed in fp32 like the reference."""
        if self._inv_freq is None or self._inv_freq.device != positions.device:
            d = self.cfg.rotary_dim
            self._inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=positions.device) / d))
        freqs = positions.float()[:, None] * self._inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    def new_state(self, batch: int, max_seq_len: int) -> ModelState:
        p = self.embed_tokens.weight  # the embedding is never quantized; lm_head may be a kernel module
        return ModelState(self.cfg, batch, max_seq_len, p.device, p.dtype)

    def forward(self, input_ids: torch.Tensor, state: ModelState | None = None, last_only: bool = False) -> torch.Tensor:
        """input_ids [B, T] -> logits [B, T or 1, vocab] (fp32). Advances state.pos by T."""
        B, T = input_ids.shape
        pos = state.pos if state is not None else 0
        x = self.embed_tokens(input_ids)
        positions = torch.arange(pos, pos + T, device=x.device)
        cos, sin = self.rotary(positions, x.dtype)
        cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, cos, sin, state)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x).float()
        if state is not None:
            state.pos += T
        return logits
