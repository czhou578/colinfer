"""Qwen3.8-Flash-Next (transformers model_type qwen4_exp) in plain PyTorch: the correctness reference of the MoE family.

It mirrors modeling_qwen4_exp.py of transformers op for op (same dtypes, same op order), as engine/model/qwen35.py does
for the dense family. The test tests/test_qwen4exp_small.py proves it against transformers on a tiny model on the CPU.
Nothing here is fast. The GDN layers, the rotary embedding and the norms come from qwen35.py; qwen35.py stays unchanged.

Architecture (48 layers, hidden 2560; docs/history/qwen4exp_plan.md):

- 36 Gated DeltaNet layers (48 / 16 heads of 128) whose output gate is sigmoid(z), not silu(z), and 12 attention layers
  (GQA 24/2, head_dim 256) with Qwen Sparse Attention (QSA): an indexer selects the 512 best 4-token blocks plus the
  tail for each query.
- Hyper-connections: the residual stream is 4 x 2560 wide. A GatedResidual before each sublayer mixes the 4 streams
  into the sublayer's input and gives the 4 injection weights of its output. No input / post norm and no final norm.
- A MoE block in every layer: 512 experts of SwiGLU (intermediate 640), top-10, plus one shared expert with a sigmoid
  gate.
- PLE at layer index 1: 16 hashed n-gram heads over a 320M-row table (engine/weights/ple.py), a gate per stream and a
  dilated depthwise conv.

The state (Qwen4ExpState) is explicit, as qwen35.ModelState, plus:

- ik[i] [B, 1, L, 128]: the raw indexer keys of attention layer i (not normed, not roped), time on dim 2
- rope_pos [B, L, 3] int64: the (t, h, w) rotary position of every cached row (a block key ropes at its first row)
- ple_conv[i] [B, 4 H, 9]: the last 9 inputs of the dilated conv of PLE layer i (zero at a sequence start)
- ple_ctx[i] [B, 2] int64: the two tokens before the next one, for the n-gram hash ((eos, eos) at a sequence start)

The routed experts of a layer are an ExpertWeights module, in one of two storages:

- dense (the tiny test; the names of transformers): gate_up_proj [E, 2 I, H] (gate rows, then up rows), down_proj [E, H, I]
- nvfp4 (the checkpoint; engine/weights/qwen4exp_loader.py): w_gu uint8 [E, 2 I, H / 2] (gate rows, then up rows),
  sf_gu e4m3 [E, 2 I, H / 16], w_dn uint8 [E, H, I / 2], sf_dn e4m3 [E, H, I / 16], gs f32 [E, 3] (the global scale
  weight_scale_2 of gate, up, down), in_scale f32 [E, 3] (the input_scale of gate, up, down). The block scales are
  row-major, as in the checkpoint; the kernel module (KernelMoE) swizzles its own copy. Element 2 k of a row is the low
  nibble of byte k.

The reference keeps transformers' MoE semantics: the softmax in fp32, top-k, the renormalized weights cast to the
model dtype, then a loop over the hit experts in ascending id that index_add_s each expert's weighted output in the
model dtype. The kernels sum in fp32 in slot order instead; their tests use a tolerance against this reference.

The module names match the HF text model without the `model.` prefix of Qwen4ExpForCausalLM (the checkpoint's
`model.language_model.` prefix), apart from the NVFP4 experts, which the loader stacks.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.model import qwen35
from engine.model.qwen35 import ModelState, Qwen35Config, RMSNorm, RMSNormGated, rotary
from engine.weights.loader import dequant_nvfp4
from engine.weights.ple import NgramParams, NgramTable, ngram_params


def _default_layer_types() -> list[str]:
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)]


@dataclass
class Qwen4ExpConfig(Qwen35Config):
    # the dense fields with the values of nvidia/Qwen3.8-Flash-Next-NVFP4
    hidden_size: int = 2560
    intermediate_size: int = 0  # no dense MLP
    num_hidden_layers: int = 48
    layer_types: list[str] = field(default_factory=_default_layer_types)
    num_key_value_heads: int = 2
    # MoE
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    num_experts: int = 512
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    output_gate_type: str = "sigmoid"  # the GDN output gate: sigmoid(z) or silu(z)
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])  # one-indexed, as in the config
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    eos_token_id: int = 248044  # the n-gram segment separator (<|endoftext|>)
    image_token_id: int = 248056  # image rows hash as this token
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # MTP (later milestone)
    mtp_num_hidden_layers: int = 1
    mtp_layer_types: list[str] = field(default_factory=lambda: ["full_attention"])

    @classmethod
    def from_checkpoint(cls, path: str) -> "Qwen4ExpConfig":
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("text_config", c)
        if c.get("model_type", t.get("model_type")) not in ("qwen4_exp", "qwen4_exp_text"):
            raise ValueError(f"{path}: model_type {c.get('model_type')!r} is not qwen4_exp")
        rp = t.get("rope_parameters", {})
        if rp.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rp['rope_type']!r}: only the default rope is implemented")
        qsa = ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio")
        missing = [k for k in qsa if t.get(k) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        n = t["num_hidden_layers"]
        types = t.get("layer_types")
        if types is None:
            iv = t.get("full_attention_interval", 4)
            types = ["linear_attention" if (i + 1) % iv else "full_attention" for i in range(n)]
        types = ["full_attention" if x == "indexed_attention" else x for x in types]
        eos = t.get("eos_token_id")
        eos = eos[0] if isinstance(eos, list) else eos
        mtp = t.get("mtp", {})
        cfg = cls(
            hidden_size=t["hidden_size"],
            intermediate_size=0,
            num_hidden_layers=n,
            layer_types=types,
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
            mrope_section=list(rp.get("mrope_section", [11, 11, 10])),
            mrope_interleaved=bool(rp.get("mrope_interleaved", False)),
            moe_intermediate_size=t["moe_intermediate_size"],
            shared_expert_intermediate_size=t["shared_expert_intermediate_size"],
            num_experts=t["num_experts"],
            num_experts_per_tok=t["num_experts_per_tok"],
            norm_topk_prob=t.get("norm_topk_prob", True),
            hidden_act=t.get("hidden_act", "silu"),
            output_gate_type=t.get("output_gate_type") or t.get("hidden_act", "silu"),
            hc_count=t.get("hc_count", 4),
            hc_lowrank=t.get("hc_lowrank", 320),
            ple_layer_ids=sorted(set(t.get("ple_layer_ids") or [])),
            ple_embed_dim=t.get("ple_embed_dim") or t["hidden_size"],
            ple_conv_kernel_size=t.get("ple_conv_kernel_size", 4),
            ngram_size=t.get("ngram_size", 3),
            heads_per_ngram=t.get("heads_per_ngram", 8),
            ngram_vocab_size_base=t.get("ngram_vocab_size_base", 20_000_000),
            make_ngram_vocab_size_divisible_by=t.get("make_ngram_vocab_size_divisible_by", 128),
            seed=t.get("seed", 1234),
            split_ngram_parts=t.get("split_ngram_parts", 512),
            eos_token_id=eos if eos is not None else -1,
            image_token_id=c.get("image_token_id", t.get("image_token_id", 248056)),
            **{k: t[k] for k in qsa},
            mtp_num_hidden_layers=t.get("mtp_num_hidden_layers", mtp.get("num_hidden_layers", 0)),
            mtp_layer_types=list(mtp.get("layer_types", ["full_attention"])),
        )
        cfg.validate()
        return cfg

    def validate(self):
        """The refusals: what the reference (and the engine) does not implement, or what transformers refuses."""
        bad = sorted(set(self.layer_types) - {"linear_attention", "full_attention"})
        if bad or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types: {bad or len(self.layer_types)}")
        if self.output_gate_type not in ("sigmoid", "silu"):
            raise ValueError(f"unsupported GDN output gate {self.output_gate_type!r}")
        if self.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {self.hidden_act!r}")
        if self.hc_count <= 1:
            raise ValueError(f"hc_count must be > 1, got {self.hc_count}")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(f"num_experts_per_tok {self.num_experts_per_tok} of {self.num_experts} experts")
        qsa = (self.indexer_n_heads, self.indexer_kv_heads, self.indexer_head_dim, self.indexer_budget, self.indexer_compress_ratio)
        if any(v <= 0 for v in qsa):
            raise ValueError(f"QSA config values must be positive: {qsa}")
        if self.indexer_kv_heads != 1:
            raise ValueError("QSA requires indexer_kv_heads = 1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        if self.rotary_dim > self.indexer_head_dim:
            raise ValueError(f"rotary_dim {self.rotary_dim} > indexer_head_dim {self.indexer_head_dim}")
        if self.ple_layer_ids:
            if self.ngram_heads <= 0 or self.ple_embed_dim % self.ngram_heads:
                raise ValueError(f"ple_embed_dim {self.ple_embed_dim} is not a multiple of the {self.ngram_heads} n-gram heads")
            for i in self.ple_layer_ids:
                if not 1 <= i <= self.num_hidden_layers or self.layer_types[i - 1] != "linear_attention":
                    raise ValueError(f"PLE layer {i} (one-indexed) must be a linear_attention layer")
            if self.eos_token_id < 0:
                raise ValueError("eos_token_id must be set when PLE layers are enabled")

    @property
    def context(self) -> int:
        """The longest sequence: the native context, times the YaRN factor when apply_yarn4 set one."""
        return int(self.max_position_embeddings * max(self.yarn_factor, 1.0))

    @property
    def ple_layers(self) -> dict[int, int]:
        """{zero-indexed layer: its index among the PLE layers}."""
        return {i - 1: j for j, i in enumerate(self.ple_layer_ids)}

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads

    @property
    def ple_state_len(self) -> int:
        """The dilated conv's window: (kernel - 1) x dilation (= ngram_size) = 9."""
        return (self.ple_conv_kernel_size - 1) * self.ngram_size


# ----------------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------------
class Qwen4ExpState(ModelState):
    """ModelState plus the QSA and PLE state (see the module docstring). kv_dtype: the K/V cache's dtype; the raw
    indexer keys keep the model dtype (fp8 would change the selection)."""

    def __init__(self, cfg: Qwen4ExpConfig, batch: int, max_seq_len: int, device, dtype=torch.bfloat16, kv_dtype=None):
        if max_seq_len > cfg.context:
            raise ValueError(f"max_seq_len {max_seq_len} > {cfg.context}: the native context {cfg.max_position_embeddings} "
                             f"x YaRN {cfg.yarn_factor:g} (apply_yarn4)")
        super().__init__(cfg, batch, max_seq_len, device, dtype, kv_dtype)
        self.ik = {i: torch.zeros(batch, 1, max_seq_len, cfg.indexer_head_dim, device=device, dtype=dtype) for i in self.k}
        self.rope_pos = torch.zeros(batch, max_seq_len, 3, device=device, dtype=torch.int64)
        hc = cfg.hc_count * cfg.hidden_size
        self.ple_conv = {i: torch.zeros(batch, hc, cfg.ple_state_len, device=device, dtype=dtype) for i in cfg.ple_layers}
        self.ple_ctx = {i: torch.full((batch, cfg.ngram_size - 1), cfg.eos_token_id, device=device, dtype=torch.int64) for i in cfg.ple_layers}

    def reset(self, kv: bool = True):
        super().reset(kv)
        for t in self.ple_conv.values():
            t.zero_()
        for t in self.ple_ctx.values():
            t.fill_(self.cfg.eos_token_id)
        if kv:
            for t in self.ik.values():
                t.zero_()
            self.rope_pos.zero_()


# ----------------------------------------------------------------------------------------------
# Norms, rope, hyper-connections
# ----------------------------------------------------------------------------------------------
class GroupRMSNorm(nn.Module):
    """Zero-centered RMSNorm over groups of `group` features (one group per hyper-connection stream)."""

    def __init__(self, dim: int, group: int, eps: float):
        super().__init__()
        if dim % group:
            raise ValueError(f"dim {dim} is not a multiple of the group {group}")
        self.eps, self.group = eps, group
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = x.float()
        h = h.reshape(*h.shape[:-1], -1, self.group)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return (h * (1.0 + self.weight.float())).type_as(x)


class RMSNormGatedSigmoid(RMSNormGated):
    """The GDN output norm with output_gate_type "sigmoid": rmsnorm(x) * w, then * sigmoid(gate)."""

    def forward(self, x, gate):
        dt = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dt)
        h = h * torch.sigmoid(gate.to(torch.float32))
        return h.to(dt)


class GatedDeltaNet4(qwen35.GatedDeltaNet):
    """qwen35.GatedDeltaNet with the output gate of the config (sigmoid on this model)."""

    def __init__(self, cfg: Qwen4ExpConfig):
        super().__init__(cfg)
        if cfg.output_gate_type == "sigmoid":
            self.norm = RMSNormGatedSigmoid(self.head_v_dim, cfg.rms_norm_eps)


def rope_first(x, cos, sin):
    """Rotates the first R dims of x [..., D] by cos / sin [..., R] (broadcast), as transformers' apply_rotary_pos_emb."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + qwen35.rotate_half(xr) * sin, xp], dim=-1)


class GatedResidual(nn.Module):
    """A hyper-connection: (x [.., 4 H]) -> the sublayer input [.., H] and the 4 injection weights [.., 4] (combine) or
    the input only (the final mixer). n = hc_norm(x) per stream; mix = sigmoid(up(silu(down(n) / 4))); the input is the
    mean over the streams of mix * n; inj = 2 sigmoid(block_inject(n) / 4)."""

    def __init__(self, cfg: Qwen4ExpConfig, combine: bool = True):
        super().__init__()
        self.hc, self.hidden = cfg.hc_count, cfg.hidden_size
        hc = self.hc * self.hidden
        self.hc_norm = GroupRMSNorm(hc, self.hidden, cfg.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, hc, bias=False)
        self.block_inject_weight = nn.Linear(hc, self.hc, bias=False) if combine else None

    def forward(self, h):
        n = self.hc_norm(h)
        mix = F.silu(self.input_mix_weight_down(n) / self.hc)
        mix = torch.sigmoid(self.input_mix_weight_up(mix)).unflatten(-1, (self.hc, self.hidden))
        x = (mix * n.unflatten(-1, (self.hc, self.hidden))).mean(dim=-2)
        if self.block_inject_weight is None:
            return x
        return x, 2 * torch.sigmoid(self.block_inject_weight(n) / self.hc)


def inject(h, y, inj):
    """h [.., 4 H] + y [.., H] * inj [.., 4] per stream."""
    return h + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


# ----------------------------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------------------------
def topk_stable(p: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The k largest of p along the last dim, by (value descending, index ascending): the tie rule of every path."""
    v, i = torch.sort(p + 0.0, dim=-1, descending=True, stable=True)  # + 0.0: -0.0 sorts as +0.0
    return v[..., :k], i[..., :k]


class Router(nn.Module):
    """The router: logits = x W^T; softmax in fp32; top-k; renormalized; cast to the model dtype."""

    def __init__(self, cfg: Qwen4ExpConfig):
        super().__init__()
        self.top_k, self.norm_topk_prob = cfg.num_experts_per_tok, cfg.norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(cfg.num_experts, cfg.hidden_size))

    def forward(self, x2):
        logits = F.linear(x2, self.weight)
        p = F.softmax(logits, dtype=torch.float, dim=-1)
        w, idx = topk_stable(p, self.top_k)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        return logits, w.to(logits.dtype), idx


class ExpertWeights(nn.Module):
    """The routed experts' weights of one layer, dense or NVFP4 (the attributes are in the module docstring).
    get(e, dtype) -> (gate_up [2 I, H], down [H, I]) of expert e in dtype."""

    def __init__(self, E: int, H: int, I: int, storage: str = "dense"):
        super().__init__()
        self.num_experts, self.hidden, self.inter, self.storage = E, H, I, storage
        if storage == "dense":
            self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
            self.down_proj = nn.Parameter(torch.zeros(E, H, I))
        elif storage == "nvfp4":
            if H % 32 or I % 32:
                raise ValueError(f"NVFP4 experts need H and I multiples of 32, got {H}, {I}")
            self.register_buffer("w_gu", torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8))
            self.register_buffer("sf_gu", torch.zeros(E, 2 * I, H // 16, dtype=torch.float8_e4m3fn))
            self.register_buffer("w_dn", torch.zeros(E, H, I // 2, dtype=torch.uint8))
            self.register_buffer("sf_dn", torch.zeros(E, H, I // 16, dtype=torch.float8_e4m3fn))
            self.register_buffer("gs", torch.ones(E, 3, dtype=torch.float32))
            self.register_buffer("in_scale", torch.ones(E, 3, dtype=torch.float32))
        else:
            raise ValueError(f"expert storage {storage!r}")

    def get(self, e: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.storage == "dense":
            return self.gate_up_proj[e].to(dtype), self.down_proj[e].to(dtype)
        I = self.inter
        gu = torch.cat([dequant_nvfp4(self.w_gu[e, :I], self.sf_gu[e, :I], self.gs[e, 0], dtype),
                        dequant_nvfp4(self.w_gu[e, I:], self.sf_gu[e, I:], self.gs[e, 1], dtype)])
        return gu, dequant_nvfp4(self.w_dn[e], self.sf_dn[e], self.gs[e, 2], dtype)


class Experts(ExpertWeights):
    """transformers' eager expert loop: the hit experts in ascending id; each adds its weighted output to its tokens'
    rows (index_add_ in the model dtype)."""

    def forward(self, x2, idx, w):
        out = torch.zeros_like(x2)
        hit = torch.unique(idx)  # sorted ascending
        for e in hit.tolist():
            pos, tok = torch.where((idx == e).T)  # [k, M] then nonzero: tokens ascending within each slot position
            gu, dn = self.get(e, x2.dtype)
            g, u = F.linear(x2[tok], gu).chunk(2, dim=-1)
            h = F.linear(F.silu(g) * u, dn) * w[tok, pos, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out


def nvfp4_experts(dense: ExpertWeights, in_scale: torch.Tensor | None = None) -> Experts:
    """The NVFP4 copy of dense experts, quantized as the checkpoint stores them (engine.weights.quantize): one global
    scale per expert shared by gate and up (as in the checkpoint), one for down. in_scale [E, 3] (default: ones)."""
    from engine.weights.quantize import nvfp4_global_scale, quantize
    E = dense.num_experts
    q = Experts(E, dense.hidden, dense.inter, "nvfp4").to(dense.gate_up_proj.device)
    with torch.no_grad():
        for e in range(E):
            gu, dn = dense.gate_up_proj[e].float(), dense.down_proj[e].float()
            g1, g2 = nvfp4_global_scale(gu), nvfp4_global_scale(dn)
            q.w_gu[e], q.sf_gu[e] = quantize(gu, g1)
            q.w_dn[e], q.sf_dn[e] = quantize(dn, g2)
            q.gs[e] = torch.tensor([g1, g1, g2])
        if in_scale is not None:
            q.in_scale.copy_(in_scale)
    return q


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoe(nn.Module):
    def __init__(self, cfg: Qwen4ExpConfig, experts: str = "dense"):
        super().__init__()
        self.gate = Router(cfg)
        self.experts = Experts(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, experts)
        self.shared_expert = SwiGLU(cfg.hidden_size, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        shared = self.shared_expert(x2)
        _, w, idx = self.gate(x2)
        out = self.experts(x2, idx, w)
        shared = torch.sigmoid(self.shared_expert_gate(x2)) * shared
        return (out + shared).reshape(shape)


# ----------------------------------------------------------------------------------------------
# PLE
# ----------------------------------------------------------------------------------------------
class NgramHash(nn.Module):
    """The n-gram hash of one PLE layer, with the buffers of transformers / the checkpoint (layer_multipliers,
    ngram_heads_vocab_sizes, ngram_heads_offsets). rows(tokens [B, T], ctx [B, 2]) -> [B, T, 16] int64: the same ids as
    engine.weights.ple.ngram_rows."""

    def __init__(self, cfg: Qwen4ExpConfig, ple_index: int):
        super().__init__()
        self.params: NgramParams = ngram_params(cfg, ple_index)
        p = self.params
        self.register_buffer("layer_multipliers", torch.tensor(p.multipliers, dtype=torch.long))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(p.sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(p.offsets, dtype=torch.long))

    def rows(self, tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        p = self.params
        tokens = tokens.long()
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= p.vocab_size):
            raise ValueError(f"n-gram token ids must be in [0, {p.vocab_size}): map image tokens to the image token id first")
        c, T = p.ngram_size - 1, tokens.shape[1]
        hist = torch.cat([ctx.long(), tokens], dim=1)
        m = self.layer_multipliers
        mixed = tokens * m[0]
        broken = torch.zeros_like(tokens, dtype=torch.bool)
        blocks = []
        for k in range(1, p.ngram_size):
            prev = hist[:, c - k:c - k + T]
            mixed = torch.bitwise_xor(mixed, torch.where(broken, p.eos, prev) * m[k])
            broken = broken | (prev == p.eos)
            h0 = (k - 1) * p.heads_per_ngram
            sizes = self.ngram_heads_vocab_sizes[h0:h0 + p.heads_per_ngram]
            offs = self.ngram_heads_offsets[h0:h0 + p.heads_per_ngram]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offs)
        return torch.cat(blocks, dim=-1)


class NgramEmbedding(NgramHash):
    """The n-gram embedding: the rows of the 16 heads, concatenated -> [B, T, 16 d]. The table is in memory
    (ngram_embedding, the tiny model; transformers' name) or an NgramTable (the checkpoint's file, read per call)."""

    def __init__(self, cfg: Qwen4ExpConfig, ple_index: int, table: NgramTable | None = None):
        super().__init__(cfg, ple_index)
        self.table = table
        self.ngram_embedding = None if table is not None else nn.Embedding(self.params.padded_rows, cfg.ngram_dim)

    def lookup(self, ids: torch.Tensor, dtype, device) -> torch.Tensor:
        if self.table is None:
            return self.ngram_embedding(ids.to(self.ngram_embedding.weight.device)).to(device)
        raw = torch.from_numpy(self.table.rows(ids.cpu().numpy()))
        return self.table.dequant(raw, dtype).view(*ids.shape, -1).to(device)

    def forward(self, tokens: torch.Tensor, state: Qwen4ExpState | None, li: int, dtype) -> torch.Tensor:
        B = tokens.shape[0]
        c = self.params.ngram_size - 1
        ctx = state.ple_ctx[li] if state is not None else tokens.new_full((B, c), self.params.eos, dtype=torch.long)
        ids = self.rows(tokens, ctx)
        if state is not None:
            state.ple_ctx[li].copy_(torch.cat([ctx, tokens.long()], dim=1)[:, -c:])
        return self.lookup(ids, dtype, tokens.device).flatten(-2)


class PLELayer(nn.Module):
    """Per-Layer Embedding: hashed n-gram features gated into every hyper-connection stream (transformers'
    Qwen4ExpTextPLELayer). forward(h [B, T, 4 H], ids [B, T]) -> the addend [B, T, 4 H]."""

    def __init__(self, cfg: Qwen4ExpConfig, ple_index: int, table: NgramTable | None = None):
        super().__init__()
        self.hidden, self.hc = cfg.hidden_size, cfg.hc_count
        hc = self.hc * self.hidden
        self.ple_embedding = NgramEmbedding(cfg, ple_index, table)
        self.key_proj = nn.Linear(cfg.ple_embed_dim, hc, bias=False)
        self.value_proj = nn.Linear(cfg.ple_embed_dim, self.hidden, bias=False)
        self.norm_key = GroupRMSNorm(hc, self.hidden, cfg.rms_norm_eps)
        self.norm_query = GroupRMSNorm(hc, self.hidden, cfg.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(hc, self.hidden, cfg.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc, hc, kernel_size=cfg.ple_conv_kernel_size, groups=hc, dilation=cfg.ngram_size, bias=False)
        self.state_len = cfg.ple_state_len

    def forward(self, h, ids, state: Qwen4ExpState | None, li: int):
        B, T, _ = h.shape
        e = self.ple_embedding(ids, state, li, h.dtype)
        kn = self.norm_key(self.key_proj(e)).unflatten(-1, (self.hc, self.hidden))
        v = self.value_proj(e)
        qn = self.norm_query(h).unflatten(-1, (self.hc, self.hidden))
        g = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        gv = torch.sigmoid(g) * v.unsqueeze(-2)
        gvn = self.norm_conv(gv.flatten(-2))
        gv = gv.flatten(-2)
        prev = state.ple_conv[li] if state is not None else h.new_zeros(B, h.shape[-1], self.state_len)
        xin = torch.cat([prev, gvn.transpose(1, 2)], dim=-1)
        if state is not None:
            state.ple_conv[li].copy_(xin[..., -self.state_len:])
        return gv + F.silu(self.conv1d(xin)).transpose(1, 2)


# ----------------------------------------------------------------------------------------------
# QSA
# ----------------------------------------------------------------------------------------------
QUERY_CHUNK = 256  # queries per scoring pass: [B, chunk, heads, blocks] fp32 is 2 GB at 262k tokens


class QSAIndexer(nn.Module):
    """The QSA indexer of an attention layer: which cached tokens each query attends to.

    The raw keys of every token are cached (state.ik). The complete 4-token blocks (aligned at absolute positions) are
    mean-pooled in fp32, k_layernorm-ed and roped at the block's first row (state.rope_pos). A query at position p sees
    the nbq = (p + 1) // 4 complete blocks before it; it scores them with sum_h relu(q_h . k_b) / sqrt(128) in fp32,
    keeps the top min(512, nbq) (score descending, index ascending) and adds the tail [4 nbq, p]."""

    def __init__(self, cfg: Qwen4ExpConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.head_dim = cfg.indexer_n_heads, cfg.indexer_head_dim
        self.ratio, self.block_topk = cfg.indexer_compress_ratio, cfg.block_topk
        self.index_qk_proj = nn.Linear(cfg.hidden_size, (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim, bias=False)
        self.q_layernorm = RMSNorm(cfg.indexer_head_dim, cfg.rms_norm_eps)
        self.k_layernorm = RMSNorm(cfg.indexer_head_dim, cfg.rms_norm_eps)

    def block_keys(self, state: Qwen4ExpState, li: int, L: int, dtype) -> torch.Tensor:
        """The keys of the complete blocks of rows [0, L): [B, L // 4, D] in dtype."""
        C, D = self.ratio, self.head_dim
        B, nb = state.ik[li].shape[0], L // C
        raw = state.ik[li][:, 0, :nb * C]
        kb = self.k_layernorm(raw.reshape(B, nb, C, D).float().mean(2).to(dtype))
        cos, sin = rotary(self.cfg, state.rope_pos[:, :nb * C:C].reshape(B * nb, 3), dtype)
        return rope_first(kb, cos.view(B, nb, -1), sin.view(B, nb, -1))

    def forward(self, x, cos, sin, state: Qwen4ExpState, li: int) -> torch.Tensor:
        """x [B, T, H]; cos / sin [B, T, R] of the rows [pos, pos + T); writes the raw keys -> the bool mask
        [B, 1, T, pos + T] of the tokens each query attends to."""
        B, T, _ = x.shape
        C, D, nh = self.ratio, self.head_dim, self.n_heads
        p0 = state.pos
        L = p0 + T
        qk = self.index_qk_proj(x)
        q = self.q_layernorm(qk[..., :nh * D].reshape(B, T, nh, D))
        q = rope_first(q, cos.unsqueeze(2), sin.unsqueeze(2))
        state.ik[li][:, 0, p0:L] = qk[..., nh * D:]
        nb = L // C
        mask = torch.zeros(B, 1, T, L, dtype=torch.bool, device=x.device)
        p = torch.arange(p0, L, device=x.device)
        nbq = (p + 1) // C  # complete blocks before each query
        tok = torch.arange(L, device=x.device)
        mask[:, 0] = (tok[None, :] >= (nbq * C)[:, None]) & (tok[None, :] <= p[:, None])  # the tail
        if nb == 0:
            return mask
        kb = self.block_keys(state, li, L, x.dtype).float()  # [B, nb, D]
        kk = min(self.block_topk, nb)
        blocks = torch.arange(nb, device=x.device)
        for t0 in range(0, T, QUERY_CHUNK):
            t1 = min(t0 + QUERY_CHUNK, T)
            s = torch.matmul(q[:, t0:t1].float(), kb[:, None].transpose(-1, -2))  # [B, t, nh, nb]
            s = torch.relu(s.transpose(-1, -2)).sum(dim=-1) / math.sqrt(D)  # [B, t, nb]
            vis = blocks[None, None, :] < nbq[t0:t1, None]
            s = s.masked_fill(~vis, float("-inf"))
            _, sel = topk_stable(s, kk)
            chosen = torch.zeros(B, t1 - t0, nb + 1, dtype=torch.bool, device=x.device)
            sel = torch.where(torch.gather(vis.expand(B, -1, -1), -1, sel), sel, nb)  # an invisible pick goes to the dropped column
            chosen.scatter_(-1, sel, True)
            mask[:, 0, t0:t1, :nb * C] |= chosen[..., :nb].repeat_interleave(C, dim=-1)
        return mask


class IndexedAttention(qwen35.Attention):
    """qwen35.Attention with the QSA indexer: SDPA over the selected tokens only."""

    def __init__(self, cfg: Qwen4ExpConfig):
        super().__init__(cfg)
        self.indexer = QSAIndexer(cfg)

    def forward(self, x, cos, sin, state: Qwen4ExpState, layer_idx: int):
        B, T, _ = x.shape
        pos = state.pos
        if pos + T > state.max_seq_len:
            raise ValueError(f"sequence length {pos + T} exceeds state max_seq_len {state.max_seq_len}")
        mask = self.indexer(x, cos, sin, state, layer_idx)
        q, gate = torch.chunk(self.q_proj(x).view(B, T, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = self.q_norm(q.reshape(B, T, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        q, k = qwen35.apply_rotary(q, k, cos, sin)
        state.k[layer_idx][:, :, pos:pos + T] = k
        state.v[layer_idx][:, :, pos:pos + T] = v
        k = state.k[layer_idx][:, :, :pos + T].to(q.dtype)
        v = state.v[layer_idx][:, :, :pos + T].to(q.dtype)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(B, T, -1).contiguous()
        return self.o_proj(attn * torch.sigmoid(gate))


# ----------------------------------------------------------------------------------------------
# Layers and model
# ----------------------------------------------------------------------------------------------
class DecoderLayer4(nn.Module):
    """PLE (layer 1 only), then the attention hyper-connection, the mixer, its injection, then the MLP
    hyper-connection, the MoE and its injection."""

    def __init__(self, cfg: Qwen4ExpConfig, layer_idx: int, experts: str = "dense", ngram_table: NgramTable | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = cfg.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet4(cfg)
        else:
            self.self_attn = IndexedAttention(cfg)
        self.mlp = SparseMoe(cfg, experts)
        j = cfg.ple_layers.get(layer_idx)
        self.ple = PLELayer(cfg, j, ngram_table) if j is not None else None
        self.attn_hyper_connection = GatedResidual(cfg)
        self.mlp_hyper_connection = GatedResidual(cfg)

    def forward(self, h, cos, sin, state: Qwen4ExpState, ple_ids):
        i = self.layer_idx
        if self.ple is not None:
            h = h + self.ple(h, ple_ids, state, i)
        x, inj = self.attn_hyper_connection(h)
        if self.block_type == "linear_attention":
            y = self.linear_attn(x, state, i)
        else:
            y = self.self_attn(x, cos, sin, state, i)
        h = inject(h, y, inj)
        x, inj = self.mlp_hyper_connection(h)
        return inject(h, self.mlp(x), inj)


class Qwen4ExpForCausalLM(nn.Module):
    """experts: "dense" or "nvfp4" (the storage of the routed experts); ngram_table: the n-gram rows on disk (else an
    in-memory table of padded_rows x d)."""

    def __init__(self, cfg: Qwen4ExpConfig, experts: str = "dense", ngram_table: NgramTable | None = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer4(cfg, i, experts, ngram_table) for i in range(cfg.num_hidden_layers)])
        self.hyper_connection_mixer = GatedResidual(cfg, combine=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def new_state(self, batch: int, max_seq_len: int, kv_dtype=None) -> Qwen4ExpState:
        p = self.embed_tokens.weight
        return Qwen4ExpState(self.cfg, batch, max_seq_len, p.device, p.dtype, kv_dtype)

    def forward(self, input_ids: torch.Tensor, state: Qwen4ExpState | None = None, last_only: bool = False,
                positions: torch.Tensor | None = None, embeds: torch.Tensor | None = None, ple_ids: torch.Tensor | None = None,
                return_hyper: bool = False):
        """input_ids [B, T] -> logits [B, T or 1, vocab] (fp32); with return_hyper also the stream before the final
        mixer [B, T, 4 H] (the MTP block's input). Advances state.pos by T. state None: a temporary state of T rows
        (transformers without a cache). positions: [T] or [T, 3] (t, h, w) rotary positions (default: the cache
        positions); embeds: input embeddings [B, T, H] instead of input_ids' (images); ple_ids: the ids the n-gram hash
        reads (default input_ids; image rows carry image_token_id)."""
        B, T = input_ids.shape
        if state is None:
            state = self.new_state(B, T)
        pos = state.pos
        if pos + T > state.max_seq_len:
            raise ValueError(f"sequence length {pos + T} exceeds state max_seq_len {state.max_seq_len}")
        x = self.embed_tokens(input_ids) if embeds is None else embeds
        if positions is None:
            positions = torch.arange(pos, pos + T, device=x.device)
        state.rope_pos[:, pos:pos + T] = (positions[:, None].expand(T, 3) if positions.dim() == 1 else positions).to(state.rope_pos)
        cos, sin = rotary(self.cfg, positions, x.dtype)
        cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
        h = x.repeat(1, 1, self.cfg.hc_count)
        ids = input_ids if ple_ids is None else ple_ids
        for layer in self.layers:
            h = layer(h, cos, sin, state, ids)
        x = self.hyper_connection_mixer(h)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x).float()
        state.pos += T
        return (logits, h) if return_hyper else logits


def fold_yarn_scale4(attn: IndexedAttention, cfg: Qwen4ExpConfig) -> None:
    """qwen35.fold_yarn_scale for a QSA layer: YaRN's attention factor folded into the rotated dims of the attention's q /
    k norms and of the indexer's q / k norms. transformers gives the indexer the attention's position embeddings (cos and
    sin times the factor), so the indexer's rotated dims scale by the same factor. The block keys get it too, since
    k_layernorm runs before their rotation. All four norms are zero-centered: (1 + w) -> a (1 + w)."""
    a = qwen35.yarn_attention_factor(cfg)
    if a == 1.0:
        return
    r = cfg.rotary_dim
    with torch.no_grad():
        for norm in (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm):
            w = norm.weight
            w[:r] = (a * (1.0 + w[:r].float()) - 1.0).to(w.dtype)


def apply_yarn4(model: Qwen4ExpForCausalLM, factor: float) -> None:
    """qwen35.apply_yarn for qwen4_exp: YaRN for `factor` times the native context. The new config (rope_inv_freq picks
    the YaRN frequencies from it) replaces the model's and the indexers' (they rotate the block keys themselves); each
    attention layer's four norms get the attention factor (fold_yarn_scale4). Call before to_fast4, which reads both."""
    if factor <= 1:
        return
    if model.cfg.yarn_factor > 1:
        raise ValueError(f"the model already has YaRN x{model.cfg.yarn_factor}")
    old, new = model.cfg, dataclasses.replace(model.cfg, yarn_factor=float(factor))
    for m in model.modules():
        if getattr(m, "cfg", None) is old:
            m.cfg = new
    for layer in model.layers:
        if layer.block_type == "full_attention":
            fold_yarn_scale4(layer.self_attn, new)


def image_ple_ids(ids: torch.Tensor, cfg: Qwen4ExpConfig) -> torch.Tensor:
    """The ids the n-gram hash reads: the engine's image placeholder ids (past the vocabulary) become image_token_id."""
    return torch.where(ids >= cfg.vocab_size, cfg.image_token_id, ids)


__all__ = ["Qwen4ExpConfig", "Qwen4ExpState", "Qwen4ExpForCausalLM", "ExpertWeights", "Experts", "nvfp4_experts", "NgramHash",
           "NgramEmbedding", "PLELayer", "QSAIndexer", "IndexedAttention", "GatedResidual", "GroupRMSNorm", "image_ple_ids", "apply_yarn4",
           "fold_yarn_scale4"]
