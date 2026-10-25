"""Decode-path model of Qwen3.8-Flash-Next (qwen4_exp): the reference (engine/model/qwen4exp.py) with each op on the CUDA
kernels of the engine, and the positions on the device. A decode step, or a verify of T rows, is one CUDA graph.

This module follows engine/model/fast.py (the 27B's decode model). The design:

- Weights: the routed experts stay NVFP4 (moe_kernels.KernelMoE, which swizzles its own scales). Everything else is
  BF16 and streams through the BF16 skinny GEMM (Bf16Linear, StackedBf16Linear): the exact path of the checkpoint.
  attach_decode_copies4 adds the requantized decode copies of tools/requant_qwen4exp.py (docs/history/qwen4exp_plan.md):
  a `.dec` of a linear (INT6 / INT5 / NVFP4), the INT6 down | inject and the INT8 W_up of the hyper-connections,
  and the INT6 gate | up of the shared experts; the prepared directory holds the classes that pass the perplexity gate
  (INT6 attention and lm_head, INT8 W_up: the decision is in the tool's docstring). Decode then streams the copies; the
  prefill keeps reading the BF16 weights (cuBLAS), except for the last token's mixer and lm_head (the decode modules).
  Without the copies, decode is the exact path.
- The residual stream is x4 [B * T, 4 x 2560]. Each sublayer reads it through a gated residual (hc_kernels
  KernelGatedResidual: the input y and the 4 injection weights), and hc_post adds the sublayer's output back in place.
- Attention layers run Qwen Sparse Attention (KernelQSAAttention): one stacked q | k | v | index pass, the fused prologues,
  the block keys of the rows that complete a block, the block scores, the top 512 blocks and the sparse attention with the
  output gate (csrc/qsa.cu).
- GDN layers run the 27B's kernels (fast.KernelGDN) with the sigmoid output gate (gate_act = 1).
- The PLE layer (layer index 1) reads 16 n-gram rows of 160 FP8 bytes per token from a device buffer that the host fills
  before a replay (feed_buffer). Its gate and its dilated conv run as torch ops inside the graph. Each reduction of
  those ops has a fixed order (a pairwise sum of elementwise adds, see _row_sum): a row's bits do not depend on the
  number of rows.

Bit identity. Every op computes a row the same way for any number of rows in the launch (the skinny GEMMs, the kernels of
hc.cu, qsa.cu, moe.cu and gdn_step.cu, and the elementwise torch ops of the PLE layer). Thus a decode step of one row
and a verify of T rows give the same bits for the same token (tests/test_fast_qwen4exp.py checks this).

Differences from the reference (tests/test_fast_qwen4exp.py measures them): the KV cache is fp8 (e4m3); the kernels
round the RoPE in fp32 (the reference rounds cos / sin to bf16); the MoE combines the expert outputs in fp32 in slot order
(the reference adds them in bf16 in expert order); the hyper-connection mix rounds u once; the PLE conv sums its 4 taps
in fp32 in tap order.

Entry points: load_fast_qwen4exp(path) = load_qwen4exp + to_fast4 -> FastQwen4Exp. FastQwen4Exp.forward is one decode
step (T = 1 per slot); verify / commit are the two halves of speculation. The state is FastState4.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.kernels import ops
from engine.model import hc_kernels, qsa_kernels
from engine.model.fast import MAX_ROWS, FastState, IntLinear, KernelGDN, Nvfp4Linear, RowsLinear, _rows
from engine.model.moe_kernels import KernelMoE
from engine.model.qwen4exp import DecoderLayer4, IndexedAttention, PLELayer, Qwen4ExpConfig, Qwen4ExpForCausalLM
from engine.model.qwen35 import rope_inv_freq
from engine.weights.ple import ngram_rows

HC_ROWS = 16  # rows per launch of the hyper-connection mix (csrc/hc.cu)
MOE_INTER = (128, 256, 384, 512, 640)  # the intermediate sizes of moe_down (csrc/moe_bind.cpp)


# ------------------------------------------------------------------------------------------------------------ linears
class Bf16Linear(RowsLinear):
    """BF16 weights w [N, K] on the BF16 skinny GEMM (ops().skinny_bf16: K % 128 == 0, N % 8 == 0). out_fp32: logits.
    The residual rows, if given, are added in the epilogue. dec: an optional decode copy (fast.IntLinear or Nvfp4Linear,
    attach_decode_copies4) that rows() streams instead; the prefill reads w."""

    def __init__(self, w: torch.Tensor, out_fp32: bool = False):
        super().__init__()
        self.register_buffer("w", w.detach().contiguous(), persistent=False)
        self.out_fp32 = out_fp32
        self.out_dtype = torch.float32 if out_fp32 else torch.bfloat16
        self.out_features, self.in_features = self.w.shape
        self.dec = None

    @property
    def weight(self) -> torch.Tensor:
        return self.w

    def rows(self, x2, r2, out):
        if self.dec is not None:  # an INT or NVFP4 copy writes bf16 or fp32 (logits) as out is
            self.dec.rows(x2, r2, out)
        else:
            _rows(lambda x, r, o: ops().skinny_bf16(x, self.w, r, o), x2, r2, out)


class StackedBf16Linear(nn.Module):
    """BF16 linears that read the same input, stacked into one weight pass (one w [sum N, K]). forward(x) returns one
    output per part: column views of one [rows, sum N] tensor. The parts stay usable (prefill): each part's weight
    becomes a view of the stacked w. dec: an optional decode copy of the whole stack (one global scale, so one launch)."""

    def __init__(self, parts: list):
        super().__init__()
        self.register_buffer("w", torch.cat([p.weight.detach() for p in parts]).contiguous(), persistent=False)
        self.sizes = [p.weight.shape[0] for p in parts]
        self.in_features = self.w.shape[1]
        self.dec = None
        off = 0
        for p, n in zip(parts, self.sizes):
            view = self.w[off:off + n]
            if isinstance(p, nn.Linear):
                p.weight = nn.Parameter(view, requires_grad=False)
            else:
                p.w = view
            off += n

    @property
    def weight(self) -> torch.Tensor:
        return self.w

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous()
        out = torch.empty(x2.shape[0], sum(self.sizes), device=x.device, dtype=torch.bfloat16)
        if self.dec is not None:
            self.dec.rows(x2, None, out)
        else:
            _rows(lambda x_, r, o: ops().skinny_bf16(x_, self.w, r, o), x2, None, out)
        return [t.reshape(*shp[:-1], -1) for t in out.split(self.sizes, dim=-1)]


# ------------------------------------------------------------------------------------------------------------ state
class FastState4(FastState):
    """The decode state of qwen4_exp for B slots of max_seq_len (Lmax) tokens: FastState (fp8 k / v [B, 2, Lmax, 256] per
    attention layer, conv [B, 10240, 3] and rec fp32 [B, 48, 128, 128] per GDN layer, pos_t / active / rope_delta int32
    [B], spec) plus:

    - ik[i]: bf16 [B, Lmax, 128], the raw indexer keys of attention layer i (no norm, no RoPE)
    - ikb[i]: bf16 [B, Lmax / 4, 128], the block keys of attention layer i (derived from ik and rope_pos)
    - rope_pos: int32 [B, Lmax, 3], the (t, h, w) rotary position of every cached row (all layers)
    - ple_conv[i]: bf16 [B, 10240, 9], the last 9 inputs of the dilated conv of PLE layer i

    Lmax must be a multiple of 16 (qsa_topk reads the scores of Lmax / 4 blocks in 16-byte rows). view(lo, hi) shares
    the tensors of slots [lo, hi)."""

    def __init__(self, cfg: Qwen4ExpConfig, batch: int, max_seq_len: int, device):
        if max_seq_len % 16:
            raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of 16 (qsa_topk reads the block scores in rows of 4 floats)")
        if max_seq_len > cfg.context:
            raise ValueError(f"max_seq_len {max_seq_len} > {cfg.context}: the native context {cfg.max_position_embeddings} "
                             f"x YaRN {cfg.yarn_factor:g} (qwen4exp.apply_yarn4)")
        super().__init__(cfg, batch, max_seq_len, device)
        D, R = cfg.indexer_head_dim, cfg.indexer_compress_ratio
        bf = dict(device=device, dtype=torch.bfloat16)
        self.ik = {i: torch.zeros(batch, max_seq_len, D, **bf) for i in self.k}
        self.ikb = {i: torch.zeros(batch, max_seq_len // R, D, **bf) for i in self.k}
        self.rope_pos = torch.zeros(batch, max_seq_len, 3, device=device, dtype=torch.int32)
        hc = cfg.hc_count * cfg.hidden_size
        self.ple_conv = {i: torch.zeros(batch, hc, cfg.ple_state_len, **bf) for i in cfg.ple_layers}

    def reset(self, kv: bool = True):
        super().reset(kv)
        for t in self.ple_conv.values():
            t.zero_()
        if kv:
            for t in list(self.ik.values()) + list(self.ikb.values()):
                t.zero_()
            self.rope_pos.zero_()

    def view(self, lo: int, hi: int) -> "FastState4":
        v = super().view(lo, hi)
        v.ik = {i: t[lo:hi] for i, t in self.ik.items()}
        v.ikb = {i: t[lo:hi] for i, t in self.ikb.items()}
        v.rope_pos = self.rope_pos[lo:hi]
        v.ple_conv = {i: t[lo:hi] for i, t in self.ple_conv.items()}
        return v

    def kv_tensors(self) -> list[torch.Tensor]:
        """The per-position tensors, time on dim 2: k[i], v[i] [B, 2, Lmax, 256] of each attention layer (ascending, the
        27B's order), then ik[i] as [B, 1, Lmax, 128], then ikb[i] as [B, 1, Lmax / 4, 128] (one row per 4 positions),
        then rope_pos as [B, 1, Lmax, 3]. kv_ratios() gives the positions per row of each."""
        layers = sorted(self.k)
        out = [t for i in layers for t in (self.k[i], self.v[i])]
        out += [self.ik[i].unsqueeze(1) for i in layers]
        out += [self.ikb[i].unsqueeze(1) for i in layers]
        return out + [self.rope_pos.unsqueeze(1)]

    def kv_ratios(self) -> list[int]:
        """The positions per row of each tensor of kv_tensors(): 1, or the indexer's compress ratio for the block keys."""
        return [self.max_seq_len // t.shape[2] for t in self.kv_tensors()]

    def snap_tensors(self) -> list[torch.Tensor]:
        """The per-slot recurrent state: conv[i] of each GDN layer, then rec[i], then ple_conv[i] (ascending layers)."""
        return [self.conv[i] for i in sorted(self.conv)] + [self.rec[i] for i in sorted(self.rec)] + \
            [self.ple_conv[i] for i in sorted(self.ple_conv)]


# ------------------------------------------------------------------------------------------------------------ layers
def hc_rows(gr: hc_kernels.KernelGatedResidual, x4: torch.Tensor):
    """gr(x4) over any number of rows, HC_ROWS at a time (each row's result does not change)."""
    if x4.shape[0] <= HC_ROWS:
        return gr(x4)
    parts = [gr(x4[i:i + HC_ROWS]) for i in range(0, x4.shape[0], HC_ROWS)]
    if not gr.has_inj:
        return torch.cat(parts)
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def gated_residual(ref) -> hc_kernels.KernelGatedResidual:
    """The kernel module of a reference GatedResidual (the final mixer has no inject weight)."""
    inj = ref.block_inject_weight.weight if ref.block_inject_weight is not None else None
    return hc_kernels.KernelGatedResidual(ref.hc_norm.weight, ref.input_mix_weight_down.weight, inj, ref.input_mix_weight_up.weight,
                                          ref.hc_norm.eps)


class KernelGDN4(KernelGDN):
    """fast.KernelGDN on BF16 projections, with the output gate of the config. forward(x [B, T, 2560], state, li) ->
    o [B, T, 2560] with no residual (the injection runs after it); verify (state.spec) and commit as in fast.KernelGDN."""
    qkvz: StackedBf16Linear  # in_proj_qkv | in_proj_z [16384, 2560]

    @classmethod
    def adopt(cls, gdn, gate_act: int) -> "KernelGDN4":
        """Moves a reference GatedDeltaNet4, with its weights, onto the kernels (in place)."""
        gdn.__class__ = cls
        gdn.qkvz = StackedBf16Linear([gdn.in_proj_qkv, gdn.in_proj_z])
        gdn.w_ba = torch.cat([gdn.in_proj_b.weight, gdn.in_proj_a.weight]).detach().contiguous()
        del gdn.in_proj_b, gdn.in_proj_a
        gdn.out_proj = Bf16Linear(gdn.out_proj.weight)
        gdn.gate_act = gate_act
        gdn._spec = None
        return gdn


class KernelQSAAttention(IndexedAttention):
    """Gated GQA 24 / 2 attention with Qwen Sparse Attention on T new rows per slot (1 for decode, k + 1 for verify):

    1. one stacked q | k | v | index pass (qkvi);
    2. attn_prologue: q / k norm, RoPE at the rows' rope_pos, the fp8 KV write at state.pos_t;
    3. qsa_prologue: the indexer's q_layernorm + RoPE, the raw key write into state.ik;
    4. qsa_block_keys: the block keys of the blocks that these rows complete (state.ikb), roped at state.rope_pos;
    5. qsa_score and qsa_topk: the top 512 blocks of each row;
    6. qsa_attn: the sparse attention with the output gate; then o_proj (no residual).

    The caller writes the rows' rotary positions into state.rope_pos before the layers run (FastQwen4Exp._rope_rows)."""
    qkvi: StackedBf16Linear  # q_proj | k_proj | v_proj | indexer.index_qk_proj [12288 + 512 + 512 + 640, 2560]
    inv_freq: torch.Tensor   # the RoPE frequencies of the rotary dims (fp32 [R / 2]); the indexer ropes the same R dims
    mrope: tuple[int, int]   # the h / w sections of interleaved multimodal RoPE

    @classmethod
    def adopt(cls, attn: IndexedAttention, inv_freq: torch.Tensor, mrope=(11, 10)) -> "KernelQSAAttention":
        """Moves a reference IndexedAttention, with its weights, onto the kernels (in place)."""
        attn.__class__ = cls
        attn.qkvi = StackedBf16Linear([attn.q_proj, attn.k_proj, attn.v_proj, attn.indexer.index_qk_proj])
        attn.o_proj = Bf16Linear(attn.o_proj.weight)
        attn.inv_freq, attn.mrope = inv_freq, tuple(mrope)
        return attn

    def forward(self, x, state: FastState4, li: int, rope_rows: torch.Tensor):
        """x bf16 [B, T, 2560]; rope_rows int32 [B * T, 3] (the rows' rotary positions) -> o bf16 [B, T, 2560]."""
        B, T, _ = x.shape
        ix = self.indexer
        qp, kp, vp, ip = self.qkvi(x)
        kc, vc, ik, ikb = state.k[li], state.v[li], state.ik[li], state.ikb[li]
        q = torch.empty(B, self.num_heads, T, self.head_dim, device=x.device, dtype=torch.bfloat16)
        ops().attn_prologue(qp, kp, vp, self.q_norm.weight, self.k_norm.weight, self.inv_freq, state.pos_t, kc, vc, q, self.q_norm.eps,
                            state.active, None, rope_rows, *self.mrope)
        q_idx = qsa_kernels.qsa_prologue(ip, ix.q_layernorm.weight, self.inv_freq, state.pos_t, ik, ix.q_layernorm.eps, T, state.active, None,
                                         rope_rows, self.mrope)
        qsa_kernels.qsa_block_keys(ik, ikb, ix.k_layernorm.weight, self.inv_freq, state.pos_t, T, ix.k_layernorm.eps, state.active, None,
                                   state.rope_pos, self.mrope)
        scores = qsa_kernels.qsa_score(q_idx, ikb, state.pos_t)
        sel, nsel = qsa_kernels.qsa_topk(scores, state.pos_t, T)
        attn = qsa_kernels.qsa_attn(q, kc, vc, sel, nsel, state.pos_t, self.head_dim ** -0.5, gate=qp.contiguous())
        return self.o_proj(attn)


def _row_sum(x: torch.Tensor) -> torch.Tensor:
    """x fp32 [R, n] -> [R]: halves the row while its length is even (x[:, :h] + x[:, h:]), then adds the rest left to
    right. Only elementwise kernels: a row's sum does not depend on R (a torch reduction may pick its order by shape)."""
    while x.shape[1] > 1 and x.shape[1] % 2 == 0:
        h = x.shape[1] // 2
        x = x[:, :h] + x[:, h:]
    s = x[:, 0]
    for j in range(1, x.shape[1]):
        s = s + x[:, j]
    return s


class KernelPLE(PLELayer):
    """The PLE layer on T new rows per slot: the n-gram rows come as FP8 bytes u8 [B, T, 16, 160] (the host reads them,
    engine/weights/ple.py). forward(x4 [B * T, 10240], rows, state, li, B, T) adds the PLE output to x4 in place.

    The math (transformers' rounding, each op in fp32 with one bf16 rounding; docs/history/qwen4exp_plan.md 5.5):
    e = bf16(e4m3 * scale); kp | vp = one stacked pass (kv_proj); kn = hc_norm(kp), qn = hc_norm(x4); g = bf16(sum(bf16(kn *
    qn))) / sqrt(2560) per stream; g = sqrt(max(|g|, 1e-6)) * sign(g); gv = bf16(sigmoid(g) * vp); u = hc_norm(gv) (norm_conv);
    c[t] = the 4 taps of the dilated conv (dilation 3) over the 9-row state and the new rows, summed in fp32 in tap order;
    x4 += bf16(gv + silu(bf16(c))).

    Plain decode advances state.ple_conv by one row (slots with active == 0 keep theirs). A verify (state.spec) leaves it
    and keeps the conv input for commit(state, li, n), which advances it by n[b] rows."""
    kv_proj: StackedBf16Linear  # key_proj | value_proj [12800, 2560]
    scale: float                # the n-gram table's dequantization scale
    _spec: torch.Tensor | None  # the conv input [B, 10240, 9 + T] of the last verify

    @classmethod
    def adopt(cls, ple: PLELayer) -> "KernelPLE":
        table = ple.ple_embedding.table
        if table is None:
            raise ValueError("the decode path reads the n-gram rows as FP8 bytes: the PLE layer needs an NgramTable (not an in-memory table)")
        ple.__class__ = cls
        ple.kv_proj = StackedBf16Linear([ple.key_proj, ple.value_proj])
        ple.scale = float(table.scale)
        C = ple.hc * ple.hidden
        ple.conv_w = ple.conv1d.weight.detach().view(C, -1).float()  # [C, 4]: the taps in fp32
        ple.dilation = ple.conv1d.dilation[0]
        ple._spec = None
        return ple

    @property
    def params(self):
        return self.ple_embedding.params

    @property
    def table(self):
        return self.ple_embedding.table

    def _norm(self, x: torch.Tensor, norm) -> torch.Tensor:
        out = torch.empty_like(x)
        ops().hc_norm(x, norm.weight, norm.eps, out, self.hc)
        return out

    def forward(self, x4: torch.Tensor, rows: torch.Tensor, state: FastState4, li: int, B: int, T: int):
        M, H, C = B * T, self.hidden, self.hc * self.hidden
        e = (rows.reshape(M, -1).view(torch.float8_e4m3fn).float() * self.scale).to(torch.bfloat16)
        kp, vp = self.kv_proj(e)
        kn = self._norm(kp.contiguous(), self.norm_key)
        qn = self._norm(x4, self.norm_query)
        g = _row_sum((kn * qn).float().view(M * self.hc, H)).to(torch.bfloat16).view(M, self.hc, 1)
        g = g / math.sqrt(H)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        gv = torch.sigmoid(g) * vp.reshape(M, 1, H)  # [M, 4, H]
        u = self._norm(gv.view(M, C), self.norm_conv)
        prev = state.ple_conv[li]  # [B, C, 9]
        hist = torch.cat([prev, u.view(B, T, C).transpose(1, 2)], dim=2)  # [B, C, 9 + T]
        hf, w, d = hist.float(), self.conv_w, self.dilation
        k = w.shape[1]
        acc = hf[:, :, 0:T] * w[:, 0:1]
        for j in range(1, k):
            acc = acc + hf[:, :, j * d:j * d + T] * w[:, j:j + 1]
        act = F.silu(acc.to(torch.bfloat16))  # [B, C, T]
        x4.view(B, T, C).add_(gv.view(B, T, C) + act.transpose(1, 2))
        n = prev.shape[2]
        if state.spec:
            self._spec = hist
        else:
            assert T == 1, "multi-token prompts go through the prefill path"
            prev.copy_(torch.where(state.active.view(B, 1, 1) != 0, hist[:, :, T:T + n], prev))

    def commit(self, state: FastState4, li: int, n: torch.Tensor):
        """Advance state.ple_conv[li] by the first n[b] rows of the last verify."""
        hist = self._spec
        B, C = hist.shape[:2]
        w = state.ple_conv[li].shape[2]
        idx = (n.long().view(B, 1, 1) + torch.arange(w, device=hist.device).view(1, 1, w)).expand(B, C, w)
        state.ple_conv[li].copy_(hist.gather(2, idx))


class FastDecoderLayer4(DecoderLayer4):
    """DecoderLayer4 on kernel modules: PLE (layer 1) on x4 in place; attn_hc -> the mixer (linear_attn or self_attn) ->
    hc_post; mlp_hc -> the MoE -> hc_post. x4 bf16 [B * T, 10240] changes in place."""
    attn_hc: hc_kernels.KernelGatedResidual
    mlp_hc: hc_kernels.KernelGatedResidual
    mlp: KernelMoE

    def forward(self, x4, state: FastState4, B: int, T: int, rope_rows: torch.Tensor, rows: torch.Tensor | None):
        i = self.layer_idx
        if self.ple is not None:
            self.ple(x4, rows, state, i, B, T)
        y, inj = hc_rows(self.attn_hc, x4)
        if self.block_type == "linear_attention":
            o = self.linear_attn(y.view(B, T, -1), state, i)
        else:
            o = self.self_attn(y.view(B, T, -1), state, i, rope_rows)
        hc_kernels.KernelGatedResidual.post(o, inj, x4)
        y, inj = hc_rows(self.mlp_hc, x4)
        hc_kernels.KernelGatedResidual.post(self.mlp(y), inj, x4)
        return x4


# ------------------------------------------------------------------------------------------------------------ model
class FastQwen4Exp(Qwen4ExpForCausalLM):
    """The decode model. forward(input_ids [B, 1], state, ngram_rows) is one step; verify(input_ids [B, T], state,
    ngram_rows) and commit(state, n) are the two halves of speculation. ngram_rows: u8 [B, T, 16, 160], the FP8 n-gram rows
    of the tokens (None: the fixed feed_buffer(B, T), which the host fills before a graph replay)."""
    mixer: hc_kernels.KernelGatedResidual  # the final gated residual (no inject)
    lm_head: Bf16Linear

    @property
    def drafter_hidden_dim(self) -> int:
        """The width of the hidden that verify returns for a drafter: the 4-stream x4 before the final mixer (10240)."""
        return self.cfg.hc_count * self.cfg.hidden_size

    def new_state(self, batch: int, max_seq_len: int) -> FastState4:
        return FastState4(self.cfg, batch, max_seq_len, self.embed_tokens.weight.device)

    def feed_buffer(self, B: int, T: int) -> torch.Tensor:
        """The fixed device buffer u8 [B, T, 16, 160] of the n-gram rows that a captured graph reads (one per (B, T))."""
        bufs = self.__dict__.setdefault("_feed", {})
        if (B, T) not in bufs:
            c = self.cfg
            with torch.inference_mode(False):  # a normal tensor: the host writes it outside inference mode
                bufs[(B, T)] = torch.zeros(B, T, c.ngram_heads, c.ngram_dim, dtype=torch.uint8, device=self.embed_tokens.weight.device)
        return bufs[(B, T)]

    def ple_layer(self) -> KernelPLE | None:
        """The PLE layer (None if the config has none)."""
        return next((layer.ple for layer in self.layers if layer.ple is not None), None)

    def host_rows(self, tokens, ctx) -> torch.Tensor:
        """The n-gram rows of tokens int [B, T] with the context ctx int [B, 2] (the two tokens before the first one;
        eos at a sequence start), read from the table: CPU u8 [B, T, 16, 160]. A host helper for tests and tools (the
        scheduler uses engine/weights/ple.py's feeder)."""
        ple = self.ple_layer()
        ids = ngram_rows(np.asarray(tokens), np.asarray(ctx), ple.params)
        B, T = ids.shape[:2]
        return torch.from_numpy(ple.table.rows(ids)).view(B, T, ids.shape[-1], -1)

    def _arange(self, T: int, device) -> torch.Tensor:
        ar = self.__dict__.setdefault("_ar", {})
        if T not in ar:
            ar[T] = torch.arange(T, dtype=torch.int32, device=device)
        return ar[T]

    def _rope_rows(self, state: FastState4, T: int) -> torch.Tensor:
        """Writes the rotary positions (p + rope_delta, three times) of the T new rows at state.rope_pos[b, pos_t[b] + t] of
        the active slots and returns them: int32 [B * T, 3]."""
        B, L = state.pos_t.shape[0], state.rope_pos.shape[1]
        p = state.pos_t.view(B, 1) + self._arange(T, state.pos_t.device).view(1, T)  # [B, T] cache positions
        r = (p + state.rope_delta.view(B, 1)).view(B, T, 1).expand(B, T, 3)
        idx = p.clamp(max=L - 1).long().view(B, T, 1).expand(B, T, 3)
        new = torch.where(state.active.view(B, 1, 1) != 0, r, state.rope_pos.gather(1, idx))
        state.rope_pos.scatter_(1, idx, new)
        return new.reshape(B * T, 3).contiguous()

    def _stream(self, input_ids: torch.Tensor, state: FastState4, ngram_rows: torch.Tensor | None) -> torch.Tensor:
        """The layers over T rows per slot: the stream before the final mixer, x4 bf16 [B * T, 10240]."""
        B, T = input_ids.shape
        rows = ngram_rows if ngram_rows is not None else self.feed_buffer(B, T)
        rope_rows = self._rope_rows(state, T)
        x4 = self.embed_tokens(input_ids).repeat(1, 1, self.cfg.hc_count).view(B * T, -1)
        for layer in self.layers:
            x4 = layer(x4, state, B, T, rope_rows, rows)
        return x4

    def forward(self, input_ids: torch.Tensor, state: FastState4, ngram_rows: torch.Tensor | None = None) -> torch.Tensor:
        """One decode step: input_ids [B, 1] at positions state.pos_t -> fp32 logits [B, V]; pos_t += active."""
        B = input_ids.shape[0]
        x4 = self._stream(input_ids, state, ngram_rows)
        logits = self.lm_head(hc_rows(self.mixer, x4))
        state.pos_t += state.active
        return logits.view(B, -1)

    def verify(self, input_ids: torch.Tensor, state: FastState4, ngram_rows: torch.Tensor | None = None):
        """Speculative verify of T <= 16 rows per slot: (fp32 logits [B, T, V], the stream before the final mixer hidden4
        [B, T, 10240]). KV, raw indexer keys, block keys and rotary positions are written for all T rows; the GDN state,
        the PLE conv state and the positions are left for commit()."""
        B, T = input_ids.shape
        assert T <= MAX_ROWS, f"verify takes at most {MAX_ROWS} rows per slot"
        state.spec = True
        try:
            x4 = self._stream(input_ids, state, ngram_rows)
            logits = self.lm_head(hc_rows(self.mixer, x4))
        finally:
            state.spec = False
        return logits.view(B, T, -1), x4.view(B, T, -1)

    def commit(self, state: FastState4, n: torch.Tensor):
        """Accept the first n[b] (int32, device) of the T verified rows per slot: GDN and PLE state forward, pos_t += n."""
        for i, layer in enumerate(self.layers):
            if layer.block_type == "linear_attention":
                layer.linear_attn.commit(state, i, n)
            if layer.ple is not None:
                layer.ple.commit(state, i, n)
        state.pos_t += n


# ------------------------------------------------------------------------------------------------------------ decode copies
def _copy_of(t: dict, mods, pad_to: int | None = None, out_fp32: bool = False):
    """The decode copy of the stacked modules `mods` (checkpoint names without .weight) from the tensors t of
    tools/requant_qwen4exp.py files: an IntLinear (INT6 / INT5: .qweight_lo / .qweight_hi / .weight_scale /
    .weight_scale_2) or an Nvfp4Linear (.weight / .weight_scale / .weight_scale_2), or None when t has none of them. The
    parts must share one global scale (one launch). pad_to: zero rows up to that many rows (code 2^(b-1): q = 0)."""
    if all(m + ".qweight_lo" in t for m in mods):
        kind, keys = "int", (".qweight_lo", ".qweight_hi", ".weight_scale")
    elif all(m + ".weight" in t and m + ".weight_scale" in t for m in mods):
        kind, keys = "nvfp4", (".weight", ".weight_scale")
    elif any(m + ".qweight_lo" in t or m + ".weight" in t for m in mods):
        raise RuntimeError(f"the decode copies hold only some of the stacked modules {list(mods)}: run tools/requant_qwen4exp.py again")
    else:
        return None
    gs = {float(t[m + ".weight_scale_2"]) for m in mods}
    if len(gs) != 1:
        raise RuntimeError(f"{list(mods)} must share one global scale to stay one launch (tools/requant_qwen4exp.py)")
    parts = [[t[m + k] for k in keys] for m in mods]
    rows = sum(p[0].shape[0] for p in parts)
    if pad_to is not None and pad_to > rows:
        if kind != "int":
            raise RuntimeError("only an INT copy takes zero rows")
        n = pad_to - rows
        lo, hi, sf = parts[0]
        bits = 6 if hi.shape[1] * 4 == lo.shape[1] * 2 else 5
        hi_zero = 0xAA if bits == 6 else 0xFF  # the high bits of code 32 (INT6: 2 per code) / 16 (INT5: 1 per code)
        parts.append([torch.zeros(n, lo.shape[1], dtype=lo.dtype, device=lo.device), torch.full((n, hi.shape[1]), hi_zero, dtype=hi.dtype,
                      device=hi.device), torch.ones(n, sf.shape[1], device=sf.device).to(sf.dtype)])
    cat = [torch.cat([p[j] for p in parts]).contiguous() for j in range(len(keys))]
    if kind == "int":
        return IntLinear(*cat, gs.pop())
    return Nvfp4Linear(*cat, gs.pop(), out_fp32=out_fp32)


def _check(dec, n: int, k: int, what: str):
    if (dec.out_features, dec.in_features) != (n, k):
        raise ValueError(f"the decode copy of {what} is [{dec.out_features}, {dec.in_features}], the model's is [{n}, {k}]")


def attach_decode_copies4(model: FastQwen4Exp, files) -> int:
    """Point the decode path of the BF16 matrices at the requantized copies in `files` (tools/requant_qwen4exp.py; each
    file holds one class, and any subset may be given): the GDN qkvz / out_proj and the attention qkvi / o_proj (.dec of
    the StackedBf16Linear / Bf16Linear), the lm_head (.dec, fp32 logits), the down | inject and W_up of each gated residual
    (KernelGatedResidual.set_decode_copies) and the shared expert's gate | up (KernelMoE.set_shared_copies). The prefill
    keeps the BF16 weights. Call after to_fast4 and before capturing graphs. Returns the number of linears attached
    (a stacked copy counts each of its parts)."""
    from engine.weights.direct import read_tensors
    from engine.weights.loader import PREFIX
    dev = model.embed_tokens.weight.device
    t: dict[str, torch.Tensor] = {}
    for f in files:
        t.update(read_tensors(f, device=dev))
    n = 0

    def linear(mod, mods, what, out_fp32=False):
        nonlocal n
        d = _copy_of(t, mods, out_fp32=out_fp32)
        if d is not None:
            _check(d, mod.w.shape[0], mod.w.shape[1], what)
            mod.dec = d
            n += len(mods)

    def residual(gr: hc_kernels.KernelGatedResidual, base: str):
        nonlocal n
        mods = [base + "input_mix_weight_down"] + ([base + "block_inject_weight"] if gr.has_inj else [])
        d = _copy_of(t, mods, pad_to=-(-gr.w_di.shape[0] // 8) * 8)
        if d is not None:
            gr.set_decode_copies(dec=d)
            n += len(mods)
        up = base + "input_mix_weight_up"
        if up + ".qweight" in t:
            gr.set_decode_copies(w_up_q=t[up + ".qweight"], w_up_s=t[up + ".weight_scale"])
            n += 1
    for i, layer in enumerate(model.layers):
        p = f"{PREFIX}layers.{i}."
        if layer.block_type == "linear_attention":
            g = layer.linear_attn
            linear(g.qkvz, [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"], f"layer {i} qkvz")
            linear(g.out_proj, [p + "linear_attn.out_proj"], f"layer {i} out_proj")
        else:
            a = layer.self_attn
            linear(a.qkvi, [p + f"self_attn.{m}" for m in ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")], f"layer {i} qkvi")
            linear(a.o_proj, [p + "self_attn.o_proj"], f"layer {i} o_proj")
        residual(layer.attn_hc, p + "attn_hyper_connection.")
        residual(layer.mlp_hc, p + "mlp_hyper_connection.")
        d = _copy_of(t, [p + "mlp.shared_expert.gate_proj", p + "mlp.shared_expert.up_proj"])
        if d is not None:
            if not isinstance(d, IntLinear):
                raise RuntimeError("the shared expert's gate | up copy must be INT6 / INT5 (KernelMoE.set_shared_copies)")
            layer.mlp.set_shared_copies(d)
            n += 2
    residual(model.mixer, PREFIX + "hyper_connection_mixer.")
    linear(model.lm_head, ["lm_head"], "lm_head", out_fp32=True)
    return n


# ------------------------------------------------------------------------------------------------------------ build
def check_dims(cfg: Qwen4ExpConfig):
    """Raises ValueError if the config's dimensions differ from the ones the kernels are compiled for."""
    k = ops()
    want = [
        ("hidden_size", cfg.hidden_size, hc_kernels.HIDDEN, "csrc/hc.cu"),
        ("hc_count", cfg.hc_count, hc_kernels.STREAMS, "csrc/hc.cu"),
        ("hc_lowrank", cfg.hc_lowrank, hc_kernels.LOWRANK, "csrc/hc.cu"),
        ("head_dim", cfg.head_dim, k.HEAD_DIM, "csrc/shapes.h"),
        ("linear_key_head_dim", cfg.linear_key_head_dim, k.GDN_DK, "csrc/shapes.h"),
        ("linear_value_head_dim", cfg.linear_value_head_dim, k.GDN_DV, "csrc/shapes.h"),
        ("num_attention_heads", cfg.num_attention_heads, 24, "csrc/qsa.cu"),
        ("num_key_value_heads", cfg.num_key_value_heads, 2, "csrc/qsa.cu"),
        ("indexer_n_heads", cfg.indexer_n_heads, qsa_kernels.INDEX_HEADS, "csrc/qsa.cu"),
        ("indexer_head_dim", cfg.indexer_head_dim, qsa_kernels.INDEX_DIM, "csrc/qsa.cu"),
        ("indexer_compress_ratio", cfg.indexer_compress_ratio, qsa_kernels.RATIO, "csrc/qsa.cu"),
        ("indexer_budget / indexer_compress_ratio", cfg.block_topk, qsa_kernels.TOPK_BLOCKS, "csrc/qsa.cu"),
        ("num_experts_per_tok", cfg.num_experts_per_tok, 10, "csrc/moe.cu"),
    ]
    bad = [f"{name} = {have} (the kernels: {need}, {src})" for name, have, need, src in want if have != need]
    if cfg.moe_intermediate_size not in MOE_INTER:
        bad.append(f"moe_intermediate_size = {cfg.moe_intermediate_size} (the kernels: one of {MOE_INTER}, csrc/moe.cu)")
    if not 10 <= cfg.num_experts <= 512:
        bad.append(f"num_experts = {cfg.num_experts} (the kernels: 10 to 512, csrc/moe.cu)")
    if cfg.shared_expert_intermediate_size % 128 or cfg.vocab_size % 8 or cfg.ple_embed_dim % 128:
        bad.append(f"shared_expert_intermediate_size {cfg.shared_expert_intermediate_size} and ple_embed_dim {cfg.ple_embed_dim} must be "
                   f"multiples of 128 and vocab_size {cfg.vocab_size} of 8 (the BF16 skinny GEMM, csrc/skinny.cu)")
    if bad:
        raise ValueError("this qwen4_exp checkpoint does not fit the compiled kernels: " + "; ".join(bad))


def to_fast4(model: Qwen4ExpForCausalLM) -> FastQwen4Exp:
    """Moves a reference model with NVFP4 experts (load_qwen4exp) onto the kernel decode path, in place: kernel
    attention / GDN / MoE / PLE / hyper-connections, stacked BF16 projections. The row-major expert scales are freed
    (KernelMoE keeps its swizzled copy). A config whose dims do not fit the kernels is refused."""
    cfg = model.cfg
    check_dims(cfg)
    dev = model.embed_tokens.weight.device
    inv_freq = rope_inv_freq(cfg, dev)
    gate_act = 1 if cfg.output_gate_type == "sigmoid" else 0
    for layer in model.layers:
        if layer.block_type == "full_attention":
            KernelQSAAttention.adopt(layer.self_attn, inv_freq, cfg.mrope_section[1:])
        else:
            KernelGDN4.adopt(layer.linear_attn, gate_act)
        if layer.ple is not None:
            KernelPLE.adopt(layer.ple)
        moe = layer.mlp
        ex = moe.experts
        if ex.storage != "nvfp4":
            raise ValueError("the decode path needs NVFP4 experts (ExpertWeights storage 'nvfp4')")
        sh = moe.shared_expert
        kmoe = KernelMoE(ex.w_gu, ex.sf_gu, ex.w_dn, ex.sf_dn, ex.gs, ex.in_scale, moe.gate.weight.detach(), moe.shared_expert_gate.weight.detach(),
                         sh.gate_proj.weight.detach(), sh.up_proj.weight.detach(), sh.down_proj.weight.detach())
        del layer.mlp, moe, ex, sh  # the row-major scales go with the reference module
        layer.mlp = kmoe
        layer.attn_hc = gated_residual(layer.attn_hyper_connection)
        layer.mlp_hc = gated_residual(layer.mlp_hyper_connection)
        del layer.attn_hyper_connection, layer.mlp_hyper_connection
        layer.__class__ = FastDecoderLayer4
    model.mixer = gated_residual(model.hyper_connection_mixer)
    del model.hyper_connection_mixer
    model.lm_head = Bf16Linear(model.lm_head.weight, out_fp32=True)
    model.__class__ = FastQwen4Exp
    model.requires_grad_(False)
    torch.cuda.empty_cache()
    return model


YARN_MAX_FACTOR = 4  # the model card: "262K natively and extensible up to 1,000,000 tokens"


def load_fast_qwen4exp(path_or_repo: str, decode_weights=None, log=print, prep=None, max_seq_len: int | None = None,
                       device="cuda") -> tuple[str, FastQwen4Exp]:
    """The family's load (engine/model/family.py): (the resolved path, the decode model) = load_qwen4exp (the reference
    tree with NVFP4 experts, the n-gram table on disk) + to_fast4. decode_weights and prep (a prepared directory) are
    accepted and not used here: the family's load attaches the requantized decode copies of the prepared directory
    (attach_decode_copies4) when decode_weights is "int". A max_seq_len beyond the native context turns on YaRN by the
    ratio (qwen4exp.apply_yarn4, as Qwen documents up to 1M tokens), before to_fast4 reads the frequencies and the norms."""
    from engine.model.qwen4exp import apply_yarn4
    from engine.model.qwen35 import yarn_attention_factor
    from engine.weights.loader import resolve
    from engine.weights.qwen4exp_loader import load_qwen4exp
    path = resolve(path_or_repo)
    native = Qwen4ExpConfig.from_checkpoint(path).max_position_embeddings
    if max_seq_len is not None and max_seq_len > YARN_MAX_FACTOR * native:  # refused from the config, before any weight is read
        raise ValueError(f"max_seq_len {max_seq_len} > {YARN_MAX_FACTOR} x the native context {native}: Qwen documents YaRN for this model up "
                         "to about 1M tokens")
    t0 = time.time()
    ref = load_qwen4exp(path, device=device, log=log)
    if max_seq_len is not None and max_seq_len > native:
        apply_yarn4(ref, max_seq_len / native)
        log(f"[fast4] YaRN x{max_seq_len / native:g}: {max_seq_len} tokens per slot over the checkpoint's {native} native positions (rope "
            f"frequencies scaled, the rotated dims of the attention's and the indexer's q / k x{yarn_attention_factor(ref.cfg):.4f})")
    model = to_fast4(ref)
    log(f"[fast4] decode model ready in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on the GPU")
    return path, model
