"""Prefill of Qwen3.8-Flash-Next (qwen4_exp): a prompt of one slot in chunks of up to CHUNK tokens, each layer over the
whole chunk (docs/history/qwen4exp_plan.md, milestone M2). The decode model (engine/model/fast_qwen4exp.py,
FastQwen4Exp) and its state (FastState4) are the same objects; after prefill the state is exactly what decode expects:
K / V, raw indexer keys, block keys and rope record written, GDN conv / recurrent state and PLE conv state advanced,
pos and pos_t moved.

The layer functions take explicit tensors (the module weights, the state at a layer index), so each one has its own
test against the reference modules of engine/model/qwen4exp.py (tests/test_prefill_qwen4exp.py):

- hc_pre / hc_post: the gated residual of the hyper-connections. The grouped norm runs on ops().hc_norm and the
  injection on ops().hc_post (both bit-exact against transformers); the products on cuBLAS bf16 matmuls; the sigmoid,
  the product and the ordered stream mean in one kernel (ops().hc_mix_rows, csrc/hc_prefill.cu); each op with the
  reference's bf16 rounding points (docs/history/qwen4exp_plan.md section 5.4).
- gdn_layer: the 27B's prefill kernels (causal_conv_silu, gdn_prefill, gated_rmsnorm with the sigmoid gate) on bf16
  projections (cuBLAS).
- qsa_layer: the stacked q | k | v | index projection, the prologues, the block keys, the selection and the sparse
  attention of engine/model/qsa_kernels.py (prefill_*), then o_proj.
- moe_layer: KernelMoE.prefill (the CUTLASS grouped NVFP4 GEMMs).
- ple_layer: the PLE layer in torch ops, with the decode model's fixed-order sums (a row's bits do not depend on T) and
  its dilated conv (4 taps in fp32 in tap order) over the slot's 9-row state, advanced by the chunk.

prefill(model, input_ids, state, chunk, ...) runs the layers chunk by chunk, with the signature of
engine/model/prefill.py's prefill plus rows / prev2: the n-gram rows of the prompt (u8 [T, 16, 160]) come in `rows`, or
the model's feeder (model.feeder, engine.weights.ple.PleFeeder) reads them from the tokens and the two tokens before
them (prev2); its worker gathers the next chunk's rows while the GPU runs the current one.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from engine.kernels import ops
from engine.model import qsa_kernels as qk

CHUNK = 2048  # tokens per chunk (bench/prefill_chunk_bench.py measures 2048, 4096 and 8192)
HC = 4  # hyper-connection streams


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _weight(m) -> torch.Tensor:
    """The tensor of a module with a .weight or .w, or the tensor itself."""
    if isinstance(m, torch.Tensor):
        return m
    return m.w if hasattr(m, "w") else m.weight


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """x fp32 [R, n] -> [R]: halves the row while its length is even (x[:, :h] + x[:, h:]), then adds the rest left to
    right. Elementwise kernels only: a row's sum does not depend on R. The same order as the decode model's PLE gate."""
    while x.shape[1] > 1 and x.shape[1] % 2 == 0:
        h = x.shape[1] // 2
        x = x[:, :h] + x[:, h:]
    s = x[:, 0]
    for j in range(1, x.shape[1]):
        s = s + x[:, j]
    return s


def group_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The zero-centered RMSNorm of each 2560-wide stream of x bf16 [T, 10240] (GroupRMSNorm) on ops().hc_norm."""
    out = torch.empty_like(x)
    ops().hc_norm(x.contiguous(), w, eps, out, HC)
    return out


# ------------------------------------------------------------------------------------------------------------ hyper-connections
def hc_pre(x4: torch.Tensor, hc):
    """The gated residual before a sublayer: x4 bf16 [T, 10240] -> (y bf16 [T, 2560], inj bf16 [T, 4]), or y for the
    final mixer (no inject weight). hc: a hc_kernels.KernelGatedResidual (norm_w [10240], w_di = [W_down; W_inj]
    [324 | 320, 10240], w_up [10240, 320], eps). The rounding of each op follows the reference GatedResidual."""
    T = x4.shape[0]
    n = group_norm(x4, hc.norm_w, hc.eps)
    de = n @ hc.w_di.t()  # down | inject: [T, 324] bf16 (one cuBLAS pass)
    r = hc.w_up.shape[1]
    a = F.silu(de[:, :r] / HC)  # bf16 ops: each computes in fp32 and rounds once, as the reference's
    y = torch.empty(T, n.shape[1] // HC, device=x4.device, dtype=torch.bfloat16)
    ops().hc_mix_rows(a @ hc.w_up.t(), n, y)  # bf16((((p0 + p1) + p2) + p3) / 4), p_s = bf16(bf16(sigmoid(u_s)) * n_s)
    if not hc.has_inj:
        return y
    return y, 2 * torch.sigmoid(de[:, r:] / HC)


def hc_post(x4: torch.Tensor, o: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
    """x4_s = bf16(x4_s + bf16(o * inj_s)) in place: x4 bf16 [T, 10240], o bf16 [T, 2560], inj bf16 [T, 4]."""
    ops().hc_post(o.reshape(-1, x4.shape[1] // HC).contiguous(), inj.contiguous(), x4)  # the decode kernel: bit-exact against the reference
    return x4


# ------------------------------------------------------------------------------------------------------------ mixers
def gdn_layer(x: torch.Tensor, g, state, li: int, eps: float = 1e-6) -> torch.Tensor:
    """A Gated DeltaNet layer over T rows of one slot (no residual): x bf16 [T, 2560] (the hyper-connection input) ->
    [T, 2560]. g: the decode model's linear_attn (qkvz.w [16384, 2560] with sizes [10240, 6144], w_ba [96, 2560],
    conv1d.weight [10240, 1, 4], A_log, dt_bias, norm.weight [128], out_proj.w [2560, 6144]). Continues state.conv[li]
    and state.rec[li] of slot 0 (a one-slot view)."""
    T, dev = x.shape[0], x.device
    w_qkvz, sizes = _weight(g.qkvz), list(g.qkvz.sizes)
    mixed, z = (x @ w_qkvz.t()).split(sizes, dim=-1)
    nv = g.w_ba.shape[0] // 2
    ba = x @ g.w_ba.t()
    b, a = ba[:, :nv], ba[:, nv:]
    w_out = _weight(g.out_proj)
    value_dim = w_out.shape[1]
    conv_dim = g.conv1d.weight.shape[0]
    key_dim = (conv_dim - value_dim) // 2
    dk, dv = 128, value_dim // nv
    q = torch.empty(T, key_dim, device=dev, dtype=torch.bfloat16)
    k = torch.empty(T, key_dim, device=dev, dtype=torch.bfloat16)
    v = torch.empty(T, value_dim, device=dev, dtype=torch.bfloat16)
    cs = state.conv[li][0]  # [C, K - 1]
    K = cs.shape[1] + 1
    ops().causal_conv_silu(mixed, cs, g.conv1d.weight, [q, k, v], 1e-6)  # q, k come out L2-normalized per head
    if T >= K - 1:
        cs.copy_(mixed[-(K - 1):].t())
    else:
        cs.copy_(torch.cat([cs, mixed.t()], dim=-1)[:, -(K - 1):])
    beta = b.sigmoid().contiguous()
    gg = (-g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)).contiguous()  # log decay
    o = torch.empty(T, nv, dv, device=dev, dtype=torch.bfloat16)
    ops().gdn_prefill(q.view(T, -1, dk), k.view(T, -1, dk), v.view(T, nv, dv), gg, beta, state.rec[li][0], o, dk ** -0.5)
    on = torch.empty(T * nv, dv, device=dev, dtype=torch.bfloat16)
    ops().gated_rmsnorm(o.view(-1, dv), z, g.norm.weight, getattr(g.norm, "eps", eps), on, 1)  # gate_act 1: sigmoid(z)
    return on.view(T, -1) @ w_out.t()


def qsa_layer(x: torch.Tensor, attn, state, li: int, pos0: int, rope_pos: torch.Tensor | None = None, eps: float = 1e-6,
              path: str | None = None) -> torch.Tensor:
    """An attention layer with Qwen Sparse Attention over the T rows of a chunk of one slot at pos0 (= state.pos): x bf16
    [T, 2560] -> [T, 2560] (no residual). attn: the decode model's self_attn (qkvi.w [13952, 2560] with sizes [12288,
    512, 512, 640], q_norm / k_norm.weight [256], indexer.q_layernorm / k_layernorm.weight [128], o_proj.w, inv_freq,
    mrope). rope_pos: int32 [T, 3] (images), else each row's cache position + state.rope_delta. Writes K / V, ik, ikb
    and the rope record of the rows; state.pos_t holds pos0 on the device."""
    T = x.shape[0]
    qp, kp, vp, idx = (x @ _weight(attn.qkvi).t()).split(list(attn.qkvi.sizes), dim=-1)
    ix = attn.indexer
    mrope = tuple(getattr(attn, "mrope", (11, 10)))
    kc, vc, ik, ikb = state.k[li], state.v[li], state.ik[li], state.ikb[li]
    q, q_idx = qk.prefill_prologue(qp, kp, vp, idx, attn.q_norm.weight, attn.k_norm.weight, ix.q_layernorm.weight, attn.inv_freq, kc, vc, ik,
                                   state.rope_pos, pos0, eps, rope_pos=rope_pos, rope_delta=state.rope_delta, mrope=mrope, pos_t=state.pos_t)
    qk.prefill_block_keys(ik, ikb, ix.k_layernorm.weight, attn.inv_freq, state.rope_pos, pos0, T, eps, mrope, pos_t=state.pos_t)
    sel, nsel = qk.prefill_select(q_idx, ikb, pos0, T)
    o = qk.prefill_attend(q, kc, vc, sel, nsel, pos0, q.shape[3] ** -0.5, qp, path)
    return o @ _weight(attn.o_proj).t()


def moe_layer(x: torch.Tensor, mlp) -> torch.Tensor:
    """The MoE block over a chunk: KernelMoE.prefill (no residual)."""
    return mlp.prefill(x)


# ------------------------------------------------------------------------------------------------------------ PLE
def ple_state(state, li: int) -> torch.Tensor:
    """The PLE conv state [10240, 9] of slot 0 at layer li."""
    return state.ple_conv[li][0]


def ple_layer(x4: torch.Tensor, rows: torch.Tensor, ple, state, li: int) -> torch.Tensor:
    """The PLE layer over the T rows of a chunk of one slot, added to x4 bf16 [T, 10240] in place. rows: u8 [T, 16, 160]
    (the FP8 n-gram rows on the device). ple: the decode model's PLE layer (kv_proj.w [12800, 2560] with sizes [10240,
    2560], norm_key / norm_query / norm_conv (GroupRMSNorm: .weight [10240], .eps), conv1d.weight [10240, 1, 4] with
    dilation 3, scale: the table's dequantization scale). Advances state.ple_conv[li] by T rows.

    e = bf16(e4m3 * scale); kp | vp = e @ kv_proj^T; kn = norm_key(kp), qn = norm_query(x4); g = bf16(sum per stream of
    bf16(kn * qn)) / sqrt(2560); g = sqrt(max(|g|, 1e-6)) * sign(g); gv = sigmoid(g) * vp (bf16); u = norm_conv(gv);
    c[t] = sum_j w_j hist[t + 3 j] in fp32 over hist = [state, u]; x4 += bf16(gv + silu(bf16(c)))."""
    T, C = x4.shape
    H = C // HC
    e = _bf(rows.reshape(T, -1).view(torch.float8_e4m3fn).float() * float(ple.scale))
    w_kv = _weight(ple.kv_proj)
    kp, vp = (e @ w_kv.t()).split(list(ple.kv_proj.sizes), dim=-1)
    kn = group_norm(kp, ple.norm_key.weight, ple.norm_key.eps)
    qn = group_norm(x4, ple.norm_query.weight, ple.norm_query.eps)
    g = _bf(row_sum((kn * qn).float().view(T * HC, H))).view(T, HC, 1)
    g = g / math.sqrt(H)
    g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
    gv = (torch.sigmoid(g) * vp.reshape(T, 1, H)).view(T, C)
    u = group_norm(gv, ple.norm_conv.weight, ple.norm_conv.eps)
    prev = ple_state(state, li)  # [C, 9]
    n = prev.shape[1]
    hist = torch.cat([prev, u.t()], dim=1)  # [C, 9 + T]
    w = ple.conv1d.weight.view(C, -1).float()
    d = ple.conv1d.dilation[0] if hasattr(ple.conv1d, "dilation") else n // (w.shape[1] - 1)
    hf = hist.float()
    acc = hf[:, 0:T] * w[:, 0:1]
    for j in range(1, w.shape[1]):
        acc = acc + hf[:, j * d:j * d + T] * w[:, j:j + 1]
    act = F.silu(_bf(acc))  # [C, T]
    x4.add_(gv + act.t())
    prev.copy_(hist[:, T:T + n])
    return x4


def _feeder(model):
    """(the model's feeder, whether it is a temporary one): model.feeder (engine.weights.ple.PleFeeder, with the row
    cache), else a feeder without a cache on the PLE layer's table."""
    f = getattr(model, "feeder", None)
    if f is not None:
        return f, False
    from engine.weights.ple import PleFeeder
    ple = next(layer.ple for layer in model.layers if getattr(layer, "ple", None) is not None)
    return PleFeeder(ple.table, cache_rows=0, params=ple.params, image_token_id=model.cfg.image_token_id), True


# ------------------------------------------------------------------------------------------------------------ the model
def _embed(model, ids: torch.Tensor, c0: int, embeds) -> torch.Tensor:
    """The input rows of the tokens ids (at offset c0 of the prompt part): the embeddings (the image placeholders past
    the vocabulary read row 0), with the rows that embeds [(offset, rows [n, 2560])] covers replaced."""
    e = model.embed_tokens
    ids = torch.where(ids >= model.cfg.vocab_size, 0, ids)
    x = F.embedding(ids, e) if isinstance(e, torch.Tensor) else e(ids)
    for off, r in embeds or ():
        a, b = max(off, c0), min(off + r.shape[0], c0 + ids.numel())
        if a < b:
            x[a - c0:b - c0] = r[a - off:b - off]
    return x


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, state, chunk: int = CHUNK, all_logits: bool = False, return_hidden: bool = False, embeds=None,
            rope_pos: torch.Tensor | None = None, logits: bool = True, rows: torch.Tensor | None = None, prev2=None, path: str | None = None):
    """input_ids [1, T] from position state.pos of a single-slot FastState4 (a view of one slot). Returns the fp32
    logits of the last token [1, vocab] (the decode model's mixer and lm_head on its row), or with all_logits the bf16
    logits of every token [T, vocab] (a bf16 matmul with lm_head.w; the perplexity tool), or None with logits=False.
    return_hidden: also the 4-stream hidden before the final mixer [T, 10240] (the MTP drafter's input).
    embeds [(offset, rows)] / rope_pos int32 [T, 3]: images, as engine/model/prefill.py takes them.
    The n-gram rows: rows u8 [T, 16, 160] (CPU or device; kept unchanged until the GPU has read them), else the feeder
    of the model (model.feeder, or a temporary one without a cache) reads them from prev2, the two tokens of the slot
    before input_ids (default (eos, eos), which only a sequence start may use). With several chunks, the feeder gathers
    the next chunk's rows on its worker thread while the GPU runs the current chunk. path: the attention path of
    qsa_kernels.prefill_attend (default "union")."""
    assert input_ids.shape[0] == 1, "one slot at a time"
    T_all = input_ids.shape[1]
    if state.pos + T_all > state.max_seq_len:
        raise ValueError("prompt exceeds the slot's max_seq_len")
    has_ple = any(getattr(layer, "ple", None) is not None for layer in model.layers)
    if rows is not None and rows.shape[0] != T_all:
        raise ValueError(f"rows: {rows.shape[0]} rows for {T_all} tokens")
    dev = input_ids.device
    feeder, own, hist = None, False, None
    if has_ple and rows is None:
        if prev2 is None:
            if state.pos != 0:
                raise ValueError("the n-gram rows of a continued prompt depend on the slot's history: pass prev2 (or rows)")
            prev2 = (model.cfg.eos_token_id,) * 2
        feeder, own = _feeder(model)
        hist = np.concatenate([np.asarray(prev2, np.int64).reshape(-1), input_ids[0].cpu().numpy().astype(np.int64)])

    def host_rows(c0: int, T: int, i: int, now: bool):
        """The rows of the chunk at c0 in staging buffer i: read now (rows_for), or started on the worker (prefetch)."""
        tok, ctx = hist[c0 + 2:c0 + 2 + T], hist[c0:c0 + 2]
        buf = feeder.staging(i, T)
        return feeder.rows_for(tok, ctx, out=buf) if now else feeder.prefetch(tok, ctx, out=buf)

    eps = model.cfg.rms_norm_eps
    outs, hid, x_last = [], [], None
    try:
        for k, c0 in enumerate(range(0, T_all, chunk)):
            ids = input_ids[0, c0:c0 + chunk]
            T = ids.numel()
            rows_c = None
            if rows is not None:
                rc = rows[c0:c0 + T]
                rows_c = rc.to(dev, non_blocking=rc.is_pinned())
            elif feeder is not None:
                rows_c = host_rows(c0, T, k % 2, True).to(dev, non_blocking=True)
                feeder.release(k % 2)
                if c0 + chunk < T_all:  # the next chunk's rows: gathered while the GPU runs this chunk
                    host_rows(c0 + chunk, min(chunk, T_all - c0 - chunk), (k + 1) % 2, False)
            rp = rope_pos[c0:c0 + T] if rope_pos is not None else None
            x4 = _embed(model, ids, c0, embeds).repeat(1, HC)  # [T, 10240]
            pos0 = state.pos
            for li, layer in enumerate(model.layers):
                if getattr(layer, "ple", None) is not None:
                    ple_layer(x4, rows_c, layer.ple, state, li)
                y, inj = hc_pre(x4, layer.attn_hc)
                if layer.block_type == "linear_attention":
                    o = gdn_layer(y, layer.linear_attn, state, li, eps)
                else:
                    o = qsa_layer(y, layer.self_attn, state, li, pos0, rp, eps, path)
                hc_post(x4, o, inj)
                y, inj = hc_pre(x4, layer.mlp_hc)
                hc_post(x4, moe_layer(y, layer.mlp), inj)
            state.pos += T
            state.pos_t += T
            x_last = x4[-1:]
            if return_hidden:
                hid.append(x4)
            if all_logits:
                outs.append(hc_pre(x4, model.mixer) @ _weight(model.lm_head).t())
    finally:
        if own:
            torch.cuda.current_stream().synchronize()  # the staging buffers die with the feeder
            feeder.close()
    if all_logits:
        out = torch.cat(outs)
    elif logits:
        out = model.lm_head(model.mixer(x_last)).float()
    else:
        out = None
    return (out, torch.cat(hid)) if return_hidden else out
