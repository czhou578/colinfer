"""Prefill path: a prompt in chunks of up to CHUNK tokens, each layer on tensor-core GEMMs over the whole chunk.

  MLP          W4A4: activations quantized to NVFP4 (static checkpoint input scale) inside the fused add + RMSNorm,
               CUTLASS SM120 NVFP4 x NVFP4 GEMMs (csrc/gemm_nvfp4.cu): the gate GEMM, then the up GEMM computes
               silu(gate) * up and quantizes it to NVFP4 in its epilogue; the down GEMM adds the residual.
  FP8 linears  W8A8: activations quantized to e4m3 with the static input scale, cuBLASLt via torch._scaled_mm (row-wise
               weight scales, so the stacked q/k/v and qkv/z keep their own scales).
  attention    the decode prologue on T rows (q/k norm + RoPE + fp8 KV write), then causal attention over the slot's
               cache: FlashInfer FA2 over a bf16 copy of the cached prefix, or past ATTN_FP8_MIN_CTX tokens of context
               csrc/attn_prefill.cu (Q K^T on FP8 tensor cores over the cache as stored: 6% faster prefill at 64k,
               11% at 128k; perplexity +0.25-0.3% from rounding Q to e4m3); output gate fused with the FP8 quantization.
  GDN          causal conv + SiLU + q/k L2 norm (csrc/prefill_ops.cu) continuing the conv window, the chunked gated
               delta rule (csrc/gdn_prefill.cu) continuing the recurrent state, gated RMSNorm, out_proj.

Works on the same FastQwen35 model / FastState as the decode graph; afterwards the state is exactly what decode expects
(KV written, conv / recurrent state advanced, pos and pos_t moved). Only the last token's logits are computed, unless
all_logits (perplexity).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import ops
from engine.model.fast import FastQwen35, FastState, Nvfp4Linear, StackedFp8Linear

CHUNK = 2048            # tokens per chunk: larger chunks make the FP8 GEMMs slower per token (docs/history/phase6_progress.md)
ATTN_FP8_MIN_CTX = 16384  # context length above which prefill attention runs on csrc/attn_prefill.cu


def _nvfp4_operands(lin_list):
    """Stack NVFP4 linears row-wise (shared K, input scale, weight global scale) into one GEMM operand
    with CUTLASS-swizzled scales; the originals become views so the decode GEMMs keep working."""
    w = torch.cat([l.w for l in lin_list]).contiguous()
    sf = torch.cat([l.sf for l in lin_list]).contiguous()
    gs, ins = {l.gscale for l in lin_list}, {l.in_scale for l in lin_list}
    if len(gs) != 1 or len(ins) != 1:
        raise ValueError("stacked NVFP4 linears must share their global scales")
    N, K = w.shape[0], w.shape[1] * 2
    sfz = torch.empty(ops().nvfp4_sf_size(N, K), dtype=torch.uint8, device=w.device)
    ops().nvfp4_swizzle_sf(sf.view(torch.uint8), sfz, K)
    off = 0
    for l in lin_list:
        n = l.out_features
        l.w, l.sf = w[off:off + n], sf[off:off + n]
        off += n
    return dict(w=w, sf=sfz, alpha=gs.pop() * next(iter(ins)), in_scale=ins.pop(), N=N, K=K)


def prepare_prefill(model: FastQwen35):
    """One-time: stacked / swizzled NVFP4 operands for the MLPs. Idempotent."""
    if getattr(model, "_prefill_ready", False):
        return
    for layer in model.layers:
        mlp = layer.mlp
        mlp.p_gu = _nvfp4_operands([mlp.gate, mlp.up])
        mlp.p_down = _nvfp4_operands([mlp.down])
        # gate and up halves of the stacked operand for the fused SwiGLU (rows: I is a multiple of 128, so the up half's
        # swizzled scales are a contiguous slice)
        I, sfh = mlp.gate.out_features, mlp.p_gu["sf"].numel() // 2
        mlp.p_g = dict(mlp.p_gu, w=mlp.p_gu["w"][:I], sf=mlp.p_gu["sf"][:sfh], N=I)
        mlp.p_u = dict(mlp.p_gu, w=mlp.p_gu["w"][I:], sf=mlp.p_gu["sf"][sfh:], N=I)
        mlp.p_down["nc"] = torch.tensor([1.0 / mlp.p_down["in_scale"]], dtype=torch.float32, device=mlp.p_gu["w"].device)
    if isinstance(model.lm_head, Nvfp4Linear):  # for all-token logits (perplexity); decode uses the skinny GEMM
        model.p_lm = _nvfp4_operands([model.lm_head])
    torch.cuda.empty_cache()
    model._prefill_ready = True


def _gemm_nvfp4(xq, xsf, op, residual=None):
    out = torch.empty(xq.shape[0], op["N"], device=xq.device, dtype=torch.bfloat16)
    ops().nvfp4_gemm(xq, xsf, op["w"], op["sf"], op["alpha"], residual, out, 0 if xq.shape[0] >= 1536 else 1)
    return out


def _fp8_gemm(xq, lin):
    """xq [M, K] e4m3 (already quantized with lin.in_scale) -> [M, N] bf16 (W8A8, row-wise weight scales)."""
    M = xq.shape[0]
    sa = torch.full((M, 1), lin.in_scale, dtype=torch.float32, device=xq.device)
    if isinstance(lin, StackedFp8Linear):
        sb = lin.rs.view(1, -1)
    else:
        sb = torch.full((1, lin.out_features), lin.scale, dtype=torch.float32, device=xq.device)
    return torch._scaled_mm(xq, lin.w.view(torch.float8_e4m3fn).t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def _norm_in(layer, x, need_bf16: bool):
    """input_layernorm of x -> (bf16 normed or None, e4m3 normed quantized for the mixer's first GEMM)."""
    lin = layer.self_attn.qkv if layer.block_type == "full_attention" else layer.linear_attn.qkvz
    q8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n = torch.empty_like(x) if need_bf16 else None
    ops().add_rmsnorm(x, None, layer.input_layernorm.weight, layer.input_layernorm.eps, n_out=n, q8=q8, in_scale8=lin.in_scale)
    return n, q8


def _mlp(layer, x, y):
    """x_new = x + y (mixer output); returns x_new + mlp(post_norm(x_new)) with the residual in the down GEMM epilogue."""
    mlp = layer.mlp
    M, K = x.shape
    x_new = torch.empty_like(x)
    xq = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
    xsf = torch.empty(ops().nvfp4_sf_size(M, K), dtype=torch.uint8, device=x.device)
    ops().add_rmsnorm(x, y, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps, x_out=x_new, q4=xq, sf4=xsf,
                      in_scale4=mlp.p_gu["in_scale"])
    I = mlp.p_gu["N"] // 2
    hq = torch.empty(M, I // 2, dtype=torch.uint8, device=x.device)
    hsf = torch.empty(ops().nvfp4_sf_size(M, I), dtype=torch.uint8, device=x.device)
    g = _gemm_nvfp4(xq, xsf, mlp.p_g)  # the gate GEMM, then the up GEMM computes silu(gate) * up and quantizes it
    u = mlp.p_u
    ops().nvfp4_gemm_swiglu(xq, xsf, u["w"], u["sf"], u["alpha"], g, hq, hsf, mlp.p_down["nc"], 0 if M >= 1536 else 1)
    return _gemm_nvfp4(hq, hsf, mlp.p_down, x_new)


def _attention(attn, q8, state: FastState, li: int):
    """Mixer output (no residual) of a full-attention layer."""
    import flashinfer
    T = q8.shape[0]
    qp, kp, vp = _fp8_gemm(q8, attn.qkv).split(attn.qkv.sizes, dim=-1)  # strided column views, read in place
    kc, vc = state.k[li], state.v[li]
    q = torch.empty(1, attn.num_heads, T, attn.head_dim, device=q8.device, dtype=torch.bfloat16)
    ops().attn_prologue(qp, kp, vp, attn.q_norm.weight, attn.k_norm.weight, attn.inv_freq, state.pos_t, kc, vc, q, attn.q_norm.eps)
    L = state.pos + T
    if L > ATTN_FP8_MIN_CTX:  # FP8 Q K^T over the cache as stored
        o = torch.empty(T, attn.num_heads * attn.head_dim, device=q8.device, dtype=torch.bfloat16)
        ops().attn_prefill_fp8(q, kc, vc, o, state.pos, attn.head_dim ** -0.5)
    else:
        # FlashInfer's FP8-KV prefill runs ~48 TFLOPS vs ~80 for BF16 on sm_121: casting the cached prefix to a BF16
        # scratch first is cheaper (and exact: e4m3 -> bf16 is lossless)
        kk, vv = kc[0, :, :L].to(torch.bfloat16), vc[0, :, :L].to(torch.bfloat16)
        o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), kk, vv, causal=True, kv_layout="HND",
                                                    sm_scale=attn.head_dim ** -0.5).reshape(T, -1)
    o8 = torch.empty(T, attn.num_heads * attn.head_dim, dtype=torch.float8_e4m3fn, device=q8.device)
    ops().gate_fp8(o, qp, attn.head_dim, attn.o_proj.in_scale, o8)
    return _fp8_gemm(o8, attn.o_proj)


def _gdn(g, n, q8, state: FastState, li: int):
    """Mixer output (no residual) of a Gated DeltaNet layer. n: bf16 normed input (for the b/a projection)."""
    T = n.shape[0]
    mixed, z = _fp8_gemm(q8, g.qkvz).split(g.qkvz.sizes, dim=-1)
    ba = n @ g.w_ba.t()
    b, a = ba[:, : g.num_v_heads], ba[:, g.num_v_heads:]
    K = g.conv_kernel_size
    dev = n.device
    q = torch.empty(T, g.key_dim, device=dev, dtype=torch.bfloat16)
    k = torch.empty(T, g.key_dim, device=dev, dtype=torch.bfloat16)
    v = torch.empty(T, g.value_dim, device=dev, dtype=torch.bfloat16)
    cs = state.conv[li][0]  # [C, K-1]
    ops().causal_conv_silu(mixed, cs, g.conv1d.weight, [q, k, v], 1e-6)  # q, k come out L2-normalized per head
    if T >= K - 1:
        cs.copy_(mixed[-(K - 1):].t())
    else:
        cs.copy_(torch.cat([cs, mixed.t()], dim=-1)[:, -(K - 1):])
    beta = b.sigmoid().contiguous()
    gg = (-g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)).contiguous()  # log decay
    o = torch.empty(T, g.num_v_heads, g.head_v_dim, device=dev, dtype=torch.bfloat16)  # continues state.rec in place
    ops().gdn_prefill(q.view(T, -1, g.head_k_dim), k.view(T, -1, g.head_k_dim), v.view(T, -1, g.head_v_dim), gg, beta, state.rec[li][0], o,
                      g.head_k_dim ** -0.5)
    on = torch.empty(T * g.num_v_heads, g.head_v_dim, device=dev, dtype=torch.bfloat16)
    ops().gated_rmsnorm(o.reshape(-1, g.head_v_dim), z, g.norm.weight, g.norm.eps, on)
    o8 = torch.empty(T, g.value_dim, dtype=torch.float8_e4m3fn, device=dev)
    ops().fp8_quant(on.view(T, -1), g.out_proj.in_scale, o8)
    return _fp8_gemm(o8, g.out_proj)


@torch.inference_mode()
def prefill(model: FastQwen35, input_ids: torch.Tensor, state: FastState, chunk: int = CHUNK, all_logits: bool = False,
            return_hidden: bool = False):
    """input_ids [1, T] from position state.pos of a single-slot state. Returns the fp32 logits of the last token
    [1, vocab], or with all_logits the bf16 logits of every token [T, vocab] (lm_head as a W4A4 GEMM; for perplexity).
    return_hidden: also the post-final-norm hidden state of every prompt position [T, H] (the MTP drafter's input)."""
    assert input_ids.shape[0] == 1, "one slot at a time"
    prepare_prefill(model)
    T_all = input_ids.shape[1]
    if state.pos + T_all > state.max_seq_len:
        raise ValueError("prompt exceeds the slot's max_seq_len")
    x_last, outs, hid = None, [], []
    for c0 in range(0, T_all, chunk):
        ids = input_ids[0, c0:c0 + chunk]
        T = ids.numel()
        x = model.embed_tokens(ids)  # [T, H] bf16
        for li, layer in enumerate(model.layers):
            gdn = layer.block_type == "linear_attention"
            n, q8 = _norm_in(layer, x, need_bf16=gdn)
            y = _gdn(layer.linear_attn, n, q8, state, li) if gdn else _attention(layer.self_attn, q8, state, li)
            x = _mlp(layer, x, y)
        state.pos += T
        state.pos_t += T
        x_last = x[-1:]
        if return_hidden:
            hid.append(model.norm(x))
        if all_logits:
            xq = torch.empty(T, x.shape[1] // 2, dtype=torch.uint8, device=x.device)
            xsf = torch.empty(ops().nvfp4_sf_size(T, x.shape[1]), dtype=torch.uint8, device=x.device)
            ops().add_rmsnorm(x, None, model.norm.weight, model.norm.eps, q4=xq, sf4=xsf, in_scale4=model.p_lm["in_scale"])
            outs.append(_gemm_nvfp4(xq, xsf, model.p_lm))
    out = torch.cat(outs) if all_logits else model.lm_head(model.norm(x_last)).float()
    return (out, torch.cat(hid)) if return_hidden else out
