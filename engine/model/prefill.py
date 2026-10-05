"""Prefill path (PLAN.md Phase 3): whole chunks of up to CHUNK tokens per layer on tensor cores.

  MLP        W4A4: activations quantized to NVFP4 (static checkpoint input scale), CUTLASS SM120
             NVFP4 x NVFP4 GEMM (csrc/gemm_nvfp4.cu); gate and up are one stacked GEMM; the down
             GEMM adds the residual in its epilogue.
  FP8 linears W8A8: activations quantized to e4m3 with the static input scale, cuBLASLt via
             torch._scaled_mm (row-wise weight scales, so stacked q/k/v and qkv/z keep their own scales).
  attention  fused q/k norm + RoPE + KV write (csrc/attn_decode.cu prologue, T rows), FlashInfer FA2
             causal prefill over the slot's cache, output gate, o_proj.
  GDN        causal conv over the chunk continuing the conv state, FLA chunk_gated_delta_rule continuing
             the recurrent state, gated RMSNorm, out_proj.

Works on the same FastQwen35 model / FastState as the decode graph; after prefill the state is
exactly what decode expects (KV written, conv / recurrent state advanced, pos and pos_t moved).
Only the logits of the last token are computed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kernels import ops
from engine.model.fast import FastQwen35, FastState, Nvfp4Linear, StackedFp8Linear

CHUNK = 2048


def _nvfp4_operands(lin_list):
    """Stack NVFP4 linears row-wise (shared K, input scale, weight global scale) into one GEMM operand
    with CUTLASS-swizzled scales; the originals become views so the decode GEMV keeps working."""
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
    if isinstance(model.lm_head, Nvfp4Linear):  # for all-token logits (perplexity); decode uses the GEMV
        model.p_lm = _nvfp4_operands([model.lm_head])
    torch.cuda.empty_cache()
    model._prefill_ready = True


def _quant_nvfp4(x2, in_scale):
    M, K = x2.shape
    q = torch.empty(M, K // 2, dtype=torch.uint8, device=x2.device)
    sf = torch.empty(ops().nvfp4_sf_size(M, K), dtype=torch.uint8, device=x2.device)
    ops().nvfp4_quant(x2, in_scale, q, sf)
    return q, sf


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
    gu = _gemm_nvfp4(xq, xsf, mlp.p_gu)
    I = gu.shape[1] // 2
    hq = torch.empty(M, I // 2, dtype=torch.uint8, device=gu.device)
    hsf = torch.empty(ops().nvfp4_sf_size(M, I), dtype=torch.uint8, device=gu.device)
    ops().silu_mul_quant(gu, mlp.p_down["in_scale"], hq, hsf)
    return _gemm_nvfp4(hq, hsf, mlp.p_down, x_new)


def _attention(attn, q8, state: FastState, li: int):
    """Mixer output (no residual) of a full-attention layer."""
    import flashinfer
    T = q8.shape[0]
    qp, kp, vp = _split(_fp8_gemm(q8, attn.qkv), attn.qkv.sizes)  # strided column views, read in place
    kc, vc = state.k[li], state.v[li]
    q = torch.empty(1, attn.num_heads, T, attn.head_dim, device=q8.device, dtype=torch.bfloat16)
    ops().attn_prologue(qp, kp, vp, attn.q_norm.weight, attn.k_norm.weight, attn.inv_freq, state.pos_t, kc, vc, q, attn.q_norm.eps)
    L = state.pos + T
    kk, vv = kc[0, :, :L], vc[0, :, :L]
    if kk.dtype == torch.uint8:  # fp4 cache rows: dequantize the prefix per head into bf16 for FlashInfer
        k16 = torch.empty(kk.shape[0], L, attn.head_dim, device=kk.device, dtype=torch.bfloat16)
        v16 = torch.empty_like(k16)
        for h in range(kk.shape[0]):
            ops().kv4_to_bf16(kk[h], k16[h])
            ops().kv4_to_bf16(vv[h], v16[h])
        kk, vv = k16, v16
    elif kk.dtype == torch.float8_e4m3fn:
        # FlashInfer's FP8-KV prefill runs ~48 TFLOPS vs ~80 for BF16 on sm_121: casting the cached prefix
        # to a BF16 scratch first costs a few ms per layer at 128k and saves tens (exact: e4m3 -> bf16 is lossless)
        kk, vv = kk.to(torch.bfloat16), vv.to(torch.bfloat16)
    o = flashinfer.single_prefill_with_kv_cache(q[0].transpose(0, 1), kk, vv, causal=True, kv_layout="HND",
                                                sm_scale=attn.head_dim ** -0.5)
    o8 = torch.empty(T, attn.num_heads * attn.head_dim, dtype=torch.float8_e4m3fn, device=q8.device)
    ops().gate_fp8(o.reshape(T, -1), qp, attn.head_dim, attn.o_proj.in_scale, o8)
    return _fp8_gemm(o8, attn.o_proj)


def _gdn(g, n, q8, state: FastState, li: int):
    """Mixer output (no residual) of a Gated DeltaNet layer. n: bf16 normed input (for the b/a projection)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    T = n.shape[0]
    mixed, z = _split(_fp8_gemm(q8, g.qkvz), g.qkvz.sizes)
    ba = n @ g.w_ba.t()
    b, a = ba[:, : g.num_v_heads], ba[:, g.num_v_heads:]
    K = g.conv_kernel_size
    dev = n.device
    q = torch.empty(T, g.key_dim, device=dev, dtype=torch.bfloat16)
    k = torch.empty(T, g.key_dim, device=dev, dtype=torch.bfloat16)
    v = torch.empty(T, g.value_dim, device=dev, dtype=torch.bfloat16)
    cs = state.conv[li][0]  # [C, K-1]
    ops().causal_conv_silu(mixed, cs, g.conv1d.weight, [q, k, v])
    if T >= K - 1:
        cs.copy_(mixed[-(K - 1):].t())
    else:
        cs.copy_(torch.cat([cs, mixed.t()], dim=-1)[:, -(K - 1):])
    beta = b.sigmoid()[None]
    gg = (-g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias))[None]
    o, s = chunk_gated_delta_rule(q.view(1, T, -1, g.head_k_dim), k.view(1, T, -1, g.head_k_dim), v.view(1, T, -1, g.head_v_dim), g=gg,
                                  beta=beta, initial_state=state.rec[li], output_final_state=True,
                                  use_qk_l2norm_in_kernel=True)  # GVA: 16 key heads, 48 value heads
    state.rec[li].copy_(s)
    on = torch.empty(T * g.num_v_heads, g.head_v_dim, device=dev, dtype=torch.bfloat16)
    ops().gated_rmsnorm(o.reshape(-1, g.head_v_dim), z, g.norm.weight, g.norm.eps, on)
    o8 = torch.empty(T, g.value_dim, dtype=torch.float8_e4m3fn, device=dev)
    ops().fp8_quant(on.view(T, -1), g.out_proj.in_scale, o8)
    return _fp8_gemm(o8, g.out_proj)


def _split(t, sizes):
    return t.split(sizes, dim=-1)


@torch.inference_mode()
def prefill(model: FastQwen35, input_ids: torch.Tensor, state: FastState, chunk: int = CHUNK, all_logits: bool = False,
            return_hidden: bool = False, return_layers: tuple = ()):
    """input_ids [1, T]. Runs the prompt through the model in chunks, advancing `state`; returns the
    fp32 logits of the last token [1, vocab], or with all_logits the bf16 logits of every token
    [T, vocab] (lm_head as a W4A4 GEMM; for perplexity). return_hidden: also return the post-final-norm
    hidden state of every prompt position [T, H] (the MTP drafter's input). return_layers: also return the residual stream
    after each of these layers, [T, len(return_layers), H] (EAGLE-3-style drafter features; return_hidden required)."""
    assert input_ids.shape[0] == 1, "one slot at a time"
    prepare_prefill(model)
    T_all = input_ids.shape[1]
    if state.pos + T_all > state.max_seq_len:
        raise ValueError("prompt exceeds the slot's max_seq_len")
    x_last, outs, hid, lay = None, [], [], []
    for c0 in range(0, T_all, chunk):
        ids = input_ids[0, c0:c0 + chunk]
        T = ids.numel()
        x = model.embed_tokens(ids)  # [T, H] bf16
        for li, layer in enumerate(model.layers):
            gdn = layer.block_type == "linear_attention"
            n, q8 = _norm_in(layer, x, need_bf16=gdn)
            y = _gdn(layer.linear_attn, n, q8, state, li) if gdn else _attention(layer.self_attn, q8, state, li)
            x = _mlp(layer, x, y)
            if li in return_layers:
                lay.append((li, x.clone()))
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
    if return_layers:  # chunk-major list -> [T, n_layers, H] in return_layers order
        per = {li: torch.cat([t for l, t in lay if l == li]) for li in return_layers}
        hid = [torch.cat(hid)]
        L = torch.stack([per[li] for li in return_layers], 1)
    if all_logits:
        if return_layers:
            return torch.cat(outs), hid[0], L
        return (torch.cat(outs), torch.cat(hid)) if return_hidden else torch.cat(outs)
    logits = model.lm_head(model.norm(x_last)).float()
    if return_layers:
        return logits, hid[0], L
    return (logits, torch.cat(hid)) if return_hidden else logits
