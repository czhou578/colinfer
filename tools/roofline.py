#!/usr/bin/env python3
"""Roofline calculator for single-user decode/prefill on DGX Spark (GB10).

Dependency-free. Computes per-token weight bytes, KV/state bytes, FLOPs, and
the resulting decode / prefill ceilings for the two target models.

Usage:
  python3 tools/roofline.py                 # default bandwidth assumptions
  python3 tools/roofline.py --bw 220 --tflops 120 --ctx 8192 32768 131072
"""
import argparse

GB = 1e9

# bytes per weight element for each storage format (incl. scale overhead)
FMT = {
    "bf16":   2.0,
    "fp8":    1.0 + 1.0 / 128,            # e4m3 + per-128 fp32-ish block scale (negligible)
    "nvfp4":  0.5 + 1.0 / 16,             # e2m1 + e4m3 scale per 16  = 0.5625
    "mxfp4":  0.5 + 1.0 / 32,             # e2m1 + e8m0 scale per 32  = 0.53125
    "int4g128": 0.5 + 2.5 / 128,          # 4-bit + fp16 scale + 4-bit zero per 128 group
    "exl3_3bpw": 3.0 / 8,                 # trellis-coded 3.0 bpw experts (what is served today)
}


def qwen38_27b():
    H, I = 5120, 17408
    n_full, n_gdn = 16, 48
    nh, nkv, hd = 24, 4, 256
    k_heads, v_heads, kd, vd = 16, 48, 128, 128
    V = 248320
    mlp = 3 * H * I
    attn = H * (nh * hd * 2) + 2 * H * (nkv * hd) + (nh * hd) * H  # gated q, k, v, o
    gdn = H * (2 * k_heads * kd + 2 * v_heads * vd) + H * 2 * v_heads \
        + (2 * k_heads * kd + v_heads * vd) * 4 + (v_heads * vd) * H
    backbone = (n_full + n_gdn) * mlp + n_full * attn + n_gdn * gdn
    lm_head = V * H
    embed = V * H
    mtp = mlp + attn + 2 * H * H
    comps = dict(mlp=(n_full + n_gdn) * mlp, full_attn=n_full * attn, gdn=n_gdn * gdn,
                 lm_head=lm_head, embed=embed, mtp=mtp)
    kv_per_tok_elems = n_full * 2 * nkv * hd                 # K and V, all full-attn layers
    gdn_state_bytes = n_gdn * v_heads * kd * vd * 4           # fp32 recurrent state
    flops_lin = 2 * backbone
    attn_flops_per_tok_per_ctx = 4 * nh * hd * n_full / 2     # causal avg, per context token
    return dict(name="Qwen3.8-27B (text only)", comps=comps, backbone=backbone,
                kv_elems=kv_per_tok_elems, state_bytes=gdn_state_bytes,
                flops_lin=flops_lin, attn_flops=attn_flops_per_tok_per_ctx)


def deepseek_v4_flash(n_experts=256):
    H, L = 4096, 43
    I_e = 2048
    nh, hd = 64, 512
    q_lora, o_lora, o_groups = 1024, 1024, 8
    idx_h, idx_d = 64, 128
    V = 129280
    expert = 3 * H * I_e
    active_experts = 6 + 1
    attn = H * q_lora + q_lora * nh * hd + H * hd \
        + (nh * hd) * (o_lora * o_groups) // o_groups + (o_lora * o_groups) * H \
        + q_lora * idx_h * idx_d + H * idx_d + 3 * H * 512  # rough: compressors/gates
    router = H * n_experts
    comps = dict(experts_total=L * n_experts * expert, experts_active=L * active_experts * expert,
                 attn=L * attn, router=L * router, lm_head=V * H, embed=V * H)
    kv_per_tok_elems = L * hd  # one shared 512-dim latent per token per layer (pre-compression)
    flops_lin = 2 * (comps["experts_active"] + comps["attn"] + comps["router"])
    attn_flops_per_tok_per_ctx = 4 * nh * hd * L / 2 / 4  # /4: CSA compression, very rough
    return dict(name=f"DeepSeek-V4-Flash ({n_experts} experts)", comps=comps,
                backbone=None, kv_elems=kv_per_tok_elems, state_bytes=0,
                flops_lin=flops_lin, attn_flops=attn_flops_per_tok_per_ctx)


def report(m, bw_peak, bw_meas, tflops, ctxs):
    print("=" * 78)
    print(m["name"])
    print("-" * 78)
    tot = 0
    for k, v in m["comps"].items():
        print(f"  {k:16s} {v/1e9:7.2f} B params")
        if k not in ("embed", "experts_total"):
            tot += v
    print(f"  per-token active (excl. embed) {tot/1e9:7.2f} B params")
    print()
    print("Decode: weight bytes read per token (batch 1, no spec decode)")
    hdr = f"  {'format':12s} {'weights GB':>11s}"
    for c in ctxs:
        hdr += f"  {'+kv@'+str(c//1024)+'k':>10s}"
    hdr += f"  {'tok/s@peak':>10s} {'tok/s@meas':>10s}"
    print(hdr)
    for fmt in ("bf16", "fp8", "nvfp4", "mxfp4", "int4g128", "exl3_3bpw"):
        if m["backbone"] is None:
            # DeepSeek: experts in `fmt`, attention in fp8 vs fp4 shown separately below
            if fmt in ("bf16",):
                continue
            w = m["comps"]["experts_active"] * FMT[fmt] + m["comps"]["attn"] * FMT["fp8"] \
                + m["comps"]["router"] * 2 + m["comps"]["lm_head"] * FMT["fp8"]
            label = f"{fmt}+fp8attn"
        else:
            if fmt == "exl3_3bpw":
                continue
            lm = m["comps"]["lm_head"] * (FMT["fp8"] if fmt != "bf16" else 2.0)
            w = m["backbone"] * FMT[fmt] + lm
            label = fmt
        row = f"  {label:12s} {w/GB:11.2f}"
        for c in ctxs:
            kv = m["kv_elems"] * c * 1.0  # fp8 kv
            row += f"  {(w + kv + 2*m['state_bytes'])/GB:10.2f}"
        base = w + m["kv_elems"] * ctxs[0] + 2 * m["state_bytes"]
        row += f"  {bw_peak*GB/base:10.1f} {bw_meas*GB/base:10.1f}"
        print(row)
    if m["backbone"] is None:
        for fmt in ("nvfp4", "exl3_3bpw"):
            w = m["comps"]["experts_active"] * FMT[fmt] + m["comps"]["attn"] * FMT["nvfp4"] \
                + m["comps"]["router"] * 2 + m["comps"]["lm_head"] * FMT["fp8"]
            base = w + m["kv_elems"] * ctxs[0]
            print(f"  {fmt+'+fp4attn':12s} {w/GB:11.2f}" + " " * (12 * len(ctxs)) +
                  f"  {bw_peak*GB/base:10.1f} {bw_meas*GB/base:10.1f}")
    print()
    print(f"  KV cache (fp8): {m['kv_elems']/1024:.1f} KB/token  "
          f"-> {m['kv_elems']*131072/GB:.2f} GB at 128k")
    if m["state_bytes"]:
        print(f"  Recurrent state (fp32): {m['state_bytes']/1e6:.0f} MB (read+write each step)")
    print()
    print("Prefill: FLOPs per token and ceiling at sustained tensor-core throughput")
    print(f"  linear layers: {m['flops_lin']/1e9:6.1f} GFLOP/token")
    for c in ctxs:
        a = m["attn_flops"] * c
        f = m["flops_lin"] + a
        print(f"  ctx {c//1024:4d}k: +attn {a/1e9:6.1f} GFLOP -> {f/1e9:6.1f} GFLOP/token "
              f"-> {tflops*1e12/f:7.0f} tok/s @ {tflops} TFLOPS sustained")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw-peak", type=float, default=273.0, help="spec LPDDR5x GB/s")
    ap.add_argument("--bw", type=float, default=230.0, help="measured achievable GB/s")
    ap.add_argument("--tflops", type=float, default=100.0, help="sustained GEMM TFLOPS")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 131072])
    a = ap.parse_args()
    for m in (qwen38_27b(), deepseek_v4_flash(256), deepseek_v4_flash(216)):
        report(m, a.bw_peak, a.bw, a.tflops, a.ctx)
