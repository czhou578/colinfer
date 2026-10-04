// attn_decode.cu -- split-KV flash-decoding for the 16 gated full-attention layers (PLAN.md 4.3 item 4).
//
// q        [B, Hq, T, D]  bf16, already q_norm'ed and RoPE'd, T query rows per slot (decode: 1)
// k/v      [B, Hkv, Lmax, D] bf16 cache, the new tokens' rows already written
// seq_lens [B] int32 on the device: valid positions after this step (pos + T). Read on the GPU,
//          so the launch shape is fixed and the kernel can live inside a CUDA graph.
// out      [B, Hq, T, D] bf16
// Row t of slot b attends to positions [0, seq_len - (T - 1 - t)) (causal among the new rows).
//
// Grid (B*T, Hkv, splits). A block owns one KV head and one contiguous slice of positions; its 8
// warps take keys round-robin, each warp keeping an online-softmax state for the G = Hq/Hkv query
// heads that share the KV head (GQA: every K/V row is read once for all G heads). Lane l holds
// head dims [8l, 8l+8). Warps are merged in shared memory, splits by a second small kernel.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace attn {

constexpr int D = 256, WARPS = 8, UNROLL = 2;

__device__ __forceinline__ void bf16x8_to_float(const uint4 v, float (&f)[8]) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(u[i] << 16);
        f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

// KV element types: bf16 (16-byte loads per lane) or fp8 e4m3 (8-byte loads per lane), 8 dims per lane.
struct KvBf16 {
    using Elem = __nv_bfloat16;
    using Vec = uint4;
    static __device__ __forceinline__ void store(Elem* p, float v) { *p = __float2bfloat16(v); }
    static __device__ __forceinline__ void to_float(const Vec v, float (&f)[8]) { bf16x8_to_float(v, f); }
};
struct KvFp8 {
    using Elem = uint8_t;
    using Vec = uint2;
    static __device__ __forceinline__ void store(Elem* p, float v) {
        *p = (Elem)__nv_cvt_float_to_fp8(fminf(fmaxf(v, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
    }
    static __device__ __forceinline__ void to_float(const Vec v, float (&f)[8]) {
        const uint32_t u[2] = {v.x, v.y};
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((u[i] >> (16 * j)) & 0xffff), __NV_E4M3);
                const float2 ff = __half22float2(*reinterpret_cast<__half2*>(&h));
                f[4 * i + 2 * j] = ff.x;
                f[4 * i + 2 * j + 1] = ff.y;
            }
    }
};

template <int G, typename KV>
__global__ void __launch_bounds__(WARPS * 32) k_split(const __nv_bfloat16* __restrict__ q, const typename KV::Elem* __restrict__ kc,
                                                       const typename KV::Elem* __restrict__ vc, const int* __restrict__ seq_lens,
                                                       float* __restrict__ part_acc, float2* __restrict__ part_ml, int Hkv, int T,
                                                       int Lmax, int splits, float scale) {
    const int b = blockIdx.x / T, t = blockIdx.x % T, kvh = blockIdx.y, s = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, Hq = Hkv * G;
    const int len = seq_lens[b] - (T - 1 - t);
    const int chunk = (len + splits - 1) / splits;
    const int start = s * chunk, end = min(len, start + chunk);

    float qr[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const uint4 v = *reinterpret_cast<const uint4*>(q + ((size_t)(b * Hq + kvh * G + g) * T + t) * D + lane * 8);
        bf16x8_to_float(v, qr[g]);
#pragma unroll
        for (int i = 0; i < 8; ++i) qr[g][i] *= scale;
    }
    float m[G], l[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -CUDART_INF_F;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[g][i] = 0.f;
    }
    const size_t base = (size_t)(b * Hkv + kvh) * Lmax * D + lane * 8;
    for (int j0 = start + warp; j0 < end; j0 += WARPS * UNROLL) {
        typename KV::Vec kr[UNROLL], vr[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int j = j0 + u * WARPS;
            if (j < end) {
                kr[u] = __ldcs(reinterpret_cast<const typename KV::Vec*>(kc + base + (size_t)j * D));
                vr[u] = __ldcs(reinterpret_cast<const typename KV::Vec*>(vc + base + (size_t)j * D));
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (j0 + u * WARPS >= end) break;
            float kf[8], vf[8];
            KV::to_float(kr[u], kf);
            KV::to_float(vr[u], vf);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float sc = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) sc = fmaf(qr[g][i], kf[i], sc);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
                const float mn = fmaxf(m[g], sc), corr = __expf(m[g] - mn), p = __expf(sc - mn);
                l[g] = l[g] * corr + p;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[g][i] = fmaf(acc[g][i], corr, p * vf[i]);
                m[g] = mn;
            }
        }
    }
    // merge the 8 warps of this block
    __shared__ float sm_ml[WARPS][G][2];
    __shared__ float sm_acc[WARPS][G][D];
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) { sm_ml[warp][g][0] = m[g]; sm_ml[warp][g][1] = l[g]; }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i) sm_acc[warp][g][lane * 8 + i] = acc[g][i];
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D; idx += WARPS * 32) {
        const int g = idx / D, d = idx % D;
        float M = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm_ml[w][g][0]);
        float L = 0.f, A = 0.f;
        if (M > -CUDART_INF_F) {
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                const float c = __expf(sm_ml[w][g][0] - M);
                L += sm_ml[w][g][1] * c;
                A += sm_acc[w][g][d] * c;
            }
        }
        const size_t row = ((size_t)(b * Hq + kvh * G + g) * T + t) * splits + s;
        part_acc[row * D + d] = A;
        if (d == 0) part_ml[row] = make_float2(M, L);
    }
}

// gate == nullptr: out[b, h, t, :] (the q layout). gate != nullptr (the q_proj output [B, T, Hq, 2D],
// gate in the second half of each head): out[b, t, h, :] = attn * sigmoid(gate), ready for o_proj.
__global__ void k_combine(const float* __restrict__ part_acc, const float2* __restrict__ part_ml, __nv_bfloat16* __restrict__ out, int splits,
                          const __nv_bfloat16* __restrict__ gate, int Hq, int T) {
    const size_t row = blockIdx.x;  // (b, h, t)
    const int d = threadIdx.x;
    float M = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[row * splits + s].x);
    float L = 0.f, A = 0.f;
    if (M > -CUDART_INF_F)
        for (int s = 0; s < splits; ++s) {
            const float2 ml = part_ml[row * splits + s];
            const float c = __expf(ml.x - M);
            L += ml.y * c;
            A += part_acc[(row * splits + s) * D + d] * c;
        }
    const float val = L > 0.f ? A / L : 0.f;
    if (gate == nullptr) {
        out[row * D + d] = __float2bfloat16(val);
    } else {
        const int t = row % T, h = (row / T) % Hq, b = row / ((size_t)T * Hq);
        const size_t o = (((size_t)b * T + t) * Hq + h) * D + d;
        const float g = __bfloat162float(gate[(((size_t)b * T + t) * Hq + h) * 2 * D + D + d]);
        out[o] = __float2bfloat16(val / (1.f + __expf(-g)));
    }
}

// Attention prologue for T new rows per slot, one block (D threads) per (b, t, head) over Hq q-heads,
// then Hkv k-heads, then Hkv v-heads:
//   q:  zero-centered RMSNorm (q_norm) + partial RoPE on the first R dims  -> q_out [B, Hq, T, D]
//   k:  RMSNorm (k_norm) + RoPE, written to k_cache[b, h, pos_t[b] + t]   (bf16 or saturating e4m3)
//   v:  written to v_cache[b, h, pos_t[b] + t]
// qp: q_proj output [B, T, Hq, 2D] (q in the first D of each head); kp, vp: [B, T, Hkv, D].
template <typename KV>
__global__ void k_prologue(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp, const __nv_bfloat16* __restrict__ vp,
                           const __nv_bfloat16* __restrict__ qn_w, const __nv_bfloat16* __restrict__ kn_w, const float* __restrict__ inv_freq,
                           const int* __restrict__ pos_t, typename KV::Elem* __restrict__ kc, typename KV::Elem* __restrict__ vc,
                           __nv_bfloat16* __restrict__ q_out, int Hq, int Hkv, int T, int Lmax, int R, float eps) {
    const int bt = blockIdx.x, b = bt / T, t = bt % T, hh = blockIdx.y, d = threadIdx.x;
    const int pos = pos_t[b] + t;
    __shared__ float xs[D], red[D / 32];
    float x;
    const __nv_bfloat16* w;
    if (hh < Hq) { x = __bfloat162float(qp[(((size_t)b * T + t) * Hq + hh) * 2 * D + d]); w = qn_w; }
    else if (hh < Hq + Hkv) { x = __bfloat162float(kp[(((size_t)b * T + t) * Hkv + (hh - Hq)) * D + d]); w = kn_w; }
    else {
        const int h = hh - Hq - Hkv;
        KV::store(vc + (((size_t)b * Hkv + h) * Lmax + pos) * D + d, __bfloat162float(vp[(((size_t)b * T + t) * Hkv + h) * D + d]));
        return;
    }
    float ss = x * x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < D / 32; ++i) tot += red[i];
    const float xn = __bfloat162float(__float2bfloat16(x * rsqrtf(tot / D + eps) * (1.f + __bfloat162float(w[d]))));
    xs[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < R) {
        const int half = R / 2, i = d % half;
        const float ang = (float)pos * inv_freq[i];
        float sn, cs;
        sincosf(ang, &sn, &cs);
        y = d < half ? xn * cs - xs[d + half] * sn : xn * cs + xs[d - half] * sn;
    }
    if (hh < Hq) q_out[(((size_t)b * Hq + hh) * T + t) * D + d] = __float2bfloat16(y);
    else KV::store(kc + (((size_t)b * Hkv + (hh - Hq)) * Lmax + pos) * D + d, __bfloat162float(__float2bfloat16(y)));
}

}  // namespace attn

cudaError_t launch_attn_decode(const void* q, const void* kc, const void* vc, const int* seq_lens, void* out, float* part_acc,
                               void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, int Dh, int splits, float scale, bool kv_fp8,
                               const void* gate, cudaStream_t st) {
    if (Dh != attn::D || Hq % Hkv) return cudaErrorInvalidValue;
    const int G = Hq / Hkv;
    dim3 grid(B * T, Hkv, splits), block(attn::WARPS * 32);
    auto qb = (const __nv_bfloat16*)q;
#define ATTN_LAUNCH(GG, KVT) \
    attn::k_split<GG, attn::KVT><<<grid, block, 0, st>>>(qb, (const attn::KVT::Elem*)kc, (const attn::KVT::Elem*)vc, seq_lens, part_acc, \
                                                         (float2*)part_ml, Hkv, T, Lmax, splits, scale)
    if (G == 6 && kv_fp8) ATTN_LAUNCH(6, KvFp8);
    else if (G == 6) ATTN_LAUNCH(6, KvBf16);
    else if (G == 4 && kv_fp8) ATTN_LAUNCH(4, KvFp8);
    else if (G == 4) ATTN_LAUNCH(4, KvBf16);
    else return cudaErrorInvalidValue;
#undef ATTN_LAUNCH
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, splits,
                                                    (const __nv_bfloat16*)gate, Hq, T);
    return cudaGetLastError();
}

cudaError_t launch_attn_prologue(const void* qp, const void* kp, const void* vp, const void* qn_w, const void* kn_w, const float* inv_freq,
                                 const int* pos_t, void* kc, void* vc, void* q_out, int B, int T, int Hq, int Hkv, int Lmax, int R, float eps,
                                 bool kv_fp8, cudaStream_t st) {
    dim3 grid(B * T, Hq + 2 * Hkv);
    if (kv_fp8)
        attn::k_prologue<attn::KvFp8><<<grid, attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp,
                                                                (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq, pos_t,
                                                                (uint8_t*)kc, (uint8_t*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R, eps);
    else
        attn::k_prologue<attn::KvBf16><<<grid, attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp,
                                                                 (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq, pos_t,
                                                                 (__nv_bfloat16*)kc, (__nv_bfloat16*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R, eps);
    return cudaGetLastError();
}
