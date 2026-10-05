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
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include "pdl.cuh"

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

// KV cache formats. A cache row is one (token, head): ROW elements. Lane l of a warp handles head dims [8l, 8l+8).
//   bf16:  256 x bf16 (16-byte loads per lane)
//   fp8:   256 x e4m3, unit scale, saturating (8-byte loads per lane)
//   fp4:   144 bytes: 128 bytes of e2m1 (dim 2i in the low nibble of byte i), then 16 e4m3 block scales (dims 16j..16j+15),
//          each stored as 16 * amax / 6 so block maxima from ~0.006 to 168 stay in e4m3's normal range (K: 0.04-23,
//          V: 0.17-108 measured on this model). 0.56x the bytes of fp8.
// store_row(row, d, v): called by all 256 threads of a block, thread d holding dim d (fp4 needs the whole 16-dim block).
struct KvBf16 {
    using Elem = __nv_bfloat16;
    using Vec = uint4;
    static constexpr int ROW = D;
    static __device__ __forceinline__ Vec load(const Elem* row, int lane) { return __ldcs(reinterpret_cast<const Vec*>(row + lane * 8)); }
    static __device__ __forceinline__ void store_row(Elem* row, int d, float v) { row[d] = __float2bfloat16(v); }
    static __device__ __forceinline__ void to_float(const Vec v, float (&f)[8]) { bf16x8_to_float(v, f); }
};
struct KvFp8 {
    using Elem = uint8_t;
    using Vec = uint2;
    static constexpr int ROW = D;
    static __device__ __forceinline__ Vec load(const Elem* row, int lane) { return __ldcs(reinterpret_cast<const Vec*>(row + lane * 8)); }
    static __device__ __forceinline__ void store_row(Elem* row, int d, float v) {
        row[d] = (Elem)__nv_cvt_float_to_fp8(fminf(fmaxf(v, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
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

struct KvFp4 {
    using Elem = uint8_t;
    struct Vec { uint32_t d, s; };
    static constexpr int ROW = D / 2 + D / 16;
    static constexpr float SCALE_BIAS = 16.f;
    static __device__ __forceinline__ Vec load(const Elem* row, int lane) {
        Vec v;
        v.d = __ldcs(reinterpret_cast<const uint32_t*>(row) + lane);
        v.s = __ldcs(row + D / 2 + (lane >> 1));
        return v;
    }
    static __device__ __forceinline__ float scale_of(uint32_t code) {
        __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)code, __NV_E4M3);
        return __half2float(*reinterpret_cast<__half*>(&h)) * (1.f / SCALE_BIAS);
    }
    static __device__ __forceinline__ void to_float(const Vec v, float (&f)[8]) {
        const float sc = scale_of(v.s);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)((v.d >> (8 * i)) & 0xff), __NV_E2M1);
            const float2 ff = __half22float2(*reinterpret_cast<__half2*>(&h));
            f[2 * i] = ff.x * sc;
            f[2 * i + 1] = ff.y * sc;
        }
    }
    static __device__ __forceinline__ void store_row(Elem* row, int d, float v) {
        float amax = fabsf(v);  // the 16 dims of a block are 16 consecutive lanes of one warp
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        // scale amax / 6: no clipping. (Choosing amax / 5 when it lowers the block's squared error, as for weights,
        // made perplexity worse: clipping a block maximum hurts attention scores more than coarser steps.)
        const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(amax / 6.f * SCALE_BIAS, __NV_SATFINITE, __NV_E4M3);
        const float sc = scale_of(code);
        const uint32_t nib = sc > 0.f ? (uint32_t)__nv_cvt_float_to_fp4(v / sc, __NV_E2M1, cudaRoundNearest) & 0xf : 0u;
        const uint32_t other = __shfl_xor_sync(0xffffffffu, nib, 1);
        if (!(d & 1)) row[d >> 1] = (Elem)(nib | (other << 4));
        if (!(d & 15)) row[D / 2 + (d >> 4)] = (Elem)code;
    }
};

template <int G, typename KV>
__global__ void __launch_bounds__(WARPS * 32) k_split(const __nv_bfloat16* __restrict__ q, const typename KV::Elem* __restrict__ kc,
                                                       const typename KV::Elem* __restrict__ vc, const int* __restrict__ seq_lens,
                                                       float* __restrict__ part_acc, float2* __restrict__ part_ml, int Hkv, int T,
                                                       int Lmax, int splits, float scale) {
    PDL_TRIGGER();
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
    const size_t base = (size_t)(b * Hkv + kvh) * Lmax * KV::ROW;
    for (int j0 = start + warp; j0 < end; j0 += WARPS * UNROLL) {
        typename KV::Vec kr[UNROLL], vr[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int j = j0 + u * WARPS;
            if (j < end) {
                kr[u] = KV::load(kc + base + (size_t)j * KV::ROW, lane);
                vr[u] = KV::load(vc + base + (size_t)j * KV::ROW, lane);
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
    PDL_TRIGGER();
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
__global__ void k_prologue(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp, const __nv_bfloat16* __restrict__ vp, int ldq, int ldk, int ldv,
                           const __nv_bfloat16* __restrict__ qn_w, const __nv_bfloat16* __restrict__ kn_w, const float* __restrict__ inv_freq,
                           const int* __restrict__ pos_t, typename KV::Elem* __restrict__ kc, typename KV::Elem* __restrict__ vc,
                           __nv_bfloat16* __restrict__ q_out, int Hq, int Hkv, int T, int Lmax, int R, float eps, const int* __restrict__ active) {
    PDL_TRIGGER();
    const int bt = blockIdx.x, b = bt / T, t = bt % T, hh = blockIdx.y, d = threadIdx.x;
    const bool upd = active == nullptr || active[b] != 0;  // inactive slots: no KV write
    const int pos = pos_t[b] + t;
    __shared__ float xs[D], red[D / 32];
    float x;
    const __nv_bfloat16* w;
    // token row (b, t) of q / k / v starts at (b * T + t) * ld{q,k,v} (row strides of the projection outputs)
    const size_t row = (size_t)b * T + t;
    if (hh < Hq) { x = __bfloat162float(qp[row * ldq + (size_t)hh * 2 * D + d]); w = qn_w; }
    else if (hh < Hq + Hkv) { x = __bfloat162float(kp[row * ldk + (size_t)(hh - Hq) * D + d]); w = kn_w; }
    else {
        const int h = hh - Hq - Hkv;
        if (upd) KV::store_row(vc + (((size_t)b * Hkv + h) * Lmax + pos) * KV::ROW, d, __bfloat162float(vp[row * ldv + (size_t)h * D + d]));
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
    else if (upd) KV::store_row(kc + (((size_t)b * Hkv + (hh - Hq)) * Lmax + pos) * KV::ROW, d, __bfloat162float(__float2bfloat16(y)));
}

// fp4 cache rows -> bf16 rows (the prefill path hands FlashInfer a bf16 copy of the cached prefix)
__global__ void k_kv4_to_bf16(const uint8_t* __restrict__ src, __nv_bfloat16* __restrict__ dst, long rows) {
    const long r = blockIdx.x * (long)(blockDim.x / 32) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    float f[8];
    KvFp4::to_float(KvFp4::load(src + r * KvFp4::ROW, lane), f);
    uint4 o;
    uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
        ou[i] = *reinterpret_cast<const uint32_t*>(&b2);
    }
    *reinterpret_cast<uint4*>(dst + r * D + lane * 8) = o;
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor-core multi-row decode (fp8 / fp4 caches). k_split runs one block per query row, so a speculative verify of
// T rows reads the KV cache T times (a k=3 cycle at 128k: 144 ms against 89 ms at 8k). Here one block owns one
// (slot, KV head, tile stripe) and serves every query row of the slot: up to 48 rows (G heads x T rows) as three
// m16 tiles of mma.sync.m16n8k16 (f16 x f16 -> fp32). Rows beyond 48 take further blocks (grid y).
//
// Bit identity. A row's result must not depend on how many other rows share the launch, so plain decode (T = 1)
// and every verify width produce the same bits:
//   - keys are processed in TK-key tiles at fixed positions; block z owns tiles z, z + NB, z + 2 NB, ... (NB fixed),
//     and the combine folds the NB partials in a fixed order;
//   - mma rows are independent; the softmax of a row is 8 lanes with fixed reduction trees, the same for every row;
//   - a tile entirely past a row's length (it exists because a longer row of the slot needs it) is an exact no-op:
//     its keys score -inf, p = 0 exactly, the running max does not move and corr is set to exactly 1.
//
// Operands. K stays in the raw cache bytes in shared memory: a lane's B fragment for k-step j is 4 consecutive head
// dims (16j + 4c .. +3, c = lane % 4), which the QK product may take in any order as long as Q's A fragment uses the
// same order (it does: 8 consecutive bytes of the f16 Q row). e4m3 and e2m1 x e4m3-scale values are exact in f16.
// V is converted to an f16 [key][dim] tile and read with ldmatrix.trans; P is rounded to f16 (~5e-4 relative error
// against fp32 attention, below the bf16 output's own rounding).
//
// Speed (bench/attn_bench.py, 128k context, fp8 KV): 228-236 GB/s for T = 1, 4 and 8, i.e. a verify of 8 rows costs
// what plain decode costs. NB = 12: one slot's 4 KV heads x 12 stripes fill the 48 SMs once (one block per SM).
namespace tc {

constexpr int TK = 32, NB = 12, WARPS = 4, QS = D + 8, VS = D + 8, PS = TK + 8, SS = TK + 4, MAXROWS = 48;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}
__device__ __forceinline__ uint32_t e2m1x2_h2(uint32_t byte, __half2 sc) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(byte & 0xff), __NV_E2M1);
    __half2 v = __hmul2(*reinterpret_cast<__half2*>(&h), sc);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ __half2 fp4_scale(uint32_t code) {  // e4m3 code / 16, exact in f16
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)code, __NV_E4M3);
    return __half2half2(__hmul(*reinterpret_cast<__half*>(&h), __float2half(1.f / KvFp4::SCALE_BIAS)));
}

// RS: shared-memory bytes per raw cache row. swz(row, chunk): where a row's 16-byte chunk lands. fp8 rows stay 256 bytes
// apart (padding them to 272 cut cp.async streaming from ~232 to ~187 GB/s) and XOR-swizzle chunks by row instead, so
// the 8 rows of a K fragment load hit 8 different bank groups.
struct TcFp8 {
    static constexpr int ROW = D, RS = D, CH = ROW / 16;
    static __device__ __forceinline__ int swz(int r, int ch) { return ch ^ (r & 7); }
    static __device__ __forceinline__ void kfrag(const uint8_t* row, int r, int j, int c, uint32_t& b0, uint32_t& b1) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 16 * swz(r, j) + 4 * c);
        b0 = e4m3x2_h2(w);
        b1 = e4m3x2_h2(w >> 16);
    }
    // 8 dims [8u, 8u+8) of one row -> 8 f16
    static __device__ __forceinline__ uint4 vunit(const uint8_t* row, int r, int u) {
        const uint2 w = *reinterpret_cast<const uint2*>(row + 16 * swz(r, u >> 1) + 8 * (u & 1));
        return make_uint4(e4m3x2_h2(w.x), e4m3x2_h2(w.x >> 16), e4m3x2_h2(w.y), e4m3x2_h2(w.y >> 16));
    }
};
struct TcFp4 {
    static constexpr int ROW = KvFp4::ROW, RS = KvFp4::ROW, CH = ROW / 16;
    static __device__ __forceinline__ int swz(int, int ch) { return ch; }
    static __device__ __forceinline__ void kfrag(const uint8_t* row, int, int j, int c, uint32_t& b0, uint32_t& b1) {
        const uint32_t w = *reinterpret_cast<const uint16_t*>(row + 8 * j + 2 * c);
        const __half2 sc = fp4_scale(row[D / 2 + j]);
        b0 = e2m1x2_h2(w, sc);
        b1 = e2m1x2_h2(w >> 8, sc);
    }
    static __device__ __forceinline__ uint4 vunit(const uint8_t* row, int, int u) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 4 * u);
        const __half2 sc = fp4_scale(row[D / 2 + u / 2]);
        return make_uint4(e2m1x2_h2(w, sc), e2m1x2_h2(w >> 8, sc), e2m1x2_h2(w >> 16, sc), e2m1x2_h2(w >> 24, sc));
    }
};

// raw K / V tile stages: four where shared memory allows (the 99 KB per block limit), else three or two
template <int MT, typename KV>
constexpr int smem_bytes(int stages) {
    return MT * 16 * QS * 2 + stages * 2 * TK * KV::RS + TK * VS * 2 + MT * 16 * SS * 4 + MT * 16 * PS * 2 + 3 * MT * 16 * 4;
}
template <int MT, typename KV>
constexpr int stages() { return smem_bytes<MT, KV>(4) <= 99 * 1024 ? 4 : smem_bytes<MT, KV>(3) <= 99 * 1024 ? 3 : 2; }

// Grid (B, Hkv * passes, NB), 128 threads. Block (b, kvh, pass, z) handles query rows r = pass * 48 + [0, 48) of slot b
// (row r = t * G + g: query head kvh * G + g, new token t) over the tiles z + n * NB. Partials go to
// part_acc / part_ml at ((b * Hq + h) * T + t) * NB + z, the layout k_combine folds (splits = NB).
template <int G, int MT, typename KV>
__global__ void __launch_bounds__(WARPS * 32) k_tc(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                    const uint8_t* __restrict__ vc, const int* __restrict__ seq_lens,
                                                    float* __restrict__ part_acc, float2* __restrict__ part_ml, int Hkv, int T, int Lmax,
                                                    float scale_log2) {
    PDL_TRIGGER();
    constexpr int MP = MT * 16, STAGES = stages<MT, KV>();
    extern __shared__ __align__(16) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    uint8_t* Kraw = smem + MP * QS * 2;
    uint8_t* Vraw = Kraw + STAGES * TK * KV::RS;
    __half* Vh = reinterpret_cast<__half*>(Vraw + STAGES * TK * KV::RS);
    float* Ss = reinterpret_cast<float*>(Vh + TK * VS);
    __half* Ps = reinterpret_cast<__half*>(Ss + MP * SS);
    float* m_s = reinterpret_cast<float*>(Ps + MP * PS);
    float* l_s = m_s + MP;
    float* corr_s = l_s + MP;

    const int b = blockIdx.x, kvh = blockIdx.y % Hkv, pass = blockIdx.y / Hkv, z = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int Hq = Hkv * G, r0 = pass * MAXROWS, R = min(G * T - r0, MP);  // rows of this block: r0 + [0, R)
    const int sl = seq_lens[b];
    const int ntiles = (sl + TK - 1) / TK;

    for (int idx = tid; idx < MP * (D / 8); idx += WARPS * 32) {
        const int r = idx / (D / 8), u = idx % (D / 8);
        uint4 o = make_uint4(0, 0, 0, 0);
        if (r < R) {
            const int rr = r0 + r, t = rr / G, g = rr % G;
            float f[8];
            bf16x8_to_float(*reinterpret_cast<const uint4*>(q + ((size_t)(b * Hq + kvh * G + g) * T + t) * D + u * 8), f);
            uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half2 h = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
                ou[i] = *reinterpret_cast<const uint32_t*>(&h);
            }
        }
        *reinterpret_cast<uint4*>(Qs + r * QS + u * 8) = o;
    }
    for (int idx = tid; idx < MP * PS; idx += WARPS * 32) Ps[idx] = __float2half(0.f);
    for (int r = tid; r < MP; r += WARPS * 32) { m_s[r] = -CUDART_INF_F; l_s[r] = 0.f; corr_s[r] = 1.f; }

    const size_t base = (size_t)(b * Hkv + kvh) * Lmax;
    auto load_tile = [&](int tile, int stage) {
        const int p0 = tile * TK;
        for (int idx = tid; idx < TK * KV::CH; idx += WARPS * 32) {
            const int kr = idx / KV::CH, ch = idx % KV::CH, p = p0 + kr;
            const bool ok = p < sl;  // zero-fill past the slot's length: stale bytes there must not reach the mma as NaN
            const size_t off = (base + (ok ? p : 0)) * KV::ROW + ch * 16;
            cp_async16(Kraw + (stage * TK + kr) * KV::RS + KV::swz(kr, ch) * 16, kc + off, ok);
            cp_async16(Vraw + (stage * TK + kr) * KV::RS + KV::swz(kr, ch) * 16, vc + off, ok);
        }
    };

    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (z + i * NB < ntiles) load_tile(z + i * NB, i);
        cp_async_commit();
    }
    for (int tile = z, it = 0; tile < ntiles; tile += NB, ++it) {
        const int stage = it % STAGES;
        // refill the stage the previous iteration consumed (its readers passed that iteration's second barrier)
        if (tile + (STAGES - 1) * NB < ntiles) load_tile(tile + (STAGES - 1) * NB, (it + STAGES - 1) % STAGES);
        cp_async_commit();
        cp_async_wait<STAGES - 1>();
        __syncthreads();  // raw tile visible; the previous tile's PV is done with Vh / Ps

        // S = Q K^T: warp w scores keys [8w, 8w + 8) of the tile for all rows
        {
            const int kr = warp * 8 + (lane >> 2);
            const uint8_t* krow = Kraw + (stage * TK + kr) * KV::RS;
            float s[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
#pragma unroll
            for (int j = 0; j < D / 16; ++j) {
                uint32_t b0, b1;
                KV::kfrag(krow, kr, j, lane & 3, b0, b1);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const __half* qa = Qs + (mt * 16 + (lane >> 2)) * QS + 16 * j + 4 * (lane & 3);
                    const uint2 x0 = *reinterpret_cast<const uint2*>(qa), x1 = *reinterpret_cast<const uint2*>(qa + 8 * QS);
                    const uint32_t a[4] = {x0.x, x1.x, x0.y, x1.y};
                    mma16816(s[mt], a, b0, b1);
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float* srow = Ss + (mt * 16 + (lane >> 2)) * SS + warp * 8 + 2 * (lane & 3);
                srow[0] = s[mt][0];
                srow[1] = s[mt][1];
                srow[8 * SS] = s[mt][2];
                srow[8 * SS + 1] = s[mt][3];
            }
        }
        for (int idx = tid; idx < TK * (D / 8); idx += WARPS * 32) {
            const int kr = idx / (D / 8), u = idx % (D / 8);
            *reinterpret_cast<uint4*>(Vh + kr * VS + u * 8) = KV::vunit(Vraw + (stage * TK + kr) * KV::RS, kr, u);
        }
        __syncthreads();

        // online softmax: 8 lanes per row (4 keys each), 4 rows per warp at a time
        {
            const int grp = lane >> 3, sub = lane & 7;
            const unsigned gmask = 0xffu << (8 * grp);
            for (int r = warp + WARPS * grp; r < R; r += WARPS * 4) {
                const int len = sl - (T - 1 - (r0 + r) / G), p = tile * TK + 4 * sub;
                const float4 sv = *reinterpret_cast<const float4*>(Ss + r * SS + 4 * sub);
                const float x[4] = {p < len ? sv.x * scale_log2 : -CUDART_INF_F, p + 1 < len ? sv.y * scale_log2 : -CUDART_INF_F,
                                    p + 2 < len ? sv.z * scale_log2 : -CUDART_INF_F, p + 3 < len ? sv.w * scale_log2 : -CUDART_INF_F};
                float tmax = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(gmask, tmax, o));
                const float mo = m_s[r], mn = fmaxf(mo, tmax);
                const float corr = mn == mo ? 1.f : ex2(mo - mn);
                __half ph[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) ph[i] = __float2half_rn(x[i] == -CUDART_INF_F ? 0.f : ex2(x[i] - mn));
                float ps = (__half2float(ph[0]) + __half2float(ph[1])) + (__half2float(ph[2]) + __half2float(ph[3]));
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) ps += __shfl_xor_sync(gmask, ps, o);
                *reinterpret_cast<uint2*>(Ps + r * PS + 4 * sub) = *reinterpret_cast<const uint2*>(ph);
                __syncwarp(gmask);
                if (sub == 0) { m_s[r] = mn; l_s[r] = l_s[r] * corr + ps; corr_s[r] = corr; }
            }
        }
        __syncthreads();

        // O = O * corr + P V: warp w owns head dims [64w, 64w + 64)
        {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float c0 = corr_s[mt * 16 + (lane >> 2)], c1 = corr_s[mt * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mt][nt][0] *= c0;
                acc[mt][nt][1] *= c0;
                acc[mt][nt][2] *= c1;
                acc[mt][nt][3] *= c1;
            }
        }
#pragma unroll
        for (int ks = 0; ks < TK / 16; ++ks) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                ldsm_x4(a[mt], Ps + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * PS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t bv[4];
                ldsm_x4_t(bv, Vh + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + warp * 64 + np * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma16816(acc[mt][2 * np], a[mt], bv[0], bv[1]);
                    mma16816(acc[mt][2 * np + 1], a[mt], bv[2], bv[3]);
                }
            }
        }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    auto prow = [&](int r) {  // partial index of block row r
        const int rr = r0 + r, t = rr / G, g = rr % G;
        return ((size_t)(b * Hq + kvh * G + g) * T + t) * NB + z;
    };
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mt * 16 + (lane >> 2) + 8 * h;
            if (r < R) {
                float* dst = part_acc + prow(r) * D + warp * 64 + 2 * (lane & 3);
#pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    *reinterpret_cast<float2*>(dst + nt * 8) = make_float2(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
            }
        }
    // m in natural-log units for k_combine (which folds with __expf)
    for (int r = tid; r < R; r += WARPS * 32) part_ml[prow(r)] = make_float2(m_s[r] * 0.69314718056f, l_s[r]);
}

}  // namespace tc
}  // namespace attn

int attn_decode_tc_nb() { return attn::tc::NB; }

// Multi-row tensor-core decode attention (fp8 / fp4 caches). part_acc: >= B * Hq * T * attn_decode_tc_nb() * 256 floats,
// part_ml: as many float2. Same contract as launch_attn_decode otherwise.
cudaError_t launch_attn_decode_tc(const void* q, const void* kc, const void* vc, const int* seq_lens, void* out, float* part_acc,
                                  void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, int Dh, float scale, int kv_kind,
                                  const void* gate, cudaStream_t st) {
    using namespace attn::tc;
    if (Dh != attn::D || Hq % Hkv || (kv_kind != 1 && kv_kind != 2)) return cudaErrorInvalidValue;
    const int G = Hq / Hkv, rows = G * T, passes = (rows + MAXROWS - 1) / MAXROWS;
    const int MT = (min(rows, MAXROWS) + 15) / 16;
    dim3 grid(B, Hkv * passes, NB), block(WARPS * 32);
    const float sl2 = scale * 1.4426950408889634f;
    auto qb = (const __nv_bfloat16*)q;
    cudaError_t err = cudaErrorInvalidValue;
#define TC_LAUNCH(GG, MM, KVT)                                                                                                     \
    do {                                                                                                                           \
        constexpr int SMEM = smem_bytes<MM, attn::tc::KVT>(stages<MM, attn::tc::KVT>());                                                                      \
        static bool init = false;                                                                                                  \
        if (!init) {                                                                                                               \
            cudaFuncSetAttribute(k_tc<GG, MM, attn::tc::KVT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                  \
            init = true;                                                                                                           \
        }                                                                                                                          \
        k_tc<GG, MM, attn::tc::KVT><<<grid, block, SMEM, st>>>(qb, (const uint8_t*)kc, (const uint8_t*)vc, seq_lens, part_acc,     \
                                                                (float2*)part_ml, Hkv, T, Lmax, sl2);                              \
        err = cudaSuccess;                                                                                                         \
    } while (0)
#define TC_MT(GG, KVT)                         \
    switch (MT) {                              \
        case 1: TC_LAUNCH(GG, 1, KVT); break;  \
        case 2: TC_LAUNCH(GG, 2, KVT); break;  \
        case 3: TC_LAUNCH(GG, 3, KVT); break;  \
    }
    if (G == 6 && kv_kind == 1) TC_MT(6, TcFp8)
    else if (G == 6) TC_MT(6, TcFp4)
    else if (G == 4 && kv_kind == 1) TC_MT(4, TcFp8)
    else if (G == 4) TC_MT(4, TcFp4)
#undef TC_MT
#undef TC_LAUNCH
    if (err != cudaSuccess) return err;
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, NB, (const __nv_bfloat16*)gate,
                                                    Hq, T);
    return cudaGetLastError();
}

cudaError_t launch_attn_decode(const void* q, const void* kc, const void* vc, const int* seq_lens, void* out, float* part_acc,
                               void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, int Dh, int splits, float scale, int kv_kind,
                               const void* gate, cudaStream_t st) {
    if (Dh != attn::D || Hq % Hkv) return cudaErrorInvalidValue;
    const int G = Hq / Hkv;
    dim3 grid(B * T, Hkv, splits), block(attn::WARPS * 32);
    auto qb = (const __nv_bfloat16*)q;
#define ATTN_LAUNCH(GG, KVT) \
    attn::k_split<GG, attn::KVT><<<grid, block, 0, st>>>(qb, (const attn::KVT::Elem*)kc, (const attn::KVT::Elem*)vc, seq_lens, part_acc, \
                                                         (float2*)part_ml, Hkv, T, Lmax, splits, scale)
    if (G == 6 && kv_kind == 1) ATTN_LAUNCH(6, KvFp8);
    else if (G == 6 && kv_kind == 2) ATTN_LAUNCH(6, KvFp4);
    else if (G == 6) ATTN_LAUNCH(6, KvBf16);
    else if (G == 4 && kv_kind == 1) ATTN_LAUNCH(4, KvFp8);
    else if (G == 4 && kv_kind == 2) ATTN_LAUNCH(4, KvFp4);
    else if (G == 4) ATTN_LAUNCH(4, KvBf16);
    else return cudaErrorInvalidValue;
#undef ATTN_LAUNCH
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, splits,
                                                    (const __nv_bfloat16*)gate, Hq, T);
    return cudaGetLastError();
}

cudaError_t launch_attn_prologue(const void* qp, const void* kp, const void* vp, int ldq, int ldk, int ldv, const void* qn_w, const void* kn_w, const float* inv_freq,
                                 const int* pos_t, void* kc, void* vc, void* q_out, int B, int T, int Hq, int Hkv, int Lmax, int R, float eps,
                                 int kv_kind, const int* active, cudaStream_t st) {
    dim3 grid(B * T, Hq + 2 * Hkv);
    if (kv_kind == 2)
        attn::k_prologue<attn::KvFp4><<<grid, attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp, ldq, ldk, ldv,
                                                                (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq, pos_t,
                                                                (uint8_t*)kc, (uint8_t*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R, eps, active);
    else if (kv_kind == 1)
        attn::k_prologue<attn::KvFp8><<<grid, attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp, ldq, ldk, ldv,
                                                                (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq, pos_t,
                                                                (uint8_t*)kc, (uint8_t*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R, eps, active);
    else
        attn::k_prologue<attn::KvBf16><<<grid, attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp, ldq, ldk, ldv,
                                                                 (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq, pos_t,
                                                                 (__nv_bfloat16*)kc, (__nv_bfloat16*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R, eps, active);
    return cudaGetLastError();
}

cudaError_t launch_kv4_to_bf16(const void* src, void* dst, long rows, cudaStream_t st) {
    attn::k_kv4_to_bf16<<<(rows + 7) / 8, 256, 0, st>>>((const uint8_t*)src, (__nv_bfloat16*)dst, rows);
    return cudaGetLastError();
}
