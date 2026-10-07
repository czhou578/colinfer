// attn_prefill.cu -- causal prefill attention over an FP8 KV cache, with Q K^T on FP8 tensor cores (PLAN.md 4.4 item 3,
// docs/history/phase6_progress.md section 8).
//
// The FA2 prefill (bf16) of FlashInfer runs at ~79 TFLOPS on this chip, 81% of the BF16 GEMM peak. At 32k context, it
// takes a third of the time of a chunk. Here S = Q K^T uses mma.sync.m16n8k32 e4m3, at twice the BF16 rate. K is the own
// e4m3 bytes of the cache (unit scale). The kernel rounds Q to e4m3, with one scale per (token, head) (amax / 448). The
// perplexity with this rounding emulated: no change at ctx 2048, +0.12% on code at ctx 8192. P V stays f16 x f16 -> fp32
// (P rounded to f16, V converted exactly from e4m3).
//
// q    [Hq, T, D] bf16 (the layout of the prologue), rows t = 0..T-1 at positions pos + t
// k, v [Hkv, Lmax, D] e4m3, one slot, with the new rows already written. Query t attends keys 0 .. pos + t.
// out  [T, Hq, D] bf16
//
// Grid (ceil(T / 64), Hq), 4 warps. Warp w owns the query rows 16w .. 16w + 15 of the 64 rows of the block. The blocks
// run the heaviest (last) query tiles first. The 6 query heads of a KV head are adjacent in y, so they share their KV
// stream in L2.
//
// Q K^T fragment order. A lane (g = lane / 4, c = lane % 4) reads 16 consecutive bytes of K row g: head dims
// 64j + 16c .. +15. It uses them as the B fragments of two k32 steps (bytes 0-7 for step 2j, 8-15 for 2j+1). The sum over
// the head dims does not depend on the mma k slot of each dim, if the A fragment of Q uses the same assignment. It does:
// the lane takes its own 16 Q bytes of the same dims. The K rows are 256 bytes apart in shared memory, and the
// 16-byte chunks are XOR'ed by bit 0 of the row. Thus the 8 lanes of a quarter-warp (rows 2r, 2r+1) hit 8 different bank
// groups.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace apf {

constexpr int D = 256, WARPS = 4, BM = 16 * WARPS, VS = D + 8;
constexpr int BN = 32;  // keys per KV tile: 32 (two blocks per SM) beat 64 (102-106 vs 86-87 TFLOPS)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ void mma_fp8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t to_e4m3x4(const float* f) {  // 4 floats -> 4 e4m3 bytes (RN, saturating)
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(f[0], f[1]), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(f[2], f[3]), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}
__device__ __forceinline__ void bf16x8_to_float(const uint4 v, float* f) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(u[i] << 16);
        f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

constexpr int SMEM = 4 * BN * D + BN * VS * 2;  // K, V raw x 2 stages, V f16

__global__ void __launch_bounds__(WARPS * 32, 2)
    k_prefill_fp8(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                  __nv_bfloat16* __restrict__ out, int T, int Hq, int Hkv, int Lmax, int pos, float scale_log2) {
    constexpr int NT = BN / 8;  // n8 tiles of keys per KV tile
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* Kraw = smem;
    uint8_t* Vraw = smem + 2 * BN * D;
    __half* Vh = reinterpret_cast<__half*>(smem + 4 * BN * D);

    const int qt = gridDim.x - 1 - blockIdx.x, h = blockIdx.y, kvh = h / (Hq / Hkv);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int tw = qt * BM + warp * 16;  // the warp's first query row

    // ---- Q rows tw + g (rr = 0) and tw + g + 8 (rr = 1): per-row e4m3 scale, A fragments in registers
    uint32_t qw[2][4][4];  // [rr][j][word]: bytes of head dims 64j + 16c .. +15
    float rs[2];           // softmax scale per row (log2 domain), including the Q scale
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int t = tw + g + 8 * rr;
        const __nv_bfloat16* qr = q + ((size_t)h * T + min(t, T - 1)) * D + 16 * c;
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f[16];
            bf16x8_to_float(__ldg(reinterpret_cast<const uint4*>(qr + 64 * j)), f);
            bf16x8_to_float(__ldg(reinterpret_cast<const uint4*>(qr + 64 * j) + 1), f + 8);
#pragma unroll
            for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const float qs = amax > 0.f ? amax / 448.f : 1.f, inv = 1.f / qs;
        rs[rr] = qs * scale_log2;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f[16];
            bf16x8_to_float(__ldg(reinterpret_cast<const uint4*>(qr + 64 * j)), f);
            bf16x8_to_float(__ldg(reinterpret_cast<const uint4*>(qr + 64 * j) + 1), f + 8);
#pragma unroll
            for (int i = 0; i < 16; ++i) f[i] *= inv;
#pragma unroll
            for (int wd = 0; wd < 4; ++wd) qw[rr][j][wd] = to_e4m3x4(f + 4 * wd);
        }
    }

    const int qlast = pos + min(qt * BM + BM, T) - 1;  // the block's last query position
    const int nkv = qlast / BN + 1;
    const size_t base = (size_t)kvh * Lmax * D;
    auto load_tile = [&](int j, int stage) {
        for (int idx = tid; idx < BN * 16; idx += WARPS * 32) {
            const int key = idx >> 4, ch = idx & 15, p = j * BN + key;
            const bool ok = p <= qlast;  // past it: zero-fill (stale bytes must not reach the mma)
            const size_t off = base + (size_t)(ok ? p : 0) * D + ch * 16;
            cp_async16(Kraw + (stage * BN + key) * D + ((ch ^ ((key & 1) << 2)) << 4), kc + off, ok);
            cp_async16(Vraw + (stage * BN + key) * D + (ch << 4), vc + off, ok);
        }
    };

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.f, 0.f};
    const int qp0 = pos + tw + g, qp1 = qp0 + 8;  // this lane's two query positions

    load_tile(0, 0);
    cp_async_commit();
    for (int j = 0; j < nkv; ++j) {
        const int stage = j & 1;
        cp_async_wait0();
        __syncthreads();  // tile j landed; every warp is done with tile j - 1 (Vh, and the stage refilled next)
        if (j + 1 < nkv) load_tile(j + 1, stage ^ 1);
        cp_async_commit();
        // V tile e4m3 -> f16 [key][dim]
        for (int idx = tid; idx < BN * 32; idx += WARPS * 32) {
            const int key = idx >> 5, u = idx & 31;
            const uint2 w = *reinterpret_cast<const uint2*>(Vraw + (stage * BN + key) * D + 8 * u);
            *reinterpret_cast<uint4*>(Vh + key * VS + 8 * u) = make_uint4(e4m3x2_h2(w.x), e4m3x2_h2(w.x >> 16), e4m3x2_h2(w.y), e4m3x2_h2(w.y >> 16));
        }
        // S = Q K^T
        float s[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            const uint8_t* kr = Kraw + (stage * BN + nt * 8 + g) * D;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const uint4 kv = *reinterpret_cast<const uint4*>(kr + (((4 * jj + c) ^ ((g & 1) << 2)) << 4));
                mma_fp8(s[nt], qw[0][jj][0], qw[1][jj][0], qw[0][jj][1], qw[1][jj][1], kv.x, kv.y);
                mma_fp8(s[nt], qw[0][jj][2], qw[1][jj][2], qw[0][jj][3], qw[1][jj][3], kv.z, kv.w);
            }
        }
        // online softmax (rows g and g + 8; a row's 4 lanes are one quad)
        const bool diag = (j + 1) * BN - 1 > qp0 - g;  // the tile reaches past the warp's first query position
        float corr[2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int qp = rr ? qp1 : qp0;
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float x = s[nt][2 * rr + e] * rs[rr];
                    if (diag && j * BN + nt * 8 + 2 * c + e > qp) x = -CUDART_INF_F;
                    s[nt][2 * rr + e] = x;
                    tmax = fmaxf(tmax, x);
                }
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            const float mn = fmaxf(m[rr], tmax);
            corr[rr] = mn == m[rr] ? 1.f : ex2(m[rr] - mn);
            float ls = 0.f;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float x = s[nt][2 * rr + e];
                    const float p = x == -CUDART_INF_F ? 0.f : __half2float(__float2half_rn(ex2(x - mn)));
                    s[nt][2 * rr + e] = p;
                    ls += p;
                }
            l[rr] = l[rr] * corr[rr] + ls;
            m[rr] = mn;
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= corr[0];
            o[i][1] *= corr[0];
            o[i][2] *= corr[1];
            o[i][3] *= corr[1];
        }
        __syncthreads();  // Vh complete
        // O += P V
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            const uint32_t pa[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                    pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < 16; ++np) {
                uint32_t bv[4];
                ldsm_x4_t(bv, Vh + (kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                mma_f16(o[2 * np], pa, bv[0], bv[1]);
                mma_f16(o[2 * np + 1], pa, bv[2], bv[3]);
            }
        }
    }
    cp_async_wait0();
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 1);
        l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 2);
        const int t = tw + g + 8 * rr;
        if (t >= T) continue;
        const float inv = 1.f / l[rr];
        __nv_bfloat16* dst = out + ((size_t)t * Hq + h) * D + 2 * c;
#pragma unroll
        for (int nt = 0; nt < 32; ++nt)
            *reinterpret_cast<__nv_bfloat162*>(dst + nt * 8) = __floats2bfloat162_rn(o[nt][2 * rr] * inv, o[nt][2 * rr + 1] * inv);
    }
}

}  // namespace apf

// FP8-QK causal prefill attention. bn: KV tile (32 or 64 keys).
cudaError_t launch_attn_prefill_fp8(const void* q, const void* kc, const void* vc, void* out, int T, int Hq, int Hkv, int Lmax, int pos,
                                    float scale, cudaStream_t st) {
    if (Hq % Hkv || T < 1) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        if (cudaError_t e = cudaFuncSetAttribute(apf::k_prefill_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, apf::SMEM)) return e;
        init = true;
    }
    dim3 grid((T + apf::BM - 1) / apf::BM, Hq), block(apf::WARPS * 32);
    apf::k_prefill_fp8<<<grid, block, apf::SMEM, st>>>((const __nv_bfloat16*)q, (const uint8_t*)kc, (const uint8_t*)vc, (__nv_bfloat16*)out, T, Hq,
                                                       Hkv, Lmax, pos, scale * 1.4426950408889634f);
    return cudaGetLastError();
}
