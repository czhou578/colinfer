// hc.cu -- hyper-connection (gated residual) kernels for qwen4_exp (docs/history/qwen4exp_plan.md section 5.4): the low-rank
// mix and the injection. The grouped norm (hc_norm) is the grouped form of csrc/norm.cu. The bindings are in hc_bind.cpp, the
// Python module (KernelGatedResidual) in engine/model/hc_kernels.py.
//
// The residual stream of a row is x4 [4 x 2560]. A GatedResidual computes, with bf16 weights, fp32 math and one rounding
// per op (the rounding points of transformers' Qwen4ExpTextGatedResidual):
//   n     = bf16(x_s * rsqrt(mean_s(x^2) + eps) * (1 + w))           per stream s       (hc_norm)
//   d, e  = bf16(n W_down^T) [320], bf16(n W_inj^T) [4]                                  (bf16_gemv / the skinny GEMM)
//   a     = bf16(silu(bf16(d / 4))),  u = bf16(a W_up^T) [10240],  mix = bf16(sigmoid(u))   (hc_mix)
//   p_s   = bf16(mix_s * n_s),  y = bf16((((p0 + p1) + p2) + p3) / 4) [2560]             (hc_mix)
//   inj_s = 2 bf16(sigmoid(bf16(e_s / 4)))                                                (hc_mix)
// and after the sublayer (o = its bf16 output [2560]):
//   x4_s  = bf16(x4_s + bf16(o * inj_s))                                                 (hc_post)
// expf is the precise one (torch's silu and sigmoid round the same way).
//
// Row invariance: every output row depends on its own input row only, with the same arithmetic for any M (<= 16), so
// decode and verify give the same bits for a token.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.cuh"
#include "pdl.cuh"

namespace hc {
using namespace cc;

constexpr int S = 4, H = 2560, R = 320, N = S * H, MAXM = 16;
constexpr int COLS = 16, THREADS = 128;  // COLS: the columns of each stream per block

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// Grid (H / 16 = 160), 128 threads. Block c owns the columns k = 16 c .. 16 c + 15 of each stream, so the 64 W_up rows
// s * 2560 + k (4 runs of 16 x 640 B). It first computes a [M, 320] (bf16, zero rows past M) into shared memory (each
// block recomputes it: 5 k silu per block). Warp s takes stream s: u = a W_up^T for its 16 rows as two n8 tiles of
// mma.sync m16n8k16 (bf16 -> fp32; the A rows are the M <= 16 token rows, so a row's u does not depend on M). Lane
// (g = lane / 4, c = lane % 4) loads the dims 32 jj + 8 c .. + 7 (jj = 0 .. 9) of W row g of each tile straight from
// global memory: the first 4 are its B fragment of k-step 2 jj, the next 4 of k-step 2 jj + 1, and the A fragments read a
// in the same dim order. Then mix and p go to shared memory, and the block writes y for its 16 columns. Block 0 also
// writes inj. d / e: row strides ldd / lde (column slices of the stacked down | inject output).
constexpr int AS = R + 32;  // the a row stride in shared memory (elements): rows 64 bytes apart in the bank space

__global__ void __launch_bounds__(THREADS) k_mix(const __nv_bfloat16* __restrict__ d, int ldd, const __nv_bfloat16* __restrict__ w_up,
                                                  const __nv_bfloat16* __restrict__ n, __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ e,
                                                  int lde, __nv_bfloat16* __restrict__ inj, int M) {
    PDL_TRIGGER();
    __shared__ __align__(16) __nv_bfloat16 as[MAXM * AS];
    __shared__ float ps[S * MAXM * COLS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, c0 = blockIdx.x * COLS;
    for (int i = tid; i < MAXM * R; i += THREADS) {
        const int m = i / R, j = i % R;
        float av = 0.f;
        if (m < M) {
            const float dv = bf(__bfloat162float(d[(size_t)m * ldd + j]) / 4.f);
            av = dv / (1.f + expf(-dv));
        }
        as[m * AS + j] = __float2bfloat16(av);
    }
    if (blockIdx.x == 0 && e != nullptr && tid < M * S) {
        const int m = tid / S, s = tid % S;
        inj[m * S + s] = __float2bfloat16(2.f * bf(sigmoid(bf(__bfloat162float(e[(size_t)m * lde + s]) / 4.f))));
    }
    const int s = warp, g = lane >> 2, c = lane & 3;
    uint4 wb[2][R / 32];  // the B fragments of the warp's two n tiles, loaded before the barrier
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const __nv_bfloat16* wr = w_up + ((size_t)s * H + c0 + nt * 8 + g) * R + 8 * c;
#pragma unroll
        for (int jj = 0; jj < R / 32; ++jj) wb[nt][jj] = __ldg(reinterpret_cast<const uint4*>(wr + 32 * jj));
    }
    __syncthreads();
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int jj = 0; jj < R / 32; ++jj) {
        const uint4 x0 = *reinterpret_cast<const uint4*>(as + g * AS + 32 * jj + 8 * c);
        const uint4 x1 = *reinterpret_cast<const uint4*>(as + (g + 8) * AS + 32 * jj + 8 * c);
        const uint32_t a0[4] = {x0.x, x1.x, x0.y, x1.y}, a1[4] = {x0.z, x1.z, x0.w, x1.w};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mma_bf16(acc[nt], a0, wb[nt][jj].x, wb[nt][jj].y);
            mma_bf16(acc[nt], a1, wb[nt][jj].z, wb[nt][jj].w);
        }
    }
    // acc[nt][q]: token row g (q < 2) or g + 8, column r = nt * 8 + 2 c + q % 2 of stream s
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int m = g + (q >= 2 ? 8 : 0), r = nt * 8 + 2 * c + (q & 1);
            if (m < M) {
                const float mix = bf(sigmoid(bf(acc[nt][q])));
                ps[(s * MAXM + m) * COLS + r] = bf(mix * __bfloat162float(n[(size_t)m * N + s * H + c0 + r]));
            }
        }
    __syncthreads();
    for (int i = tid; i < M * COLS; i += THREADS) {
        const int m = i / COLS, r = i % COLS;
        const float sum = ((ps[(0 * MAXM + m) * COLS + r] + ps[(1 * MAXM + m) * COLS + r]) + ps[(2 * MAXM + m) * COLS + r]) +
                          ps[(3 * MAXM + m) * COLS + r];
        y[(size_t)m * H + c0 + r] = __float2bfloat16(sum / 4.f);
    }
}

// One thread per 8 elements of x4 [M, 4 x 2560] (in place): x4_s = bf16(x4_s + bf16(o * inj_s)).
__global__ void k_post(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ inj, __nv_bfloat16* __restrict__ x4, int M) {
    PDL_TRIGGER();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * (N / 8)) return;
    const int m = i / (N / 8), c = (i % (N / 8)) * 8, s = c / H, k = c % H;
    const float g = __bfloat162float(inj[m * S + s]);
    float xv[8], ov[8];
    bf16x8_to_float(*reinterpret_cast<const uint4*>(x4 + (size_t)m * N + c), xv);
    bf16x8_to_float(*reinterpret_cast<const uint4*>(o + (size_t)m * H + k), ov);
    uint32_t r[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 p = __floats2bfloat162_rn(xv[2 * j] + bf(ov[2 * j] * g), xv[2 * j + 1] + bf(ov[2 * j + 1] * g));
        r[j] = *reinterpret_cast<const uint32_t*>(&p);
    }
    *reinterpret_cast<uint4*>(x4 + (size_t)m * N + c) = make_uint4(r[0], r[1], r[2], r[3]);
}

}  // namespace hc

cudaError_t launch_hc_mix(const void* d, int ldd, const void* w_up, const void* n, void* y, const void* e, int lde, void* inj, int M, cudaStream_t st) {
    if (M < 1 || M > hc::MAXM) return cudaErrorInvalidValue;
    hc::k_mix<<<hc::H / hc::COLS, hc::THREADS, 0, st>>>((const __nv_bfloat16*)d, ldd, (const __nv_bfloat16*)w_up, (const __nv_bfloat16*)n,
                                                        (__nv_bfloat16*)y, (const __nv_bfloat16*)e, lde, (__nv_bfloat16*)inj, M);
    return cudaGetLastError();
}

cudaError_t launch_hc_post(const void* o, const void* inj, void* x4, int M, cudaStream_t st) {
    const int n = M * (hc::N / 8);
    hc::k_post<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)inj, (__nv_bfloat16*)x4, M);
    return cudaGetLastError();
}
