// gemv.cu -- weight-streaming GEMV for decode (PLAN.md 4.3 item 2), W4A16 / W8A16, M <= 8.
//
//   out[m, n] = sum_k x[m, k] * W[n, k]  (+ residual[m, n])
//
// NVFP4 weights: packed e2m1 [N, K/2] (element 2i in the low nibble), e4m3 block scales [N, K/16],
// fp32 global scale. FP8 weights: e4m3 [N, K], fp32 per-tensor scale. Activations bf16 [M, K]
// (decode keeps activations in bf16: free at this arithmetic intensity, and 1.7% better
// perplexity than FP4 activations, docs/phase1_results.md).
//
// One warp per output row; lane l reads the 16-byte chunks l, l+32, ... of the row, so a warp
// streams 512 contiguous bytes per step. UNROLL chunks are loaded before any is consumed to keep
// enough bytes in flight (bench/bw_bench.cu: 4 x 16 B per thread saturates DRAM). Dequantization
// in registers via cvt.e2m1x2 / cvt.e4m3x2 -> half2 -> float; fp32 accumulation; one warp reduce.
// Activations come through the read-only cache: every warp in the grid reads the same x.
//
// The SwiGLU variant computes the gate and up rows of the same index in one warp and writes
// silu(g) * u, so the MLP intermediate is produced in one launch and never round-trips as two tensors.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "pdl.cuh"

namespace gemv {

constexpr int WARPS = 8;  // rows per block

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float2 fp4x2_to_float2(uint8_t b) {
    __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b, __NV_E2M1);
    return __half22float2(*reinterpret_cast<__half2*>(&h));
}
__device__ __forceinline__ float2 fp8x2_to_float2(uint16_t b) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)b, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&h));
}
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}
__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

// Dot of 32 e2m1 weights (one uint4) with 32 bf16 activations starting at xk, two block scales.
template <int M>
__device__ __forceinline__ void dot_nvfp4_chunk(const uint4 w, const uint16_t sf2, const __nv_bfloat16* __restrict__ x,
                                                int K, int k0, float (&acc)[M]) {
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    const float s0 = e4m3_to_float(sf2 & 0xff), s1 = e4m3_to_float(sf2 >> 8);
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * K + k0);
        float part[2] = {0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // 4 x (8 weights in a uint32, 8 activations in a uint4)
            const uint4 xv = __ldg(xp + q);
            const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float2 wf = fp4x2_to_float2((wv[q] >> (8 * b)) & 0xff);
                const float2 xf = bf16x2_to_float2(xs[b]);
                part[q >> 1] = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, part[q >> 1]));
            }
        }
        acc[m] = fmaf(part[0], s0, fmaf(part[1], s1, acc[m]));
    }
}

template <int M, int UNROLL>
__device__ __forceinline__ void row_nvfp4(const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ srow,
                                          const __nv_bfloat16* __restrict__ x, int K, int lane, float (&acc)[M]) {
    const int chunks = K / 32;  // 32 weights = 16 bytes per chunk
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const uint16_t* s2 = reinterpret_cast<const uint16_t*>(srow);
    int c = lane;
    for (; c + 32 * (UNROLL - 1) < chunks; c += 32 * UNROLL) {
        uint4 w[UNROLL];
        uint16_t s[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            w[u] = __ldcs(w4 + c + 32 * u);  // streamed once: evict-first
            s[u] = __ldcs(s2 + c + 32 * u);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) dot_nvfp4_chunk<M>(w[u], s[u], x, K, (c + 32 * u) * 32, acc);
    }
    for (; c < chunks; c += 32) dot_nvfp4_chunk<M>(__ldcs(w4 + c), __ldcs(s2 + c), x, K, c * 32, acc);
}

template <int M, typename OutT>
__device__ __forceinline__ void store_row(OutT* out, const __nv_bfloat16* residual, int N, int n, float (&acc)[M], float scale, int lane) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_sum(acc[m]) * scale;
        if (lane == 0) {
            if (residual) v += __bfloat162float(residual[(size_t)m * N + n]);
            if constexpr (sizeof(OutT) == 4) out[(size_t)m * N + n] = v;
            else out[(size_t)m * N + n] = __float2bfloat16(v);
        }
    }
}

template <int M, int UNROLL, typename OutT>
__global__ void __launch_bounds__(WARPS * 32) k_nvfp4(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                       const uint8_t* __restrict__ sf, float gscale, const __nv_bfloat16* residual,
                                                       OutT* __restrict__ out, int N, int K) {
    PDL_TRIGGER();
    const int lane = threadIdx.x & 31, n = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    row_nvfp4<M, UNROLL>(w + (size_t)n * (K / 2), sf + (size_t)n * (K / 16), x, K, lane, acc);
    store_row<M>(out, residual, N, n, acc, gscale, lane);
}

template <int M, int UNROLL>
__global__ void __launch_bounds__(WARPS * 32) k_nvfp4_swiglu(const __nv_bfloat16* __restrict__ x,
                                                              const uint8_t* __restrict__ wg, const uint8_t* __restrict__ sg, float gg,
                                                              const uint8_t* __restrict__ wu, const uint8_t* __restrict__ su, float gu,
                                                              __nv_bfloat16* __restrict__ out, int N, int K) {
    PDL_TRIGGER();
    const int lane = threadIdx.x & 31, n = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    float ag[M], au[M];
#pragma unroll
    for (int m = 0; m < M; ++m) ag[m] = au[m] = 0.f;
    row_nvfp4<M, UNROLL>(wg + (size_t)n * (K / 2), sg + (size_t)n * (K / 16), x, K, lane, ag);
    row_nvfp4<M, UNROLL>(wu + (size_t)n * (K / 2), su + (size_t)n * (K / 16), x, K, lane, au);
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float g = warp_sum(ag[m]) * gg, u = warp_sum(au[m]) * gu;
        if (lane == 0) out[(size_t)m * N + n] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
    }
}

// ---- FP8 (per-tensor scale) ----
template <int M>
__device__ __forceinline__ void dot_fp8_chunk(const uint4 w, const __nv_bfloat16* __restrict__ x, int K, int k0, float (&acc)[M]) {
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)m * K + k0);
        const uint4 xa = __ldg(xp), xb = __ldg(xp + 1);
        const uint32_t xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float a = acc[m];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 w0 = fp8x2_to_float2(wv[q] & 0xffff), w1 = fp8x2_to_float2(wv[q] >> 16);
            const float2 x0 = bf16x2_to_float2(xs[2 * q]), x1 = bf16x2_to_float2(xs[2 * q + 1]);
            a = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, a))));
        }
        acc[m] = a;
    }
}

template <int M, int UNROLL, typename OutT>
__global__ void __launch_bounds__(WARPS * 32) k_fp8(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, float scale,
                                                     const float* __restrict__ row_scale, const __nv_bfloat16* residual,
                                                     OutT* __restrict__ out, int N, int K) {
    PDL_TRIGGER();
    const int lane = threadIdx.x & 31, n = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (n >= N) return;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const uint4* w4 = reinterpret_cast<const uint4*>(w + (size_t)n * K);
    const int chunks = K / 16;
    int c = lane;
    for (; c + 32 * (UNROLL - 1) < chunks; c += 32 * UNROLL) {
        uint4 wr[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) wr[u] = __ldcs(w4 + c + 32 * u);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) dot_fp8_chunk<M>(wr[u], x, K, (c + 32 * u) * 16, acc);
    }
    for (; c < chunks; c += 32) dot_fp8_chunk<M>(__ldcs(w4 + c), x, K, c * 16, acc);
    store_row<M>(out, residual, N, n, acc, row_scale ? row_scale[n] : scale, lane);
}

// ---- BF16 weights (tiny projections only): one block per row, K split over the 8 warps ----
template <int M>
__global__ void __launch_bounds__(WARPS * 32) k_bf16(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                      __nv_bfloat16* __restrict__ out, int N, int K) {
    PDL_TRIGGER();
    const int n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    const uint4* w4 = reinterpret_cast<const uint4*>(w + (size_t)n * K);
    for (int c = tid; c < K / 8; c += WARPS * 32) {
        const uint4 wv = __ldcs(w4 + c);
        const uint32_t wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * K) + c);
            const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = bf16x2_to_float2(wu[j]), b = bf16x2_to_float2(xu[j]);
                acc[m] = fmaf(a.x, b.x, fmaf(a.y, b.y, acc[m]));
            }
        }
    }
    __shared__ float red[M][WARPS];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float v = warp_sum(acc[m]);
        if (lane == 0) red[m][warp] = v;
    }
    __syncthreads();
    if (tid < M) {
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i) v += red[tid][i];
        out[(size_t)tid * N + n] = __float2bfloat16(v);
    }
}

}  // namespace gemv

// ---------------------------------------------------------------------------------------------
// Host launchers (called from bindings.cpp). M is dispatched to a template instance.
// ---------------------------------------------------------------------------------------------
#define DISPATCH_M(M_, ...)                                       \
    switch (M_) {                                                 \
        case 1: { constexpr int M = 1; __VA_ARGS__; break; }      \
        case 2: { constexpr int M = 2; __VA_ARGS__; break; }      \
        case 3: { constexpr int M = 3; __VA_ARGS__; break; }      \
        case 4: { constexpr int M = 4; __VA_ARGS__; break; }      \
        case 5: { constexpr int M = 5; __VA_ARGS__; break; }      \
        case 6: { constexpr int M = 6; __VA_ARGS__; break; }      \
        case 7: { constexpr int M = 7; __VA_ARGS__; break; }      \
        case 8: { constexpr int M = 8; __VA_ARGS__; break; }      \
        default: return cudaErrorInvalidValue;                    \
    }

constexpr int UNROLL_FP4 = 4, UNROLL_FP8 = 4;

cudaError_t launch_nvfp4_gemv(const void* x, const void* w, const void* sf, float gscale, const void* residual, void* out,
                              bool out_fp32, int M_, int N, int K, cudaStream_t st) {
    if (K % 32) return cudaErrorInvalidValue;
    dim3 grid((N + gemv::WARPS - 1) / gemv::WARPS), block(gemv::WARPS * 32);
    auto xb = (const __nv_bfloat16*)x;
    auto rb = (const __nv_bfloat16*)residual;
    DISPATCH_M(M_, if (out_fp32) gemv::k_nvfp4<M, UNROLL_FP4, float><<<grid, block, 0, st>>>(xb, (const uint8_t*)w, (const uint8_t*)sf, gscale, rb, (float*)out, N, K);
                   else gemv::k_nvfp4<M, UNROLL_FP4, __nv_bfloat16><<<grid, block, 0, st>>>(xb, (const uint8_t*)w, (const uint8_t*)sf, gscale, rb, (__nv_bfloat16*)out, N, K));
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_swiglu(const void* x, const void* wg, const void* sg, float gg, const void* wu, const void* su, float gu,
                                void* out, int M_, int N, int K, cudaStream_t st) {
    if (K % 32) return cudaErrorInvalidValue;
    dim3 grid((N + gemv::WARPS - 1) / gemv::WARPS), block(gemv::WARPS * 32);
    DISPATCH_M(M_, gemv::k_nvfp4_swiglu<M, UNROLL_FP4><<<grid, block, 0, st>>>((const __nv_bfloat16*)x, (const uint8_t*)wg, (const uint8_t*)sg, gg,
                                                                               (const uint8_t*)wu, (const uint8_t*)su, gu, (__nv_bfloat16*)out, N, K));
    return cudaGetLastError();
}

cudaError_t launch_fp8_gemv(const void* x, const void* w, float scale, const float* row_scale, const void* residual, void* out, bool out_fp32,
                            int M_, int N, int K, cudaStream_t st) {
    if (K % 16) return cudaErrorInvalidValue;
    dim3 grid((N + gemv::WARPS - 1) / gemv::WARPS), block(gemv::WARPS * 32);
    auto xb = (const __nv_bfloat16*)x;
    auto rb = (const __nv_bfloat16*)residual;
    DISPATCH_M(M_, if (out_fp32) gemv::k_fp8<M, UNROLL_FP8, float><<<grid, block, 0, st>>>(xb, (const uint8_t*)w, scale, row_scale, rb, (float*)out, N, K);
                   else gemv::k_fp8<M, UNROLL_FP8, __nv_bfloat16><<<grid, block, 0, st>>>(xb, (const uint8_t*)w, scale, row_scale, rb, (__nv_bfloat16*)out, N, K));
    return cudaGetLastError();
}

cudaError_t launch_bf16_gemv(const void* x, const void* w, void* out, int M_, int N, int K, cudaStream_t st) {
    if (K % 8) return cudaErrorInvalidValue;
    dim3 grid(N), block(gemv::WARPS * 32);
    DISPATCH_M(M_, gemv::k_bf16<M><<<grid, block, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)out, N, K));
    return cudaGetLastError();
}
