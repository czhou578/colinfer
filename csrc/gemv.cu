// gemv.cu -- BF16 GEMV for the small decode projections (the b / a gates of the GDN layers: 96 rows of 5120),
// M <= 8 rows.
//
//   out[m, n] = sum_k x[m, k] * W[n, k]
//
// One block per output row. Its 8 warps split K, and each lane streams 16-byte chunks of the row. The accumulation is
// fp32, with one warp reduce and a shared-memory sum over the warps. All quantized linears run on the tensor-core skinny
// GEMM (skinny.cu) instead.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "pdl.cuh"

namespace gemv {

constexpr int WARPS = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

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


cudaError_t launch_bf16_gemv(const void* x, const void* w, void* out, int M_, int N, int K, cudaStream_t st) {
    if (K % 8) return cudaErrorInvalidValue;
    dim3 grid(N), block(gemv::WARPS * 32);
    DISPATCH_M(M_, gemv::k_bf16<M><<<grid, block, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)out, N, K));
    return cudaGetLastError();
}
