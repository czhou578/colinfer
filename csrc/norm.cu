// norm.cu -- zero-centered RMSNorm for decode rows: out = bf16( x / rms(x) * (1 + w) ), fp32 math,
// same rounding as engine/model/qwen35.RMSNorm. One block per row, 16-byte loads.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "pdl.cuh"

namespace rmsn {

constexpr int THREADS = 256;

__global__ void __launch_bounds__(THREADS) k_rmsnorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                      __nv_bfloat16* __restrict__ out, int K, float eps) {
    PDL_TRIGGER();
    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * K;
    __nv_bfloat16* orow = out + (size_t)blockIdx.x * K;
    float ss = 0.f;
    for (int i = threadIdx.x * 8; i < K; i += THREADS * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(xr + i);
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = __uint_as_float(u[j] << 16), b = __uint_as_float(u[j] & 0xffff0000u);
            ss = fmaf(a, a, fmaf(b, b, ss));
        }
    }
    __shared__ float red[THREADS / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) tot += red[i];
    const float r = rsqrtf(tot / K + eps);
    for (int i = threadIdx.x * 8; i < K; i += THREADS * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(xr + i);
        const uint4 wv = *reinterpret_cast<const uint4*>(w + i);
        const uint32_t u[4] = {v.x, v.y, v.z, v.w}, wu[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = __uint_as_float(u[j] << 16) * r * (1.f + __uint_as_float(wu[j] << 16));
            const float b = __uint_as_float(u[j] & 0xffff0000u) * r * (1.f + __uint_as_float(wu[j] & 0xffff0000u));
            const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
            o[j] = *reinterpret_cast<const uint32_t*>(&p);
        }
        *reinterpret_cast<uint4*>(orow + i) = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

}  // namespace rmsn

cudaError_t launch_rmsnorm(const void* x, const void* w, void* out, int M, int K, float eps, cudaStream_t st) {
    if (K % 8) return cudaErrorInvalidValue;
    rmsn::k_rmsnorm<<<M, rmsn::THREADS, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)out, K, eps);
    return cudaGetLastError();
}
