// sampling.cu -- per-slot seeded uniforms for speculative sampling (PLAN.md 4.5).
//
// out[b, i] = U(0, 1] from Philox4x32-10, keyed by mix(seed[b]), at counter offset[b] + i. The key is a mixed seed, so
// this stream never coincides with the sampling stream of FlashInfer. That stream uses seed[b] directly, with the row
// index as the subsequence.
#include <cuda_runtime.h>
#include <stdint.h>

// Philox4x32-10 (Salmon et al. 2011), bit-identical to cuRAND's curand_init(key, 0, n) + curand_uniform: counter
// (n / 4, 0), output word n % 4, mapped to (0, 1] as u32 * 2^-32 + 2^-33.
__device__ __forceinline__ uint32_t philox_u32(uint64_t key, uint64_t n) {
    uint4 c = make_uint4((uint32_t)(n >> 2), (uint32_t)(n >> 34), 0u, 0u);
    uint2 k = make_uint2((uint32_t)key, (uint32_t)(key >> 32));
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r) { k.x += 0x9E3779B9u; k.y += 0xBB67AE85u; }
        const uint32_t lo0 = 0xD2511F53u * c.x, hi0 = __umulhi(0xD2511F53u, c.x);
        const uint32_t lo1 = 0xCD9E8D57u * c.z, hi1 = __umulhi(0xCD9E8D57u, c.z);
        c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    const uint32_t w[4] = {c.x, c.y, c.z, c.w};
    return w[n & 3];
}

__global__ void k_philox_uniform(const int64_t* __restrict__ seed, const int64_t* __restrict__ offset, float* __restrict__ out, int n) {
    const int b = blockIdx.x;
    const uint64_t key = (uint64_t)seed[b] * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[(size_t)b * n + i] = philox_u32(key, (uint64_t)offset[b] + i) * 2.3283064e-10f + 2.3283064e-10f / 2.0f;  // (0, 1]
}

cudaError_t launch_philox_uniform(const int64_t* seed, const int64_t* offset, float* out, int B, int n, cudaStream_t st) {
    k_philox_uniform<<<B, n < 256 ? n : 256, 0, st>>>(seed, offset, out, n);
    return cudaGetLastError();
}

// Exact logits of candidate rows of an NVFP4 matrix (the rescoring of the low-rank draft head, engine/spec/mtp.py):
// out[b, j] = gs * sum_k x[b, k] * W[cand[b, j], k]. One warp per candidate. Each lane takes the 16-element scale blocks
// lane, lane + 32, ... (fp32 accumulation).
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

__global__ void __launch_bounds__(128) k_rescore_nvfp4(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                                      const uint8_t* __restrict__ sf, float gs, const int64_t* __restrict__ cand,
                                                      float* __restrict__ out, int K, int NC) {
    const int b = blockIdx.y, j = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (j >= NC) return;
    const int64_t row = cand[(size_t)b * NC + j];
    const uint8_t* wr = w + (size_t)row * (K / 2);
    const uint8_t* sr = sf + (size_t)row * (K / 16);
    const __nv_bfloat16* xr = x + (size_t)b * K;
    float acc = 0.f;
    for (int blk = lane; blk < K / 16; blk += 32) {
        const uint2 pw = *reinterpret_cast<const uint2*>(wr + blk * 8);
        const uint4 x0 = *reinterpret_cast<const uint4*>(xr + blk * 16), x1 = *reinterpret_cast<const uint4*>(xr + blk * 16 + 8);
        const __nv_bfloat162* xv0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const __nv_bfloat162* xv1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
        const uint32_t pk[2] = {pw.x, pw.y};
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {  // byte i: elements 2i (low nibble), 2i + 1 (high)
            const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)((pk[i >> 2] >> (8 * (i & 3))) & 0xff), __NV_E2M1);
            const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            const float2 xf = __bfloat1622float2(i < 4 ? xv0[i] : xv1[i - 4]);
            s = fmaf(wf.x, xf.x, s);
            s = fmaf(wf.y, xf.y, s);
        }
        const __half_raw sc = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)sr[blk], __NV_E4M3);
        acc = fmaf(s, __half2float(*reinterpret_cast<const __half*>(&sc)), acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[(size_t)b * NC + j] = acc * gs;
}

cudaError_t launch_rescore_nvfp4(const void* x, const void* w, const void* sf, float gs, const int64_t* cand, float* out, int B, int K, int NC,
                                 cudaStream_t st) {
    if (K % 16) return cudaErrorInvalidValue;
    k_rescore_nvfp4<<<dim3((NC + 3) / 4, B), 128, 0, st>>>((const __nv_bfloat16*)x, (const uint8_t*)w, (const uint8_t*)sf, gs, cand, out, K, NC);
    return cudaGetLastError();
}
