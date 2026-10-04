// sampling.cu -- per-slot seeded uniforms for speculative sampling (PLAN.md 4.5).
//
// out[b, i] = U[0, 1) from Philox4x32-10 keyed by mix(seed[b]) at counter offset[b] + i. The key is a mixed seed
// so this stream never coincides with FlashInfer's sampling stream, which uses seed[b] directly with the
// row index as the subsequence.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

__global__ void k_philox_uniform(const int64_t* __restrict__ seed, const int64_t* __restrict__ offset, float* __restrict__ out, int n) {
    const int b = blockIdx.x, i = threadIdx.x;
    if (i >= n) return;
    const uint64_t key = (uint64_t)seed[b] * 6364136223846793005ULL + 1442695040888963407ULL;
    curandStatePhilox4_32_10_t st;
    curand_init(key, 0, (uint64_t)offset[b] + i, &st);
    out[(size_t)b * n + i] = 1.f - curand_uniform(&st);  // curand_uniform is (0, 1]
}

cudaError_t launch_philox_uniform(const int64_t* seed, const int64_t* offset, float* out, int B, int n, cudaStream_t st) {
    if (n > 1024) return cudaErrorInvalidValue;
    k_philox_uniform<<<B, n, 0, st>>>(seed, offset, out, n);
    return cudaGetLastError();
}
