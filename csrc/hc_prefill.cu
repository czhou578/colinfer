// hc_prefill.cu -- the tail of the gated residual of qwen4_exp over the rows of a prefill chunk (engine/model/
// prefill_qwen4exp.py hc_pre): y = bf16((((p0 + p1) + p2) + p3) / 4) with p_s = bf16(bf16(sigmoid(u_s)) * n_s), the
// rounding points of the reference GatedResidual (docs/history/qwen4exp_plan.md section 5.4, steps 5 to 7). In torch
// ops, the sum over the 4 strided streams takes three slow strided passes; one kernel reads u and n once. The binding is
// in qsa_bind.cpp (ops().hc_mix_rows). Tests: tests/test_prefill_qwen4exp.py.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common.cuh"

namespace hcp {
using namespace cc;

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }  // torch's sigmoid in fp32 (precise expf)

// Grid (ceil(H / 8 / 128), T), 128 threads: thread (x, t) owns the 8 columns c = 8 (128 x + tid) .. + 7 of row t.
// u: [T, 4 H] rows ldu elements apart; n: [T, 4 H] contiguous; y: [T, H].
__global__ void __launch_bounds__(128) k_mix_rows(const __nv_bfloat16* __restrict__ u, int ldu, const __nv_bfloat16* __restrict__ n,
                                                  __nv_bfloat16* __restrict__ y, int H) {
    const int t = blockIdx.y, c = 8 * (blockIdx.x * 128 + threadIdx.x);
    if (c >= H) return;
    float acc[8];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float fu[8], fn[8];
        bf16x8_to_float(*reinterpret_cast<const uint4*>(u + (size_t)t * ldu + s * H + c), fu);
        bf16x8_to_float(*reinterpret_cast<const uint4*>(n + ((size_t)t * 4 + s) * H + c), fn);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float p = bf(bf(sigm(fu[i])) * fn[i]);
            acc[i] = s == 0 ? p : acc[i] + p;
        }
    }
    uint4 o;
    uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(acc[2 * i] / 4.f, acc[2 * i + 1] / 4.f);
        ou[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(y + (size_t)t * H + c) = o;
}

}  // namespace hcp

cudaError_t launch_hc_mix_rows(const void* u, int ldu, const void* n, void* y, int T, int H, cudaStream_t st) {
    if (H % 8 || ldu % 8 || T < 1) return cudaErrorInvalidValue;
    const dim3 grid((H / 8 + 127) / 128, T);
    hcp::k_mix_rows<<<grid, 128, 0, st>>>((const __nv_bfloat16*)u, ldu, (const __nv_bfloat16*)n, (__nv_bfloat16*)y, H);
    return cudaGetLastError();
}
