// gdn_step.cu -- Gated DeltaNet decode kernels: T new tokens per slot (T = 1 plain decode, T = k + 1 speculative
// verify), on the slot's conv window and fp32 recurrent state.
//
// One set of kernels serves all three uses, so a token's output does not depend on how it is processed:
//   plain decode   k_conv (T = 1) + k_conv_commit (n = active) + k_delta (outputs, state advanced by n = active)
//   verify         k_conv (T rows) + k_delta (outputs for all T rows, state untouched)
//   commit         k_conv_commit (n accepted) + k_delta (no outputs, state advanced by n and written)
// Verifying T tokens and decoding them one at a time give the same bits (tests/test_spec_gdn.py).
//
// k_conv         out[b, t, c] = bf16(silu(bf16(sum_k w[c, k] * x[k]))), x = the conv window followed by tokens 0 .. t:
//                the depthwise causal conv (kernel 4) of the reference. The window is not written.
// k_conv_commit  conv window <- the last 3 inputs of [window, mixed[b, 0 .. n_b - 1]].
// k_delta        one block per (slot, value head h), 512 threads. Head h reads q / k of key head h / (Hv / Hk) and v of head
//                h, L2-normalizes q and k, then per token on its fp32 state S [dk = 128, dv = 128]:
//                    S *= exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
//                and the gated RMSNorm  out = bf16(bf16(w * bf16(rmsnorm(bf16(o)))) * silu(z)).
//                beta = bf16(sigmoid(b)), g = -exp(A_log) * softplus(a + dt_bias). Rounding points follow transformers'
//                torch_recurrent_gated_delta_rule + Qwen3_5RMSNormGated (engine/model/qwen35.py).
//
// k_delta layout. Warp w owns state rows [32 (w / 4), +32) and columns [32 (w % 4), +32): lane = column, so every state
// load / store is a coalesced 128-byte row segment and a thread keeps its 32 state values in registers for all T tokens.
// The kernel is a chain of T dependent steps on 64 KB of state per block (read once), so the per-token work that does not
// depend on the state goes before the chain (q / k norms, beta and decay of all T tokens: warp t takes token t, with the
// sums in the same order as a one-token block reduction), the outputs' gated RMSNorm after it (from per-step column
// partials), and a step is two passes over the registers and one __syncthreads (column partials alternate between two
// shared buffers). Time at T = 8: ≈17.5 µs fixed (the 3 MB of state per layer) + ≈0.5 µs per token.
//
// Shapes: mixed, qkv [B, T, C = 2 Hk 128 + Hv 128]; z [B, T, Hv 128]; b, a [B, T, Hv]; n int32 [B] (0 .. T). The rows of
// mixed, z, b and a may be strided (ldm / ldz / ldba elements apart: column slices of the projection outputs, no copies).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "pdl.cuh"

namespace gdn {

constexpr int DK = 128, DV = 128, THREADS = 512, MAX_T = THREADS / 32;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__global__ void k_conv(const __nv_bfloat16* __restrict__ mixed, const __nv_bfloat16* __restrict__ conv_state, const __nv_bfloat16* __restrict__ w,
                       __nv_bfloat16* __restrict__ out, int T, int C, int ldm) {
    PDL_TRIGGER();
    const int bt = blockIdx.y, b = bt / T, t = bt % T, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    float xs[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        xs[k] = tt >= 0 ? __bfloat162float(mixed[((size_t)b * T + tt) * ldm + c]) : __bfloat162float(cs[3 + tt]);
    }
    const __nv_bfloat16* wc = w + (size_t)c * 4;
    float acc = __bfloat162float(wc[0]) * xs[0];
    acc = fmaf(__bfloat162float(wc[1]), xs[1], acc);
    acc = fmaf(__bfloat162float(wc[2]), xs[2], acc);
    acc = fmaf(__bfloat162float(wc[3]), xs[3], acc);
    out[((size_t)b * T + t) * C + c] = __float2bfloat16(silu(bf(acc)));
}

__global__ void k_conv_commit(const __nv_bfloat16* __restrict__ mixed, __nv_bfloat16* __restrict__ conv_state, const int* __restrict__ n_ptr,
                              int T, int C, int ldm) {
    PDL_TRIGGER();
    const int b = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = n_ptr[b];
    if (n == 0) return;
    __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    __nv_bfloat16 hist[3 + MAX_T];
    hist[0] = cs[0]; hist[1] = cs[1]; hist[2] = cs[2];
    for (int t = 0; t < n; ++t) hist[3 + t] = mixed[((size_t)b * T + t) * ldm + c];
    cs[0] = hist[n]; cs[1] = hist[n + 1]; cs[2] = hist[n + 2];
}

// out != nullptr: outputs for all T tokens. n_ptr != nullptr: the state as of token n[b] (0 .. T) is written back.
__global__ void __launch_bounds__(THREADS) k_delta(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ z,
                                                    const __nv_bfloat16* __restrict__ bvec, const __nv_bfloat16* __restrict__ avec,
                                                    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                                                    const __nv_bfloat16* __restrict__ norm_w, float* __restrict__ state,
                                                    __nv_bfloat16* __restrict__ out, int Hk, int Hv, float eps, int T,
                                                    const int* __restrict__ n_ptr, int ldz, int ldba) {
    PDL_TRIGGER();
    const int b = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kh = h / (Hv / Hk), C = 2 * Hk * DK + Hv * DV;
    const int n_write = n_ptr ? n_ptr[b] : 0;                 // tokens after which the state is written back (0: never)
    const int steps = out ? T : n_write;
    extern __shared__ float dsm[];
    float* qs = dsm;                     // [T][DK] normalized q (times 1/sqrt(dk))
    float* ks = qs + T * DK;             // [T][DK] normalized k
    float* vs = ks + T * DK;             // [T][DV]
    float* opart = vs + T * DV;          // [T][4][DV] column partials of o (outputs only)
    float* colred = opart + (out ? T * 4 * DV : 0);  // [2][4][DV]
    float* gb = colred + 2 * 4 * DV;     // [T][2]: beta, decay
    const int rg = warp >> 2, col = ((warp & 3) << 5) + lane;
    float* S = state + ((size_t)b * Hv + h) * DK * DV + (size_t)(rg * 32) * DV + col;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = S[(size_t)r * DV];
    if (warp < steps) {  // warp t: the state-independent part of token t
        const int t = warp;
        const size_t bt = (size_t)b * T + t;
        const __nv_bfloat16* src = qkv + bt * C;
        float q[4], k[4], qq = 0.f, kk = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __bfloat162float(src[kh * DK + 32 * j + lane]);
            k[j] = __bfloat162float(src[Hk * DK + kh * DK + 32 * j + lane]);
            vs[t * DV + 32 * j + lane] = __bfloat162float(src[2 * Hk * DK + h * DV + 32 * j + lane]);
            float sq = q[j] * q[j], sk = k[j] * k[j];
            for (int o = 16; o > 0; o >>= 1) {
                sq += __shfl_xor_sync(0xffffffffu, sq, o);
                sk += __shfl_xor_sync(0xffffffffu, sk, o);
            }
            qq = j ? qq + sq : sq;  // ((r0 + r1) + r2) + r3 over the four 32-element groups
            kk = j ? kk + sk : sk;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qs[t * DK + 32 * j + lane] = q[j] * rsqrtf(qq + 1e-6f) * rsqrtf((float)DK);
            ks[t * DK + 32 * j + lane] = k[j] * rsqrtf(kk + 1e-6f);
        }
        if (lane == 0) {
            const float beta = bf(1.f / (1.f + __expf(-__bfloat162float(bvec[bt * ldba + h]))));
            const float av = __bfloat162float(avec[bt * ldba + h]) + __bfloat162float(dt_bias[h]);
            const float softplus = av > 20.f ? av : log1pf(__expf(av));
            gb[2 * t] = beta;
            gb[2 * t + 1] = __expf(-__expf(__bfloat162float(A_log[h])) * softplus);
        }
    }
    __syncthreads();
    for (int t = 0; t < steps; ++t) {
        const float* kt = ks + t * DK + rg * 32;
        const float* qt = qs + t * DK + rg * 32;
        float* cr = colred + (t & 1) * 4 * DV;
        const float decay = gb[2 * t + 1];
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = s[r] * decay;
        float part = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) part = fmaf(s[r], kt[r], part);
        cr[rg * DV + col] = part;
        __syncthreads();
        const float kv = cr[col] + cr[DV + col] + cr[2 * DV + col] + cr[3 * DV + col];
        const float delta = (vs[t * DV + col] - kv) * gb[2 * t];
        part = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            s[r] = fmaf(kt[r], delta, s[r]);
            part = fmaf(s[r], qt[r], part);
        }
        if (out) opart[(t * 4 + rg) * DV + col] = part;
        if (t + 1 == n_write) {
#pragma unroll
            for (int r = 0; r < 32; ++r) S[(size_t)r * DV] = s[r];
        }
    }
    if (!out) return;
    __syncthreads();
    if (warp < T) {  // warp t: gated RMSNorm of token t's 128 outputs
        const int t = warp;
        const size_t bt = (size_t)b * T + t;
        const float* op = opart + t * 4 * DV;
        float o[4], var = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = 32 * j + lane;
            o[j] = bf(op[c] + op[DV + c] + op[2 * DV + c] + op[3 * DV + c]);
            float sq = o[j] * o[j];
            for (int m = 16; m > 0; m >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, m);
            var = j ? var + sq : sq;
        }
        var = var / DV;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = 32 * j + lane;
            const float xn = bf(o[j] * rsqrtf(var + eps));
            const float y = bf(__bfloat162float(norm_w[c]) * xn);
            const float g = __bfloat162float(z[bt * ldz + h * DV + c]);
            out[bt * Hv * DV + h * DV + c] = __float2bfloat16(y * silu(g));
        }
    }
}

}  // namespace gdn

cudaError_t launch_gdn_conv(const void* mixed, int ldm, const void* conv_state, const void* w, void* out, int B, int T, int C, cudaStream_t st) {
    gdn::k_conv<<<dim3((C + 255) / 256, B * T), 256, 0, st>>>((const __nv_bfloat16*)mixed, (const __nv_bfloat16*)conv_state,
                                                              (const __nv_bfloat16*)w, (__nv_bfloat16*)out, T, C, ldm);
    return cudaGetLastError();
}

cudaError_t launch_gdn_conv_commit(const void* mixed, int ldm, void* conv_state, const int* n, int B, int T, int C, cudaStream_t st) {
    if (T > gdn::MAX_T) return cudaErrorInvalidValue;
    gdn::k_conv_commit<<<dim3((C + 255) / 256, B), 256, 0, st>>>((const __nv_bfloat16*)mixed, (__nv_bfloat16*)conv_state, n, T, C, ldm);
    return cudaGetLastError();
}

cudaError_t launch_gdn_delta(const void* qkv, const void* z, int ldz, const void* b, const void* a, int ldba, const void* A_log, const void* dt_bias,
                             const void* norm_w, float* state, void* out, const int* n, int B, int Hk, int Hv, float eps, int T, cudaStream_t st) {
    if (T < 1 || T > gdn::MAX_T || (!out && !n)) return cudaErrorInvalidValue;
    const int smem = (T * (3 * 128 + (out ? 4 * 128 : 0)) + 2 * 4 * 128 + 2 * T) * (int)sizeof(float);
    static bool attr = false;
    if (!attr) {
        if (cudaError_t e = cudaFuncSetAttribute(gdn::k_delta, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * (7 * 128 + 2) * 4 + 4096))
            return e;
        attr = true;
    }
    gdn::k_delta<<<dim3(Hv, B), gdn::THREADS, smem, st>>>((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)z, (const __nv_bfloat16*)b,
                                                          (const __nv_bfloat16*)a, (const __nv_bfloat16*)A_log, (const __nv_bfloat16*)dt_bias,
                                                          (const __nv_bfloat16*)norm_w, state, (__nv_bfloat16*)out, Hk, Hv, eps, T, n, ldz, ldba);
    return cudaGetLastError();
}
