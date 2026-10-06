// gdn_step.cu -- fused Gated DeltaNet decode step (PLAN.md 4.3 item 5), T = 1 per slot.
//
// k_conv:  qkv[b, c] = bf16(silu(bf16(sum_k w[c,k] * [conv_state[b,c,0..2], mixed[b,c]][k])));
//          conv_state[b, c] <- last three inputs. One thread per channel.
// k_delta: one block per (slot, value head h), 512 threads. Head h reads q/k of key head h / (Hv/Hk)
//          and v of head h from qkv, L2-normalises q and k, then on its fp32 state S [dk=128, dv=128]:
//              S *= exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
//          followed by the gated RMSNorm  out = bf16( bf16(w * bf16(rmsnorm(bf16(o)))) * silu(z) ).
//          beta = bf16(sigmoid(b)), g = -exp(A_log) * softplus(a + dt_bias). Rounding points follow
//          transformers' torch_recurrent_gated_delta_rule + Qwen3_5RMSNormGated.
//          Warp w owns rows [32 (w/4), +32) and columns [32 (w%4), +32): lane = column, so every
//          state load/store is a coalesced 128-byte row segment. Column sums over the 4 row groups
//          go through shared memory.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "pdl.cuh"

namespace gdn {

constexpr int DK = 128, DV = 128, THREADS = 512;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// active (optional, int32 [B]): slots with active[b] == 0 keep their conv state (idle or mid-prefill slots)
__global__ void k_conv(const __nv_bfloat16* __restrict__ mixed, __nv_bfloat16* __restrict__ conv_state, const __nv_bfloat16* __restrict__ w,
                       __nv_bfloat16* __restrict__ out, int C, const int* __restrict__ active) {
    PDL_TRIGGER();
    const int b = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const bool upd = active == nullptr || active[b] != 0;
    __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    const float x0 = __bfloat162float(cs[0]), x1 = __bfloat162float(cs[1]), x2 = __bfloat162float(cs[2]);
    const __nv_bfloat16 xn = mixed[(size_t)b * C + c];
    const float x3 = __bfloat162float(xn);
    const __nv_bfloat16* wc = w + (size_t)c * 4;
    float acc = __bfloat162float(wc[0]) * x0;
    acc = fmaf(__bfloat162float(wc[1]), x1, acc);
    acc = fmaf(__bfloat162float(wc[2]), x2, acc);
    acc = fmaf(__bfloat162float(wc[3]), x3, acc);
    out[(size_t)b * C + c] = __float2bfloat16(silu(bf(acc)));
    if (upd) {
        cs[0] = cs[1];
        cs[1] = cs[2];
        cs[2] = xn;
    }
}

__device__ __forceinline__ float block_sum_128(float v, float* red, int tid) {
    // sum of v over threads 0..127 (4 warps); result broadcast to all 512 threads
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (tid < 128 && (tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    const float s = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return s;
}

__global__ void __launch_bounds__(THREADS) k_delta(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ z,
                                                    const __nv_bfloat16* __restrict__ bvec, const __nv_bfloat16* __restrict__ avec,
                                                    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                                                    const __nv_bfloat16* __restrict__ norm_w, float* __restrict__ state,
                                                    __nv_bfloat16* __restrict__ out, int Hk, int Hv, float eps, const int* __restrict__ active) {
    PDL_TRIGGER();
    const int b = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool upd = active == nullptr || active[b] != 0;  // inactive slots: compute, but leave the state untouched
    const int kh = h / (Hv / Hk), C = 2 * Hk * DK + Hv * DV;
    const __nv_bfloat16* src = qkv + (size_t)b * C;
    __shared__ float qs[DK], ks[DK], vs[DV], red[4], colred[4][DV];
    if (tid < DK) {
        qs[tid] = __bfloat162float(src[kh * DK + tid]);
        ks[tid] = __bfloat162float(src[Hk * DK + kh * DK + tid]);
        vs[tid] = __bfloat162float(src[2 * Hk * DK + h * DV + tid]);
    }
    __syncthreads();
    // l2norm (eps 1e-6) and the 1/sqrt(dk) query scale, in fp32
    const float qq = block_sum_128(tid < DK ? qs[tid] * qs[tid] : 0.f, red, tid);
    const float kk = block_sum_128(tid < DK ? ks[tid] * ks[tid] : 0.f, red, tid);
    if (tid < DK) {
        qs[tid] = qs[tid] * rsqrtf(qq + 1e-6f) * rsqrtf((float)DK);
        ks[tid] = ks[tid] * rsqrtf(kk + 1e-6f);
    }
    __syncthreads();
    const float beta = bf(1.f / (1.f + __expf(-__bfloat162float(bvec[(size_t)b * Hv + h]))));
    const float av = __bfloat162float(avec[(size_t)b * Hv + h]) + __bfloat162float(dt_bias[h]);
    const float softplus = av > 20.f ? av : log1pf(__expf(av));
    const float decay = __expf(-__expf(__bfloat162float(A_log[h])) * softplus);

    const int rg = warp >> 2, col = ((warp & 3) << 5) + lane;
    float* S = state + ((size_t)b * Hv + h) * DK * DV + (size_t)(rg * 32) * DV + col;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = S[(size_t)r * DV] * decay;
    float part = 0.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) part = fmaf(s[r], ks[rg * 32 + r], part);
    colred[rg][col] = part;
    __syncthreads();
    const float kv = colred[0][col] + colred[1][col] + colred[2][col] + colred[3][col];
    const float delta = (vs[col] - kv) * beta;
    __syncthreads();
    part = 0.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        s[r] = fmaf(ks[rg * 32 + r], delta, s[r]);
        if (upd) S[(size_t)r * DV] = s[r];
        part = fmaf(s[r], qs[rg * 32 + r], part);
    }
    colred[rg][col] = part;
    __syncthreads();
    // gated RMSNorm over the 128 outputs of this head (threads 0..127 hold columns 0..127)
    float o = 0.f;
    if (tid < DV) o = bf(colred[0][tid] + colred[1][tid] + colred[2][tid] + colred[3][tid]);
    const float var = block_sum_128(tid < DV ? o * o : 0.f, red, tid) / DV;
    if (tid < DV) {
        const float xn = bf(o * rsqrtf(var + eps));
        const float y = bf(__bfloat162float(norm_w[tid]) * xn);
        const float g = __bfloat162float(z[(size_t)b * Hv * DV + h * DV + tid]);
        out[(size_t)b * Hv * DV + h * DV + tid] = __float2bfloat16(y * silu(g));
    }
}

// ---------------------------------------------------------------------------------------------
// Speculative-decoding verify / commit (PLAN.md 4.5): T tokens per slot through the same arithmetic as
// k_conv / k_delta applied token after token, so greedy outputs match plain decode bit for bit.
//   k_conv_multi   out[b, t] from the conv state plus tokens < t of this step; conv state NOT written
//   k_conv_commit  conv state <- last 3 inputs of [state, mixed[b, 0 .. n_b - 1]]
//   k_delta_multi  verify (n == nullptr): outputs for all T tokens, state untouched;
//                  commit (n != nullptr): no outputs, state advanced by n_b tokens and written back
// mixed / qkv: [B, T, C]; z: [B, T, Hv*128]; b, a: [B, T, Hv]; n: int32 [B] tokens to commit (1..T). mixed, z, b and a
// rows may be strided (ldm / ldz / ldba elements apart: views into the projection outputs, no copies).
// ---------------------------------------------------------------------------------------------
__global__ void k_conv_multi(const __nv_bfloat16* __restrict__ mixed, const __nv_bfloat16* __restrict__ conv_state,
                             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int T, int C, int ldm) {
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
    __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    __nv_bfloat16 hist[3 + 16];
    hist[0] = cs[0]; hist[1] = cs[1]; hist[2] = cs[2];
    for (int t = 0; t < n; ++t) hist[3 + t] = mixed[((size_t)b * T + t) * ldm + c];
    cs[0] = hist[n]; cs[1] = hist[n + 1]; cs[2] = hist[n + 2];
}

// Latency layout (the kernel is a chain of T dependent steps, its 64 KB of state per block are read once): the q / k
// L2 norms, beta and the decay of all T tokens are computed up front (warp t: token t, the same sums in the same
// order as k_delta's block_sum_128), the outputs' gated RMSNorm after the loop from per-step column partials, so a
// step is the two state passes and one __syncthreads (the kv column partials alternate between two buffers).
__global__ void __launch_bounds__(THREADS) k_delta_multi(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ z,
                                                          const __nv_bfloat16* __restrict__ bvec, const __nv_bfloat16* __restrict__ avec,
                                                          const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                                                          const __nv_bfloat16* __restrict__ norm_w, float* __restrict__ state,
                                                          __nv_bfloat16* __restrict__ out, int Hk, int Hv, float eps, int T,
                                                          const int* __restrict__ n_ptr, int ldz, int ldba) {
    PDL_TRIGGER();
    const int b = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kh = h / (Hv / Hk), C = 2 * Hk * DK + Hv * DV;
    const bool commit = n_ptr != nullptr;
    const int steps = commit ? n_ptr[b] : T;
    extern __shared__ float dsm[];
    float* qs = dsm;                 // [T][DK] normalised q (times 1/sqrt(dk))
    float* ks = qs + T * DK;         // [T][DK] normalised k
    float* vs = ks + T * DK;         // [T][DV]
    float* opart = vs + T * DV;      // [T][4][DV] column partials of o (verify)
    float* colred = opart + (commit ? 0 : T * 4 * DV);  // [2][4][DV]
    float* gb = colred + 2 * 4 * DV;  // [T][2]: beta, decay
    const int rg = warp >> 2, col = ((warp & 3) << 5) + lane;
    float* S = state + ((size_t)b * Hv + h) * DK * DV + (size_t)(rg * 32) * DV + col;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = S[(size_t)r * DV];
    if (warp < steps) {
        const int t = warp;
        const size_t bt = (size_t)b * T + t;
        const __nv_bfloat16* src = qkv + bt * C;
        float q[4], k[4], qq = 0.f, kk = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __bfloat162float(src[kh * DK + 32 * j + lane]);
            k[j] = __bfloat162float(src[Hk * DK + kh * DK + 32 * j + lane]);
            vs[t * DV + 32 * j + lane] = __bfloat162float(src[2 * Hk * DK + h * DV + 32 * j + lane]);
            float a = q[j] * q[j], c = k[j] * k[j];
            for (int o = 16; o > 0; o >>= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, o);
                c += __shfl_xor_sync(0xffffffffu, c, o);
            }
            qq = j ? qq + a : a;
            kk = j ? kk + c : c;
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
        if (!commit) opart[(t * 4 + rg) * DV + col] = part;
    }
    if (commit) {
#pragma unroll
        for (int r = 0; r < 32; ++r) S[(size_t)r * DV] = s[r];
        return;
    }
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
            float a = o[j] * o[j];
            for (int m = 16; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
            var = j ? var + a : a;
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

cudaError_t launch_gdn_conv(const void* mixed, void* conv_state, const void* w, void* out, int B, int C, const int* active, cudaStream_t st) {
    dim3 grid((C + 255) / 256, B);
    gdn::k_conv<<<grid, 256, 0, st>>>((const __nv_bfloat16*)mixed, (__nv_bfloat16*)conv_state, (const __nv_bfloat16*)w, (__nv_bfloat16*)out, C, active);
    return cudaGetLastError();
}

cudaError_t launch_gdn_delta(const void* qkv, const void* z, const void* b, const void* a, const void* A_log, const void* dt_bias,
                             const void* norm_w, float* state, void* out, int B, int Hk, int Hv, float eps, const int* active, cudaStream_t st) {
    dim3 grid(Hv, B);
    gdn::k_delta<<<grid, gdn::THREADS, 0, st>>>((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (const __nv_bfloat16*)a,
                                                (const __nv_bfloat16*)A_log, (const __nv_bfloat16*)dt_bias, (const __nv_bfloat16*)norm_w, state,
                                                (__nv_bfloat16*)out, Hk, Hv, eps, active);
    return cudaGetLastError();
}

cudaError_t launch_gdn_conv_multi(const void* mixed, const void* conv_state, const void* w, void* out, int B, int T, int C, int ldm, cudaStream_t st) {
    dim3 grid((C + 255) / 256, B * T);
    gdn::k_conv_multi<<<grid, 256, 0, st>>>((const __nv_bfloat16*)mixed, (const __nv_bfloat16*)conv_state, (const __nv_bfloat16*)w,
                                            (__nv_bfloat16*)out, T, C, ldm);
    return cudaGetLastError();
}

cudaError_t launch_gdn_conv_commit(const void* mixed, void* conv_state, const int* n, int B, int T, int C, int ldm, cudaStream_t st) {
    if (T > 16) return cudaErrorInvalidValue;
    dim3 grid((C + 255) / 256, B);
    gdn::k_conv_commit<<<grid, 256, 0, st>>>((const __nv_bfloat16*)mixed, (__nv_bfloat16*)conv_state, n, T, C, ldm);
    return cudaGetLastError();
}

cudaError_t launch_gdn_delta_multi(const void* qkv, const void* z, const void* b, const void* a, const void* A_log, const void* dt_bias,
                                   const void* norm_w, float* state, void* out, int B, int Hk, int Hv, float eps, int T, const int* n,
                                   int ldz, int ldba, cudaStream_t st) {
    if (T < 1 || T > gdn::THREADS / 32) return cudaErrorInvalidValue;
    dim3 grid(Hv, B);
    const int smem = (T * (3 * 128 + (n ? 0 : 4 * 128)) + 2 * 4 * 128 + 2 * T) * (int)sizeof(float);
    static bool attr = false;
    if (!attr) {
        if (cudaError_t e = cudaFuncSetAttribute(gdn::k_delta_multi, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * (7 * 128 + 2) * 4 + 4096))
            return e;
        attr = true;
    }
    gdn::k_delta_multi<<<grid, gdn::THREADS, smem, st>>>((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)z, (const __nv_bfloat16*)b,
                                                      (const __nv_bfloat16*)a, (const __nv_bfloat16*)A_log, (const __nv_bfloat16*)dt_bias,
                                                      (const __nv_bfloat16*)norm_w, state, (__nv_bfloat16*)out, Hk, Hv, eps, T, n, ldz, ldba);
    return cudaGetLastError();
}
