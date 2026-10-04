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

namespace gdn {

constexpr int DK = 128, DV = 128, THREADS = 512;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// active (optional, int32 [B]): slots with active[b] == 0 keep their conv state (idle or mid-prefill slots)
__global__ void k_conv(const __nv_bfloat16* __restrict__ mixed, __nv_bfloat16* __restrict__ conv_state, const __nv_bfloat16* __restrict__ w,
                       __nv_bfloat16* __restrict__ out, int C, const int* __restrict__ active) {
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
// mixed / qkv: [B, T, C]; z: [B, T, Hv*128]; b, a: [B, T, Hv]; n: int32 [B] tokens to commit (1..T).
// ---------------------------------------------------------------------------------------------
__global__ void k_conv_multi(const __nv_bfloat16* __restrict__ mixed, const __nv_bfloat16* __restrict__ conv_state,
                             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int T, int C) {
    const int bt = blockIdx.y, b = bt / T, t = bt % T, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    float xs[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        xs[k] = tt >= 0 ? __bfloat162float(mixed[((size_t)b * T + tt) * C + c]) : __bfloat162float(cs[3 + tt]);
    }
    const __nv_bfloat16* wc = w + (size_t)c * 4;
    float acc = __bfloat162float(wc[0]) * xs[0];
    acc = fmaf(__bfloat162float(wc[1]), xs[1], acc);
    acc = fmaf(__bfloat162float(wc[2]), xs[2], acc);
    acc = fmaf(__bfloat162float(wc[3]), xs[3], acc);
    out[((size_t)b * T + t) * C + c] = __float2bfloat16(silu(bf(acc)));
}

__global__ void k_conv_commit(const __nv_bfloat16* __restrict__ mixed, __nv_bfloat16* __restrict__ conv_state, const int* __restrict__ n_ptr,
                              int T, int C) {
    const int b = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = n_ptr[b];
    __nv_bfloat16* cs = conv_state + ((size_t)b * C + c) * 3;
    __nv_bfloat16 hist[3 + 16];
    hist[0] = cs[0]; hist[1] = cs[1]; hist[2] = cs[2];
    for (int t = 0; t < n; ++t) hist[3 + t] = mixed[((size_t)b * T + t) * C + c];
    cs[0] = hist[n]; cs[1] = hist[n + 1]; cs[2] = hist[n + 2];
}

__global__ void __launch_bounds__(THREADS) k_delta_multi(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ z,
                                                          const __nv_bfloat16* __restrict__ bvec, const __nv_bfloat16* __restrict__ avec,
                                                          const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                                                          const __nv_bfloat16* __restrict__ norm_w, float* __restrict__ state,
                                                          __nv_bfloat16* __restrict__ out, int Hk, int Hv, float eps, int T,
                                                          const int* __restrict__ n_ptr) {
    const int b = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kh = h / (Hv / Hk), C = 2 * Hk * DK + Hv * DV;
    const bool commit = n_ptr != nullptr;
    const int steps = commit ? n_ptr[b] : T;
    __shared__ float qs[DK], ks[DK], vs[DV], red[4], colred[4][DV];
    const int rg = warp >> 2, col = ((warp & 3) << 5) + lane;
    float* S = state + ((size_t)b * Hv + h) * DK * DV + (size_t)(rg * 32) * DV + col;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = S[(size_t)r * DV];
    const float A = __expf(__bfloat162float(A_log[h])), dtb = __bfloat162float(dt_bias[h]);
    for (int t = 0; t < steps; ++t) {
        const __nv_bfloat16* src = qkv + ((size_t)b * T + t) * C;
        __syncthreads();
        if (tid < DK) {
            qs[tid] = __bfloat162float(src[kh * DK + tid]);
            ks[tid] = __bfloat162float(src[Hk * DK + kh * DK + tid]);
            vs[tid] = __bfloat162float(src[2 * Hk * DK + h * DV + tid]);
        }
        __syncthreads();
        const float qq = block_sum_128(tid < DK ? qs[tid] * qs[tid] : 0.f, red, tid);
        const float kk = block_sum_128(tid < DK ? ks[tid] * ks[tid] : 0.f, red, tid);
        if (tid < DK) {
            qs[tid] = qs[tid] * rsqrtf(qq + 1e-6f) * rsqrtf((float)DK);
            ks[tid] = ks[tid] * rsqrtf(kk + 1e-6f);
        }
        __syncthreads();
        const size_t bt = (size_t)b * T + t;
        const float beta = bf(1.f / (1.f + __expf(-__bfloat162float(bvec[bt * Hv + h]))));
        const float av = __bfloat162float(avec[bt * Hv + h]) + dtb;
        const float softplus = av > 20.f ? av : log1pf(__expf(av));
        const float decay = __expf(-A * softplus);
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r] = s[r] * decay;
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
            part = fmaf(s[r], qs[rg * 32 + r], part);
        }
        if (commit) continue;
        colred[rg][col] = part;
        __syncthreads();
        float o = 0.f;
        if (tid < DV) o = bf(colred[0][tid] + colred[1][tid] + colred[2][tid] + colred[3][tid]);
        const float var = block_sum_128(tid < DV ? o * o : 0.f, red, tid) / DV;
        if (tid < DV) {
            const float xn = bf(o * rsqrtf(var + eps));
            const float y = bf(__bfloat162float(norm_w[tid]) * xn);
            const float g = __bfloat162float(z[bt * Hv * DV + h * DV + tid]);
            out[bt * Hv * DV + h * DV + tid] = __float2bfloat16(y * silu(g));
        }
    }
    if (commit) {
#pragma unroll
        for (int r = 0; r < 32; ++r) S[(size_t)r * DV] = s[r];
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

cudaError_t launch_gdn_conv_multi(const void* mixed, const void* conv_state, const void* w, void* out, int B, int T, int C, cudaStream_t st) {
    dim3 grid((C + 255) / 256, B * T);
    gdn::k_conv_multi<<<grid, 256, 0, st>>>((const __nv_bfloat16*)mixed, (const __nv_bfloat16*)conv_state, (const __nv_bfloat16*)w,
                                            (__nv_bfloat16*)out, T, C);
    return cudaGetLastError();
}

cudaError_t launch_gdn_conv_commit(const void* mixed, void* conv_state, const int* n, int B, int T, int C, cudaStream_t st) {
    if (T > 16) return cudaErrorInvalidValue;
    dim3 grid((C + 255) / 256, B);
    gdn::k_conv_commit<<<grid, 256, 0, st>>>((const __nv_bfloat16*)mixed, (__nv_bfloat16*)conv_state, n, T, C);
    return cudaGetLastError();
}

cudaError_t launch_gdn_delta_multi(const void* qkv, const void* z, const void* b, const void* a, const void* A_log, const void* dt_bias,
                                   const void* norm_w, float* state, void* out, int B, int Hk, int Hv, float eps, int T, const int* n,
                                   cudaStream_t st) {
    dim3 grid(Hv, B);
    gdn::k_delta_multi<<<grid, gdn::THREADS, 0, st>>>((const __nv_bfloat16*)qkv, (const __nv_bfloat16*)z, (const __nv_bfloat16*)b,
                                                      (const __nv_bfloat16*)a, (const __nv_bfloat16*)A_log, (const __nv_bfloat16*)dt_bias,
                                                      (const __nv_bfloat16*)norm_w, state, (__nv_bfloat16*)out, Hk, Hv, eps, T, n);
    return cudaGetLastError();
}
