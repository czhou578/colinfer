// prefill_ops.cu -- bandwidth-bound glue of the prefill path, fused (PLAN.md 4.4).
//   k_fp8_quant         bf16 [M, K] -> e4m3 [M, K] = sat(x / in_scale)            (W8A8 inputs)
//   k_silu_mul_quant    gu bf16 [M, 2I] (gate | up) -> h = bf16(bf16(silu(g)) * u) quantized to NVFP4:
//                       packed e2m1 [M, I/2] + CUTLASS-swizzled e4m3 scales     (down_proj input)
//   k_causal_conv_silu  mixed bf16 [T, C] (token-major), state bf16 [C, 3] -> bf16(silu(bf16(conv))) (_l2: + q / k L2 norm)
//   k_gated_rmsnorm     o, z bf16 [N, 128] -> bf16(bf16(w * bf16(rmsnorm(o))) * silu(z))
// Rounding points follow the PyTorch reference (engine/model/qwen35.py).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pf {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }  // precise: must round like torch
__device__ __forceinline__ uint32_t e2m1_code(float a) {
    return (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
}
__device__ __forceinline__ size_t sf_offset(int r, int kb, int kb4) {
    return ((size_t)(r >> 7) * kb4 + (kb >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (kb & 3);
}

__global__ void k_fp8_quant(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out, size_t n8, float inv_scale) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = __uint_as_float(u[j] << 16) * inv_scale, b = __uint_as_float(u[j] & 0xffff0000u) * inv_scale;
        const uint32_t pa = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3), pb = __nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3);
        o[j >> 1] |= (pa | (pb << 8)) << (16 * (j & 1));
    }
    reinterpret_cast<uint2*>(out)[i] = make_uint2(o[0], o[1]);
}

// one thread per 16-element block of h
__global__ void k_silu_mul_quant(const __nv_bfloat16* __restrict__ gu, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, int I,
                                 float inv_in_scale, float in_scale) {
    const int KB = I / 16, kb4 = (KB + 3) / 4;
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * KB) return;
    const int r = idx / KB, kb = idx % KB;
    const uint4* gp = reinterpret_cast<const uint4*>(gu + (size_t)r * 2 * I + kb * 16);
    const uint4* up = reinterpret_cast<const uint4*>(gu + (size_t)r * 2 * I + I + kb * 16);
    const uint4 g0 = gp[0], g1 = gp[1], u0 = up[0], u1 = up[1];
    const uint32_t gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, uw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    float h[16], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float ga = __uint_as_float(gw[i] << 16), gb = __uint_as_float(gw[i] & 0xffff0000u);
        const float ua = __uint_as_float(uw[i] << 16), ub = __uint_as_float(uw[i] & 0xffff0000u);
        h[2 * i] = bf(bf(silu(ga)) * ua);
        h[2 * i + 1] = bf(bf(silu(gb)) * ub);
        amax = fmaxf(amax, fmaxf(fabsf(h[2 * i]), fabsf(h[2 * i + 1])));
    }
    const __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(amax / 6.f * inv_in_scale, __NV_SATFINITE, __NV_E4M3);
    __half_raw hr = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
    const float sfv = __half2float(*reinterpret_cast<__half*>(&hr));
    const float os = sfv != 0.f ? 1.f / (sfv * in_scale) : 0.f;
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s = h[i] * os;
        packed[i >> 3] |= (e2m1_code(fabsf(s)) | (s < 0.f ? 8u : 0u)) << (4 * (i & 7));
    }
    *reinterpret_cast<uint2*>(q + (size_t)r * (I / 2) + kb * 8) = make_uint2(packed[0], packed[1]);
    sf[sf_offset(r, kb, kb4)] = sfb;
}

// one thread per (t, channel pair)
// output channels [0, c1) -> o0 [T, c1], [c1, c2) -> o1 [T, c2 - c1], [c2, C) -> o2 [T, C - c2] (all contiguous)
__global__ void k_causal_conv_silu(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ state,
                                   const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ o0, __nv_bfloat16* __restrict__ o1,
                                   __nv_bfloat16* __restrict__ o2, int c1, int c2, int T, int C) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int C2 = C / 2;
    if (idx >= (size_t)T * C2) return;
    const int t = idx / C2, c = (idx % C2) * 2;
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        float x0, x1;
        if (tt >= 0) {
            const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)tt * ldx + c);
            x0 = __low2float(v); x1 = __high2float(v);
        } else {
            x0 = __bfloat162float(state[(size_t)c * 3 + (3 + tt)]);
            x1 = __bfloat162float(state[(size_t)(c + 1) * 3 + (3 + tt)]);
        }
        acc0 = fmaf(__bfloat162float(w[(size_t)c * 4 + k]), x0, acc0);
        acc1 = fmaf(__bfloat162float(w[(size_t)(c + 1) * 4 + k]), x1, acc1);
    }
    __nv_bfloat16* dst = c < c1 ? o0 + (size_t)t * c1 + c : c < c2 ? o1 + (size_t)t * (c2 - c1) + (c - c1) : o2 + (size_t)t * (C - c2) + (c - c2);
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(silu(bf(acc0)), silu(bf(acc1)));
}

// One warp per (t, 128-channel head): lane l owns channels 4l .. 4l+3. l2_eps >= 0: the q and k outputs (channels < c2) are
// also L2-normalized per head, as FLA's l2norm would (x / sqrt(sum x^2 + eps) on the bf16 values), so
// chunk_gated_delta_rule runs with use_qk_l2norm_in_kernel=False and its separate l2norm pass goes away.
__global__ void k_causal_conv_silu_l2(const __nv_bfloat16* __restrict__ x, int ldx, const __nv_bfloat16* __restrict__ state,
                                      const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ o0, __nv_bfloat16* __restrict__ o1,
                                      __nv_bfloat16* __restrict__ o2, int c1, int c2, int T, int C, float l2_eps) {
    const size_t wi = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, G = C / 128;
    if (wi >= (size_t)T * G) return;
    const int t = wi / G, c = (wi % G) * 128 + lane * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        float xv[4];
        if (tt >= 0) {
            const uint2 v = *reinterpret_cast<const uint2*>(x + (size_t)tt * ldx + c);
            xv[0] = __uint_as_float(v.x << 16); xv[1] = __uint_as_float(v.x & 0xffff0000u);
            xv[2] = __uint_as_float(v.y << 16); xv[3] = __uint_as_float(v.y & 0xffff0000u);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = __bfloat162float(state[(size_t)(c + i) * 3 + (3 + tt)]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(__bfloat162float(w[(size_t)(c + i) * 4 + k]), xv[i], acc[i]);
    }
    float y[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[i] = bf(silu(bf(acc[i])));
        ss += y[i] * y[i];
    }
    if (l2_eps >= 0.f && c < c2) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = 1.f / sqrtf(ss + l2_eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) y[i] *= r;
    }
    __nv_bfloat16* dst = c < c1 ? o0 + (size_t)t * c1 + c : c < c2 ? o1 + (size_t)t * (c2 - c1) + (c - c1) : o2 + (size_t)t * (C - c2) + (c - c2);
    const __nv_bfloat162 a = __floats2bfloat162_rn(y[0], y[1]), b = __floats2bfloat162_rn(y[2], y[3]);
    *reinterpret_cast<uint2*>(dst) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

// one warp per row of 128; row = token * heads + head. z row (token, head) lives at z + token * ldz + head * 128.
__global__ void k_gated_rmsnorm(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ z, int ldz, int heads,
                                const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int N, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    float x[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = __bfloat162float(o[(size_t)row * 128 + lane * 4 + i]);
        ss = fmaf(x[i], x[i], ss);
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);
    const float r = rsqrtf(ss / 128.f + eps);
    const __nv_bfloat16* zr = z + (size_t)(row / heads) * ldz + (row % heads) * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float xn = bf(x[i] * r);
        const float y = bf(__bfloat162float(w[lane * 4 + i]) * xn);
        const float g = __bfloat162float(zr[lane * 4 + i]);
        out[(size_t)row * 128 + lane * 4 + i] = __float2bfloat16(y * silu(g));
    }
}

// One block (256 threads) per row of K (K % 8 == 0). x_new = bf16(x + y) (y optional), n = bf16(rmsnorm(x_new) * (1 + w)).
// Outputs (each optional): x_out (x_new), n_out (bf16), q4/sf4 (NVFP4 of n, static in_scale, swizzled scales),
// q8 (e4m3 of n / in_scale8). Thread t owns 8-element chunks t, t + 256, ...; the two chunks of a 16-element
// NVFP4 block are owned by neighbouring threads (t, t ^ 1), which exchange their amax with one shuffle.
__global__ void __launch_bounds__(256) k_add_rmsnorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y,
                                                      const __nv_bfloat16* __restrict__ w, float eps, int K, __nv_bfloat16* __restrict__ x_out,
                                                      __nv_bfloat16* __restrict__ n_out, uint8_t* __restrict__ q4, uint8_t* __restrict__ sf4,
                                                      float in_scale4, uint8_t* __restrict__ q8, float in_scale8) {
    const int r = blockIdx.x, tid = threadIdx.x;
    const size_t base = (size_t)r * K;
    const int chunks = K / 8;
    float ss = 0.f;
    for (int j = tid; j < chunks; j += 256) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x + base + j * 8);
        uint32_t u[4] = {xv.x, xv.y, xv.z, xv.w};
        if (y) {
            const uint4 yv = *reinterpret_cast<const uint4*>(y + base + j * 8);
            const uint32_t yu[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __nv_bfloat162 s2 = __floats2bfloat162_rn(__uint_as_float(u[i] << 16) + __uint_as_float(yu[i] << 16),
                                                                __uint_as_float(u[i] & 0xffff0000u) + __uint_as_float(yu[i] & 0xffff0000u));
                u[i] = *reinterpret_cast<const uint32_t*>(&s2);
            }
            if (x_out) *reinterpret_cast<uint4*>(x_out + base + j * 8) = make_uint4(u[0], u[1], u[2], u[3]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a = __uint_as_float(u[i] << 16), b = __uint_as_float(u[i] & 0xffff0000u);
            ss = fmaf(a, a, fmaf(b, b, ss));
        }
    }
    __shared__ float red[8];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float rs = rsqrtf(tot / K + eps);
    const __nv_bfloat16* src = y ? (x_out ? x_out : nullptr) : x;
    const int KB = K / 16, kb4 = (KB + 3) / 4;
    const int iters = (chunks + 255) / 256;
    for (int it = 0; it < iters; ++it) {
        const int j = it * 256 + tid;
        const bool active = j < chunks;
        float nv[8], amax = 0.f;
        if (active) {
            uint32_t u[4];
            if (src) {
                const uint4 xv = *reinterpret_cast<const uint4*>(src + base + j * 8);
                u[0] = xv.x; u[1] = xv.y; u[2] = xv.z; u[3] = xv.w;
            } else {  // y given but x_out not: recompute the sum
                const uint4 xv = *reinterpret_cast<const uint4*>(x + base + j * 8), yv = *reinterpret_cast<const uint4*>(y + base + j * 8);
                const uint32_t a[4] = {xv.x, xv.y, xv.z, xv.w}, b[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat162 s2 = __floats2bfloat162_rn(__uint_as_float(a[i] << 16) + __uint_as_float(b[i] << 16),
                                                                    __uint_as_float(a[i] & 0xffff0000u) + __uint_as_float(b[i] & 0xffff0000u));
                    u[i] = *reinterpret_cast<const uint32_t*>(&s2);
                }
            }
            const uint4 wv = *reinterpret_cast<const uint4*>(w + j * 8);
            const uint32_t wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                nv[2 * i] = bf(__uint_as_float(u[i] << 16) * rs * (1.f + __uint_as_float(wu[i] << 16)));
                nv[2 * i + 1] = bf(__uint_as_float(u[i] & 0xffff0000u) * rs * (1.f + __uint_as_float(wu[i] & 0xffff0000u)));
                amax = fmaxf(amax, fmaxf(fabsf(nv[2 * i]), fabsf(nv[2 * i + 1])));
            }
            if (n_out) {
                uint32_t o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat162 p = __floats2bfloat162_rn(nv[2 * i], nv[2 * i + 1]);
                    o[i] = *reinterpret_cast<const uint32_t*>(&p);
                }
                *reinterpret_cast<uint4*>(n_out + base + j * 8) = make_uint4(o[0], o[1], o[2], o[3]);
            }
            if (q8) {
                uint32_t o[2] = {0, 0};
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    o[i >> 2] |= (uint32_t)__nv_cvt_float_to_fp8(nv[i] / in_scale8, __NV_SATFINITE, __NV_E4M3) << (8 * (i & 3));
                *reinterpret_cast<uint2*>(q8 + base + j * 8) = make_uint2(o[0], o[1]);
            }
        }
        if (q4) {  // all threads of the warp take part in the shuffle
            const float am = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            if (active) {
                const __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(am / 6.f / in_scale4, __NV_SATFINITE, __NV_E4M3);
                __half_raw hr = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
                const float sfv = __half2float(*reinterpret_cast<__half*>(&hr));
                const float os = sfv != 0.f ? 1.f / (sfv * in_scale4) : 0.f;
                uint32_t packed = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = nv[i] * os;
                    packed |= (e2m1_code(fabsf(v)) | (v < 0.f ? 8u : 0u)) << (4 * i);
                }
                *reinterpret_cast<uint32_t*>(q4 + (size_t)r * (K / 2) + j * 4) = packed;
                if ((j & 1) == 0) sf4[sf_offset(r, j >> 1, kb4)] = sfb;
            }
        }
    }
}

// attention output: out = e4m3( bf16(o * sigmoid(gate)) / in_scale ); o [T, H*D] contiguous, gate rows strided (ldg)
__global__ void k_gate_fp8(const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ gate, int ldg, int H, int D,
                           uint8_t* __restrict__ out, size_t n, float inv_scale) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int HD = H * D;
    const size_t t = i / HD;
    const int hd = i % HD, h = hd / D, d = hd % D;
    const float g = __bfloat162float(gate[t * ldg + (size_t)h * 2 * D + D + d]);
    const float v = bf(bf(__bfloat162float(o[i]) * bf(1.f / (1.f + expf(-g)))));
    out[i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace pf

cudaError_t launch_fp8_quant(const void* x, void* out, size_t n, float scale, cudaStream_t st) {
    if (n % 8) return cudaErrorInvalidValue;
    const size_t n8 = n / 8;
    pf::k_fp8_quant<<<(n8 + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, (uint8_t*)out, n8, 1.f / scale);
    return cudaGetLastError();
}

size_t nvfp4_sf_bytes(int R, int K);
cudaError_t launch_silu_mul_quant(const void* gu, void* q, void* sf, int M, int I, float in_scale, cudaStream_t st) {
    if (I % 16) return cudaErrorInvalidValue;
    cudaMemsetAsync(sf, 0, nvfp4_sf_bytes(M, I), st);
    const size_t n = (size_t)M * (I / 16);
    pf::k_silu_mul_quant<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)gu, (uint8_t*)q, (uint8_t*)sf, M, I, 1.f / in_scale, in_scale);
    return cudaGetLastError();
}

cudaError_t launch_causal_conv_silu(const void* x, int ldx, const void* state, const void* w, void* o0, void* o1, void* o2, int c1, int c2, int T,
                                    int C, float l2_eps, cudaStream_t st) {
    if (C % 128 == 0 && c1 % 128 == 0 && c2 % 128 == 0 && ldx % 4 == 0) {
        const size_t threads = (size_t)T * (C / 128) * 32;
        pf::k_causal_conv_silu_l2<<<(threads + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)state,
                                                                         (const __nv_bfloat16*)w, (__nv_bfloat16*)o0, (__nv_bfloat16*)o1,
                                                                         (__nv_bfloat16*)o2, c1, c2, T, C, l2_eps);
        return cudaGetLastError();
    }
    if (l2_eps >= 0.f || C % 2 || ldx % 2 || c1 % 2 || c2 % 2) return cudaErrorInvalidValue;
    const size_t n = (size_t)T * (C / 2);
    pf::k_causal_conv_silu<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)state, (const __nv_bfloat16*)w,
                                                            (__nv_bfloat16*)o0, (__nv_bfloat16*)o1, (__nv_bfloat16*)o2, c1, c2, T, C);
    return cudaGetLastError();
}

cudaError_t launch_gated_rmsnorm(const void* o, const void* z, int ldz, int heads, const void* w, void* out, int N, float eps, cudaStream_t st) {
    pf::k_gated_rmsnorm<<<(N + 7) / 8, 256, 0, st>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)z, ldz, heads, (const __nv_bfloat16*)w,
                                                     (__nv_bfloat16*)out, N, eps);
    return cudaGetLastError();
}

cudaError_t launch_add_rmsnorm(const void* x, const void* y, const void* w, float eps, int M, int K, void* x_out, void* n_out, void* q4, void* sf4,
                               float in_scale4, void* q8, float in_scale8, cudaStream_t st) {
    if (K % 16) return cudaErrorInvalidValue;
    if (sf4) cudaMemsetAsync(sf4, 0, nvfp4_sf_bytes(M, K), st);
    pf::k_add_rmsnorm<<<M, 256, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)y, (const __nv_bfloat16*)w, eps, K, (__nv_bfloat16*)x_out,
                                         (__nv_bfloat16*)n_out, (uint8_t*)q4, (uint8_t*)sf4, in_scale4, (uint8_t*)q8, in_scale8);
    return cudaGetLastError();
}

cudaError_t launch_gate_fp8(const void* o, const void* gate, int ldg, int T, int H, int D, void* out, float scale, cudaStream_t st) {
    const size_t n = (size_t)T * H * D;
    pf::k_gate_fp8<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)gate, ldg, H, D, (uint8_t*)out, n, 1.f / scale);
    return cudaGetLastError();
}
