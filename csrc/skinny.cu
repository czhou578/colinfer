// skinny.cu -- weight-streaming skinny GEMM on tensor cores, for each decode-time linear.
//
//   out[m, n] = sum_k x[m, k] * W[n, k]   (* scale, + residual),  M <= 16 rows of bf16 activations.
//
// The weights stay in the layout of the checkpoint, the same tensors that the CUTLASS prefill GEMM reads:
//   NVFP4  packed e2m1 [N, K/2] + e4m3 block scales [N, K/16] + fp32 global scale
//   FP8    e4m3 [N, K] + per-tensor or per-row fp32 scale
// The dequantization into bf16 is exact (e2m1 x e4m3 has <= 5 significant bits), and the product runs on
// mma.sync.m16n8k16 bf16 x bf16 -> fp32. Thus 16 activation rows cost about the same as 1 row: the weight stream, not the
// math, sets the speed of the kernel. All decode linears run here: plain decode (1-3 rows), the speculative verify (up to
// 16 rows) and the MTP drafter.
//
// Memory pattern. A warp owns 16 weight rows (two mma n-tiles: rows n0..n0+15, or the gate and up rows n0..n0+7 for
// SwiGLU). It walks K in chunks of 256 bytes per row. It prefetches the weights into registers one chunk ahead, with
// 16-byte loads that each cover 256 contiguous bytes of 2 rows.
//
// On this LPDDR5x, a 600 MB tensor streams at 235-238 GB/s with runs of >= 256 B. With 128 B it streams at ~225, and
// with 64 B at 190-210 GB/s. A direct load of mma fragments from global memory gives the last pattern, because each
// lane needs its own row. Thus the kernel transposes each chunk into fragment order through a small per-warp
// shared-memory scratch. It uses __syncwarp only, so the warps stream independently, and the L1 stays free for the
// activations. The NVFP4 block scales (1/8 of the bytes) come 4 chunks at a time, 128 B per row.
//
// Fragment order. Within each 64-wide group S of a chunk, lane t takes 16 consecutive k at 64S + 16q, four per mma
// k-step j. (g = t/4 is the weight row / output column that the lane serves, and q = t%4.) The physical k is
// 64S + 16q + 4j + e for fragment element e. This is a fixed permutation of the k order of the mma inside the step,
// applied to A and B alike. Thus the B fragments of a lane are 8 (FP4) or 16 (FP8) contiguous scratch bytes per group.
// Its A fragments are 32 contiguous bytes of each activation row, read directly from global memory (all warps read the
// same few KB, from L1 / L2). The order is fixed, so each output row is bit-identical for any M: the rows >= M are zeros
// and never touch the other rows.
//
// INT6 / INT5 (tools/int6_requant.py): w = (c - 32 | 16) * e4m3 block scale (16 k) * global scale. c is a 6- or 5-bit
// code in two planes. The low 4 bits are exactly like the nibbles of NVFP4, so they stream and land in the scratch the
// same way. The high 2 (1) bits are a [N, K/4] ([N, K/8]) plane, 128 (64) B per row per chunk.
//
// The dequantization: 0x4300 | c is the bf16 value 128 + c, and minus 160 (144) this is exactly q. The product with the
// bf16 block scale rounds once (q * s has up to 9 significant bits, and bf16 holds 8). This is ~0.2% relative, against
// the 2.2% / 4.2% quantization error of the format.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

#include "pdl.cuh"

namespace skinny {

enum Fmt { NVFP4 = 0, FP8 = 1, INT6 = 2, INT5 = 3 };

constexpr int WARPS = 4;  // warps per block (each 16 weight rows, or 8 gate + 8 up rows)

template <int F> struct Cfg {
    static constexpr bool BLK = F != FP8;                 // block-scaled (NVFP4, INT6): 4-bit plane, e4m3 scales per 16 k
    static constexpr int KC = BLK ? 512 : 256;            // k per chunk: 256 weight bytes per row
    static constexpr int GROUPS = KC / 64;
    static constexpr int ROWB = BLK ? 288 : 320;          // scratch row stride: conflict-free fragment reads
    static constexpr int SCH = 4;                         // chunks per scale load (128 B per row)
    static constexpr bool INT = F == INT6 || F == INT5;
    static constexpr int HB = F == INT6 ? 2 : 1;          // INT6 / INT5: high bits per code
    static constexpr int HC = 32 * HB * 2;                // high-plane bytes per row per chunk (512 k): 128 / 64
    static constexpr int ROWH = HC + 16;                  // its scratch row (+ pad: conflict-free fragment reads)
    static constexpr int SCRATCH = 16 * ROWB + (BLK ? 16 * 128 : 0) + (INT ? 16 * ROWH : 0);
};

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t h2_to_bf2(__half2 h) {
    const float2 f = __half22float2(h);
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&b);
}

// two e2m1 (one byte) times an f16 block scale -> bf16x2, exact
__device__ __forceinline__ uint32_t fp4x2_scaled(uint32_t byte, __half2 scale) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
    return h2_to_bf2(__hmul2(*reinterpret_cast<__half2*>(&r), scale));
}

// two e4m3 -> bf16x2, exact
__device__ __forceinline__ uint32_t fp8x2_bf(uint32_t two) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    return h2_to_bf2(*reinterpret_cast<__half2*>(&r));
}

// two (4 + HB)-bit codes (low nibbles in `lo`'s two halves, high fields in hb's bits 0..HB-1 and HB..2HB-1) times a
// bf16x2 block scale -> bf16x2. Codes are q + 2^(3 + HB).
template <int HB>
__device__ __forceinline__ uint32_t intx2_scaled(uint32_t lo, uint32_t hb, __nv_bfloat162 sc) {
    constexpr uint32_t M = (1u << HB) - 1;
    const uint32_t c0 = (lo & 15u) | ((hb & M) << 4), c1 = ((lo >> 4) & 15u) | (((hb >> HB) & M) << 4);
    const uint32_t w = 0x43004300u | c0 | (c1 << 16);
    const __nv_bfloat162 q = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&w), __float2bfloat162_rn(128.f + (1 << (3 + HB))));
    const __nv_bfloat162 r = __hmul2(q, sc);
    return *reinterpret_cast<const uint32_t*>(&r);
}

__device__ __forceinline__ __nv_bfloat162 e4m3_to_bf2(uint32_t byte) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)byte, __NV_E4M3);
    return __float2bfloat162_rn(__half2float(*reinterpret_cast<__half*>(&r)));
}

__device__ __forceinline__ __half2 e4m3_to_h2(uint32_t byte) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)byte, __NV_E4M3);
    const __half h = *reinterpret_cast<__half*>(&r);
    return __halves2half2(h, h);
}

template <typename T> __device__ __forceinline__ T zero_of() {
    if constexpr (sizeof(T) == 4) return 0.f;
    else return __float2bfloat16(0.f);
}

// out = (x W^T) * scale (+ residual). SWIGLU: out = silu((x W^T) * scale) * ((x W2^T) * scale2).
//
// Split-K: the work item of a warp is (tile of 16 rows, K range s of S). S depends only on the matrix shape, never on M,
// so the row results stay bit-identical for each M. With S > 1, each item writes fp32 partials. The last item of a tile
// to finish (a per-tile counter that resets itself) adds them in the order s = 0..S-1 and runs the epilogue.
template <int F, bool SWIGLU, typename OutT>
__global__ void __launch_bounds__(WARPS * 32) k_skinny(const __nv_bfloat16* __restrict__ x, int M, int N, int K,
                                                        const uint8_t* __restrict__ w, const uint8_t* __restrict__ sf,
                                                        const uint8_t* __restrict__ wh, float scale,
                                                        const float* __restrict__ row_scale,
                                                        const uint8_t* __restrict__ w2, const uint8_t* __restrict__ sf2, float scale2,
                                                        const __nv_bfloat16* __restrict__ residual, OutT* __restrict__ out,
                                                        int S, float* __restrict__ ws, int* __restrict__ cnt, const int* __restrict__ skip) {
    using C = Cfg<F>;
    PDL_TRIGGER();
    extern __shared__ __align__(16) uint8_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    constexpr int NT = 2;  // mma n-tiles per warp
    const int item = blockIdx.x * WARPS + warp, tile = item / S, split = item % S;
    const int n0 = tile * (SWIGLU ? 8 : 16);
    if (n0 >= N) return;
    if (skip && *skip) {  // a draft step after the drafter stopped (skinny_set_skip): no weight reads, zero output
        constexpr int COLS = SWIGLU ? 8 : 16;
        PDL_WAIT();  // out may be memory the predecessor still reads
        if (split == 0)
            for (int i = lane; i < M * COLS; i += 32)
                if (n0 + i % COLS < N) out[(size_t)(i / COLS) * N + n0 + i % COLS] = zero_of<OutT>();
        return;
    }
    uint8_t* wsc = smem + warp * C::SCRATCH;  // [16 rows][ROWB]: one chunk in fragment-friendly layout
    uint8_t* ssc = wsc + 16 * C::ROWB;        // NVFP4 / INT6: [16 rows][128]: block scales of 4 chunks
    uint8_t* hsc = ssc + 16 * 128;            // INT6: [16 rows][ROWH]: high-bit plane of the chunk
    const size_t rowbytes = C::BLK ? K / 2 : K;
    const int chunks = K / C::KC;
    // this item's chunk range; block-scaled ranges start on a scale load boundary (multiples of SCH chunks)
    const int unit = C::BLK ? C::SCH : 1, units = (chunks + unit - 1) / unit;
    const int cb = split * units / S * unit, ce = min((split + 1) * units / S * unit, chunks);
    // logical row r (0..15) of this warp: tile r / 8, column g = r % 8
    auto wrow = [&](int r) -> const uint8_t* {
        if constexpr (SWIGLU) return (r < 8 ? w : w2) + (size_t)(n0 + (r & 7)) * rowbytes;
        else return w + (size_t)min(n0 + r, N - 1) * rowbytes;  // rows past N (N % 16 == 8) read a valid row, unused
    };
    auto srow = [&](int r) -> const uint8_t* {
        if constexpr (SWIGLU) return (r < 8 ? sf : sf2) + (size_t)(n0 + (r & 7)) * (K / 16);
        else return sf + (size_t)min(n0 + r, N - 1) * (K / 16);
    };
    // prefetch registers: chunk = 16 rows x 256 B = 8 x 16 B per lane; instruction i covers rows 2i, 2i+1
    const int hr = lane >> 4, seg = lane & 15;
    constexpr int LD = NT * 4;  // 16-byte loads per lane per chunk
    uint4 v[LD], sv[4], hv[C::INT ? C::HC / 32 : 1];
    auto load_w = [&](int c) {
#pragma unroll
        for (int i = 0; i < LD; ++i) v[i] = __ldcs(reinterpret_cast<const uint4*>(wrow(2 * i + hr) + (size_t)c * 256) + seg);
        if constexpr (C::INT) {  // high bits of the chunk: 16 rows x HC bytes, 512 B per instruction
            constexpr int LPR = C::HC / 16, RPI = 32 / LPR;  // lanes per row, rows per instruction
#pragma unroll
            for (int i = 0; i < 16 / RPI; ++i)
                hv[i] = __ldcs(reinterpret_cast<const uint4*>(wh + (size_t)min(n0 + RPI * i + lane / LPR, N - 1) * (K / 16 * C::HB * 2) +
                                                              (size_t)c * C::HC) + lane % LPR);
        }
    };
    auto load_s = [&](int p) {  // scales of chunks 4p .. 4p+3: 16 rows x 128 B, 4 rows x 128 B per instruction
        if constexpr (C::BLK) {
            const int have = K / 16 - p * 128, col = (lane & 7) * 16;  // a multiple of 32 bytes are left in the row
#pragma unroll
            for (int i = 0; i < NT * 2; ++i)
                sv[i] = col < have ? __ldcs(reinterpret_cast<const uint4*>(srow(4 * i + (lane >> 3)) + p * 128 + col)) : make_uint4(0, 0, 0, 0);
        }
    };
    const __nv_bfloat16* xlo = g < M ? x + (size_t)g * K : nullptr;
    const __nv_bfloat16* xhi = g + 8 < M ? x + (size_t)(g + 8) * K : nullptr;
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    load_w(cb);
    load_s(cb / C::SCH);
    // Everything above reads only weights. Launched with programmatic serialization (PDL), this kernel runs while its
    // predecessor finishes: also pull the next chunks into L2, then wait for the predecessor before reading activations
    // or writing anything.
#pragma unroll
    for (int c = cb + 1; c < cb + 3; ++c)
        if (c < ce)
#pragma unroll
            for (int i = 0; i < 8; ++i) asm volatile("prefetch.global.L2 [%0];" ::"l"(wrow(2 * i + hr) + (size_t)c * 256 + seg * 16));
    PDL_WAIT();
    for (int c = cb; c < ce; ++c) {
        // registers -> scratch (the previous chunk's readers finished at the trailing __syncwarp)
#pragma unroll
        for (int i = 0; i < LD; ++i) *reinterpret_cast<uint4*>(wsc + (2 * i + hr) * C::ROWB + seg * 16) = v[i];
        if constexpr (C::INT) {
            constexpr int LPR = C::HC / 16, RPI = 32 / LPR;
#pragma unroll
            for (int i = 0; i < 16 / RPI; ++i) *reinterpret_cast<uint4*>(hsc + (RPI * i + lane / LPR) * C::ROWH + (lane % LPR) * 16) = hv[i];
        }
        if constexpr (C::BLK) {
            if (c % C::SCH == 0) {
#pragma unroll
                for (int i = 0; i < NT * 2; ++i) *reinterpret_cast<uint4*>(ssc + (4 * i + (lane >> 3)) * 128 + (lane & 7) * 16) = sv[i];
            }
        }
        __syncwarp();
        if (c + 1 < ce) load_w(c + 1);  // in flight while this chunk is multiplied
        if constexpr (C::BLK) {
            if (c % C::SCH == 0 && (c / C::SCH + 1) * C::SCH < ce) load_s(c / C::SCH + 1);
        }
#pragma unroll 2
        for (int S = 0; S < C::GROUPS; ++S) {
            const int xk = c * C::KC + 64 * S + 16 * q;  // this lane's 16 activations per row for the group
            uint4 al[2], ah[2];
            al[0] = al[1] = ah[0] = ah[1] = make_uint4(0, 0, 0, 0);
            if (xlo) { al[0] = __ldg(reinterpret_cast<const uint4*>(xlo + xk)); al[1] = __ldg(reinterpret_cast<const uint4*>(xlo + xk) + 1); }
            if (xhi) { ah[0] = __ldg(reinterpret_cast<const uint4*>(xhi + xk)); ah[1] = __ldg(reinterpret_cast<const uint4*>(xhi + xk) + 1); }
            const uint32_t* lw = reinterpret_cast<const uint32_t*>(al);
            const uint32_t* hw = reinterpret_cast<const uint32_t*>(ah);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint8_t* rowp = wsc + (8 * j + g) * C::ROWB;
                if constexpr (F == NVFP4) {
                    const uint2 wv = *reinterpret_cast<const uint2*>(rowp + 32 * S + 8 * q);
                    const __half2 sc = e4m3_to_h2(ssc[(8 * j + g) * 128 + (c % C::SCH) * 32 + 4 * S + q]);  // one block of 16 k
                    const uint32_t ww[2] = {wv.x, wv.y};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t two = (ww[jj >> 1] >> (16 * (jj & 1))) & 0xffff;
                        const uint32_t a[4] = {lw[2 * jj], hw[2 * jj], lw[2 * jj + 1], hw[2 * jj + 1]};
                        mma_bf16(acc[j], a, fp4x2_scaled(two & 0xff, sc), fp4x2_scaled(two >> 8, sc));
                    }
                } else if constexpr (C::INT) {
                    // this lane's 16 codes: 8 nibble bytes, and HB * 2 bytes of high bits (4 HB bits per k-step)
                    const uint2 wv = *reinterpret_cast<const uint2*>(rowp + 32 * S + 8 * q);
                    const uint8_t* hp = hsc + (8 * j + g) * C::ROWH + (16 * S + 4 * q) * C::HB / 2;
                    const uint32_t hb = C::HB == 2 ? *reinterpret_cast<const uint32_t*>(hp) : *reinterpret_cast<const uint16_t*>(hp);
                    const __nv_bfloat162 sc = e4m3_to_bf2(ssc[(8 * j + g) * 128 + (c % C::SCH) * 32 + 4 * S + q]);
                    const uint32_t ww[2] = {wv.x, wv.y};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t two = (ww[jj >> 1] >> (16 * (jj & 1))) & 0xffff;
                        const uint32_t h = (hb >> (4 * C::HB * jj)) & ((1u << (4 * C::HB)) - 1);
                        const uint32_t a[4] = {lw[2 * jj], hw[2 * jj], lw[2 * jj + 1], hw[2 * jj + 1]};
                        mma_bf16(acc[j], a, intx2_scaled<C::HB>(two & 0xff, h, sc), intx2_scaled<C::HB>(two >> 8, h >> (2 * C::HB), sc));
                    }
                } else {
                    const uint4 wv = *reinterpret_cast<const uint4*>(rowp + 64 * S + 16 * q);
                    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {lw[2 * jj], hw[2 * jj], lw[2 * jj + 1], hw[2 * jj + 1]};
                        mma_bf16(acc[j], a, fp8x2_bf(ww[jj] & 0xffff), fp8x2_bf(ww[jj] >> 16));
                    }
                }
            }
        }
        __syncwarp();
    }
    if (S > 1) {  // partials -> workspace [S][NT][M][N] (plain: tile j lives in columns n0 + 8j..; SwiGLU: gate / up)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int jj = SWIGLU ? j : 0, col = n0 + (SWIGLU ? 0 : 8 * j) + 2 * q;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = g + 8 * h;
                if (m < M) *reinterpret_cast<float2*>(ws + (((size_t)split * NT + jj) * M + m) * N + col) = make_float2(acc[j][2 * h], acc[j][2 * h + 1]);
            }
        }
        __threadfence();
        __syncwarp();
        int last = 0;
        if (lane == 0) last = atomicAdd(cnt + tile, 1) == S - 1;
        if (!__shfl_sync(0xffffffffu, last, 0)) return;
        __threadfence();
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int jj = SWIGLU ? j : 0, col = n0 + (SWIGLU ? 0 : 8 * j) + 2 * q;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = g + 8 * h;
                if (m >= M) continue;
                float a0 = 0.f, a1 = 0.f;
                for (int t = 0; t < S; ++t) {  // fixed order: the result does not depend on which item finished last
                    const float2 v = __ldcg(reinterpret_cast<const float2*>(ws + (((size_t)t * NT + jj) * M + m) * N + col));
                    a0 += v.x;
                    a1 += v.y;
                }
                acc[j][2 * h] = a0;
                acc[j][2 * h + 1] = a1;
            }
        }
        if (lane == 0) cnt[tile] = 0;
    }
    // epilogue: acc[j][0..1] = rows g, cols 2q + {0,1} of tile j; acc[j][2..3] = row g + 8
#pragma unroll
    for (int j = 0; j < (SWIGLU ? 1 : NT); ++j) {
        if (n0 + 8 * j >= N) break;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = g + 8 * h;
            if (m >= M) continue;
            const int n = n0 + 8 * j + 2 * q;
            float v0, v1;
            if constexpr (SWIGLU) {
                const float g0 = acc[0][2 * h] * scale, g1 = acc[0][2 * h + 1] * scale;
                const float u0 = acc[1][2 * h] * scale2, u1 = acc[1][2 * h + 1] * scale2;
                v0 = g0 / (1.f + __expf(-g0)) * u0;
                v1 = g1 / (1.f + __expf(-g1)) * u1;
            } else {
                const float s0 = row_scale ? row_scale[n] : scale, s1 = row_scale ? row_scale[n + 1] : scale;
                v0 = acc[j][2 * h] * s0;
                v1 = acc[j][2 * h + 1] * s1;
            }
            if (residual) {
                const __nv_bfloat162 r = *reinterpret_cast<const __nv_bfloat162*>(residual + (size_t)m * N + n);
                v0 += __bfloat162float(r.x);
                v1 += __bfloat162float(r.y);
            }
            if constexpr (sizeof(OutT) == 4) {
                *reinterpret_cast<float2*>(out + (size_t)m * N + n) = make_float2(v0, v1);
            } else {
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * N + n) = __floats2bfloat162_rn(v0, v1);
            }
        }
    }
}

constexpr int ITEMS = 2048;  // warp work items per launch that keep every SM streaming to the end of the grid

// Per-tile split counters, zeroed once and reset by each tile's last item. Allocated on the first (eager) call,
// before any CUDA graph capture.
static int* g_cnt = nullptr;
constexpr int CNT = 1 << 17;

// Launches made while the skip flag is on contain the flag (skinny_set_skip). The draft steps of the speculative cycle
// after the first one read a device int. The drafter sets this int when the verify is unlikely to accept its chain
// (engine/spec/mtp.py). A kernel must write the int. That kernel must complete before the stream predecessor of the GEMM
// starts, because the GEMM reads the flag before griddepcontrol.wait.
static const int* g_skip = nullptr;

// Splits for a shape: enough work items (ITEMS) to keep every SM streaming to the end of the grid, at least one
// scale unit (4 chunks; 1 for FP8) per split.
inline int splits_for(int F, bool swiglu, int N, int K) {
    const int tiles = (N + (swiglu ? 8 : 16) - 1) / (swiglu ? 8 : 16);
    const int kc = F == FP8 ? 256 : 512, unit = F == FP8 ? 1 : 4;
    const int units = (K / kc + unit - 1) / unit;
    const int maxs = units * unit;
    const int S = (ITEMS + tiles - 1) / tiles;
    return S < 1 ? 1 : (S > maxs ? (maxs < 1 ? 1 : maxs) : S);
}

template <int F, bool SWIGLU, typename OutT>
cudaError_t launch(const void* x, int M, int N, int K, const void* w, const void* sf, float scale, const float* row_scale, const void* w2,
                   const void* sf2, float scale2, const void* residual, void* out, float* ws, cudaStream_t st, const void* wh = nullptr) {
    using C = Cfg<F>;
    if (M < 1 || M > 16 || N % 8 || K % C::KC) return cudaErrorInvalidValue;
    const int rows = SWIGLU ? 8 : 16, tiles = (N + rows - 1) / rows;
    const int S = ws ? splits_for(F, SWIGLU, N, K) : 1;
    if (S > 1 && !g_cnt) {
        if (cudaError_t e = cudaMalloc(&g_cnt, CNT * sizeof(int))) return e;
        if (cudaError_t e = cudaMemset(g_cnt, 0, CNT * sizeof(int))) return e;
    }
    if (tiles > CNT) return cudaErrorInvalidValue;
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((tiles * S + WARPS - 1) / WARPS);
    lc.blockDim = dim3(WARPS * 32);
    lc.dynamicSmemBytes = WARPS * C::SCRATCH;
    lc.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
    return cudaLaunchKernelEx(&lc, k_skinny<F, SWIGLU, OutT>, (const __nv_bfloat16*)x, M, N, K, (const uint8_t*)w, (const uint8_t*)sf,
                              (const uint8_t*)wh, scale,
                              row_scale, (const uint8_t*)w2, (const uint8_t*)sf2, scale2, (const __nv_bfloat16*)residual, (OutT*)out, S, ws,
                              g_cnt, g_skip);
}

}  // namespace skinny

void skinny_set_skip(const int* p) { skinny::g_skip = p; }

// workspace floats a call needs (0: no split); the caller passes a buffer of that size as ws
int skinny_ws_floats(int fmt, bool swiglu, int M, int N, int K) {
    const int S = skinny::splits_for(fmt, swiglu, N, K);
    return S > 1 ? S * 2 * M * N : 0;
}

cudaError_t launch_skinny_nvfp4(const void* x, const void* w, const void* sf, float gscale, const void* residual, void* out, bool out_fp32,
                                int M, int N, int K, float* ws, cudaStream_t st) {
    using namespace skinny;
    return out_fp32 ? launch<NVFP4, false, float>(x, M, N, K, w, sf, gscale, nullptr, nullptr, nullptr, 0.f, residual, out, ws, st)
                    : launch<NVFP4, false, __nv_bfloat16>(x, M, N, K, w, sf, gscale, nullptr, nullptr, nullptr, 0.f, residual, out, ws, st);
}

cudaError_t launch_skinny_swiglu(const void* x, const void* wg, const void* sg, float gg, const void* wu, const void* su, float gu, void* out,
                                 int M, int N, int K, float* ws, cudaStream_t st) {
    using namespace skinny;
    return launch<NVFP4, true, __nv_bfloat16>(x, M, N, K, wg, sg, gg, nullptr, wu, su, gu, nullptr, out, ws, st);
}

// INT6 / INT5 and FP8 weights only feed bf16 activations (fp32 output: NVFP4 only, for logits)
cudaError_t launch_skinny_int(int bits, const void* x, const void* wlo, const void* whi, const void* sf, float gscale, const void* residual,
                              void* out, int M, int N, int K, float* ws, cudaStream_t st) {
    using namespace skinny;
    if (bits == 6) return launch<INT6, false, __nv_bfloat16>(x, M, N, K, wlo, sf, gscale, nullptr, nullptr, nullptr, 0.f, residual, out, ws, st, whi);
    if (bits == 5) return launch<INT5, false, __nv_bfloat16>(x, M, N, K, wlo, sf, gscale, nullptr, nullptr, nullptr, 0.f, residual, out, ws, st, whi);
    return cudaErrorInvalidValue;
}

cudaError_t launch_skinny_fp8(const void* x, const void* w, float scale, const float* row_scale, const void* residual, void* out, int M, int N,
                              int K, float* ws, cudaStream_t st) {
    using namespace skinny;
    return launch<FP8, false, __nv_bfloat16>(x, M, N, K, w, nullptr, scale, row_scale, nullptr, nullptr, 0.f, residual, out, ws, st);
}
