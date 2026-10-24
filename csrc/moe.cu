// moe.cu -- decode MoE kernels for qwen4_exp (docs/history/qwen4exp_plan.md section 5.2): expert routing, the expert SwiGLU and
// down projections on the skinny-GEMM streaming pattern (csrc/skinny.cu), and the fixed-order combine. The bindings are in
// moe_bind.cpp; engine/model/moe_kernels.py chains them. Tests: tests/test_moe.py; bench: bench/moe_bench.py.
//
//   moe_route    logits bf16 [M, NL] (E router columns, then the shared-expert gate at column E) -> the top-10 experts per row
//   moe_gate_up  h  bf16 [10M, I] = silu(x W_gate^T * gs_g) * (x W_up^T * gs_u), one row per (token, slot) pair
//   moe_down     yp f32  [10M, K] = (h W_down^T) * gs_d
//   moe_combine  y  bf16 [M, K]   = bf16(sum_{k = 0..9} w[m, k] * yp[10m + k] + sg[m] * shared[m])
//
// Pairs. Pair p = 10m + k is slot k of row m; ids, w, h and yp are in pair order. An expert occurs at most once in a row, so an
// expert has at most M pairs in a launch.
//
// Weights. Per expert e: w_gu u8 [E, 2I, K/2] (gate rows 0..I-1, then up rows I..2I-1, packed e2m1) and w_dn u8 [E, K, I/2].
// The e4m3 block scales sf_gu [E, 2I * K/16] and sf_dn [E, K * I/16] are in the swizzled layout of gemm_nvfp4.cu (each
// expert swizzled as one [2I, K] or [K, I] operand; the prefill GEMM reads the same bytes):
//     offset(r, kb) = ((r / 128) * KB4 + kb / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4,   KB4 = (K / 16) / 4
// The global scales gs f32 [E, 3] are (gate, up, down).
//
// Warp mapping of the expert GEMMs. The grid covers 10M pair slots x the column tiles of one expert (16 output columns each).
// The warp of slot s reads e = ids[s]. If an earlier pair has the same expert, the warp exits. Thus the warp that holds the
// first pair of an expert owns the tile of that expert. It collects the pairs of e in increasing pair order with ballots.
// These pairs are the rows of its mma: 16 per row group, RG = ceil(M / 16) row groups. A row's sum does not depend on its
// position in the mma or on the other rows. The k order is fixed, and there is no split-K and no atomic. Thus each output row
// is bit-identical for any set of rows in the launch.
//
// A warp owns the 16 weight rows G(rho) = 128 rb + 32 (rho / 4) + 4 w8 + rho % 4, rho = 0..15, of the 128-row block rb (tile
// ct = 8 rb + w8). In the swizzled layout, the scales of these rows for one 64-k atom are 64 contiguous bytes at
// atom * 512 + 64 w8: rho % 4 picks the 16-byte line, rho / 4 the 4-byte word, and the k-block the byte. The gate and up
// rows of a column are I apart (I % 128 == 0), so the up scales are the same run in block rb + I / 128. moe_gate_up streams
// 32 rows (16 gate + 16 up: four mma n-tiles) in chunks of 512 k (256-byte runs per row, 64-byte scale runs). It keeps one
// chunk ahead in registers and goes through a per-warp shared scratch in the fragment order of skinny.cu. moe_down streams
// the 16 rows of all I k at once (I / 2-byte runs), before its PDL wait.
//
// The output column of n-tile t (rows rho = 8 (t % 2) + g), lane quad q and element b is G(8 (t % 2) + 2q + b) =
// 128 rb + 32 (2 (t % 2) + q / 2) + 4 w8 + 2 (q % 2) + b: two adjacent columns per lane, one 4- or 8-byte store.
//
// PDL. moe_route and moe_combine trigger at once and wait before they read their inputs. moe_gate_up waits first (it needs
// the routing) and triggers after its wait. Thus moe_down, which starts on that trigger, can read ids before its own wait and
// prefetch its weights. Every block waits before it exits. Under the skip flag (skinny_skip), the route and expert kernels
// write nothing and moe_combine writes zeros.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.cuh"
#include "pdl.cuh"
#include "skinny_core.cuh"

namespace moe {
using namespace cc;
using skinny::e4m3_to_h2;
using skinny::fp4x2_scaled;

constexpr int TOPK = 10;
constexpr int WARPS = 4;      // warps per block of the expert kernels
constexpr int MAXP = 48;      // pairs of one expert in a launch: M <= 48
constexpr int ROUTE_W = 16;   // warps (rows) per block of moe_route
constexpr int MAX_E = 512;    // experts that moe_route handles (16 logits per lane)
constexpr int GU_ROWB = 288;  // gate_up scratch row stride (256 B + 32: conflict-free 8-byte fragment reads)
constexpr int GU_SCRATCH = 32 * GU_ROWB + 1024 + MAXP * 4;

// ------------------------------------------------------------------------------------------------------------ routing
// A total order of (logit descending, index ascending) as one unsigned key: the order-preserving bits of the bf16 logit
// (-0.0 counts as +0.0), then 0xffff - index. Key 0 marks no entry.
__device__ __forceinline__ uint32_t route_key(uint32_t u, int idx) {
    if ((u & 0x7fffu) == 0) u = 0;
    const uint32_t ord = (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
    return (ord << 16) | (uint32_t)(0xffff - idx);
}
__device__ __forceinline__ float key_logit(uint32_t key) {
    const uint32_t ord = key >> 16, u = (ord & 0x8000u) ? (ord & 0x7fffu) : (~ord & 0xffffu);
    return __uint_as_float(u << 16);
}

// One warp per row. Lane t holds the logits 32 i + t. Ten rounds of a warp-wide max pick the top 10 in order. Then
// p_k = exp(l_k - max) / sum_e exp(l_e - max) (fp32, the precise expf), w_k = bf16(p_k / (p_0 + ... + p_9)), and
// sg = bf16(sigmoid(logit[E])). stats (optional): stats[0] += the distinct experts of each block of ROUTE_W rows,
// stats[1] += 1 per launch.
__global__ void __launch_bounds__(ROUTE_W * 32) k_route(const __nv_bfloat16* __restrict__ logits, int ld, int M, int E, int* __restrict__ ids,
                                                        float* __restrict__ w, float* __restrict__ sg, int* __restrict__ stats,
                                                        const int* __restrict__ skip) {
    PDL_TRIGGER();
    __shared__ uint32_t seen[MAX_E / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (stats && threadIdx.x < MAX_E / 32) seen[threadIdx.x] = 0;
    PDL_WAIT();
    if (skip && *skip) return;  // the whole block: uniform
    if (stats) __syncthreads();
    const int m = blockIdx.x * ROUTE_W + warp;
    if (m < M) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(logits) + (size_t)m * ld;
        uint32_t key[MAX_E / 32];
        float val[MAX_E / 32];
#pragma unroll
        for (int i = 0; i < MAX_E / 32; ++i) {
            const int idx = 32 * i + lane;
            const uint32_t u = idx < E ? row[idx] : 0u;
            key[i] = idx < E ? route_key(u, idx) : 0u;
            val[i] = __uint_as_float(u << 16);
        }
        uint32_t mine = 0;  // lane k < 10 keeps the key of slot k
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k) {
            uint32_t best = 0;
#pragma unroll
            for (int i = 0; i < MAX_E / 32; ++i) best = max(best, key[i]);
            best = __reduce_max_sync(0xffffffffu, best);
#pragma unroll
            for (int i = 0; i < MAX_E / 32; ++i) key[i] = key[i] == best ? 0u : key[i];
            if (lane == k) mine = best;
        }
        const float mx = key_logit(__shfl_sync(0xffffffffu, mine, 0));
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < MAX_E / 32; ++i)
            if (32 * i + lane < E) s += expf(val[i] - mx);
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const float p = expf(key_logit(mine) - mx) / s;
        float tot = 0.f;  // p_0 + ... + p_9 in slot order
#pragma unroll
        for (int k = 0; k < TOPK; ++k) tot += __shfl_sync(0xffffffffu, p, k);
        if (lane < TOPK) {
            const int e = 0xffff - (int)(mine & 0xffffu);
            ids[(size_t)m * TOPK + lane] = e;
            w[(size_t)m * TOPK + lane] = bf(p / tot);
            if (stats) atomicOr(&seen[e >> 5], 1u << (e & 31));
        }
        if (lane == 0) {
            const float l = __uint_as_float((uint32_t)row[E] << 16);
            sg[m] = bf(1.f / (1.f + expf(-l)));
        }
    }
    if (stats) {
        __syncthreads();
        if (threadIdx.x < 32) {
            int c = threadIdx.x < MAX_E / 32 ? __popc(seen[threadIdx.x]) : 0;
#pragma unroll
            for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
            if (threadIdx.x == 0) {
                atomicAdd(stats, c);
                if (blockIdx.x == 0) atomicAdd(stats + 1, 1);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------ ownership
// The pairs of expert e = ids[slot] in increasing pair order into plist; returns their count. Returns -1 when an earlier pair
// has the same expert (another warp owns e) or when e is out of range. LDCG: read ids through L2 (moe_down reads them before
// its PDL wait). The rows of ids hold distinct experts (moe_route makes them so), so the count is at most M <= MAXP. Ids
// with a repeated expert in a row give wrong rows but no write past plist: the count stops at MAXP.
template <bool LDCG>
__device__ __forceinline__ int collect_pairs(const int* __restrict__ ids, int P, int slot, int E, int* plist, int lane) {
    const int e = LDCG ? __ldcg(ids + slot) : ids[slot];
    if ((unsigned)e >= (unsigned)E) return -1;
    int n = 0;
    for (int b = 0; b < P; b += 32) {
        const int p = b + lane;
        const int v = p < P ? (LDCG ? __ldcg(ids + p) : ids[p]) : -1;
        const uint32_t hit = __ballot_sync(0xffffffffu, v == e);
        const uint32_t ge = slot <= b ? 0xffffffffu : slot - b >= 32 ? 0u : 0xffffffffu << (slot - b);
        if (hit & ~ge) return -1;
        const uint32_t mine = hit & ge;
        const int pos = n + __popc(mine & ((1u << lane) - 1u));
        if (((mine >> lane) & 1u) && pos < MAXP) plist[pos] = p;
        n += __popc(mine);
    }
    __syncwarp();
    return min(n, MAXP);
}

// 16 bf16 activations (32 bytes) of a row at k, or zeros
__device__ __forceinline__ void load_a(const __nv_bfloat16* row, int k, uint4 (&a)[2]) {
    if (row) {
        a[0] = __ldg(reinterpret_cast<const uint4*>(row + k));
        a[1] = __ldg(reinterpret_cast<const uint4*>(row + k) + 1);
    } else {
        a[0] = a[1] = make_uint4(0, 0, 0, 0);
    }
}

// The 4 mma k-steps of one 64-k group for one n-tile: B = 8 scratch bytes (16 e2m1) at rowp, one e4m3 block scale.
__device__ __forceinline__ void mma_group(float (&acc)[4], const uint4 (&al)[2], const uint4 (&ah)[2], const uint8_t* rowp, uint32_t sbyte) {
    const uint2 wv = *reinterpret_cast<const uint2*>(rowp);
    const __half2 sc = e4m3_to_h2(sbyte);
    const uint32_t* lw = reinterpret_cast<const uint32_t*>(al);
    const uint32_t* hw = reinterpret_cast<const uint32_t*>(ah);
    const uint32_t ww[2] = {wv.x, wv.y};
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        const uint32_t two = (ww[jj >> 1] >> (16 * (jj & 1))) & 0xffff;
        const uint32_t a[4] = {lw[2 * jj], hw[2 * jj], lw[2 * jj + 1], hw[2 * jj + 1]};
        mma_bf16(acc, a, fp4x2_scaled(two & 0xff, sc), fp4x2_scaled(two >> 8, sc));
    }
}

// ------------------------------------------------------------------------------------------------------------ gate | up
// Item = (slot, ct) = (item / (I / 16), item % (I / 16)). I / 16 is a multiple of WARPS, so a block holds one slot.
template <int RG>
__global__ void __launch_bounds__(WARPS * 32) k_gate_up(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, int M, int K, int I,
                                                         int E, const uint8_t* __restrict__ w_gu, const uint8_t* __restrict__ sf_gu,
                                                         const float* __restrict__ gs, __nv_bfloat16* __restrict__ h,
                                                         const int* __restrict__ skip) {
    extern __shared__ __align__(16) uint8_t smem[];
    PDL_WAIT();
    PDL_TRIGGER();  // after the wait: moe_down reads ids before its own wait
    if (skip && *skip) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    const int tiles = I / 16, item = blockIdx.x * WARPS + warp, slot = item / tiles, ct = item % tiles, P = M * TOPK;
    if (slot >= P) return;
    uint8_t* wsc = smem + warp * GU_SCRATCH;  // [32 rows][GU_ROWB]: one chunk in fragment order
    uint8_t* ssc = wsc + 32 * GU_ROWB;        // [2][8 atoms][64 B]: gate, then up scales of the chunk
    int* plist = reinterpret_cast<int*>(ssc + 1024);
    const int n = collect_pairs<false>(ids, P, slot, E, plist, lane);
    if (n < 0) return;
    const int e = ids[slot], rb = ct >> 3, w8 = ct & 7, kb4 = K / 64, chunks = K / 512;
    const size_t rowbytes = K / 2;
    const uint8_t* we = w_gu + (size_t)e * 2 * I * rowbytes;
    const uint8_t* se = sf_gu + (size_t)e * 2 * I * (K / 16);
    // logical row r = 8 t + (rho % 8) of n-tile t (0, 1 gate; 2, 3 up) -> physical row
    auto wrow = [&](int r) -> const uint8_t* {
        const int t = r >> 3, rho = 8 * (t & 1) + (r & 7);
        return we + (size_t)(128 * rb + 32 * (rho >> 2) + 4 * w8 + (rho & 3) + (t >> 1) * I) * rowbytes;
    };
    const int hr = lane >> 4, seg = lane & 15;
    uint4 v[16], sv[2];
    auto load = [&](int c) {
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __ldcs(reinterpret_cast<const uint4*>(wrow(2 * i + hr) + (size_t)c * 256) + seg);
#pragma unroll
        for (int u = 0; u < 2; ++u)
            sv[u] = __ldcs(reinterpret_cast<const uint4*>(se + ((size_t)((rb + u * (I / 128)) * kb4 + 8 * c + (lane >> 2)) * 512 + 64 * w8 +
                                                               16 * (lane & 3))));
    };
    const __nv_bfloat16* xr[RG][2];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int j = 16 * rg + g + 8 * hh;
            xr[rg][hh] = j < n ? x + (size_t)(plist[j] / TOPK) * K : nullptr;
        }
    float acc[RG][4][4];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int t = 0; t < 4; ++t) acc[rg][t][0] = acc[rg][t][1] = acc[rg][t][2] = acc[rg][t][3] = 0.f;
    load(0);
    for (int c = 0; c < chunks; ++c) {
#pragma unroll
        for (int i = 0; i < 16; ++i) *reinterpret_cast<uint4*>(wsc + (2 * i + hr) * GU_ROWB + seg * 16) = v[i];
        *reinterpret_cast<uint4*>(ssc + lane * 16) = sv[0];
        *reinterpret_cast<uint4*>(ssc + 512 + lane * 16) = sv[1];
        __syncwarp();
        if (c + 1 < chunks) load(c + 1);  // in flight while this chunk is multiplied
#pragma unroll 2
        for (int S = 0; S < 8; ++S) {
            uint4 al[RG][2], ah[RG][2];
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) {
                load_a(xr[rg][0], c * 512 + 64 * S + 16 * q, al[rg]);
                load_a(xr[rg][1], c * 512 + 64 * S + 16 * q, ah[rg]);
            }
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint8_t* rowp = wsc + (8 * t + g) * GU_ROWB + 32 * S + 8 * q;
                const uint32_t sb = ssc[(t >> 1) * 512 + S * 64 + (g & 3) * 16 + (2 * (t & 1) + (g >> 2)) * 4 + q];
#pragma unroll
                for (int rg = 0; rg < RG; ++rg) mma_group(acc[rg][t], al[rg], ah[rg], rowp, sb);
            }
        }
        __syncwarp();
    }
    const float gg = gs[3 * e], gu = gs[3 * e + 1];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int j = 16 * rg + g + 8 * hh;
            if (j >= n) continue;
            const int p = plist[j];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int col = 128 * rb + 32 * (2 * t + (q >> 1)) + 4 * w8 + 2 * (q & 1);
                const float g0 = acc[rg][t][2 * hh] * gg, g1 = acc[rg][t][2 * hh + 1] * gg;
                const float u0 = acc[rg][t + 2][2 * hh] * gu, u1 = acc[rg][t + 2][2 * hh + 1] * gu;
                *reinterpret_cast<__nv_bfloat162*>(h + (size_t)p * I + col) =
                    __floats2bfloat162_rn(g0 / (1.f + __expf(-g0)) * u0, g1 / (1.f + __expf(-g1)) * u1);
            }
        }
}

// ------------------------------------------------------------------------------------------------------------ down
template <int NI>
struct DownCfg {
    static constexpr int I = 64 * NI;
    static constexpr int RB = I / 2;                                 // weight bytes per row
    static constexpr int SEGS = RB / 16;                             // 16-byte loads per row
    static constexpr int ROWB = RB + ((32 - RB) % 128 + 128) % 128;  // scratch row stride = 32 mod 128: conflict-free
    static constexpr int SL = (4 * NI + 31) / 32;                    // 16-byte scale loads per lane (NI atoms x 64 B)
    static constexpr int SCRATCH = 16 * ROWB + NI * 64 + MAXP * 4;
};

// Item = (slot, ct) = (item / (K / 16), item % (K / 16)). Before the PDL wait: the ownership (ids from moe_route, which is
// complete because moe_gate_up triggers after its wait) and the whole weight tile in registers.
template <int NI, int RG>
__global__ void __launch_bounds__(WARPS * 32) k_down(const __nv_bfloat16* __restrict__ h, const int* __restrict__ ids, int M, int K, int E,
                                                      const uint8_t* __restrict__ w_dn, const uint8_t* __restrict__ sf_dn,
                                                      const float* __restrict__ gs, float* __restrict__ yp, const int* __restrict__ skip) {
    using C = DownCfg<NI>;
    extern __shared__ __align__(16) uint8_t smem[];
    PDL_TRIGGER();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    const int tiles = K / 16, item = blockIdx.x * WARPS + warp, slot = item / tiles, ct = item % tiles, P = M * TOPK;
    uint8_t* wsc = smem + warp * C::SCRATCH;  // [16 rows][ROWB]
    uint8_t* ssc = wsc + 16 * C::ROWB;        // [NI atoms][64 B]
    int* plist = reinterpret_cast<int*>(ssc + NI * 64);
    const bool skipped = skip && *skip;
    const int n = skipped || slot >= P ? -1 : collect_pairs<true>(ids, P, slot, E, plist, lane);
    if (n < 0) {
        PDL_WAIT();
        return;
    }
    const int e = __ldcg(ids + slot), rb = ct >> 3, w8 = ct & 7;
    const uint8_t* we = w_dn + (size_t)e * K * C::RB;
    const uint8_t* se = sf_dn + (size_t)e * K * (C::I / 16);
    uint4 v[NI], sv[C::SL];
#pragma unroll
    for (int i = 0; i < NI; ++i) {  // 16 rows x SEGS loads: NI per lane
        const int f = 32 * i + lane, r = f / C::SEGS, sg = f % C::SEGS;
        v[i] = __ldcs(reinterpret_cast<const uint4*>(we + (size_t)(128 * rb + 32 * (r >> 2) + 4 * w8 + (r & 3)) * C::RB) + sg);
    }
#pragma unroll
    for (int i = 0; i < C::SL; ++i) {
        const int f = 32 * i + lane;
        sv[i] = f < 4 * NI ? __ldcs(reinterpret_cast<const uint4*>(se + ((size_t)(rb * NI + (f >> 2)) * 512 + 64 * w8 + 16 * (f & 3))))
                           : make_uint4(0, 0, 0, 0);
    }
    PDL_WAIT();
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int f = 32 * i + lane;
        *reinterpret_cast<uint4*>(wsc + (f / C::SEGS) * C::ROWB + (f % C::SEGS) * 16) = v[i];
    }
#pragma unroll
    for (int i = 0; i < C::SL; ++i)
        if (32 * i + lane < 4 * NI) *reinterpret_cast<uint4*>(ssc + (32 * i + lane) * 16) = sv[i];
    __syncwarp();
    const __nv_bfloat16* hr[RG][2];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int j = 16 * rg + g + 8 * hh;
            hr[rg][hh] = j < n ? h + (size_t)plist[j] * C::I : nullptr;
        }
    float acc[RG][2][4];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int t = 0; t < 2; ++t) acc[rg][t][0] = acc[rg][t][1] = acc[rg][t][2] = acc[rg][t][3] = 0.f;
#pragma unroll 2
    for (int S = 0; S < NI; ++S) {
        uint4 al[RG][2], ah[RG][2];
#pragma unroll
        for (int rg = 0; rg < RG; ++rg) {
            load_a(hr[rg][0], 64 * S + 16 * q, al[rg]);
            load_a(hr[rg][1], 64 * S + 16 * q, ah[rg]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint8_t* rowp = wsc + (8 * t + g) * C::ROWB + 32 * S + 8 * q;
            const uint32_t sb = ssc[S * 64 + (g & 3) * 16 + (2 * t + (g >> 2)) * 4 + q];
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) mma_group(acc[rg][t], al[rg], ah[rg], rowp, sb);
        }
    }
    const float gd = gs[3 * e + 2];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int j = 16 * rg + g + 8 * hh;
            if (j >= n) continue;
            const int p = plist[j];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int col = 128 * rb + 32 * (2 * t + (q >> 1)) + 4 * w8 + 2 * (q & 1);
                *reinterpret_cast<float2*>(yp + (size_t)p * K + col) = make_float2(acc[rg][t][2 * hh] * gd, acc[rg][t][2 * hh + 1] * gd);
            }
        }
}

// ------------------------------------------------------------------------------------------------------------ combine
// One thread per 8 columns of a row: y = bf16(w_0 yp_0 + ... + w_9 yp_9 + sg * shared), added in this order in fp32.
__global__ void k_combine(const float* __restrict__ yp, const float* __restrict__ w, const float* __restrict__ sg,
                          const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y, int M, int K, const int* __restrict__ skip) {
    PDL_TRIGGER();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x, per = K / 8;
    PDL_WAIT();
    if (idx >= M * per) return;
    const int m = idx / per, n = (idx % per) * 8;
    uint4 o = make_uint4(0, 0, 0, 0);
    if (!(skip && *skip)) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const float wk = w[m * TOPK + k];
            const float4* r = reinterpret_cast<const float4*>(yp + ((size_t)m * TOPK + k) * K + n);
            const float4 a = r[0], b = r[1];
            const float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += wk * f[i];
        }
        if (shared) {
            float f[8];
            bf16x8_to_float(*reinterpret_cast<const uint4*>(shared + (size_t)m * K + n), f);
            const float s = sg[m];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += s * f[i];
        }
        __nv_bfloat162 b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        o = *reinterpret_cast<const uint4*>(b);
    }
    *reinterpret_cast<uint4*>(y + (size_t)m * K + n) = o;
}

// ------------------------------------------------------------------------------------------------------------ launch
template <typename... KArgs, typename... Args>
cudaError_t launch_pdl(void (*kern)(KArgs...), int grid, int block, size_t smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(grid);
    lc.blockDim = dim3(block);
    lc.dynamicSmemBytes = smem;
    lc.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = attr;
    lc.numAttrs = 1;
    return cudaLaunchKernelEx(&lc, kern, args...);
}

inline int row_groups(int M) { return (M + 15) / 16; }

template <int NI>
cudaError_t launch_down_ni(const void* h, const int* ids, int M, int K, int E, const void* w_dn, const void* sf_dn, const float* gs, float* yp,
                           cudaStream_t st) {
    const int blocks = (M * TOPK * (K / 16) + WARPS - 1) / WARPS;
    const size_t smem = WARPS * DownCfg<NI>::SCRATCH;
    const auto* hh = (const __nv_bfloat16*)h;
    const auto* w = (const uint8_t*)w_dn;
    const auto* s = (const uint8_t*)sf_dn;
    const int* skip = skinny_skip_flag();
    switch (row_groups(M)) {
        case 1: return launch_pdl(k_down<NI, 1>, blocks, WARPS * 32, smem, st, hh, ids, M, K, E, w, s, gs, yp, skip);
        case 2: return launch_pdl(k_down<NI, 2>, blocks, WARPS * 32, smem, st, hh, ids, M, K, E, w, s, gs, yp, skip);
        default: return launch_pdl(k_down<NI, 3>, blocks, WARPS * 32, smem, st, hh, ids, M, K, E, w, s, gs, yp, skip);
    }
}

}  // namespace moe

cudaError_t launch_moe_route(const void* logits, int ld, int M, int E, int* ids, float* w, float* sg, int* stats, cudaStream_t st) {
    using namespace moe;
    if (M < 1 || E < TOPK || E > MAX_E || ld < E + 1) return cudaErrorInvalidValue;
    return launch_pdl(k_route, (M + ROUTE_W - 1) / ROUTE_W, ROUTE_W * 32, 0, st, (const __nv_bfloat16*)logits, ld, M, E, ids, w, sg, stats,
                      skinny_skip_flag());
}

cudaError_t launch_moe_gate_up(const void* x, const int* ids, int M, int K, int I, int E, const void* w_gu, const void* sf_gu, const float* gs,
                               void* h, cudaStream_t st) {
    using namespace moe;
    if (M < 1 || M > MAXP || K % 512 || I % 128 || I < 128) return cudaErrorInvalidValue;
    const int blocks = M * TOPK * (I / 16) / WARPS;
    const size_t smem = WARPS * GU_SCRATCH;
    const auto* xx = (const __nv_bfloat16*)x;
    const auto* w = (const uint8_t*)w_gu;
    const auto* s = (const uint8_t*)sf_gu;
    auto* hh = (__nv_bfloat16*)h;
    const int* skip = skinny_skip_flag();
    switch (row_groups(M)) {
        case 1: return launch_pdl(k_gate_up<1>, blocks, WARPS * 32, smem, st, xx, ids, M, K, I, E, w, s, gs, hh, skip);
        case 2: return launch_pdl(k_gate_up<2>, blocks, WARPS * 32, smem, st, xx, ids, M, K, I, E, w, s, gs, hh, skip);
        default: return launch_pdl(k_gate_up<3>, blocks, WARPS * 32, smem, st, xx, ids, M, K, I, E, w, s, gs, hh, skip);
    }
}

cudaError_t launch_moe_down(const void* h, const int* ids, int M, int K, int I, int E, const void* w_dn, const void* sf_dn, const float* gs,
                            float* yp, cudaStream_t st) {
    using namespace moe;
    if (M < 1 || M > MAXP || K % 128) return cudaErrorInvalidValue;
    switch (I) {
        case 128: return launch_down_ni<2>(h, ids, M, K, E, w_dn, sf_dn, gs, yp, st);
        case 256: return launch_down_ni<4>(h, ids, M, K, E, w_dn, sf_dn, gs, yp, st);
        case 384: return launch_down_ni<6>(h, ids, M, K, E, w_dn, sf_dn, gs, yp, st);
        case 512: return launch_down_ni<8>(h, ids, M, K, E, w_dn, sf_dn, gs, yp, st);
        case 640: return launch_down_ni<10>(h, ids, M, K, E, w_dn, sf_dn, gs, yp, st);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t launch_moe_combine(const float* yp, const float* w, const float* sg, const void* shared, void* y, int M, int K, cudaStream_t st) {
    using namespace moe;
    if (M < 1 || K % 8) return cudaErrorInvalidValue;
    const int n = M * (K / 8);
    return launch_pdl(k_combine, (n + 255) / 256, 256, 0, st, yp, w, sg, (const __nv_bfloat16*)shared, (__nv_bfloat16*)y, M, K, skinny_skip_flag());
}
