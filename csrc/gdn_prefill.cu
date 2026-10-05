// gdn_prefill.cu -- chunked Gated DeltaNet forward for prefill (PLAN.md 4.4 item 4; replaces FLA's chunk_gated_delta_rule
// in engine/model/prefill.py, docs/phase6_progress.md section 15).
//
// Per value head h (key head h / (Hv / Hk)), state S [K=128][V=128] fp32, chunks of C = 64 tokens, in-chunk cumulative
// log-decay G_i = sum_{t <= i} g_t:
//   A_ij = beta_i (k_i . k_j) e^(G_i - G_j)  (j < i),   T = (I + A)^-1                      k_wy (all chunks in parallel)
//   W = T diag(beta e^G) K,   U = T diag(beta) V,   V_new = U - W S
//   O = scale (diag(e^G) Q S + (Q K^T o M) V_new),   M_ij = e^(G_i - G_j) for j <= i, else 0
//   S <- e^(G_C) S + K^T diag(e^(G_C - G)) V_new                                            k_chunk (sequential over chunks)
// FLA runs this as cumsum, kkt/solve, w/u recompute, state recurrence (writing every chunk's state) and output kernels; here
// the recurrence keeps S on chip and computes W, U, V_new, O and the update per chunk with mma.sync bf16 (fp32 accumulate),
// one block per (value head, 64-wide V slice). Inputs are token-major: q, k [T, Hk, 128] (L2-normalized), v [T, Hv, 128]
// bf16, g [T, Hv] fp32 (log decay), beta [T, Hv] bf16; o [T, Hv, 128] bf16; state [Hv, 128, 128] fp32, read and written.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace gdn {

constexpr int C = 64, DK = 128, DV = 128, BV = 64, WARPS = 4;
constexpr int KS = DK + 8, VS = BV + 8, AS = C + 1;  // padded shared rows (bf16 / bf16 / fp32)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&b);
}
__device__ __forceinline__ float2 unpack(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

// Fragment conventions (m16n8k16, g = lane / 4, q = lane % 4):
//   A: a0 (row g, k 2q..2q+1), a1 (row g+8, same k), a2 (row g, k 8+2q..), a3 (row g+8, k 8+2q..)
//   B: b0 (k 2q..2q+1, col g), b1 (k 8+2q.., col g);   C: c0 c1 (row g, cols 2q, 2q+1), c2 c3 (row g+8)
// ldsm4 of A from row-major [m][k]: lane row (l & 7) + ((l >> 3) & 1) * 8, col (l >> 4) * 8.
// B from row-major [k][n] (n contiguous): ldsm4t, lane k (l & 7) + ((l >> 3) & 1) * 8, n (l >> 4) * 8 -> n-tiles n0, n0 + 8.
// B from row-major [n][k] (k contiguous): ldsm4, lane n (l & 7) + (l >> 4) * 8, k ((l >> 3) & 1) * 8 -> n-tiles n0, n0 + 8.
// A from row-major [k][m] (A transposed): ldsm4t, lane k (l & 7) + (l >> 4) * 8, m ((l >> 3) & 1) * 8.

// in-chunk cumulative decay of value head h: G[i] for i < C (padding rows: g = 0), by one warp
__device__ __forceinline__ void chunk_cumsum(const float* __restrict__ g, int Hv, int h, int t0, int nt, float* Gs, int lane) {
    float a = lane * 2 < nt ? g[(size_t)(t0 + lane * 2) * Hv + h] : 0.f;
    float b = lane * 2 + 1 < nt ? g[(size_t)(t0 + lane * 2 + 1) * Hv + h] : 0.f;
    b += a;
    float s = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffffu, s, o);
        if (lane >= o) s += n;
    }
    const float base = s - b;  // exclusive prefix of the pairs
    Gs[lane * 2] = base + a;
    Gs[lane * 2 + 1] = base + b;
}

// ---- k_wy: T = (I + A)^-1 per (chunk, value head), bf16 [Hv][chunks][64][64] ----
__global__ void __launch_bounds__(WARPS * 32) k_wy(const __nv_bfloat16* __restrict__ k, const float* __restrict__ g,
                                                   const __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ Tout, int T, int Hk,
                                                   int Hv) {
    extern __shared__ __align__(16) uint8_t wsm[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(wsm);           // [C][KS]
    float* Am = reinterpret_cast<float*>(Ks + C * KS);                     // [C][AS], then X [3][16][16]; T reuses Ks's space
    __shared__ float Gs[C], Bs[C];
    const int c = blockIdx.x, h = blockIdx.y, hk = h / (Hv / Hk), t0 = c * C, nt = min(C, T - t0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
    for (int idx = tid; idx < C * (DK / 8); idx += WARPS * 32) {
        const int r = idx / (DK / 8), u = idx % (DK / 8);
        cp_async16(Ks + r * KS + u * 8, k + ((size_t)(t0 + min(r, nt - 1)) * Hk + hk) * DK + u * 8, r < nt);
    }
    cp_async_commit();
    if (warp == 0) chunk_cumsum(g, Hv, h, t0, nt, Gs, lane);
    for (int i = tid; i < C; i += WARPS * 32) Bs[i] = i < nt ? __bfloat162float(beta[(size_t)(t0 + i) * Hv + h]) : 0.f;
    cp_async_wait0();
    __syncthreads();
    // K K^T: warp w rows 16w .. 16w + 15, all 64 columns
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int s = 0; s < DK / 16; ++s) {
        uint32_t a[4];
        ldsm4(a, Ks + (warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + s * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            uint32_t b[4];
            ldsm4(b, Ks + (np * 16 + (lane & 7) + (lane >> 4) * 8) * KS + s * 16 + ((lane >> 3) & 1) * 8);
            mma(acc[2 * np], a, b[0], b[1]);
            mma(acc[2 * np + 1], a, b[2], b[3]);
        }
    }
#pragma unroll
    for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int i = warp * 16 + gq + (e >> 1) * 8, j = nt8 * 8 + 2 * q + (e & 1);
            Am[i * AS + j] = j < i ? Bs[i] * __expf(Gs[i] - Gs[j]) * acc[nt8][e] : 0.f;
        }
    __syncthreads();
    // T = (I + A)^-1 by 16 x 16 blocks (unit lower triangular): diagonal blocks by substitution, then block rows
    // i = 1..3: T_ij = -T_ii sum_{j <= m < i} A_im T_mj. Tm holds T (fp32); the upper blocks stay zero.
    float* Tm = reinterpret_cast<float*>(Ks);  // K is no longer needed: K K^T is in Am
    __syncthreads();
    for (int idx = tid; idx < C * AS; idx += WARPS * 32) Tm[idx] = 0.f;
    __syncthreads();
    if (tid < C) {  // diagonal block b = tid / 16, column c of it
        const int b = tid >> 4, c = tid & 15, o0 = b * 16;
        float tc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float s = i == c ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < i; ++j) s -= Am[(o0 + i) * AS + o0 + j] * tc[j];  // tc[j] = 0 for j < c
            tc[i] = i >= c ? s : 0.f;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) Tm[(o0 + i) * AS + o0 + c] = tc[i];
    }
    __syncthreads();
    float* X = Am + C * AS;  // [3 blocks][16][16]: sum_m A_im T_mj for the block row being solved
    for (int bi = 1; bi < 4; ++bi) {
        // X_j = sum_{m = j}^{bi - 1} A[bi][m] T[m][j], for j < bi: bi * 256 elements over 128 threads
        for (int idx = tid; idx < bi * 256; idx += WARPS * 32) {
            const int bj = idx >> 8, r = (idx >> 4) & 15, cc = idx & 15;
            float s = 0.f;
            for (int bm = bj; bm < bi; ++bm)
#pragma unroll
                for (int t = 0; t < 16; ++t) s += Am[(bi * 16 + r) * AS + bm * 16 + t] * Tm[(bm * 16 + t) * AS + bj * 16 + cc];
            X[idx] = s;
        }
        __syncthreads();
        for (int idx = tid; idx < bi * 256; idx += WARPS * 32) {  // T[bi][j] = -T[bi][bi] X_j
            const int bj = idx >> 8, r = (idx >> 4) & 15, cc = idx & 15;
            float s = 0.f;
#pragma unroll
            for (int t = 0; t < 16; ++t) s -= Tm[(bi * 16 + r) * AS + bi * 16 + t] * X[(bj << 8) + t * 16 + cc];
            Tm[(bi * 16 + r) * AS + bj * 16 + cc] = s;
        }
        __syncthreads();
    }
    __nv_bfloat16* dst = Tout + (((size_t)h * gridDim.x + c) * C) * C;
    for (int idx = tid; idx < C * C / 2; idx += WARPS * 32) {
        const int r = idx / (C / 2), cc = (idx % (C / 2)) * 2;
        *reinterpret_cast<uint32_t*>(dst + r * C + cc) = pack(Tm[r * AS + cc], Tm[r * AS + cc + 1]);
    }
}

// ---- k_chunk: the recurrence, one block per (value head, V slice), sequential over chunks ----
__global__ void __launch_bounds__(WARPS * 32, 1) k_chunk(const __nv_bfloat16* __restrict__ qg, const __nv_bfloat16* __restrict__ k,
                                                         const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                                         const __nv_bfloat16* __restrict__ beta, const __nv_bfloat16* __restrict__ Tm,
                                                         float* __restrict__ state, __nv_bfloat16* __restrict__ o, int T, int Hk, int Hv,
                                                         float scale) {
    extern __shared__ __align__(16) uint8_t smem[];
    __nv_bfloat16* Ksm = reinterpret_cast<__nv_bfloat16*>(smem);  // [2][C][KS]
    __nv_bfloat16* Vsm = Ksm + 2 * C * KS;                         // [2][C][VS]
    __nv_bfloat16* Sb = Vsm + 2 * C * VS;                          // [DK][VS]   S (bf16) for the mma
    __nv_bfloat16* Vn = Sb + DK * VS;                              // [C][VS]    V_new
    __nv_bfloat16* Vd = Vn + C * VS;                               // [C][VS]    V_new * e^(G_C - G_j)
    float* Gs = reinterpret_cast<float*>(Vd + C * VS);             // [2][C]
    float* Bs = Gs + 2 * C;                                        // [2][C]

    const int h = blockIdx.x, vs = blockIdx.y, hk = h / (Hv / Hk);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
    const int nchunks = (T + C - 1) / C;

    auto load = [&](int c, int buf) {
        const int t0 = c * C, nt = min(C, T - t0);
        for (int idx = tid; idx < C * (DK / 8); idx += WARPS * 32) {
            const int r = idx / (DK / 8), u = idx % (DK / 8);
            cp_async16(Ksm + (buf * C + r) * KS + u * 8, k + ((size_t)(t0 + min(r, nt - 1)) * Hk + hk) * DK + u * 8, r < nt);
        }
        for (int idx = tid; idx < C * (BV / 8); idx += WARPS * 32) {
            const int r = idx / (BV / 8), u = idx % (BV / 8);
            cp_async16(Vsm + (buf * C + r) * VS + u * 8, v + ((size_t)(t0 + min(r, nt - 1)) * Hv + h) * DV + vs * BV + u * 8, r < nt);
        }
        cp_async_commit();
    };

    // S rows (k) 32w .. 32w + 31 of this V slice, fp32 in registers: [m-tile][n-tile][4]
    float S[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kr = warp * 32 + mt * 16 + gq + (e >> 1) * 8, vc = nt8 * 8 + 2 * q + (e & 1);
                S[mt][nt8][e] = state[((size_t)h * DK + kr) * DV + vs * BV + vc];
            }
    auto store_sb = [&]() {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int kr = warp * 32 + mt * 16 + gq + hh * 8, vc = nt8 * 8 + 2 * q;
                    *reinterpret_cast<uint32_t*>(Sb + kr * VS + vc) = pack(S[mt][nt8][2 * hh], S[mt][nt8][2 * hh + 1]);
                }
    };
    store_sb();
    load(0, 0);

    for (int c = 0; c < nchunks; ++c) {
        const int buf = c & 1, t0 = c * C, nt = min(C, T - t0);
        const __nv_bfloat16* Kc = Ksm + buf * C * KS;
        const __nv_bfloat16* Vc = Vsm + buf * C * VS;
        float* G = Gs + buf * C;
        float* Bt = Bs + buf * C;
        if (warp == 0) chunk_cumsum(g, Hv, h, t0, nt, G, lane);
        for (int i = tid; i < C; i += WARPS * 32) Bt[i] = i < nt ? __bfloat162float(beta[(size_t)(t0 + i) * Hv + h]) : 0.f;
        cp_async_wait0();
        __syncthreads();  // chunk c's K / V / G / beta, and S (bf16) of the chunk start, visible to all
        if (c + 1 < nchunks) load(c + 1, buf ^ 1);
        const float GC = G[nt - 1];
        const int r0 = warp * 16 + gq, r1 = r0 + 8;  // this lane's two token rows

        // T rows r0, r1 (A fragments, k = token j), from k_wy
        uint32_t ta[4][4];
        {
            const __nv_bfloat16* tr = Tm + (((size_t)h * nchunks + c) * C) * C;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int j = s * 16 + 2 * q;
                ta[s][0] = *reinterpret_cast<const uint32_t*>(tr + r0 * C + j);
                ta[s][1] = *reinterpret_cast<const uint32_t*>(tr + r1 * C + j);
                ta[s][2] = *reinterpret_cast<const uint32_t*>(tr + r0 * C + j + 8);
                ta[s][3] = *reinterpret_cast<const uint32_t*>(tr + r1 * C + j + 8);
            }
        }
        // column scales: W uses T diag(beta e^G), U uses T diag(beta)
        auto scaled = [&](int s, bool decay, uint32_t (&a)[4]) {
            const int j0 = s * 16 + 2 * q, j1 = j0 + 8;
            const float s00 = Bt[j0] * (decay ? __expf(G[j0]) : 1.f), s01 = Bt[j0 + 1] * (decay ? __expf(G[j0 + 1]) : 1.f);
            const float s10 = Bt[j1] * (decay ? __expf(G[j1]) : 1.f), s11 = Bt[j1 + 1] * (decay ? __expf(G[j1 + 1]) : 1.f);
            float2 f;
            f = unpack(ta[s][0]); a[0] = pack(f.x * s00, f.y * s01);
            f = unpack(ta[s][1]); a[1] = pack(f.x * s00, f.y * s01);
            f = unpack(ta[s][2]); a[2] = pack(f.x * s10, f.y * s11);
            f = unpack(ta[s][3]); a[3] = pack(f.x * s10, f.y * s11);
        };

        // V_new = T diag(beta) V - (T diag(beta e^G) K) S  (rows r0, r1; this V slice)
        float vn[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) vn[i][0] = vn[i][1] = vn[i][2] = vn[i][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t a[4];
            scaled(s, false, a);
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t b[4];
                ldsm4t(b, Vc + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                mma(vn[2 * np], a, b[0], b[1]);
                mma(vn[2 * np + 1], a, b[2], b[3]);
            }
        }
        {
            float w[16][4];  // W rows r0, r1 over the 128 key dims
#pragma unroll
            for (int i = 0; i < 16; ++i) w[i][0] = w[i][1] = w[i][2] = w[i][3] = 0.f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                uint32_t a[4];
                scaled(s, true, a);
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    uint32_t b[4];
                    ldsm4t(b, Kc + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * KS + np * 16 + (lane >> 4) * 8);
                    mma(w[2 * np], a, b[0], b[1]);
                    mma(w[2 * np + 1], a, b[2], b[3]);
                }
            }
            // vn -= W S: A = -W (C -> A fragments), B = S [k][v] (ldsm4t)
#pragma unroll
            for (int s = 0; s < DK / 16; ++s) {
                const uint32_t a[4] = {pack(-w[2 * s][0], -w[2 * s][1]), pack(-w[2 * s][2], -w[2 * s][3]), pack(-w[2 * s + 1][0], -w[2 * s + 1][1]),
                                       pack(-w[2 * s + 1][2], -w[2 * s + 1][3])};
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t b[4];
                    ldsm4t(b, Sb + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                    mma(vn[2 * np], a, b[0], b[1]);
                    mma(vn[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        // V_new rows -> shared, plain and decayed to the chunk end (for the state update)
        {
            const float d0 = r0 < nt ? __expf(GC - G[r0]) : 0.f, d1 = r1 < nt ? __expf(GC - G[r1]) : 0.f;
#pragma unroll
            for (int nt8 = 0; nt8 < 8; ++nt8) {
                const int vc = nt8 * 8 + 2 * q;
                *reinterpret_cast<uint32_t*>(Vn + r0 * VS + vc) = pack(vn[nt8][0], vn[nt8][1]);
                *reinterpret_cast<uint32_t*>(Vn + r1 * VS + vc) = pack(vn[nt8][2], vn[nt8][3]);
                *reinterpret_cast<uint32_t*>(Vd + r0 * VS + vc) = pack(vn[nt8][0] * d0, vn[nt8][1] * d0);
                *reinterpret_cast<uint32_t*>(Vd + r1 * VS + vc) = pack(vn[nt8][2] * d1, vn[nt8][3] * d1);
            }
        }
        // Q rows r0, r1 (A fragments over the key dims)
        uint32_t qa[8][4];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int kk = s * 16 + 2 * q;
            const __nv_bfloat16* q0 = qg + ((size_t)(t0 + min(r0, nt - 1)) * Hk + hk) * DK;
            const __nv_bfloat16* q1 = qg + ((size_t)(t0 + min(r1, nt - 1)) * Hk + hk) * DK;
            qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + kk);
            qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + kk);
            qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + kk + 8);
            qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + kk + 8);
        }
        // P = (Q K^T) o M  (rows r0, r1; 64 key tokens)
        float p[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i][0] = p[i][1] = p[i][2] = p[i][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t b[4];
                ldsm4(b, Kc + (np * 16 + (lane & 7) + (lane >> 4) * 8) * KS + s * 16 + ((lane >> 3) & 1) * 8);
                mma(p[2 * np], qa[s], b[0], b[1]);
                mma(p[2 * np + 1], qa[s], b[2], b[3]);
            }
#pragma unroll
        for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = e < 2 ? r0 : r1, j = nt8 * 8 + 2 * q + (e & 1);
                p[nt8][e] = j <= i ? p[nt8][e] * __expf(G[i] - G[j]) : 0.f;
            }
        __syncthreads();  // V_new / V_d of every row
        // O = scale (e^G Q S + P V_new)
        {
            float acc[8][4];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
            for (int s = 0; s < DK / 16; ++s)
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t b[4];
                    ldsm4t(b, Sb + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                    mma(acc[2 * np], qa[s], b[0], b[1]);
                    mma(acc[2 * np + 1], qa[s], b[2], b[3]);
                }
            const float e0 = __expf(G[r0]), e1 = __expf(G[r1]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i][0] *= e0; acc[i][1] *= e0; acc[i][2] *= e1; acc[i][3] *= e1;
            }
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t a[4] = {pack(p[2 * s][0], p[2 * s][1]), pack(p[2 * s][2], p[2 * s][3]), pack(p[2 * s + 1][0], p[2 * s + 1][1]),
                                       pack(p[2 * s + 1][2], p[2 * s + 1][3])};
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t b[4];
                    ldsm4t(b, Vn + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                    mma(acc[2 * np], a, b[0], b[1]);
                    mma(acc[2 * np + 1], a, b[2], b[3]);
                }
            }
#pragma unroll
            for (int nt8 = 0; nt8 < 8; ++nt8) {
                const int vc = vs * BV + nt8 * 8 + 2 * q;
                if (r0 < nt) *reinterpret_cast<uint32_t*>(o + ((size_t)(t0 + r0) * Hv + h) * DV + vc) = pack(acc[nt8][0] * scale, acc[nt8][1] * scale);
                if (r1 < nt) *reinterpret_cast<uint32_t*>(o + ((size_t)(t0 + r1) * Hv + h) * DV + vc) = pack(acc[nt8][2] * scale, acc[nt8][3] * scale);
            }
        }
        // S = e^(G_C) S + K^T V_d  (rows k 32w .. 32w + 31): A = K^T from K [j][k] (ldsm4t), B = V_d [j][v] (ldsm4t)
        {
            const float eC = __expf(GC);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
                    for (int e = 0; e < 4; ++e) S[mt][nt8][e] *= eC;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    uint32_t a[4];
                    ldsm4t(a, Kc + (s * 16 + (lane & 7) + (lane >> 4) * 8) * KS + warp * 32 + mt * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                    for (int np = 0; np < 4; ++np) {
                        uint32_t b[4];
                        ldsm4t(b, Vd + (s * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + np * 16 + (lane >> 4) * 8);
                        mma(S[mt][2 * np], a, b[0], b[1]);
                        mma(S[mt][2 * np + 1], a, b[2], b[3]);
                    }
                }
            }
        }
        __syncthreads();  // every warp is done reading Sb, Vn, Vd of this chunk
        store_sb();
    }
    cp_async_wait0();
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kr = warp * 32 + mt * 16 + gq + (e >> 1) * 8, vc = nt8 * 8 + 2 * q + (e & 1);
                state[((size_t)h * DK + kr) * DV + vs * BV + vc] = S[mt][nt8][e];
            }
}

constexpr int wy_smem() { return C * KS * 2 + C * AS * 4 + 3 * 256 * 4; }  // C * KS * 2 >= C * AS * 4: T fits in K's space
constexpr int chunk_smem() { return (2 * C * KS + 2 * C * VS + DK * VS + 2 * C * VS) * 2 + 4 * C * 4; }

}  // namespace gdn

// workspace: bf16 [Hv][ceil(T / 64)][64][64] (the T matrices)
size_t gdn_prefill_ws_bytes(int T, int Hv) { return (size_t)Hv * ((T + gdn::C - 1) / gdn::C) * gdn::C * gdn::C * 2; }

cudaError_t launch_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta, float* state, void* o, void* ws, int T,
                               int Hk, int Hv, float scale, cudaStream_t st) {
    if (T < 1 || Hv % Hk) return cudaErrorInvalidValue;
    const int nchunks = (T + gdn::C - 1) / gdn::C;
    static bool init_wy = false;
    if (!init_wy) {
        cudaFuncSetAttribute(gdn::k_wy, cudaFuncAttributeMaxDynamicSharedMemorySize, gdn::wy_smem());
        init_wy = true;
    }
    gdn::k_wy<<<dim3(nchunks, Hv), gdn::WARPS * 32, gdn::wy_smem(), st>>>((const __nv_bfloat16*)k, g, (const __nv_bfloat16*)beta, (__nv_bfloat16*)ws, T, Hk, Hv);
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(gdn::k_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize, gdn::chunk_smem());
        init = true;
    }
    gdn::k_chunk<<<dim3(Hv, gdn::DV / gdn::BV), gdn::WARPS * 32, gdn::chunk_smem(), st>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, (const __nv_bfloat16*)beta, (const __nv_bfloat16*)ws, state,
        (__nv_bfloat16*)o, T, Hk, Hv, scale);
    return cudaGetLastError();
}
