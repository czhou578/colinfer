// qsa.cu -- Qwen Sparse Attention kernels for qwen4_exp (docs/history/qwen4exp_plan.md section 5.3): the indexer
// prologue, the block keys, the block scores, the top-k selection and the sparse decode attention. The bindings are in
// qsa_bind.cpp, the Python wrappers in engine/model/qsa_kernels.py. Tests: tests/test_qsa.py.
//
// The indexer of a layer scores 4-token blocks of the context for each query row. Each row then attends to its top 512
// blocks plus the tail (the tokens of its incomplete block). A row's selection and its attention depend only on the row,
// so decode (T = 1), verify (T <= 16) and prefill give the same bits for a token.
//
//   k_prologue    q_idx [B, T, 4, 128] = RoPE(q_layernorm(q)) of the 4 indexer query heads; the raw key of each row goes
//                 to ik[b, pos_t[b] + t] (no norm, no RoPE).
//   k_block_keys  the block key of each block j that the new rows complete (4 j + 3 in [pos, pos + T)):
//                 ikb[b, j] = RoPE_{position of token 4 j}(k_layernorm(bf16(mean of the 4 raw keys in fp32))).
//   k_score       scores[b, t, j] = sum_h relu(q_idx[b, t, h] . ikb[b, j]) / sqrt(128) in fp32, -inf for the blocks
//                 that row t cannot see (j >= (p_t + 1) / 4, p_t = pos_t[b] + t).
//   k_topk        sel[b, t] = the top 512 blocks of row t in ascending index order, nsel[b, t] = their count.
//   attn::tc::k_tc<LIST = true> + k_combine (csrc/attn_tc.cuh): the decode attention over the keys of the list.
//
// Layouts (per slot b): ik bf16 [B, Lmax, 128]; ikb bf16 [B, Lmax / 4, 128]; rope_pos int32 [B, Lmax, 3] (the (t, h, w)
// rotary positions of the cache rows, for the block keys); scores fp32 [B, S, Lmax / 4], sel int32 [B, S, 512],
// nsel int32 [B, S], with S >= T rows (S = 16: the widest verify).
#include "attn_tc.cuh"

namespace qsa {
using namespace cc;

constexpr int DI = 128;   // indexer head dim
constexpr int HI = 4;     // indexer query heads
constexpr int TOPK = 512; // selected blocks per row (indexer_budget 2048 / indexer_compress_ratio 4)

// The zero-centered RMSNorm of one 128-wide row, one element per thread of a 128-thread block (the formula of
// attn_decode.cu k_prologue): bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)). red: 4 floats of shared memory.
__device__ __forceinline__ float norm128(float x, float w, float eps, float* red) {
    float ss = x * x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < DI / 32; ++i) tot += red[i];
    return bf(x * rsqrtf(tot / DI + eps) * (1.f + w));
}

// Grid (B * T, 5), 128 threads: block (b, t, h) normalizes and rotates indexer query head h < 4 of row (b, t), or (h = 4)
// copies the raw key of the row to ik[b, pos_t[b] + t] (active slots only). idx: the rows [q 4 x 128 | k 128] of the
// index projection, ld elements apart. The RoPE position follows attn_decode.cu k_prologue: rope_pos [B * T, 3], else
// pos + rope_delta[b], else pos. R: the rotated dims (64), with the frequencies inv_freq [R / 2].
__global__ void __launch_bounds__(DI) k_prologue(const __nv_bfloat16* __restrict__ idx, int ld, const __nv_bfloat16* __restrict__ qn_w,
                                                  const float* __restrict__ inv_freq, const int* __restrict__ pos_t, __nv_bfloat16* __restrict__ ik,
                                                  __nv_bfloat16* __restrict__ q_out, int T, int Lmax, int R, float eps, const int* __restrict__ active,
                                                  const int* __restrict__ rope_delta, const int* __restrict__ rope_pos, int sec_h, int sec_w) {
    PDL_TRIGGER();
    const int bt = blockIdx.x, b = bt / T, t = bt % T, h = blockIdx.y, d = threadIdx.x;
    const int pos = pos_t[b] + t;
    const size_t row = (size_t)b * T + t;
    const __nv_bfloat16* src = idx + row * ld;
    if (h == HI) {
        if ((active == nullptr || active[b] != 0) && pos < Lmax) ik[((size_t)b * Lmax + pos) * DI + d] = src[HI * DI + d];
        return;
    }
    __shared__ float xs[DI], red[DI / 32];
    const float xn = norm128(__bfloat162float(src[h * DI + d]), __bfloat162float(qn_w[d]), eps, red);
    xs[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < R) {
        const int i = d % (R / 2);
        const int rp = rope_position(i, rope_delta != nullptr ? pos + rope_delta[b] : pos, rope_pos != nullptr ? rope_pos + row * 3 : nullptr,
                                     sec_h, sec_w);
        y = rope_rotate(xs, d, R, xn, rp, inv_freq[i]);
    }
    q_out[(row * HI + h) * DI + d] = __float2bfloat16(y);
}

// Grid (B, T / 4 + 1), 128 threads (one per dim): block (b, y) builds block j = pos_t[b] / 4 + y if the new rows
// [pos, pos + T) complete it (4 j + 3 < pos + T); the other blocks exit. The mean is a sequential fp32 sum of the 4 raw
// keys, divided by 4 and rounded to bf16 (the reference's float().mean().to(bf16)). The RoPE position is the one of
// token 4 j: rope_pos[b, 4 j] (the slot's [Lmax, 3] record), else 4 j + rope_delta[b], else 4 j. Inactive slots write
// nothing.
__global__ void __launch_bounds__(DI) k_block_keys(const __nv_bfloat16* __restrict__ ik, __nv_bfloat16* __restrict__ ikb,
                                                    const __nv_bfloat16* __restrict__ kn_w, const float* __restrict__ inv_freq,
                                                    const int* __restrict__ pos_t, int T, int Lmax, int R, float eps, const int* __restrict__ active,
                                                    const int* __restrict__ rope_delta, const int* __restrict__ rope_pos, int sec_h, int sec_w) {
    PDL_TRIGGER();
    const int b = blockIdx.x, d = threadIdx.x, pos = pos_t[b], j = pos / 4 + blockIdx.y;
    if ((active != nullptr && active[b] == 0) || 4 * j + 3 >= pos + T || 4 * j + 3 >= Lmax) return;  // uniform over the block
    const __nv_bfloat16* kr = ik + ((size_t)b * Lmax + 4 * j) * DI + d;
    float s = __bfloat162float(kr[0]);
    s += __bfloat162float(kr[DI]);
    s += __bfloat162float(kr[2 * DI]);
    s += __bfloat162float(kr[3 * DI]);
    __shared__ float xs[DI], red[DI / 32];
    const float xn = norm128(bf(s / 4.f), __bfloat162float(kn_w[d]), eps, red);
    xs[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < R) {
        const int i = d % (R / 2), p0 = 4 * j;
        const int rp = rope_position(i, rope_delta != nullptr ? p0 + rope_delta[b] : p0,
                                     rope_pos != nullptr ? rope_pos + ((size_t)b * Lmax + p0) * 3 : nullptr, sec_h, sec_w);
        y = rope_rotate(xs, d, R, xn, rp, inv_freq[i]);
    }
    ikb[((size_t)b * (Lmax / 4) + j) * DI + d] = __float2bfloat16(y);
}

// ------------------------------------------------------------------------------------------------------------------
// Block scores. Grid (ceil(NBL / 256), B), 128 threads: block (x, b) scores the blocks [256 x, 256 x + 256) of slot b for
// all T rows; the blocks at or past (pos + T) / 4 (no row sees them) exit. mma.sync m16n8k16 bf16 -> fp32: the A rows are
// the 4 T query rows r = 4 t + h (MT = ceil(T / 4) m16 tiles in shared memory, zero past 4 T), the B columns the block
// keys. Warp w takes the blocks [64 w, 64 w + 64) of the 256, in two rounds of 4 n8 tiles, and loads its key fragments
// straight from global memory: lane (g = lane / 4, c = lane % 4) holds the dims 32 jj + 8 c .. + 7 (jj = 0 .. 3) of key g
// of each n tile. The first 4 of them are its B fragment of k-step 2 jj, the next 4 of k-step 2 jj + 1, and the A
// fragments read Q in the same dim order (the dot product is the same sum in another order).
// Epilogue: the lanes with g % 4 == 0 hold head 0 of a row; heads 1, 2, 3 are 4, 8, 12 lanes up. The score is
// (((r0 + r1) + r2) + r3) / sqrt(128) with r = relu(dot), -0 as +0.
// Bit identity: an mma row depends on its own A row and the keys only, so the score of a row does not depend on T.
constexpr int SC_KEYS = 256, SC_THREADS = 128, QSI = DI + 32;  // QSI: the Q row stride in shared memory (elements)

template <int MT>
__global__ void __launch_bounds__(SC_THREADS) k_score(const __nv_bfloat16* __restrict__ q_idx, const __nv_bfloat16* __restrict__ ikb,
                                                       const int* __restrict__ pos_t, float* __restrict__ scores, int T, int NBL, int srows) {
    PDL_TRIGGER();
    const int b = blockIdx.y, j0 = blockIdx.x * SC_KEYS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pos = pos_t[b], nbmax = min((pos + T) / 4, NBL);
    if (j0 >= nbmax) return;
    __shared__ __align__(16) __nv_bfloat16 Qs[MT * 16 * QSI];
    for (int idx = tid; idx < MT * 16 * (DI / 8); idx += SC_THREADS) {
        const int r = idx / (DI / 8), u = idx % (DI / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < 4 * T) v = *reinterpret_cast<const uint4*>(q_idx + ((size_t)b * T * HI + r) * DI + u * 8);
        *reinterpret_cast<uint4*>(Qs + r * QSI + u * 8) = v;
    }
    __syncthreads();
    const int g = lane >> 2, c = lane & 3;
    const float sqrt_dim = 11.313708498984761f;  // sqrt(128), as the reference divides by math.sqrt(128)
    for (int round = 0; round < 2; ++round) {
        const int jw = j0 + warp * 64 + round * 32;
        if (jw >= nbmax) break;  // uniform over the warp
        uint4 kb[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key = jw + nt * 8 + g;
            const __nv_bfloat16* kp = ikb + ((size_t)b * NBL + key) * DI + 8 * c;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) kb[nt][jj] = key < nbmax ? __ldg(reinterpret_cast<const uint4*>(kp + 32 * jj)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const uint4 x0 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g) * QSI + 32 * jj + 8 * c);
                const uint4 x1 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g + 8) * QSI + 32 * jj + 8 * c);
                const uint32_t a0[4] = {x0.x, x1.x, x0.y, x1.y}, a1[4] = {x0.z, x1.z, x0.w, x1.w};
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_bf16(acc[nt], a0, kb[nt][jj].x, kb[nt][jj].y);
                    mma_bf16(acc[nt], a1, kb[nt][jj].z, kb[nt][jj].w);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float v = acc[nt][e] > 0.f ? acc[nt][e] : 0.f;  // relu; -0 and NaN become +0
                    const float r1 = __shfl_down_sync(0xffffffffu, v, 4), r2 = __shfl_down_sync(0xffffffffu, v, 8),
                                r3 = __shfl_down_sync(0xffffffffu, v, 12);
                    const int t = mt * 4 + (g >> 2) + (e >= 2 ? 2 : 0), col = jw + nt * 8 + 2 * c + (e & 1);
                    if ((g & 3) == 0 && t < T && col < NBL) {
                        const float s = (((v + r1) + r2) + r3) / sqrt_dim;
                        scores[((size_t)b * srows + t) * NBL + col] = col < (pos + t + 1) / 4 ? s : -CUDART_INF_F;
                    }
                }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Top-k. Grid (B * T), 1024 threads: block (b, t) selects the blocks of row t of slot b. With nb = (p_t + 1) / 4 <= 512
// complete blocks it takes them all. Else it takes the 512 largest scores, ties to the lower block index (the order of
// torch.topk on CUDA once -0.0 is +0.0), and writes their indices in ascending order:
//   1. a radix select over the ordered keys (4 passes of 8 bits, a shared histogram each, one column per lane) finds
//      thr, the key of the 512th largest score, and need, the number of keys equal to thr that belong to the top 512;
//   2. a compaction in index order keeps a block if its key is > thr, or == thr and fewer than need equal keys come
//      before it (chunks of 4096 indices, one block-wide scan each).
// sel entries past nsel are -1.
__device__ __forceinline__ uint32_t okey(float f) {  // the order of the floats as unsigned keys; -0 counts as +0
    uint32_t u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int TOPK_THREADS = 1024, TOPK_WARPS = TOPK_THREADS / 32;

__global__ void __launch_bounds__(TOPK_THREADS) k_topk(const float* __restrict__ scores, const int* __restrict__ pos_t, int* __restrict__ sel,
                                                        int* __restrict__ nsel, int T, int NBL, int srows) {
    PDL_TRIGGER();
    const int b = blockIdx.x / T, t = blockIdx.x % T, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nb = min((pos_t[b] + t + 1) / 4, NBL);
    const float* sr = scores + ((size_t)b * srows + t) * NBL;
    int* so = sel + ((size_t)b * srows + t) * TOPK;
    if (nb <= TOPK) {
        if (tid < TOPK) so[tid] = tid < nb ? tid : -1;
        if (tid == 0) nsel[b * srows + t] = nb;
        return;
    }
    __shared__ uint32_t lh[256 * 32], hist[256], wsum[TOPK_WARPS], s_prefix, s_need;
    // the rows are 16-byte aligned when NBL % 4 == 0 (the binding checks it): float4 loads, then the tail
    const int nb4 = nb / 4;
    const float4* sr4 = reinterpret_cast<const float4*>(sr);
    uint32_t prefix = 0, mask = 0, need = TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256 * 32; i += TOPK_THREADS) lh[i] = 0;
        __syncthreads();
        // The scores of a row crowd into a few bins of the high digits. Each lane counts into its own column of lh, so
        // the 32 adds of a warp never meet on one address or bank.
        auto add = [&](float f) {
            const uint32_t u = okey(f);
            if ((u & mask) == prefix) atomicAdd(&lh[((u >> shift) & 255u) * 32 + lane], 1u);
        };
        // one block on one SM reads the whole row: issue 8 loads before the adds (the loop is latency-bound)
        for (int i0 = tid; i0 < nb4; i0 += 8 * TOPK_THREADS) {
            float4 v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int i = i0 + u * TOPK_THREADS;
                v[u] = i < nb4 ? __ldg(sr4 + i) : make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
            }
#pragma unroll
            for (int u = 0; u < 8; ++u)
                if (i0 + u * TOPK_THREADS < nb4) {
                    add(v[u].x);
                    add(v[u].y);
                    add(v[u].z);
                    add(v[u].w);
                }
        }
        if (tid < nb - 4 * nb4) add(sr[4 * nb4 + tid]);
        __syncthreads();
        if (tid < 256) {  // the column sums; thread d starts at column d % 32 (no bank conflicts)
            uint32_t s = 0;
            for (int k = 0; k < 32; ++k) s += lh[tid * 32 + ((k + tid) & 31)];
            hist[tid] = s;
        }
        __syncthreads();
        if (warp == 0) {  // lane l owns the digits 255 - 8 l - k, k = 0 .. 7 (descending)
            uint32_t cnt[8], s = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) s += (cnt[k] = hist[255 - 8 * lane - k]);
            uint32_t incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            uint32_t above = incl - s;  // the matching keys with a larger digit
            if (above < need && need <= incl)
                for (int k = 0; k < 8; ++k) {
                    if (above + cnt[k] >= need) {
                        s_prefix = prefix | ((uint32_t)(255 - 8 * lane - k) << shift);
                        s_need = need - above;
                        break;
                    }
                    above += cnt[k];
                }
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask |= 255u << shift;
    }
    const uint32_t thr = prefix;
    // Compaction in chunks of 4 x 1024 indices: thread tid owns the 4 consecutive indices c0 + 4 tid .. + 3 (one float4).
    // One block scan per chunk of the packed counts (> thr in the low half, == thr in the high half) orders the threads.
    // The loop waits on the latency of its loads: the next chunk's float4 loads before the scan of this one.
    auto load4 = [&](int i0) {
        if (i0 + 3 < nb) return __ldg(sr4 + i0 / 4);
        float f[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) f[e] = i0 + e < nb ? sr[i0 + e] : -CUDART_INF_F;
        return make_float4(f[0], f[1], f[2], f[3]);
    };
    uint32_t out = 0, eq_seen = 0;  // the kept indices and the equal keys before the chunk
    float4 next = load4(4 * tid);
    for (int c0 = 0; c0 < nb && out < TOPK; c0 += 4 * TOPK_THREADS) {  // uniform over the block
        const int i0 = c0 + 4 * tid;
        const float4 cur = next;
        if (c0 + 4 * TOPK_THREADS < nb) next = load4(i0 + 4 * TOPK_THREADS);
        const float f[4] = {cur.x, cur.y, cur.z, cur.w};
        uint32_t k[4], v = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            k[e] = okey(f[e]);
            const bool ok = i0 + e < nb;
            v += (ok && k[e] > thr ? 1u : 0u) + (ok && k[e] == thr ? 0x10000u : 0u);
        }
        uint32_t incl = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t x = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += x;
        }
        if (lane == 31) wsum[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            uint32_t w = wsum[lane];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t x = __shfl_up_sync(0xffffffffu, w, o);
                if (lane >= o) w += x;
            }
            wsum[lane] = w;
        }
        __syncthreads();
        const uint32_t ex = incl - v + (warp ? wsum[warp - 1] : 0u), tot = wsum[TOPK_WARPS - 1];
        const uint32_t rem = need > eq_seen ? need - eq_seen : 0u;  // equal keys still to take
        uint32_t gt_r = ex & 0xffffu, eq_r = ex >> 16;  // the (> thr, == thr) keys of the chunk before this element
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (i0 + e >= nb) break;
            if (k[e] > thr) so[out + gt_r++ + min(eq_r, rem)] = i0 + e;
            else if (k[e] == thr) {
                if (eq_r < rem) so[out + gt_r + eq_r] = i0 + e;
                ++eq_r;
            }
        }
        out += (tot & 0xffffu) + min(tot >> 16, rem);
        eq_seen += tot >> 16;
        __syncthreads();  // wsum is rewritten by the next chunk
    }
    if (tid == 0) nsel[b * srows + t] = TOPK;
}

}  // namespace qsa

cudaError_t launch_qsa_prologue(const void* idx, int ld, const void* qn_w, const float* inv_freq, const int* pos_t, void* ik, void* q_out, int B,
                                int T, int Lmax, int R, float eps, const int* active, const int* rope_delta, const int* rope_pos, int sec_h, int sec_w,
                                cudaStream_t st) {
    qsa::k_prologue<<<dim3(B * T, qsa::HI + 1), qsa::DI, 0, st>>>((const __nv_bfloat16*)idx, ld, (const __nv_bfloat16*)qn_w, inv_freq, pos_t,
                                                                  (__nv_bfloat16*)ik, (__nv_bfloat16*)q_out, T, Lmax, R, eps, active, rope_delta,
                                                                  rope_pos, sec_h, sec_w);
    return cudaGetLastError();
}

cudaError_t launch_qsa_block_keys(const void* ik, void* ikb, const void* kn_w, const float* inv_freq, const int* pos_t, int B, int T, int Lmax,
                                  int R, float eps, const int* active, const int* rope_delta, const int* rope_pos, int sec_h, int sec_w,
                                  cudaStream_t st) {
    qsa::k_block_keys<<<dim3(B, T / 4 + 1), qsa::DI, 0, st>>>((const __nv_bfloat16*)ik, (__nv_bfloat16*)ikb, (const __nv_bfloat16*)kn_w,
                                                              inv_freq, pos_t, T, Lmax, R, eps, active, rope_delta, rope_pos, sec_h, sec_w);
    return cudaGetLastError();
}

cudaError_t launch_qsa_score(const void* q_idx, const void* ikb, const int* pos_t, float* scores, int B, int T, int NBL, int srows,
                             cudaStream_t st) {
    if (T < 1 || T > 16) return cudaErrorInvalidValue;
    const dim3 grid((NBL + qsa::SC_KEYS - 1) / qsa::SC_KEYS, B);
    auto q = (const __nv_bfloat16*)q_idx;
    auto k = (const __nv_bfloat16*)ikb;
    switch ((T + 3) / 4) {
        case 1: qsa::k_score<1><<<grid, qsa::SC_THREADS, 0, st>>>(q, k, pos_t, scores, T, NBL, srows); break;
        case 2: qsa::k_score<2><<<grid, qsa::SC_THREADS, 0, st>>>(q, k, pos_t, scores, T, NBL, srows); break;
        case 3: qsa::k_score<3><<<grid, qsa::SC_THREADS, 0, st>>>(q, k, pos_t, scores, T, NBL, srows); break;
        default: qsa::k_score<4><<<grid, qsa::SC_THREADS, 0, st>>>(q, k, pos_t, scores, T, NBL, srows); break;
    }
    return cudaGetLastError();
}

cudaError_t launch_qsa_topk(const float* scores, const int* pos_t, int* sel, int* nsel, int B, int T, int NBL, int srows, cudaStream_t st) {
    qsa::k_topk<<<B * T, qsa::TOPK_THREADS, 0, st>>>(scores, pos_t, sel, nsel, T, NBL, srows);
    return cudaGetLastError();
}

// The stripe count of qsa_attn: the 2 KV heads x 24 stripes of a token fill the 48 SMs.
int qsa_attn_nb() { return 24; }

// GQA 24 / 2, head dim 256. part_acc: >= B * 24 * T * qsa_attn_nb() * 256 floats, part_ml: as many float2.
cudaError_t launch_qsa_attn(const void* q, const void* kc, const void* vc, const int* sel, const int* nsel, int sel_rows, const int* pos_t, void* out,
                            float* part_acc, void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, float scale, const void* gate, cudaStream_t st) {
    using namespace attn::tc;
    constexpr int G = 12, NB = 24;
    if (Hkv * G != Hq || T < 1 || T > sel_rows) return cudaErrorInvalidValue;
    const dim3 grid(B * T, Hkv, NB);
    if (cudaError_t e = launch_tc<G, 1, NB, true>(grid, st, (const __nv_bfloat16*)q, kc, vc, pos_t, sel, nsel, sel_rows, qsa::TOPK, part_acc,
                                                  part_ml, Hkv, T, Lmax, scale * 1.4426950408889634f))
        return e;
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, NB, (const __nv_bfloat16*)gate,
                                                    Hq, T);
    return cudaGetLastError();
}
