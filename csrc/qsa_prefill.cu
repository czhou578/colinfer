// qsa_prefill.cu -- Qwen Sparse Attention over a prefill chunk of one slot (docs/history/qwen4exp_plan.md section 5.3,
// milestone M2): T rows at the positions pos0 .. pos0 + T - 1, T up to 8192. The bindings are in qsa_bind.cpp, the
// Python wrappers in engine/model/qsa_kernels.py (prefill_select, prefill_attend). Tests: tests/test_qsa_prefill.py.
//
//   k_score_rows  the block scores of all T rows in one launch: the computation of qsa.cu k_score<4> (16 rows per
//                 group, the same mma order and the same epilogue), with the row group on grid y. A row's scores are
//                 bit-identical to the decode scores of that row, so qsa_topk selects the same blocks.
//   k_union       per group of 4 consecutive rows: the union of their selected blocks and their tail blocks, ascending,
//                 each entry (block << 4) | the mask of the rows that attend to it.
//   k_union_attn  the attention of a group of 4 rows: the 12 query heads of one KV head x 4 rows = 48 rows (three m16
//                 tiles) over the keys of the group's union, with each row masked to its own blocks and tail. The
//                 tensor-core core is the one of the decode kernel (attn_tc.cuh k_tc: f16 Q, the e4m3 keys converted
//                 exactly to f16, P in f16, fp32 accumulation), with one stripe and the output gate in the epilogue.
//
// The union kernel replaces the dense FA2 with a custom mask (cost: every row times the whole prefix) and the decode
// kernel per row (cost: 24 stripe partials per row and the K / V of every row read on its own). The 4 rows of a group
// share the K / V reads of their common blocks, and the 12 heads of a KV head share them all.
#include "attn_tc.cuh"

namespace qsap {
using namespace cc;

constexpr int DI = 128, HI = 4, TOPK = 512;
constexpr int GROUP = 4;  // rows per union group
constexpr int UCAP = GROUP * (TOPK + 1);  // union entries per group at most: 512 selected blocks and a tail block per row

// ------------------------------------------------------------------------------------------------------------------
// Block scores of T rows of one slot. Grid (ceil(NBL / 256), ceil(T / 16)), 128 threads. Group y holds the rows
// t = 16 y .. 16 y + 15 at the positions pos_t[0] + t. The body is qsa.cu k_score<4> with b = 0 and the group's row
// offset: keep the two in step (tests/test_qsa_prefill.py compares the bits).
constexpr int SC_KEYS = 256, SC_THREADS = 128, QSI = DI + 32;

__global__ void __launch_bounds__(SC_THREADS) k_score_rows(const __nv_bfloat16* __restrict__ q_idx, const __nv_bfloat16* __restrict__ ikb,
                                                            const int* __restrict__ pos_t, float* __restrict__ scores, int T_all, int NBL, int srows) {
    constexpr int MT = 4;
    const int t0 = blockIdx.y * 16, T = min(16, T_all - t0);
    const int j0 = blockIdx.x * SC_KEYS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pos = pos_t[0] + t0, nbmax = min((pos + T) / 4, NBL);
    if (j0 >= nbmax) return;
    __shared__ __align__(16) __nv_bfloat16 Qs[MT * 16 * QSI];
    for (int idx = tid; idx < MT * 16 * (DI / 8); idx += SC_THREADS) {
        const int r = idx / (DI / 8), u = idx % (DI / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < 4 * T) v = *reinterpret_cast<const uint4*>(q_idx + ((size_t)t0 * HI + r) * DI + u * 8);
        *reinterpret_cast<uint4*>(Qs + r * QSI + u * 8) = v;
    }
    __syncthreads();
    const int g = lane >> 2, c = lane & 3;
    const float sqrt_dim = 11.313708498984761f;
    for (int round = 0; round < 2; ++round) {
        const int jw = j0 + warp * 64 + round * 32;
        if (jw >= nbmax) break;
        uint4 kb[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key = jw + nt * 8 + g;
            const __nv_bfloat16* kp = ikb + (size_t)key * DI + 8 * c;
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
                    const float v = acc[nt][e] > 0.f ? acc[nt][e] : 0.f;
                    const float r1 = __shfl_down_sync(0xffffffffu, v, 4), r2 = __shfl_down_sync(0xffffffffu, v, 8),
                                r3 = __shfl_down_sync(0xffffffffu, v, 12);
                    const int t = mt * 4 + (g >> 2) + (e >= 2 ? 2 : 0), col = jw + nt * 8 + 2 * c + (e & 1);
                    if ((g & 3) == 0 && t < T && col < NBL) {
                        const float s = (((v + r1) + r2) + r3) / sqrt_dim;
                        scores[((size_t)(t0 + t)) * NBL + col] = col < (pos + t + 1) / 4 ? s : -CUDART_INF_F;
                    }
                }
        }
    }
    (void)srows;
}

// ------------------------------------------------------------------------------------------------------------------
// The union of a group. Grid (ceil(T / 4)), 256 threads; dynamic shared memory: 4 bitmaps of W = nw words over the
// blocks [0, 32 W). Row tt of the group (token t = 4 gi + tt at p = pos0 + t, nbq = (p + 1) / 4 complete blocks) marks
// its nsel selected blocks and, when (p + 1) % 4 > 0, its tail block nbq. The entries go out in ascending block order:
// ul[gi, k] = (block << 4) | (the rows that mark it), uc[gi] = the count.
constexpr int UN_THREADS = 256;

__global__ void __launch_bounds__(UN_THREADS) k_union(const int* __restrict__ sel, const int* __restrict__ nsel, const int* __restrict__ pos_t,
                                                       int* __restrict__ ul, int* __restrict__ uc, int T, int nw) {
    extern __shared__ uint32_t bm[];  // [GROUP][nw]
    __shared__ uint32_t wsum[UN_THREADS / 32];
    const int gi = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ntok = min(GROUP, T - gi * GROUP), pos0 = pos_t[0];
    for (int i = tid; i < GROUP * nw; i += UN_THREADS) bm[i] = 0u;
    __syncthreads();
    for (int tt = 0; tt < ntok; ++tt) {
        const int t = gi * GROUP + tt, p = pos0 + t, nbq = (p + 1) / 4, n = nsel[t];
        uint32_t* m = bm + tt * nw;
        for (int i = tid; i < n; i += UN_THREADS) {
            const int b = sel[(size_t)t * TOPK + i];
            atomicOr(m + (b >> 5), 1u << (b & 31));
        }
        if (tid == 0 && (p + 1) % 4) atomicOr(m + (nbq >> 5), 1u << (nbq & 31));
    }
    __syncthreads();
    // thread tid owns the words [w0, w1): count, block-wide exclusive scan, then emit in order
    const int per = (nw + UN_THREADS - 1) / UN_THREADS, w0 = min(nw, tid * per), w1 = min(nw, w0 + per);
    uint32_t cnt = 0;
    for (int w = w0; w < w1; ++w) cnt += __popc(bm[w] | bm[nw + w] | bm[2 * nw + w] | bm[3 * nw + w]);
    uint32_t incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t x = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += x;
    }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    uint32_t before = 0, total = 0;
    for (int w = 0; w < UN_THREADS / 32; ++w) {
        if (w < warp) before += wsum[w];
        total += wsum[w];
    }
    uint32_t k = before + incl - cnt;
    int* out = ul + (size_t)gi * UCAP;
    for (int w = w0; w < w1; ++w) {
        const uint32_t a = bm[w], b = bm[nw + w], c = bm[2 * nw + w], d = bm[3 * nw + w];
        uint32_t o = a | b | c | d;
        while (o) {
            const int bit = __ffs(o) - 1;
            o &= o - 1;
            const uint32_t msk = ((a >> bit) & 1u) | (((b >> bit) & 1u) << 1) | (((c >> bit) & 1u) << 2) | (((d >> bit) & 1u) << 3);
            out[k++] = ((w * 32 + bit) << 4) | (int)msk;
        }
    }
    if (tid == 0) uc[gi] = (int)total;
}

// ------------------------------------------------------------------------------------------------------------------
// The attention of the groups. Grid (ceil(T / 4), Hkv), 128 threads. Block (gi, kvh) owns the rows r = tt * G + g
// (row tt of the group, query head kvh * G + g), R = G * ntok <= 48. Logical key i is the position 4 (ul[i / 4] >> 4) +
// i % 4. In the softmax, the lane of a row that holds the 4 keys of union entry e masks them for its row tt: a key is
// in when bit tt of e is set and the block is complete for the row (block < nbq) or the key is in the row's tail
// (key % 4 < (p + 1) % 4). The rest is attn_tc.cuh k_tc with LIST and NB = 1; the epilogue writes
// out[t, h] = bf16((acc / l) * sigmoid(gate)) (k_combine with one split). gate: the q rows [T, Hq, 2 D] (the gate in the
// second half of each head), ldg elements apart.
using attn::tc::TcFp8;
constexpr int G = 12, MT = 3, MP = MT * 16, TK = attn::tc::TK, WARPS = attn::tc::WARPS, QS = attn::tc::QS, VS = attn::tc::VS,
              SS = attn::tc::SS, PS = attn::tc::PS, D = attn::D;
constexpr int STAGES = attn::tc::stages<MT, TcFp8>();
constexpr int SMEM = attn::tc::smem_bytes<MT, TcFp8>(STAGES);

__global__ void __launch_bounds__(WARPS * 32) k_union_attn(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                            const uint8_t* __restrict__ vc, const int* __restrict__ pos_t,
                                                            const int* __restrict__ ul, const int* __restrict__ uc,
                                                            const __nv_bfloat16* __restrict__ gate, int ldg, __nv_bfloat16* __restrict__ out,
                                                            int Hkv, int T, int Lmax, float scale_log2) {
    using KV = TcFp8;
    extern __shared__ __align__(16) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    uint8_t* Kraw = smem + MP * QS * 2;
    uint8_t* Vraw = Kraw + STAGES * TK * KV::RS;
    __half* Vh = reinterpret_cast<__half*>(Vraw + STAGES * TK * KV::RS);
    float* Ss = reinterpret_cast<float*>(Vh + TK * VS);
    __half* Ps = reinterpret_cast<__half*>(Ss + MP * SS);
    float* m_s = reinterpret_cast<float*>(Ps + MP * PS);
    float* l_s = m_s + MP;
    float* corr_s = l_s + MP;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = blockIdx.x, kvh = blockIdx.y, Hq = Hkv * G;
    const int ntok = min(GROUP, T - gi * GROUP), R = G * ntok, pos0 = pos_t[0];
    const int* ue = ul + (size_t)gi * UCAP;
    const int nu = uc[gi], sl = 4 * nu, ntiles = (sl + TK - 1) / TK;

    for (int idx = tid; idx < MP * (D / 8); idx += WARPS * 32) {
        const int r = idx / (D / 8), u = idx % (D / 8);
        uint4 o = make_uint4(0, 0, 0, 0);
        if (r < R) {
            const int t = gi * GROUP + r / G, h = kvh * G + r % G;
            float f[8];
            bf16x8_to_float(*reinterpret_cast<const uint4*>(q + ((size_t)h * T + t) * D + u * 8), f);
            uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half2 hh = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
                ou[i] = *reinterpret_cast<const uint32_t*>(&hh);
            }
        }
        *reinterpret_cast<uint4*>(Qs + r * QS + u * 8) = o;
    }
    for (int idx = tid; idx < MP * PS; idx += WARPS * 32) Ps[idx] = __float2half(0.f);
    for (int r = tid; r < MP; r += WARPS * 32) { m_s[r] = -CUDART_INF_F; l_s[r] = 0.f; corr_s[r] = 1.f; }

    const size_t base = (size_t)kvh * Lmax;
    auto load_tile = [&](int tile, int stage) {
        const int p0 = tile * TK;
        for (int idx = tid; idx < TK * KV::CH; idx += WARPS * 32) {
            const int kr = idx / KV::CH, ch = idx % KV::CH, i = p0 + kr;
            const bool ok = i < sl;
            const int p = ok ? 4 * (__ldg(ue + (i >> 2)) >> 4) + (i & 3) : 0;
            const size_t off = (base + p) * KV::ROW + ch * 16;
            cp_async16(Kraw + (stage * TK + kr) * KV::RS + KV::swz(kr, ch) * 16, kc + off, ok);
            cp_async16(Vraw + (stage * TK + kr) * KV::RS + KV::swz(kr, ch) * 16, vc + off, ok);
        }
    };

    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (i < ntiles) load_tile(i, i);
        cp_async_commit();
    }
    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile % STAGES;
        if (tile + STAGES - 1 < ntiles) load_tile(tile + STAGES - 1, (tile + STAGES - 1) % STAGES);
        cp_async_commit();
        cp_async_wait<STAGES - 1>();
        __syncthreads();

        {  // S = Q K^T: warp w scores keys [8w, 8w + 8) of the tile for all rows
            const int kr = warp * 8 + (lane >> 2);
            const uint8_t* krow = Kraw + (stage * TK + kr) * KV::RS;
            float s[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
#pragma unroll
            for (int j = 0; j < D / 16; ++j) {
                uint32_t b0, b1;
                KV::kfrag(krow, kr, j, lane & 3, b0, b1);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const __half* qa = Qs + (mt * 16 + (lane >> 2)) * QS + 16 * j + 4 * (lane & 3);
                    const uint2 x0 = *reinterpret_cast<const uint2*>(qa), x1 = *reinterpret_cast<const uint2*>(qa + 8 * QS);
                    const uint32_t a[4] = {x0.x, x1.x, x0.y, x1.y};
                    mma_f16(s[mt], a, b0, b1);
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float* srow = Ss + (mt * 16 + (lane >> 2)) * SS + warp * 8 + 2 * (lane & 3);
                srow[0] = s[mt][0];
                srow[1] = s[mt][1];
                srow[8 * SS] = s[mt][2];
                srow[8 * SS + 1] = s[mt][3];
            }
        }
        for (int idx = tid; idx < TK * (D / 8); idx += WARPS * 32) {
            const int kr = idx / (D / 8), u = idx % (D / 8);
            *reinterpret_cast<uint4*>(Vh + kr * VS + u * 8) = KV::vunit(Vraw + (stage * TK + kr) * KV::RS, kr, u);
        }
        __syncthreads();

        {  // online softmax: 8 lanes per row, lane sub holds the 4 keys of union entry tile * 8 + sub
            const int grp = lane >> 3, sub = lane & 7;
            const unsigned gmask = 0xffu << (8 * grp);
            const int ei = tile * (TK / 4) + sub;
            const int e = ei < nu ? __ldg(ue + ei) : 0;
            for (int r = warp + WARPS * grp; r < R; r += WARPS * 4) {
                const int tt = r / G, p = pos0 + gi * GROUP + tt, nbq = (p + 1) / 4, rem = (p + 1) % 4, blk = e >> 4;
                const bool on = ei < nu && ((e >> tt) & 1);
                const int nk = !on ? 0 : blk < nbq ? 4 : rem;  // the keys of the entry this row attends to
                const float4 sv = *reinterpret_cast<const float4*>(Ss + r * SS + 4 * sub);
                const float x[4] = {0 < nk ? sv.x * scale_log2 : -CUDART_INF_F, 1 < nk ? sv.y * scale_log2 : -CUDART_INF_F,
                                    2 < nk ? sv.z * scale_log2 : -CUDART_INF_F, 3 < nk ? sv.w * scale_log2 : -CUDART_INF_F};
                float tmax = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(gmask, tmax, o));
                const float mo = m_s[r], mn = fmaxf(mo, tmax);
                const float corr = mn == mo ? 1.f : ex2(mo - mn);
                __half ph[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) ph[i] = __float2half_rn(x[i] == -CUDART_INF_F ? 0.f : ex2(x[i] - mn));
                float ps = (__half2float(ph[0]) + __half2float(ph[1])) + (__half2float(ph[2]) + __half2float(ph[3]));
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) ps += __shfl_xor_sync(gmask, ps, o);
                *reinterpret_cast<uint2*>(Ps + r * PS + 4 * sub) = *reinterpret_cast<const uint2*>(ph);
                __syncwarp(gmask);
                if (sub == 0) { m_s[r] = mn; l_s[r] = l_s[r] * corr + ps; corr_s[r] = corr; }
            }
        }
        __syncthreads();

        {  // O = O * corr + P V: warp w owns head dims [64w, 64w + 64)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float c0 = corr_s[mt * 16 + (lane >> 2)], c1 = corr_s[mt * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    acc[mt][nt][0] *= c0;
                    acc[mt][nt][1] *= c0;
                    acc[mt][nt][2] *= c1;
                    acc[mt][nt][3] *= c1;
                }
            }
#pragma unroll
            for (int ks = 0; ks < TK / 16; ++ks) {
                uint32_t a[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                    ldsm_x4(a[mt], Ps + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * PS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int np = 0; np < 4; ++np) {
                    uint32_t bv[4];
                    ldsm_x4_t(bv, Vh + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * VS + warp * 64 + np * 16 + (lane >> 4) * 8);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        mma_f16(acc[mt][2 * np], a[mt], bv[0], bv[1]);
                        mma_f16(acc[mt][2 * np + 1], a[mt], bv[2], bv[3]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int r = mt * 16 + (lane >> 2) + 8 * hh;
            if (r < R) {
                const int t = gi * GROUP + r / G, h = kvh * G + r % G;
                const float L = l_s[r];
                const int d0 = warp * 64 + 2 * (lane & 3);
                const __nv_bfloat16* gr = gate + (size_t)t * ldg + h * 2 * D + D + d0;
                __nv_bfloat16* dst = out + ((size_t)t * Hq + h) * D + d0;
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float v0 = L > 0.f ? acc[mt][nt][2 * hh] / L : 0.f, v1 = L > 0.f ? acc[mt][nt][2 * hh + 1] / L : 0.f;
                    const float g0 = __bfloat162float(gr[nt * 8]), g1 = __bfloat162float(gr[nt * 8 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(dst + nt * 8) = __floats2bfloat162_rn(v0 / (1.f + __expf(-g0)), v1 / (1.f + __expf(-g1)));
                }
            }
        }
}

}  // namespace qsap

// All T rows of a chunk, one launch: scores fp32 [T, NBL] (row t at pos_t[0] + t; the blocks no row sees are not
// written).
cudaError_t launch_qsa_score_rows(const void* q_idx, const void* ikb, const int* pos_t, float* scores, int T, int NBL, cudaStream_t st) {
    if (T < 1) return cudaErrorInvalidValue;
    const dim3 grid((NBL + qsap::SC_KEYS - 1) / qsap::SC_KEYS, (T + 15) / 16);
    qsap::k_score_rows<<<grid, qsap::SC_THREADS, 0, st>>>((const __nv_bfloat16*)q_idx, (const __nv_bfloat16*)ikb, pos_t, scores, T, NBL, T);
    return cudaGetLastError();
}

int qsa_union_cap() { return qsap::UCAP; }

// ul int32 [ceil(T / 4), qsa_union_cap()], uc int32 [ceil(T / 4)]; nblocks: an upper bound of the block indices + 1
// ((pos0 + T) / 4 + 1).
cudaError_t launch_qsa_union(const int* sel, const int* nsel, const int* pos_t, int* ul, int* uc, int T, int nblocks, cudaStream_t st) {
    const int nw = (nblocks + 31) / 32;
    const size_t smem = (size_t)qsap::GROUP * nw * 4;
    static int set = 0;
    if ((int)smem > set) {
        if (cudaError_t e = cudaFuncSetAttribute(qsap::k_union, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)) return e;
        set = (int)smem;
    }
    qsap::k_union<<<(T + qsap::GROUP - 1) / qsap::GROUP, qsap::UN_THREADS, smem, st>>>(sel, nsel, pos_t, ul, uc, T, nw);
    return cudaGetLastError();
}

// q bf16 [1, Hq = 12 Hkv, T, 256]; caches e4m3 [1, Hkv, Lmax, 256]; gate rows [T, Hq, 512] ldg elements apart;
// out bf16 [T, Hq * 256].
cudaError_t launch_qsa_union_attn(const void* q, const void* kc, const void* vc, const int* pos_t, const int* ul, const int* uc, const void* gate,
                                  int ldg, void* out, int Hq, int Hkv, int T, int Lmax, float scale, cudaStream_t st) {
    if (Hq != qsap::G * Hkv || T < 1) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        if (cudaError_t e = cudaFuncSetAttribute(qsap::k_union_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, qsap::SMEM)) return e;
        init = true;
    }
    const dim3 grid((T + qsap::GROUP - 1) / qsap::GROUP, Hkv);
    qsap::k_union_attn<<<grid, qsap::WARPS * 32, qsap::SMEM, st>>>((const __nv_bfloat16*)q, (const uint8_t*)kc, (const uint8_t*)vc, pos_t, ul, uc,
                                                                    (const __nv_bfloat16*)gate, ldg, (__nv_bfloat16*)out, Hkv, T, Lmax,
                                                                    scale * 1.4426950408889634f);
    return cudaGetLastError();
}
