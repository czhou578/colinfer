// attn_tc.cuh -- the tensor-core multi-row decode attention over an fp8 (e4m3, unit scale) KV cache, and the combine of
// its stripe partials. Two files instantiate it:
//   csrc/attn_decode.cu  dense keys: row t of slot b attends to the positions [0, seq_len - (T - 1 - t)).
//   csrc/qsa.cu          a key list per row (Qwen Sparse Attention): the selected 4-token blocks, then the tail.
//
// k              [B, Hkv, Lmax, D] e4m3 cache, with the rows of the new tokens already written (v the same)
// q              [B, Hq, T, D] bf16
// part_acc       fp32 [B * Hq * T * NB, D], part_ml float2 [B * Hq * T * NB]: the partials of the NB stripes, at the
//                index ((b * Hq + h) * T + t) * NB + z
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include "common.cuh"
#include "pdl.cuh"

namespace attn {
using namespace cc;

constexpr int D = HEAD_DIM;

// Folds the NB partials (part_acc [rows, NB, D], part_ml [rows, NB] = (running max, sum)) of each (b, h, t) row in a
// fixed order.
//   gate == nullptr: out[b, h, t, :] (the q layout).
//   gate != nullptr (the q_proj output [B, T, Hq, 2D], with the gate in the second half of each head):
//   out[b, t, h, :] = attn * sigmoid(gate), ready for o_proj.
// static: each kernel file that includes this header has its own copy.
static __global__ void k_combine(const float* __restrict__ part_acc, const float2* __restrict__ part_ml, __nv_bfloat16* __restrict__ out,
                                 int splits, const __nv_bfloat16* __restrict__ gate, int Hq, int T) {
    PDL_TRIGGER();
    const size_t row = blockIdx.x;  // (b, h, t)
    const int d = threadIdx.x;
    float M = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[row * splits + s].x);
    float L = 0.f, A = 0.f;
    if (M > -CUDART_INF_F)
        for (int s = 0; s < splits; ++s) {
            const float2 ml = part_ml[row * splits + s];
            const float c = __expf(ml.x - M);
            L += ml.y * c;
            A += part_acc[(row * splits + s) * D + d] * c;
        }
    const float val = L > 0.f ? A / L : 0.f;
    if (gate == nullptr) {
        out[row * D + d] = __float2bfloat16(val);
    } else {
        const int t = row % T, h = (row / T) % Hq, b = row / ((size_t)T * Hq);
        const size_t o = (((size_t)b * T + t) * Hq + h) * D + d;
        const float g = __bfloat162float(gate[(((size_t)b * T + t) * Hq + h) * 2 * D + D + d]);
        out[o] = __float2bfloat16(val / (1.f + __expf(-g)));
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor-core multi-row decode. One block owns one (slot, KV head, tile stripe) and serves a set of query rows that
// share the keys. These are up to 48 rows, as three m16 tiles of mma.sync.m16n8k16 (f16 x f16 -> fp32):
//   dense (LIST = false): the G heads x T new tokens of the slot; rows beyond 48 take more blocks (grid y).
//   list (LIST = true):   the G heads of one new token, which own one key list (grid x = B * T).
// The keys are a sequence of logical indices i = 0, 1, ... with a cache position each: i itself (dense), or the list
// entry (list).
//
// Bit identity. The result of a row must not depend on how many other rows share the launch. Thus plain decode (T = 1)
// and each verify width give the same bits:
//   - The kernel processes the keys in TK-key tiles at fixed logical indices. Block z owns tiles z, z + NB, z + 2 NB, ...
//     (NB fixed for a head layout), and the combine folds the NB partials in a fixed order.
//   - The mma rows are independent. The softmax of a row is 8 lanes with fixed reduction trees, the same for each row.
//   - A tile that is fully past the length of a row is an exact no-op. (It exists because a longer row of the slot needs
//     it.) Its keys score -inf, p = 0 exactly, the running max does not move, and corr is exactly 1.
// A list that holds the positions 0 .. n - 1 in order thus gives the same bits as the dense kernel at the same NB.
//
// Operands. K stays in the raw cache bytes in shared memory. The B fragment of a lane for k-step j is 4 consecutive head
// dims (16j + 4c .. +3, c = lane % 4). The QK product can take them in any order, if the A fragment of Q uses the same
// order. It does: 8 consecutive bytes of the f16 Q row. e4m3 values are exact in f16.
// The kernel converts V to an f16 [key][dim] tile and reads it with ldmatrix.trans. It rounds P to f16 (~5e-4 relative
// error against fp32 attention, below the rounding of the bf16 output).
//
// Speed (bench/attn_bench.py, 128k context, fp8 KV): 228-236 GB/s for T = 1, 4 and 8. Thus a verify of 8 rows costs the
// same as plain decode. NB = 48 / Hkv: the KV heads x NB stripes of one slot fill the 48 SMs once (one block per SM).
namespace tc {

constexpr int TK = 32, WARPS = 4, QS = D + 8, VS = D + 8, PS = TK + 8, SS = TK + 4, MAXROWS = 48;

// RS: the shared-memory bytes per raw cache row. swz(row, chunk): where the 16-byte chunk of a row lands. The fp8 rows
// stay 256 bytes apart, and the kernel XOR-swizzles the chunks by row instead. (A padding to 272 decreased the cp.async
// streaming from ~232 to ~187 GB/s.) Thus the 8 rows of a K fragment load hit 8 different bank groups.
struct TcFp8 {
    static constexpr int ROW = D, RS = D, CH = ROW / 16;
    static __device__ __forceinline__ int swz(int r, int ch) { return ch ^ (r & 7); }
    static __device__ __forceinline__ void kfrag(const uint8_t* row, int r, int j, int c, uint32_t& b0, uint32_t& b1) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(row + 16 * swz(r, j) + 4 * c);
        b0 = e4m3x2_h2(w);
        b1 = e4m3x2_h2(w >> 16);
    }
    // 8 dims [8u, 8u+8) of one row -> 8 f16
    static __device__ __forceinline__ uint4 vunit(const uint8_t* row, int r, int u) {
        const uint2 w = *reinterpret_cast<const uint2*>(row + 16 * swz(r, u >> 1) + 8 * (u & 1));
        return make_uint4(e4m3x2_h2(w.x), e4m3x2_h2(w.x >> 16), e4m3x2_h2(w.y), e4m3x2_h2(w.y >> 16));
    }
};
// raw K / V tile stages: four where shared memory allows (the 99 KB per block limit), else three or two
template <int MT, typename KV>
constexpr int smem_bytes(int stages) {
    return MT * 16 * QS * 2 + stages * 2 * TK * KV::RS + TK * VS * 2 + MT * 16 * SS * 4 + MT * 16 * PS * 2 + 3 * MT * 16 * 4;
}
template <int MT, typename KV>
constexpr int stages() { return smem_bytes<MT, KV>(4) <= 99 * 1024 ? 4 : smem_bytes<MT, KV>(3) <= 99 * 1024 ? 3 : 2; }

// The key list of one query token (LIST = true): the 4 nsel selected blocks of sel (ascending block indices), then the
// tail positions 4 nbq .. p of the incomplete block (nbq = (p + 1) / 4 complete blocks). Logical key i < 4 nsel is the
// position 4 sel[i / 4] + i % 4.
struct KeyList {
    const int* sel;
    int nfull, tail0, n;  // nfull = 4 nsel, tail0 = 4 nbq, n = the key count
    __device__ __forceinline__ int pos(int i) const { return i < nfull ? 4 * __ldg(sel + (i >> 2)) + (i & 3) : tail0 + (i - nfull); }
};

// Dense (LIST = false): grid (B, Hkv * passes, NB), 128 threads. Block (b, kvh, pass, z) handles the query rows
// r = pass * 48 + [0, 48) of slot b, over the tiles z + n * NB. Row r = t * G + g is query head kvh * G + g of new token t.
// lens = seq_lens [B] (the valid positions after this step).
// List (LIST = true): grid (B * T, Hkv, NB), 128 threads. Block (b, t, kvh, z) handles the G rows of query token t:
// row r = g is query head kvh * G + g. lens = pos_t [B] (token t is at the position p = pos_t[b] + t); sel int32
// [B, sel_rows, 512] and nsel int32 [B, sel_rows] give the selected blocks of each token.
// The partials go to part_acc / part_ml at ((b * Hq + h) * T + t) * NB + z, the layout that k_combine folds (splits = NB).
template <int G, int MT, typename KV, int NB, bool LIST>
__global__ void __launch_bounds__(WARPS * 32) k_tc(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                    const uint8_t* __restrict__ vc, const int* __restrict__ lens,
                                                    const int* __restrict__ sel, const int* __restrict__ nsel, int sel_rows, int sel_cap,
                                                    float* __restrict__ part_acc, float2* __restrict__ part_ml, int Hkv, int T, int Lmax,
                                                    float scale_log2) {
    PDL_TRIGGER();
    constexpr int MP = MT * 16, STAGES = stages<MT, KV>();
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

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, z = blockIdx.z, Hq = Hkv * G;
    int b, kvh, r0, R, sl;  // rows of this block: r0 + [0, R); sl: the key count of the longest row
    KeyList kl{nullptr, 0, 0, 0};
    int tq = 0;  // LIST: the query token of the block
    if constexpr (LIST) {
        b = blockIdx.x / T;
        tq = blockIdx.x % T;
        kvh = blockIdx.y;
        r0 = 0;
        R = G;
        const int p = lens[b] + tq, nbq = (p + 1) / 4, ns = nsel[b * sel_rows + tq];
        kl = KeyList{sel + ((size_t)b * sel_rows + tq) * sel_cap, 4 * ns, 4 * nbq, 4 * ns + (p + 1 - 4 * nbq)};
        sl = kl.n;
    } else {
        b = blockIdx.x;
        kvh = blockIdx.y % Hkv;
        r0 = (blockIdx.y / Hkv) * MAXROWS;
        R = min(G * T - r0, MP);
        sl = lens[b];
    }
    // block row r -> (token, head in the group)
    auto row_t = [&](int r) { if constexpr (LIST) return tq; else return (r0 + r) / G; };
    auto row_g = [&](int r) { if constexpr (LIST) return r; else return (r0 + r) % G; };
    const int ntiles = (sl + TK - 1) / TK;

    for (int idx = tid; idx < MP * (D / 8); idx += WARPS * 32) {
        const int r = idx / (D / 8), u = idx % (D / 8);
        uint4 o = make_uint4(0, 0, 0, 0);
        if (r < R) {
            const int t = row_t(r), g = row_g(r);
            float f[8];
            bf16x8_to_float(*reinterpret_cast<const uint4*>(q + ((size_t)(b * Hq + kvh * G + g) * T + t) * D + u * 8), f);
            uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half2 h = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
                ou[i] = *reinterpret_cast<const uint32_t*>(&h);
            }
        }
        *reinterpret_cast<uint4*>(Qs + r * QS + u * 8) = o;
    }
    for (int idx = tid; idx < MP * PS; idx += WARPS * 32) Ps[idx] = __float2half(0.f);
    for (int r = tid; r < MP; r += WARPS * 32) { m_s[r] = -CUDART_INF_F; l_s[r] = 0.f; corr_s[r] = 1.f; }

    const size_t base = (size_t)(b * Hkv + kvh) * Lmax;
    auto load_tile = [&](int tile, int stage) {
        const int p0 = tile * TK;
        for (int idx = tid; idx < TK * KV::CH; idx += WARPS * 32) {
            const int kr = idx / KV::CH, ch = idx % KV::CH, i = p0 + kr;
            const bool ok = i < sl;  // zero-fill past the key count: stale bytes there must not reach the mma as NaN
            int p = 0;
            if constexpr (LIST) p = ok ? kl.pos(i) : 0;
            else p = ok ? i : 0;
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
        if (z + i * NB < ntiles) load_tile(z + i * NB, i);
        cp_async_commit();
    }
    for (int tile = z, it = 0; tile < ntiles; tile += NB, ++it) {
        const int stage = it % STAGES;
        // refill the stage the previous iteration consumed (its readers passed that iteration's second barrier)
        if (tile + (STAGES - 1) * NB < ntiles) load_tile(tile + (STAGES - 1) * NB, (it + STAGES - 1) % STAGES);
        cp_async_commit();
        cp_async_wait<STAGES - 1>();
        __syncthreads();  // raw tile visible; the previous tile's PV is done with Vh / Ps

        // S = Q K^T: warp w scores keys [8w, 8w + 8) of the tile for all rows
        {
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

        // online softmax: 8 lanes per row (4 keys each), 4 rows per warp at a time
        {
            const int grp = lane >> 3, sub = lane & 7;
            const unsigned gmask = 0xffu << (8 * grp);
            for (int r = warp + WARPS * grp; r < R; r += WARPS * 4) {
                const int len = LIST ? sl : sl - (T - 1 - row_t(r)), p = tile * TK + 4 * sub;
                const float4 sv = *reinterpret_cast<const float4*>(Ss + r * SS + 4 * sub);
                const float x[4] = {p < len ? sv.x * scale_log2 : -CUDART_INF_F, p + 1 < len ? sv.y * scale_log2 : -CUDART_INF_F,
                                    p + 2 < len ? sv.z * scale_log2 : -CUDART_INF_F, p + 3 < len ? sv.w * scale_log2 : -CUDART_INF_F};
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

        // O = O * corr + P V: warp w owns head dims [64w, 64w + 64)
        {
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

    auto prow = [&](int r) {  // partial index of block row r
        return ((size_t)(b * Hq + kvh * G + row_g(r)) * T + row_t(r)) * NB + z;
    };
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = mt * 16 + (lane >> 2) + 8 * h;
            if (r < R) {
                float* dst = part_acc + prow(r) * D + warp * 64 + 2 * (lane & 3);
#pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    *reinterpret_cast<float2*>(dst + nt * 8) = make_float2(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
            }
        }
    // m in natural-log units for k_combine (which folds with __expf)
    for (int r = tid; r < R; r += WARPS * 32) part_ml[prow(r)] = make_float2(m_s[r] * 0.69314718056f, l_s[r]);
}

// Sets the dynamic shared memory limit of one instantiation once, then launches it.
template <int G, int MT, int NB, bool LIST>
cudaError_t launch_tc(dim3 grid, cudaStream_t st, const __nv_bfloat16* q, const void* kc, const void* vc, const int* lens, const int* sel,
                      const int* nsel, int sel_rows, int sel_cap, float* part_acc, void* part_ml, int Hkv, int T, int Lmax, float scale_log2) {
    constexpr int SMEM = smem_bytes<MT, TcFp8>(stages<MT, TcFp8>());
    static bool init = false;
    if (!init) {
        if (cudaError_t e = cudaFuncSetAttribute(k_tc<G, MT, TcFp8, NB, LIST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)) return e;
        init = true;
    }
    k_tc<G, MT, TcFp8, NB, LIST><<<grid, WARPS * 32, SMEM, st>>>(q, (const uint8_t*)kc, (const uint8_t*)vc, lens, sel, nsel, sel_rows, sel_cap,
                                                                 part_acc, (float2*)part_ml, Hkv, T, Lmax, scale_log2);
    return cudaGetLastError();
}

}  // namespace tc
}  // namespace attn
