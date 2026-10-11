// attn_decode.cu -- decode attention for the 16 gated full-attention layers, over an fp8 (e4m3, unit scale) KV cache.
//
//   k_prologue  per new token row: q / k zero-centered RMSNorm + partial RoPE. It writes k and v to the cache at the
//               device position of the slot (so the launch shape is fixed, and a CUDA graph can hold the step), and q
//               to q_out.
//   tc::k_tc    multi-row tensor-core attention: one pass over the KV of a slot for all of its query rows (G heads x T
//               new tokens), split over NB key-tile stripes.
//   k_combine   folds the partials of the stripes in a fixed order, times sigmoid(gate) (the output gate), ready for
//               o_proj.
//
// q        [B, Hq, T, D] bf16, from k_prologue
// k/v      [B, Hkv, Lmax, D] e4m3 cache, with the rows of the new tokens already written
// seq_lens [B] int32 on the device: the valid positions after this step (pos + T). Row t of slot b attends to positions
//          [0, seq_len - (T - 1 - t)) (causal among the new rows).
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

constexpr int D = 256;

// e4m3 cache row writer: saturating (values beyond +-448 clip)
struct KvFp8 {
    using Elem = uint8_t;
    static constexpr int ROW = D;
    static __device__ __forceinline__ void store_row(Elem* row, int d, float v) {
        row[d] = (Elem)__nv_cvt_float_to_fp8(fminf(fmaxf(v, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
    }
};

// Folds the NB partials (part_acc [rows, NB, D], part_ml [rows, NB] = (running max, sum)) of each (b, h, t) row in a
// fixed order.
//   gate == nullptr: out[b, h, t, :] (the q layout).
//   gate != nullptr (the q_proj output [B, T, Hq, 2D], with the gate in the second half of each head):
//   out[b, t, h, :] = attn * sigmoid(gate), ready for o_proj.
__global__ void k_combine(const float* __restrict__ part_acc, const float2* __restrict__ part_ml, __nv_bfloat16* __restrict__ out, int splits,
                          const __nv_bfloat16* __restrict__ gate, int Hq, int T) {
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

// Attention prologue for T new rows per slot. One block (D threads) per (b, t, head) runs over Hq q-heads, then Hkv
// k-heads, then Hkv v-heads:
//   q:  zero-centered RMSNorm (q_norm) + partial RoPE on the first R dims  -> q_out [B, Hq, T, D]
//   k:  RMSNorm (k_norm) + RoPE, written to k_cache[b, h, pos_t[b] + t]   (saturating e4m3)
//   v:  written to v_cache[b, h, pos_t[b] + t]
// qp: q_proj output [B, T, Hq, 2D] (q in the first D of each head). kp, vp: [B, T, Hkv, D].
__global__ void k_prologue(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp, const __nv_bfloat16* __restrict__ vp, int ldq, int ldk, int ldv,
                           const __nv_bfloat16* __restrict__ qn_w, const __nv_bfloat16* __restrict__ kn_w, const float* __restrict__ inv_freq,
                           const int* __restrict__ pos_t, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                           __nv_bfloat16* __restrict__ q_out, int Hq, int Hkv, int T, int Lmax, int R, float eps, const int* __restrict__ active) {
    PDL_TRIGGER();
    const int bt = blockIdx.x, b = bt / T, t = bt % T, hh = blockIdx.y, d = threadIdx.x;
    const bool upd = active == nullptr || active[b] != 0;  // inactive slots: no KV write
    const int pos = pos_t[b] + t;
    __shared__ float xs[D], red[D / 32];
    float x;
    const __nv_bfloat16* w;
    // token row (b, t) of q / k / v starts at (b * T + t) * ld{q,k,v} (row strides of the projection outputs)
    const size_t row = (size_t)b * T + t;
    if (hh < Hq) { x = __bfloat162float(qp[row * ldq + (size_t)hh * 2 * D + d]); w = qn_w; }
    else if (hh < Hq + Hkv) { x = __bfloat162float(kp[row * ldk + (size_t)(hh - Hq) * D + d]); w = kn_w; }
    else {
        const int h = hh - Hq - Hkv;
        if (upd) KvFp8::store_row(vc + (((size_t)b * Hkv + h) * Lmax + pos) * D, d, __bfloat162float(vp[row * ldv + (size_t)h * D + d]));
        return;
    }
    float ss = x * x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < D / 32; ++i) tot += red[i];
    const float xn = __bfloat162float(__float2bfloat16(x * rsqrtf(tot / D + eps) * (1.f + __bfloat162float(w[d]))));
    xs[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < R) {
        const int half = R / 2, i = d % half;
        const float ang = (float)pos * inv_freq[i];
        float sn, cs;
        sincosf(ang, &sn, &cs);
        y = d < half ? xn * cs - xs[d + half] * sn : xn * cs + xs[d - half] * sn;
    }
    if (hh < Hq) q_out[(((size_t)b * Hq + hh) * T + t) * D + d] = __float2bfloat16(y);
    else if (upd) KvFp8::store_row(kc + (((size_t)b * Hkv + (hh - Hq)) * Lmax + pos) * D, d, __bfloat162float(__float2bfloat16(y)));
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor-core multi-row decode. One block owns one (slot, KV head, tile stripe) and serves all query rows of the slot.
// These are up to 48 rows (G heads x T rows), as three m16 tiles of mma.sync.m16n8k16 (f16 x f16 -> fp32). Rows beyond
// 48 take more blocks (grid y).
//
// Bit identity. The result of a row must not depend on how many other rows share the launch. Thus plain decode (T = 1)
// and each verify width give the same bits:
//   - The kernel processes the keys in TK-key tiles at fixed positions. Block z owns tiles z, z + NB, z + 2 NB, ... (NB
//     fixed), and the combine folds the NB partials in a fixed order.
//   - The mma rows are independent. The softmax of a row is 8 lanes with fixed reduction trees, the same for each row.
//   - A tile that is fully past the length of a row is an exact no-op. (It exists because a longer row of the slot needs
//     it.) Its keys score -inf, p = 0 exactly, the running max does not move, and corr is exactly 1.
//
// Operands. K stays in the raw cache bytes in shared memory. The B fragment of a lane for k-step j is 4 consecutive head
// dims (16j + 4c .. +3, c = lane % 4). The QK product can take them in any order, if the A fragment of Q uses the same
// order. It does: 8 consecutive bytes of the f16 Q row. e4m3 values are exact in f16.
// The kernel converts V to an f16 [key][dim] tile and reads it with ldmatrix.trans. It rounds P to f16 (~5e-4 relative
// error against fp32 attention, below the rounding of the bf16 output).
//
// Speed (bench/attn_bench.py, 128k context, fp8 KV): 228-236 GB/s for T = 1, 4 and 8. Thus a verify of 8 rows costs the
// same as plain decode. NB = 12: the 4 KV heads x 12 stripes of one slot fill the 48 SMs once (one block per SM).
namespace tc {

constexpr int TK = 32, NB = 12, WARPS = 4, QS = D + 8, VS = D + 8, PS = TK + 8, SS = TK + 4, MAXROWS = 48;

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

// Grid (B, Hkv * passes, NB), 128 threads. Block (b, kvh, pass, z) handles the query rows r = pass * 48 + [0, 48) of
// slot b, over the tiles z + n * NB. Row r = t * G + g is query head kvh * G + g of new token t. The partials go to
// part_acc / part_ml at ((b * Hq + h) * T + t) * NB + z, the layout that k_combine folds (splits = NB).
template <int G, int MT, typename KV>
__global__ void __launch_bounds__(WARPS * 32) k_tc(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                    const uint8_t* __restrict__ vc, const int* __restrict__ seq_lens,
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

    const int b = blockIdx.x, kvh = blockIdx.y % Hkv, pass = blockIdx.y / Hkv, z = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int Hq = Hkv * G, r0 = pass * MAXROWS, R = min(G * T - r0, MP);  // rows of this block: r0 + [0, R)
    const int sl = seq_lens[b];
    const int ntiles = (sl + TK - 1) / TK;

    for (int idx = tid; idx < MP * (D / 8); idx += WARPS * 32) {
        const int r = idx / (D / 8), u = idx % (D / 8);
        uint4 o = make_uint4(0, 0, 0, 0);
        if (r < R) {
            const int rr = r0 + r, t = rr / G, g = rr % G;
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
            const int kr = idx / KV::CH, ch = idx % KV::CH, p = p0 + kr;
            const bool ok = p < sl;  // zero-fill past the slot's length: stale bytes there must not reach the mma as NaN
            const size_t off = (base + (ok ? p : 0)) * KV::ROW + ch * 16;
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
                const int len = sl - (T - 1 - (r0 + r) / G), p = tile * TK + 4 * sub;
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
        const int rr = r0 + r, t = rr / G, g = rr % G;
        return ((size_t)(b * Hq + kvh * G + g) * T + t) * NB + z;
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

}  // namespace tc
}  // namespace attn

int attn_decode_tc_nb() { return attn::tc::NB; }

// part_acc: >= B * Hq * T * attn_decode_tc_nb() * 256 floats, part_ml: as many float2. gate: the q_proj output
// [B, T, Hq, 2D] (gate in the second half of each head); out [B, T, Hq * D] = attention * sigmoid(gate).
cudaError_t launch_attn_decode_tc(const void* q, const void* kc, const void* vc, const int* seq_lens, void* out, float* part_acc,
                                  void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, int Dh, float scale, const void* gate, cudaStream_t st) {
    using namespace attn::tc;
    if (Dh != attn::D || Hq % Hkv) return cudaErrorInvalidValue;
    const int G = Hq / Hkv, rows = G * T, passes = (rows + MAXROWS - 1) / MAXROWS;
    const int MT = (min(rows, MAXROWS) + 15) / 16;
    dim3 grid(B, Hkv * passes, NB), block(WARPS * 32);
    const float sl2 = scale * 1.4426950408889634f;
    auto qb = (const __nv_bfloat16*)q;
    cudaError_t err = cudaErrorInvalidValue;
#define TC_LAUNCH(GG, MM)                                                                                                          \
    do {                                                                                                                           \
        constexpr int SMEM = smem_bytes<MM, TcFp8>(stages<MM, TcFp8>());                                                           \
        static bool init = false;                                                                                                  \
        if (!init) {                                                                                                               \
            cudaFuncSetAttribute(k_tc<GG, MM, TcFp8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                          \
            init = true;                                                                                                           \
        }                                                                                                                          \
        k_tc<GG, MM, TcFp8><<<grid, block, SMEM, st>>>(qb, (const uint8_t*)kc, (const uint8_t*)vc, seq_lens, part_acc,             \
                                                        (float2*)part_ml, Hkv, T, Lmax, sl2);                                      \
        err = cudaSuccess;                                                                                                         \
    } while (0)
#define TC_MT(GG)                         \
    switch (MT) {                         \
        case 1: TC_LAUNCH(GG, 1); break;  \
        case 2: TC_LAUNCH(GG, 2); break;  \
        case 3: TC_LAUNCH(GG, 3); break;  \
    }
    if (G == 6) TC_MT(6)  // Qwen3.8-27B: 24 query heads over 4 KV heads (the target and the MTP layer)
#undef TC_MT
#undef TC_LAUNCH
    if (err != cudaSuccess) return err;
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, NB, (const __nv_bfloat16*)gate,
                                                    Hq, T);
    return cudaGetLastError();
}

cudaError_t launch_attn_prologue(const void* qp, const void* kp, const void* vp, int ldq, int ldk, int ldv, const void* qn_w, const void* kn_w,
                                 const float* inv_freq, const int* pos_t, void* kc, void* vc, void* q_out, int B, int T, int Hq, int Hkv, int Lmax,
                                 int R, float eps, const int* active, cudaStream_t st) {
    attn::k_prologue<<<dim3(B * T, Hq + 2 * Hkv), attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp,
                                                                    ldq, ldk, ldv, (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq,
                                                                    pos_t, (uint8_t*)kc, (uint8_t*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R,
                                                                    eps, active);
    return cudaGetLastError();
}
