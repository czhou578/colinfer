// attn_decode.cu -- decode attention for the 16 gated full-attention layers, over an fp8 (e4m3, unit scale) KV cache.
//
//   k_prologue  per new token row: q / k zero-centered RMSNorm + partial RoPE. It writes k and v to the cache at the
//               device position of the slot (so the launch shape is fixed, and a CUDA graph can hold the step), and q
//               to q_out.
//   tc::k_tc    multi-row tensor-core attention (csrc/attn_tc.cuh): one pass over the KV of a slot for all of its query
//               rows (G heads x T new tokens), split over NB key-tile stripes.
//   k_combine   folds the partials of the stripes in a fixed order, times sigmoid(gate) (the output gate), ready for
//               o_proj (csrc/attn_tc.cuh).
// Head layouts: G = 6 query heads per KV head with NB = 12 (Qwen3.8-27B: 24 / 4) and G = 12 with NB = 24 (qwen4_exp:
// 24 / 2). NB = 48 / Hkv: the stripes of the KV heads of one slot fill the 48 SMs at T = 1.
//
// q        [B, Hq, T, D] bf16, from k_prologue
// k/v      [B, Hkv, Lmax, D] e4m3 cache, with the rows of the new tokens already written
// seq_lens [B] int32 on the device: the valid positions after this step (pos + T). Row t of slot b attends to positions
//          [0, seq_len - (T - 1 - t)) (causal among the new rows).
#include "attn_tc.cuh"

namespace attn {

// e4m3 cache row writer: saturating (values beyond +-448 clip)
struct KvFp8 {
    using Elem = uint8_t;
    static constexpr int ROW = D;
    static __device__ __forceinline__ void store_row(Elem* row, int d, float v) {
        row[d] = (Elem)__nv_cvt_float_to_fp8(fminf(fmaxf(v, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
    }
};

// Attention prologue for T new rows per slot. One block (D threads) per (b, t, head) runs over Hq q-heads, then Hkv
// k-heads, then Hkv v-heads:
//   q:  zero-centered RMSNorm (q_norm) + partial RoPE on the first R dims  -> q_out [B, Hq, T, D]
//   k:  RMSNorm (k_norm) + RoPE, written to k_cache[b, h, pos_t[b] + t]   (saturating e4m3)
//   v:  written to v_cache[b, h, pos_t[b] + t]
// qp: q_proj output [B, T, Hq, 2D] (q in the first D of each head). kp, vp: [B, T, Hkv, D].
// RoPE position (multimodal RoPE: the cache row and the rotary position differ after an image):
//   rope_pos (optional, [B * T, 3]): the (t, h, w) positions of each row. Frequency i rotates by h when i % 3 == 1 and
//     i < 3 * sec_h, by w when i % 3 == 2 and i < 3 * sec_w, else by t (the interleaved layout of Qwen3-VL / 3.5).
//   else rope_delta (optional, [B]): every frequency rotates by pos_t[b] + t + rope_delta[b].
//   else pos_t[b] + t.
__global__ void k_prologue(const __nv_bfloat16* __restrict__ qp, const __nv_bfloat16* __restrict__ kp, const __nv_bfloat16* __restrict__ vp, int ldq, int ldk, int ldv,
                           const __nv_bfloat16* __restrict__ qn_w, const __nv_bfloat16* __restrict__ kn_w, const float* __restrict__ inv_freq,
                           const int* __restrict__ pos_t, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                           __nv_bfloat16* __restrict__ q_out, int Hq, int Hkv, int T, int Lmax, int R, float eps, const int* __restrict__ active,
                           const int* __restrict__ rope_delta, const int* __restrict__ rope_pos, int sec_h, int sec_w) {
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
        const int i = d % (R / 2);
        const int rp = rope_position(i, rope_delta != nullptr ? pos + rope_delta[b] : pos,
                                     rope_pos != nullptr ? rope_pos + ((size_t)b * T + t) * 3 : nullptr, sec_h, sec_w);
        y = rope_rotate(xs, d, R, xn, rp, inv_freq[i]);
    }
    if (hh < Hq) q_out[(((size_t)b * Hq + hh) * T + t) * D + d] = __float2bfloat16(y);
    else if (upd) KvFp8::store_row(kc + (((size_t)b * Hkv + (hh - Hq)) * Lmax + pos) * D, d, __bfloat162float(__float2bfloat16(y)));
}

}  // namespace attn

// The stripe count of a head layout (G query heads per KV head), 0 if the kernel is not compiled for it. The partial
// buffers of attn_decode hold B * Hq * T * attn_decode_tc_nb(G) rows.
int attn_decode_tc_nb(int G) { return G == 6 ? 12 : G == 12 ? 24 : 0; }

// part_acc: >= B * Hq * T * attn_decode_tc_nb(G) * 256 floats, part_ml: as many float2. gate: the q_proj output
// [B, T, Hq, 2D] (gate in the second half of each head); out [B, T, Hq * D] = attention * sigmoid(gate).
cudaError_t launch_attn_decode_tc(const void* q, const void* kc, const void* vc, const int* seq_lens, void* out, float* part_acc,
                                  void* part_ml, int B, int Hq, int Hkv, int T, int Lmax, int Dh, float scale, const void* gate, cudaStream_t st) {
    using namespace attn::tc;
    if (Dh != attn::D || Hkv < 1 || Hq % Hkv) return cudaErrorInvalidValue;
    const int G = Hq / Hkv, rows = G * T, passes = (rows + MAXROWS - 1) / MAXROWS, NB = attn_decode_tc_nb(G);
    if (NB == 0) return cudaErrorInvalidValue;
    const int MT = (min(rows, MAXROWS) + 15) / 16;
    const dim3 grid(B, Hkv * passes, NB);
    const float sl2 = scale * 1.4426950408889634f;
    auto qb = (const __nv_bfloat16*)q;
    cudaError_t err = cudaErrorInvalidValue;
#define TC_LAUNCH(GG, MM, NN) \
    err = launch_tc<GG, MM, NN, false>(grid, st, qb, kc, vc, seq_lens, nullptr, nullptr, 0, 0, part_acc, part_ml, Hkv, T, Lmax, sl2)
#define TC_MT(GG, NN)                         \
    switch (MT) {                             \
        case 1: TC_LAUNCH(GG, 1, NN); break;  \
        case 2: TC_LAUNCH(GG, 2, NN); break;  \
        case 3: TC_LAUNCH(GG, 3, NN); break;  \
    }
    if (G == 6) TC_MT(6, 12)        // Qwen3.8-27B: 24 query heads over 4 KV heads (the target and the MTP layer)
    else if (G == 12) TC_MT(12, 24)  // qwen4_exp: 24 query heads over 2 KV heads
#undef TC_MT
#undef TC_LAUNCH
    if (err != cudaSuccess) return err;
    attn::k_combine<<<B * Hq * T, attn::D, 0, st>>>(part_acc, (const float2*)part_ml, (__nv_bfloat16*)out, NB, (const __nv_bfloat16*)gate,
                                                    Hq, T);
    return cudaGetLastError();
}

cudaError_t launch_attn_prologue(const void* qp, const void* kp, const void* vp, int ldq, int ldk, int ldv, const void* qn_w, const void* kn_w,
                                 const float* inv_freq, const int* pos_t, void* kc, void* vc, void* q_out, int B, int T, int Hq, int Hkv, int Lmax,
                                 int R, float eps, const int* active, const int* rope_delta, const int* rope_pos, int sec_h, int sec_w,
                                 cudaStream_t st) {
    attn::k_prologue<<<dim3(B * T, Hq + 2 * Hkv), attn::D, 0, st>>>((const __nv_bfloat16*)qp, (const __nv_bfloat16*)kp, (const __nv_bfloat16*)vp,
                                                                    ldq, ldk, ldv, (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w, inv_freq,
                                                                    pos_t, (uint8_t*)kc, (uint8_t*)vc, (__nv_bfloat16*)q_out, Hq, Hkv, T, Lmax, R,
                                                                    eps, active, rope_delta, rope_pos, sec_h, sec_w);
    return cudaGetLastError();
}
