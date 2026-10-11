// Torch bindings for the CUDA kernels of the engine. engine/kernels/__init__.py builds them as `colinfer_kernels`, and
// Python calls them as engine.kernels.ops().<name>. Each wrapper checks the shapes / dtypes / strides and launches on the
// current stream, so a CUDA graph can capture each op. The sections follow the source files:
//
//   skinny.cu        decode linears: skinny_nvfp4 / skinny_swiglu / skinny_int / skinny_fp8, skinny_skip
//   gemv.cu, norm.cu bf16_gemv (tiny projections), rmsnorm
//   attn_decode.cu   attn_prologue (q/k norm + RoPE + KV write), attn_decode (multi-row tensor-core attention)
//   gdn_step.cu      gdn_conv, gdn_conv_commit, gdn_delta (Gated DeltaNet decode / verify / commit)
//   gemm_nvfp4.cu    prefill NVFP4 GEMMs and activation quantization
//   prefill_ops.cu   prefill glue: fp8_quant, add_rmsnorm, causal_conv_silu, gated_rmsnorm, gate_fp8
//   gdn_prefill.cu   gdn_prefill (chunked delta rule)
//   attn_prefill.cu  attn_prefill_fp8
//   sampling.cu      philox_uniform (sampling), rescore_nvfp4 (low-rank draft head)
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CHECK_CUDA_TENSOR(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dt), #t " must be a contiguous CUDA " #dt " tensor")
#define CHECK_LAUNCH(e) TORCH_CHECK((e) == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e))

static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

// optional per-slot int32 [B] tensor (activity mask, token counts) -> device pointer
static const int* opt_i32(const c10::optional<torch::Tensor>& a, int64_t B, const char* name) {
    if (!a) return nullptr;
    TORCH_CHECK(a->is_cuda() && a->is_contiguous() && a->scalar_type() == torch::kInt32 && a->numel() == B, name, ": int32 [B]");
    return a->data_ptr<int>();
}

// Row stride (elements) of a [..., n] bf16 view whose last dim is contiguous and whose rows are evenly spaced (a column
// slice of a wider matrix, or a contiguous tensor). Size-1 dims carry arbitrary strides and are skipped.
static int64_t row_stride(const torch::Tensor& t, int64_t n, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.size(-1) == n && t.stride(-1) == 1, name, ": bf16 CUDA [..., ", n,
                "] with a contiguous last dim");
    int64_t ld = n, span = 0;  // span: the stride the next outer dim must have for the rows to stay evenly spaced
    for (int64_t i = t.dim() - 2; i >= 0; --i) {
        if (t.size(i) == 1) continue;
        if (span == 0) ld = t.stride(i);
        else TORCH_CHECK(t.stride(i) == span, name, ": rows must be evenly strided");
        span = t.stride(i) * t.size(i);
    }
    TORCH_CHECK(ld >= n, name, ": overlapping rows");
    return ld;
}

// ===================================================================================================== skinny.cu
cudaError_t launch_skinny_nvfp4(const void*, const void*, const void*, float, const void*, void*, bool, int, int, int, float*, cudaStream_t);
cudaError_t launch_skinny_swiglu(const void*, const void*, const void*, float, const void*, const void*, float, void*, int, int, int, float*,
                                 cudaStream_t);
cudaError_t launch_skinny_fp8(const void*, const void*, float, const float*, const void*, void*, int, int, int, float*, cudaStream_t);
cudaError_t launch_skinny_int(int, const void*, const void*, const void*, const void*, float, const void*, void*, int, int, int, float*,
                              cudaStream_t);
int skinny_ws_floats(int, bool, int, int, int);
void skinny_set_skip(const int*);

// split-K workspace for one skinny call (fp32, from the caching allocator: graph-capture safe)
static float* skinny_ws(int fmt, bool swiglu, const torch::Tensor& x, int64_t N, int64_t K, torch::Tensor& hold) {
    const int n = skinny_ws_floats(fmt, swiglu, x.size(0), N, K);
    if (!n) return nullptr;
    hold = torch::empty({n}, x.options().dtype(torch::kFloat32));
    return hold.data_ptr<float>();
}

static void check_skinny(const torch::Tensor& x, const torch::Tensor& out, int64_t N, int64_t K) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16 && x.size(1) == K && K % 256 == 0 && N % 8 == 0,
                "skinny GEMM: x [M <= 16, K], K % 256 == 0 (512 for block-scaled weights), N % 8 == 0");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == x.size(0) && out.size(1) == N, "out must be [M, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat32, "out must be bf16 or fp32");
}

static const void* residual_ptr(const c10::optional<torch::Tensor>& r, const torch::Tensor& out) {
    if (!r) return nullptr;
    CHECK_CUDA_TENSOR(*r, torch::kBFloat16);
    TORCH_CHECK(r->sizes() == out.sizes(), "residual must match out");
    return r->data_ptr();
}

static void check_sf(const torch::Tensor& sf, int64_t N, int64_t K) {
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1 && sf.size(0) == N && sf.size(1) == K / 16,
                "block scales: 1-byte contiguous CUDA [N, K/16]");
}

// out [M, N] (bf16 or fp32) = x [M, K] @ dequant(W)^T * gscale (+ residual); NVFP4 W: uint8 [N, K/2], e4m3 scales [N, K/16].
void skinny_nvfp4(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double gscale, c10::optional<torch::Tensor> residual, torch::Tensor out) {
    CHECK_CUDA_TENSOR(w, torch::kUInt8);
    const int64_t N = w.size(0), K = w.size(1) * 2;
    check_skinny(x, out, N, K);
    check_sf(sf, N, K);
    torch::Tensor hold;
    float* ws = skinny_ws(0, false, x, N, K, hold);
    CHECK_LAUNCH(launch_skinny_nvfp4(x.data_ptr(), w.data_ptr(), sf.data_ptr(), (float)gscale, residual_ptr(residual, out), out.data_ptr(),
                                     out.scalar_type() == torch::kFloat32, x.size(0), N, K, ws, stream()));
}

// out bf16 [M, N] = silu(x Wg^T * gg) * (x Wu^T * gu), both NVFP4 (the MLP's gate and up in one weight pass).
void skinny_swiglu(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, double gg, torch::Tensor wu, torch::Tensor su, double gu, torch::Tensor out) {
    CHECK_CUDA_TENSOR(wg, torch::kUInt8);
    CHECK_CUDA_TENSOR(wu, torch::kUInt8);
    const int64_t N = wg.size(0), K = wg.size(1) * 2;
    TORCH_CHECK(wu.sizes() == wg.sizes(), "gate / up shapes");
    check_skinny(x, out, N, K);
    check_sf(sg, N, K);
    check_sf(su, N, K);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
    torch::Tensor hold;
    float* ws = skinny_ws(0, true, x, N, K, hold);
    CHECK_LAUNCH(launch_skinny_swiglu(x.data_ptr(), wg.data_ptr(), sg.data_ptr(), (float)gg, wu.data_ptr(), su.data_ptr(), (float)gu,
                                      out.data_ptr(), x.size(0), N, K, ws, stream()));
}

// INT6 / INT5 weights (engine/weights/quantize.py): wlo uint8 [N, K/2] low nibbles, whi uint8 [N, K/4] (INT6: 2-bit fields)
// or [N, K/8] (INT5: one bit per code), sf e4m3 [N, K/16]. The width of whi gives the format. out: bf16.
void skinny_int(torch::Tensor x, torch::Tensor wlo, torch::Tensor whi, torch::Tensor sf, double gscale, c10::optional<torch::Tensor> residual,
                torch::Tensor out) {
    CHECK_CUDA_TENSOR(wlo, torch::kUInt8);
    CHECK_CUDA_TENSOR(whi, torch::kUInt8);
    const int64_t N = wlo.size(0), K = wlo.size(1) * 2;
    check_skinny(x, out, N, K);
    check_sf(sf, N, K);
    const int bits = whi.size(1) == K / 4 ? 6 : whi.size(1) == K / 8 ? 5 : 0;
    TORCH_CHECK(bits && whi.size(0) == N, "high-bit plane: [N, K/4] (INT6) or [N, K/8] (INT5)");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
    torch::Tensor hold;
    float* ws = skinny_ws(2, false, x, N, K, hold);
    CHECK_LAUNCH(launch_skinny_int(bits, x.data_ptr(), wlo.data_ptr(), whi.data_ptr(), sf.data_ptr(), (float)gscale, residual_ptr(residual, out),
                                   out.data_ptr(), x.size(0), N, K, ws, stream()));
}

// FP8 W: e4m3 [N, K] with a per-tensor scale, or per-row scales row_scale fp32 [N] (stacked projections). out: bf16.
void skinny_fp8(torch::Tensor x, torch::Tensor w, double scale, c10::optional<torch::Tensor> residual, torch::Tensor out,
                c10::optional<torch::Tensor> row_scale) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.element_size() == 1 && w.dim() == 2, "w must be 1-byte [N, K]");
    const int64_t N = w.size(0), K = w.size(1);
    check_skinny(x, out, N, K);
    const float* rs = nullptr;
    if (row_scale) {
        CHECK_CUDA_TENSOR(*row_scale, torch::kFloat32);
        TORCH_CHECK(row_scale->numel() == N, "row_scale: fp32 [N]");
        rs = row_scale->data_ptr<float>();
    }
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
    torch::Tensor hold;
    float* ws = skinny_ws(1, false, x, N, K, hold);
    CHECK_LAUNCH(launch_skinny_fp8(x.data_ptr(), w.data_ptr(), (float)scale, rs, residual_ptr(residual, out), out.data_ptr(), x.size(0), N, K, ws,
                                   stream()));
}

// Skinny GEMMs launched (or captured) while a flag is set return at once, writing zeros, whenever the int32 it points to
// is nonzero (the draft early exit, engine/spec/mtp.py). None: back to normal launches.
void skinny_skip(c10::optional<torch::Tensor> flag) {
    if (flag) {
        CHECK_CUDA_TENSOR(*flag, torch::kInt32);
        TORCH_CHECK(flag->numel() == 1, "flag: one int32");
    }
    skinny_set_skip(flag ? flag->data_ptr<int>() : nullptr);
}

// ===================================================================================================== gemv.cu, norm.cu
cudaError_t launch_bf16_gemv(const void*, const void*, void*, int, int, int, cudaStream_t);
cudaError_t launch_rmsnorm(const void*, const void*, void*, int, int, float, cudaStream_t);

// out bf16 [M, N] = x [M, K] @ W^T, bf16 W [N, K], M <= 8.
void bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8 && x.size(1) == w.size(1) && out.size(0) == x.size(0) && out.size(1) == w.size(0),
                "x [M <= 8, K], out [M, N]");
    CHECK_LAUNCH(launch_bf16_gemv(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0), w.size(0), w.size(1), stream()));
}

// out = bf16(x / rms(x) * (1 + w)); x, out: bf16 [M, K]; w: bf16 [K] (the zero-centered RMSNorm).
void rmsnorm(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor out) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && w.numel() == x.size(1) && out.sizes() == x.sizes(), "bad shapes");
    CHECK_LAUNCH(launch_rmsnorm(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0), x.size(1), (float)eps, stream()));
}

// ===================================================================================================== attn_decode.cu
int attn_decode_tc_nb();
cudaError_t launch_attn_decode_tc(const void*, const void*, const void*, const int*, void*, float*, void*, int, int, int, int, int, int, float,
                                  const void*, cudaStream_t);
cudaError_t launch_attn_prologue(const void*, const void*, const void*, int, int, int, const void*, const void*, const float*, const int*, void*,
                                 void*, void*, int, int, int, int, int, int, float, const int*, cudaStream_t);

static void check_kv(const torch::Tensor& k, const torch::Tensor& v) {
    CHECK_CUDA_TENSOR(k, torch::kFloat8_e4m3fn);
    CHECK_CUDA_TENSOR(v, torch::kFloat8_e4m3fn);
    TORCH_CHECK(k.dim() == 4 && k.sizes() == v.sizes() && k.size(3) == 256, "KV caches: e4m3 [B, Hkv, Lmax, 256]");
}

// Fused q/k RMSNorm + partial RoPE + KV-cache write at the device positions pos_t[b] + t.
// qp, kp, vp: bf16 [.., features] views with unit-stride rows (e.g. column slices of one stacked q|k|v GEMM output).
// The qp rows hold [Hq, 512] (q | gate per head), and the kp / vp rows [Hkv, 256]. inv_freq: fp32 [R/2]. q_out:
// [B, Hq, T, 256]. active (optional): int32 [B]. Slots with 0 do not write KV.
void attn_prologue(torch::Tensor qp, torch::Tensor kp, torch::Tensor vp, torch::Tensor qn_w, torch::Tensor kn_w, torch::Tensor inv_freq,
                   torch::Tensor pos_t, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor q_out, double eps,
                   c10::optional<torch::Tensor> active) {
    for (auto* t : {&qp, &kp, &vp})
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->stride(-1) == 1, "q/k/v rows must be unit-stride bf16");
    for (auto* t : {&qn_w, &kn_w, &q_out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(inv_freq, torch::kFloat32);
    CHECK_CUDA_TENSOR(pos_t, torch::kInt32);
    check_kv(k_cache, v_cache);
    const int64_t B = q_out.size(0), Hq = q_out.size(1), T = q_out.size(2), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    auto q2 = qp.reshape({-1, qp.size(-1)}), k2 = kp.reshape({-1, kp.size(-1)}), v2 = vp.reshape({-1, vp.size(-1)});
    TORCH_CHECK(q2.size(0) == B * T && k2.size(0) == B * T && v2.size(0) == B * T && q2.size(1) == Hq * 512 && k2.size(1) == Hkv * 256 &&
                v2.size(1) == Hkv * 256 && q_out.size(3) == 256 && pos_t.numel() == B, "bad shapes / strides");
    CHECK_LAUNCH(launch_attn_prologue(q2.data_ptr(), k2.data_ptr(), v2.data_ptr(), (int)q2.stride(0), (int)k2.stride(0), (int)v2.stride(0),
                                      qn_w.data_ptr(), kn_w.data_ptr(), inv_freq.data_ptr<float>(), pos_t.data_ptr<int>(), k_cache.data_ptr(),
                                      v_cache.data_ptr(), q_out.data_ptr(), B, T, Hq, Hkv, Lmax, 2 * inv_freq.numel(), (float)eps,
                                      opt_i32(active, B, "active"), stream()));
}

// out[b, h, t] = softmax(q k^T * scale) v over the first seq_lens[b] - (T-1-t) cached positions, with one KV pass for all
// T rows of a slot. q: bf16 [B, Hq, T, 256]. caches: e4m3 [B, Hkv, Lmax, 256]. seq_lens: int32 [B] (device).
// gate (optional): the q_proj output [B, T, Hq, 2*256]. With it, out is [B, T, Hq*256] = attn * sigmoid(gate).
void attn_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor seq_lens, torch::Tensor out, double scale,
                 c10::optional<torch::Tensor> gate) {
    CHECK_CUDA_TENSOR(q, torch::kBFloat16);
    check_kv(k_cache, v_cache);
    CHECK_CUDA_TENSOR(seq_lens, torch::kInt32);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && out.numel() == q.numel(), "bad shapes");
    const void* gp = nullptr;
    if (gate) {
        CHECK_CUDA_TENSOR(*gate, torch::kBFloat16);
        TORCH_CHECK(gate->numel() == 2 * q.numel(), "gate: [B, T, Hq, 512]");
        gp = gate->data_ptr();
    }
    const int64_t B = q.size(0), Hq = q.size(1), T = q.size(2), Dh = q.size(3), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(k_cache.size(0) == B && Dh == 256 && seq_lens.numel() == B, "bad shapes");
    const int NB = attn_decode_tc_nb();
    auto part_acc = torch::empty({B * Hq * T * NB, Dh}, q.options().dtype(torch::kFloat32));
    auto part_ml = torch::empty({B * Hq * T * NB, 2}, q.options().dtype(torch::kFloat32));
    CHECK_LAUNCH(launch_attn_decode_tc(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), seq_lens.data_ptr<int>(), out.data_ptr(),
                                       part_acc.data_ptr<float>(), part_ml.data_ptr(), B, Hq, Hkv, T, Lmax, Dh, (float)scale, gp, stream()));
}

// ===================================================================================================== gdn_step.cu
cudaError_t launch_gdn_conv(const void*, int, const void*, const void*, void*, int, int, int, cudaStream_t);
cudaError_t launch_gdn_conv_commit(const void*, int, void*, const int*, int, int, int, cudaStream_t);
cudaError_t launch_gdn_delta(const void*, const void*, int, const void*, const void*, int, const void*, const void*, const void*, float*, void*,
                             const int*, int, int, int, float, int, cudaStream_t);

// out bf16 [B, T, C] = silu(causal depthwise conv) of T new tokens per slot from the conv window (not written).
// mixed: bf16 [B, T, C] (rows may be strided); conv_state: bf16 [B, C, 3]; w: bf16 [C, 4] (or [C, 1, 4]).
void gdn_conv(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor w, torch::Tensor out) {
    for (auto* t : {&conv_state, &w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    const int64_t B = conv_state.size(0), C = conv_state.size(1), T = mixed.numel() / (B * C);
    const int64_t ldm = row_stride(mixed, C, "mixed");
    TORCH_CHECK(mixed.numel() == B * T * C && out.numel() == mixed.numel() && w.numel() == C * 4, "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv(mixed.data_ptr(), (int)ldm, conv_state.data_ptr(), w.data_ptr(), out.data_ptr(), B, T, C, stream()));
}

// conv window advanced by n[b] (int32 [B], 0 .. T) of the T tokens in mixed.
void gdn_conv_commit(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor n) {
    CHECK_CUDA_TENSOR(conv_state, torch::kBFloat16);
    const int64_t B = conv_state.size(0), C = conv_state.size(1), T = mixed.numel() / (B * C);
    const int64_t ldm = row_stride(mixed, C, "mixed");
    TORCH_CHECK(mixed.numel() == B * T * C, "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv_commit(mixed.data_ptr(), (int)ldm, conv_state.data_ptr(), opt_i32(n, B, "n"), B, T, C, stream()));
}

// Gated delta rule + gated RMSNorm over T tokens per slot. The caller must give at least one of out and n:
//   out (optional): bf16 [B, T, Hv*128], the outputs of all T tokens
//   n (optional): int32 [B]. The op writes back the state advanced by n[b] tokens (0: no change).
// qkv bf16 [B, T, 2*Hk*128 + Hv*128] (the conv output). z [B, T, Hv*128]. b, a [B, T, Hv]. z, b and a can be column
// slices of wider rows, and b and a share a row stride. A_log, dt_bias bf16 [Hv]. norm_w bf16 [128]. state fp32
// [B, Hv, 128, 128].
void gdn_delta(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a, torch::Tensor A_log, torch::Tensor dt_bias,
               torch::Tensor norm_w, torch::Tensor state, c10::optional<torch::Tensor> out, int64_t Hk, double eps, c10::optional<torch::Tensor> n) {
    for (auto* t : {&qkv, &A_log, &dt_bias, &norm_w}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(state, torch::kFloat32);
    const int64_t B = state.size(0), Hv = state.size(1), C = 2 * Hk * 128 + Hv * 128, T = qkv.numel() / (B * C);
    const int64_t ldz = row_stride(z, Hv * 128, "z"), ldba = row_stride(b, Hv, "b");
    TORCH_CHECK(row_stride(a, Hv, "a") == ldba, "b and a must share a row stride");
    TORCH_CHECK(state.dim() == 4 && state.size(2) == 128 && state.size(3) == 128 && qkv.numel() == B * T * C && z.numel() == B * T * Hv * 128 &&
                b.numel() == B * T * Hv && a.numel() == b.numel() && Hv % Hk == 0, "bad shapes");
    TORCH_CHECK(out || n, "gdn_delta: outputs, a commit, or both");
    void* op = nullptr;
    if (out) {
        CHECK_CUDA_TENSOR(*out, torch::kBFloat16);
        TORCH_CHECK(out->numel() == z.numel(), "out: [B, T, Hv*128]");
        op = out->data_ptr();
    }
    CHECK_LAUNCH(launch_gdn_delta(qkv.data_ptr(), z.data_ptr(), (int)ldz, b.data_ptr(), a.data_ptr(), (int)ldba, A_log.data_ptr(), dt_bias.data_ptr(),
                                  norm_w.data_ptr(), state.data_ptr<float>(), op, opt_i32(n, B, "n"), B, Hk, Hv, (float)eps, T, stream()));
}

// ===================================================================================================== gemm_nvfp4.cu
size_t nvfp4_sf_bytes(int, int);
cudaError_t launch_nvfp4_quant(const void*, void*, void*, int, int, float, cudaStream_t);
cudaError_t launch_nvfp4_swizzle_sf(const void*, void*, int, int, cudaStream_t);
cudaError_t launch_nvfp4_gemm(const void*, const void*, const void*, const void*, float, const void*, void*, int, int, int, int, void*, size_t,
                              size_t*, cudaStream_t);
cudaError_t launch_nvfp4_gemm_swiglu(const void*, const void*, const void*, const void*, float, const void*, void*, void*, const float*, int, int,
                                     int, int, void*, size_t, size_t*, cudaStream_t);

int64_t nvfp4_sf_size(int64_t rows, int64_t K) { return (int64_t)nvfp4_sf_bytes(rows, K); }

// x bf16 [M, K] -> q uint8 [M, K/2] (packed e2m1), sf uint8 [nvfp4_sf_size(M, K)] (swizzled e4m3), static global scale.
void nvfp4_quant(torch::Tensor x, double in_scale, torch::Tensor q, torch::Tensor sf) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(q, torch::kUInt8);
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1);
    const int64_t M = x.size(0), K = x.size(1);
    TORCH_CHECK(x.dim() == 2 && q.size(0) == M && q.size(1) == K / 2 && sf.numel() >= (int64_t)nvfp4_sf_bytes(M, K), "bad shapes");
    CHECK_LAUNCH(launch_nvfp4_quant(x.data_ptr(), q.data_ptr(), sf.data_ptr(), M, K, (float)in_scale, stream()));
}

// row-major e4m3 scales [R, K/16] -> swizzled [nvfp4_sf_size(R, K)] (CUTLASS's 128 x 4 scale-factor layout)
void nvfp4_swizzle_sf(torch::Tensor src, torch::Tensor dst, int64_t K) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.element_size() == 1 && dst.is_cuda() && dst.element_size() == 1);
    const int64_t R = src.size(0);
    TORCH_CHECK(src.size(1) == K / 16 && dst.numel() >= (int64_t)nvfp4_sf_bytes(R, K), "bad shapes");
    CHECK_LAUNCH(launch_nvfp4_swizzle_sf(src.data_ptr(), dst.data_ptr(), R, K, stream()));
}

// out bf16 [M, N] = alpha * (A . B^T) (+ residual). a [M, K/2], b [N, K/2] packed e2m1; sfa, sfb swizzled. tile 0: 128 x 256
// (M >= 1536), 1: 128 x 128.
void nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, double alpha, c10::optional<torch::Tensor> residual,
                torch::Tensor out, int64_t tile) {
    CHECK_CUDA_TENSOR(a, torch::kUInt8);
    CHECK_CUDA_TENSOR(b, torch::kUInt8);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    const int64_t M = a.size(0), K = a.size(1) * 2, N = b.size(0);
    TORCH_CHECK(b.size(1) * 2 == K && out.size(0) == M && out.size(1) == N && K % 128 == 0 && N % 8 == 0, "bad shapes");
    TORCH_CHECK(sfa.numel() >= (int64_t)nvfp4_sf_bytes(M, K) && sfb.numel() >= (int64_t)nvfp4_sf_bytes(N, K), "scale tensors too small");
    const void* c = residual_ptr(residual, out);
    size_t need = 0;
    launch_nvfp4_gemm(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, c, out.data_ptr(), M, N, K, (int)tile, nullptr, 0,
                      &need, stream());
    auto ws = torch::empty({(int64_t)std::max<size_t>(need, 1)}, a.options());
    CHECK_LAUNCH(launch_nvfp4_gemm(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, c, out.data_ptr(), M, N, K, (int)tile,
                                   ws.data_ptr(), need, nullptr, stream()));
}

// The MLP's up GEMM with the SwiGLU fused: hq / hsf = NVFP4(silu(gate) * alpha * (a . b^T)), quantized for the down GEMM
// (norm_const: float32 [1] on the device = 1 / the down projection's input scale). gate: bf16 [M, N] (the gate GEMM).
void nvfp4_gemm_swiglu(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, double alpha, torch::Tensor gate, torch::Tensor hq,
                       torch::Tensor hsf, torch::Tensor norm_const, int64_t tile) {
    CHECK_CUDA_TENSOR(a, torch::kUInt8);
    CHECK_CUDA_TENSOR(b, torch::kUInt8);
    CHECK_CUDA_TENSOR(gate, torch::kBFloat16);
    CHECK_CUDA_TENSOR(hq, torch::kUInt8);
    CHECK_CUDA_TENSOR(norm_const, torch::kFloat32);
    const int64_t M = a.size(0), K = a.size(1) * 2, N = b.size(0);
    TORCH_CHECK(b.size(1) * 2 == K && gate.size(0) == M && gate.size(1) == N && hq.size(0) == M && hq.size(1) * 2 == N && K % 128 == 0 &&
                N % 128 == 0, "bad shapes");
    TORCH_CHECK(sfa.numel() >= (int64_t)nvfp4_sf_bytes(M, K) && sfb.numel() >= (int64_t)nvfp4_sf_bytes(N, K) &&
                hsf.numel() >= (int64_t)nvfp4_sf_bytes(M, N), "scale tensors too small");
    size_t need = 0;
    launch_nvfp4_gemm_swiglu(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, gate.data_ptr(), hq.data_ptr(),
                             hsf.data_ptr(), norm_const.data_ptr<float>(), M, N, K, (int)tile, nullptr, 0, &need, stream());
    auto ws = torch::empty({(int64_t)std::max<size_t>(need, 1)}, a.options());
    CHECK_LAUNCH(launch_nvfp4_gemm_swiglu(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, gate.data_ptr(), hq.data_ptr(),
                                          hsf.data_ptr(), norm_const.data_ptr<float>(), M, N, K, (int)tile, ws.data_ptr(), need, nullptr, stream()));
}

// ===================================================================================================== prefill_ops.cu
cudaError_t launch_fp8_quant(const void*, void*, size_t, float, cudaStream_t);
cudaError_t launch_causal_conv_silu(const void*, int, const void*, const void*, void*, void*, void*, int, int, int, int, float, cudaStream_t);
cudaError_t launch_add_rmsnorm(const void*, const void*, const void*, float, int, int, void*, void*, void*, void*, float, void*, float, cudaStream_t);
cudaError_t launch_gate_fp8(const void*, const void*, int, int, int, int, void*, float, cudaStream_t);
cudaError_t launch_gated_rmsnorm(const void*, const void*, int, int, const void*, void*, int, float, cudaStream_t);

// out (float8_e4m3fn, same shape) = saturate(x / scale)
void fp8_quant(torch::Tensor x, double scale, torch::Tensor out) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kFloat8_e4m3fn);
    TORCH_CHECK(out.numel() == x.numel());
    CHECK_LAUNCH(launch_fp8_quant(x.data_ptr(), out.data_ptr(), x.numel(), (float)scale, stream()));
}

// bf16(silu(bf16(depthwise causal conv(x)))) for x [T, C] bf16 (rows with any stride), state [C, 3] (the previous 3
// inputs) and w [C, 4]. The output channels split into three contiguous tensors outs[0..2] = [T, c1], [T, c2 - c1],
// [T, C - c2] (q, k, v). With l2_eps >= 0, the op also L2-normalizes the first two outputs (q, k) per 128-channel head.
void causal_conv_silu(torch::Tensor x, torch::Tensor state, torch::Tensor w, std::vector<torch::Tensor> outs, double l2_eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x: bf16 [T, C], unit-stride rows");
    TORCH_CHECK(outs.size() == 3, "three outputs");
    for (auto* t : {&state, &w, &outs[0], &outs[1], &outs[2]}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    const int64_t T = x.size(0), C = x.size(1), c1 = outs[0].size(-1), c2 = c1 + outs[1].size(-1);
    TORCH_CHECK(state.numel() == C * 3 && w.numel() == C * 4 && c2 + outs[2].size(-1) == C, "bad shapes");
    for (auto& o : outs) TORCH_CHECK(o.numel() == T * o.size(-1), "outputs must be [T, channels]");
    CHECK_LAUNCH(launch_causal_conv_silu(x.data_ptr(), (int)x.stride(0), state.data_ptr(), w.data_ptr(), outs[0].data_ptr(), outs[1].data_ptr(),
                                         outs[2].data_ptr(), (int)c1, (int)c2, T, C, (float)l2_eps, stream()));
}

// Fused residual add + zero-centered RMSNorm + quantization, one row of K per block. All outputs optional:
// x_out = bf16(x + y); n_out = bf16 normed; (q4, sf4) = NVFP4 of normed (in_scale4); q8 = e4m3(normed / in_scale8).
void add_rmsnorm(torch::Tensor x, c10::optional<torch::Tensor> y, torch::Tensor w, double eps, c10::optional<torch::Tensor> x_out,
                 c10::optional<torch::Tensor> n_out, c10::optional<torch::Tensor> q4, c10::optional<torch::Tensor> sf4, double in_scale4,
                 c10::optional<torch::Tensor> q8, double in_scale8) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    const int64_t M = x.size(0), K = x.size(1);
    auto ptr = [&](const c10::optional<torch::Tensor>& t, int64_t numel) -> void* {
        if (!t) return nullptr;
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->numel() >= numel, "bad optional tensor");
        return t->data_ptr();
    };
    TORCH_CHECK(x.dim() == 2 && w.numel() == K && (!y || y->sizes() == x.sizes()), "bad shapes");
    TORCH_CHECK(!q4 == !sf4, "q4 and sf4 go together");
    CHECK_LAUNCH(launch_add_rmsnorm(x.data_ptr(), ptr(y, M * K), w.data_ptr(), (float)eps, M, K, ptr(x_out, M * K), ptr(n_out, M * K),
                                    ptr(q4, M * K / 2), ptr(sf4, (int64_t)nvfp4_sf_bytes(M, K)), (float)in_scale4, ptr(q8, M * K), (float)in_scale8,
                                    stream()));
}

// out e4m3 [T, H*D] = (o * sigmoid(gate)) / scale; o bf16 [T, H*D]; gate: rows [H, 2D] (q | gate), strided
void gate_fp8(torch::Tensor o, torch::Tensor gate, int64_t D, double scale, torch::Tensor out) {
    CHECK_CUDA_TENSOR(o, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kFloat8_e4m3fn);
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 && gate.dim() == 2 && gate.stride(1) == 1, "gate rows unit-stride");
    const int64_t T = o.size(0), H = o.size(1) / D;
    TORCH_CHECK(gate.size(0) == T && gate.size(1) == 2 * H * D && out.numel() == o.numel(), "bad shapes");
    CHECK_LAUNCH(launch_gate_fp8(o.data_ptr(), gate.data_ptr(), (int)gate.stride(0), T, H, D, out.data_ptr(), (float)scale, stream()));
}

// o, out: bf16 [T*heads, 128] contiguous; z: bf16 [T, heads*128] (rows may be strided); w [128]
void gated_rmsnorm(torch::Tensor o, torch::Tensor z, torch::Tensor w, double eps, torch::Tensor out) {
    for (auto* t : {&o, &w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.dim() == 2 && z.stride(1) == 1, "z: bf16 [T, heads*128], unit-stride rows");
    const int64_t heads = z.size(1) / 128;
    TORCH_CHECK(z.size(1) % 128 == 0 && o.numel() == z.size(0) * z.size(1) && out.numel() == o.numel() && w.numel() == 128, "bad shapes");
    CHECK_LAUNCH(launch_gated_rmsnorm(o.data_ptr(), z.data_ptr(), (int)z.stride(0), (int)heads, w.data_ptr(), out.data_ptr(), o.numel() / 128,
                                      (float)eps, stream()));
}

// ===================================================================================================== gdn_prefill.cu
size_t gdn_prefill_ws_bytes(int, int);
cudaError_t launch_gdn_prefill(const void*, const void*, const void*, const float*, const void*, float*, void*, void*, int, int, int, float, cudaStream_t);

// Chunked Gated DeltaNet forward: q, k bf16 [T, Hk, 128] (L2-normalized), v bf16 [T, Hv, 128], g fp32 [T, Hv] (log decay),
// beta bf16 [T, Hv], state fp32 [Hv, 128, 128] (continued in place), o bf16 [T, Hv, 128].
void gdn_prefill(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor state, torch::Tensor o,
                 double scale) {
    for (auto* t : {&q, &k, &v, &beta, &o}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(g, torch::kFloat32);
    CHECK_CUDA_TENSOR(state, torch::kFloat32);
    const int64_t T = q.size(0), Hk = q.size(1), Hv = v.size(1);
    TORCH_CHECK(q.dim() == 3 && q.sizes() == k.sizes() && q.size(2) == 128 && v.dim() == 3 && v.size(0) == T && v.size(2) == 128 &&
                g.numel() == T * Hv && beta.numel() == T * Hv && state.numel() == Hv * 128 * 128 && o.sizes() == v.sizes() && Hv % Hk == 0,
                "bad shapes");
    auto ws = torch::empty({(int64_t)gdn_prefill_ws_bytes(T, Hv)}, q.options().dtype(torch::kUInt8));
    CHECK_LAUNCH(launch_gdn_prefill(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr<float>(), beta.data_ptr(), state.data_ptr<float>(),
                                    o.data_ptr(), ws.data_ptr(), T, Hk, Hv, (float)scale, stream()));
}

// ===================================================================================================== attn_prefill.cu
cudaError_t launch_attn_prefill_fp8(const void*, const void*, const void*, void*, int, int, int, int, int, float, cudaStream_t);

// Causal prefill attention over the e4m3 KV cache of one slot, with Q K^T on FP8 tensor cores.
// q: bf16 [1, Hq, T, 256] (rows at positions pos .. pos + T - 1). caches: e4m3 [1, Hkv, Lmax, 256], with the T new rows
// written. out: bf16 [T, Hq * 256].
void attn_prefill_fp8(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor out, int64_t pos, double scale) {
    CHECK_CUDA_TENSOR(q, torch::kBFloat16);
    check_kv(k_cache, v_cache);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 256 && k_cache.size(0) == 1 && out.numel() == q.numel(), "bad shapes");
    const int64_t Hq = q.size(1), T = q.size(2), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(pos >= 0 && pos + T <= Lmax, "positions past the cache");
    CHECK_LAUNCH(launch_attn_prefill_fp8(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), out.data_ptr(), T, Hq, Hkv, Lmax, pos, (float)scale,
                                         stream()));
}

// ===================================================================================================== sampling.cu
cudaError_t launch_philox_uniform(const int64_t*, const int64_t*, float*, int, int, cudaStream_t);
cudaError_t launch_rescore_nvfp4(const void*, const void*, const void*, float, const int64_t*, float*, int, int, int, cudaStream_t);

// out fp32 [B, n] = per-slot seeded uniforms in (0, 1] at counters offset[b] + i (seed, offset: int64 [B], device).
void philox_uniform(torch::Tensor seed, torch::Tensor offset, torch::Tensor out) {
    CHECK_CUDA_TENSOR(seed, torch::kInt64);
    CHECK_CUDA_TENSOR(offset, torch::kInt64);
    CHECK_CUDA_TENSOR(out, torch::kFloat32);
    TORCH_CHECK(out.dim() == 2 && seed.numel() == out.size(0) && offset.numel() == out.size(0), "out [B, n], seed / offset [B]");
    CHECK_LAUNCH(launch_philox_uniform(seed.data_ptr<int64_t>(), offset.data_ptr<int64_t>(), out.data_ptr<float>(), out.size(0), out.size(1), stream()));
}

// exact logits of candidate rows of an NVFP4 matrix: x bf16 [B, K], w u8 [V, K/2], sf e4m3 [V, K/16], cand int64 [B, NC] -> [B, NC] fp32
torch::Tensor rescore_nvfp4(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double gs, torch::Tensor cand) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kUInt8);
    CHECK_CUDA_TENSOR(cand, torch::kInt64);
    const int64_t K = x.size(-1), B = x.numel() / K, NC = cand.size(-1);
    TORCH_CHECK(w.size(1) * 2 == K && cand.numel() == B * NC, "shapes");
    check_sf(sf, w.size(0), K);
    auto out = torch::empty({B, NC}, x.options().dtype(torch::kFloat32));
    CHECK_LAUNCH(launch_rescore_nvfp4(x.data_ptr(), w.data_ptr(), sf.data_ptr(), (float)gs, cand.data_ptr<int64_t>(), out.data_ptr<float>(), (int)B,
                                      (int)K, (int)NC, stream()));
    return out;
}

// ===================================================================================================== module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // decode linears
    m.def("skinny_nvfp4", &skinny_nvfp4, "tensor-core NVFP4 x bf16 skinny GEMM, M <= 16", py::arg("x"), py::arg("w"), py::arg("sf"),
          py::arg("gscale"), py::arg("residual"), py::arg("out"));
    m.def("skinny_swiglu", &skinny_swiglu, "tensor-core silu(x Wg^T) * (x Wu^T), NVFP4, M <= 16", py::arg("x"), py::arg("wg"), py::arg("sg"), py::arg("gg"), py::arg("wu"), py::arg("su"), py::arg("gu"), py::arg("out"));
    m.def("skinny_int", &skinny_int, "tensor-core skinny GEMM, INT6 / INT5 block-16 weights, M <= 16", py::arg("x"), py::arg("wlo"), py::arg("whi"),
          py::arg("sf"), py::arg("gscale"), py::arg("residual"), py::arg("out"));
    m.def("skinny_fp8", &skinny_fp8, "tensor-core FP8 x bf16 skinny GEMM, M <= 16", py::arg("x"), py::arg("w"), py::arg("scale"),
          py::arg("residual"), py::arg("out"), py::arg("row_scale") = py::none());
    m.def("skinny_skip", &skinny_skip, "skinny GEMMs launched from now on return early while *flag != 0 (None: never)", py::arg("flag"));
    m.def("bf16_gemv", &bf16_gemv, "bf16-weight GEMV, M <= 8", py::arg("x"), py::arg("w"), py::arg("out"));
    m.def("rmsnorm", &rmsnorm, "zero-centered RMSNorm (1 + w)", py::arg("x"), py::arg("w"), py::arg("eps"), py::arg("out"));
    // decode attention
    m.def("attn_prologue", &attn_prologue, "fused q/k norm + partial RoPE + fp8 KV write", py::arg("qp"), py::arg("kp"), py::arg("vp"),
          py::arg("qn_w"), py::arg("kn_w"), py::arg("inv_freq"), py::arg("pos_t"), py::arg("k_cache"), py::arg("v_cache"), py::arg("q_out"),
          py::arg("eps"), py::arg("active") = py::none());
    m.def("attn_decode", &attn_decode, "tensor-core multi-row GQA decode attention over an fp8 KV cache", py::arg("q"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("seq_lens"), py::arg("out"), py::arg("scale"), py::arg("gate") = py::none());
    // Gated DeltaNet decode / verify / commit
    m.def("gdn_conv", &gdn_conv, "GDN causal conv + SiLU over T tokens per slot (window read-only)", py::arg("mixed"), py::arg("conv_state"), py::arg("w"), py::arg("out"));
    m.def("gdn_conv_commit", &gdn_conv_commit, "advance the GDN conv window by n tokens", py::arg("mixed"), py::arg("conv_state"), py::arg("n"));
    m.def("gdn_delta", &gdn_delta, "GDN gated delta rule + gated RMSNorm over T tokens per slot", py::arg("qkv"), py::arg("z"), py::arg("b"),
          py::arg("a"), py::arg("A_log"), py::arg("dt_bias"), py::arg("norm_w"), py::arg("state"), py::arg("out"), py::arg("Hk"), py::arg("eps"),
          py::arg("n") = py::none());
    // prefill
    m.def("nvfp4_sf_size", &nvfp4_sf_size, "bytes of a swizzled NVFP4 scale tensor for [rows, K]", py::arg("rows"), py::arg("K"));
    m.def("nvfp4_quant", &nvfp4_quant, "bf16 -> NVFP4 (packed e2m1 + swizzled e4m3 scales), static global scale", py::arg("x"), py::arg("in_scale"), py::arg("q"), py::arg("sf"));
    m.def("nvfp4_swizzle_sf", &nvfp4_swizzle_sf, "row-major NVFP4 scales -> CUTLASS 128x4 layout", py::arg("src"), py::arg("dst"), py::arg("K"));
    m.def("nvfp4_gemm", &nvfp4_gemm, "CUTLASS SM120 NVFP4 x NVFP4 GEMM, bf16 out", py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"),
          py::arg("alpha"), py::arg("residual"), py::arg("out"), py::arg("tile") = 0);
    m.def("nvfp4_gemm_swiglu", &nvfp4_gemm_swiglu, "NVFP4 up GEMM with silu(gate) * acc and NVFP4 quantization fused in the epilogue",
          py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"), py::arg("alpha"), py::arg("gate"), py::arg("hq"), py::arg("hsf"),
          py::arg("norm_const"), py::arg("tile") = 0);
    m.def("fp8_quant", &fp8_quant, "bf16 -> e4m3, static scale", py::arg("x"), py::arg("scale"), py::arg("out"));
    m.def("causal_conv_silu", &causal_conv_silu, "GDN causal depthwise conv (k=4) + SiLU, token-major, split q|k|v outputs", py::arg("x"),
          py::arg("state"), py::arg("w"), py::arg("outs"), py::arg("l2_eps") = -1.0);
    m.def("add_rmsnorm", &add_rmsnorm, "fused residual add + RMSNorm (+ NVFP4 / FP8 quantization)", py::arg("x"), py::arg("y"), py::arg("w"),
          py::arg("eps"), py::arg("x_out") = py::none(), py::arg("n_out") = py::none(), py::arg("q4") = py::none(), py::arg("sf4") = py::none(),
          py::arg("in_scale4") = 1.0, py::arg("q8") = py::none(), py::arg("in_scale8") = 1.0);
    m.def("gate_fp8", &gate_fp8, "attention output gate fused with FP8 quantization", py::arg("o"), py::arg("gate"), py::arg("D"), py::arg("scale"), py::arg("out"));
    m.def("gated_rmsnorm", &gated_rmsnorm, "GDN gated RMSNorm over rows of 128", py::arg("o"), py::arg("z"), py::arg("w"), py::arg("eps"), py::arg("out"));
    m.def("gdn_prefill", &gdn_prefill, "chunked Gated DeltaNet forward (prefill), continuing state in place", py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("state"), py::arg("o"), py::arg("scale"));
    m.def("attn_prefill_fp8", &attn_prefill_fp8, "causal prefill attention, FP8 Q K^T over an e4m3 KV cache", py::arg("q"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("out"), py::arg("pos"), py::arg("scale"));
    // sampling and drafting
    m.def("philox_uniform", &philox_uniform, "per-slot seeded uniforms (position-keyed sampling)", py::arg("seed"), py::arg("offset"), py::arg("out"));
    m.def("rescore_nvfp4", &rescore_nvfp4, "exact logits of candidate rows of an NVFP4 matrix (low-rank draft head)", py::arg("x"), py::arg("w"), py::arg("sf"), py::arg("gs"), py::arg("cand"));
}
