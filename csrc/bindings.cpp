// Torch bindings for the decode kernels (built by engine/kernels/__init__.py).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t launch_nvfp4_gemv(const void*, const void*, const void*, float, const void*, void*, bool, int, int, int, cudaStream_t);
cudaError_t launch_nvfp4_swiglu(const void*, const void*, const void*, float, const void*, const void*, float, void*, int, int, int, cudaStream_t);
cudaError_t launch_fp8_gemv(const void*, const void*, float, const float*, const void*, void*, bool, int, int, int, cudaStream_t);
cudaError_t launch_attn_decode(const void*, const void*, const void*, const int*, void*, float*, void*, int, int, int, int, int, int, int, float,
                               bool, const void*, cudaStream_t);
cudaError_t launch_attn_prologue(const void*, const void*, const void*, int, int, int, const void*, const void*, const float*, const int*, void*, void*, void*,
                                 int, int, int, int, int, int, float, bool, const int*, cudaStream_t);

#define CHECK_CUDA_TENSOR(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dt), #t " must be a contiguous CUDA " #dt " tensor")
#define CHECK_LAUNCH(e) TORCH_CHECK((e) == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e))

// optional per-slot activity mask (int32 [B]) for the decode kernels
static const int* active_ptr(const c10::optional<torch::Tensor>& a, int64_t B) {
    if (!a) return nullptr;
    TORCH_CHECK(a->is_cuda() && a->is_contiguous() && a->scalar_type() == torch::kInt32 && a->numel() == B, "active: int32 [B]");
    return a->data_ptr<int>();
}

static void check_x_out(const torch::Tensor& x, const torch::Tensor& out, int64_t N) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 4, "x must be [M<=4, K]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == x.size(0) && out.size(1) == N, "out must be [M, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat32, "out must be bf16 or fp32");
}

// out = x @ dequant(W)^T (+ residual). w: uint8 [N, K/2], sf: float8_e4m3fn or uint8 [N, K/16].
void nvfp4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor sf, double gscale, c10::optional<torch::Tensor> residual, torch::Tensor out) {
    CHECK_CUDA_TENSOR(w, torch::kUInt8);
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1, "sf must be 1-byte contiguous CUDA");
    const int64_t N = w.size(0), K = w.size(1) * 2;
    check_x_out(x, out, N);
    TORCH_CHECK(x.size(1) == K && sf.size(0) == N && sf.size(1) == K / 16, "shape mismatch");
    const void* r = nullptr;
    if (residual) { CHECK_CUDA_TENSOR(*residual, torch::kBFloat16); TORCH_CHECK(residual->sizes() == out.sizes()); r = residual->data_ptr(); }
    CHECK_LAUNCH(launch_nvfp4_gemv(x.data_ptr(), w.data_ptr(), sf.data_ptr(), (float)gscale, r, out.data_ptr(),
                                   out.scalar_type() == torch::kFloat32, x.size(0), N, K, at::cuda::getCurrentCUDAStream()));
}

// out = silu(x @ Wg^T) * (x @ Wu^T), both NVFP4.
void nvfp4_swiglu(torch::Tensor x, torch::Tensor wg, torch::Tensor sg, double gg, torch::Tensor wu, torch::Tensor su, double gu, torch::Tensor out) {
    CHECK_CUDA_TENSOR(wg, torch::kUInt8);
    CHECK_CUDA_TENSOR(wu, torch::kUInt8);
    const int64_t N = wg.size(0), K = wg.size(1) * 2;
    TORCH_CHECK(wu.sizes() == wg.sizes() && sg.sizes() == su.sizes() && sg.size(0) == N && sg.size(1) == K / 16, "shape mismatch");
    check_x_out(x, out, N);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && x.size(1) == K);
    CHECK_LAUNCH(launch_nvfp4_swiglu(x.data_ptr(), wg.data_ptr(), sg.data_ptr(), (float)gg, wu.data_ptr(), su.data_ptr(), (float)gu,
                                     out.data_ptr(), x.size(0), N, K, at::cuda::getCurrentCUDAStream()));
}

// out = (x @ W^T) * scale (+ residual). w: float8_e4m3fn or uint8 [N, K]. row_scale: optional fp32 [N]
// per-row scales (several projections with their own per-tensor scales stacked into one launch).
void fp8_gemv(torch::Tensor x, torch::Tensor w, double scale, c10::optional<torch::Tensor> residual, torch::Tensor out,
              c10::optional<torch::Tensor> row_scale) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.element_size() == 1 && w.dim() == 2, "w must be 1-byte [N, K]");
    const int64_t N = w.size(0), K = w.size(1);
    check_x_out(x, out, N);
    TORCH_CHECK(x.size(1) == K, "shape mismatch");
    const void* r = nullptr;
    if (residual) { CHECK_CUDA_TENSOR(*residual, torch::kBFloat16); TORCH_CHECK(residual->sizes() == out.sizes()); r = residual->data_ptr(); }
    const float* rs = nullptr;
    if (row_scale) { CHECK_CUDA_TENSOR(*row_scale, torch::kFloat32); TORCH_CHECK(row_scale->numel() == N); rs = row_scale->data_ptr<float>(); }
    CHECK_LAUNCH(launch_fp8_gemv(x.data_ptr(), w.data_ptr(), (float)scale, rs, r, out.data_ptr(), out.scalar_type() == torch::kFloat32,
                                 x.size(0), N, K, at::cuda::getCurrentCUDAStream()));
}

// out[b, h, t] = softmax(q k^T * scale) v over the first seq_lens[b] - (T-1-t) cached positions.
// q, out: bf16 [B, Hq, T, 256]; k_cache, v_cache: bf16 or float8_e4m3fn [B, Hkv, Lmax, 256]; seq_lens: int32 [B] (device).
// gate (optional): the q_proj output [B, T, Hq, 2*256]; then out is [B, T, Hq*256] = attn * sigmoid(gate).
void attn_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor seq_lens, torch::Tensor out, int64_t splits,
                 double scale, c10::optional<torch::Tensor> gate) {
    CHECK_CUDA_TENSOR(q, torch::kBFloat16);
    const bool kv_fp8 = k_cache.scalar_type() == torch::kFloat8_e4m3fn;
    CHECK_CUDA_TENSOR(k_cache, kv_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    CHECK_CUDA_TENSOR(v_cache, k_cache.scalar_type());
    CHECK_CUDA_TENSOR(seq_lens, torch::kInt32);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes() && out.numel() == q.numel(), "bad shapes");
    const void* gp = nullptr;
    if (gate) { CHECK_CUDA_TENSOR(*gate, torch::kBFloat16); TORCH_CHECK(gate->numel() == 2 * q.numel()); gp = gate->data_ptr(); }
    const int64_t B = q.size(0), Hq = q.size(1), T = q.size(2), Dh = q.size(3), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(3) == Dh && seq_lens.numel() == B && splits >= 1, "bad shapes");
    auto part_acc = torch::empty({B * Hq * T * splits, Dh}, q.options().dtype(torch::kFloat32));
    auto part_ml = torch::empty({B * Hq * T * splits, 2}, q.options().dtype(torch::kFloat32));
    CHECK_LAUNCH(launch_attn_decode(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), seq_lens.data_ptr<int>(), out.data_ptr(),
                                    part_acc.data_ptr<float>(), part_ml.data_ptr(), B, Hq, Hkv, T, Lmax, Dh, splits, (float)scale, kv_fp8,
                                    gp, at::cuda::getCurrentCUDAStream()));
}

// Fused q/k RMSNorm + partial RoPE + KV-cache write at the device positions pos_t[b] + t.
// qp, kp, vp: bf16 [.., features] views with unit-stride rows (e.g. column slices of one stacked q|k|v GEMM output):
// qp rows hold [Hq, 512] (q | gate per head), kp / vp rows [Hkv, 256]. inv_freq: fp32 [R/2]; q_out: [B, Hq, T, 256].
// active (optional): int32 [B]; slots with 0 do not write KV.
void attn_prologue(torch::Tensor qp, torch::Tensor kp, torch::Tensor vp, torch::Tensor qn_w, torch::Tensor kn_w, torch::Tensor inv_freq,
                   torch::Tensor pos_t, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor q_out, double eps,
                   c10::optional<torch::Tensor> active) {
    for (auto* t : {&qp, &kp, &vp}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->stride(-1) == 1, "q/k/v rows must be unit-stride bf16");
    for (auto* t : {&qn_w, &kn_w, &q_out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(inv_freq, torch::kFloat32);
    CHECK_CUDA_TENSOR(pos_t, torch::kInt32);
    const bool kv_fp8 = k_cache.scalar_type() == torch::kFloat8_e4m3fn;
    CHECK_CUDA_TENSOR(k_cache, kv_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    CHECK_CUDA_TENSOR(v_cache, k_cache.scalar_type());
    const int64_t B = q_out.size(0), Hq = q_out.size(1), T = q_out.size(2), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    auto q2 = qp.reshape({-1, qp.size(-1)}), k2 = kp.reshape({-1, kp.size(-1)}), v2 = vp.reshape({-1, vp.size(-1)});
    TORCH_CHECK(q2.size(0) == B * T && k2.size(0) == B * T && v2.size(0) == B * T && q2.size(1) == Hq * 512 && k2.size(1) == Hkv * 256 &&
                v2.size(1) == Hkv * 256 && q_out.size(3) == 256 && pos_t.numel() == B, "bad shapes / strides");
    CHECK_LAUNCH(launch_attn_prologue(q2.data_ptr(), k2.data_ptr(), v2.data_ptr(), (int)q2.stride(0), (int)k2.stride(0), (int)v2.stride(0), qn_w.data_ptr(), kn_w.data_ptr(), inv_freq.data_ptr<float>(),
                                      pos_t.data_ptr<int>(), k_cache.data_ptr(), v_cache.data_ptr(), q_out.data_ptr(), B, T, Hq, Hkv, Lmax,
                                      2 * inv_freq.numel(), (float)eps, kv_fp8, active_ptr(active, B), at::cuda::getCurrentCUDAStream()));
}

cudaError_t launch_gdn_conv(const void*, void*, const void*, void*, int, int, const int*, cudaStream_t);
cudaError_t launch_gdn_delta(const void*, const void*, const void*, const void*, const void*, const void*, const void*, float*, void*, int, int,
                             int, float, const int*, cudaStream_t);

// GDN decode step, part 1: depthwise causal conv (kernel 4) + SiLU, updating conv_state in place.
// mixed, out: bf16 [B, C]; conv_state: bf16 [B, C, 3]; w: bf16 [C, 4] (or [C, 1, 4]).
void gdn_conv(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor w, torch::Tensor out, c10::optional<torch::Tensor> active) {
    CHECK_CUDA_TENSOR(mixed, torch::kBFloat16);
    CHECK_CUDA_TENSOR(conv_state, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    const int64_t B = mixed.size(0), C = mixed.size(1);
    TORCH_CHECK(mixed.dim() == 2 && conv_state.size(0) == B && conv_state.size(1) == C && conv_state.size(2) == 3 && w.numel() == C * 4 &&
                out.sizes() == mixed.sizes(), "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv(mixed.data_ptr(), conv_state.data_ptr(), w.data_ptr(), out.data_ptr(), B, C, active_ptr(active, B),
                                 at::cuda::getCurrentCUDAStream()));
}

// GDN decode step, part 2: gated delta rule on the fp32 state + gated RMSNorm.
// qkv: bf16 [B, 2*Hk*128 + Hv*128]; z: bf16 [B, Hv*128]; b, a: bf16 [B, Hv]; A_log, dt_bias: bf16 [Hv];
// norm_w: bf16 [128]; state: fp32 [B, Hv, 128, 128] (in place); out: bf16 [B, Hv*128].
void gdn_delta(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor norm_w,
               torch::Tensor state, torch::Tensor out, int64_t Hk, double eps, c10::optional<torch::Tensor> active) {
    for (auto* t : {&qkv, &z, &b, &a, &A_log, &dt_bias, &norm_w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(state, torch::kFloat32);
    const int64_t B = qkv.size(0), Hv = state.size(1);
    TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(2) == 128 && state.size(3) == 128 && qkv.size(1) == 2 * Hk * 128 + Hv * 128 &&
                z.size(1) == Hv * 128 && b.numel() == B * Hv && a.numel() == B * Hv && A_log.numel() == Hv && dt_bias.numel() == Hv &&
                norm_w.numel() == 128 && out.sizes() == z.sizes() && Hv % Hk == 0, "bad shapes");
    CHECK_LAUNCH(launch_gdn_delta(qkv.data_ptr(), z.data_ptr(), b.data_ptr(), a.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), norm_w.data_ptr(),
                                  state.data_ptr<float>(), out.data_ptr(), B, Hk, Hv, (float)eps, active_ptr(active, B),
                                  at::cuda::getCurrentCUDAStream()));
}

cudaError_t launch_rmsnorm(const void*, const void*, void*, int, int, float, cudaStream_t);
cudaError_t launch_bf16_gemv(const void*, const void*, void*, int, int, int, cudaStream_t);

// out = bf16(x / rms(x) * (1 + w)); x, out: bf16 [M, K]; w: bf16 [K].
void rmsnorm(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor out) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && w.numel() == x.size(1) && out.sizes() == x.sizes(), "bad shapes");
    CHECK_LAUNCH(launch_rmsnorm(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0), x.size(1), (float)eps, at::cuda::getCurrentCUDAStream()));
}

// out = x @ W^T with bf16 W [N, K]; M <= 4.
void bf16_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    check_x_out(x, out, w.size(0));
    TORCH_CHECK(x.size(1) == w.size(1) && out.scalar_type() == torch::kBFloat16, "bad shapes");
    CHECK_LAUNCH(launch_bf16_gemv(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0), w.size(0), w.size(1), at::cuda::getCurrentCUDAStream()));
}

size_t nvfp4_sf_bytes(int, int);
cudaError_t launch_nvfp4_quant(const void*, void*, void*, int, int, float, cudaStream_t);
cudaError_t launch_nvfp4_swizzle_sf(const void*, void*, int, int, cudaStream_t);
cudaError_t launch_nvfp4_gemm(const void*, const void*, const void*, const void*, float, const void*, void*, int, int, int, int, void*, size_t,
                              size_t*, cudaStream_t);

int64_t nvfp4_sf_size(int64_t rows, int64_t K) { return (int64_t)nvfp4_sf_bytes(rows, K); }

// x bf16 [M, K] -> q uint8 [M, K/2] (packed e2m1), sf uint8 [nvfp4_sf_size(M, K)] (swizzled e4m3), static global scale.
void nvfp4_quant(torch::Tensor x, double in_scale, torch::Tensor q, torch::Tensor sf) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(q, torch::kUInt8);
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1);
    const int64_t M = x.size(0), K = x.size(1);
    TORCH_CHECK(x.dim() == 2 && q.size(0) == M && q.size(1) == K / 2 && sf.numel() >= (int64_t)nvfp4_sf_bytes(M, K), "bad shapes");
    CHECK_LAUNCH(launch_nvfp4_quant(x.data_ptr(), q.data_ptr(), sf.data_ptr(), M, K, (float)in_scale, at::cuda::getCurrentCUDAStream()));
}

// row-major e4m3 scales [R, K/16] -> swizzled [nvfp4_sf_size(R, K)]
void nvfp4_swizzle_sf(torch::Tensor src, torch::Tensor dst, int64_t K) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.element_size() == 1 && dst.is_cuda() && dst.element_size() == 1);
    const int64_t R = src.size(0);
    TORCH_CHECK(src.size(1) == K / 16 && dst.numel() >= (int64_t)nvfp4_sf_bytes(R, K), "bad shapes");
    CHECK_LAUNCH(launch_nvfp4_swizzle_sf(src.data_ptr(), dst.data_ptr(), R, K, at::cuda::getCurrentCUDAStream()));
}

// out bf16 [M, N] = alpha * (A . B^T) (+ residual). a [M, K/2], b [N, K/2] packed e2m1; sfa, sfb swizzled.
void nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, double alpha, c10::optional<torch::Tensor> residual,
                torch::Tensor out, int64_t tile) {
    CHECK_CUDA_TENSOR(a, torch::kUInt8);
    CHECK_CUDA_TENSOR(b, torch::kUInt8);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    const int64_t M = a.size(0), K = a.size(1) * 2, N = b.size(0);
    TORCH_CHECK(b.size(1) * 2 == K && out.size(0) == M && out.size(1) == N && K % 128 == 0 && N % 8 == 0, "bad shapes");
    TORCH_CHECK(sfa.numel() >= (int64_t)nvfp4_sf_bytes(M, K) && sfb.numel() >= (int64_t)nvfp4_sf_bytes(N, K), "scale tensors too small");
    const void* c = nullptr;
    if (residual) { CHECK_CUDA_TENSOR(*residual, torch::kBFloat16); TORCH_CHECK(residual->sizes() == out.sizes()); c = residual->data_ptr(); }
    size_t need = 0;
    launch_nvfp4_gemm(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, c, out.data_ptr(), M, N, K, (int)tile, nullptr, 0,
                      &need, at::cuda::getCurrentCUDAStream());
    auto ws = torch::empty({(int64_t)std::max<size_t>(need, 1)}, a.options());
    CHECK_LAUNCH(launch_nvfp4_gemm(a.data_ptr(), sfa.data_ptr(), b.data_ptr(), sfb.data_ptr(), (float)alpha, c, out.data_ptr(), M, N, K, (int)tile,
                                   ws.data_ptr(), need, nullptr, at::cuda::getCurrentCUDAStream()));
}

cudaError_t launch_fp8_quant(const void*, void*, size_t, float, cudaStream_t);
cudaError_t launch_silu_mul_quant(const void*, void*, void*, int, int, float, cudaStream_t);
cudaError_t launch_causal_conv_silu(const void*, int, const void*, const void*, void*, void*, void*, int, int, int, int, cudaStream_t);
cudaError_t launch_add_rmsnorm(const void*, const void*, const void*, float, int, int, void*, void*, void*, void*, float, void*, float, cudaStream_t);
cudaError_t launch_gate_fp8(const void*, const void*, int, int, int, int, void*, float, cudaStream_t);
cudaError_t launch_gated_rmsnorm(const void*, const void*, int, int, const void*, void*, int, float, cudaStream_t);

// out (float8_e4m3fn, same shape) = saturate(x / scale)
void fp8_quant(torch::Tensor x, double scale, torch::Tensor out) {
    CHECK_CUDA_TENSOR(x, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kFloat8_e4m3fn);
    TORCH_CHECK(out.numel() == x.numel());
    CHECK_LAUNCH(launch_fp8_quant(x.data_ptr(), out.data_ptr(), x.numel(), (float)scale, at::cuda::getCurrentCUDAStream()));
}

// gu bf16 [M, 2I] (gate | up) -> NVFP4 of silu(gate) * up: q uint8 [M, I/2], sf swizzled [nvfp4_sf_size(M, I)]
void silu_mul_quant(torch::Tensor gu, double in_scale, torch::Tensor q, torch::Tensor sf) {
    CHECK_CUDA_TENSOR(gu, torch::kBFloat16);
    CHECK_CUDA_TENSOR(q, torch::kUInt8);
    const int64_t M = gu.size(0), I = gu.size(1) / 2;
    TORCH_CHECK(q.size(0) == M && q.size(1) == I / 2 && sf.numel() >= (int64_t)nvfp4_sf_bytes(M, I), "bad shapes");
    CHECK_LAUNCH(launch_silu_mul_quant(gu.data_ptr(), q.data_ptr(), sf.data_ptr(), M, I, (float)in_scale, at::cuda::getCurrentCUDAStream()));
}

// bf16(silu(bf16(depthwise causal conv(x)))) for x [T, C] bf16 (rows may be strided), state [C, 3] (previous 3 inputs), w [C, 4];
// output channels are split into three contiguous tensors outs[0..2] = [T, c1], [T, c2 - c1], [T, C - c2].
void causal_conv_silu(torch::Tensor x, torch::Tensor state, torch::Tensor w, std::vector<torch::Tensor> outs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1, "x: bf16 [T, C], unit-stride rows");
    TORCH_CHECK(outs.size() == 3, "three outputs");
    for (auto* t : {&state, &w, &outs[0], &outs[1], &outs[2]}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    const int64_t T = x.size(0), C = x.size(1), c1 = outs[0].size(-1), c2 = c1 + outs[1].size(-1);
    TORCH_CHECK(state.numel() == C * 3 && w.numel() == C * 4 && c2 + outs[2].size(-1) == C, "bad shapes");
    for (auto& o : outs) TORCH_CHECK(o.numel() == T * o.size(-1), "outputs must be [T, channels]");
    CHECK_LAUNCH(launch_causal_conv_silu(x.data_ptr(), (int)x.stride(0), state.data_ptr(), w.data_ptr(), outs[0].data_ptr(), outs[1].data_ptr(),
                                         outs[2].data_ptr(), (int)c1, (int)c2, T, C, at::cuda::getCurrentCUDAStream()));
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
                                    at::cuda::getCurrentCUDAStream()));
}

// out e4m3 [T, H*D] = (o * sigmoid(gate)) / scale; o bf16 [T, H*D]; gate: rows [H, 2D] (q | gate), strided
void gate_fp8(torch::Tensor o, torch::Tensor gate, int64_t D, double scale, torch::Tensor out) {
    CHECK_CUDA_TENSOR(o, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kFloat8_e4m3fn);
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 && gate.dim() == 2 && gate.stride(1) == 1, "gate rows unit-stride");
    const int64_t T = o.size(0), H = o.size(1) / D;
    TORCH_CHECK(gate.size(0) == T && gate.size(1) == 2 * H * D && out.numel() == o.numel(), "bad shapes");
    CHECK_LAUNCH(launch_gate_fp8(o.data_ptr(), gate.data_ptr(), (int)gate.stride(0), T, H, D, out.data_ptr(), (float)scale,
                                 at::cuda::getCurrentCUDAStream()));
}

// o, out: bf16 [T*heads, 128] contiguous; z: bf16 [T, heads*128] (rows may be strided); w [128]
void gated_rmsnorm(torch::Tensor o, torch::Tensor z, torch::Tensor w, double eps, torch::Tensor out) {
    for (auto* t : {&o, &w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    TORCH_CHECK(z.is_cuda() && z.scalar_type() == torch::kBFloat16 && z.dim() == 2 && z.stride(1) == 1, "z: bf16 [T, heads*128], unit-stride rows");
    const int64_t heads = z.size(1) / 128;
    TORCH_CHECK(z.size(1) % 128 == 0 && o.numel() == z.size(0) * z.size(1) && out.numel() == o.numel() && w.numel() == 128, "bad shapes");
    CHECK_LAUNCH(launch_gated_rmsnorm(o.data_ptr(), z.data_ptr(), (int)z.stride(0), (int)heads, w.data_ptr(), out.data_ptr(), o.numel() / 128,
                                      (float)eps, at::cuda::getCurrentCUDAStream()));
}

cudaError_t launch_gdn_conv_multi(const void*, const void*, const void*, void*, int, int, int, cudaStream_t);
cudaError_t launch_gdn_conv_commit(const void*, void*, const int*, int, int, int, cudaStream_t);
cudaError_t launch_gdn_delta_multi(const void*, const void*, const void*, const void*, const void*, const void*, const void*, float*, void*, int, int,
                                   int, float, int, const int*, cudaStream_t);

// speculative verify: conv over T tokens per slot from the (unchanged) conv state. mixed, out: bf16 [B, T, C].
void gdn_conv_multi(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor w, torch::Tensor out) {
    for (auto* t : {&mixed, &conv_state, &w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    const int64_t B = conv_state.size(0), C = conv_state.size(1), T = mixed.numel() / (B * C);
    TORCH_CHECK(mixed.numel() == B * T * C && out.numel() == mixed.numel() && w.numel() == C * 4, "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv_multi(mixed.data_ptr(), conv_state.data_ptr(), w.data_ptr(), out.data_ptr(), B, T, C, at::cuda::getCurrentCUDAStream()));
}

// speculative commit: conv state advanced by n[b] of the T verified tokens. n: int32 [B] on the device.
void gdn_conv_commit(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor n) {
    CHECK_CUDA_TENSOR(mixed, torch::kBFloat16);
    CHECK_CUDA_TENSOR(conv_state, torch::kBFloat16);
    CHECK_CUDA_TENSOR(n, torch::kInt32);
    const int64_t B = conv_state.size(0), C = conv_state.size(1), T = mixed.numel() / (B * C);
    TORCH_CHECK(n.numel() == B && mixed.numel() == B * T * C, "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv_commit(mixed.data_ptr(), conv_state.data_ptr(), n.data_ptr<int>(), B, T, C, at::cuda::getCurrentCUDAStream()));
}

// speculative verify (n = None: outputs for all T tokens, state untouched) or commit (n: int32 [B]: state
// advanced by n[b] tokens and written, no outputs). qkv [B, T, C]; z [B, T, Hv*128]; b, a [B, T, Hv].
void gdn_delta_multi(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a, torch::Tensor A_log, torch::Tensor dt_bias,
                     torch::Tensor norm_w, torch::Tensor state, torch::Tensor out, int64_t Hk, double eps, c10::optional<torch::Tensor> n) {
    for (auto* t : {&qkv, &z, &b, &a, &A_log, &dt_bias, &norm_w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(state, torch::kFloat32);
    const int64_t B = state.size(0), Hv = state.size(1), C = 2 * Hk * 128 + Hv * 128, T = qkv.numel() / (B * C);
    TORCH_CHECK(qkv.numel() == B * T * C && z.numel() == B * T * Hv * 128 && b.numel() == B * T * Hv && a.numel() == b.numel() &&
                out.numel() == z.numel(), "bad shapes");
    const int* np = nullptr;
    if (n) { CHECK_CUDA_TENSOR(*n, torch::kInt32); TORCH_CHECK(n->numel() == B); np = n->data_ptr<int>(); }
    CHECK_LAUNCH(launch_gdn_delta_multi(qkv.data_ptr(), z.data_ptr(), b.data_ptr(), a.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
                                        norm_w.data_ptr(), state.data_ptr<float>(), out.data_ptr(), B, Hk, Hv, (float)eps, T, np,
                                        at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_conv_multi", &gdn_conv_multi, "spec verify: GDN conv over T tokens, state read-only");
    m.def("gdn_conv_commit", &gdn_conv_commit, "spec commit: advance the GDN conv state by n tokens");
    m.def("gdn_delta_multi", &gdn_delta_multi, "spec verify / commit: GDN delta rule over T tokens", py::arg("qkv"), py::arg("z"), py::arg("b"),
          py::arg("a"), py::arg("A_log"), py::arg("dt_bias"), py::arg("norm_w"), py::arg("state"), py::arg("out"), py::arg("Hk"), py::arg("eps"),
          py::arg("n") = py::none());
    m.def("fp8_quant", &fp8_quant, "bf16 -> e4m3, static scale");
    m.def("silu_mul_quant", &silu_mul_quant, "fused silu(gate)*up -> NVFP4 (swizzled scales)");
    m.def("causal_conv_silu", &causal_conv_silu, "GDN causal depthwise conv (k=4) + SiLU, token-major, split q|k|v outputs");
    m.def("add_rmsnorm", &add_rmsnorm, "fused residual add + RMSNorm (+ NVFP4 / FP8 quantization)", py::arg("x"), py::arg("y"), py::arg("w"),
          py::arg("eps"), py::arg("x_out") = py::none(), py::arg("n_out") = py::none(), py::arg("q4") = py::none(), py::arg("sf4") = py::none(),
          py::arg("in_scale4") = 1.0, py::arg("q8") = py::none(), py::arg("in_scale8") = 1.0);
    m.def("gate_fp8", &gate_fp8, "attention output gate fused with FP8 quantization");
    m.def("gated_rmsnorm", &gated_rmsnorm, "GDN gated RMSNorm over rows of 128");
    m.def("nvfp4_sf_size", &nvfp4_sf_size, "bytes of a swizzled NVFP4 scale tensor for [rows, K]");
    m.def("nvfp4_quant", &nvfp4_quant, "bf16 -> NVFP4 (packed e2m1 + swizzled e4m3 scales), static global scale");
    m.def("nvfp4_swizzle_sf", &nvfp4_swizzle_sf, "row-major NVFP4 scales -> CUTLASS 128x4 layout");
    m.def("nvfp4_gemm", &nvfp4_gemm, "CUTLASS SM120 NVFP4 x NVFP4 GEMM, bf16 out", py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"),
          py::arg("alpha"), py::arg("residual"), py::arg("out"), py::arg("tile") = 0);
    m.def("rmsnorm", &rmsnorm, "zero-centered RMSNorm (1 + w)");
    m.def("bf16_gemv", &bf16_gemv, "bf16-weight GEMV, M<=4");
    m.def("gdn_conv", &gdn_conv, "GDN decode: causal conv step + SiLU", py::arg("mixed"), py::arg("conv_state"), py::arg("w"), py::arg("out"),
          py::arg("active") = py::none());
    m.def("gdn_delta", &gdn_delta, "GDN decode: gated delta rule + gated RMSNorm", py::arg("qkv"), py::arg("z"), py::arg("b"), py::arg("a"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("norm_w"), py::arg("state"), py::arg("out"), py::arg("Hk"), py::arg("eps"),
          py::arg("active") = py::none());
    m.def("attn_decode", &attn_decode, "split-KV GQA decode attention, head_dim 256", py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("seq_lens"), py::arg("out"), py::arg("splits"), py::arg("scale"), py::arg("gate") = py::none());
    m.def("attn_prologue", &attn_prologue, "fused q/k norm + partial RoPE + KV write", py::arg("qp"), py::arg("kp"), py::arg("vp"),
          py::arg("qn_w"), py::arg("kn_w"), py::arg("inv_freq"), py::arg("pos_t"), py::arg("k_cache"), py::arg("v_cache"), py::arg("q_out"),
          py::arg("eps"), py::arg("active") = py::none());
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 W4A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("sf"), py::arg("gscale"),
          py::arg("residual"), py::arg("out"));
    m.def("nvfp4_swiglu", &nvfp4_swiglu, "fused silu(gate)*up NVFP4 GEMV, M<=4");
    m.def("fp8_gemv", &fp8_gemv, "FP8 per-tensor (or per-row) W8A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("scale"),
          py::arg("residual"), py::arg("out"), py::arg("row_scale") = py::none());
}
