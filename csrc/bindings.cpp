// Torch bindings for the decode kernels (built by engine/kernels/__init__.py).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t launch_nvfp4_gemv(const void*, const void*, const void*, float, const void*, void*, bool, int, int, int, cudaStream_t);
cudaError_t launch_nvfp4_swiglu(const void*, const void*, const void*, float, const void*, const void*, float, void*, int, int, int, cudaStream_t);
cudaError_t launch_fp8_gemv(const void*, const void*, float, const void*, void*, bool, int, int, int, cudaStream_t);
cudaError_t launch_attn_decode(const void*, const void*, const void*, const int*, void*, float*, void*, int, int, int, int, int, int, int, float,
                               bool, cudaStream_t);

#define CHECK_CUDA_TENSOR(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dt), #t " must be a contiguous CUDA " #dt " tensor")
#define CHECK_LAUNCH(e) TORCH_CHECK((e) == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e))

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

// out = (x @ W^T) * scale (+ residual). w: float8_e4m3fn or uint8 [N, K].
void fp8_gemv(torch::Tensor x, torch::Tensor w, double scale, c10::optional<torch::Tensor> residual, torch::Tensor out) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.element_size() == 1 && w.dim() == 2, "w must be 1-byte [N, K]");
    const int64_t N = w.size(0), K = w.size(1);
    check_x_out(x, out, N);
    TORCH_CHECK(x.size(1) == K, "shape mismatch");
    const void* r = nullptr;
    if (residual) { CHECK_CUDA_TENSOR(*residual, torch::kBFloat16); TORCH_CHECK(residual->sizes() == out.sizes()); r = residual->data_ptr(); }
    CHECK_LAUNCH(launch_fp8_gemv(x.data_ptr(), w.data_ptr(), (float)scale, r, out.data_ptr(), out.scalar_type() == torch::kFloat32,
                                 x.size(0), N, K, at::cuda::getCurrentCUDAStream()));
}

// out[b, h, t] = softmax(q k^T * scale) v over the first seq_lens[b] - (T-1-t) cached positions.
// q, out: bf16 [B, Hq, T, 256]; k_cache, v_cache: bf16 or float8_e4m3fn [B, Hkv, Lmax, 256]; seq_lens: int32 [B] (device).
void attn_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor seq_lens, torch::Tensor out, int64_t splits,
                 double scale) {
    CHECK_CUDA_TENSOR(q, torch::kBFloat16);
    const bool kv_fp8 = k_cache.scalar_type() == torch::kFloat8_e4m3fn;
    CHECK_CUDA_TENSOR(k_cache, kv_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    CHECK_CUDA_TENSOR(v_cache, k_cache.scalar_type());
    CHECK_CUDA_TENSOR(seq_lens, torch::kInt32);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes() && out.sizes() == q.sizes(), "bad shapes");
    const int64_t B = q.size(0), Hq = q.size(1), T = q.size(2), Dh = q.size(3), Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(3) == Dh && seq_lens.numel() == B && splits >= 1, "bad shapes");
    auto part_acc = torch::empty({B * Hq * T * splits, Dh}, q.options().dtype(torch::kFloat32));
    auto part_ml = torch::empty({B * Hq * T * splits, 2}, q.options().dtype(torch::kFloat32));
    CHECK_LAUNCH(launch_attn_decode(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), seq_lens.data_ptr<int>(), out.data_ptr(),
                                    part_acc.data_ptr<float>(), part_ml.data_ptr(), B, Hq, Hkv, T, Lmax, Dh, splits, (float)scale, kv_fp8,
                                    at::cuda::getCurrentCUDAStream()));
}

cudaError_t launch_gdn_conv(const void*, void*, const void*, void*, int, int, cudaStream_t);
cudaError_t launch_gdn_delta(const void*, const void*, const void*, const void*, const void*, const void*, const void*, float*, void*, int, int,
                             int, float, cudaStream_t);

// GDN decode step, part 1: depthwise causal conv (kernel 4) + SiLU, updating conv_state in place.
// mixed, out: bf16 [B, C]; conv_state: bf16 [B, C, 3]; w: bf16 [C, 4] (or [C, 1, 4]).
void gdn_conv(torch::Tensor mixed, torch::Tensor conv_state, torch::Tensor w, torch::Tensor out) {
    CHECK_CUDA_TENSOR(mixed, torch::kBFloat16);
    CHECK_CUDA_TENSOR(conv_state, torch::kBFloat16);
    CHECK_CUDA_TENSOR(w, torch::kBFloat16);
    CHECK_CUDA_TENSOR(out, torch::kBFloat16);
    const int64_t B = mixed.size(0), C = mixed.size(1);
    TORCH_CHECK(mixed.dim() == 2 && conv_state.size(0) == B && conv_state.size(1) == C && conv_state.size(2) == 3 && w.numel() == C * 4 &&
                out.sizes() == mixed.sizes(), "bad shapes");
    CHECK_LAUNCH(launch_gdn_conv(mixed.data_ptr(), conv_state.data_ptr(), w.data_ptr(), out.data_ptr(), B, C, at::cuda::getCurrentCUDAStream()));
}

// GDN decode step, part 2: gated delta rule on the fp32 state + gated RMSNorm.
// qkv: bf16 [B, 2*Hk*128 + Hv*128]; z: bf16 [B, Hv*128]; b, a: bf16 [B, Hv]; A_log, dt_bias: bf16 [Hv];
// norm_w: bf16 [128]; state: fp32 [B, Hv, 128, 128] (in place); out: bf16 [B, Hv*128].
void gdn_delta(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor norm_w,
               torch::Tensor state, torch::Tensor out, int64_t Hk, double eps) {
    for (auto* t : {&qkv, &z, &b, &a, &A_log, &dt_bias, &norm_w, &out}) CHECK_CUDA_TENSOR(*t, torch::kBFloat16);
    CHECK_CUDA_TENSOR(state, torch::kFloat32);
    const int64_t B = qkv.size(0), Hv = state.size(1);
    TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(2) == 128 && state.size(3) == 128 && qkv.size(1) == 2 * Hk * 128 + Hv * 128 &&
                z.size(1) == Hv * 128 && b.numel() == B * Hv && a.numel() == B * Hv && A_log.numel() == Hv && dt_bias.numel() == Hv &&
                norm_w.numel() == 128 && out.sizes() == z.sizes() && Hv % Hk == 0, "bad shapes");
    CHECK_LAUNCH(launch_gdn_delta(qkv.data_ptr(), z.data_ptr(), b.data_ptr(), a.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), norm_w.data_ptr(),
                                  state.data_ptr<float>(), out.data_ptr(), B, Hk, Hv, (float)eps, at::cuda::getCurrentCUDAStream()));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "zero-centered RMSNorm (1 + w)");
    m.def("bf16_gemv", &bf16_gemv, "bf16-weight GEMV, M<=4");
    m.def("gdn_conv", &gdn_conv, "GDN decode: causal conv step + SiLU");
    m.def("gdn_delta", &gdn_delta, "GDN decode: gated delta rule + gated RMSNorm");
    m.def("attn_decode", &attn_decode, "split-KV GQA decode attention, head_dim 256");
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 W4A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("sf"), py::arg("gscale"),
          py::arg("residual"), py::arg("out"));
    m.def("nvfp4_swiglu", &nvfp4_swiglu, "fused silu(gate)*up NVFP4 GEMV, M<=4");
    m.def("fp8_gemv", &fp8_gemv, "FP8 per-tensor W8A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("residual"),
          py::arg("out"));
}
