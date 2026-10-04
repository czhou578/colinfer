// Torch bindings for the decode kernels (built by engine/kernels/__init__.py).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t launch_nvfp4_gemv(const void*, const void*, const void*, float, const void*, void*, bool, int, int, int, cudaStream_t);
cudaError_t launch_nvfp4_swiglu(const void*, const void*, const void*, float, const void*, const void*, float, void*, int, int, int, cudaStream_t);
cudaError_t launch_fp8_gemv(const void*, const void*, float, const void*, void*, bool, int, int, int, cudaStream_t);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 W4A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("sf"), py::arg("gscale"),
          py::arg("residual"), py::arg("out"));
    m.def("nvfp4_swiglu", &nvfp4_swiglu, "fused silu(gate)*up NVFP4 GEMV, M<=4");
    m.def("fp8_gemv", &fp8_gemv, "FP8 per-tensor W8A16 GEMV, M<=4", py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("residual"),
          py::arg("out"));
}
