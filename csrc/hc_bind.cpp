// hc_bind.cpp -- torch bindings of hc.cu (and the grouped norm of norm.cu). bindings.cpp calls bind_hc(m) from the module;
// each kernel file of qwen4_exp has its own binding file. Each wrapper checks the shapes / dtypes / strides and launches
// on the current stream (CUDA-graph safe). Python module: engine/model/hc_kernels.py.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace {

constexpr int64_t S = 4, H = 2560, R = 320, MAXM = 16;

#define HC_CHECK_TENSOR(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dt), #t " must be a contiguous CUDA " #dt " tensor")
#define HC_CHECK_LAUNCH(e) TORCH_CHECK((e) == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e))

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

// the row stride of a 2-D bf16 [M, n] view with a contiguous last dim (a column slice of a wider matrix is fine)
int64_t ld_of(const torch::Tensor& t, int64_t M, int64_t n, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.dim() == 2 && t.size(0) == M && t.size(1) == n && t.stride(1) == 1, name,
                ": bf16 CUDA [", M, ", ", n, "] with a contiguous last dim");
    return M == 1 ? n : t.stride(0);
}

}  // namespace

cudaError_t launch_rmsnorm_grouped(const void*, const void*, void*, int, int, int, float, cudaStream_t);
cudaError_t launch_hc_mix(const void*, int, const void*, const float*, const void*, void*, const void*, int, void*, int, cudaStream_t);
cudaError_t launch_hc_post(const void*, const void*, void*, int, cudaStream_t);

// out = the zero-centered RMSNorm of each of the `groups` equal slices of a row, with the matching slice of w:
// x, out bf16 [M, groups * K], w bf16 [groups * K] (hc_norm: groups 4, K 2560).
void hc_norm(torch::Tensor x, torch::Tensor w, double eps, torch::Tensor out, int64_t groups) {
    HC_CHECK_TENSOR(x, torch::kBFloat16);
    HC_CHECK_TENSOR(w, torch::kBFloat16);
    HC_CHECK_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && groups >= 1 && x.size(1) % groups == 0 && (x.size(1) / groups) % 8 == 0,
                "hc_norm: x [M, groups * K] with K % 8 == 0");
    TORCH_CHECK(w.numel() == x.size(1) && out.sizes() == x.sizes(), "hc_norm: w [groups * K], out like x");
    HC_CHECK_LAUNCH(launch_rmsnorm_grouped(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.size(0), x.size(1) / groups, groups, (float)eps, stream()));
}

// The low-rank mix: y bf16 [M, 2560] = the mean over the 4 streams of bf16(sigmoid(a W_up^T)) * n, a = silu(d / 4).
// d bf16 [M, 320] (the down product; rows may be strided); w_up bf16 [10240, 320]; n bf16 [M, 10240] (the hc_norm
// output). With e (bf16 [M, 4], the inject product, rows may be strided) it also writes inj bf16 [M, 4] = 2 sigmoid(e / 4).
// M <= 16. With w_scale (fp32 [10240]), w_up is the INT8 decode copy int8 [10240, 320]: W_up = w_up * w_scale per row.
void hc_mix(torch::Tensor d, torch::Tensor w_up, torch::Tensor n, torch::Tensor y, c10::optional<torch::Tensor> e,
            c10::optional<torch::Tensor> inj, c10::optional<torch::Tensor> w_scale) {
    if (w_scale) {
        HC_CHECK_TENSOR(w_up, torch::kChar);
        HC_CHECK_TENSOR(*w_scale, torch::kFloat32);
        TORCH_CHECK(w_scale->numel() == S * H, "hc_mix: w_scale fp32 [10240]");
    } else {
        HC_CHECK_TENSOR(w_up, torch::kBFloat16);
    }
    HC_CHECK_TENSOR(n, torch::kBFloat16);
    HC_CHECK_TENSOR(y, torch::kBFloat16);
    const int64_t M = n.size(0);
    TORCH_CHECK(n.dim() == 2 && n.size(1) == S * H && M >= 1 && M <= MAXM, "hc_mix: n [M <= 16, 10240]");
    TORCH_CHECK(w_up.dim() == 2 && w_up.size(0) == S * H && w_up.size(1) == R, "hc_mix: w_up [10240, 320]");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == M && y.size(1) == H, "hc_mix: y [M, 2560]");
    const int64_t ldd = ld_of(d, M, R, "d");
    TORCH_CHECK(!e == !inj, "hc_mix: e and inj go together");
    int64_t lde = 0;
    if (e) {
        lde = ld_of(*e, M, S, "e");
        HC_CHECK_TENSOR(*inj, torch::kBFloat16);
        TORCH_CHECK(inj->numel() == M * S, "hc_mix: inj [M, 4]");
    }
    HC_CHECK_LAUNCH(launch_hc_mix(d.data_ptr(), (int)ldd, w_up.data_ptr(), w_scale ? w_scale->data_ptr<float>() : nullptr, n.data_ptr(), y.data_ptr(),
                                  e ? e->data_ptr() : nullptr, (int)lde, inj ? inj->data_ptr() : nullptr, (int)M, stream()));
}

// The injection after the sublayer, in place: x4[:, s] = bf16(x4[:, s] + bf16(o * inj[:, s])). o bf16 [M, 2560]; inj bf16
// [M, 4]; x4 bf16 [M, 10240].
void hc_post(torch::Tensor o, torch::Tensor inj, torch::Tensor x4) {
    HC_CHECK_TENSOR(o, torch::kBFloat16);
    HC_CHECK_TENSOR(inj, torch::kBFloat16);
    HC_CHECK_TENSOR(x4, torch::kBFloat16);
    const int64_t M = x4.size(0);
    TORCH_CHECK(x4.dim() == 2 && x4.size(1) == S * H, "hc_post: x4 [M, 10240]");
    TORCH_CHECK(o.numel() == M * H && inj.numel() == M * S, "hc_post: o [M, 2560], inj [M, 4]");
    HC_CHECK_LAUNCH(launch_hc_post(o.data_ptr(), inj.data_ptr(), x4.data_ptr(), (int)M, stream()));
}

void bind_hc(pybind11::module_& m) {
    namespace py = pybind11;
    m.def("hc_norm", &hc_norm, "grouped zero-centered RMSNorm (hyper-connection streams)", py::arg("x"), py::arg("w"), py::arg("eps"),
          py::arg("out"), py::arg("groups") = 4);
    m.def("hc_mix", &hc_mix, "hyper-connection low-rank mix: up product, sigmoid, stream mean (+ the injection weights)", py::arg("d"),
          py::arg("w_up"), py::arg("n"), py::arg("y"), py::arg("e") = py::none(), py::arg("inj") = py::none(), py::arg("w_scale") = py::none());
    m.def("hc_post", &hc_post, "hyper-connection injection, in place: x4 += o * inj per stream", py::arg("o"), py::arg("inj"), py::arg("x4"));
}
