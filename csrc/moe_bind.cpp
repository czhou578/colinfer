// moe_bind.cpp -- torch bindings of moe.cu. bindings.cpp calls bind_moe(m) from the module. Kept apart from bindings.cpp so that
// the kernel files of the qwen4_exp work can be developed in parallel. Each wrapper checks the shapes, dtypes and the
// constraints of the kernels, and launches on the current stream (CUDA graph capture safe: no allocation, no sync).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t launch_moe_route(const void*, int, int, int, int*, float*, float*, int*, cudaStream_t);
cudaError_t launch_moe_gate_up(const void*, const int*, int, int, int, int, const void*, const void*, const float*, void*, cudaStream_t);
cudaError_t launch_moe_down(const void*, const int*, int, int, int, int, const void*, const void*, const float*, float*, cudaStream_t);
cudaError_t launch_moe_combine(const float*, const float*, const float*, const void*, void*, int, int, cudaStream_t);

namespace {

constexpr int64_t TOPK = 10, MAX_M = 48, MAX_E = 512;

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

void check(const torch::Tensor& t, torch::ScalarType dt, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt, name, " must be a contiguous CUDA ", c10::toString(dt), " tensor");
}

void check_launch(cudaError_t e, const char* name) { TORCH_CHECK(e == cudaSuccess, name, ": kernel launch failed: ", cudaGetErrorString(e)); }

// ids int32 [M, 10] on the device of x
void check_ids(const torch::Tensor& ids, int64_t M) {
    check(ids, torch::kInt32, "ids");
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) == M && ids.size(1) == TOPK, "ids must be int32 [M, 10]");
}

// gs float32 [E, 3] (the weight_scale_2 of gate, up, down)
void check_gs(const torch::Tensor& gs, int64_t E) {
    check(gs, torch::kFloat32, "gs");
    TORCH_CHECK(gs.dim() == 2 && gs.size(0) == E && gs.size(1) == 3, "gs must be float32 [E, 3]");
}

// swizzled e4m3 scales, 1-byte (uint8 or float8_e4m3fn), E x rows x K / 16 bytes with size(0) == E
void check_sf(const torch::Tensor& sf, int64_t E, int64_t rows, int64_t K, const char* name) {
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1 && sf.dim() >= 1 && sf.size(0) == E && sf.numel() == E * rows * (K / 16),
                name, ": the swizzled 1-byte scales of each expert, [E, ", rows * (K / 16), "]");
}

}  // namespace

// ids int32 [M, 10], w f32 [M, 10] (bf16-rounded), sg f32 [M] = bf16(sigmoid(logits[:, E]))
// from logits bf16 [M, NL >= E + 1]; stats: optional int32 [2] (+= distinct experts, += 1 per launch)
void moe_route(torch::Tensor logits, int64_t E, torch::Tensor ids, torch::Tensor w, torch::Tensor sg, c10::optional<torch::Tensor> stats) {
    check(logits, torch::kBFloat16, "logits");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= 1, "logits must be bf16 [M, NL]");
    const int64_t M = logits.size(0), NL = logits.size(1);
    TORCH_CHECK(E >= TOPK && E <= MAX_E && NL >= E + 1, "moe_route: 10 <= E <= 512 and NL >= E + 1 (column E is the shared-expert gate)");
    check_ids(ids, M);
    check(w, torch::kFloat32, "w");
    TORCH_CHECK(w.sizes() == ids.sizes(), "w must be float32 [M, 10]");
    check(sg, torch::kFloat32, "sg");
    TORCH_CHECK(sg.numel() == M, "sg must be float32 [M]");
    int* st = nullptr;
    if (stats) {
        check(*stats, torch::kInt32, "stats");
        TORCH_CHECK(stats->numel() == 2, "stats must be int32 [2]");
        st = stats->data_ptr<int>();
    }
    check_launch(launch_moe_route(logits.data_ptr(), (int)NL, (int)M, (int)E, ids.data_ptr<int>(), w.data_ptr<float>(), sg.data_ptr<float>(), st,
                                  stream()),
                 "moe_route");
}

// h bf16 [10M, I] = silu(x Wg^T * gs[e, 0]) * (x Wu^T * gs[e, 1]) for each pair (m, k), e = ids[m, k]
// x bf16 [M <= 48, K], w_gu uint8 [E, 2I, K/2] (gate rows, then up rows), sf_gu swizzled [E, 2I * K/16]. K % 512 == 0, I % 128 == 0.
void moe_gate_up(torch::Tensor x, torch::Tensor ids, torch::Tensor w_gu, torch::Tensor sf_gu, torch::Tensor gs, torch::Tensor h) {
    check(x, torch::kBFloat16, "x");
    check(w_gu, torch::kUInt8, "w_gu");
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= MAX_M, "x must be bf16 [M <= 48, K]");
    const int64_t M = x.size(0), K = x.size(1);
    TORCH_CHECK(w_gu.dim() == 3 && w_gu.size(2) * 2 == K && w_gu.size(1) % 256 == 0, "w_gu must be uint8 [E, 2I, K/2] with I % 128 == 0");
    const int64_t E = w_gu.size(0), I = w_gu.size(1) / 2;
    TORCH_CHECK(K % 512 == 0, "moe_gate_up: K % 512 == 0");
    check_ids(ids, M);
    check_sf(sf_gu, E, 2 * I, K, "sf_gu");
    check_gs(gs, E);
    check(h, torch::kBFloat16, "h");
    TORCH_CHECK(h.dim() == 2 && h.size(0) == M * TOPK && h.size(1) == I, "h must be bf16 [10M, I]");
    check_launch(launch_moe_gate_up(x.data_ptr(), ids.data_ptr<int>(), (int)M, (int)K, (int)I, (int)E, w_gu.data_ptr(), sf_gu.data_ptr(),
                                    gs.data_ptr<float>(), h.data_ptr(), stream()),
                 "moe_gate_up");
}

// yp f32 [10M, K] = (h W_down^T) * gs[e, 2] for each pair; h bf16 [10M, I], w_dn uint8 [E, K, I/2], sf_dn swizzled [E, K * I/16].
// K % 128 == 0, I in {128, 256, 384, 512, 640}, M <= 48.
void moe_down(torch::Tensor h, torch::Tensor ids, torch::Tensor w_dn, torch::Tensor sf_dn, torch::Tensor gs, torch::Tensor yp) {
    check(h, torch::kBFloat16, "h");
    check(w_dn, torch::kUInt8, "w_dn");
    TORCH_CHECK(h.dim() == 2 && h.size(0) % TOPK == 0 && h.size(0) >= TOPK && h.size(0) <= MAX_M * TOPK, "h must be bf16 [10M, I], M <= 48");
    const int64_t M = h.size(0) / TOPK, I = h.size(1);
    TORCH_CHECK(w_dn.dim() == 3 && w_dn.size(2) * 2 == I, "w_dn must be uint8 [E, K, I/2]");
    const int64_t E = w_dn.size(0), K = w_dn.size(1);
    TORCH_CHECK(K % 128 == 0 && (I == 128 || I == 256 || I == 384 || I == 512 || I == 640), "moe_down: K % 128 == 0, I in {128, 256, 384, 512, 640}");
    check_ids(ids, M);
    check_sf(sf_dn, E, K, I, "sf_dn");
    check_gs(gs, E);
    check(yp, torch::kFloat32, "yp");
    TORCH_CHECK(yp.dim() == 2 && yp.size(0) == M * TOPK && yp.size(1) == K, "yp must be float32 [10M, K]");
    check_launch(launch_moe_down(h.data_ptr(), ids.data_ptr<int>(), (int)M, (int)K, (int)I, (int)E, w_dn.data_ptr(), sf_dn.data_ptr(),
                                 gs.data_ptr<float>(), yp.data_ptr<float>(), stream()),
                 "moe_down");
}

// y bf16 [M, K] = bf16(sum_{k = 0..9} w[m, k] * yp[10m + k] + sg[m] * shared[m]) (fp32, slot order); shared: bf16 [M, K] or None
void moe_combine(torch::Tensor yp, torch::Tensor w, torch::Tensor sg, c10::optional<torch::Tensor> shared, torch::Tensor y) {
    check(y, torch::kBFloat16, "y");
    TORCH_CHECK(y.dim() == 2 && y.size(0) >= 1 && y.size(1) % 8 == 0, "y must be bf16 [M, K], K % 8 == 0");
    const int64_t M = y.size(0), K = y.size(1);
    check(yp, torch::kFloat32, "yp");
    TORCH_CHECK(yp.dim() == 2 && yp.size(0) == M * TOPK && yp.size(1) == K, "yp must be float32 [10M, K]");
    check(w, torch::kFloat32, "w");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == M && w.size(1) == TOPK, "w must be float32 [M, 10]");
    check(sg, torch::kFloat32, "sg");
    TORCH_CHECK(sg.numel() == M, "sg must be float32 [M]");
    const void* sh = nullptr;
    if (shared) {
        check(*shared, torch::kBFloat16, "shared");
        TORCH_CHECK(shared->sizes() == y.sizes(), "shared must be bf16 [M, K]");
        sh = shared->data_ptr();
    }
    check_launch(launch_moe_combine(yp.data_ptr<float>(), w.data_ptr<float>(), sg.data_ptr<float>(), sh, y.data_ptr(), (int)M, (int)K, stream()),
                 "moe_combine");
}

void bind_moe(pybind11::module_& m) {
    namespace py = pybind11;
    m.def("moe_route", &moe_route, "MoE routing: top-10 of the softmax by (logit desc, index asc), renormalized, bf16-rounded; sigmoid gate",
          py::arg("logits"), py::arg("E"), py::arg("ids"), py::arg("w"), py::arg("sg"), py::arg("stats") = py::none());
    m.def("moe_gate_up", &moe_gate_up, "decode MoE: silu(x Wg^T) * (x Wu^T) per (row, expert) pair, NVFP4 experts, swizzled scales",
          py::arg("x"), py::arg("ids"), py::arg("w_gu"), py::arg("sf_gu"), py::arg("gs"), py::arg("h"));
    m.def("moe_down", &moe_down, "decode MoE: h W_down^T per pair, fp32 partials", py::arg("h"), py::arg("ids"), py::arg("w_dn"),
          py::arg("sf_dn"), py::arg("gs"), py::arg("yp"));
    m.def("moe_combine", &moe_combine, "decode MoE: the weighted sum of the pair outputs in slot order + the gated shared expert",
          py::arg("yp"), py::arg("w"), py::arg("sg"), py::arg("shared"), py::arg("y"));
}
