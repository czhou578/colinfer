// moe_bind.cpp -- torch bindings of moe.cu (decode) and moe_prefill.cu (prefill). bindings.cpp calls bind_moe(m) from the
// module (qsa_bind.cpp and hc_bind.cpp bind the other qwen4_exp kernels). Each wrapper checks the shapes, dtypes and the
// constraints of the kernels, and launches on the current stream (CUDA graph capture safe: no allocation, no sync).
// Python module: engine/model/moe_kernels.py.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

cudaError_t launch_moe_route(const void*, int, int, int, int*, float*, float*, int*, cudaStream_t);
cudaError_t launch_moe_gate_up(const void*, const int*, int, int, int, int, const void*, const void*, const float*, void*, cudaStream_t);
cudaError_t launch_moe_down(const void*, const int*, int, int, int, int, const void*, const void*, const float*, float*, cudaStream_t);
cudaError_t launch_moe_combine(const float*, const float*, const float*, const void*, void*, int, int, cudaStream_t);
cudaError_t launch_moe_grouped_gemm(int, int, int, int, const int*, const int*, const void*, const void*, const void*, const void*, void*,
                                    const float*, const float*, void*, size_t, size_t*, cudaStream_t);
cudaError_t launch_moe_gather_quant(const void*, const int*, const int*, const int*, const int*, const float*, int, int, void*, void*, cudaStream_t);
cudaError_t launch_moe_swiglu_quant(const void*, const int*, const int*, const int*, const int*, const float*, int, int, void*, void*, cudaStream_t);
cudaError_t launch_moe_combine_prefill(const void*, const int*, const float*, const float*, const void*, void*, int, int, cudaStream_t);

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

// int32 [n], the plan tensors of the prefill path
void check_i32(const torch::Tensor& t, int64_t n, const char* name) {
    check(t, torch::kInt32, name);
    TORCH_CHECK(t.dim() == 1 && t.size(0) == n, name, " must be int32 [", n, "]");
}

// The scale rows of the gathered operand of P pairs over E experts, each group padded to 128 rows: an upper bound.
int64_t sf_rows_max(int64_t P, int64_t E) { return P + 127 * std::min(P, E); }

// the 1-byte swizzled scales of a gathered operand of P rows (groups of E experts) and K columns
void check_gathered_sf(const torch::Tensor& sf, int64_t P, int64_t E, int64_t K, const char* name) {
    TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && sf.element_size() == 1 && sf.numel() >= sf_rows_max(P, E) * (K / 16), name,
                ": 1-byte CUDA, at least moe_sf_rows(P, E) * K/16 bytes");
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

// h bf16 [10M, I] = silu(x Wg^T * gs[e, 0]) * (x Wu^T * gs[e, 1]) for each pair (m, k), e = ids[m, k]. Each row of ids holds 10
// distinct experts in [0, E) (moe_route makes them so): a pair with an expert out of range gets no output row.
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

// ===================================================================================================== prefill (moe_prefill.cu)
// The plan, from the sorted pairs: order int32 [P] (pair indices sorted by expert, stable), off int32 [E + 1] (first sorted row
// of each expert, off[E] = P), sfo int32 [E + 1] (first scale row of each group, the groups padded to 128 rows).

// q uint8 [P, K/2], sf: gathered row r = x[order[r] / 10], quantized with in_scale[e, 0] into group e's swizzled scales
void moe_gather_quant(torch::Tensor x, torch::Tensor ids, torch::Tensor order, torch::Tensor off, torch::Tensor sfo, torch::Tensor in_scale,
                      torch::Tensor q, torch::Tensor sf) {
    check(x, torch::kBFloat16, "x");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 64 == 0, "x must be bf16 [T, K], K % 64 == 0");
    const int64_t T = x.size(0), K = x.size(1), P = T * TOPK, E = in_scale.size(0);
    check_ids(ids, T);
    check_i32(order, P, "order");
    check_i32(off, E + 1, "off");
    check_i32(sfo, E + 1, "sfo");
    check_gs(in_scale, E);
    check(q, torch::kUInt8, "q");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == P && q.size(1) * 2 == K, "q must be uint8 [10T, K/2]");
    check_gathered_sf(sf, P, E, K, "sf");
    check_launch(launch_moe_gather_quant(x.data_ptr(), ids.data_ptr<int>(), order.data_ptr<int>(), off.data_ptr<int>(), sfo.data_ptr<int>(),
                                         in_scale.data_ptr<float>(), (int)P, (int)K, q.data_ptr(), sf.data_ptr(), stream()),
                 "moe_gather_quant");
}

// q uint8 [P, I/2], sf = NVFP4(silu(d1[:, :I]) * d1[:, I:]) per sorted row, with in_scale[e, 2]
void moe_swiglu_quant(torch::Tensor d1, torch::Tensor ids, torch::Tensor order, torch::Tensor off, torch::Tensor sfo, torch::Tensor in_scale,
                      torch::Tensor q, torch::Tensor sf) {
    check(d1, torch::kBFloat16, "d1");
    TORCH_CHECK(d1.dim() == 2 && d1.size(0) % TOPK == 0 && d1.size(1) % 128 == 0, "d1 must be bf16 [10T, 2I], I % 64 == 0");
    const int64_t P = d1.size(0), I = d1.size(1) / 2, E = in_scale.size(0);
    check_ids(ids, P / TOPK);
    check_i32(order, P, "order");
    check_i32(off, E + 1, "off");
    check_i32(sfo, E + 1, "sfo");
    check_gs(in_scale, E);
    check(q, torch::kUInt8, "q");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == P && q.size(1) * 2 == I, "q must be uint8 [10T, I/2]");
    check_gathered_sf(sf, P, E, I, "sf");
    check_launch(launch_moe_swiglu_quant(d1.data_ptr(), ids.data_ptr<int>(), order.data_ptr<int>(), off.data_ptr<int>(), sfo.data_ptr<int>(),
                                         in_scale.data_ptr<float>(), (int)P, (int)I, q.data_ptr(), sf.data_ptr(), stream()),
                 "moe_swiglu_quant");
}

// d bf16 [P, N] = alpha_e * (a_e . w[e]^T) per expert group (the CUTLASS SM120 block-scaled grouped GEMM). which 0: gate | up
// (w = w_gu [E, 2I, K/2], K % 256 == 0, alpha = in_scale[e, 0] * gs[e, 0]); which 1: down (w = w_dn [E, K, I/2], I % 128 == 0,
// alpha = in_scale[e, 2] * gs[e, 2]). N % 128 == 0. sfw: the swizzled weight scales [E, N * K/16].
void moe_grouped_gemm(int64_t which, torch::Tensor a, torch::Tensor sfa, torch::Tensor w, torch::Tensor sfw, torch::Tensor d, torch::Tensor off,
                      torch::Tensor sfo, torch::Tensor in_scale, torch::Tensor gs) {
    TORCH_CHECK(which == 0 || which == 1, "which: 0 (gate | up) or 1 (down)");
    check(a, torch::kUInt8, "a");
    check(w, torch::kUInt8, "w");
    check(d, torch::kBFloat16, "d");
    TORCH_CHECK(w.dim() == 3 && a.dim() == 2 && a.size(1) == w.size(2), "a [P, K/2] and w [E, N, K/2]");
    const int64_t P = a.size(0), K = a.size(1) * 2, E = w.size(0), N = w.size(1);
    TORCH_CHECK(N % 128 == 0 && K % (which == 0 ? 256 : 128) == 0, "moe_grouped_gemm: N % 128 == 0, K % 256 == 0 (gate | up) or 128 (down)");
    TORCH_CHECK(d.dim() == 2 && d.size(0) == P && d.size(1) == N, "d must be bf16 [P, N]");
    check_sf(sfw, E, N, K, "sfw");
    check_gathered_sf(sfa, P, E, K, "sfa");
    check_i32(off, E + 1, "off");
    check_i32(sfo, E + 1, "sfo");
    check_gs(in_scale, E);
    check_gs(gs, E);
    size_t need = 0;
    check_launch(launch_moe_grouped_gemm((int)which, (int)E, (int)N, (int)K, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr, 0, &need, stream()),
                 "moe_grouped_gemm (size)");
    auto ws = torch::empty({(int64_t)need}, a.options());
    check_launch(launch_moe_grouped_gemm((int)which, (int)E, (int)N, (int)K, off.data_ptr<int>(), sfo.data_ptr<int>(), a.data_ptr(), sfa.data_ptr(),
                                         w.data_ptr(), sfw.data_ptr(), d.data_ptr(), in_scale.data_ptr<float>(), gs.data_ptr<float>(), ws.data_ptr(),
                                         need, nullptr, stream()),
                 "moe_grouped_gemm");
}

// y bf16 [T, K] = bf16(sum_{k = 0..9} w[t, k] * d2[inv[10t + k]] + sg[t] * shared[t]) (fp32, slot order); inv int32 [10T] = the
// sorted row of each pair
void moe_combine_prefill(torch::Tensor d2, torch::Tensor inv, torch::Tensor w, torch::Tensor sg, c10::optional<torch::Tensor> shared, torch::Tensor y) {
    check(y, torch::kBFloat16, "y");
    TORCH_CHECK(y.dim() == 2 && y.size(1) % 8 == 0, "y must be bf16 [T, K], K % 8 == 0");
    const int64_t T = y.size(0), K = y.size(1);
    check(d2, torch::kBFloat16, "d2");
    TORCH_CHECK(d2.dim() == 2 && d2.size(0) == T * TOPK && d2.size(1) == K, "d2 must be bf16 [10T, K]");
    check_i32(inv, T * TOPK, "inv");
    check(w, torch::kFloat32, "w");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == T && w.size(1) == TOPK, "w must be float32 [T, 10]");
    check(sg, torch::kFloat32, "sg");
    TORCH_CHECK(sg.numel() == T, "sg must be float32 [T]");
    const void* sh = nullptr;
    if (shared) {
        check(*shared, torch::kBFloat16, "shared");
        TORCH_CHECK(shared->sizes() == y.sizes(), "shared must be bf16 [T, K]");
        sh = shared->data_ptr();
    }
    check_launch(launch_moe_combine_prefill(d2.data_ptr(), inv.data_ptr<int>(), w.data_ptr<float>(), sg.data_ptr<float>(), sh, y.data_ptr(), (int)T,
                                            (int)K, stream()),
                 "moe_combine_prefill");
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
    m.def("moe_sf_rows", &sf_rows_max, "scale rows of a gathered prefill operand of P pairs over E experts (groups padded to 128)",
          py::arg("P"), py::arg("E"));
    m.def("moe_gather_quant", &moe_gather_quant, "prefill MoE: the pairs' rows of x, sorted by expert, quantized to NVFP4 per group",
          py::arg("x"), py::arg("ids"), py::arg("order"), py::arg("off"), py::arg("sfo"), py::arg("in_scale"), py::arg("q"), py::arg("sf"));
    m.def("moe_swiglu_quant", &moe_swiglu_quant, "prefill MoE: silu(gate) * up of the grouped gate | up output, quantized to NVFP4 per group",
          py::arg("d1"), py::arg("ids"), py::arg("order"), py::arg("off"), py::arg("sfo"), py::arg("in_scale"), py::arg("q"), py::arg("sf"));
    m.def("moe_grouped_gemm", &moe_grouped_gemm, "prefill MoE: CUTLASS SM120 NVFP4 grouped GEMM, one group per expert, bf16 out",
          py::arg("which"), py::arg("a"), py::arg("sfa"), py::arg("w"), py::arg("sfw"), py::arg("d"), py::arg("off"), py::arg("sfo"),
          py::arg("in_scale"), py::arg("gs"));
    m.def("moe_combine_prefill", &moe_combine_prefill, "prefill MoE: the weighted sum of each token's pair rows in slot order + the gated shared expert",
          py::arg("d2"), py::arg("inv"), py::arg("w"), py::arg("sg"), py::arg("shared"), py::arg("y"));
}
