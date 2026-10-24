// qsa_bind.cpp -- torch bindings of qsa.cu. bindings.cpp calls bind_qsa(m) from the module. Kept apart from bindings.cpp so that
// the kernel files of the qwen4_exp work can be developed in parallel. Each wrapper checks the shapes / dtypes / strides and
// launches on the current stream, so a CUDA graph can capture each op. Python wrappers: engine/model/qsa_kernels.py.
#include <c10/cuda/CUDAStream.h>
#include <string.h>
#include <torch/extension.h>

#include "shapes.h"

namespace {

constexpr int64_t DI = 128, HI = 4, TOPK = 512, MAX_ROWS = 16;

#define QSA_CHECK_TENSOR(t, dt) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dt), #t " must be a contiguous CUDA " #dt " tensor")
#define QSA_CHECK_LAUNCH(e) TORCH_CHECK((e) == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(e))

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

const int* opt_i32(const c10::optional<torch::Tensor>& a, int64_t n, const char* name) {
    if (!a) return nullptr;
    TORCH_CHECK(a->is_cuda() && a->is_contiguous() && a->scalar_type() == torch::kInt32 && a->numel() == n, name, ": contiguous int32 CUDA with ",
                n, " elements");
    return a->data_ptr<int>();
}

void check_pos(const torch::Tensor& pos_t, int64_t B) {
    QSA_CHECK_TENSOR(pos_t, torch::kInt32);
    TORCH_CHECK(pos_t.numel() == B, "pos_t: int32 [B = ", B, "]");
}

}  // namespace

cudaError_t launch_qsa_prologue(const void*, int, const void*, const float*, const int*, void*, void*, int, int, int, int, float, const int*,
                                const int*, const int*, int, int, cudaStream_t);
cudaError_t launch_qsa_block_keys(const void*, void*, const void*, const float*, const int*, int, int, int, int, float, const int*, const int*,
                                  const int*, int, int, cudaStream_t);
cudaError_t launch_qsa_score(const void*, const void*, const int*, float*, int, int, int, int, cudaStream_t);
cudaError_t launch_qsa_topk(const float*, const int*, int*, int*, int, int, int, int, cudaStream_t);
int qsa_attn_nb();
cudaError_t launch_qsa_attn(const void*, const void*, const void*, const int*, const int*, int, const int*, void*, float*, void*, int, int, int, int,
                            int, float, const void*, cudaStream_t);

// Indexer prologue for T new rows per slot. idx: bf16 [B, T, 640] (or [B * T, 640]) rows of the index projection (4 query
// heads of 128, then the key), unit stride inside a row, rows evenly strided (a column slice of the stacked q | k | v |
// index output is fine). q_out: bf16 [B, T, 4, 128] = RoPE(q_layernorm(q)). ik: bf16 [B, Lmax, 128]; the raw key of row t
// goes to ik[b, pos_t[b] + t] in active slots. RoPE: rope_pos (int32 [B * T, 3]), else pos + rope_delta (int32 [B]),
// else pos; the first 2 * len(inv_freq) dims rotate.
void qsa_prologue(torch::Tensor idx, torch::Tensor qn_w, torch::Tensor inv_freq, torch::Tensor pos_t, torch::Tensor ik, torch::Tensor q_out,
                  double eps, c10::optional<torch::Tensor> active, c10::optional<torch::Tensor> rope_delta, c10::optional<torch::Tensor> rope_pos,
                  int64_t sec_h, int64_t sec_w) {
    TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == torch::kBFloat16 && idx.size(-1) == (HI + 1) * DI && idx.stride(-1) == 1,
                "idx: bf16 CUDA [..., 640] with a contiguous last dim");
    QSA_CHECK_TENSOR(qn_w, torch::kBFloat16);
    QSA_CHECK_TENSOR(inv_freq, torch::kFloat32);
    QSA_CHECK_TENSOR(ik, torch::kBFloat16);
    QSA_CHECK_TENSOR(q_out, torch::kBFloat16);
    TORCH_CHECK(q_out.dim() == 4 && q_out.size(2) == HI && q_out.size(3) == DI, "q_out: bf16 [B, T, 4, 128]");
    const int64_t B = q_out.size(0), T = q_out.size(1);
    check_pos(pos_t, B);
    TORCH_CHECK(ik.dim() == 3 && ik.size(0) == B && ik.size(2) == DI, "ik: bf16 [B, Lmax, 128]");
    TORCH_CHECK(qn_w.numel() == DI && inv_freq.numel() >= 1 && 2 * inv_freq.numel() <= DI, "qn_w [128], inv_freq [R / 2] with R <= 128");
    auto i2 = idx.reshape({-1, idx.size(-1)});
    TORCH_CHECK(i2.size(0) == B * T && i2.stride(1) == 1 && i2.stride(0) >= i2.size(1), "idx: B * T rows, evenly strided (got ", i2.size(0), ")");
    QSA_CHECK_LAUNCH(launch_qsa_prologue(i2.data_ptr(), (int)i2.stride(0), qn_w.data_ptr(), inv_freq.data_ptr<float>(), pos_t.data_ptr<int>(),
                                         ik.data_ptr(), q_out.data_ptr(), B, T, ik.size(1), 2 * inv_freq.numel(), (float)eps,
                                         opt_i32(active, B, "active"), opt_i32(rope_delta, B, "rope_delta"), opt_i32(rope_pos, B * T * 3, "rope_pos"),
                                         (int)sec_h, (int)sec_w, stream()));
}

// Block keys of the blocks that the rows [pos_t[b], pos_t[b] + T) complete (4 j + 3 in the range): ikb[b, j] =
// RoPE(k_layernorm(bf16(mean(ik[b, 4 j .. 4 j + 3])))) at the rotary position of token 4 j. ik bf16 [B, Lmax, 128]; ikb bf16
// [B, Lmax / 4, 128]. The rotary position: rope_pos (int32 [B, Lmax, 3], the slot's record of the rows' (t, h, w)
// positions), else 4 j + rope_delta[b], else 4 j. Decode, verify, prefill (any T) and a rebuild (pos 0, T = L) use it.
void qsa_block_keys(torch::Tensor ik, torch::Tensor ikb, torch::Tensor kn_w, torch::Tensor inv_freq, torch::Tensor pos_t, int64_t T, double eps,
                    c10::optional<torch::Tensor> active, c10::optional<torch::Tensor> rope_delta, c10::optional<torch::Tensor> rope_pos,
                    int64_t sec_h, int64_t sec_w) {
    QSA_CHECK_TENSOR(ik, torch::kBFloat16);
    QSA_CHECK_TENSOR(ikb, torch::kBFloat16);
    QSA_CHECK_TENSOR(kn_w, torch::kBFloat16);
    QSA_CHECK_TENSOR(inv_freq, torch::kFloat32);
    TORCH_CHECK(ik.dim() == 3 && ik.size(2) == DI && ik.size(1) % 4 == 0, "ik: bf16 [B, Lmax, 128], Lmax % 4 == 0");
    const int64_t B = ik.size(0), Lmax = ik.size(1);
    TORCH_CHECK(ikb.dim() == 3 && ikb.size(0) == B && ikb.size(1) == Lmax / 4 && ikb.size(2) == DI, "ikb: bf16 [B, Lmax / 4, 128]");
    TORCH_CHECK(kn_w.numel() == DI && inv_freq.numel() >= 1 && 2 * inv_freq.numel() <= DI, "kn_w [128], inv_freq [R / 2] with R <= 128");
    TORCH_CHECK(T >= 1 && T <= Lmax, "T must be in [1, Lmax]");
    check_pos(pos_t, B);
    QSA_CHECK_LAUNCH(launch_qsa_block_keys(ik.data_ptr(), ikb.data_ptr(), kn_w.data_ptr(), inv_freq.data_ptr<float>(), pos_t.data_ptr<int>(), B, T,
                                           Lmax, 2 * inv_freq.numel(), (float)eps, opt_i32(active, B, "active"), opt_i32(rope_delta, B, "rope_delta"),
                                           opt_i32(rope_pos, B * Lmax * 3, "rope_pos"), (int)sec_h, (int)sec_w, stream()));
}

// scores[b, t, j] = sum_h relu(q_idx[b, t, h] . ikb[b, j]) / sqrt(128), fp32, for the T rows; -inf for j >= (p_t + 1) / 4.
// q_idx bf16 [B, T <= 16, 4, 128]; ikb bf16 [B, NBL, 128]; scores fp32 [B, S >= T, NBL] (rows t >= T untouched; the
// blocks at or past (pos + T) / 4 are not written).
void qsa_score(torch::Tensor q_idx, torch::Tensor ikb, torch::Tensor pos_t, torch::Tensor scores) {
    QSA_CHECK_TENSOR(q_idx, torch::kBFloat16);
    QSA_CHECK_TENSOR(ikb, torch::kBFloat16);
    QSA_CHECK_TENSOR(scores, torch::kFloat32);
    TORCH_CHECK(q_idx.dim() == 4 && q_idx.size(2) == HI && q_idx.size(3) == DI, "q_idx: bf16 [B, T, 4, 128]");
    const int64_t B = q_idx.size(0), T = q_idx.size(1);
    TORCH_CHECK(T >= 1 && T <= MAX_ROWS, "qsa_score: T must be in [1, 16]");
    TORCH_CHECK(ikb.dim() == 3 && ikb.size(0) == B && ikb.size(2) == DI, "ikb: bf16 [B, NBL, 128]");
    TORCH_CHECK(scores.dim() == 3 && scores.size(0) == B && scores.size(1) >= T && scores.size(2) == ikb.size(1), "scores: fp32 [B, S >= T, NBL]");
    check_pos(pos_t, B);
    QSA_CHECK_LAUNCH(launch_qsa_score(q_idx.data_ptr(), ikb.data_ptr(), pos_t.data_ptr<int>(), scores.data_ptr<float>(), B, T, ikb.size(1),
                                      scores.size(1), stream()));
}

// The top 512 blocks of each of the T rows (all blocks when nb = (p_t + 1) / 4 <= 512), ties to the lower index, in
// ascending order: sel int32 [B, S, 512] (-1 past the count), nsel int32 [B, S]. scores fp32 [B, S, NBL] from qsa_score.
void qsa_topk(torch::Tensor scores, torch::Tensor pos_t, int64_t T, torch::Tensor sel, torch::Tensor nsel) {
    QSA_CHECK_TENSOR(scores, torch::kFloat32);
    QSA_CHECK_TENSOR(sel, torch::kInt32);
    QSA_CHECK_TENSOR(nsel, torch::kInt32);
    TORCH_CHECK(scores.dim() == 3, "scores: fp32 [B, S, NBL]");
    const int64_t B = scores.size(0), S = scores.size(1), NBL = scores.size(2);
    TORCH_CHECK(T >= 1 && T <= S, "qsa_topk: T must be in [1, S]");
    TORCH_CHECK(NBL % 4 == 0, "qsa_topk: NBL must be a multiple of 4 (16-byte aligned score rows)");
    TORCH_CHECK(sel.dim() == 3 && sel.size(0) == B && sel.size(1) == S && sel.size(2) == TOPK, "sel: int32 [B, S, 512]");
    TORCH_CHECK(nsel.dim() == 2 && nsel.size(0) == B && nsel.size(1) == S, "nsel: int32 [B, S]");
    check_pos(pos_t, B);
    QSA_CHECK_LAUNCH(launch_qsa_topk(scores.data_ptr<float>(), pos_t.data_ptr<int>(), sel.data_ptr<int>(), nsel.data_ptr<int>(), B, T, NBL, S,
                                     stream()));
}

// Sparse decode attention, GQA 24 / 2 over an fp8 KV cache. Row t of slot b (position p_t = pos_t[b] + t) attends to the
// 4 nsel[b, t] tokens of its selected blocks sel[b, t, :nsel] and the tail positions 4 ((p_t + 1) / 4) .. p_t.
// q bf16 [B, 24, T, 256]; caches e4m3 [B, 2, Lmax, 256]; sel int32 [B, S, 512], nsel int32 [B, S] (S >= T).
// gate (optional): the q_proj output [B, T, 24, 512]; with it, out is [B, T, 24 * 256] = attn * sigmoid(gate), else
// out has the q layout.
void qsa_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor sel, torch::Tensor nsel, torch::Tensor pos_t,
              torch::Tensor out, double scale, c10::optional<torch::Tensor> gate) {
    QSA_CHECK_TENSOR(q, torch::kBFloat16);
    QSA_CHECK_TENSOR(k_cache, torch::kFloat8_e4m3fn);
    QSA_CHECK_TENSOR(v_cache, torch::kFloat8_e4m3fn);
    QSA_CHECK_TENSOR(sel, torch::kInt32);
    QSA_CHECK_TENSOR(nsel, torch::kInt32);
    QSA_CHECK_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && q.size(3) == HEAD_DIM, "q: bf16 [B, Hq, T, 256]");
    const int64_t B = q.size(0), Hq = q.size(1), T = q.size(2);
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes() && k_cache.size(0) == B && k_cache.size(3) == HEAD_DIM,
                "caches: e4m3 [B, Hkv, Lmax, 256]");
    const int64_t Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(Hq == 24 && Hkv == 2, "qsa_attn: compiled for 24 query heads over 2 KV heads, got ", Hq, " / ", Hkv);
    TORCH_CHECK(sel.dim() == 3 && sel.size(0) == B && sel.size(1) >= T && sel.size(2) == TOPK, "sel: int32 [B, S >= T, 512]");
    TORCH_CHECK(nsel.dim() == 2 && nsel.size(0) == B && nsel.size(1) == sel.size(1), "nsel: int32 [B, S]");
    TORCH_CHECK(out.numel() == q.numel(), "out: B * Hq * T * 256 elements");
    check_pos(pos_t, B);
    const void* gp = nullptr;
    if (gate) {
        QSA_CHECK_TENSOR(*gate, torch::kBFloat16);
        TORCH_CHECK(gate->numel() == 2 * q.numel(), "gate: [B, T, Hq, 512]");
        gp = gate->data_ptr();
    }
    const int64_t NB = qsa_attn_nb();
    auto part_acc = torch::empty({B * Hq * T * NB, HEAD_DIM}, q.options().dtype(torch::kFloat32));
    auto part_ml = torch::empty({B * Hq * T * NB, 2}, q.options().dtype(torch::kFloat32));
    QSA_CHECK_LAUNCH(launch_qsa_attn(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), sel.data_ptr<int>(), nsel.data_ptr<int>(), sel.size(1),
                                     pos_t.data_ptr<int>(), out.data_ptr(), part_acc.data_ptr<float>(), part_ml.data_ptr(), B, Hq, Hkv, T, Lmax,
                                     (float)scale, gp, stream()));
}

int ple_read_rows(int, const int64_t*, uint8_t*, int64_t, int64_t, bool, int);

// The PLE n-gram rows (csrc/ple_io.cpp): out[i] = the out.size(1) bytes of the file fd at the byte offset offsets[i], read by
// a pool of `threads` threads (the pool keeps the largest count asked). offsets: int64 [n] on the CPU; out: uint8 [n, row <=
// 4096] on the CPU (pinned or not). direct: fd is open with O_DIRECT (the reads then take the aligned page(s) of a row).
// Releases the GIL while it reads.
void ple_gather(int64_t fd, torch::Tensor offsets, torch::Tensor out, bool direct, int64_t threads) {
    TORCH_CHECK(offsets.device().is_cpu() && offsets.is_contiguous() && offsets.scalar_type() == torch::kInt64 && offsets.dim() == 1,
                "ple_gather: offsets must be a contiguous CPU int64 [n] tensor");
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous() && out.scalar_type() == torch::kUInt8 && out.dim() == 2 && out.size(0) == offsets.size(0),
                "ple_gather: out must be a contiguous CPU uint8 [n, row] tensor");
    TORCH_CHECK(out.size(1) >= 1 && out.size(1) <= 4096 && threads >= 1 && threads <= 256 && fd >= 0, "ple_gather: row in [1, 4096], threads in [1, 256]");
    TORCH_CHECK(offsets.numel() == 0 || offsets.min().item<int64_t>() >= 0, "ple_gather: negative offset");
    int err;
    {
        pybind11::gil_scoped_release nogil;
        err = ple_read_rows((int)fd, offsets.data_ptr<int64_t>(), out.data_ptr<uint8_t>(), out.size(0), out.size(1), direct, (int)threads);
    }
    TORCH_CHECK(err == 0, "ple_gather: read failed: ", strerror(err));
}

void bind_qsa(pybind11::module_& m) {
    namespace py = pybind11;
    m.def("ple_gather", &ple_gather, "PLE n-gram table rows: pread of n rows at byte offsets by a thread pool (GIL released)", py::arg("fd"),
          py::arg("offsets"), py::arg("out"), py::arg("direct") = false, py::arg("threads") = 32);
    m.def("qsa_prologue", &qsa_prologue, "QSA indexer prologue: q_layernorm + RoPE of the 4 query heads, raw key write", py::arg("idx"),
          py::arg("qn_w"), py::arg("inv_freq"), py::arg("pos_t"), py::arg("ik"), py::arg("q_out"), py::arg("eps"), py::arg("active") = py::none(),
          py::arg("rope_delta") = py::none(), py::arg("rope_pos") = py::none(), py::arg("sec_h") = 11, py::arg("sec_w") = 10);
    m.def("qsa_block_keys", &qsa_block_keys, "QSA block keys of the blocks that the new rows complete", py::arg("ik"), py::arg("ikb"),
          py::arg("kn_w"), py::arg("inv_freq"), py::arg("pos_t"), py::arg("T"), py::arg("eps"), py::arg("active") = py::none(),
          py::arg("rope_delta") = py::none(), py::arg("rope_pos") = py::none(), py::arg("sec_h") = 11, py::arg("sec_w") = 10);
    m.def("qsa_score", &qsa_score, "QSA block scores (tensor cores, fp32)", py::arg("q_idx"), py::arg("ikb"), py::arg("pos_t"), py::arg("scores"));
    m.def("qsa_topk", &qsa_topk, "QSA top-512 blocks per row, ties to the lower index, ascending", py::arg("scores"), py::arg("pos_t"), py::arg("T"),
          py::arg("sel"), py::arg("nsel"));
    m.def("qsa_attn", &qsa_attn, "QSA sparse decode attention (GQA 24 / 2, fp8 KV, output gate)", py::arg("q"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("sel"), py::arg("nsel"), py::arg("pos_t"), py::arg("out"), py::arg("scale"), py::arg("gate") = py::none());
}
