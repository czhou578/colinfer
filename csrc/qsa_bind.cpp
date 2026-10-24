// qsa_bind.cpp -- torch bindings of qsa.cu, qsa_prefill.cu and hc_prefill.cu, and of the PLE row reader of ple_io.cpp (ple_gather). bindings.cpp calls
// bind_qsa(m) from the module; each kernel file of qwen4_exp has its own binding file. Each wrapper checks the shapes /
// dtypes / strides and launches on the current stream, so a CUDA graph can capture each op (ple_gather is a host function and runs no kernel). Python wrappers:
// engine/model/qsa_kernels.py.
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

// ------------------------------------------------------------------------------------------------------------------
// qsa_prefill.cu: the sparse attention of a prefill chunk of one slot (T rows at pos_t[0] .. pos_t[0] + T - 1).
cudaError_t launch_qsa_score_rows(const void*, const void*, const int*, float*, int, int, cudaStream_t);
int qsa_union_cap();
cudaError_t launch_qsa_union(const int*, const int*, const int*, int*, int*, int, int, cudaStream_t);
cudaError_t launch_qsa_union_attn(const void*, const void*, const void*, const int*, const int*, const int*, const void*, int, void*, int, int, int,
                                  int, float, cudaStream_t);

// The block scores of all T rows of a chunk in one launch, bit-identical per row to qsa_score: q_idx bf16 [T, 4, 128]
// (any leading shape with T * 4 rows of 128); ikb bf16 [NBL, 128] (or [1, NBL, 128]); scores fp32 [T, NBL].
void qsa_score_rows(torch::Tensor q_idx, torch::Tensor ikb, torch::Tensor pos_t, torch::Tensor scores) {
    QSA_CHECK_TENSOR(q_idx, torch::kBFloat16);
    QSA_CHECK_TENSOR(ikb, torch::kBFloat16);
    QSA_CHECK_TENSOR(scores, torch::kFloat32);
    TORCH_CHECK(q_idx.numel() % (HI * DI) == 0 && q_idx.size(-1) == DI, "q_idx: bf16 [T, 4, 128]");
    const int64_t T = q_idx.numel() / (HI * DI);
    TORCH_CHECK(ikb.size(-1) == DI && ikb.numel() / DI == ikb.size(-2) && (ikb.dim() == 2 || ikb.size(0) == 1), "ikb: bf16 [NBL, 128] of one slot");
    const int64_t NBL = ikb.size(-2);
    TORCH_CHECK(scores.numel() == T * NBL && scores.size(-1) == NBL, "scores: fp32 [T, NBL]");
    check_pos(pos_t, 1);
    QSA_CHECK_LAUNCH(launch_qsa_score_rows(q_idx.data_ptr(), ikb.data_ptr(), pos_t.data_ptr<int>(), scores.data_ptr<float>(), T, NBL, stream()));
}

// The union of the selected and tail blocks of each group of 4 rows: sel int32 [T, 512], nsel int32 [T] (the chunk's
// selection); ul int32 [ceil(T / 4), qsa_union_cap()] gets (block << 4) | (the mask of the group's rows that attend to
// it) in ascending block order, uc int32 [ceil(T / 4)] the count. nblocks: (pos0 + T) / 4 + 1 or more.
void qsa_union(torch::Tensor sel, torch::Tensor nsel, torch::Tensor pos_t, torch::Tensor ul, torch::Tensor uc, int64_t nblocks) {
    for (auto* t : {&sel, &nsel, &ul, &uc}) QSA_CHECK_TENSOR(*t, torch::kInt32);
    const int64_t T = nsel.numel(), NG = (T + 3) / 4;
    TORCH_CHECK(T >= 1 && sel.numel() == T * TOPK, "sel: int32 [T, 512], nsel: int32 [T]");
    TORCH_CHECK(ul.numel() == NG * qsa_union_cap() && uc.numel() == NG, "ul: int32 [ceil(T / 4), qsa_union_cap()], uc: int32 [ceil(T / 4)]");
    TORCH_CHECK(nblocks >= 1 && nblocks <= (1 << 22), "nblocks out of range");
    check_pos(pos_t, 1);
    QSA_CHECK_LAUNCH(launch_qsa_union(sel.data_ptr<int>(), nsel.data_ptr<int>(), pos_t.data_ptr<int>(), ul.data_ptr<int>(), uc.data_ptr<int>(), T,
                                      nblocks, stream()));
}

// The sparse attention of a chunk over the group unions (GQA 12 Hkv / Hkv): q bf16 [1, Hq, T, 256] (the prologue
// output), caches e4m3 [1, Hkv, Lmax, 256] with the chunk's rows written, gate: the q rows bf16 [T, >= Hq * 512] with
// unit-stride rows (q | gate per head; a column view is fine), out bf16 [T, Hq * 256] = attn * sigmoid(gate).
void qsa_union_attn(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor pos_t, torch::Tensor ul, torch::Tensor uc,
                    torch::Tensor gate, torch::Tensor out, double scale) {
    QSA_CHECK_TENSOR(q, torch::kBFloat16);
    QSA_CHECK_TENSOR(k_cache, torch::kFloat8_e4m3fn);
    QSA_CHECK_TENSOR(v_cache, torch::kFloat8_e4m3fn);
    QSA_CHECK_TENSOR(ul, torch::kInt32);
    QSA_CHECK_TENSOR(uc, torch::kInt32);
    QSA_CHECK_TENSOR(out, torch::kBFloat16);
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == HEAD_DIM, "q: bf16 [1, Hq, T, 256]");
    const int64_t Hq = q.size(1), T = q.size(2);
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes() && k_cache.size(0) == 1 && k_cache.size(3) == HEAD_DIM,
                "caches: e4m3 [1, Hkv, Lmax, 256]");
    const int64_t Hkv = k_cache.size(1), Lmax = k_cache.size(2);
    TORCH_CHECK(Hq == 12 * Hkv, "qsa_union_attn: 12 query heads per KV head");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 && gate.dim() == 2 && gate.size(0) == T && gate.stride(1) == 1 &&
                    gate.size(1) >= Hq * 2 * HEAD_DIM,
                "gate: bf16 [T, Hq * 512] with unit-stride rows");
    const int64_t NG = (T + 3) / 4;
    TORCH_CHECK(ul.numel() == NG * qsa_union_cap() && uc.numel() == NG, "ul / uc: the qsa_union output of T rows");
    TORCH_CHECK(out.numel() == T * Hq * HEAD_DIM, "out: bf16 [T, Hq * 256]");
    check_pos(pos_t, 1);
    QSA_CHECK_LAUNCH(launch_qsa_union_attn(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), pos_t.data_ptr<int>(), ul.data_ptr<int>(),
                                           uc.data_ptr<int>(), gate.data_ptr(), (int)gate.stride(0), out.data_ptr(), Hq, Hkv, T, Lmax, (float)scale,
                                           stream()));
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

// The PLE row cache (csrc/ple_io.cpp). The handle frees the cache when Python drops it.
struct PleRowCache;
PleRowCache* ple_cache_create(int64_t, int64_t, int64_t);
void ple_cache_destroy(PleRowCache*);
void ple_cache_lookup(PleRowCache*, const int64_t*, int64_t, uint8_t*, uint8_t*);
void ple_cache_insert(PleRowCache*, const int64_t*, int64_t, const uint8_t*);
void ple_cache_counters(PleRowCache*, int64_t*);
void ple_cache_reset_counters(PleRowCache*);

struct PleCache {
    PleRowCache* c;
    int64_t row;
    PleCache(int64_t capacity, int64_t row_, int64_t ways) : c(ple_cache_create(capacity, row_, ways)), row(row_) {}
    ~PleCache() { ple_cache_destroy(c); }
    PleCache(const PleCache&) = delete;
    PleCache& operator=(const PleCache&) = delete;
};

namespace {
void check_rows(const torch::Tensor& ids, const torch::Tensor& rows, int64_t row, const char* fn) {
    TORCH_CHECK(ids.device().is_cpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt64 && ids.dim() == 1, fn,
                ": ids must be a contiguous CPU int64 [n] tensor");
    TORCH_CHECK(rows.device().is_cpu() && rows.is_contiguous() && rows.scalar_type() == torch::kUInt8 && rows.dim() == 2 &&
                    rows.size(0) == ids.size(0) && rows.size(1) == row,
                fn, ": rows must be a contiguous CPU uint8 [n, ", row, "] tensor");
}
}  // namespace

// A row cache of `capacity` rows of `row` bytes, `ways` per set.
std::shared_ptr<PleCache> ple_cache_new(int64_t capacity, int64_t row, int64_t ways) {
    TORCH_CHECK(capacity >= 1 && row >= 1 && row <= 4096 && ways >= 1 && ways <= 64, "ple_cache_new: capacity >= 1, row in [1, 4096], ways in [1, 64]");
    return std::make_shared<PleCache>(capacity, row, ways);
}

// out[i] = the cached row of ids[i]; returns the miss mask uint8 [n] (1: not cached, out[i] untouched). GIL released.
torch::Tensor ple_cache_lookup_t(const std::shared_ptr<PleCache>& h, torch::Tensor ids, torch::Tensor out) {
    check_rows(ids, out, h->row, "ple_cache_lookup");
    auto miss = torch::empty({ids.size(0)}, torch::dtype(torch::kUInt8));
    {
        pybind11::gil_scoped_release nogil;
        ple_cache_lookup(h->c, ids.data_ptr<int64_t>(), ids.size(0), out.data_ptr<uint8_t>(), miss.data_ptr<uint8_t>());
    }
    return miss;
}

// Stores rows[i] as the row of ids[i]. GIL released.
void ple_cache_insert_t(const std::shared_ptr<PleCache>& h, torch::Tensor ids, torch::Tensor rows) {
    check_rows(ids, rows, h->row, "ple_cache_insert");
    pybind11::gil_scoped_release nogil;
    ple_cache_insert(h->c, ids.data_ptr<int64_t>(), ids.size(0), rows.data_ptr<uint8_t>());
}

// (hits, misses, rows held, capacity in rows) since the creation or the last reset.
std::vector<int64_t> ple_cache_stats(const std::shared_ptr<PleCache>& h, bool reset) {
    std::vector<int64_t> out(4);
    ple_cache_counters(h->c, out.data());
    if (reset) ple_cache_reset_counters(h->c);
    return out;
}

// ------------------------------------------------------------------------------------------------------------------
// hc_prefill.cu: the tail of the gated residual over the rows of a prefill chunk.
cudaError_t launch_hc_mix_rows(const void*, int, const void*, void*, int, int, cudaStream_t);

// y [T, H] = bf16((((p0 + p1) + p2) + p3) / 4), p_s = bf16(bf16(sigmoid(u_s)) * n_s): u bf16 [T, 4 H] (unit-stride rows,
// any row stride), n bf16 [T, 4 H] contiguous, y bf16 [T, H] contiguous.
void hc_mix_rows(torch::Tensor u, torch::Tensor n, torch::Tensor y) {
    QSA_CHECK_TENSOR(n, torch::kBFloat16);
    QSA_CHECK_TENSOR(y, torch::kBFloat16);
    TORCH_CHECK(u.is_cuda() && u.scalar_type() == torch::kBFloat16 && u.dim() == 2 && u.stride(1) == 1, "u: bf16 [T, 4 H] with unit-stride rows");
    const int64_t T = y.size(0), H = y.size(1);
    TORCH_CHECK(y.dim() == 2 && u.size(0) == T && u.size(1) == 4 * H && n.numel() == T * 4 * H && H % 8 == 0 && u.stride(0) % 8 == 0,
                "hc_mix_rows: u, n [T, 4 H], y [T, H], H % 8 == 0");
    QSA_CHECK_LAUNCH(launch_hc_mix_rows(u.data_ptr(), (int)u.stride(0), n.data_ptr(), y.data_ptr(), T, H, stream()));
}

void bind_qsa(pybind11::module_& m) {
    namespace py = pybind11;
    m.def("hc_mix_rows", &hc_mix_rows, "gated residual tail over prefill rows: sigmoid, product, ordered stream mean", py::arg("u"), py::arg("n"),
          py::arg("y"));
    py::class_<PleCache, std::shared_ptr<PleCache>>(m, "PleCache");
    m.def("ple_cache_new", &ple_cache_new, "PLE row cache: set-associative, LRU per set, capacity in rows", py::arg("capacity"), py::arg("row") = 160,
          py::arg("ways") = 8);
    m.def("ple_cache_lookup", &ple_cache_lookup_t, "PLE row cache lookup: the cached rows into out, returns the miss mask", py::arg("cache"),
          py::arg("ids"), py::arg("out"));
    m.def("ple_cache_insert", &ple_cache_insert_t, "PLE row cache insert", py::arg("cache"), py::arg("ids"), py::arg("rows"));
    m.def("ple_cache_stats", &ple_cache_stats, "PLE row cache (hits, misses, rows held, capacity)", py::arg("cache"), py::arg("reset") = false);
    m.def("ple_gather", &ple_gather, "PLE n-gram table rows: pread of n rows at byte offsets by a thread pool (GIL released)", py::arg("fd"),
          py::arg("offsets"), py::arg("out"), py::arg("direct") = false, py::arg("threads") = 32);
    m.def("qsa_score_rows", &qsa_score_rows, "QSA block scores of all rows of a prefill chunk (bit-identical to qsa_score per row)", py::arg("q_idx"),
          py::arg("ikb"), py::arg("pos_t"), py::arg("scores"));
    m.def("qsa_union_cap", &qsa_union_cap, "the entries per group of the qsa_union output");
    m.def("qsa_union", &qsa_union, "QSA prefill: the union of the selected and tail blocks of each group of 4 rows", py::arg("sel"), py::arg("nsel"),
          py::arg("pos_t"), py::arg("ul"), py::arg("uc"), py::arg("nblocks"));
    m.def("qsa_union_attn", &qsa_union_attn, "QSA prefill attention over the group unions (fp8 KV, output gate)", py::arg("q"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("pos_t"), py::arg("ul"), py::arg("uc"), py::arg("gate"), py::arg("out"), py::arg("scale"));
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
