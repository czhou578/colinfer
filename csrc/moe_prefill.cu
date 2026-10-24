// moe_prefill.cu -- prefill MoE for qwen4_exp (docs/history/qwen4exp_plan.md section 3, "MoE prefill"): the (token, slot) pairs
// sorted by expert run through two grouped NVFP4 GEMMs, one group per expert, with the CUTLASS 4.8 SM120 block-scaled
// ptr-array kernel. Tests: tests/test_moe_prefill.py. engine/model/moe_kernels.py (KernelMoE.prefill) chains the steps:
//
//   plan (torch, no host sync)  order = the pairs sorted by expert (stable), off[e] = the first sorted row of expert e,
//                               sfo[e] = the first scale row of group e (each group padded to 128 rows)
//   moe_gather_quant            A [P, K/2] e2m1 + SFA: sorted row r = x[order[r] / 10] quantized with in_scale[e, 0];
//                               the scales of group e in the CUTLASS 128x4 layout at sfo[e] * K/16 bytes
//   moe_grouped_gemm (gate|up)  D1 bf16 [P, 2I] = alpha_e * (A_e . w_gu[e]^T), alpha_e = in_scale[e, 0] * gs[e, 0]
//   moe_swiglu_quant            A2 [P, I/2] + SFA2 = NVFP4(silu(gate) * up) with in_scale[e, 2]
//   moe_grouped_gemm (down)     D2 bf16 [P, K] = alpha_e * (A2_e . w_dn[e]^T), alpha_e = in_scale[e, 2] * gs[e, 2]
//   moe_combine_prefill         y[t] = bf16(sum_{k = 0..9} w[t, k] * D2[inv[10 t + k]] + sg[t] * shared[t]), fp32, slot order
//
// The group arguments (problem shapes, pointers, strides, scale layouts, alphas) are written on the device by k_group_args,
// one thread per expert, from off and sfo. The kernel gets no host problem shapes, so its grid is the SM count and
// nothing waits for the host. A group with 0 rows has 0 tiles: the scheduler skips it. The A and D rows of a group end at
// its row count (TMA fills the rows past it with zeros and does not store them). The scales of the padding rows of a group
// are not written: the mma of a row reads only that row's scales, and the padding rows are never stored.
//
// The raster swizzle of the persistent scheduler is 8, as in gemm_nvfp4.cu. Without it, the grouped GEMM runs at a third of
// the speed: 10.5 ms instead of 4.6 ms for the gate | up GEMM of a 2048-token chunk (512 groups of 40 rows), and 3.5 ms instead
// of 1.1 ms for one group of 20480 rows (the dense GEMM's time). The pingpong schedule and a 128 x 128 x 128 tile (4 stages
// instead of 2) change the time by less than 5%.
//
// The gate | up GEMM uses the 128 x 128 x 256 tile, the down GEMM 128 x 128 x 128 (its K = I = 640 = 5 x 128). Requires
// K % 256 == 0 for the gate | up GEMM, I % 128 == 0, and the weight scales in the swizzled layout of gemm_nvfp4.cu.
// The GEMM gate equals the up global scale per expert (the checkpoint stores it so): one alpha per group.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace mp {

constexpr int TOPK = 10;

template <int TM, int TN, int TK>
struct GCfg {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementC = void;
    using ElementD = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using Fusion = cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, ElementC, LayoutD*, 8, ElementD, LayoutD*, 8, cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
        Fusion>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA*, 32, ElementB, LayoutB*, 32, float, TileShape,
        ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    using PS = typename ProblemShape::UnderlyingProblemShape;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutSFA = typename Main::InternalLayoutSFA;
    using LayoutSFB = typename Main::InternalLayoutSFB;
    using SfCfg = typename Main::Sm1xxBlkScaledConfig;
    using DA = typename ElementA::DataType;
    using DB = typename ElementB::DataType;
    using SF = typename ElementA::ScaleFactorType;
};

// The per-group argument arrays, carved from one device buffer.
template <class C>
struct Arrays {
    typename C::PS* shapes;
    const typename C::DA** a;
    const typename C::DB** b;
    const typename C::SF** sfa;
    const typename C::SF** sfb;
    typename C::ElementD** d;
    typename C::StrideA* sa;
    typename C::StrideB* sb;
    typename C::StrideD* sd;
    typename C::LayoutSFA* lsfa;
    typename C::LayoutSFB* lsfb;
    float* alpha;
    const float** alpha_ptr;
};

inline size_t align16(size_t n) { return (n + 15) & ~size_t(15); }

template <class C>
size_t carve(Arrays<C>* arr, uint8_t* base, int E) {
    size_t o = 0;
    auto take = [&](auto*& p, size_t bytes) {
        p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(base ? base + o : nullptr);
        o += align16(bytes);
    };
    take(arr->shapes, E * sizeof(typename C::PS));
    take(arr->a, E * sizeof(void*));
    take(arr->b, E * sizeof(void*));
    take(arr->sfa, E * sizeof(void*));
    take(arr->sfb, E * sizeof(void*));
    take(arr->d, E * sizeof(void*));
    take(arr->sa, E * sizeof(typename C::StrideA));
    take(arr->sb, E * sizeof(typename C::StrideB));
    take(arr->sd, E * sizeof(typename C::StrideD));
    take(arr->lsfa, E * sizeof(typename C::LayoutSFA));
    take(arr->lsfb, E * sizeof(typename C::LayoutSFB));
    take(arr->alpha, E * sizeof(float));
    take(arr->alpha_ptr, E * sizeof(void*));
    return o;
}

// One thread per expert: the arguments of group e. A, SFA, D: the gathered operands (rows off[e].., scale rows sfo[e]..);
// W [E, N, K/2], SFW [E, N * K/16] swizzled per expert.
template <class C>
__global__ void k_group_args(Arrays<C> arr, int E, int N, int K, const int* __restrict__ off, const int* __restrict__ sfo, const uint8_t* A,
                             const uint8_t* SFA, const uint8_t* W, const uint8_t* SFW, __nv_bfloat16* D, const float* __restrict__ in_scale,
                             const float* __restrict__ gs, int col) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int m = off[e + 1] - off[e];
    arr.shapes[e] = typename C::PS(m, N, K);
    arr.a[e] = reinterpret_cast<const typename C::DA*>(A + (size_t)off[e] * (K / 2));
    arr.sfa[e] = reinterpret_cast<const typename C::SF*>(SFA + (size_t)sfo[e] * (K / 16));
    arr.b[e] = reinterpret_cast<const typename C::DB*>(W + (size_t)e * N * (K / 2));
    arr.sfb[e] = reinterpret_cast<const typename C::SF*>(SFW + (size_t)e * N * (K / 16));
    arr.d[e] = reinterpret_cast<typename C::ElementD*>(D + (size_t)off[e] * N);
    arr.sa[e] = cutlass::make_cute_packed_stride(typename C::StrideA{}, {m, K, 1});
    arr.sb[e] = cutlass::make_cute_packed_stride(typename C::StrideB{}, {N, K, 1});
    arr.sd[e] = cutlass::make_cute_packed_stride(typename C::StrideD{}, {m, N, 1});
    arr.lsfa[e] = C::SfCfg::tile_atom_to_shape_SFA(make_shape(m, N, K, 1));
    arr.lsfb[e] = C::SfCfg::tile_atom_to_shape_SFB(make_shape(m, N, K, 1));
    arr.alpha[e] = in_scale[3 * e + col] * gs[3 * e + col];
    arr.alpha_ptr[e] = arr.alpha + e;
}

template <class C>
cudaError_t run_grouped(int E, int N, int K, const int* off, const int* sfo, const void* A, const void* SFA, const void* W, const void* SFW, void* D,
                        const float* in_scale, const float* gs, int col, void* ws, size_t ws_bytes, size_t* ws_needed, cudaStream_t st) {
    using Gemm = typename C::Gemm;
    Arrays<C> arr{};
    const size_t arr_bytes = carve<C>(&arr, nullptr, E);
    typename Gemm::Arguments args;
    static int sm_count = 0, device = -1;
    if (sm_count == 0) {
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    auto make_args = [&]() {
        decltype(args.epilogue.thread) fusion{};
        fusion.alpha_ptr_array = arr.alpha_ptr;
        fusion.dAlpha = {_0{}, _0{}, 1};
        cutlass::KernelHardwareInfo hw;
        hw.device_id = device;
        hw.sm_count = sm_count;
        typename Gemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGrouped,
                                   {E, arr.shapes, nullptr},
                                   {arr.a, arr.sa, arr.b, arr.sb, arr.sfa, arr.lsfa, arr.sfb, arr.lsfb},
                                   {fusion, nullptr, nullptr, arr.d, arr.sd},
                                   hw};
        a.scheduler.max_swizzle_size = 8;
        return a;
    };
    args = make_args();
    const size_t gemm_ws = Gemm::get_workspace_size(args);
    if (ws_needed) {  // size query only
        *ws_needed = align16(arr_bytes) + gemm_ws;
        return cudaSuccess;
    }
    if (align16(arr_bytes) + gemm_ws > ws_bytes) return cudaErrorMemoryAllocation;
    carve<C>(&arr, static_cast<uint8_t*>(ws), E);
    args = make_args();
    k_group_args<C><<<(E + 127) / 128, 128, 0, st>>>(arr, E, N, K, off, sfo, (const uint8_t*)A, (const uint8_t*)SFA, (const uint8_t*)W,
                                                     (const uint8_t*)SFW, (__nv_bfloat16*)D, in_scale, gs, col);
    if (cudaError_t e = cudaGetLastError()) return e;
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    if (gemm.initialize(args, static_cast<uint8_t*>(ws) + align16(arr_bytes), st) != cutlass::Status::kSuccess) return cudaErrorUnknown;
    if (gemm.run(st) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    return cudaGetLastError();
}

using GateUp = GCfg<128, 128, 256>;
using Down = GCfg<128, 128, 128>;

// ------------------------------------------------------------------------------------------------------------ quantization
// 16 fp32 values -> 8 bytes of e2m1 + the e4m3 block scale, as k_quant of gemm_nvfp4.cu (bit for bit on bf16 inputs).
__device__ __forceinline__ uint8_t quant16(const float (&f)[16], float inv_in_scale, float in_scale, uint2& packed_out) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
    const __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(amax / 6.f * inv_in_scale, __NV_SATFINITE, __NV_E4M3);
    __half_raw hr = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
    const float sfv = __half2float(*reinterpret_cast<__half*>(&hr));
    const float os = sfv != 0.f ? 1.f / (sfv * in_scale) : 0.f;
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s = f[i] * os;
        const uint32_t code = cc::e2m1_code(fabsf(s)) | (s < 0.f ? 8u : 0u);
        packed[i >> 3] |= code << (4 * (i & 7));
    }
    packed_out = make_uint2(packed[0], packed[1]);
    return sfb;
}

// One thread per (sorted row r, 16-element block kb): A[r] = NVFP4(x[order[r] / 10]) with the input scale of its expert.
__global__ void k_gather_quant(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const int* __restrict__ order,
                               const int* __restrict__ off, const int* __restrict__ sfo, const float* __restrict__ in_scale, int col, int P, int K,
                               uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
    const int KB = K / 16;
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)P * KB) return;
    const int r = idx / KB, kb = idx % KB, pair = order[r], e = ids[pair], t = pair / TOPK;
    const float s_in = in_scale[3 * e + col];
    float f[16];
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * K + kb * 16);
    cc::bf16x8_to_float(src[0], f);
    cc::bf16x8_to_float(src[1], f + 8);
    uint2 packed;
    const uint8_t s = quant16(f, 1.f / s_in, s_in, packed);
    *reinterpret_cast<uint2*>(q + (size_t)r * (K / 2) + kb * 8) = packed;
    sf[(size_t)sfo[e] * KB + cc::sf_offset(r - off[e], kb, KB / 4)] = s;
}

// One thread per (sorted row r, 16-element block kb of I): A2[r] = NVFP4(silu(g) * u), g = d1[r, :I], u = d1[r, I:], fp32
// (the precise expf), quantized with the down input scale of its expert.
__global__ void k_swiglu_quant(const __nv_bfloat16* __restrict__ d1, const int* __restrict__ ids, const int* __restrict__ order,
                               const int* __restrict__ off, const int* __restrict__ sfo, const float* __restrict__ in_scale, int P, int I,
                               uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
    const int KB = I / 16;
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)P * KB) return;
    const int r = idx / KB, kb = idx % KB, e = ids[order[r]];
    const float s_in = in_scale[3 * e + 2];
    float g[16], u[16], f[16];
    const uint4* gp = reinterpret_cast<const uint4*>(d1 + (size_t)r * 2 * I + kb * 16);
    const uint4* up = reinterpret_cast<const uint4*>(d1 + (size_t)r * 2 * I + I + kb * 16);
    cc::bf16x8_to_float(gp[0], g);
    cc::bf16x8_to_float(gp[1], g + 8);
    cc::bf16x8_to_float(up[0], u);
    cc::bf16x8_to_float(up[1], u + 8);
#pragma unroll
    for (int i = 0; i < 16; ++i) f[i] = g[i] / (1.f + expf(-g[i])) * u[i];
    uint2 packed;
    const uint8_t s = quant16(f, 1.f / s_in, s_in, packed);
    *reinterpret_cast<uint2*>(q + (size_t)r * (I / 2) + kb * 8) = packed;
    sf[(size_t)sfo[e] * KB + cc::sf_offset(r - off[e], kb, KB / 4)] = s;
}

// One thread per 8 columns of a token: y = bf16(w_0 d_0 + ... + w_9 d_9 + sg * shared), fp32 in this order.
__global__ void k_combine_prefill(const __nv_bfloat16* __restrict__ d2, const int* __restrict__ inv, const float* __restrict__ w,
                                  const float* __restrict__ sg, const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y, int T, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x, per = K / 8;
    if (idx >= T * per) return;
    const int t = idx / per, n = (idx % per) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, f[8];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const float wk = w[t * TOPK + k];
        cc::bf16x8_to_float(*reinterpret_cast<const uint4*>(d2 + (size_t)inv[t * TOPK + k] * K + n), f);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += wk * f[i];
    }
    if (shared) {
        cc::bf16x8_to_float(*reinterpret_cast<const uint4*>(shared + (size_t)t * K + n), f);
        const float s = sg[t];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += s * f[i];
    }
    __nv_bfloat162 b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    *reinterpret_cast<uint4*>(y + (size_t)t * K + n) = *reinterpret_cast<const uint4*>(b);
}

}  // namespace mp

// which: 0 = gate | up (tile 128 x 128 x 256, alpha = in_scale[e, 0] * gs[e, 0]), 1 = down (128 x 128 x 128, column 2).
// ws_needed != nullptr: only the workspace size.
cudaError_t launch_moe_grouped_gemm(int which, int E, int N, int K, const int* off, const int* sfo, const void* A, const void* SFA, const void* W,
                                    const void* SFW, void* D, const float* in_scale, const float* gs, void* ws, size_t ws_bytes, size_t* ws_needed,
                                    cudaStream_t st) {
    if (which == 0)
        return mp::run_grouped<mp::GateUp>(E, N, K, off, sfo, A, SFA, W, SFW, D, in_scale, gs, 0, ws, ws_bytes, ws_needed, st);
    return mp::run_grouped<mp::Down>(E, N, K, off, sfo, A, SFA, W, SFW, D, in_scale, gs, 2, ws, ws_bytes, ws_needed, st);
}

cudaError_t launch_moe_gather_quant(const void* x, const int* ids, const int* order, const int* off, const int* sfo, const float* in_scale, int P,
                                    int K, void* q, void* sf, cudaStream_t st) {
    const size_t n = (size_t)P * (K / 16);
    if (n == 0) return cudaSuccess;
    mp::k_gather_quant<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, ids, order, off, sfo, in_scale, 0, P, K, (uint8_t*)q, (uint8_t*)sf);
    return cudaGetLastError();
}

cudaError_t launch_moe_swiglu_quant(const void* d1, const int* ids, const int* order, const int* off, const int* sfo, const float* in_scale, int P,
                                    int I, void* q, void* sf, cudaStream_t st) {
    const size_t n = (size_t)P * (I / 16);
    if (n == 0) return cudaSuccess;
    mp::k_swiglu_quant<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)d1, ids, order, off, sfo, in_scale, P, I, (uint8_t*)q, (uint8_t*)sf);
    return cudaGetLastError();
}

cudaError_t launch_moe_combine_prefill(const void* d2, const int* inv, const float* w, const float* sg, const void* shared, void* y, int T, int K,
                                       cudaStream_t st) {
    const int n = T * (K / 8);
    if (n == 0) return cudaSuccess;
    mp::k_combine_prefill<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)d2, inv, w, sg, (const __nv_bfloat16*)shared, (__nv_bfloat16*)y,
                                                           T, K);
    return cudaGetLastError();
}
