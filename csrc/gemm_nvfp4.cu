// gemm_nvfp4.cu -- prefill GEMM: NVFP4 x NVFP4 block-scaled, with CUTLASS 4.8 SM120 collectives (PLAN.md 4.4 item 2).
//
//   D[M, N] (bf16) = alpha * (A[M, K] . B[N, K]^T) (+ C[M, N])
//   A: activations, packed e2m1 [M, K/2] (row-major, element 2i in the low nibble) + e4m3 block scales (16)
//   B: weights,     packed e2m1 [N, K/2] (K contiguous)                            + e4m3 block scales (16)
//   Both scale-factor tensors use CUTLASS's 128x4 atom layout (Sm1xxBlockScaledConfig<16>):
//     offset(r, kb) = ((r / 128) * ceil(KB / 4) + kb / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4
//   with rows padded to 128 and KB = K / 16 padded to 4 (padding zero-filled). See quant_nvfp4 below.
//   alpha = input_scale * weight_scale_2 (the two NVFP4 global scales).
// The tile configs come from the phase-0 CUTLASS sweep (docs/history/baseline.md section 2). The raster swizzle of the
// persistent scheduler is 8. Without it, the 5120 x 17408 down projection loses 60% to L2 thrash at M = 4096.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace g4 {

template <int TM, int TN, int TK>
struct Cfg {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, ElementC, cutlass::layout::RowMajor, 8, ElementD, cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, cutlass::layout::RowMajor, 32, ElementB,
        cutlass::layout::ColumnMajor, 32, float, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// ---- the SwiGLU of the MLP, fused into the epilogue of the up GEMM -------------------------------------------------------
// D = NVFP4(silu(C) * alpha * acc), with an e4m3 scale per 16 outputs, in the CUTLASS scale layout that the down GEMM
// reads as its A scales. C is the bf16 output of the gate GEMM. Block scale = e4m3(amax * nc / 6), and the values are
// values * nc / scale, with nc = 1 / in_scale of the down projection (Sm120BlockScaleFactorRowStore), the same as
// quant_nvfp4. The traits come from LinCombBlockScaleFactor (NVFP4 output, bf16 source). The callbacks are the tree
// below.
struct SwigluNvfp4 : cutlass::epilogue::fusion::LinCombBlockScaleFactor<16, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
                                                                       cutlass::layout::RowMajor, cutlass::bfloat16_t> {};

}  // namespace g4

namespace cutlass::epilogue::fusion {
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>, g4::SwigluNvfp4,
                       CtaTileShapeMNK, EpilogueTile>
    : Sm90EVT<Sm120BlockScaleFactorRowStore<16, EpilogueTile, CtaTileShapeMNK, FragmentSize, float_e2m1_t, float, float_ue4m3_t,
                                            FloatRoundStyle::round_to_nearest>,
              Sm90EVT<Sm90Compute<multiplies, float, float, FloatRoundStyle::round_to_nearest>,
                      Sm90EVT<Sm90Compute<epilogue::thread::SiLu, float, float, FloatRoundStyle::round_to_nearest>, Sm90SrcFetch<bfloat16_t>>,
                      Sm90EVT<Sm90Compute<multiplies, float, float, FloatRoundStyle::round_to_nearest>, Sm90ScalarBroadcast<float>,
                              Sm90AccFetch>>> {
    using Impl =
        Sm90EVT<Sm120BlockScaleFactorRowStore<16, EpilogueTile, CtaTileShapeMNK, FragmentSize, float_e2m1_t, float, float_ue4m3_t,
                                              FloatRoundStyle::round_to_nearest>,
                Sm90EVT<Sm90Compute<multiplies, float, float, FloatRoundStyle::round_to_nearest>,
                        Sm90EVT<Sm90Compute<epilogue::thread::SiLu, float, float, FloatRoundStyle::round_to_nearest>, Sm90SrcFetch<bfloat16_t>>,
                        Sm90EVT<Sm90Compute<multiplies, float, float, FloatRoundStyle::round_to_nearest>, Sm90ScalarBroadcast<float>,
                                Sm90AccFetch>>>;
    using Operation = g4::SwigluNvfp4;

    struct Arguments {
        float alpha = 1.f;
        float_ue4m3_t* block_scale_factor_ptr = nullptr;
        float const* norm_constant_ptr = nullptr;
        using StrideNormConst = Stride<_0, _0, int64_t>;
        StrideNormConst dNormConst = {_0{}, _0{}, 0};

        operator typename Impl::Arguments() const {
            return {
                {
                    // silu(C) * (alpha * acc)
                    {{}, {}},                                                // silu(C): source, silu
                    {{{alpha}, {nullptr}, {}}, {}, {}},                     // alpha * acc
                    {}                                                       // multiplies
                },
                {block_scale_factor_ptr, norm_constant_ptr, dNormConst}     // scale factors, norm constant
            };
        }
    };

    using Impl::Impl;
};
}  // namespace cutlass::epilogue::fusion

namespace g4 {

template <int TM, int TN, int TK>
struct CfgSwiglu {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::float_e2m1_t;
    using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, ElementC, cutlass::layout::RowMajor, 8, ElementD, cutlass::layout::RowMajor, 32,
        cutlass::epilogue::collective::EpilogueScheduleAuto, SwigluNvfp4>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, cutlass::layout::RowMajor, 32, ElementB,
        cutlass::layout::ColumnMajor, 32, float, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class C>
cudaError_t run_swiglu(const void* a, const void* sfa, const void* b, const void* sfb, float alpha, const void* c, void* d, void* sfd,
                       const float* norm_const, int M, int N, int K, void* workspace, size_t ws_bytes, size_t* ws_needed, cudaStream_t st) {
    using Gemm = typename C::Gemm;
    using SfCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    auto lSFA = SfCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto lSFB = SfCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K, 1},
                                  {(const typename C::ElementA::DataType*)a, sA, (const typename C::ElementB::DataType*)b, sB,
                                   (const typename C::ElementA::ScaleFactorType*)sfa, lSFA, (const typename C::ElementB::ScaleFactorType*)sfb, lSFB},
                                  {{}, (const typename C::ElementC*)c, sC, (typename C::ElementD*)d, sD}};
    args.epilogue.thread.alpha = alpha;
    args.epilogue.thread.block_scale_factor_ptr = (cutlass::float_ue4m3_t*)sfd;
    args.epilogue.thread.norm_constant_ptr = norm_const;
    args.scheduler.max_swizzle_size = 8;
    static int sm_count = 0, device = -1;
    if (sm_count == 0) {
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    args.hw_info.device_id = device;
    args.hw_info.sm_count = sm_count;
    const size_t need = Gemm::get_workspace_size(args);
    if (ws_needed) {
        *ws_needed = need;
        return cudaSuccess;
    }
    if (need > ws_bytes) return cudaErrorMemoryAllocation;
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    if (gemm.initialize(args, workspace, st) != cutlass::Status::kSuccess) return cudaErrorUnknown;
    if (gemm.run(st) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    return cudaGetLastError();
}

template <class C>
cudaError_t run(const void* a, const void* sfa, const void* b, const void* sfb, float alpha, const void* c, void* d, int M, int N, int K,
                void* workspace, size_t ws_bytes, size_t* ws_needed, cudaStream_t st) {
    using Gemm = typename C::Gemm;
    using SfCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    auto lSFA = SfCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto lSFB = SfCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K, 1},
                                  {(const typename C::ElementA::DataType*)a, sA, (const typename C::ElementB::DataType*)b, sB,
                                   (const typename C::ElementA::ScaleFactorType*)sfa, lSFA, (const typename C::ElementB::ScaleFactorType*)sfb, lSFB},
                                  {{alpha, c ? 1.0f : 0.0f}, (const typename C::ElementC*)(c ? c : d), sC, (typename C::ElementD*)d, sD}};
    args.scheduler.max_swizzle_size = 8;
    // Without an explicit SM count CUTLASS calls cudaGetDeviceProperties on every initialize (milliseconds).
    static int sm_count = 0, device = -1;
    if (sm_count == 0) {
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    args.hw_info.device_id = device;
    args.hw_info.sm_count = sm_count;
    const size_t need = Gemm::get_workspace_size(args);
    if (ws_needed) {  // size query only: do not run
        *ws_needed = need;
        return cudaSuccess;
    }
    if (need > ws_bytes) return cudaErrorMemoryAllocation;
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    if (gemm.initialize(args, workspace, st) != cutlass::Status::kSuccess) return cudaErrorUnknown;
    if (gemm.run(st) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
    return cudaGetLastError();
}

// ---- activation quantization: bf16 [M, K] -> packed e2m1 [M, K/2] + swizzled e4m3 scales ----
__device__ __forceinline__ uint32_t e2m1_code(float a) {  // |a| -> 3-bit magnitude code, RN-even, saturating
    // grid 0, .5, 1, 1.5, 2, 3, 4, 6 ; midpoints .25 .75 1.25 1.75 2.5 3.5 5 ; ties to even mantissa
    uint32_t c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
    return c;
}

__device__ __forceinline__ size_t sf_offset(int r, int kb, int kb4) {
    return ((size_t)(r >> 7) * kb4 + (kb >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (kb & 3);
}

// one thread per 16-element block
__global__ void k_quant(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, int K, float inv_in_scale,
                        float in_scale) {
    const int KB = K / 16, kb4 = (KB + 3) / 4;
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * KB) return;
    const int r = idx / KB, kb = idx % KB;
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * K + kb * 16);
    const uint4 v0 = src[0], v1 = src[1];
    const uint32_t u[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float f[16], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f[2 * i] = __uint_as_float(u[i] << 16);
        f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        amax = fmaxf(amax, fmaxf(fabsf(f[2 * i]), fabsf(f[2 * i + 1])));
    }
    const __nv_fp8_storage_t sfb = __nv_cvt_float_to_fp8(amax / 6.f * inv_in_scale, __NV_SATFINITE, __NV_E4M3);
    __half_raw hr = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
    const float sfv = __half2float(*reinterpret_cast<__half*>(&hr));
    const float os = sfv != 0.f ? 1.f / (sfv * in_scale) : 0.f;
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float s = f[i] * os;
        const uint32_t code = e2m1_code(fabsf(s)) | (s < 0.f ? 8u : 0u);
        packed[i >> 3] |= code << (4 * (i & 7));
    }
    *reinterpret_cast<uint2*>(q + (size_t)r * (K / 2) + kb * 8) = make_uint2(packed[0], packed[1]);
    sf[sf_offset(r, kb, kb4)] = sfb;
}

// row-major e4m3 [R, KB] -> swizzled layout (weights, at load time)
__global__ void k_swizzle_sf(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int R, int KB) {
    const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * KB) return;
    const int r = idx / KB, kb = idx % KB;
    dst[sf_offset(r, kb, (KB + 3) / 4)] = src[idx];
}

}  // namespace g4

size_t nvfp4_sf_bytes(int R, int K) { return (size_t)((R + 127) / 128) * 128 * (((K / 16) + 3) / 4) * 4; }

cudaError_t launch_nvfp4_quant(const void* x, void* q, void* sf, int M, int K, float in_scale, cudaStream_t st) {
    if (K % 16) return cudaErrorInvalidValue;
    cudaMemsetAsync(sf, 0, nvfp4_sf_bytes(M, K), st);
    const size_t n = (size_t)M * (K / 16);
    g4::k_quant<<<(n + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)x, (uint8_t*)q, (uint8_t*)sf, M, K, 1.f / in_scale, in_scale);
    return cudaGetLastError();
}

cudaError_t launch_nvfp4_swizzle_sf(const void* src, void* dst, int R, int K, cudaStream_t st) {
    cudaMemsetAsync(dst, 0, nvfp4_sf_bytes(R, K), st);
    const size_t n = (size_t)R * (K / 16);
    g4::k_swizzle_sf<<<(n + 255) / 256, 256, 0, st>>>((const uint8_t*)src, (uint8_t*)dst, R, K / 16);
    return cudaGetLastError();
}

// The up GEMM of the MLP with the SwiGLU fused: d = NVFP4(silu(c) * alpha * a.b^T) + its swizzled e4m3 scales (sfd, zeroed
// here), c = the gate GEMM's bf16 output [M, N], norm_const = device float 1 / in_scale (down projection).
cudaError_t launch_nvfp4_gemm_swiglu(const void* a, const void* sfa, const void* b, const void* sfb, float alpha, const void* c, void* d, void* sfd,
                                     const float* norm_const, int M, int N, int K, int tile, void* ws, size_t ws_bytes, size_t* ws_needed,
                                     cudaStream_t st) {
    if (!ws_needed) cudaMemsetAsync(sfd, 0, nvfp4_sf_bytes(M, N), st);
    if (tile == 0)
        return g4::run_swiglu<g4::CfgSwiglu<256, 128, 128>>(a, sfa, b, sfb, alpha, c, d, sfd, norm_const, M, N, K, ws, ws_bytes, ws_needed, st);
    return g4::run_swiglu<g4::CfgSwiglu<128, 128, 256>>(a, sfa, b, sfb, alpha, c, d, sfd, norm_const, M, N, K, ws, ws_bytes, ws_needed, st);
}

// tile: 0 = 256x128x128 cooperative (large M), 1 = 128x128x256 cooperative (small M)
cudaError_t launch_nvfp4_gemm(const void* a, const void* sfa, const void* b, const void* sfb, float alpha, const void* c, void* d, int M, int N, int K,
                              int tile, void* ws, size_t ws_bytes, size_t* ws_needed, cudaStream_t st) {
    if (tile == 0) return g4::run<g4::Cfg<256, 128, 128>>(a, sfa, b, sfb, alpha, c, d, M, N, K, ws, ws_bytes, ws_needed, st);
    return g4::run<g4::Cfg<128, 128, 256>>(a, sfa, b, sfb, alpha, c, d, M, N, K, ws, ws_bytes, ws_needed, st);
}
