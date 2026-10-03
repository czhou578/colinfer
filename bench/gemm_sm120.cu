// gemm_sm120.cu -- CUTLASS SM120 GEMM ceiling probe for DGX Spark (GB10, sm_121a).
//
// Measures one CUTLASS kernel configuration (chosen at compile time) at a given M x N x K,
// with D = A (M x K, K-major) * B^T (N x K, K-major), bf16 output, fp32 accumulate.
// Lineage: examples/79_blackwell_geforce_gemm/79b (NVFP4 block-scaled) and the
// test/unit/gemm/device/sm120_tensorop_gemm FP8 tests.
//
// Compile-time config (see bench/Makefile):
//   -DKIND_NVFP4 | -DKIND_FP8         nv_float4 (e2m1 + ue4m3 scale per 16) or e4m3 x e4m3
//   -DTILE_M=.. -DTILE_N=.. -DTILE_K=..
//   -DSCHED_PINGPONG | -DSCHED_COOP
//
// Usage: gemm_sm120_<cfg> --m M --n N --k K [--iters 20] [--warmup 3] [--verify 1] [--pairs 65536]
//                         [--swizzle S] [--raster h|m|n]     tile-scheduler raster swizzle (L2 reuse) and order
// Prints one line:  RESULT kind=.. tile=.. sched=.. m= n= k= ms= best_ms= tflops= gbps= verify=ok|FAIL maxerr=
// or               RESULT ... status=<reason>     when can_implement() rejects the problem.
//
// Verification: a sampled set of (m, n) output elements is recomputed by an independent naive
// kernel that dequantizes A, B and the scale factors itself. This exists because an FP4 kernel
// built for the wrong ISA has produced wrong answers on sm_121 with no CUDA error (PLAN.md 1.x).

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#error "Needs CUDA >= 12.9 and -arch=sm_120a / sm_121a"
#endif

using namespace cute;

#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif

#if defined(SCHED_PINGPONG)
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#define SCHED_NAME "pingpong"
#elif defined(SCHED_COOP)
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#define SCHED_NAME "cooperative"
#else
#error "define SCHED_PINGPONG or SCHED_COOP"
#endif

using TileShape    = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<_1, _1, _1>;  // no multicast on this arch
using ArchTag      = cutlass::arch::Sm120;
using ElementC     = cutlass::bfloat16_t;
using ElementD     = cutlass::bfloat16_t;
using LayoutC      = cutlass::layout::RowMajor;
using LayoutD      = cutlass::layout::RowMajor;
constexpr int AlignC = 8, AlignD = 8;

#if defined(KIND_NVFP4)
#define KIND_NAME "nvfp4"
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignA = 32, AlignB = 32;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
constexpr double BytesPerElem = 0.5;
#elif defined(KIND_FP8)
#define KIND_NAME "fp8"
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
constexpr int AlignA = 16, AlignB = 16;
using OperatorClass = cutlass::arch::OpClassTensorOp;
constexpr double BytesPerElem = 1.0;
#else
#error "define KIND_NVFP4 or KIND_FP8"
#endif
using LayoutA = cutlass::layout::RowMajor;     // A is M x K, K contiguous
using LayoutB = cutlass::layout::ColumnMajor;  // B is N x K, K contiguous  (TN: the only SM120 layout)

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    Schedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA    = typename Gemm::GemmKernel::StrideA;
using StrideB    = typename Gemm::GemmKernel::StrideB;
using StrideC    = typename Gemm::GemmKernel::StrideC;
using StrideD    = typename Gemm::GemmKernel::StrideD;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(2);                                                                                       \
        }                                                                                                  \
    } while (0)

// ---- deterministic fills ---------------------------------------------------------------
__device__ __forceinline__ uint32_t hash32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (uint32_t)x;
}
// two random e2m1 codes per byte (all 16 codes are finite: 0, .5, 1, 1.5, 2, 3, 4, 6 and negatives)
__global__ void k_fill_fp4(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = hash32(i ^ ((uint64_t)seed << 32)) & 0xFF;
}
// ue4m3 scale factors with exponent in {6,7,8} -> values in [0.5, 3.75]
__global__ void k_fill_ue4m3(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = hash32(i ^ ((uint64_t)seed << 32));
        p[i] = (uint8_t)(((6 + h % 3) << 3) | ((h >> 4) & 7));
    }
}
// e4m3 with |x| in [0.25, 2) and random sign
__global__ void k_fill_e4m3(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = hash32(i ^ ((uint64_t)seed << 32));
        p[i] = (uint8_t)(((h & 1) << 7) | ((5 + (h >> 1) % 3) << 3) | ((h >> 4) & 7));
    }
}

// ---- independent decoders for the reference ---------------------------------------------
__device__ __forceinline__ float e2m1_to_f(uint32_t c) {
    float m = (float)(c & 1);
    int e = (c >> 1) & 3;
    float v = e == 0 ? 0.5f * m : ldexpf(1.0f + 0.5f * m, e - 1);
    return (c & 8) ? -v : v;
}
__device__ __forceinline__ float ue4m3_to_f(uint32_t b) {
    uint32_t e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? ldexpf((float)m / 8.0f, -6) : ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
}
__device__ __forceinline__ float e4m3_to_f(uint32_t b) {
    float v = ue4m3_to_f(b & 0x7F);
    return (b & 0x80) ? -v : v;
}

#if defined(KIND_NVFP4)
template <class LSFA, class LSFB>
__global__ void k_ref(const uint8_t* __restrict__ A, const uint8_t* __restrict__ SFA, LSFA lsfa,
                      const uint8_t* __restrict__ B, const uint8_t* __restrict__ SFB, LSFB lsfb, int K,
                      const uint2* __restrict__ pairs, int npairs, int low_first, float* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npairs) return;
    const int m = pairs[t].x, n = pairs[t].y;
    const uint8_t* a = A + (size_t)m * K / 2;
    const uint8_t* b = B + (size_t)n * K / 2;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const float sa = ue4m3_to_f(SFA[lsfa(m, k0, 0)]);
        const float sb = ue4m3_to_f(SFB[lsfb(n, k0, 0)]);
        float blk = 0.f;
        for (int k = k0; k < k0 + 16; k += 2) {
            const uint32_t ab = a[k >> 1], bb = b[k >> 1];
            const uint32_t a0 = low_first ? (ab & 15) : (ab >> 4), a1 = low_first ? (ab >> 4) : (ab & 15);
            const uint32_t b0 = low_first ? (bb & 15) : (bb >> 4), b1 = low_first ? (bb >> 4) : (bb & 15);
            blk += e2m1_to_f(a0) * e2m1_to_f(b0) + e2m1_to_f(a1) * e2m1_to_f(b1);
        }
        acc += sa * sb * blk;
    }
    out[t] = acc;
}
#else
__global__ void k_ref(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int K, const uint2* __restrict__ pairs,
                      int npairs, float* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npairs) return;
    const int m = pairs[t].x, n = pairs[t].y;
    const uint8_t* a = A + (size_t)m * K;
    const uint8_t* b = B + (size_t)n * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += e4m3_to_f(a[k]) * e4m3_to_f(b[k]);
    out[t] = acc;
}
#endif

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    int M = 256, N = 17408, K = 5120, iters = 20, warmup = 3, verify = 1, npairs = 1 << 16, swizzle = 0;
    char raster = 'h';
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--m")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--n")) N = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--k")) K = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--verify")) verify = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--pairs")) npairs = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--swizzle")) swizzle = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--raster")) raster = argv[i + 1][0];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    const char* tag = "RESULT kind=" KIND_NAME " tile=" ;
    char head[128];
    snprintf(head, sizeof head, "%s%dx%dx%d sched=%s m=%d n=%d k=%d swizzle=%d raster=%c", tag, TILE_M, TILE_N, TILE_K,
             SCHED_NAME, M, N, K, swizzle, raster);

    // ---- strides / scale-factor layouts ----
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    const size_t a_bytes = (size_t)(M * (double)K * BytesPerElem), b_bytes = (size_t)(N * (double)K * BytesPerElem);
    const size_t d_bytes = (size_t)M * N * sizeof(ElementD);
    uint8_t *A, *B;
    ElementC* C;
    ElementD* D;
    CK(cudaMalloc(&A, a_bytes));
    CK(cudaMalloc(&B, b_bytes));
    CK(cudaMalloc(&C, d_bytes));
    CK(cudaMalloc(&D, d_bytes));
    CK(cudaMemset(C, 0, d_bytes));
    CK(cudaMemset(D, 0, d_bytes));
    double sf_bytes = 0;

#if defined(KIND_NVFP4)
    using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    const size_t sfa_n = size(filter_zeros(layout_SFA)), sfb_n = size(filter_zeros(layout_SFB));
    uint8_t *SFA, *SFB;
    CK(cudaMalloc(&SFA, sfa_n));
    CK(cudaMalloc(&SFB, sfb_n));
    sf_bytes = (double)sfa_n + sfb_n;
    k_fill_fp4<<<1024, 256>>>(A, a_bytes, 1);
    k_fill_fp4<<<1024, 256>>>(B, b_bytes, 2);
    k_fill_ue4m3<<<1024, 256>>>(SFA, sfa_n, 3);
    k_fill_ue4m3<<<1024, 256>>>(SFB, sfb_n, 4);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K, 1},
                                  {reinterpret_cast<ElementA::DataType const*>(A), stride_A,
                                   reinterpret_cast<ElementB::DataType const*>(B), stride_B,
                                   reinterpret_cast<ElementA::ScaleFactorType const*>(SFA), layout_SFA,
                                   reinterpret_cast<ElementB::ScaleFactorType const*>(SFB), layout_SFB},
                                  {{1.0f, 0.0f}, C, stride_C, D, stride_D}};
#else
    k_fill_e4m3<<<1024, 256>>>(A, a_bytes, 1);
    k_fill_e4m3<<<1024, 256>>>(B, b_bytes, 2);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K, 1},
                                  {reinterpret_cast<ElementA const*>(A), stride_A, reinterpret_cast<ElementB const*>(B), stride_B},
                                  {{1.0f, 0.0f}, C, stride_C, D, stride_D}};
#endif
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.max_swizzle_size = swizzle;
    args.scheduler.raster_order = raster == 'm' ? RO::AlongM : raster == 'n' ? RO::AlongN : RO::Heuristic;

    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        printf("%s status=unsupported:%s\n", head, cutlassGetStatusString(st));
        return 0;
    }
    size_t ws_size = Gemm::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_size) CK(cudaMalloc(&ws, ws_size));
    st = gemm.initialize(args, ws);
    if (st != cutlass::Status::kSuccess) {
        printf("%s status=init_failed:%s\n", head, cutlassGetStatusString(st));
        return 0;
    }
    st = gemm.run();
    if (st != cutlass::Status::kSuccess) {
        printf("%s status=run_failed:%s\n", head, cutlassGetStatusString(st));
        return 0;
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // ---- verification on sampled outputs ----
    const char* vres = "skipped";
    double maxerr = 0;
    if (verify) {
        npairs = (int)std::min<long long>(npairs, (long long)M * N);
        std::vector<uint2> hp(npairs);
        uint64_t s = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < npairs; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            hp[i].x = (unsigned)((s >> 33) % M);
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            hp[i].y = (unsigned)((s >> 33) % N);
        }
        // make sure the corners are covered
        hp[0] = make_uint2(0, 0);
        hp[npairs - 1] = make_uint2(M - 1, N - 1);
        uint2* dp;
        float* dref;
        CK(cudaMalloc(&dp, npairs * sizeof(uint2)));
        CK(cudaMalloc(&dref, npairs * sizeof(float)));
        CK(cudaMemcpy(dp, hp.data(), npairs * sizeof(uint2), cudaMemcpyHostToDevice));
#if defined(KIND_NVFP4)
        // Which nibble holds element 0 in CUTLASS's packed fp4 arrays? Ask CUTLASS rather than assume.
        cutlass::Array<cutlass::float_e2m1_t, 2> probe;
        probe[0] = cutlass::float_e2m1_t(1.0f);  // e2m1 code 0b0010
        probe[1] = cutlass::float_e2m1_t(0.0f);
        uint8_t raw;
        memcpy(&raw, &probe, 1);
        int low_first = (raw & 0x0F) == 0x2 ? 1 : ((raw >> 4) == 0x2 ? 0 : -1);
        if (low_first < 0) { fprintf(stderr, "cannot determine fp4 nibble order (raw=0x%02x)\n", raw); return 2; }
        k_ref<<<(npairs + 255) / 256, 256>>>(A, SFA, layout_SFA, B, SFB, layout_SFB, K, dp, npairs, low_first, dref);
#else
        k_ref<<<(npairs + 255) / 256, 256>>>(A, B, K, dp, npairs, dref);
#endif
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> href(npairs);
        std::vector<ElementD> hD((size_t)M * N);
        CK(cudaMemcpy(href.data(), dref, npairs * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hD.data(), D, d_bytes, cudaMemcpyDeviceToHost));
        int nbad = 0;
        for (int i = 0; i < npairs; ++i) {
            double got = (double)float(hD[(size_t)hp[i].x * N + hp[i].y]);
            double ref = href[i];
            double err = fabs(got - ref), tol = 1e-2 * fabs(ref) + 0.5;  // bf16 out + fp32 reassociation
            maxerr = std::max(maxerr, err);
            if (!(err <= tol)) ++nbad;
        }
        vres = nbad == 0 ? "ok" : "FAIL";
        if (nbad) fprintf(stderr, "verify: %d / %d sampled outputs mismatch (max abs err %g)\n", nbad, npairs, maxerr);
        CK(cudaFree(dp));
        CK(cudaFree(dref));
    }

    // ---- timing ----
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) gemm.run();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(e0));
        gemm.run();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float t;
        CK(cudaEventElapsedTime(&t, e0, e1));
        ms.push_back(t);
    }
    CK(cudaGetLastError());
    const double med = median(ms), best = *std::min_element(ms.begin(), ms.end());
    const double flop = 2.0 * M * N * (double)K;
    const double bytes = (double)a_bytes + b_bytes + sf_bytes + d_bytes;
    printf("%s ms=%.4f best_ms=%.4f tflops=%.1f gbps=%.1f verify=%s maxerr=%.3g\n", head, med, best, flop / med / 1e9,
           bytes / med / 1e6, vres, maxerr);
    return strcmp(vres, "FAIL") == 0 ? 1 : 0;
}
