// common.cuh -- device helpers that more than one kernel file uses: PTX wrappers (cp.async, ldmatrix, mma.sync,
// ex2.approx), bf16 / fp8 / fp4 conversions, and the CUTLASS scale-factor layout of the NVFP4 GEMMs.
//
// Each helper is defined once, here, so that a change (the e2m1 rounding rule, the scale swizzle) cannot drift between
// the files. silu() is not here on purpose: csrc/prefill_ops.cu uses the precise expf, which must round like torch, and
// csrc/gdn_step.cu the fast __expf.
//
// The helpers live in namespace cc: a kernel file brings them into its own namespace with `using namespace cc;`.
// csrc/gemm_nvfp4.cu, which has CUTLASS's `using namespace cute`, names the two it uses instead (cute has its own
// cp_async_wait, and a second one at global scope makes CUTLASS's calls ambiguous).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace cc {

// ---- shared memory and cp.async ----
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// 16-byte global -> shared copy; valid = false copies zeros (src-size 0), so a tail can be padded without a branch
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void cp_async_wait0() { cp_async_wait<0>(); }

// ---- ldmatrix: four 8x8 b16 tiles per warp, straight and transposed ----
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

// ---- mma.sync m16n8k16, fp32 accumulate: f16 and bf16 operands ----
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- math ----
__device__ __forceinline__ float ex2(float x) {  // 2^x, the approximate hardware instruction (softmax exponentials)
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }  // round through bf16

// ---- conversions ----
// 8 bf16 in a uint4 -> 8 floats
__device__ __forceinline__ void bf16x8_to_float(const uint4 v, float* f) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(u[i] << 16);
        f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}
// two e4m3 (the low 16 bits) -> a packed half2
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t two) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}
// |a| -> the 3-bit e2m1 magnitude code: grid 0, .5, 1, 1.5, 2, 3, 4, 6; midpoints .25 .75 1.25 1.75 2.5 3.5 5; ties to
// even mantissa; saturating at 6
__device__ __forceinline__ uint32_t e2m1_code(float a) {
    return (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
}

// ---- the CUTLASS SM120 block-scale layout (128 rows x 4 scale columns per 512-byte atom) ----
// element offset of the scale of row r, k-block kb (16 elements each); kb4 = ceil(K / 64): atoms per row of 128
__device__ __forceinline__ size_t sf_offset(int r, int kb, int kb4) {
    return ((size_t)(r >> 7) * kb4 + (kb >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (kb & 3);
}

}  // namespace cc
