// skinny_core.cuh -- the dequantization helpers of the weight-streaming GEMMs: csrc/skinny.cu (the decode linears) and
// csrc/moe.cu (the decode experts). Both files include this header, so an expert weight dequantizes into exactly the same
// bf16 values as a dense weight of the same format.
//
// Also the host accessor of the skip flag of the draft early exit (skinny_set_skip in skinny.cu): the MoE launches read the
// same flag as the skinny launches. Tests: tests/test_skinny.py and tests/test_moe.py.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace skinny {

__device__ __forceinline__ uint32_t h2_to_bf2(__half2 h) {
    const float2 f = __half22float2(h);
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&b);
}

// two e2m1 (one byte) times an f16 block scale -> bf16x2, exact
__device__ __forceinline__ uint32_t fp4x2_scaled(uint32_t byte, __half2 scale) {
    __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)byte, __NV_E2M1);
    return h2_to_bf2(__hmul2(*reinterpret_cast<__half2*>(&r), scale));
}

// two e4m3 -> bf16x2, exact
__device__ __forceinline__ uint32_t fp8x2_bf(uint32_t two) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    return h2_to_bf2(*reinterpret_cast<__half2*>(&r));
}

__device__ __forceinline__ __nv_bfloat162 e4m3_to_bf2(uint32_t byte) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)byte, __NV_E4M3);
    return __float2bfloat162_rn(__half2float(*reinterpret_cast<__half*>(&r)));
}

// one e4m3 block scale -> f16x2 (both halves), exact
__device__ __forceinline__ __half2 e4m3_to_h2(uint32_t byte) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)byte, __NV_E4M3);
    const __half h = *reinterpret_cast<__half*>(&r);
    return __halves2half2(h, h);
}

}  // namespace skinny

// The device int of the skip flag that launches made now contain (nullptr: none). skinny.cu owns the pointer.
const int* skinny_skip_flag();
