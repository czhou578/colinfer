// moe.cu -- decode MoE kernels for qwen4_exp (docs/history/qwen4exp_plan.md section 5.2): expert routing, the expert SwiGLU and
// down projections on the skinny-GEMM streaming pattern, and the fixed-order combine. The bindings are in moe_bind.cpp.
#include "common.cuh"
