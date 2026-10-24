// qsa.cu -- Qwen Sparse Attention kernels for qwen4_exp (docs/history/qwen4exp_plan.md section 5.3): the indexer prologue, the
// block keys, the block scores, the top-k selection and the sparse decode attention. The bindings are in qsa_bind.cpp.
#include "common.cuh"
