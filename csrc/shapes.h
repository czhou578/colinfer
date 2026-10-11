// shapes.h -- the model dimensions the kernels are compiled for (nvidia/Qwen3.8-27B-NVFP4).
//
// Plain C++: the kernels include it through common.cuh, and csrc/bindings.cpp (the host compiler) checks tensor shapes
// against it and exports the values to Python, where engine/model/fast.py to_fast refuses a checkpoint with other
// dimensions before anything runs.
#pragma once

constexpr int HEAD_DIM = 256;  // attention head dim: the rows of the KV cache, half a q row (q | gate)
constexpr int GDN_DK = 128;    // Gated DeltaNet key head dim
constexpr int GDN_DV = 128;    // Gated DeltaNet value head dim
