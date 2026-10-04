"""Force transformers' Qwen3.5 linear-attention functions back to their PyTorch reference paths.

transformers swaps in flash-linear-attention / causal-conv1d kernels at import time when those
packages are installed (FLA is a project dependency since Phase 3). The Phase 1 parity results were
taken against the PyTorch paths (bit-exact), and FLA's Triton kernels do not run on the CPU, so the
reference side of every comparison calls this first.
"""


def force_hf_torch_fallbacks():
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        f = getattr(m, name)
        setattr(m, name, getattr(f, "__wrapped__", f))
