"""Force the linear-attention functions of transformers' Qwen3.5 and Qwen4-Exp back to their PyTorch reference paths.

At import time, transformers swaps in flash-linear-attention / causal-conv1d kernels when those packages are present.
(The engine does not depend on them, but an environment can have them.) The Phase 1 parity results used the PyTorch
paths (bit-exact), and the Triton kernels of FLA do not run on the CPU. Thus the reference side of each comparison
calls this first.
"""

FUNCTIONS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


def force_hf_torch_fallbacks():
    import transformers.models.qwen3_5.modeling_qwen3_5 as m35
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m4
    for m in (m35, m4):
        for name in FUNCTIONS:
            f = getattr(m, name)
            setattr(m, name, getattr(f, "__wrapped__", f))
