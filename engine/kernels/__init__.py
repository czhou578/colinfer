"""Builds and loads the CUDA decode kernels in csrc/ (JIT via torch.utils.cpp_extension + ninja,
cached under build/torch_ext; rebuilt automatically when a source changes)."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_ext = None


def ops():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        build = os.path.join(_ROOT, "build", "torch_ext")
        os.makedirs(build, exist_ok=True)
        _ext = load(
            name="colinfer_kernels",
            sources=[os.path.join(_ROOT, "csrc", f) for f in ("bindings.cpp", "gemv.cu", "attn_decode.cu", "attn_prefill.cu", "gdn_step.cu", "norm.cu", "gemm_nvfp4.cu", "prefill_ops.cu", "sampling.cu", "skinny.cu")],
            extra_cflags=["-O3"],
            extra_include_paths=[os.path.join(_ROOT, "csrc", "third_party", "cutlass", p) for p in ("include", "tools/util/include")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo", "--expt-relaxed-constexpr", "--fmad=false",
                               "-Xptxas=-v" if os.environ.get("COLINFER_PTXAS_VERBOSE") else "-DNOVERBOSE"],
            build_directory=build,
            verbose=bool(os.environ.get("COLINFER_BUILD_VERBOSE")),
        )
    return _ext
