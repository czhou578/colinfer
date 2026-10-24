---
paths:
  - "csrc/**"
  - "engine/model/fast.py"
  - "engine/model/moe_kernels.py"
  - "engine/spec/mtp.py"
  - "engine/runtime/scheduler.py"
  - "engine/runtime/sessions.py"
  - "engine/weights/direct.py"
  - "engine/weights/ple.py"
  - "engine/server/worker.py"
---

# Sanitizer checks while you build

These files hold the CUDA kernels, the side streams, the pinned buffers and the threads of the engine. After you change
one of them, run the check that matches the change before you end the turn. The Stop hook runs only ruff and pytest.

> **CAUTION:** The GPU and the host share 128 GB of unified memory, and an over-commit powers off the machine. Run
> these checks only with 16 GB or more of `MemAvailable`, and never next to `tests/golden.py` or another model load.

| You changed | Run |
|---|---|
| A kernel in `csrc/` | Its tests (the list below) under memcheck: `PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest tests/test_<kernel>.py -q -x`. Without the variable, the caching allocator hides out-of-bounds accesses. |
| Shared memory or `__syncthreads()` in a kernel | Also `--tool racecheck` and `--tool synccheck` on the same tests. Filter racecheck to the engine's kernels (below). |
| A side stream, a `wait_stream` or a non-blocking copy | The module's tests with `TORCH_CUDA_SANITIZER=1`. This sanitizer sees torch ops only, not the kernels of `csrc/`. |
| A thread, a lock, a condition or a queue | The module's tests three times: `.venv/bin/python -X dev -m pytest tests/test_<module>.py -q -x`. |

Most kernel files have one test file of the same name. These files do not:

- `csrc/moe.cu`: `tests/test_moe.py`. `csrc/moe_prefill.cu`: `tests/test_moe_prefill.py`.
- `csrc/attn_tc.cuh` (the tensor-core core of the decode attention): `tests/test_attn_decode.py` and `tests/test_qsa.py`.
- `csrc/skinny_core.cuh` (shared by `skinny.cu` and `moe.cu`): `tests/test_skinny.py` and `tests/test_moe.py`.
- `csrc/ple_io.cpp`: `tests/test_ple_io.py`. It holds a host thread pool, not a kernel, so the thread rule applies.

**racecheck.** PyTorch's own fused-attention kernels (`fmha_cutlass*`), which the test references call, report hazards
of their own. Filter racecheck to the engine's kernels with one `--kernel-name kns=<kernel>` per kernel, or with one
regex:

```bash
compute-sanitizer --tool racecheck --kernel-name kns=k_topk --kernel-name kns=k_tc ...
compute-sanitizer --tool racecheck --kernel-name regex='k_topk|k_tc' ...
```

- The filter matches the mangled kernel name. A namespace such as `qsa::` matches no kernel, and racecheck then
  checks nothing and reports 0 hazards. Take the names from `grep -n __global__ csrc/<file>`.
- A comma list in one option (`kns=k_topk,kns=k_tc`) is one filter that must match all its parts. Repeat the option
  for an alternative.
- `regex:<expr>` with a colon is the syntax of ncu. compute-sanitizer refuses it.

**synccheck.** The CUTLASS grouped GEMM (`tests/test_moe_prefill.py`) needs `--num-cuda-barriers 4096`.

A test that passes under these tools is not proof that the code has no race. `/reviewing-features` writes a targeted repro
for each suspected race (`.claude/agents/memrace-verifier.md`).
