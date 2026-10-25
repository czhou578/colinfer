---
paths:
  - "csrc/**"
  - "engine/model/fast.py"
  - "engine/model/family.py"
  - "engine/model/moe_kernels.py"
  - "engine/model/prefill_qwen4exp.py"
  - "engine/spec/mtp.py"
  - "engine/spec/mtp4.py"
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
- `csrc/ple_io.cpp`: `tests/test_ple_io.py` and `tests/test_ple_feeder.py` (the row cache). It holds a host thread
  pool and a mutex, not a kernel, so the thread rule applies.
- `csrc/qsa_prefill.cu`: `tests/test_qsa_prefill.py`.
- `csrc/hc_prefill.cu`: `tests/test_prefill_qwen4exp.py` (`test_hc_pre_post`).
- `csrc/hc.cu`: `tests/test_hc.py`, and also `tests/test_requant_qwen4exp.py` for the INT8 mode of `hc_mix`.

The Python files of the list hold these: a side stream in `moe_kernels.py` (the shared expert) and in `mtp4.py` (the
commit branch of `Mtp4Cycle`), non-blocking copies from pinned buffers in `family.py` (`ngram_before_replay`) and in
`prefill_qwen4exp.py` (the n-gram rows of a chunk), and the prefetch thread and the locks of `PleFeeder` in
`engine/weights/ple.py`.

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

## Scripts that load a model

> **CAUTION:** Two model loads at the same time over-commit the unified memory, and the machine powers off. Run each
> script that loads a model under the GPU lock, and check `MemAvailable` first.

1. Make sure that no server runs: `systemctl --user is-active colinfer` must not show `active`.
2. Read `MemAvailable` in `/proc/meminfo`. Qwen3.8-Flash-Next needs 100 GB or more. The 27B at 3 slots needs 105 GB or
   more.
3. Start the script under the lock: `flock -w 14400 ~/.cache/colinfer/gpu-model.lock <command>`.

These scripts load a model: `tests/golden.py`, `tests/scheduler_check.py`, `tests/passkey.py`, `tests/perplexity.py`,
`tests/golden_qwen4exp*.py`, `tests/scheduler_check_qwen4exp.py`, `tests/passkey_qwen4exp.py`,
`tools/qwen4exp_reference.py`, `tools/requant_qwen4exp.py` (up to 14 GB on the GPU) and `tests/check.sh` without
`--quick`. The sanitizer runs of the unit tests above load no model, but they too need the 16 GB of the caution box.

A test that passes under these tools is not proof that the code has no race. `/reviewing-features` writes a targeted repro
for each suspected race (`.claude/agents/memrace-verifier.md`).
