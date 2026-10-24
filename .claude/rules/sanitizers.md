---
paths:
  - "csrc/**"
  - "engine/model/fast.py"
  - "engine/spec/mtp.py"
  - "engine/runtime/scheduler.py"
  - "engine/runtime/sessions.py"
  - "engine/weights/direct.py"
  - "engine/server/worker.py"
---

# Sanitizer checks while you build

These files hold the CUDA kernels, the side streams, the pinned buffers and the threads of the engine. After you change
one of them, run the check that matches the change before you end the turn. The Stop hook runs only ruff and pytest.

> **CAUTION:** The GPU and the host share 128 GB of unified memory, and an over-commit powers off the machine. Run
> these checks only with 16 GB or more of `MemAvailable`, and never next to `tests/golden.py` or another model load.

| You changed | Run |
|---|---|
| A kernel in `csrc/` | Its test file under memcheck: `PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest tests/test_<kernel>.py -q -x`. Without the variable, the caching allocator hides out-of-bounds accesses. |
| Shared memory or `__syncthreads()` in a kernel | Also `--tool racecheck` and `--tool synccheck` on the same tests. |
| A side stream, a `wait_stream` or a non-blocking copy | The module's tests with `TORCH_CUDA_SANITIZER=1`. This sanitizer sees torch ops only, not the kernels of `csrc/`. |
| A thread, a lock, a condition or a queue | The module's tests three times: `.venv/bin/python -X dev -m pytest tests/test_<module>.py -q -x`. |

A test that passes under these tools is not proof that the code has no race. `/reviewing-features` writes a targeted repro
for each suspected race (`.claude/agents/memrace-verifier.md`).
