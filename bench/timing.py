"""The timing loop of the benchmarks."""
import time

import torch


def timed(fn, n: int = 20, warmup: int = 3) -> float:
    """Seconds per call of fn() on the GPU: `warmup` calls first (first-use compilation, autotuning), then n calls
    between two synchronizations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n
