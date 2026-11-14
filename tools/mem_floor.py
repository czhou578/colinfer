#!/usr/bin/env python3
"""mem_floor.py -- the memory that an engine process takes before it loads a model (docs/history/PLAN_QWEN35_MOE.md
section 5, step 1). The tool starts a child process that runs the steps of STEPS one after another, and the parent
measures the unified memory after each step:

- avail: the fall of MemAvailable since the start of the child (the measure of ENGINE_FIXED_BYTES in
  engine/runtime/build.py). The lowest value during the step gives its peak.
- rss, anon, file: the resident memory of the child, and its anonymous and file-backed parts (Pss of
  /proc/<pid>/smaps_rollup). File-backed pages are the libraries; the kernel can drop them.
- gpu: the memory of the child's CUDA context, from nvidia-smi. It is not part of the resident memory.
- torch: the bytes that the torch allocator holds (reserved), and the bytes in use (allocated).

The parent imports no torch, so it adds nothing to the numbers. Other processes also move MemAvailable: the tool
prints the range of MemAvailable over two seconds before the start, and its drift from the start to the time after
the child's end. A drift near zero means that the avail column is the child's own memory.

CAUTION: The GPU and the host share the unified memory, and an over-commit powers off the machine. The tool needs
about 3 GB. It does not start with less than --min-avail-gb of MemAvailable (default 16), and it stops the child when
MemAvailable falls below --stop-avail-gb (default 8).

Run it with uv run: the kernel extension needs the ninja of the venv on the PATH.

    uv run python tools/mem_floor.py                     # the table
    uv run python tools/mem_floor.py --json floor.json   # also the rows as JSON

Tests: tests/test_mem_floor.py (the parsers and the table; no GPU).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

GB = 10**9
MARK = "MEMFLOOR "  # the child's report line of a step: MARK + JSON
STEPS = (
    ("python", "the interpreter"),
    ("torch", "import torch"),
    ("cuda", "the CUDA context and one tensor"),
    ("kernels", "the kernel extension (engine/kernels)"),
    ("server", "the imports of the server (engine.server.api, the drafter)"),
    ("tokenizer", "the tokenizer of the default model"),
    ("cublas", "one bf16 matmul (cuBLAS)"),
    ("selftest", "each matmul path once (engine/selftest.py)"),
    ("graph", "one small CUDA graph in a graph pool"),
    ("release", "gc and torch.cuda.empty_cache()"),
)


def meminfo(text: str, key: str) -> int:
    """A field of /proc/meminfo in bytes. A missing field is a KeyError."""
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise KeyError(key)


def smaps(text: str) -> dict:
    """rss, anon and file of /proc/<pid>/smaps_rollup in bytes (Pss, so shared pages count by their share)."""
    f = {p[0].rstrip(":"): int(p[1]) * 1024 for p in (line.split() for line in text.splitlines()) if len(p) >= 3 and p[2] == "kB"}
    return {"rss": f.get("Pss", 0), "anon": f.get("Pss_Anon", 0), "file": f.get("Pss_File", 0) + f.get("Pss_Shmem", 0)}


def gpu_bytes(text: str, pid: int) -> int:
    """The memory of process pid in the output of nvidia-smi --query-compute-apps=pid,used_memory (csv, MiB); 0 when
    the process has no CUDA context, or when the driver reports no value."""
    for line in text.splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) == 2 and p[0] == str(pid) and p[1].isdigit():
            return int(p[1]) * 2**20
    return 0


def table(rows: list[dict]) -> str:
    """The Markdown table of the rows, in GB (1e9), with the growth of avail against the row before."""
    head = ("Step", "What", "avail", "step", "peak", "rss", "anon", "file", "gpu", "torch reserved", "allocated")
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    prev = 0
    for r in rows:
        v = [r["avail"], r["avail"] - prev, r["peak"], r["rss"], r["anon"], r["file"], r["gpu"], r["reserved"], r["allocated"]]
        out.append(f"| {r['step']} | {r['what']} | " + " | ".join(f"{x / GB:+.2f}" if i == 1 else f"{x / GB:.2f}" for i, x in enumerate(v)) + " |")
        prev = r["avail"]
    return "\n".join(out)


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def available(samples: int = 5, gap: float = 0.2) -> int:
    """The median of `samples` reads of MemAvailable, `gap` seconds apart."""
    vals = []
    for i in range(samples):
        vals.append(meminfo(read("/proc/meminfo"), "MemAvailable"))
        if i + 1 < samples:
            time.sleep(gap)
    return int(statistics.median(vals))


# ------------------------------------------------------------------------------------------------------------ the child
def child() -> None:
    """Runs one step for each line of stdin and answers with a MARK line: the torch allocator's bytes, or an error."""
    keep = {}

    def step(name: str) -> None:
        if name == "python":
            return
        import torch
        if name == "cuda":
            torch.cuda.init()
            keep["x"] = torch.zeros(1, device="cuda")
        elif name == "kernels":
            from engine.kernels import ops
            keep["ops"] = ops()
        elif name == "server":
            import engine.server.api  # noqa: F401  the HTTP layer, the worker, the build, the scheduler, the grammar
            import engine.spec.mtp  # noqa: F401
        elif name == "tokenizer":
            from transformers import AutoTokenizer

            from engine.weights.loader import MODEL, resolve
            keep["tok"] = AutoTokenizer.from_pretrained(resolve(MODEL))
            keep["tok"]("The memory floor of the engine.")
        elif name == "cublas":
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            keep["y"] = a @ a
        elif name == "selftest":
            from engine.selftest import run_selftest
            run_selftest()
        elif name == "graph":
            from engine.model.fast import capture
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            keep["graph"] = capture(lambda: a @ a, pool=torch.cuda.graph_pool_handle())
            keep["graph"][0].replay()
        elif name == "release":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

    for line in sys.stdin:
        name, rep = line.strip(), {}
        try:
            step(name)
            if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
                cuda = sys.modules["torch"].cuda
                rep = {"reserved": cuda.memory_reserved(), "allocated": cuda.memory_allocated()}
        except Exception as e:  # the parent prints the error and goes on: a missing checkpoint skips only its step
            rep = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(MARK + json.dumps(rep), flush=True)


# ----------------------------------------------------------------------------------------------------------- the parent
def run(min_avail_gb: float, stop_avail_gb: float, settle: float) -> tuple[list[dict], int, int]:
    """(the rows of the steps, the range of MemAvailable before the start, its drift over the run), in bytes. The drift
    is the fall of MemAvailable from the start to the time after the child's end: the work of the other processes."""
    before = []
    for _ in range(10):  # two seconds: the noise of MemAvailable from the other processes
        before.append(available(1))
        time.sleep(0.2)
    if min(before) < min_avail_gb * GB:
        raise SystemExit(f"mem_floor: {min(before) / GB:.1f} GB of memory available; the tool needs {min_avail_gb:.0f} GB (--min-avail-gb)")
    base = int(statistics.median(before))
    p = subprocess.Popen([sys.executable, __file__, "--child"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    low, done = [base], threading.Event()

    def watch() -> None:  # the lowest MemAvailable of the step, and the stop of the child under the limit
        while not done.wait(0.25):
            low[0] = min(low[0], meminfo(read("/proc/meminfo"), "MemAvailable"))
            if low[0] < stop_avail_gb * GB:
                p.kill()

    threading.Thread(target=watch, daemon=True).start()
    rows = []
    try:
        for name, what in STEPS:
            low[0] = meminfo(read("/proc/meminfo"), "MemAvailable")
            p.stdin.write(name + "\n")
            p.stdin.flush()
            rep = None
            for line in p.stdout:
                if line.startswith(MARK):
                    rep = json.loads(line[len(MARK):])
                    break
                sys.stderr.write(line)
            if rep is None:
                raise SystemExit(f"mem_floor: the child ended in step {name!r} (lowest MemAvailable {low[0] / GB:.1f} GB)")
            if "error" in rep:
                print(f"mem_floor: step {name!r} failed and is not in the table: {rep['error']}", file=sys.stderr)
                continue
            time.sleep(settle)
            smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            rows.append({"step": name, "what": what, "avail": base - available(), "peak": base - low[0], "gpu": gpu_bytes(smi, p.pid),
                         "reserved": rep.get("reserved", 0), "allocated": rep.get("allocated", 0), **smaps(read(f"/proc/{p.pid}/smaps_rollup"))})
    finally:
        done.set()
        p.stdin.close()
        p.wait(timeout=60)
    time.sleep(settle)
    return rows, max(before) - min(before), base - available()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="The memory of an engine process before it loads a model.")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-avail-gb", type=float, default=16.0, help="do not start with less MemAvailable (GB of 1e9 bytes)")
    ap.add_argument("--stop-avail-gb", type=float, default=8.0, help="stop the child when MemAvailable falls below this value")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds between the end of a step and its measurement")
    ap.add_argument("--json", help="write the rows and the noise to this file")
    args = ap.parse_args(argv)
    if args.child:
        return child()
    rows, noise, drift = run(args.min_avail_gb, args.stop_avail_gb, args.settle)
    print(table(rows))
    print(f"\nGB = 1e9 bytes. avail: the fall of MemAvailable since the start; step: its growth in the step; peak: the largest fall "
          f"during the step. MemAvailable moved by {noise / GB:.2f} GB in the two seconds before the start. After the end of the child "
          f"it was {drift / GB:+.2f} GB below the start: the drift of the other processes over the run.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "noise": noise, "drift": drift}, f, indent=1)


if __name__ == "__main__":
    main()
