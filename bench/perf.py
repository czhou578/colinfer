#!/usr/bin/env python3
"""Repeatable performance baseline of the engine in its server configuration, in-process, with no HTTP. The server
configuration is 3 slots x 262,144 tokens, 32 prefix checkpoints, INT6 / INT5 decode copies and the drafter files when
present, and suffix-match drafts.

  prefill   random-token prompts of 2k / 8k / 32k tokens, one token out: TTFT (submit -> first token) and prefill tok/s
  decode    12 chat prompts of the frozen 40-prompt mix (3 per kind), one at a time, 256 tokens greedy, MTP:
            per-request decode tok/s = (tokens - 1) / (last token - first token), and their TTFT
  plain     4 of those prompts, 128 tokens, without speculation (a 1-slot scheduler built after the first is freed)

Each section runs --warmup untimed rounds, then --runs timed rounds. The report gives the mean, std, min and max over
the timed rounds. Each round uses fresh prompts or a fresh cache_salt, so prefix checkpoints never shorten a prefill.

Memory, measured three ways, because nvidia-smi reports no per-process GPU memory on the unified memory of GB10:
  torch     the caching allocator's peak (torch.cuda.max_memory_allocated / _reserved)
  system    MemTotal - min(MemAvailable) over the run, minus the same before the model loaded (/proc/meminfo, sampled 2 Hz)
  rss       the process's peak resident set (VmHWM in /proc/self/status; GPU allocations do not show up there)
The bench also samples the GPU temperature, SM clock and power (nvidia-smi), to find thermal throttling between rounds.

   uv run python bench/perf.py [--runs 3] [--warmup 1] [--suffix N] [--json out.json]
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import threading
import time
import uuid

import torch

from engine.spec.suffix import MIN_MATCH
from engine.runtime.build import build_engine
from tests.golden import EOS, load_prompts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def meminfo_kb(key: str) -> int:
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise KeyError(key)


def status_kb(key: str) -> int:
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return 0


class Monitor(threading.Thread):
    """Samples system MemAvailable (2 Hz) and GPU temperature / SM clock / power (1 Hz) until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.min_avail = meminfo_kb("MemAvailable")
        self.gpu = []  # (t, temp C, sm clock MHz, power W)

    def run(self):
        n = 0
        while not self.stop.wait(0.5):
            self.min_avail = min(self.min_avail, meminfo_kb("MemAvailable"))
            n += 1
            if n % 2 == 0:
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                    self.gpu.append((time.time(), *[float(v) if v.strip().replace(".", "").isdigit() else None for v in out]))
                except Exception:  # noqa: BLE001  (monitoring must never break the benchmark)
                    pass


def stats(xs):
    xs = list(xs)
    return dict(mean=statistics.mean(xs), std=statistics.stdev(xs) if len(xs) > 1 else 0.0, min=min(xs), max=max(xs), n=len(xs))


def chat_prompts(n_per_kind: int):
    """The first n_per_kind prompts of each kind of the frozen 40-prompt mix (tests/golden/prompts.json)."""
    by_kind = {}
    for m in load_prompts()["mix"]:
        if len(by_kind.setdefault(m["kind"], [])) < n_per_kind:
            by_kind[m["kind"]].append(m["ids"])
    return [(k, x) for k, xs in sorted(by_kind.items()) for x in xs]


def run_requests(sched, reqs):
    sched.run(reqs)
    torch.cuda.synchronize()
    return reqs


def prefill_round(sched, lens, seed):
    from engine.runtime.scheduler import Request
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in lens:
        ids = torch.randint(1000, 150000, (L,), generator=g).tolist()
        r = run_requests(sched, [Request(ids, max_new_tokens=1)])[0]
        out[L] = dict(ttft=r.t_first - r.t_submit, tok_s=L / (r.t_first - r.t_admit))
    return out


def decode_round(sched, prompts, max_new):
    from engine.runtime.scheduler import Request
    salt = uuid.uuid4().hex  # no prefix reuse across rounds
    reqs = []
    for kind, x in prompts:
        r = run_requests(sched, [Request(x, max_new_tokens=max_new, eos_ids=EOS, cache_salt=salt)])[0]
        reqs.append((kind, r))
    tok = sum(len(r.output) - 1 for _, r in reqs)
    dec = sum(r.t_done - r.t_first for _, r in reqs)
    per_kind = {}
    for kind, r in reqs:
        a = per_kind.setdefault(kind, [0, 0.0])
        a[0] += len(r.output) - 1
        a[1] += r.t_done - r.t_first
    return dict(tok_s=tok / dec, ttft=statistics.mean(r.t_first - r.t_submit for _, r in reqs),
                per_kind={k: v[0] / v[1] for k, v in per_kind.items()}, tokens=tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--lens", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--suffix", type=int, default=MIN_MATCH, help="suffix-match drafts of at least N tokens (the server's --suffix-drafts; 0: off)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    avail0, total = meminfo_kb("MemAvailable"), meminfo_kb("MemTotal")
    mon = Monitor()
    mon.start()
    res = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), git=subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                                                            capture_output=True, text=True).stdout.strip())
    with torch.inference_mode():
        t0 = time.perf_counter()
        sched, _ = build_engine(suffix_drafts=a.suffix)  # the server's engine
        res["startup_s"] = time.perf_counter() - t0
        prompts = chat_prompts(3)
        rounds = []
        for i in range(a.warmup + a.runs):
            rounds.append(prefill_round(sched, a.lens, seed=i))
            print(f"[perf] prefill round {i}: " + ", ".join(f"{L}: {v['ttft']:.3f} s" for L, v in rounds[-1].items()), flush=True)
        rounds = rounds[a.warmup:]
        res["prefill"] = {L: dict(ttft_s=stats(r[L]["ttft"] for r in rounds), tok_s=stats(r[L]["tok_s"] for r in rounds)) for L in a.lens}
        rounds = []
        for i in range(a.warmup + a.runs):
            rounds.append(decode_round(sched, prompts, 256))
            print(f"[perf] MTP decode round {i}: {rounds[-1]['tok_s']:.2f} tok/s, TTFT {rounds[-1]['ttft']:.3f} s", flush=True)
        rounds = rounds[a.warmup:]
        res["decode_mtp"] = dict(tok_s=stats(r["tok_s"] for r in rounds), ttft_s=stats(r["ttft"] for r in rounds),
                                 per_kind={k: stats(r["per_kind"][k] for r in rounds) for k in rounds[0]["per_kind"]})
        res["memory_gb"] = dict(torch_allocated=torch.cuda.max_memory_allocated() / 1e9, torch_reserved=torch.cuda.max_memory_reserved() / 1e9,
                                system=((avail0 - mon.min_avail) / 1e6), rss=status_kb("VmHWM") / 1e6, mem_total=total / 1e6)
        del sched
        gc.collect()
        torch.cuda.empty_cache()
        plain, _ = build_engine(spec="none", slots=1, max_seq_len=8192, checkpoints=0)
        rounds = []
        for i in range(a.warmup + a.runs):
            rounds.append(decode_round(plain, prompts[::3], 128))
            print(f"[perf] plain decode round {i}: {rounds[-1]['tok_s']:.2f} tok/s", flush=True)
        rounds = rounds[a.warmup:]
        res["decode_plain"] = dict(tok_s=stats(r["tok_s"] for r in rounds))
    mon.stop.set()
    mon.join()
    temps = [g[1] for g in mon.gpu if g[1] is not None]
    clocks = [g[2] for g in mon.gpu if g[2] is not None]
    res["gpu"] = dict(temp_c=stats(temps) if temps else None, sm_clock_mhz=stats(clocks) if clocks else None)
    f = lambda s: f"{s['mean']:8.3f} ± {s['std']:.3f}  [{s['min']:.3f}, {s['max']:.3f}]"  # noqa: E731
    print(f"\n[perf] {res['git']}  startup {res['startup_s']:.1f} s  ({a.runs} timed rounds after {a.warmup} warm-up)")
    for L, v in res["prefill"].items():
        print(f"  prefill {L:>6}: TTFT {f(v['ttft_s'])} s   {v['tok_s']['mean']:7.0f} ± {v['tok_s']['std']:.0f} tok/s")
    d = res["decode_mtp"]
    print(f"  decode, MTP:   {f(d['tok_s'])} tok/s   TTFT (chat prompts) {d['ttft_s']['mean']:.3f} s   "
          + "  ".join(f"{k} {v['mean']:.1f}" for k, v in d["per_kind"].items()))
    print(f"  decode, plain: {f(res['decode_plain']['tok_s'])} tok/s")
    m = res["memory_gb"]
    print(f"  peak memory: torch allocated {m['torch_allocated']:.1f} GB, reserved {m['torch_reserved']:.1f} GB; system (MemAvailable drop) "
          f"{m['system']:.1f} GB of {m['mem_total']:.0f}; process RSS {m['rss']:.1f} GB")
    if res["gpu"]["temp_c"]:
        print(f"  GPU: {res['gpu']['temp_c']['min']:.0f}-{res['gpu']['temp_c']['max']:.0f} C, SM clock "
              f"{res['gpu']['sm_clock_mhz']['min']:.0f}-{res['gpu']['sm_clock_mhz']['max']:.0f} MHz")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
