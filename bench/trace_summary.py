#!/usr/bin/env python3
"""trace_summary.py -- summarize an nsys trace of decode steps (bench/traces/*.sqlite).

The script splits the kernel timeline into steps at the lm_head GEMV (the one fp32-output NVFP4 GEMV). Then it reports
these values per step: the wall span, the busy time (the union of the kernel intervals) and the idle gaps. It also
reports the per-kernel time breakdown of the median step.

  nsys profile --cuda-graph-trace=node --trace=cuda -o bench/traces/X python bench/decode_bench.py ...
  nsys export --type sqlite bench/traces/X.nsys-rep   (nsys stats does this implicitly)
  python bench/trace_summary.py bench/traces/X.sqlite
"""
import collections
import sqlite3
import statistics
import sys


def main(path):
    db = sqlite3.connect(path)
    rows = db.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                         JOIN StringIds s ON k.shortName = s.id ORDER BY k.start""").fetchall()
    # lm_head is the only NVFP4 GEMV / skinny GEMM instance whose duration is ~3 ms; use duration to find it
    steps, cur = [], []
    for st, en, name in rows:
        cur.append((st, en, name))
        if ("k_nvfp4" in name or "k_skinny" in name) and "swiglu" not in name and (en - st) > 2_000_000:
            steps.append(cur)
            cur = []
    steps = [s for s in steps if len(s) > 300]  # full decode steps only (skip warm-up fragments)
    if not steps:
        sys.exit("no complete decode steps found")
    stats = []
    for s in steps[1:]:
        span = s[-1][1] - s[0][0]
        busy, last_end = 0, s[0][0]
        for st, en, _ in s:
            st = max(st, last_end)
            if en > st:
                busy += en - st
                last_end = en
        stats.append((span, busy, len(s)))
    spans = [x[0] for x in stats]
    med = sorted(range(len(stats)), key=lambda i: spans[i])[len(stats) // 2]
    span, busy, n = stats[med]
    print(f"{path}: {len(stats)} steps analysed")
    print(f"median step: span {span / 1e6:.2f} ms, GPU busy {busy / 1e6:.2f} ms, idle {(span - busy) / 1e6:.2f} ms "
          f"({100 * (span - busy) / span:.1f}%), {n} kernels")
    print(f"span over steps: min {min(spans) / 1e6:.2f} / median {statistics.median(spans) / 1e6:.2f} / max {max(spans) / 1e6:.2f} ms")
    by = collections.defaultdict(lambda: [0, 0])
    for st, en, name in steps[1:][med]:
        key = name.split("(")[0]
        by[key][0] += en - st
        by[key][1] += 1
    gaps = collections.defaultdict(lambda: [0, 0])  # idle time by (previous kernel, next kernel)
    st_ = steps[1:][med]
    last_end, last_name = st_[0][1], st_[0][2]
    for st, en, name in st_[1:]:
        if st > last_end:
            k = (last_name.split("(")[0][:40], name.split("(")[0][:40])
            gaps[k][0] += st - last_end
            gaps[k][1] += 1
        if en > last_end:
            last_end, last_name = en, name
    print(f"\n{'idle gap: previous kernel -> next kernel':84s} {'ms':>7s} {'count':>6s}")
    for (a, b), (t, c) in sorted(gaps.items(), key=lambda x: -x[1][0])[:12]:
        print(f"{a:40s} -> {b:40s} {t / 1e6:7.3f} {c:6d}")
    print(f"\n{'kernel':60s} {'ms':>7s} {'count':>6s} {'%':>5s}")
    for k, (t, c) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"{k[:60]:60s} {t / 1e6:7.2f} {c:6d} {100 * t / busy:5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
