#!/usr/bin/env python3
"""Side-by-side markdown tables from model-benchmarks run directories (core_runner.py output).

   python tools/harness_table.py NAME=~/Projects/model-benchmarks/results/<model>/<ts> [NAME=...]
"""
import json
import os
import sys


def load(d, name):
    p = os.path.join(os.path.expanduser(d), name)
    return json.load(open(p)) if os.path.exists(p) else {}


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def fmt(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    names = [n for n, _ in runs]
    lat = {n: load(d, "latency.json") for n, d in runs}
    lens = sorted({int(k) for v in lat.values() for k in v}, key=int)
    table("Prefill: TTFT median (s) / prefill tok/s, single request, no prefix reuse", ["prompt tokens"] + names,
          [[f"{L:,}"] + [f"{fmt(lat[n].get(str(L), {}).get('ttft_median_s'), 3)} / {fmt(lat[n].get(str(L), {}).get('prefill_tps_avg'), 0)}"
                         for n in names] for L in lens])
    deep = {n: load(d, "deep_context.json") for n, d in runs}
    dl = sorted({int(k) for v in deep.values() for k in v})
    if dl:
        table("Deep context: TTFT median (s) / prefill tok/s", ["context tokens"] + names,
              [[f"{L:,}"] + [f"{fmt(deep[n].get(str(L), {}).get('ttft_median_s'), 2)} / {fmt(deep[n].get(str(L), {}).get('prefill_tps_avg'), 0)}"
                             for n in names] for L in dl])
    dec = {n: load(d, "decode.json") for n, d in runs}
    ol = sorted({int(k) for v in dec.values() for k in v})
    table("Decode: average tok/s, single stream, prose prompt (thinking on)", ["output tokens"] + names,
          [[f"{L:,}"] + [fmt(dec[n].get(str(L), {}).get("tok_per_sec_avg")) for n in names] for L in ol])
    con = {n: load(d, "concurrency.json").get("per_concurrency_level", {}) for n, d in runs}
    cl = sorted({int(k) for v in con.values() for k in v})
    table("Concurrency: aggregate tok/s (256-token greedy outputs, 8 requests per level)", ["streams"] + names,
          [[str(c)] + [fmt(con[n].get(str(c), {}).get("aggregate_throughput_tok_s")) for n in names] for c in cl])
    tc = {n: load(d, "tool_calling.json") for n, d in runs}
    table("Tool calling (harness task set)", ["", *names], [["correct"] + [f"{tc[n].get('total_correct', '-')}/{tc[n].get('total_tasks', '-')}"
                                                                         for n in names]])


if __name__ == "__main__":
    main()
