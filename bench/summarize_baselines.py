#!/usr/bin/env python3
"""summarize_baselines.py -- condense model-benchmarks run directories into markdown tables.

Usage: python bench/summarize_baselines.py <run_dir_or_model_name> [...]
  A model name resolves to the newest run under ~/Projects/model-benchmarks/results/<name>/.
Reads latency.json, ttft_breakdown.json, deep_context.json, decode.json, concurrency.json and the
spec_* files written by core_runner.py; missing files are skipped.
"""
import glob
import json
import os
import sys

RESULTS = os.path.expanduser("~/Projects/model-benchmarks/results")


def resolve(arg):
    if os.path.isdir(arg) and os.path.exists(os.path.join(arg, "model_config.yml")):
        return arg
    runs = sorted(glob.glob(os.path.join(RESULTS, arg, "*")))
    if not runs:
        sys.exit(f"no run found for {arg}")
    return runs[-1]


def load(run, name):
    p = os.path.join(run, name)
    return json.load(open(p)) if os.path.exists(p) else None


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def summarize(run):
    name = os.path.basename(os.path.dirname(run.rstrip("/")))
    env = load(run, "environment.json") or {}
    srv = load(run, "resolved_server.json") or {}
    out = [f"### {name}  ({os.path.basename(run.rstrip('/'))})", ""]
    out.append(f"- server: `{g(srv, 'command_display', default='?')[:300]}`")
    out.append(f"- env: torch {env.get('torch_version')}, vllm {env.get('vllm_version')} (harness venv), gpu {env.get('gpu_name')}")
    summ = load(run, "summary.json") or {}
    out.append(f"- status: {summ.get('status')} {summ.get('error', '')}")
    out.append("")

    lat = load(run, "latency.json")
    if lat:
        out += ["| prompt tokens | TTFT median s | TTFT p95 s | prefill tok/s (avg) |", "|---|---|---|---|"]
        for k, v in lat.items():
            out.append(f"| {k} | {fmt(v.get('ttft_median_s'), 3)} | {fmt(v.get('ttft_p95_s'), 3)} | {fmt(v.get('prefill_tps_avg'), 0)} |")
        out.append("")
    dc = load(run, "deep_context.json")
    if dc:
        out += ["| context tokens | TTFT median s | TTFT p95 s | prefill tok/s |", "|---|---|---|---|"]
        for k, v in dc.items():
            tps = v.get("prefill_tps_avg") or v.get("prefill_tps_median") or (
                (v["requested_context_tokens"] / v["ttft_median_s"]) if v.get("ttft_median_s") else None)
            out.append(f"| {k} | {fmt(v.get('ttft_median_s'), 3)} | {fmt(v.get('ttft_p95_s'), 3)} | {fmt(tps, 0)} |")
        out.append("")
    dec = load(run, "decode.json")
    if dec:
        out += ["| output tokens | decode tok/s avg | median | peak | TTFT s |", "|---|---|---|---|---|"]
        for k, v in dec.items():
            out.append(f"| {k} | {fmt(v.get('tok_per_sec_avg'))} | {fmt(v.get('tok_per_sec_median'))} | {fmt(v.get('tok_per_sec_peak'))} | {fmt(v.get('ttft_s'), 3)} |")
        out.append("")
    for label in ("spec_enabled", "spec_disabled"):
        sp = load(run, f"{label}.json")
        if sp:
            out += [f"| {label}: output tokens | decode tok/s avg | median | peak |", "|---|---|---|---|"]
            for k, v in sp.items():
                out.append(f"| {k} | {fmt(v.get('tok_per_sec_avg'))} | {fmt(v.get('tok_per_sec_median'))} | {fmt(v.get('tok_per_sec_peak'))} |")
            out.append("")
    cmp_ = load(run, "spec_comparison.json")
    if cmp_:
        out.append("spec comparison: " + json.dumps(cmp_)[:600])
        out.append("")
    con = load(run, "concurrency.json")
    if con:
        out += ["| concurrency | aggregate tok/s | per-request tok/s | mean latency s | failed |", "|---|---|---|---|---|"]
        for k, v in g(con, "per_concurrency_level", default={}).items():
            per_req = v.get("per_request_tok_s_avg") or v.get("avg_tok_s_per_request") or (
                v["aggregate_throughput_tok_s"] / int(k) if v.get("aggregate_throughput_tok_s") else None)
            lat_mean = v.get("latency_avg_s") or v.get("avg_latency_s") or v.get("mean_latency_s")
            out.append(f"| {k} | {fmt(v.get('aggregate_throughput_tok_s'))} | {fmt(per_req)} | {fmt(lat_mean, 2)} | {v.get('n_failed')} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(summarize(resolve(a)))
        print()
