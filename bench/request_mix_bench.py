#!/usr/bin/env python3
"""End-to-end speed on the 40-request mix (docs/history/phase6_progress.md sections 16-19). The prompts are those of
tools/drafter_data.py (seed 1: code, prose, Q&A, structured, half with thinking on), frozen as token ids in
tests/golden/prompts.json. The bench sends one request at a time, greedy, 256 tokens. It reports the completion tokens
per second of wall time (prefill included) per kind. It works against a running colinfer server or an SGLang server
(whose replies also give the accepted tokens per verify step).

   uv run python bench/request_mix_bench.py --port 8002                  # colinfer (python -m engine.server ...)
   uv run python bench/request_mix_bench.py --port 8010 --engine sglang
"""
import argparse
import collections
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--engine", choices=("colinfer", "sglang"), default="colinfer")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    from tests.golden import load_prompts
    if a.engine == "sglang":  # SGLang gets token ids and no template, so it needs the stop ids
        from engine.server.chat import ChatFormat
        stop_ids = list(ChatFormat.from_checkpoint().eos_ids)
    res = collections.defaultdict(lambda: [0, 0, 0.0])  # completion tokens, verify steps (SGLang), seconds
    for m in load_prompts()["mix"][:a.n]:
        x, kind = m["ids"], m["kind"]
        t0 = time.time()
        if a.engine == "colinfer":
            r = requests.post(f"http://127.0.0.1:{a.port}/v1/completions", json={"prompt": x, "temperature": 0, "max_tokens": a.max_tokens},
                              timeout=600).json()
            ct, vc = r["usage"]["completion_tokens"], 0
        else:
            r = requests.post(f"http://127.0.0.1:{a.port}/generate", json={"input_ids": x, "sampling_params": {
                "temperature": 0, "max_new_tokens": a.max_tokens, "stop_token_ids": stop_ids}}, timeout=600).json()
            ct, vc = r["meta_info"]["completion_tokens"], r["meta_info"].get("spec_verify_ct") or 0
        res[kind][0] += ct
        res[kind][1] += vc
        res[kind][2] += time.time() - t0
    tot = [0, 0, 0.0]
    print(f"{a.engine}: wall tok/s incl. prefill" + (", accepted tokens per verify step" if a.engine == "sglang" else ""))
    for kind, v in sorted(res.items()) + [("all", None)]:
        if v is None:
            v = tot
        else:
            tot = [p + q for p, q in zip(tot, v)]
        print(f"  {kind:8s} {v[0] / v[2]:6.1f} tok/s" + (f"   {v[0] / max(v[1], 1):.2f}" if a.engine == "sglang" else ""))


if __name__ == "__main__":
    main()
