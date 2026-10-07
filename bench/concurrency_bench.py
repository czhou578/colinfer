#!/usr/bin/env python3
"""Per-request and aggregate decode speed at 1..N concurrent streams, against a running server.

   uv run python bench/concurrency_bench.py --url http://127.0.0.1:8001 [--levels 1 2 3] [--max-tokens 256]

Each stream gets a different prompt (a mix of code, JSON and prose), greedy, with thinking off. The requests of a level
start together. Decode tok/s per request = (completion_tokens - 1) / (t_done - t_first), from the timings of the
server.
"""
import argparse
import concurrent.futures as cf
import time

import requests

PROMPTS = [
    "Write a Python class LRUCache with get and put methods, O(1) each, using an OrderedDict. Include docstrings.",
    "Convert to a JSON array of objects with keys name, role, team: Alice engineer platform; Bob designer web; Carol manager "
    "platform; Dave engineer infra; Erin analyst data; Frank engineer web; Grace engineer data; Heidi designer infra.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash map works, with a small C implementation using open addressing.",
]


def one(url, prompt, max_tokens, temperature):
    d = requests.post(url + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "min_tokens": max_tokens, "temperature": temperature,
        "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}, timeout=1200).json()
    t = d["timings"]
    return d["usage"]["completion_tokens"], t["decode_s"], t["ttft_s"], t["decode_tok_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    print("| streams | per-request decode tok/s (each) | mean | aggregate tok/s |")
    print("|---|---|---|---|")
    for n in a.levels:
        t0 = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda i: one(a.url, PROMPTS[i % len(PROMPTS)], a.max_tokens, a.temperature), range(n)))
        wall = time.time() - t0
        per = [r[3] for r in res]
        print(f"| {n} | {', '.join(f'{x:.1f}' for x in per)} | {sum(per) / n:.1f} | {sum(r[0] for r in res) / wall:.1f} |", flush=True)


if __name__ == "__main__":
    main()
