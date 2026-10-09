#!/usr/bin/env python3
"""Long-context retrieval check (PLAN.md Phase 3 week 11: FP8 KV numerical stability at depth).

The check hides a random 6-digit pass key at a given depth inside filler text of the requested length. Then it asks the
model to repeat the key, greedy, with thinking off. This tests the chunked prefill (FlashInfer over a long FP8 or BF16
cache) and the decode attention kernel at depth.

  uv run python tests/passkey.py [--lens 16384 65536 131072] [--depths 0.1 0.5 0.9]
  uv run python tests/passkey.py --url http://127.0.0.1:8000 --lens 262000 --depths 0.5   # through the server
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.runtime.build import build_engine  # noqa: E402
from engine.runtime.scheduler import Request  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")


def build(tok, n_tokens, depth, key):
    unit = tok(FILLER, add_special_tokens=False).input_ids
    needle = tok(f" The pass key is {key}. Remember it. {key} is the pass key. ", add_special_tokens=False).input_ids
    question = "\n\nWhat is the pass key? Answer with the number only."
    overhead = len(tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)) + len(needle) + 16
    body = (unit * (n_tokens // len(unit) + 1))[: max(0, n_tokens - overhead)]
    at = int(len(body) * depth)
    text = tok.decode(body[:at] + needle + body[at:])
    ids = tok.apply_chat_template([{"role": "user", "content": text + question}], add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[16384, 65536, 131072])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--url", default=None, help="test a running server (engine/server) over HTTP instead of a local scheduler")
    a = ap.parse_args()
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    rng = random.Random(0)
    hits = 0
    if a.url:
        import requests
        print(f"server {a.url}")
        for L in a.lens:
            for d in a.depths:
                key = rng.randint(100000, 999999)
                ids = build(tok, L, d, key)
                r = requests.post(a.url + "/v1/completions", json={"prompt": ids, "max_tokens": 12, "temperature": 0, "cache_prompt": False},
                                  timeout=3600).json()
                ans = r["choices"][0]["text"].strip()
                ok = str(key) in re.sub(r"[^0-9]", "", ans) or str(key) in ans
                hits += ok
                print(f"  len {len(ids):6d} depth {d:.1f}: key {key} -> {ans!r:24s} {'OK' if ok else 'MISS'}  "
                      f"(TTFT {r['timings']['ttft_s']:6.1f} s)", flush=True)
        print(f"PASSKEY {hits}/{len(a.lens) * len(a.depths)}")
        return
    eng, _ = build_engine(path, spec="none", slots=1, max_seq_len=max(a.lens) + 64, checkpoints=0, decode_weights="checkpoint", boundary=None)
    for L in a.lens:
        for d in a.depths:
            key = rng.randint(100000, 999999)
            ids = build(tok, L, d, key)
            r = eng.run([Request(ids, max_new_tokens=12, eos_ids=(248046, 248044))])[0]
            ans = tok.decode(r.output, skip_special_tokens=True).strip()
            ok = str(key) in re.sub(r"[^0-9]", "", ans) or str(key) in ans
            hits += ok
            print(f"  len {len(ids):6d} depth {d:.1f}: key {key} -> {ans!r:24s} {'OK' if ok else 'MISS'}  "
                  f"(TTFT {r.t_first - r.t_submit:6.1f} s)", flush=True)
    print(f"PASSKEY {hits}/{len(a.lens) * len(a.depths)}")


if __name__ == "__main__":
    main()
