#!/usr/bin/env python3
"""Long-context retrieval check (PLAN.md Phase 3 week 11: FP8 KV numerical stability at depth).

Hides a random 6-digit pass key at a given depth inside filler text of the requested length and asks
the model to repeat it, greedy, thinking off. Exercises chunked prefill (FlashInfer over a long FP8 or
BF16 cache) and the decode attention kernel at depth.

  uv run python tests/passkey.py [--lens 16384 65536 131072] [--depths 0.1 0.5 0.9] [--kv-fp8]
"""
import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.model.fast import load_fast_model, to_fast  # noqa: E402
from engine.runtime.scheduler import Request, Scheduler  # noqa: E402
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
    ap.add_argument("--kv-fp8", action="store_true")
    a = ap.parse_args()
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    eng = Scheduler(to_fast(load_fast_model(path), kv_fp8=a.kv_fp8), n_slots=1, max_seq_len=max(a.lens) + 64, n_checkpoints=0)
    rng = random.Random(0)
    hits = 0
    print(f"KV {'fp8' if a.kv_fp8 else 'bf16'}")
    for L in a.lens:
        for d in a.depths:
            key = rng.randint(100000, 999999)
            ids = build(tok, L, d, key)
            t0 = time.perf_counter()
            r = eng.run([Request(ids, max_new_tokens=12, eos_ids=(248046, 248044))])[0]
            ans = tok.decode(r.output, skip_special_tokens=True).strip()
            ok = str(key) in re.sub(r"[^0-9]", "", ans) or str(key) in ans
            hits += ok
            print(f"  len {len(ids):6d} depth {d:.1f}: key {key} -> {ans!r:24s} {'OK' if ok else 'MISS'}  "
                  f"(TTFT {r.t_first - r.t_submit:6.1f} s)", flush=True)
    print(f"PASSKEY {hits}/{len(a.lens) * len(a.depths)}")


if __name__ == "__main__":
    main()
