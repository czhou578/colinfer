#!/usr/bin/env python3
"""The long-context retrieval check of tests/passkey.py for the qwen4_exp family (Qwen3.8-Flash-Next), through its prefill
(engine/model/prefill_qwen4exp.py) and its decode graph, without the scheduler (milestone M2).

The check hides a random 6-digit pass key at a given depth inside filler text of the requested length (tests/passkey.py
build). One slot of the longest length (plus a margin) holds the prompt. The prefill runs in 4096-token chunks with an
n-gram feeder; then the decode graph produces up to 12 greedy tokens. Beyond 2051 tokens the QSA selection is sparse, so
this checks the indexer, the selection and the sparse attention at depth.

Usage (loads about 80 GB; take the GPU lock):
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tests/passkey_qwen4exp.py [--lens 16384 32768 65536] [--depths 0.1 0.5 0.9]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time

import torch


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--lens", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.server.chat import ChatFormat
    from engine.weights.ple import PleFeeder
    from tests.golden_qwen4exp import FEEDER_ROWS, Stepper
    from tests.passkey import build
    path, model = load_fast_qwen4exp(a.model)
    ple = model.ple_layer()
    model.feeder = PleFeeder(ple.table, cache_rows=FEEDER_ROWS, params=ple.params, image_token_id=model.cfg.image_token_id)
    fmt = ChatFormat.from_checkpoint(path)
    tok = fmt.tok
    step = Stepper(model, -(-(max(a.lens) + 256) // 16) * 16)
    rng = random.Random(0)
    hits, n = 0, 0
    for L in a.lens:
        for d in a.depths:
            key = rng.randint(100000, 999999)
            ids = build(fmt, L, d, key)
            torch.cuda.synchronize()
            t0 = time.time()
            logits = step.prefill(ids, a.chunk)[0]
            torch.cuda.synchronize()
            dt = time.time() - t0
            out = []
            for _ in range(12):
                t = int(logits.argmax())
                if t in fmt.eos_ids:
                    break
                out.append(t)
                logits = step(t)
            ans = tok.decode(out, skip_special_tokens=True).strip()
            ok = str(key) in re.sub(r"[^0-9]", "", ans)
            hits += ok
            n += 1
            print(f"  len {len(ids):6d} depth {d:.1f}: key {key} -> {ans!r:24s} {'OK' if ok else 'MISS'}  (prefill {dt:6.1f} s, "
                  f"{len(ids) / dt:.0f} tok/s)", flush=True)
    h, m, _, _ = model.feeder.stats()
    print(f"PASSKEY {hits}/{n} (feeder hit rate {h / max(1, h + m):.1%})")
    sys.exit(0 if hits == n else 1)


if __name__ == "__main__":
    main()
