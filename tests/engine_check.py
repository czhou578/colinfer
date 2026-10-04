#!/usr/bin/env python3
"""End-to-end checks of engine/runtime/engine.py on the real model (needs ~25 GB of GPU memory).

1. Slot isolation: a greedy request yields identical tokens alone and while other requests are
   prefilled (chunked) and decoded alongside it.
2. Multi-turn prefix reuse: turn 2 (= turn-1 prompt + turn-1 answer + new user message) served from
   a checkpoint vs from scratch; reports reused tokens, TTFT and token agreement.
   uv run python tests/engine_check.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.model.fast import load_fast_model, to_fast  # noqa: E402
from engine.runtime.engine import Engine, Request  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402

EOS = (248046, 248044)


def chat_ids(tok, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def main():
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    model = to_fast(load_fast_model(path), kv_fp8=True)
    eng = Engine(model, n_slots=3, max_seq_len=16384)
    ok = True

    # ---- 1. isolation ----
    pa = chat_ids(tok, [{"role": "user", "content": "List five prime numbers and explain what makes a number prime."}])
    alone = eng.run([Request(pa, max_new_tokens=64, eos_ids=EOS)])[0].output
    long_doc = torch.randint(1000, 200000, (5000,)).tolist()  # forces 3 prefill chunks interleaved with A's decode
    pb = chat_ids(tok, [{"role": "user", "content": "Write a haiku about autumn."}])
    reqs = eng.run([Request(pa, max_new_tokens=64, eos_ids=EOS), Request(long_doc, max_new_tokens=16),
                    Request(pb, max_new_tokens=48, temperature=0.8, top_p=0.95, seed=3, eos_ids=EOS)])
    same = reqs[0].output == alone
    ok &= same
    print(f"[isolation] request alone vs with 2 concurrent requests (one with a 5000-token chunked prefill): "
          f"{'IDENTICAL' if same else 'DIFFERENT'} ({len(alone)} tokens); slots used {[r.slot for r in reqs]}")
    print("   ", repr(tok.decode(alone[:40])))

    # ---- 2. multi-turn prefix reuse ----
    sys_msg = {"role": "system", "content": "You are a careful assistant. " + "Background notes: " + " ".join(f"item {i}" for i in range(1500))}
    m1 = [sys_msg, {"role": "user", "content": "Summarize the background notes in one sentence."}]
    p1 = chat_ids(tok, m1)
    r1 = eng.run([Request(p1, max_new_tokens=48, eos_ids=EOS)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
    m2 = m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Now count how many items there were."}]
    p2 = chat_ids(tok, m2)
    common = next((i for i in range(min(len(p1), len(p2))) if p1[i] != p2[i]), min(len(p1), len(p2)))
    r2 = eng.run([Request(p2, max_new_tokens=32, eos_ids=EOS)])[0]
    fresh = Engine.__new__(Engine)  # same engine, checkpoints disabled
    saved = eng.ckpts
    eng.ckpts = type(saved)(maxlen=0)
    r2f = eng.run([Request(p2, max_new_tokens=32, eos_ids=EOS)])[0]
    eng.ckpts = saved
    agree = sum(a == b for a, b in zip(r2.output, r2f.output))
    print(f"[multi-turn] turn-1 prompt {len(p1)} tok, turn-2 prompt {len(p2)} tok (shares {common} with turn 1)")
    print(f"   with checkpoint: reused {r2.reused} tok, TTFT {r2.t_first - r2.t_submit:.3f} s")
    print(f"   from scratch   : reused {r2f.reused} tok, TTFT {r2f.t_first - r2f.t_submit:.3f} s")
    print(f"   greedy tokens agreeing position-wise: {agree}/{len(r2.output)}; first 12: {r2.output[:12]} vs {r2f.output[:12]}")
    print("   ", repr(tok.decode(r2.output[:30])))
    ok &= r2.reused > 0
    print("ENGINE CHECK", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
