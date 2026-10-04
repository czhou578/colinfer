#!/usr/bin/env python3
"""End-to-end checks of engine/runtime/scheduler.py on the real model (needs ~40 GB of GPU memory).

Reference: a plain-decode scheduler (no speculation), each request alone. The MTP scheduler must give the same
greedy tokens:
  1. each request alone (batch width 1);
  2. all four together (widths 3 -> 1 as they finish, a 4th request queued);
  3. together with a sampled request and a 5000-token chunked prefill (sampled cycle graphs, masked slots);
plus
  4. seeded sampling: same seed alone and alongside other requests -> identical tokens;
  5. multi-turn: turn 2 restores the end-of-turn-1 checkpoint (stop-token cut keeps it a prefix);
  6. a second conversation with the same long system prompt, while the first is still decoding, restores the
     system-prompt checkpoint by copying its KV prefix into another slot; its greedy output matches a run
     without checkpoints;
  7. speed: single-slot MTP tok/s per prompt (Phase 4 numbers: 32.4 tok/s mean at T=0).
   uv run python tests/scheduler_check.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer  # noqa: E402

from engine.model.fast import load_fast_model, to_fast  # noqa: E402
from engine.runtime.scheduler import Request, Scheduler  # noqa: E402
from engine.spec.mtp import Mtp  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_check import PROMPTS  # noqa: E402

EOS = (248046, 248044)
IM_START = 248045


def chat_ids(tok, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def main():
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    model = to_fast(load_fast_model(path), kv_fp8=True)
    mtp = Mtp(model, path, fp8=True, draft_vocab=65536)
    t0 = time.perf_counter()
    ref = Scheduler(model, n_slots=1, max_seq_len=8192, n_checkpoints=0, selftest=True)
    spec = Scheduler(model, n_slots=3, max_seq_len=32768, n_checkpoints=32, mtp=mtp, k=3, selftest=False, boundary_token=IM_START)
    print(f"schedulers built in {time.perf_counter() - t0:.1f} s; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")
    ok = True
    names = list(PROMPTS)
    prompts = [chat_ids(tok, [{"role": "user", "content": PROMPTS[n]}]) for n in names]
    N = 200

    def R(p, **kw):
        return Request(p, max_new_tokens=kw.pop("max_new_tokens", N), eos_ids=EOS, **kw)

    want = [ref.run([R(p)])[0].output for p in prompts]

    # 1. alone
    for n, p, w in zip(names, prompts, want):
        r = spec.run([R(p)])[0]
        same = r.output == w
        ok &= same
        dt = r.t_done - r.t_first
        print(f"[alone] {n:9s} {'IDENTICAL' if same else 'DIFFERENT'} {len(r.output)} tok, {(len(r.output) - 1) / dt:5.1f} tok/s")

    # 2. all together (3 slots, the 4th waits)
    rs = spec.run([R(p) for p in prompts])
    same = [r.output == w for r, w in zip(rs, want)]
    ok &= all(same)
    print(f"[together] identical {sum(same)}/{len(same)}; slots {[r.slot for r in rs]}; "
          f"aggregate {sum(len(r.output) for r in rs) / (max(r.t_done for r in rs) - min(r.t_submit for r in rs)):.1f} tok/s")

    # 3. with a sampled request and a long chunked prefill in flight
    long_doc = torch.randint(1000, 150000, (5000,), generator=torch.Generator().manual_seed(0)).tolist()
    rs = spec.run([R(prompts[0]), R(long_doc, max_new_tokens=24), R(prompts[2], temperature=0.8, top_p=0.95, top_k=20, seed=7), R(prompts[3])])
    same = [rs[0].output == want[0], rs[3].output == want[3]]
    ok &= all(same)
    print(f"[mixed] greedy requests identical {sum(same)}/2 next to a sampled request and a 5000-token prefill")

    # 4. seeded sampling reproducibility
    kw = dict(temperature=0.8, top_p=0.95, top_k=20, seed=1234, max_new_tokens=120)
    a1 = spec.run([R(prompts[3], **kw)])[0].output
    a2 = spec.run([R(prompts[3], **kw)])[0].output
    a3 = spec.run([R(prompts[0]), R(prompts[3], **kw), R(prompts[1], temperature=0.5, seed=9)])[1].output
    rep = a1 == a2 == a3
    ok &= rep
    print(f"[seeded] same seed alone x2 and next to two others: {'IDENTICAL' if rep else 'DIFFERENT'} ({len(a1)} tok)")

    # 5. multi-turn: turn 2 restores the end of turn 1
    sys_msg = {"role": "system", "content": "You are a careful assistant. Background notes: " + " ".join(f"item {i}" for i in range(1500))}
    m1 = [sys_msg, {"role": "user", "content": "Summarize the background notes in one sentence."}]
    r1 = spec.run([R(chat_ids(tok, m1), max_new_tokens=60)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
    p2 = chat_ids(tok, m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Now count how many items there were."}])
    r2 = spec.run([R(p2, max_new_tokens=40)])[0]
    ok &= r2.reused >= len(chat_ids(tok, m1)) + len(r1.output) - 1
    print(f"[multi-turn] turn 1: {len(r1.prompt)} + {len(r1.output)} tok; turn 2: {len(p2)} tok, reused {r2.reused}, "
          f"TTFT {r2.t_first - r2.t_submit:.3f} s")

    # 6. shared system prompt, cross-slot restore while the first conversation is still decoding
    sys2 = {"role": "system", "content": "You are an agent. Tool manual: " + " ".join(f"rule {i}: be precise." for i in range(900))}
    pa = chat_ids(tok, [sys2, {"role": "user", "content": "Write a long essay about rivers."}])
    pb = chat_ids(tok, [sys2, {"role": "user", "content": "List three prime numbers."}])
    ra = spec.submit(R(pa, max_new_tokens=300))
    while not ra.output:
        spec.step()
    for _ in range(5):
        spec.step()
    rb = spec.submit(R(pb, max_new_tokens=40))
    while spec.busy():
        spec.step()
    saved, spec.ckpts, spec.free_bufs = spec.ckpts, [], []  # same request without checkpoints
    rb2 = spec.run([R(pb, max_new_tokens=40)])[0]
    spec.ckpts, spec.free_bufs = saved, [i for i in range(len(spec.ring_h)) if i not in {c.buf for c in saved}]
    sysn = next(i for i in range(1, len(pb)) if pb[i] == IM_START)
    same = rb.output == rb2.output
    ok &= rb.reused == sysn and rb.slot != ra.slot
    print(f"[shared system prompt] B reused {rb.reused} (system prompt {sysn}) in slot {rb.slot} while A decoded in slot {ra.slot}; "
          f"TTFT {rb.t_first - rb.t_submit:.3f} s vs {rb2.t_first - rb2.t_submit:.3f} s from scratch; greedy output "
          f"{'IDENTICAL' if same else 'DIFFERENT'} to the uncached run ({sum(x == y for x, y in zip(rb.output, rb2.output))}/{len(rb.output)})")

    m = spec.metrics
    print(f"[spec] drafted {m.drafted.get():.0f}, accepted {m.accepted.get():.0f} ({m.accepted.get() / max(m.drafted.get(), 1):.2f})")
    print("SCHEDULER CHECK", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
