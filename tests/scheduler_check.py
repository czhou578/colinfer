#!/usr/bin/env python3
"""End-to-end checks of engine/runtime/scheduler.py on the real model (needs ~40 GB of GPU memory).

The reference is a plain-decode scheduler (no speculation), with each request alone. The MTP scheduler must give the
same greedy tokens:
  1. each request alone (batch width 1)
  2. all four together (widths 3 -> 1 as they finish, a 4th request queued)
  3. together with a sampled request and a 5000-token chunked prefill (sampled cycle graphs, masked slots)
More checks:
  4. seeded sampling at T=0.8: the MTP scheduler emits exactly what plain decode samples, alone and next to one or two
     other requests (position-keyed draws, engine/spec/accept.py)
  5. multi-turn: turn 2 restores the end-of-turn-1 checkpoint (the stop-token cut keeps it a prefix)
  6. a second conversation with the same long system prompt starts while the first one still decodes. It restores the
     system-prompt checkpoint, with a copy of its KV prefix into another slot. Its greedy output matches a run without
     checkpoints.
  7. a client sends the reply back without its reasoning (as Hermes Agent does). Turn 2 then restores the snapshot at
     the start of turn 1's reply, not only the start of turn 1's last message. The output of turn 2 is compared with a
     run without checkpoints.
  8. speed: single-slot MTP tok/s per prompt (Phase 4 numbers: 32.4 tok/s mean at T=0)
   uv run python tests/scheduler_check.py
"""
import time

import torch

from transformers import AutoTokenizer

from engine.runtime.build import build_engine
from engine.runtime.scheduler import Request, Scheduler
from engine.weights.loader import resolve

EOS = (248046, 248044)
IM_START = 248045
CODE = '''def parse_config(path):
    with open(path) as f:
        data = json.load(f)
    config = Config()
    config.name = data["name"]
    config.version = data["version"]
    config.author = data["author"]
    config.license = data["license"]
    config.description = data["description"]
    return config
'''
PROMPTS = {
    "code-edit": "Rewrite this function so every field access uses data.get(key, default) with a sensible default, "
                 "and add type hints. Return only the code.\n\n" + CODE,
    "json": "Convert this list to a JSON array of objects with keys name, role and team: Alice engineer platform; "
            "Bob designer web; Carol manager platform; Dave engineer infra; Erin analyst data; Frank engineer web.",
    "code-gen": "Write a Python class LRUCache with get and put methods, O(1) each, using an OrderedDict. Include docstrings.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
}


def chat_ids(tok, msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--only-alone", action="store_true", help="just the single-request speeds")
    ap.add_argument("--checkpoint-weights", action="store_true", help="decode the FP8 projections instead of the INT6 / INT5 copies")
    a = ap.parse_args()
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.perf_counter()
    # the speculative scheduler: the server's engine at 32k slots, with the checkpoint's drafter and no suffix drafts
    spec, _ = build_engine(path, max_seq_len=32768, k=a.k, drafter_weights="none", suffix_drafts=0,
                           decode_weights="checkpoint" if a.checkpoint_weights else "int", boundary=IM_START)
    ref = Scheduler(spec.model, n_slots=1, max_seq_len=8192, n_checkpoints=0)  # plain decode, one request at a time
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
    if a.only_alone:
        print("SCHEDULER CHECK", "PASSED" if ok else "FAILED")
        return

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

    # 4. seeded sampling: speculative output = plain sampling, at any batch width
    kw = dict(temperature=0.8, top_p=0.95, top_k=20, seed=1234, max_new_tokens=120)
    plain = ref.run([R(prompts[3], **kw)])[0].output
    a1 = spec.run([R(prompts[3], **kw)])[0].output
    a2 = spec.run([R(prompts[1], temperature=0.5, seed=9), R(prompts[3], **kw)])[1].output
    a3 = spec.run([R(prompts[0]), R(prompts[3], **kw), R(prompts[2], temperature=1.0, seed=5)])[1].output
    rep = plain == a1 == a2 == a3
    ok &= rep
    print(f"[seeded] T=0.8 plain decode vs MTP alone / next to one / next to two: "
          f"{'IDENTICAL' if rep else 'DIFFERENT'} ({len(a1)} tok; agree {[sum(x == y for x, y in zip(plain, o)) for o in (a1, a2, a3)]})")

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

    # 7. the reply comes back without its reasoning: turn 2 restores the start of turn 1's reply
    def think_ids(msgs):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="medium", tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    table = "Quarterly figures: " + " ".join(f"Q{i % 4 + 1} {2000 + i // 4}: revenue {900 + 13 * i}, net income {40 + 5 * i}." for i in range(120))
    m1 = [{"role": "system", "content": "You are a research agent."}, {"role": "user", "content": table + "\nWhich year grew the most?"}]
    p1 = think_ids(m1)
    r1 = spec.run([R(p1, max_new_tokens=60)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True).split("</think>")[-1].strip()
    p2 = think_ids(m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Now give the total revenue."}])
    reply_start = max(i for i, t in enumerate(p1) if t == IM_START)
    r2 = spec.run([R(p2, max_new_tokens=40)])[0]
    # the same request without checkpoints, with the prefill chunk boundaries of turn 1 and the restored turn 2 (other
    # boundaries change the prefill numerics, and a near tie can then change the greedy tokens)
    saved, spec.ckpts, spec.free_bufs = spec.ckpts, [], []
    r2b = spec.submit(R(p2, max_new_tokens=40))
    spec._admit()
    spec.slots[r2b.slot].splits = [reply_start]  # turn 1's only split: its first message is shorter than 256 tokens
    while spec.busy():
        spec.step()
    spec.ckpts, spec.free_bufs = saved, [i for i in range(len(spec.ring_h)) if i not in {c.buf for c in saved}]
    same = r2.output == r2b.output
    ok &= r2.reused == reply_start and same
    print(f"[reply sent back changed] turn 1 last message {reply_start - max(i for i, t in enumerate(p1[:reply_start]) if t == IM_START)} tok; "
          f"turn 2 reused {r2.reused} (reply start {reply_start}), TTFT {r2.t_first - r2.t_submit:.3f} s vs "
          f"{r2b.t_first - r2b.t_submit:.3f} s from scratch; greedy output {'IDENTICAL' if same else 'DIFFERENT'} "
          f"to the uncached run with the same chunk boundaries ({sum(x == y for x, y in zip(r2.output, r2b.output))}/{len(r2.output)})")

    m = spec.metrics
    print(f"[spec] drafted {m.drafted.get():.0f}, accepted {m.accepted.get():.0f} ({m.accepted.get() / max(m.drafted.get(), 1):.2f})")
    print("SCHEDULER CHECK", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
