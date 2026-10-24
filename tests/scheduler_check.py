#!/usr/bin/env python3
"""End-to-end checks of engine/runtime/scheduler.py on the real model, as the server builds it (needs ~90 GB of memory).

The engine is the server's: 3 slots of 524288 tokens, thus the rope with YaRN x2 (engine/runtime/build.py load_model).
Plain decode and MTP share that model, so a wrong rope gives the same wrong tokens to both. Thus check 0 reads the rope
itself: the kernels use the YaRN frequencies, and the drafter's q norm carries the attention factor.

The reference is a plain-decode scheduler (no speculation), with each request alone. The MTP scheduler must give the
same greedy tokens:
  1. each request alone
  2. queued around a sampled request and a 5000-token chunked prefill (sampled cycle graphs, slot reuse)
More checks:
  3. seeded sampling at T=0.8: the MTP scheduler emits exactly what plain decode samples, alone and after one or two
     other requests (position-keyed draws, engine/spec/accept.py)
  4. multi-turn: turn 2 restores the end-of-turn-1 checkpoint (the stop-token cut keeps it a prefix)
  5. a second conversation with the same long system prompt, queued while the first one decodes, restores the
     system-prompt checkpoint. Its greedy output matches a run without checkpoints. It runs in another slot, with a
     copy of the KV prefix, so the first conversation (its long last message and its reply) stays cached.
  6. a client sends the reply back without its reasoning (as Hermes Agent does). Turn 2 then restores the snapshot at
     the start of turn 1's reply, not only the start of turn 1's last message. The output of turn 2 is compared with a
     run without checkpoints.
  7. the disk cache (engine/runtime/sessions.py): after a restart (memory forgotten, a new store on the same directory),
     turn 2 of a long conversation restores the end of turn 1 from disk. Its output must be the output of turn 2 after
     a restore from memory, and its TTFT far below a prefill from scratch.
  8. an image (engine/model/vision.py): the model reads the text drawn in it; the MTP scheduler gives the tokens of
     plain decode; turn 2 restores the prefix that holds the image and does not run the vision tower again.
  9. constrained decoding (engine/runtime/grammar.py): a JSON-schema reply parses and matches its schema, and MTP
     gives the tokens of plain decode under the same grammar; a forced tool call parses, with typed arguments; with
     reasoning on, the JSON follows the free reasoning.
 10. speed: MTP tok/s per prompt (Phase 4 numbers: 32.4 tok/s mean at T=0)
 11. the counts of each request (cycles, drafted and accepted tokens; the server's log and timings read them) sum to
     the metrics
   uv run python tests/scheduler_check.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time

import torch

from engine.model.qwen35 import rope_inv_freq, yarn_attention_factor
from engine.runtime.build import build_engine
from engine.runtime.grammar import Grammars
from engine.runtime.scheduler import Request, Scheduler
from engine.runtime.sessions import SessionStore
from engine.server.chat import ChatFormat, parse_tool_call
from engine.server.images import ImageProcessor, expand
from engine.weights.direct import read_tensors, release_buffers
from engine.weights.loader import MODEL, resolve, weight_map

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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--only-alone", action="store_true", help="just the single-request speeds")
    ap.add_argument("--checkpoint-weights", action="store_true", help="decode the FP8 projections instead of the INT6 / INT5 copies")
    a = ap.parse_args()
    fmt = ChatFormat.from_checkpoint()
    tok, EOS, IM_START = fmt.tok, fmt.eos_ids, fmt.im_start

    def chat_ids(msgs):
        return fmt.render(msgs, enable_thinking=False)
    t0 = time.perf_counter()
    # the speculative scheduler: the server's engine (its slots and its slot length, thus its YaRN rope), with the
    # checkpoint's drafter and no suffix drafts
    spec, _ = build_engine(k=a.k, drafter_weights="none", suffix_drafts=0,
                           decode_weights="checkpoint" if a.checkpoint_weights else "int", boundary=IM_START)
    ref = Scheduler(spec.model, n_slots=1, max_seq_len=8192, n_checkpoints=0)  # plain decode on the same model, one request at a time
    print(f"schedulers built in {time.perf_counter() - t0:.1f} s; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")
    ok = True

    # 0. the rope of the served model: the kernels read the YaRN frequencies (apply_yarn runs before to_fast), and the
    # drafter's q norm carries the attention factor (fold_yarn_scale)
    cfg, da = spec.model.cfg, spec.mtp.layer.self_attn
    inv = rope_inv_freq(cfg, da.inv_freq.device)
    # the factor first: inv comes from the same config as the kernels' frequencies, so it alone cannot see a lost apply_yarn
    rope = cfg.yarn_factor > 1 and cfg.yarn_factor == spec.max_seq_len / cfg.max_position_embeddings
    rope = rope and torch.equal(da.inv_freq, inv) and all(torch.equal(x.self_attn.inv_freq, inv) for x in spec.model.layers
                                                          if x.block_type == "full_attention")
    path, qn = resolve(MODEL), "mtp.layers.0.self_attn.q_norm.weight"
    raw = read_tensors(os.path.join(path, weight_map(path)[qn]), [qn], da.inv_freq.device)[qn].to(torch.bfloat16)
    release_buffers()  # the direct reader pins 384 MB of staging buffers for a read; the check does not need them again
    f, r = yarn_attention_factor(cfg), cfg.rotary_dim
    fold = torch.equal(da.q_norm.weight[:r], raw[:r] if f == 1 else (f * (1 + raw[:r].float()) - 1).to(raw.dtype))
    ok &= rope and fold
    print(f"[rope] YaRN x{cfg.yarn_factor:g} at {spec.max_seq_len} tokens per slot: kernel frequencies "
          f"{'RIGHT' if rope else 'WRONG'}; drafter q norm x{f:.4f} {'FOLDED' if fold else 'NOT FOLDED'}")

    spec_done = []  # the requests that the MTP scheduler ran since its metrics were last zeroed
    counts_ok = [True]
    submit, reset = spec.submit, spec.reset

    def counted_submit(req):
        spec_done.append(req)
        return submit(req)

    def check_counts():
        """The counts of each request sum to the metrics, and each request with decode steps counts them."""
        m = spec.metrics
        good = (sum(x.drafted for x in spec_done) == m.drafted.total() and sum(x.accepted for x in spec_done) == m.accepted.total()
                and all(x.cycles >= 1 and x.accepted <= x.drafted and len(x.output) <= 1 + x.cycles + x.accepted
                        for x in spec_done if len(x.output) > 1))
        counts_ok[0] &= good

    def counted_reset():
        check_counts()
        spec_done.clear()
        reset()
    spec.submit, spec.reset = counted_submit, counted_reset
    names = list(PROMPTS)
    prompts = [chat_ids([{"role": "user", "content": PROMPTS[n]}]) for n in names]
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
        sys.exit(0 if ok else 1)

    # 2. queued around a sampled request and a long chunked prefill
    long_doc = torch.randint(1000, 150000, (5000,), generator=torch.Generator().manual_seed(0)).tolist()
    rs = spec.run([R(prompts[0]), R(long_doc, max_new_tokens=24), R(prompts[2], temperature=0.8, top_p=0.95, top_k=20, seed=7), R(prompts[3])])
    same = [rs[0].output == want[0], rs[3].output == want[3]]
    ok &= all(same)
    print(f"[queued] greedy requests identical {sum(same)}/2 around a sampled request and a 5000-token prefill")

    # 3. seeded sampling: speculative output = plain sampling, whatever ran before
    kw = dict(temperature=0.8, top_p=0.95, top_k=20, seed=1234, max_new_tokens=120)
    plain = ref.run([R(prompts[3], **kw)])[0].output
    a1 = spec.run([R(prompts[3], **kw)])[0].output
    a2 = spec.run([R(prompts[1], temperature=0.5, seed=9), R(prompts[3], **kw)])[1].output
    a3 = spec.run([R(prompts[0]), R(prompts[3], **kw), R(prompts[2], temperature=1.0, seed=5)])[1].output
    rep = plain == a1 == a2 == a3
    ok &= rep
    print(f"[seeded] T=0.8 plain decode vs MTP alone / after one / after two: "
          f"{'IDENTICAL' if rep else 'DIFFERENT'} ({len(a1)} tok; agree {[sum(x == y for x, y in zip(plain, o)) for o in (a1, a2, a3)]})")

    # 4. multi-turn: turn 2 restores the end of turn 1
    sys_msg = {"role": "system", "content": "You are a careful assistant. Background notes: " + " ".join(f"item {i}" for i in range(1500))}
    m1 = [sys_msg, {"role": "user", "content": "Summarize the background notes in one sentence."}]
    r1 = spec.run([R(chat_ids(m1), max_new_tokens=60)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
    p2 = chat_ids(m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Now count how many items there were."}])
    r2 = spec.run([R(p2, max_new_tokens=40)])[0]
    ok &= r2.reused >= len(chat_ids(m1)) + len(r1.output) - 1
    print(f"[multi-turn] turn 1: {len(r1.prompt)} + {len(r1.output)} tok; turn 2: {len(p2)} tok, reused {r2.reused}, "
          f"TTFT {r2.t_first - r2.t_submit:.3f} s")

    # 5. shared system prompt: a second conversation, queued while the first one decodes
    sys2 = {"role": "system", "content": "You are an agent. Tool manual: " + " ".join(f"rule {i}: be precise." for i in range(900))}
    notes = " ".join(f"River {i} is {40 + 7 * i} km long and crosses {i % 5 + 1} towns." for i in range(90))
    pa = chat_ids([sys2, {"role": "user", "content": "Write a long essay about rivers. Notes: " + notes}])
    pb = chat_ids([sys2, {"role": "user", "content": "List three prime numbers."}])
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
    kept = rb.slot != ra.slot and spec.slots[ra.slot].tokens[:len(pa)] == pa
    ok &= rb.reused == sysn and kept
    print(f"[shared system prompt] B (queued while A decoded) reused {rb.reused} (system prompt {sysn}); "
          f"admission to first token {rb.t_first - rb.t_admit:.3f} s vs {rb2.t_first - rb2.t_admit:.3f} s from scratch; greedy output "
          f"{'IDENTICAL' if same else 'DIFFERENT'} to the uncached run ({sum(x == y for x, y in zip(rb.output, rb2.output))}/{len(rb.output)}); "
          f"A ({len(pa)} + {len(ra.output)} tok) {'stays cached in its slot' if kept else 'LOST ITS SLOT'}")

    # 6. the reply comes back without its reasoning: turn 2 restores the start of turn 1's reply
    def think_ids(msgs):
        return fmt.render(msgs, reasoning_effort="medium")
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

    # 7. the disk cache across a restart
    d = tempfile.mkdtemp(prefix="sessions-check-", dir=os.path.expanduser("~/.cache/colinfer"))
    try:
        notes = {"role": "system", "content": "You are a careful assistant. Field log: " + " ".join(
            f"day {i}: sensor {i % 7} read {(37 * i) % 101} units." for i in range(1400))}
        m1 = [notes, {"role": "user", "content": "Which sensor appears most often? Answer in one sentence."}]
        p1 = chat_ids(m1)

        def turn2(r1):
            ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
            return chat_ids(m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "And on which day was the highest reading?"}])
        spec.reset()
        spec.store = SessionStore(spec.state, spec.mst, spec.fingerprint, 20 * 10**9, d)
        r1 = spec.run([R(p1, max_new_tokens=60)])[0]
        spec.store.flush()
        spec.reset()  # a restart: nothing in memory, and a new store that finds the files
        spec.store = SessionStore(spec.state, spec.mst, spec.fingerprint, 20 * 10**9, d)
        p2 = turn2(r1)
        rd = spec.run([R(p2, max_new_tokens=60)])[0]
        disk_tok = spec.metrics.disk_tokens.get()
        spec.store.flush()
        spec.store = None
        spec.reset()  # the reference: the same two turns, turn 2 restored from memory
        r1b = spec.run([R(p1, max_new_tokens=60)])[0]
        rm = spec.run([R(turn2(r1b), max_new_tokens=60)])[0]
        spec.reset()
        rc = spec.run([R(p2, max_new_tokens=60)])[0]  # and from scratch, for the time
        same = r1b.output == r1.output and rd.output == rm.output and rd.reused == rm.reused == disk_tok == len(p1) + len(r1.output) - 1
        ok &= same
        print(f"[disk cache] turn 1 {len(p1)} + {len(r1.output)} tok; after a restart turn 2 ({len(p2)} tok) restored {rd.reused} from disk: "
              f"TTFT {rd.t_first - rd.t_submit:.3f} s (from memory {rm.t_first - rm.t_submit:.3f} s, from scratch {rc.t_first - rc.t_submit:.3f} s); "
              f"output {'IDENTICAL' if rd.output == rm.output else 'DIFFERENT'} to the memory restore "
              f"({sum(x == y for x, y in zip(rd.output, rm.output))}/{len(rm.output)})")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 8. an image
    if spec.vision is not None:
        from PIL import Image as PILImage
        from PIL import ImageDraw, ImageFont
        pic, lines = PILImage.new("RGB", (1024, 768), "white"), ["Quarterly report", "Revenue: 4,210", "Costs: 3,005", "Margin: 28.6%"]
        for i, line in enumerate(lines):
            ImageDraw.Draw(pic).text((60, 60 + 110 * i), line, fill="black", font=ImageFont.load_default(size=40))
        buf = io.BytesIO()
        pic.save(buf, "PNG")
        proc = ImageProcessor(resolve(MODEL))
        ref.vision, ref.image_pad = spec.vision, spec.image_pad
        V = spec.model.cfg.vocab_size

        def image_turn(msgs):
            im = proc.load(buf.getvalue())
            return expand(fmt.render(msgs, enable_thinking=False), [im], spec.image_pad, V), [im]
        m1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe every line of the image."}]}]
        pi, ims = image_turn(m1)
        spec.reset()
        ri = spec.run([R(pi, max_new_tokens=60, images=ims)])[0]
        pi2, ims2 = image_turn(m1)
        rp = ref.run([R(pi2, max_new_tokens=60, images=ims2)])[0]
        ans = tok.decode([t for t in ri.output if t not in EOS], skip_special_tokens=True)
        calls = [0]
        encode = spec.vision.encode
        spec.vision.encode = lambda im: calls.__setitem__(0, calls[0] + 1) or encode(im)
        p3, ims3 = image_turn(m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "What is the margin?"}])
        r3 = spec.run([R(p3, max_new_tokens=20, images=ims3)])[0]
        spec.vision.encode = encode
        same = ri.output == rp.output
        ok &= same and all(line in ans for line in lines) and r3.reused == len(pi) + len(ri.output) - 1 and calls[0] == 0
        print(f"[image] {ims[0].n} image tokens; answer {ans!r}; MTP vs plain decode {'IDENTICAL' if same else 'DIFFERENT'}; "
              f"turn 2 reused {r3.reused} of {len(p3)} with {calls[0]} vision tower runs, TTFT {r3.t_first - r3.t_submit:.3f} s "
              f"(turn 1: {ri.t_first - ri.t_submit:.3f} s)")

    # 9. constrained decoding
    gr = Grammars(fmt, spec.model.cfg.vocab_size)
    person = {"type": "object", "properties": {"name": {"type": "string"}, "birth_year": {"type": "integer"},
                                               "fields": {"type": "array", "items": {"type": "string"}}},
              "required": ["name", "birth_year", "fields"], "additionalProperties": False}
    pj = chat_ids([{"role": "user", "content": "Give a short record of Ada Lovelace: her name, birth year and fields of work."}])
    spec.reset()
    rj = spec.run([R(pj, max_new_tokens=300, constraint=gr.json(person, think=False))])[0]
    rjp = ref.run([R(pj, max_new_tokens=300, constraint=gr.json(person, think=False))])[0]
    rjf = spec.run([R(pj, max_new_tokens=300)])[0]
    try:
        obj = json.loads(tok.decode([t for t in rj.output if t not in EOS], skip_special_tokens=True))
        valid = set(obj) == set(person["properties"]) and isinstance(obj["birth_year"], int) and all(isinstance(x, str) for x in obj["fields"])
    except (ValueError, KeyError, TypeError):
        obj, valid = None, False
    same = rj.output == rjp.output
    ok &= valid and same and rj.finish_reason == "stop"
    tps = lambda r: (len(r.output) - 1) / max(r.t_done - r.t_first, 1e-9)  # noqa: E731
    print(f"[constrained] JSON schema: {json.dumps(obj)}; {'VALID' if valid else 'INVALID'}; "
          f"MTP vs plain decode {'IDENTICAL' if same else 'DIFFERENT'}; "
          f"{tps(rj):.1f} tok/s (unconstrained reply: {tps(rjf):.1f} tok/s, {tok.decode(rjf.output[:12])!r}...)")
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
    pt = fmt.render([{"role": "user", "content": "Tell me a joke about Paris."}], tools=tools, enable_thinking=False)
    rt = spec.run([R(pt, max_new_tokens=200, constraint=gr.tools(tools, names=["get_weather"], single=True))])[0]
    text = tok.decode(rt.output, skip_special_tokens=False)
    body = text.split("<tool_call>")[-1].split("</tool_call>")[0] if "<tool_call>" in text else ""
    call = parse_tool_call(body, tools)
    good = call is not None and call["name"] == "get_weather" and isinstance(json.loads(call["arguments"]).get("city"), str)
    ok &= good
    print(f"[constrained] forced tool call on a prompt that needs none: {call}; {'VALID' if good else 'INVALID'}")
    pr = fmt.render([{"role": "user", "content": "Give a short record of Ada Lovelace: her name, birth year and fields of work."}],
                    reasoning_effort="low")
    rr = spec.run([R(pr, max_new_tokens=2000, constraint=gr.json(person, think=fmt.opens_in_reasoning(pr)))])[0]
    out = tok.decode([t for t in rr.output if t not in EOS], skip_special_tokens=False)
    try:
        robj = json.loads(out.split("</think>")[-1])
        rvalid = set(robj) == set(person["properties"]) and "</think>" in out
    except ValueError:
        robj, rvalid = None, False
    ok &= rvalid
    print(f"[constrained] with reasoning: {len(out.split('</think>')[0])} characters of reasoning, then {json.dumps(robj)}; "
          f"{'VALID' if rvalid else 'INVALID'}")

    m = spec.metrics
    check_counts()
    ok &= counts_ok[0]
    print(f"[spec] request counts (cycles, drafted, accepted) {'MATCH' if counts_ok[0] else 'DO NOT MATCH'} the metrics")
    print(f"[spec] drafted {m.drafted.total():.0f}, accepted {m.accepted.total():.0f} ({m.accepted.total() / max(m.drafted.total(), 1):.2f}); "
          + ", ".join(f"{src}: {m.accepted.get(source=src) / max(m.drafted.get(source=src), 1):.2f} of {m.drafted.get(source=src):.0f}"
                      for src in ("mtp", "suffix") if m.drafted.get(source=src)))
    print("SCHEDULER CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
