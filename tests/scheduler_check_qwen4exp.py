#!/usr/bin/env python3
"""The end-to-end check of Qwen3.8-Flash-Next (qwen4_exp) through the scheduler, as the server builds it.

build_engine(model=the Flash-Next checkpoint) takes the family's defaults (engine/model/family.py): 2 slots of 262,144
tokens, 16 checkpoints, speculation with the MTP drafter (engine/spec/mtp4.py: up to 3 drafts, a cycle of 1, 2 or 3,
the suffix-match drafts), prefill chunks of 4096 tokens, the n-gram rows through a PleFeeder with a host row cache. A
second scheduler on the same model, with one slot and no drafter, is the reference of plain decode ("spec off"). Its
checkpoints follow the same rules, so both prefill the same chunks. The checks:
  0. the memory: the fall of MemAvailable from the start of this process to the end of the build, against the state
     of the slots and of the drafter and the n-gram row cache (the rest is QWEN4EXP_FIXED_BYTES), and after the checks
     (the growth)
  1. greedy decode of the three prompts of tests/data/qwen4exp_golden.json with speculation: the tokens equal the
     reference's, or the first divergence is a near-tie of the reference (tests/golden_qwen4exp.py NEAR_TIE), and they
     equal plain decode's. The prompt goes through the prefill here (golden_qwen4exp.py feeds it token by token), and
     each cycle reads the n-gram rows of its k + 1 rows through before_replay
  2. multi-turn: turn 2 restores the end-of-turn-1 checkpoint; its time to first token against a prefill from scratch;
     both turns equal plain decode
  3. a second slot: another conversation runs in the other slot, and turn 3 of the first one restores from memory
  4. the disk cache (engine/runtime/sessions.py): two other conversations evict a long conversation from both slots;
     its next turn restores from disk (the drafter's rows and hidden included), with the output of a restore from memory
  5. a prompt of more than 4096 tokens (two prefill chunks, the second with prev2): the model finds a code that the
     first chunk holds; the output equals plain decode
  6. the speed: two generations of --tokens tokens (code, prose) with and without speculation: the tokens per second,
     the accepted drafts per cycle and the draft lengths that the scheduler chose; the outputs are equal
  7. the fingerprint of the cached state names the n-gram table file and the drafter "checkpoint"

Usage (loads about 80 GB of weights; take the GPU lock, MemAvailable >= 100 GB):
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock uv run python tests/scheduler_check_qwen4exp.py [--sessions-dir DIR] [--json FILE]
    [--tokens N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
import time

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise KeyError("MemAvailable")


AVAIL0 = mem_available()  # before torch and the CUDA context: the start of a fresh process

import torch  # noqa: E402

from engine.model.family import NGRAM_CACHE_GB, family_of  # noqa: E402
from engine.runtime.build import build_engine  # noqa: E402
from engine.runtime.scheduler import Request, Scheduler  # noqa: E402
from engine.runtime.sessions import SessionStore  # noqa: E402
from engine.server.chat import ChatFormat  # noqa: E402
from engine.weights.loader import resolve  # noqa: E402
from tests.golden_qwen4exp import FILE as GOLDEN  # noqa: E402
from tests.golden_qwen4exp import NEAR_TIE, STOP  # noqa: E402


def golden_verdict(got: list[int], want: dict) -> tuple[bool, str]:
    """The near-tie rule of tests/golden_qwen4exp.py: equal tokens, or a first divergence where the reference's margin is
    below NEAR_TIE."""
    ref = want["tokens"]
    if got == ref:
        return True, f"{len(got)} tokens EQUAL"
    i = next((j for j, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
    m = want["margins"][i] if i < len(want["margins"]) else float("nan")
    near = m < NEAR_TIE
    return near, (f"first divergence at token {i} of {len(ref)} (engine {got[i] if i < len(got) else None}, reference "
                  f"{ref[i] if i < len(ref) else None}, reference margin {m:.3f}) -> {'near-tie' if near else 'REAL DIVERGENCE'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sessions-dir", help="check 4 writes its session files into this new directory and keeps them (default: a temporary one)")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--tokens", type=int, default=256, help="the tokens of each generation of the speed check")
    a = ap.parse_args()
    path = resolve(a.model)
    fam = family_of(path)
    fmt = ChatFormat.from_checkpoint(path)
    tok, EOS, IM_START = fmt.tok, fmt.eos_ids, fmt.im_start
    res: dict = {"mem_available_start_gb": round(AVAIL0 / 1e9, 2)}
    ok = True

    def chat_ids(msgs, **kw):
        return fmt.render(msgs, **({"enable_thinking": False} | kw))

    def R(p, n=64, **kw):
        return Request(p, max_new_tokens=n, eos_ids=EOS, **kw)

    def ttft(r):
        return round(r.t_first - r.t_submit, 3)

    def tps(r):
        return round((len(r.output) - 1) / max(r.t_done - r.t_first, 1e-9), 1)

    def per_cycle(r):
        return round(len(r.output[1:]) / max(r.cycles, 1), 2)  # the first token comes from the prefill

    def same(x, y):
        return x.output == y.output

    def agree(x, y):
        return f"{sum(p == q for p, q in zip(x.output, y.output))}/{len(y.output)}"

    # ---- the build, and check 0: the startup memory
    t0 = time.perf_counter()
    sched, phases = build_engine(path, boundary=IM_START)
    torch.cuda.synchronize()
    startup_s = time.perf_counter() - t0
    avail1 = mem_available()
    d = fam.defaults
    assert sched.mtp is not None, "the family's default is speculation on"
    state = fam.state_bytes(path, sched.n_slots, sched.max_seq_len, "mtp")
    drafter_state = state - fam.state_bytes(path, sched.n_slots, sched.max_seq_len, "none")
    cache = fam.host_bytes(NGRAM_CACHE_GB)
    ring = sum(r.nbytes for r in sched.ring) + sched.ring_h.nbytes
    fall = AVAIL0 - avail1
    res.update(startup_s=round(startup_s, 1), phases=phases, slots=sched.n_slots, max_seq_len=sched.max_seq_len, checkpoints=sched.n_checkpoints,
               k=sched.k, k_opts=sched.k_opts, kmax=sched.kmax, cycle_ms_startup={str(kk): round(v * 1e3, 1) for kk, v in sched.cycle_s.items()},
               mem_fall_gb=round(fall / 1e9, 2), state_gb=round(state / 1e9, 2), drafter_state_gb=round(drafter_state / 1e9, 3),
               drafter_weights_gb=round(sched.mtp.nbytes() / 1e9, 2), ngram_cache_gb=round(cache / 1e9, 2), ring_gb=round(ring / 1e9, 2),
               fixed_measured_gb=round((fall - state - cache) / 1e9, 2), fixed_constant_gb=round(fam.fixed_bytes / 1e9, 2),
               allocated_gb=round(torch.cuda.memory_allocated() / 1e9, 2), reserved_gb=round(torch.cuda.memory_reserved() / 1e9, 2))
    print(f"[startup] {startup_s:.1f} s {phases}; {sched.n_slots} slots x {sched.max_seq_len} tokens, {sched.n_checkpoints} checkpoints "
          f"(defaults {d}); speculation k = {sched.k}, options {sched.k_opts} (+ {sched.kmax} for suffix drafts), cycle times at startup "
          f"{res['cycle_ms_startup']} ms; MemAvailable fell {fall / 1e9:.2f} GB: the slot state {state / 1e9:.2f} GB (the drafter's "
          f"{drafter_state / 1e9:.3f} GB of it), the n-gram row cache {cache / 1e9:.2f} GB, the rest {(fall - state - cache) / 1e9:.2f} GB "
          f"(QWEN4EXP_FIXED_BYTES {fam.fixed_bytes / 1e9:.2f} GB; the drafter's weights and head {res['drafter_weights_gb']:.2f} GB and the "
          f"ring {ring / 1e9:.2f} GB of it); torch allocated {res['allocated_gb']:.2f} GB, reserved {res['reserved_gb']:.2f} GB", flush=True)
    m = sched.model
    ref = Scheduler(m, n_slots=1, max_seq_len=32768, n_checkpoints=sched.n_checkpoints, boundary_token=IM_START, prefill_fn=fam.prefill,
                    prepare_fn=fam.prepare_prefill, before_replay=fam.before_replay(m), feed_buffer=fam.feed_buffer(m),
                    prefill_chunk=fam.prefill_chunk, prefill_prev2=fam.prefill_prev2)  # plain decode, the same checkpoint rules
    picks: collections.Counter = collections.Counter()
    pick = sched._pick_k

    def counted(s):
        kk = pick(s)
        picks[kk] += 1
        return kk
    sched._pick_k = counted

    def ks():
        """The draft lengths chosen since the last call, {k: cycles}."""
        out = dict(sorted(picks.items()))
        picks.clear()
        return out

    # ---- 1. greedy decode against the golden tokens
    want = json.load(open(GOLDEN))["prompts"]
    res["golden"] = []
    for w in want:
        r = sched.run([Request(w["prompt_ids"], max_new_tokens=len(w["tokens"]), eos_ids=STOP)])[0]
        k_used = ks()
        rp = ref.run([Request(w["prompt_ids"], max_new_tokens=len(w["tokens"]), eos_ids=STOP)])[0]
        good, msg = golden_verdict(r.output, w)
        ok &= good and same(r, rp)
        res["golden"].append(dict(prompt=w["prompt"][:40], verdict=msg, spec_equals_plain=same(r, rp), tok_s=tps(r), plain_tok_s=tps(rp),
                                  tokens_per_cycle=per_cycle(r), k=k_used))
        print(f"[golden] {w['prompt'][:40]!r}: {msg}; spec {'EQUALS' if same(r, rp) else 'DIFFERS FROM'} plain ({agree(r, rp)}); TTFT "
              f"{ttft(r)} s; {tps(r)} tok/s ({per_cycle(r)} tokens per cycle, k {k_used}) vs {tps(rp)} tok/s plain; "
              f"{tok.decode(r.output)[:60]!r}", flush=True)

    # ---- 2. multi-turn: turn 2 restores the end of turn 1
    sched.reset()
    items = " ".join(f"item {i} costs {(i * 37) % 500} coins." for i in range(400))
    sys_a = {"role": "system", "content": "You are a careful assistant. Background notes: " + items}
    m1 = [sys_a, {"role": "user", "content": "Which item costs the most? Answer in one sentence."}]
    p1 = chat_ids(m1)
    r1 = sched.run([R(p1, 48)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
    m2 = m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "And which item costs the least?"}]
    p2 = chat_ids(m2)
    r2 = sched.run([R(p2, 48)])[0]
    ks()
    ref.reset()
    rp1 = ref.run([R(p1, 48)])[0]
    rp2 = ref.run([R(p2, 48)])[0]
    sched.ckpts, sched.free_bufs = [], []  # the same prompt without checkpoints
    sched.slots[r2.slot].tokens = []
    r2c = sched.run([R(p2, 48)])[0]
    sched.ckpts, sched.free_bufs = [], list(range(sched.n_checkpoints))
    reuse_ok = r2.reused >= len(p1) + len(r1.output) - 1 and r2c.reused == 0 and rp2.reused == r2.reused
    turns_ok = same(r1, rp1) and same(r2, rp2)
    ok &= reuse_ok and turns_ok
    res.update(turn1_tokens=len(p1), turn1_ttft=ttft(r1), turn2_tokens=len(p2), turn2_reused=r2.reused, turn2_ttft=ttft(r2),
               turn2_scratch_ttft=ttft(r2c), turn2_tok_s=tps(r2), turns_spec_equal_plain=turns_ok)
    print(f"[multi-turn] turn 1: {len(p1)} + {len(r1.output)} tok, TTFT {ttft(r1)} s, {tps(r1)} tok/s; {ans!r}\n"
          f"[multi-turn] turn 2: {len(p2)} tok, reused {r2.reused} ({'RIGHT' if reuse_ok else 'WRONG'}), TTFT {ttft(r2)} s vs {ttft(r2c)} s "
          f"from scratch; greedy output agrees with the uncached run on {agree(r2, r2c)} tokens\n"
          f"[multi-turn] spec vs plain decode (the same checkpoints): turn 1 {agree(r1, rp1)}, turn 2 {agree(r2, rp2)} -> "
          f"{'EQUAL' if turns_ok else 'DIFFERENT'}", flush=True)

    # ---- 3. a second slot: another conversation, then turn 3 of the first one from memory
    sched.reset()
    r1 = sched.run([R(p1, 48)])[0]
    fares = " ".join(f"route {i} costs {(i * 53) % 900} euros." for i in range(300))
    sys_b = {"role": "system", "content": "You are a travel agent. Fares: " + fares}
    pb = chat_ids([sys_b, {"role": "user", "content": "Which route is the cheapest?"}])
    rb = sched.run([R(pb, 32)])[0]
    ans = tok.decode([t for t in r1.output if t not in EOS], skip_special_tokens=True)
    p3 = chat_ids(m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "How many items are in the notes?"}])
    r3 = sched.run([R(p3, 32)])[0]
    slot_ok = rb.slot != r1.slot and r3.slot == r1.slot and r3.reused >= len(p1) + len(r1.output) - 1
    ok &= slot_ok
    res.update(second_slot=rb.slot, first_slot=r1.slot, turn3_reused=r3.reused, turn3_ttft=ttft(r3), second_ttft=ttft(rb))
    print(f"[second slot] A in slot {r1.slot}, B ({len(pb)} tok) in slot {rb.slot}, TTFT {ttft(rb)} s; A's next turn in slot {r3.slot} reused "
          f"{r3.reused} of {len(p3)}, TTFT {ttft(r3)} s -> {'RIGHT' if slot_ok else 'WRONG'}", flush=True)

    # ---- 4. the disk cache: an eviction from both slots, then a restore from disk
    if a.sessions_dir:
        os.makedirs(a.sessions_dir)
    sdir = a.sessions_dir or tempfile.mkdtemp(prefix="sessions-check4-", dir=os.path.expanduser("~/.cache/colinfer"))
    try:
        notes = {"role": "system", "content": "You are a careful assistant. Field log: " + " ".join(
            f"day {i}: sensor {i % 7} read {(37 * i) % 101} units." for i in range(500))}
        mc = [notes, {"role": "user", "content": "Which sensor appears most often? Answer in one sentence."}]
        pc = chat_ids(mc)

        def turn2(r):
            ans = tok.decode([t for t in r.output if t not in EOS], skip_special_tokens=True)
            return chat_ids(mc + [{"role": "assistant", "content": ans}, {"role": "user", "content": "And on which day was the highest reading?"}])
        sched.reset()
        sched.store = SessionStore(sched.state, sched.mst, sched.fingerprint, 20 * 10**9, sdir, h_dim=sched.h_dim)
        rc1 = sched.run([R(pc, 48)])[0]
        for i in range(2):  # two other conversations take both slots
            sched.run([R(chat_ids([{"role": "user", "content": f"Write a haiku about the number {i + 2}."}]), 24)])
        evicted = not any(s.tokens[:len(pc)] == pc for s in sched.slots)
        pc2 = turn2(rc1)
        rd = sched.run([R(pc2, 48)])[0]
        disk_tok = sched.metrics.disk_tokens.get()
        sched.store.flush()
        files = sched.store.status()
        sched.store = None
        sched.reset()  # the reference: the same two turns, turn 2 restored from memory
        rc1b = sched.run([R(pc, 48)])[0]
        rm = sched.run([R(turn2(rc1b), 48)])[0]
        n = len(pc) + len(rc1.output) - 1
        drafts_same = (rd.cycles, rd.accepted) == (rm.cycles, rm.accepted)  # the drafts depend on the drafter's restored rows
        disk_ok = evicted and rc1b.output == rc1.output and rd.reused == rm.reused == disk_tok == n and rd.output == rm.output and drafts_same
        ok &= disk_ok
        res.update(disk_turn1_tokens=len(pc), disk_reused=rd.reused, disk_ttft=ttft(rd), memory_ttft=ttft(rm), disk_files=files["files"],
                   disk_bytes=files["bytes"], disk_identical=rd.output == rm.output, disk_accepted=[rd.accepted, rm.accepted],
                   disk_cycles=[rd.cycles, rm.cycles])
        print(f"[disk cache] turn 1 {len(pc)} + {len(rc1.output)} tok; evicted from both slots: {evicted}; turn 2 ({len(pc2)} tok) restored "
              f"{rd.reused} from disk (want {n}): TTFT {ttft(rd)} s (from memory {ttft(rm)} s); output "
              f"{'IDENTICAL' if rd.output == rm.output else 'DIFFERENT'} to the memory restore "
              f"({agree(rd, rm)}); accepted drafts / cycles {rd.accepted}/{rd.cycles} vs {rm.accepted}/{rm.cycles} from memory; "
              f"{files['files']} files, {files['bytes'] / 1e6:.0f} MB -> {'RIGHT' if disk_ok else 'WRONG'}", flush=True)
    finally:
        if not a.sessions_dir:
            shutil.rmtree(sdir, ignore_errors=True)

    # ---- 5. a prompt of two prefill chunks
    sched.reset()
    code = "47193"
    filler = " ".join(f"Line {i}: the weather on day {i} was {['sunny', 'rainy', 'windy', 'foggy'][i % 4]}." for i in range(380))
    pl = chat_ids([{"role": "user", "content": f"Remember this: the secret code is {code}.\n\n{filler}\n\nWhat is the secret code? "
                                               "Answer with the number only."}])
    rl = sched.run([R(pl, 16)])[0]
    ks()
    ref.reset()
    rlp = ref.run([R(pl, 16)])[0]
    text = tok.decode([t for t in rl.output if t not in EOS], skip_special_tokens=True)
    long_ok = len(pl) > 5120 and code in text and same(rl, rlp)  # more than 5 / 4 chunks: the scheduler runs two
    ok &= long_ok
    res.update(long_tokens=len(pl), long_ttft=ttft(rl), long_prefill_tok_s=round(len(pl) / (rl.t_first - rl.t_admit), 0), long_answer=text,
               long_spec_equals_plain=same(rl, rlp))
    print(f"[long prompt] {len(pl)} tok (two chunks of at most 4096): TTFT {ttft(rl)} s ({res['long_prefill_tok_s']:.0f} tok/s of prefill); "
          f"answer {text!r}; spec {'EQUALS' if same(rl, rlp) else 'DIFFERS FROM'} plain ({agree(rl, rlp)}) -> {'RIGHT' if long_ok else 'WRONG'}",
          flush=True)

    # ---- 6. the speed with and without speculation
    res["speed"] = []
    for name, msgs in (("code", [{"role": "user", "content": "Write a Python function that parses an ISO 8601 date string without "
                                                            "any library, with a docstring and three doctests."}]),
                       ("prose", [{"role": "user", "content": "Write a short story of about 300 words about a lighthouse keeper who "
                                                             "finds a message in a bottle."}])):
        p = chat_ids(msgs)
        sched.reset()
        ref.reset()
        rs = sched.run([Request(p, max_new_tokens=a.tokens)])[0]  # no stop token: the full length
        k_used = ks()
        rp = ref.run([Request(p, max_new_tokens=a.tokens)])[0]
        ok &= same(rs, rp)
        row = dict(prompt=name, tokens=len(rs.output), spec_tok_s=tps(rs), plain_tok_s=tps(rp), speedup=round(tps(rs) / tps(rp), 2),
                   tokens_per_cycle=per_cycle(rs), accepted_per_cycle=round(rs.accepted / max(rs.cycles, 1), 2),
                   acceptance=round(rs.accepted / max(rs.drafted, 1), 2), k=k_used, spec_equals_plain=same(rs, rp))
        res["speed"].append(row)
        print(f"[speed] {name}: {len(rs.output)} tokens, spec {tps(rs)} tok/s vs plain {tps(rp)} tok/s ({row['speedup']}x); "
              f"{row['tokens_per_cycle']} tokens per cycle ({rs.accepted} of {rs.drafted} drafts accepted, {row['accepted_per_cycle']} per "
              f"cycle); k chosen {k_used}; spec {'EQUALS' if same(rs, rp) else 'DIFFERS FROM'} plain ({agree(rs, rp)})", flush=True)
    sched.reset()

    # ---- 7. the fingerprint
    fp = sched.fingerprint
    fp_ok = fp.get("drafter") == "checkpoint" and fp.get("ngram", [None])[0] == "model-fp8-mtp-ple.safetensors"
    ok &= fp_ok
    print(f"[fingerprint] {fp} -> {'RIGHT' if fp_ok else 'WRONG'}")
    st = sched.model.feeder.stats()
    torch.cuda.synchronize()
    avail2 = mem_available()
    res.update(mem_fall_end_gb=round((AVAIL0 - avail2) / 1e9, 2), ngram_cache_stats=list(st), ok=bool(ok),
               allocated_end_gb=round(torch.cuda.memory_allocated() / 1e9, 2), reserved_end_gb=round(torch.cuda.memory_reserved() / 1e9, 2))
    print(f"[memory] after the checks: MemAvailable {(AVAIL0 - avail2) / 1e9:.2f} GB below the start ({(avail1 - avail2) / 1e9:+.2f} GB since the "
          f"build); torch allocated {res['allocated_end_gb']:.2f} GB, reserved {res['reserved_end_gb']:.2f} GB; n-gram row cache (hits, misses, "
          f"rows, capacity) {st}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("SCHEDULER CHECK QWEN4EXP", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
