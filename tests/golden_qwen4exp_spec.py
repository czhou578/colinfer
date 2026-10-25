#!/usr/bin/env python3
"""The exactness check of speculation for qwen4_exp on the real checkpoint: greedy tokens with the MTP
drafter (engine/spec/mtp4.py, Mtp4Cycle with k = 1, 2, 3) against plain decode, and the numbers of the scheduler's
cost model (engine/model/family.py QWEN4EXP_CYCLE_MS comes from this fit).

The script loads the decode model and the drafter once (one slot of 4096 tokens), then for each prompt of
tests/data/qwen4exp_golden.json:

1. plain decode: the prompt token by token through the decode graph of tests/golden_qwen4exp.py (Stepper), then
   --tokens greedy tokens (no stop);
2. speculation for each k: the prompt through verify + commit in chunks of 16 rows (its last logits must equal the
   plain path's, bit for bit), the drafter's prompt rows (Mtp4.first_drafts), then cycles until --tokens tokens. The
   host fills the cycle's n-gram rows before each replay. The tokens must equal plain decode's; the script prints the
   first divergence if any (that is a bug: the accept rule makes any drafts exact).

A fourth prompt, --long tokens of WikiText (default 1024), measures the same at 1k context with a real state (it also
runs the drafter's prefill path, Mtp4.prefill, on the real weights).

It reports: the accepted tokens per cycle per k and prompt; the GPU time of each cycle and of each plain decode step in
the generation; the distinct experts of each verify (the target's KernelMoE stats: the mean over the 48 layers); a
least-squares fit cycle_ms = a + b x distinct + c x k over the cycles whose drafting ran in full (no early exit); the
replay times at 1k context on a garbage state (the median of 20 replays: the verify alone, the cycle, the plain step;
the routing of garbage rows spreads over more experts, so these are upper bounds); the tokens per second of each k
against plain decode; the drafter's load time and memory.

--draft-stop sets the threshold of the draft early exit (engine/spec/mtp.py DRAFT_STOP) for the cycles of the run.
With --prepared, the model and the drafter run as the server runs them: the decode copies and the low-rank draft head of
the prepared directory (tools/prepare.py). --ctx adds the garbage-state timings at longer contexts (the slot grows to
the longest): the block selection is the only part that grows with the context, so the difference to 1k checks the
context term of the cost model (QWEN4EXP_SELECT_S).

Usage (loads about 80 GB on the GPU; take the GPU lock):
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock uv run python tests/golden_qwen4exp_spec.py [--json out.json]
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock uv run python tests/golden_qwen4exp_spec.py --prepared --ctx 1024 65536 262000
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from tests.golden_qwen4exp import FILE, Stepper, timed

CHUNK = 16  # prompt rows per verify


def _ctx(eos, hist):
    return ([eos, eos] + list(hist))[-2:]


class Spec:
    """The host side of speculation for slot 0 at one draft length k (what the scheduler does)."""

    def __init__(self, model, mtp, st, mst, k, early_exit=True):
        from engine.spec.mtp4 import Mtp4Cycle
        self.model, self.mtp, self.st, self.mst, self.k = model, mtp, st, mst, k
        self.eos = model.cfg.eos_token_id
        self.cyc = Mtp4Cycle(model, mtp, st, mst, k, early_exit=early_exit)

    def start(self, prompt: list[int]) -> torch.Tensor:
        """Feeds the prompt and writes the first cycle's input; returns the last prompt row's logits [V]."""
        m, st = self.model, self.st
        st.reset()
        self.mst.reset()
        hs, done = [], []
        for c0 in range(0, len(prompt), CHUNK):
            ch = prompt[c0:c0 + CHUNK]
            rows = m.host_rows([ch], [_ctx(self.eos, done)]).cuda()
            logits, h4 = m.verify(torch.tensor([ch], device="cuda"), st, rows)
            m.commit(st, torch.tensor([len(ch)], dtype=torch.int32, device="cuda"))
            hs.append(h4[0])
            done += ch
        last = logits[0, -1].clone()
        y = int(last.argmax())
        d = self.mtp.first_drafts(torch.tensor(prompt[1:] + [y], device="cuda"), torch.cat(hs), self.mst.slot(0), self.k)
        self.cyc.tok.copy_(torch.tensor([[y] + d], device="cuda"))
        self.prompt, self.out = list(prompt), [y]
        return last

    def cycle(self, stats=None):
        """One replay: (n accepted, GPU ms, the mean distinct experts per layer of the verify or None, the early exit flag)."""
        tok = self.cyc.tok[0].tolist()
        self.cyc.feed.copy_(self.model.host_rows([tok], [_ctx(self.eos, self.prompt + self.out[:-1])]))
        s0 = stats() if stats else None
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self.cyc.graph.replay()
        b.record()
        b.synchronize()
        n = int(self.cyc.n[0])
        self.out += self.cyc.out_tok[0, :n].tolist()
        dist = None
        if stats:
            s1 = stats()
            dist = (s1[0] - s0[0]) / max(1, s1[1] - s0[1])
        return n, a.elapsed_time(b), dist, int(self.cyc.skip[0])


def plain(step: Stepper, prompt: list[int], n: int):
    """(n greedy tokens after the prompt, the last prompt row's logits, the GPU ms of each generation step)."""
    step.reset()
    for t in prompt:
        logits = step(t)
    last = logits.clone()
    out, ms = [], []
    while len(out) < n:
        t = int(logits.argmax())
        out.append(t)
        step.buf.copy_(step.model.host_rows([[t]], [step.ctx]))  # Stepper.__call__ with the host rows outside the timed window
        step.ctx = [step.ctx[1], t]
        step.tok.fill_(t)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step.graph.replay()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
        logits = step.out[0]
    return out, last, ms


def fit(rows):
    """Least squares ms = a + b x distinct + c x k over rows (k, distinct, ms) -> (a, b, c)."""
    import numpy as np
    k, x, y = (np.array(c, dtype=np.float64) for c in zip(*rows))
    A = np.stack([np.ones_like(x), x, k], 1) if len(set(k.tolist())) > 1 else np.stack([np.ones_like(x), x, np.zeros_like(x)], 1)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return tuple(float(v) for v in sol)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--k", default="1,2,3")
    ap.add_argument("--long", type=int, default=1024, help="the WikiText prompt's tokens (0: none)")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--prepared", action="store_true", help="the decode copies and the low-rank draft head of the prepared directory")
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024], help="the contexts of the garbage-state timings")
    ap.add_argument("--draft-stop", type=float, default=None, help="the threshold of the draft early exit (default: DRAFT_STOP)")
    args = ap.parse_args()
    from engine.model.fast import capture
    from engine.model.fast_qwen4exp import attach_decode_copies4, load_fast_qwen4exp
    from engine.runtime.prepared import read as read_prepared
    from engine.spec import mtp4
    from engine.spec.mtp4 import Mtp4, Mtp4State
    if args.draft_stop is not None:  # read when each cycle graph is captured
        mtp4.DRAFT_STOP = args.draft_stop
    want = json.load(open(FILE))
    ks = [int(x) for x in args.k.split(",")]
    res: dict = {"k": ks, "tokens": args.tokens}
    L = -(-max(4096, max(args.ctx) + 64) // 16) * 16
    path, model = load_fast_qwen4exp(args.model, max_seq_len=L)
    prep = read_prepared(path) if args.prepared else None
    if prep is not None and prep.decode:
        res["decode_copies"] = attach_decode_copies4(model, prep.decode)
        print(f"[spec4] decode copies of {prep.dir}: {res['decode_copies']} linears")
    for layer in model.layers:  # distinct experts per decode launch (before any graph capture: the graphs keep the pointer)
        layer.mlp.stats = torch.zeros(2, dtype=torch.int32, device="cuda")

    def stats():
        s = torch.stack([layer.mlp.stats for layer in model.layers]).sum(0).tolist()
        return s
    torch.cuda.synchronize()
    gb0 = torch.cuda.memory_allocated()
    t0 = time.time()
    mtp = Mtp4(model, path, lowrank=prep.lowrank if prep is not None else None)
    torch.cuda.synchronize()
    res.update(drafter_load_s=time.time() - t0, drafter_gb=(torch.cuda.memory_allocated() - gb0) / 1e9, drafter_own_gb=mtp.nbytes() / 1e9,
               drafter_cached=mtp.cached)
    print(f"[spec4] drafter: {res['drafter_load_s']:.1f}s ({'from the cache' if mtp.cached else 'requantized'}), "
          f"{res['drafter_gb']:.2f} GB on the GPU ({res['drafter_own_gb']:.2f} GB of weights and head)")
    step = Stepper(model, 4096)
    st = model.new_state(1, L)
    mst = Mtp4State(model.cfg, L, "cuda", active=st.active, rope_delta=st.rope_delta)
    specs = {k: Spec(model, mtp, st, mst, k) for k in ks}
    torch.cuda.synchronize()
    res["gb_total"] = torch.cuda.memory_allocated() / 1e9
    print(f"[spec4] memory_allocated {res['gb_total']:.2f} GB with the plain graph and {len(ks)} cycle graphs")
    ok = True
    prompts = [(w["prompt"][:40], w["prompt_ids"], w["tokens"]) for w in want["prompts"]]
    if args.long:
        from transformers import AutoTokenizer

        from tests.perplexity import wikitext_test
        tk = AutoTokenizer.from_pretrained(path)
        prompts.append((f"WikiText, {args.long} tokens", tk(wikitext_test()[:20 * args.long]).input_ids[:args.long], None))
    groups = {"golden": [p for p in prompts if p[2] is not None], "long": [p for p in prompts if p[2] is None]}
    res["prompts"], res["per_k"] = [], {}
    for gname, plist in groups.items():
        per = {k: dict(n=[], ms=[], dist=[], skip=[], wall_s=0.0, tokens=0) for k in ks}
        plain_ms = []
        for tag, prompt, golden in plist:
            mtp.set_prompt_vocab(prompt)
            ref, last_plain, pms = plain(step, prompt, args.tokens)
            plain_ms += pms
            pr = {"prompt": tag, "plain_equals_golden": ref[:len(golden)] == golden if golden else None}
            for k in ks:
                sp = specs[k]
                last = sp.start(prompt)
                same_logits = torch.equal(last, last_plain)
                ns = []
                t1 = time.time()
                while len(sp.out) < args.tokens:
                    n, ms, dist, skip = sp.cycle(stats)
                    ns.append(n)
                    for key, v in (("n", n), ("ms", ms), ("dist", dist), ("skip", skip)):
                        per[k][key].append(v)
                per[k]["wall_s"] += time.time() - t1
                per[k]["tokens"] += sum(ns)
                got = sp.out[:args.tokens]
                eq = got == ref
                ok &= eq and same_logits
                div = None if eq else next(i for i, (a, b) in enumerate(zip(got, ref)) if a != b)
                pr[f"k{k}"] = dict(equal=eq, first_divergence=div, prompt_logits_equal=same_logits, cycles=len(ns),
                                   tokens_per_cycle=sum(ns) / len(ns))
                print(f"[spec4] {tag!r} k = {k}: {'EQUAL' if eq else f'DIVERGES at token {div}: spec {got[div]}, plain {ref[div]}'} "
                      f"({len(ns)} cycles, {sum(ns) / len(ns):.2f} tokens per cycle; the prompt's last logits "
                      f"{'equal' if same_logits else 'DIFFER'})", flush=True)
            res["prompts"].append(pr)
        pm = statistics.median(plain_ms)
        g = res["per_k"][gname] = {"plain_step_ms": pm}
        print(f"[spec4] {gname} prompts: plain decode step {pm:.1f} ms median in the generation ({1000 / pm:.1f} tok/s)")
        rows = []
        for k in ks:
            d = per[k]
            tpc = sum(d["n"]) / len(d["n"])
            med = statistics.median(d["ms"])
            full = [(k, x, y) for x, y, sk in zip(d["dist"], d["ms"], d["skip"]) if not sk]
            rows += full
            r = dict(tokens_per_cycle=tpc, acceptance=(tpc - 1) / k, cycle_ms=med, early_exits=sum(d["skip"]) / len(d["skip"]),
                     cycle_ms_full=statistics.median([y for _, _, y in full]) if full else None, distinct=sum(d["dist"]) / len(d["dist"]),
                     tok_s=1000 * tpc / med, speedup=pm * tpc / med, tok_s_wall=d["tokens"] / d["wall_s"])
            g[k] = r
            print(f"[spec4] {gname} k = {k}: {tpc:.2f} tokens per cycle (acceptance {r['acceptance']:.2f} of the drafts); cycle {med:.1f} ms "
                  f"median ({r['cycle_ms_full'] or float('nan'):.1f} ms without an early exit, which {r['early_exits']:.0%} of the cycles took); "
                  f"{r['distinct']:.1f} distinct experts per layer; {r['tok_s']:.1f} tok/s on the GPU ({r['speedup']:.2f}x plain), "
                  f"{r['tok_s_wall']:.1f} tok/s wall with the host rows", flush=True)
        if rows:
            a, b, c = fit(rows)
            g["fit"] = dict(a=a, b=b, c=c, n=len(rows))
            print(f"[spec4] {gname}: cycle_ms = {a:.2f} + {b:.3f} x distinct + {c:.2f} x k over {len(rows)} cycles without an early exit")
    # timing at 1k context on a garbage state: the rows below 1024 hold the last run's tokens
    def one():
        step.st.pos_t.fill_(1024)
        step.graph.replay()
    res["decode_ms_1k"] = timed(one)
    print(f"[spec4] at 1k context on a garbage state: plain decode step {res['decode_ms_1k']:.2f} ms")
    for k in ks:
        sp = specs[k]
        toks = sp.cyc.tok.clone()
        st.pos_t.fill_(1024)
        vg, _ = capture(lambda: model.verify(toks, st), None)  # noqa: B023 (captured now)
        for ctx in args.ctx:
            def ver():
                st.pos_t.fill_(ctx)  # noqa: B023
                mst.pos_t.fill_(ctx)  # noqa: B023
                vg.replay()  # noqa: B023

            def cyc():
                st.pos_t.fill_(ctx)  # noqa: B023
                mst.pos_t.fill_(ctx)  # noqa: B023
                sp.cyc.graph.replay()  # noqa: B023
            v_ms, c_ms = timed(ver), timed(cyc)
            res.setdefault("garbage", {}).setdefault(str(ctx), {})[k] = dict(verify_ms=v_ms, cycle_ms=c_ms)
            print(f"[spec4]   k = {k} at {ctx} tokens: verify of {k + 1} rows {v_ms:.1f} ms, cycle {c_ms:.1f} ms (drafting {c_ms - v_ms:.1f} ms)",
                  flush=True)
    print(f"[spec4] SPEC {'PASSED' if ok else 'FAILED'}: the tokens with speculation {'equal' if ok else 'DIFFER FROM'} plain decode")
    res["ok"] = ok
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1, default=str)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
