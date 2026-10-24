#!/usr/bin/env python3
"""The golden check of the qwen4_exp decode path (milestone M1) on the real checkpoint, against the plain-PyTorch reference.

tools/qwen4exp_reference.py wrote tests/data/qwen4exp_golden.json: the reference's greedy tokens for three prompts, the
margin (top-1 minus top-2 logit) of each, and the mean NLL of the first 2048 WikiText tokens. This script loads the decode
model (engine/model/fast_qwen4exp.py), captures one decode step (T = 1, one slot of 4096 tokens) as a CUDA graph and:

1. feeds each prompt token by token (the host computes the n-gram rows of each token and fills the graph's feed buffer),
   then decodes 32 greedy tokens. It reports the first divergence from the reference with the reference's margin there.
   Acceptance: the tokens are equal, or each prompt diverges first where the reference's margin is below 0.2 (a near-tie:
   the reference's logits are bf16, so a margin of one or two bf16 steps does not order two tokens reliably).
2. scores the WikiText windows of the golden file (2048 tokens each, each from a fresh state) through the same graph, token
   by token, and compares the perplexity with the reference's. Acceptance: the ratio is within 0.5%. The file may also
   hold the reference with its expert math in fp32 (tools/qwen4exp_reference.py --fp32-experts): the distance between
   the two references shows the reference's own rounding spread.
3. times the decode step (T = 1) and a verify of 8 rows at 1k context (the median of 20 graph replays), and, with
   --profile, breaks the decode step down by kernel class.

The reference runs with a bf16 KV cache; the engine's KV cache is fp8. That is one source of small logit differences.

Usage (loads about 78 GB on the GPU; take the GPU lock):
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tests/golden_qwen4exp.py [--profile] [--no-ppl]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(ROOT, "tests", "data", "qwen4exp_golden.json")
NEAR_TIE = 0.2      # a divergence where the reference's margin is below this is a near-tie: two bf16 steps at a logit of about 22
                    # (the reference's lm_head rounds its logits to bf16; its fp32 logits picked the engine's token at the first such case)
PPL_GATE = 0.005    # the perplexity ratio must be within 0.5%
STOP = (248044, 248046)


class Stepper:
    """One decode step of slot 0 as a CUDA graph: the host writes the token and its n-gram rows, then replays."""

    def __init__(self, model, L: int):
        from engine.model.fast import capture
        self.model, self.st = model, model.new_state(1, L)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        self.buf = model.feed_buffer(1, 1)
        self.graph, self.out = capture(lambda: model(self.tok, self.st), None)
        self.eos = model.cfg.eos_token_id
        self.reset()

    def reset(self):
        self.st.reset()
        self.ctx = [self.eos, self.eos]

    def __call__(self, t: int) -> torch.Tensor:
        """Feeds token t at the next position; returns the fp32 logits [V] of the next token (valid until the next call)."""
        self.buf.copy_(self.model.host_rows([[t]], [self.ctx]))
        self.ctx = [self.ctx[1], t]
        self.tok.fill_(t)
        self.graph.replay()
        return self.out[0]


def margin(logits: torch.Tensor) -> float:
    v = logits.topk(2).values
    return float(v[0] - v[1])


def golden(step: Stepper, want: list[dict]) -> bool:
    ok = True
    for w in want:
        step.reset()
        for t in w["prompt_ids"]:
            logits = step(t)
        got, margins = [], []
        for _ in range(len(w["tokens"])):
            nxt = int(logits.argmax())
            got.append(nxt)
            margins.append(margin(logits))
            if nxt in STOP:
                break
            logits = step(nxt)
        tag = repr(w["prompt"][:40])
        if got == w["tokens"]:
            print(f"[golden4] {tag}: {len(got)} tokens equal (smallest margin: reference {min(w['margins']):.3f}, engine {min(margins):.3f})")
            continue
        i = next(j for j, (a, b) in enumerate(zip(got, w["tokens"])) if a != b) if any(a != b for a, b in zip(got, w["tokens"])) \
            else min(len(got), len(w["tokens"]))
        m = w["margins"][i] if i < len(w["margins"]) else float("nan")
        if i < len(got):  # the engine's logits at the divergence: replay the reference's prefix
            step.reset()
            for t in w["prompt_ids"] + w["tokens"][:i]:
                logits = step(t)
            v, k = logits.topk(3)
            print(f"[golden4]   engine top-3 at token {i} after the reference's prefix: {list(zip(k.tolist(), [round(a, 4) for a in v.tolist()]))}")
        near = m < NEAR_TIE
        ok &= near
        print(f"[golden4] {tag}: first divergence at token {i} of {len(w['tokens'])}: engine {got[i] if i < len(got) else None}, reference "
              f"{w['tokens'][i] if i < len(w['tokens']) else None}; reference margin {m:.3f}, engine margin "
              f"{margins[i] if i < len(margins) else float('nan'):.3f} -> {'near-tie' if near else 'REAL DIVERGENCE'}")
    return ok


def perplexity(step: Stepper, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(the NLL of each token 1..N-1 from the fp32 logits, the same from the logits rounded to bf16 as the reference's
    lm_head rounds them, the seconds) through the decode graph, one window from a fresh state."""
    step.reset()
    nll = torch.zeros(len(ids) - 1, dtype=torch.float64, device="cuda")
    nll16 = torch.zeros_like(nll)
    t0 = time.time()
    for j in range(len(ids) - 1):
        logits = step(ids[j])
        nll[j] = -torch.log_softmax(logits, -1)[ids[j + 1]]
        nll16[j] = -torch.log_softmax(logits.bfloat16().float(), -1)[ids[j + 1]]
    return nll.cpu(), nll16.cpu(), time.time() - t0


def timed(fn, n: int = 20) -> float:
    """The median ms of n calls of fn (CUDA events)."""
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


KERNEL_CLASSES = [("MoE experts (moe.cu)", ("moe::",)), ("BF16 skinny GEMM (linears, router, shared expert, lm_head)", ("skinny::",)),
                  ("hyper-connection mix / post (hc.cu)", ("hc::",)), ("grouped norms (norm.cu)", ("rmsn::",)),
                  ("bf16_gemv (HC down | inject, GDN b / a)", ("gemv::",)), ("QSA indexer (qsa.cu)", ("qsa::",)),
                  ("attention prologue + sparse attention (attn::)", ("attn::",)), ("GDN conv + delta (gdn_step.cu)", ("gdn::",)),
                  ("torch ops (PLE, embedding, positions)", ("",))]


def profile(step: Stepper, n: int = 5):
    """The decode step's GPU time by kernel class (torch.profiler over n graph replays)."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as prof
    torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            step.st.pos_t.fill_(1024)
            step.graph.replay()
        torch.cuda.synchronize()
    cls: dict[str, float] = {}
    for e in p.key_averages():
        us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if us <= 0 or "Memcpy" in e.key or "Memset" in e.key:
            continue
        name = next(c for c, keys in KERNEL_CLASSES if any(k in e.key for k in keys))
        cls[name] = cls.get(name, 0.0) + us / n
    total = sum(cls.values())
    print(f"[golden4] profile of the decode step (GPU time, {total / 1000:.1f} ms of kernels per step):")
    for c, us in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"    {c:58s} {us / 1000:7.2f} ms  {100 * us / total:5.1f}%")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--dump", help="save the per-token WikiText NLLs here (torch.save)")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    from engine.model.fast import capture
    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.weights.loader import resolve
    want = json.load(open(FILE))
    res: dict = {}
    _, model = load_fast_qwen4exp(args.model, max_seq_len=4096)
    torch.cuda.synchronize()
    res["gb_after_load"] = torch.cuda.memory_allocated() / 1e9
    step = Stepper(model, 4096)
    torch.cuda.synchronize()
    res["gb_after_state_and_graph"] = torch.cuda.memory_allocated() / 1e9
    print(f"[golden4] memory_allocated: {res['gb_after_load']:.2f} GB after the load, {res['gb_after_state_and_graph']:.2f} GB with the "
          f"4096-token state and the decode graph")
    ok = golden(step, want["prompts"])
    res["golden_ok"] = ok
    print(f"[golden4] GOLDEN {'PASSED' if ok else 'FAILED'}")
    if not args.no_ppl and "wikitext" in want:
        from transformers import AutoTokenizer

        from tests.perplexity import wikitext_test
        w = want["wikitext"]
        n, nw = w["tokens"], w.get("windows", 1)
        tok = AutoTokenizer.from_pretrained(resolve(args.model))
        all_ids = tok(wikitext_test(), return_tensors="pt").input_ids[0][:n * nw].tolist()
        per, per16, secs = [], [], 0.0
        for i in range(nw):
            a, b, s_ = perplexity(step, all_ids[i * n:(i + 1) * n])
            per.append(a)
            per16.append(b)
            secs += s_
            ref_i = math.exp(w.get("window_nll", [w["nll"]])[i])
            print(f"[golden4]   window {i}: decode path ppl {math.exp(float(a.mean())):.4f}, reference {ref_i:.4f}", flush=True)
        per_tok, per_tok16 = torch.cat(per), torch.cat(per16)
        nll = float(per_tok.mean())
        ratio = math.exp(nll) / w["ppl"]
        res.update(ppl=math.exp(nll), ppl_ref=w["ppl"], ppl_ratio=ratio, ppl_ok=abs(ratio - 1) <= PPL_GATE,
                   ppl_bf16_logits=math.exp(float(per_tok16.mean())))
        if args.dump:
            torch.save(dict(nll=per_tok, nll_bf16_logits=per_tok16, ids=all_ids), args.dump)
        print(f"[golden4] WikiText {nw} windows of {n} tokens: decode path ppl {math.exp(nll):.4f}, reference {w['ppl']:.4f}, ratio {ratio:.5f} "
              f"({secs:.0f}s, {1000 * secs / per_tok.numel():.1f} ms per token with the host rows) -> "
              f"{'PASSED' if res['ppl_ok'] else 'FAILED'} (gate {PPL_GATE:.1%}); with the logits rounded to bf16 {res['ppl_bf16_logits']:.4f}")
        w32 = want.get("wikitext_fp32_experts")
        if w32:
            res.update(ppl_ref_fp32_experts=w32["ppl"], ppl_ratio_fp32_experts=math.exp(nll) / w32["ppl"])
            print(f"[golden4] the reference with fp32 expert math: ppl {w32['ppl']:.4f}; the decode path / it {res['ppl_ratio_fp32_experts']:.5f}; "
                  f"the bf16 reference / it {w['ppl'] / w32['ppl']:.5f}")
    if args.no_timing:
        sys.exit(0 if ok and res.get("ppl_ok", True) else 1)
    # timing at 1k context: the KV rows below 1024 hold the last run's tokens
    st = step.st

    def one():
        st.pos_t.fill_(1024)
        step.graph.replay()
    res["decode_ms"] = timed(one)
    T = 8
    toks = torch.randint(0, model.cfg.vocab_size, (1, T), device="cuda")
    st.pos_t.fill_(1024)
    vg, _ = capture(lambda: model.verify(toks, st), None)

    def ver():
        st.pos_t.fill_(1024)
        vg.replay()
    res["verify8_ms"] = timed(ver)
    res["gb_with_verify_graph"] = torch.cuda.memory_allocated() / 1e9
    print(f"[golden4] at 1k context: decode step {res['decode_ms']:.2f} ms ({1000 / res['decode_ms']:.1f} tok/s), verify of {T} rows "
          f"{res['verify8_ms']:.2f} ms (median of 20 graph replays); memory_allocated {res['gb_with_verify_graph']:.2f} GB")
    if args.profile:
        profile(step)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    sys.exit(0 if ok and res.get("ppl_ok", True) else 1)


if __name__ == "__main__":
    main()
