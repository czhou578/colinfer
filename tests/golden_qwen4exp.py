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

With --prefill (milestone M2), the script checks the prefill path (engine/model/prefill_qwen4exp.py) instead of 1-3:

4. attaches an n-gram feeder with a row cache to the model (model.feeder), prefills each golden prompt as one chunk, then
   decodes 32 greedy tokens through the decode graph. It compares the tokens with the reference under the near-tie rule,
   and the prefill logits of the last prompt token with the decode path's logits of the same token (max |delta|, argmax).
5. a long prompt (--long-tokens WikiText tokens, default 3000): the prefill against the token-by-token decode path for the
   same prompt. It compares the per-token NLL and argmax of every prompt token, then 16 greedy tokens after the prompt:
   they must be equal or diverge first at a near-tie of both paths.

The reference runs with a bf16 KV cache; the engine's KV cache is fp8. That is one source of small logit differences.

With --decode-copies (phase B), the decode model streams the requantized copies of tools/requant_qwen4exp.py
(attach_decode_copies4): those of the prepared directory's manifest, or the files of a given directory or list. Then the
perplexity gate of 2 compares with the exact decode path (wikitext_decode_exact of the golden file: 2.3274), within 0.5%.
1 and 3 run as before (the same near-tie rule against the reference).

Usage (loads about 78 GB on the GPU; take the GPU lock):
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tests/golden_qwen4exp.py [--profile] [--no-ppl]
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tests/golden_qwen4exp.py --prefill [--long-tokens 3000]
    flock -w 14400 ~/.cache/colinfer/gpu-model.lock ./py tests/golden_qwen4exp.py --decode-copies [DIR | FILE,FILE...]
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
FEEDER_ROWS = 6_250_000  # the row cache of the prefill's n-gram feeder: 1 GB of 160-byte rows


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

    def prefill(self, ids: list[int], chunk: int | None = None, **kw):
        """Prefills ids into a fresh state (engine/model/prefill_qwen4exp.py; model.feeder reads the n-gram rows) and
        returns what prefill returns (default: the fp32 logits [1, V] of the next token). The decode graph continues there."""
        from engine.model import prefill_qwen4exp as pq
        self.reset()
        out = pq.prefill(self.model, torch.tensor([ids], device="cuda"), self.st, chunk=chunk or pq.CHUNK, **kw)
        self.ctx = ([self.eos, self.eos] + list(ids))[-2:]
        return out


def margin(logits: torch.Tensor) -> float:
    v = logits.topk(2).values
    return float(v[0] - v[1])


def feed(step: Stepper, ids: list[int]) -> torch.Tensor:
    """The decode path over ids from a fresh state, token by token: the fp32 logits [V] after the last one."""
    step.reset()
    for t in ids:
        logits = step(t)
    return logits


def golden(step: Stepper, want: list[dict], start=None) -> bool:
    """Greedy decoding of each golden prompt against the reference's tokens. start(ids) -> the logits [V] after the
    prompt ids from a fresh state: the decode path token by token (feed, the default) or a prefill."""
    start = start or (lambda ids: feed(step, ids))
    ok = True
    for w in want:
        logits = start(w["prompt_ids"])
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
            logits = start(w["prompt_ids"] + w["tokens"][:i])
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


def greedy(step: Stepper, logits: torch.Tensor, n: int) -> tuple[list[int], list[float], list[torch.Tensor]]:
    """n greedy tokens from logits [V] through the decode graph: (the tokens, the margin of each, the logits of each)."""
    got, margins, seen = [], [], []
    for _ in range(n):
        nxt = int(logits.argmax())
        got.append(nxt)
        margins.append(margin(logits))
        seen.append(logits.clone())
        if nxt in STOP:
            break
        logits = step(nxt)
    return got, margins, seen


def prefill_logits(step: Stepper, want: list[dict]) -> list[dict]:
    """For each golden prompt: the prefill's fp32 logits of the last prompt token against the decode path's logits of
    the same token (max |delta|, relative L2, the argmax of both, the decode path's margin)."""
    out = []
    for w in want:
        ids = w["prompt_ids"]
        dec = feed(step, ids).clone()
        pf = step.prefill(ids)[0].clone()
        r = dict(prompt=w["prompt"][:40], tokens=len(ids), max_abs=float((pf - dec).abs().max()),
                 rel=float((pf - dec).norm() / dec.norm()), argmax_equal=int(pf.argmax()) == int(dec.argmax()), decode_margin=margin(dec))
        print(f"[golden4] prefill vs decode, last prompt token of {r['prompt']!r} ({len(ids)} tokens): max |delta| {r['max_abs']:.4f}, "
              f"rel {r['rel']:.2e}, argmax {'equal' if r['argmax_equal'] else 'DIFFERENT'} (decode margin {r['decode_margin']:.3f})")
        out.append(r)
    return out


def long_prompt(step: Stepper, ids: list[int], n_new: int = 16) -> dict:
    """A long prompt through the decode path token by token and through the prefill (one chunk of up to pq.CHUNK tokens,
    more chunks beyond): the NLL and argmax of every prompt token, the logits of the last one, then n_new greedy tokens.
    The greedy tokens pass if they are equal, or if the first divergence is a near-tie: the gap between the two paths'
    tokens is below NEAR_TIE in the logits of either path at that point."""
    from engine.model import prefill_qwen4exp as pq
    N = len(ids)
    tgt = torch.tensor(ids[1:], device="cuda")
    nll_dec = torch.zeros(N - 1, dtype=torch.float64, device="cuda")
    arg_dec = torch.zeros(N - 1, dtype=torch.long, device="cuda")
    t0 = time.time()
    step.reset()
    for j, t in enumerate(ids):
        logits = step(t)
        if j < N - 1:
            nll_dec[j] = -torch.log_softmax(logits, -1)[ids[j + 1]]
            arg_dec[j] = logits.argmax()
    t_dec = time.time() - t0
    tok_dec, m_dec, l_dec = greedy(step, logits, n_new)
    last_dec = l_dec[0]
    torch.cuda.synchronize()
    t0 = time.time()
    pf_all = step.prefill(ids, all_logits=True)  # bf16 [N, V]
    torch.cuda.synchronize()
    t_pf = time.time() - t0
    nll_pf = torch.zeros(N - 1, dtype=torch.float64, device="cuda")
    arg_pf = torch.zeros(N - 1, dtype=torch.long, device="cuda")
    for a in range(0, N - 1, 512):
        b = min(N - 1, a + 512)
        lg = pf_all[a:b].float()
        nll_pf[a:b] = torch.nn.functional.cross_entropy(lg, tgt[a:b], reduction="none").double()
        arg_pf[a:b] = lg.argmax(-1)
    del pf_all
    pf = step.prefill(ids)[0]
    tok_pf, m_pf, l_pf = greedy(step, pf, n_new)
    r = dict(tokens=N, decode_s=t_dec, prefill_s=t_pf, ppl_decode=math.exp(float(nll_dec.mean())), ppl_prefill=math.exp(float(nll_pf.mean())),
             argmax_agree=float((arg_dec == arg_pf).double().mean()), nll_max_abs=float((nll_dec - nll_pf).abs().max()),
             last_max_abs=float((l_pf[0] - last_dec).abs().max()), last_argmax_equal=int(l_pf[0].argmax()) == int(last_dec.argmax()),
             greedy_decode=tok_dec, greedy_prefill=tok_pf)
    print(f"[golden4] long prompt of {N} tokens: the prefill (all logits, {pq.CHUNK}-token chunks) {t_pf:.2f} s, the decode path {t_dec:.0f} s; "
          f"ppl prefill {r['ppl_prefill']:.4f} / decode {r['ppl_decode']:.4f}; argmax agreement {r['argmax_agree']:.2%}; per-token NLL "
          f"max |delta| {r['nll_max_abs']:.3f}; last token: logits max |delta| {r['last_max_abs']:.4f}, argmax "
          f"{'equal' if r['last_argmax_equal'] else 'DIFFERENT'}")
    if tok_pf == tok_dec:
        r["greedy_ok"] = True
        print(f"[golden4] long prompt: {len(tok_pf)} greedy tokens equal (smallest margin: decode {min(m_dec):.3f}, prefill {min(m_pf):.3f})")
        return r
    i = next((j for j, (a, b) in enumerate(zip(tok_dec, tok_pf)) if a != b), min(len(tok_dec), len(tok_pf)))
    if i >= min(len(tok_dec), len(tok_pf)):  # one stopped early
        r["greedy_ok"] = False
        print(f"[golden4] long prompt: the greedy tokens differ in length: decode {tok_dec}, prefill {tok_pf}")
        return r
    a, b = tok_dec[i], tok_pf[i]
    gap_dec = float(l_dec[i][a] - l_dec[i][b])  # the decode path's preference for its token
    gap_pf = float(l_pf[i][b] - l_pf[i][a])
    near = min(gap_dec, gap_pf) < NEAR_TIE
    r.update(greedy_ok=near, divergence=i, gap_decode=gap_dec, gap_prefill=gap_pf)
    print(f"[golden4] long prompt: first greedy divergence at token {i} of {n_new}: decode {a}, prefill {b}; the gap in the decode path's "
          f"logits {gap_dec:.3f}, in the prefill's {gap_pf:.3f} -> {'near-tie' if near else 'REAL DIVERGENCE'}")
    return r


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


def prefill_mode(step: Stepper, want: dict, args, res: dict) -> bool:
    """4 and 5 of the docstring, with the model's feeder (a row cache of FEEDER_ROWS rows). Returns the verdict."""
    from engine.weights.ple import PleFeeder
    model = step.model
    ple = model.ple_layer()
    model.feeder = PleFeeder(ple.table, cache_rows=FEEDER_ROWS, params=ple.params, image_token_id=model.cfg.image_token_id)
    res["prefill_logits"] = prefill_logits(step, want["prompts"])
    ok = golden(step, want["prompts"], start=lambda ids: step.prefill(ids)[0])
    res["golden_prefill_ok"] = ok
    print(f"[golden4] GOLDEN WITH PREFILL {'PASSED' if ok else 'FAILED'}")
    if args.long_tokens:
        from transformers import AutoTokenizer

        from engine.weights.loader import resolve
        from tests.perplexity import wikitext_test
        tok = AutoTokenizer.from_pretrained(resolve(args.model))
        ids = tok(wikitext_test(), return_tensors="pt").input_ids[0][:args.long_tokens].tolist()
        r = long_prompt(step, ids)
        res["long"] = r
        ok &= r["greedy_ok"]
    h, m, held, cap = model.feeder.stats()
    res["feeder"] = dict(hits=h, misses=m, held=held, capacity=cap, file_rows=model.feeder.gather_rows)
    print(f"[golden4] feeder: {h} hits, {m} misses ({h / max(1, h + m):.1%} of the distinct rows of each call from the cache), "
          f"{held} rows held of {cap}, {model.feeder.gather_rows} rows read from the file")
    print(f"[golden4] PREFILL CHECK {'PASSED' if ok else 'FAILED'}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    return ok


def decode_copies(model, path: str, spec: str) -> dict:
    """Attaches the requantized decode copies (attach_decode_copies4) named by spec: "prepared" (the manifest of the
    checkpoint's prepared directory), a directory (its decode4_*.safetensors) or a comma list of files."""
    from engine.model.fast_qwen4exp import attach_decode_copies4
    from engine.runtime.prepared import read as read_prepared
    if spec == "prepared":
        prep = read_prepared(path)
        if prep is None or not prep.decode:
            raise SystemExit(f"--decode-copies: no decode copies in the prepared directory of {path} (tools/prepare.py)")
        files = list(prep.decode)
    elif os.path.isdir(spec):
        files = sorted(os.path.join(spec, f) for f in os.listdir(spec) if f.startswith("decode4_") and f.endswith(".safetensors"))
    else:
        files = spec.split(",")
    before = torch.cuda.memory_allocated()
    t0 = time.time()
    n = attach_decode_copies4(model, files)
    torch.cuda.synchronize()
    gb = (torch.cuda.memory_allocated() - before) / 1e9
    print(f"[golden4] decode copies: {n} linears from {len(files)} files ({', '.join(os.path.basename(f) for f in files)}) in "
          f"{time.time() - t0:.1f}s, +{gb:.2f} GB on the GPU")
    return dict(copies=[os.path.basename(f) for f in files], copies_linears=n, copies_gb=gb)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--dump", help="save the per-token WikiText NLLs here (torch.save)")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--prefill", action="store_true", help="check the prefill path (4 and 5 of the docstring) instead of the decode path")
    ap.add_argument("--long-tokens", type=int, default=3000, help="--prefill: the WikiText tokens of the long prompt (0: skip it)")
    ap.add_argument("--decode-copies", nargs="?", const="prepared", default=None,
                    help="attach the requantized decode copies: those of the prepared directory's manifest (no value), a directory's "
                         "decode4_*.safetensors, or a comma list of files")
    args = ap.parse_args()
    from engine.model.fast import capture
    from engine.model.fast_qwen4exp import load_fast_qwen4exp
    from engine.weights.loader import resolve
    want = json.load(open(FILE))
    res: dict = {}
    path, model = load_fast_qwen4exp(args.model, max_seq_len=4096)
    torch.cuda.synchronize()
    res["gb_after_load"] = torch.cuda.memory_allocated() / 1e9
    if args.decode_copies:
        res.update(decode_copies(model, path, args.decode_copies))
    step = Stepper(model, 4096)
    torch.cuda.synchronize()
    res["gb_after_state_and_graph"] = torch.cuda.memory_allocated() / 1e9
    print(f"[golden4] memory_allocated: {res['gb_after_load']:.2f} GB after the load, {res['gb_after_state_and_graph']:.2f} GB with the "
          f"4096-token state and the decode graph")
    if args.prefill:
        sys.exit(0 if prefill_mode(step, want, args, res) else 1)
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
        if args.decode_copies:  # the gate of the copies: the exact decode path
            ex = want["wikitext_decode_exact"]
            r_ex = math.exp(nll) / ex["ppl"]
            res.update(ppl_exact=ex["ppl"], ppl_ratio_exact=r_ex, ppl_ok=abs(r_ex - 1) <= PPL_GATE,
                       window_ppl=[math.exp(float(a.mean())) for a in per])
            print(f"[golden4] decode copies: ppl {math.exp(nll):.4f} against the exact decode path's {ex['ppl']:.4f}: ratio {r_ex:.5f} -> "
                  f"{'PASSED' if res['ppl_ok'] else 'FAILED'} (gate {PPL_GATE:.1%}); per window "
                  + ", ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(res["window_ppl"], ex["window_ppl"])))
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
