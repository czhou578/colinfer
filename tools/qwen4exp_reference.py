"""Greedy continuations and a WikiText perplexity from the plain-PyTorch reference of Qwen3.8-Flash-Next on the real
checkpoint.

The reference (engine/model/qwen4exp.py, loaded by engine/weights/qwen4exp_loader.py) is slow: about one second per
token, because it dequantizes each routed expert on the fly. It is the ground truth of the kernel path. The golden test
of the decode path (tests/golden_qwen4exp.py) compares the tokens of the engine with the tokens that this tool
writes. For each greedy token, the tool also records the reference's margin: the top-1 logit minus the top-2 logit. A
divergence at a small margin is a near-tie, not a bug.

With --ppl N the tool also scores --windows consecutive windows of N tokens of the WikiText test text
(tests/perplexity.py), each from a fresh state, in chunks of --chunk tokens through the reference's state, and records the
mean NLL. tests/golden_qwen4exp.py compares the decode path's perplexity on the same tokens with it. --fp32-experts adds
the same windows with the expert loop in fp32. The reference's own bf16 rounding moves the perplexity of one window by up to
2.4% (the first 2048 tokens: 1.876 against 1.831), and that of 8 windows (16,376 tokens) by 0.03% (2.3335 against 2.3329,
2026-10-24). One window cannot resolve a 0.5% gate; 8 windows can.

Usage (loads about 78 GB on the GPU; stop the server first, and put each command under the GPU lock:
flock -w 14400 ~/.cache/colinfer/gpu-model.lock <command>):
    uv run python tools/qwen4exp_reference.py                        # three built-in prompts, 24 tokens each
    uv run python tools/qwen4exp_reference.py --tokens 32 --ppl 2048 --windows 8 --fp32-experts --out tests/data/qwen4exp_golden.json
    uv run python tools/qwen4exp_reference.py --tokens 48 --prompt "The capital of France is"

The first run on 2026-10-24 gave, for "The capital of France is": " Paris. The capital of Germany is Berlin. The capital
of Italy is Rome. The capital of Spain is Madrid." The load took 34 s.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from engine.weights.loader import resolve
from engine.weights.qwen4exp_loader import load_qwen4exp

REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"
PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "chat:In one sentence, what does a mixture-of-experts layer do?",  # chat: = a user message through the chat template, thinking off
]
STOP = (248044, 248046)  # <|endoftext|>, <|im_end|>


def prompt_ids(tok, text: str) -> list[int]:
    """The token ids of a prompt; 'chat:<text>' renders a user message through the chat template, thinking off."""
    if text.startswith("chat:"):
        msgs = [{"role": "user", "content": text[len("chat:"):]}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]
    return tok(text)["input_ids"]


def margin(logits: torch.Tensor) -> float:
    """The top-1 logit minus the top-2 logit."""
    v = logits.float().topk(2).values
    return float(v[0] - v[1])


@torch.inference_mode()
def generate(model, ids: list[int], n: int) -> tuple[list[int], list[float], float, float]:
    """(the greedy tokens, their margins, the prefill seconds, the seconds per decode token). The prompt runs as one
    chunk; each token then runs through the recurrent path, as the engine's decode does."""
    st = model.new_state(1, len(ids) + n + 8)
    t0 = time.time()
    logits = model(torch.tensor([ids], device="cuda"), st, last_only=True)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    out: list[int] = []
    margins: list[float] = []
    t0 = time.time()
    for _ in range(n):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        margins.append(margin(logits[0, -1]))
        if nxt in STOP:
            break
        logits = model(torch.tensor([[nxt]], device="cuda"), st, last_only=True)
    torch.cuda.synchronize()
    return out, margins, t_prefill, (time.time() - t0) / max(len(out), 1)


def experts_fp32(self, x2, idx, w):
    """engine.model.qwen4exp.Experts.forward with the expert math in fp32 (weights dequantized to fp32, products and the
    weighted sum in fp32, one rounding at the end): --fp32-experts measures how far the reference's own bf16 rounding
    moves the perplexity."""
    out = torch.zeros(x2.shape, dtype=torch.float32, device=x2.device)
    for e in torch.unique(idx).tolist():
        pos, tok = torch.where((idx == e).T)
        gu, dn = self.get(e, torch.float32)
        g, u = torch.nn.functional.linear(x2[tok].float(), gu).chunk(2, dim=-1)
        out.index_add_(0, tok, torch.nn.functional.linear(torch.nn.functional.silu(g) * u, dn) * w[tok, pos, None].float())
    return out.to(x2.dtype)


@torch.inference_mode()
def wikitext_nll(model, ids: list[int], chunk: int) -> tuple[float, int]:
    """(the mean NLL of tokens 1..N-1 given the ones before, the count) in one window, chunked through the state."""
    st = model.new_state(1, len(ids))
    x = torch.tensor([ids], device="cuda")
    nll, count = 0.0, 0
    for c0 in range(0, len(ids), chunk):
        logits = model(x[:, c0:c0 + chunk], st)[0]  # [t, V] fp32: row j predicts token c0 + j + 1
        tgt = x[0, c0 + 1:c0 + chunk + 1]
        logits = logits[:tgt.numel()]
        nll += float(torch.nn.functional.cross_entropy(logits, tgt, reduction="sum"))
        count += tgt.numel()
        del logits
    return nll / count, count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", action="append", help="a prompt (repeatable); 'chat:<text>' sends it as a user message, thinking off")
    ap.add_argument("--ppl", type=int, default=0, help="also score the first N tokens of the WikiText test text in one window")
    ap.add_argument("--chunk", type=int, default=512, help="the chunk of the WikiText forward")
    ap.add_argument("--windows", type=int, default=1, help="score this many consecutive windows of --ppl tokens, each from a fresh state")
    ap.add_argument("--fp32-experts", action="store_true", help="also score the windows with the expert math in fp32 (experts_fp32)")
    ap.add_argument("--out", help="write the prompts, their token ids, the greedy tokens and margins (and the perplexity) as JSON here")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    path = resolve(args.model)
    t0 = time.time()
    model = load_qwen4exp(path, device="cuda")
    torch.cuda.synchronize()
    print(f"[reference] loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on the GPU", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    results = []
    for text in args.prompt or PROMPTS:
        ids = prompt_ids(tok, text)
        out, margins, tp, td = generate(model, ids, args.tokens)
        print(f"[reference] prompt of {len(ids)} tokens: prefill {tp:.2f}s, {td * 1000:.0f} ms per token; smallest margin "
              f"{min(margins):.3f}", flush=True)
        print("    " + repr(tok.decode(out)), flush=True)
        results.append(dict(prompt=text, prompt_ids=ids, tokens=out, margins=[round(m, 4) for m in margins], text=tok.decode(out)))
    doc = dict(model=args.model, max_new_tokens=args.tokens, kv="bf16", prompts=results)
    if args.ppl:
        from engine.model import qwen4exp
        from tests.perplexity import wikitext_test
        all_ids = tok(wikitext_test(), return_tensors="pt").input_ids[0][:args.ppl * args.windows].tolist()
        windows = [all_ids[i * args.ppl:(i + 1) * args.ppl] for i in range(args.windows)]
        variants = [("wikitext", None)] + ([("wikitext_fp32_experts", experts_fp32)] if args.fp32_experts else [])
        for key, fwd in variants:
            saved = qwen4exp.Experts.forward
            if fwd is not None:
                qwen4exp.Experts.forward = fwd
            try:
                t0 = time.time()
                per = [wikitext_nll(model, ids, args.chunk) for ids in windows]
            finally:
                qwen4exp.Experts.forward = saved
            count = sum(c for _, c in per)
            nll = sum(n * c for n, c in per) / count
            print(f"[reference] {key}: {args.windows} windows of {args.ppl} tokens: ppl {math.exp(nll):.4f} (nll {nll:.5f} over {count} "
                  f"tokens; per window {[round(math.exp(n), 4) for n, _ in per]}; {time.time() - t0:.0f}s)", flush=True)
            doc[key] = dict(tokens=args.ppl, windows=args.windows, scored=count, chunk=args.chunk, nll=nll, ppl=math.exp(nll),
                            window_nll=[n for n, _ in per])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"[reference] wrote {args.out}")


if __name__ == "__main__":
    main()
