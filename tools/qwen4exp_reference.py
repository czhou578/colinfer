"""Greedy continuations from the plain-PyTorch reference of Qwen3.8-Flash-Next on the real checkpoint.

The reference (engine/model/qwen4exp.py, loaded by engine/weights/qwen4exp_loader.py) is slow: about one second per
token, because it dequantizes each routed expert on the fly. It is the ground truth of the kernel path: the golden test
of the decode milestone compares the tokens of the engine with the tokens that this tool writes.

Usage (loads about 78 GB on the GPU; stop the server first):
    uv run python tools/qwen4exp_reference.py                       # three built-in prompts, 24 tokens each
    uv run python tools/qwen4exp_reference.py --tokens 48 --out golden.json --prompt "The capital of France is"

The first run on 2026-10-24 gave, for "The capital of France is": " Paris. The capital of Germany is Berlin. The capital
of Italy is Rome. The capital of Spain is Madrid." The load took 34 s.
"""
from __future__ import annotations

import argparse
import json
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


@torch.inference_mode()
def generate(model, ids: list[int], n: int) -> tuple[list[int], float, float]:
    """(the greedy tokens, the prefill seconds, the seconds per decode token). The prompt runs as one chunk; each token
    then runs through the recurrent path, as the engine's decode does."""
    st = model.new_state(1, len(ids) + n + 8)
    t0 = time.time()
    logits = model(torch.tensor([ids], device="cuda"), st, last_only=True)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    out: list[int] = []
    t0 = time.time()
    for _ in range(n):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt in STOP:
            break
        logits = model(torch.tensor([[nxt]], device="cuda"), st, last_only=True)
    torch.cuda.synchronize()
    return out, t_prefill, (time.time() - t0) / max(len(out), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=REPO)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", action="append", help="a prompt (repeatable); 'chat:<text>' sends it as a user message, thinking off")
    ap.add_argument("--out", help="write the prompts and their token ids as JSON here")
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
        if text.startswith("chat:"):
            msgs = [{"role": "user", "content": text[len("chat:"):]}]
            ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]
        else:
            ids = tok(text)["input_ids"]
        out, tp, td = generate(model, ids, args.tokens)
        print(f"[reference] prompt of {len(ids)} tokens: prefill {tp:.2f}s, {td * 1000:.0f} ms per token", flush=True)
        print("    " + repr(tok.decode(out)), flush=True)
        results.append(dict(prompt=text, prompt_ids=ids, tokens=out, text=tok.decode(out)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[reference] wrote {args.out}")


if __name__ == "__main__":
    main()
