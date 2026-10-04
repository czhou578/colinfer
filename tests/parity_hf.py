#!/usr/bin/env python3
"""Model parity harness (PLAN.md 4.7): greedy generation on 30 fixed prompts x N tokens, our
plain-PyTorch model vs HF transformers BF16, token-exact plus logit agreement on the first steps.

The two models do not fit on the GPU together with headroom, so each side runs in its own process
and writes a .pt file; `compare` reads both.

  uv run python tests/parity_hf.py ref  --ckpt Qwen/Qwen3.8-27B --out tests/parity_out/hf_bf16.pt
  uv run python tests/parity_hf.py ours --ckpt Qwen/Qwen3.8-27B --out tests/parity_out/ours_bf16.pt
  uv run python tests/parity_hf.py compare tests/parity_out/hf_bf16.pt tests/parity_out/ours_bf16.pt

Options: --max-new-tokens 128 --keep-logits 32 --prompts 30 (use fewer for a quick check).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.prompts import PROMPTS  # noqa: E402


def resolve(p):
    from engine.weights.loader import resolve as r
    return r(p)


def build_inputs(tok, prompt: str) -> torch.Tensor:
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt", return_dict=False)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return ids


def run_ref(args):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    path = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[ref] loaded in {time.time() - t0:.0f}s, attn_implementation={model.config._attn_implementation}")
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    results = []
    for i, p in enumerate(PROMPTS[: args.prompts]):
        ids = build_inputs(tok, p).to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, output_logits=True, return_dict_in_generate=True, eos_token_id=eos)
        gen = out.sequences[0, ids.shape[1]:].tolist()
        logits = torch.stack([l[0].float() for l in out.logits[: args.keep_logits]]).cpu()
        results.append(dict(prompt=p, input_ids=ids[0].cpu(), tokens=gen, logits=logits))
        print(f"[ref] {i:2d} {len(gen):3d} tok {time.time() - t0:5.1f}s  {tok.decode(gen[:24])!r}")
    torch.save(dict(side="hf", ckpt=args.ckpt, results=results, eos=eos), args.out)
    print(f"[ref] wrote {args.out}")


def run_ours(args):
    from transformers import AutoTokenizer
    from engine.runtime.generate import generate
    from engine.weights.loader import load_model
    path = resolve(args.ckpt)
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    model = load_model(path)
    print(f"[ours] loaded in {time.time() - t0:.0f}s")
    import json
    gc = json.load(open(os.path.join(path, "generation_config.json")))
    eos = gc.get("eos_token_id", [])
    eos = [eos] if isinstance(eos, int) else list(eos)
    results = []
    for i, p in enumerate(PROMPTS[: args.prompts]):
        ids = build_inputs(tok, p).to("cuda")
        t0 = time.time()
        gen, logits = generate(model, ids, args.max_new_tokens, eos_ids=eos, keep_logits=args.keep_logits)
        results.append(dict(prompt=p, input_ids=ids[0].cpu(), tokens=gen, logits=logits.cpu() if logits is not None else None))
        print(f"[ours] {i:2d} {len(gen):3d} tok {time.time() - t0:5.1f}s  {tok.decode(gen[:24])!r}")
    torch.save(dict(side="ours", ckpt=args.ckpt, results=results, eos=eos), args.out)
    print(f"[ours] wrote {args.out}")


def compare(a_path, b_path):
    a = torch.load(a_path)
    b = torch.load(b_path)
    print(f"A = {a['side']} ({a['ckpt']}),  B = {b['side']} ({b['ckpt']})")
    n_exact = 0
    rows = []
    for i, (ra, rb) in enumerate(zip(a["results"], b["results"])):
        assert torch.equal(ra["input_ids"], rb["input_ids"]), f"prompt {i}: input ids differ"
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        first_diff = next((j for j in range(n) if ta[j] != tb[j]), None)
        exact = first_diff is None and len(ta) == len(tb)
        n_exact += exact
        la, lb = ra["logits"], rb["logits"]
        kl = maxabs = top1 = None
        if la is not None and lb is not None:
            m = min(la.shape[0], lb.shape[0])
            # only compare steps before the first divergence (after that the contexts differ)
            m = min(m, first_diff if first_diff is not None else m)
            if m > 0:
                pa = torch.log_softmax(la[:m], -1)
                pb = torch.log_softmax(lb[:m], -1)
                kl = (pa.exp() * (pa - pb)).sum(-1).mean().item()
                maxabs = (la[:m] - lb[:m]).abs().max().item()
                top1 = (la[:m].argmax(-1) == lb[:m].argmax(-1)).float().mean().item()
        rows.append((i, len(ta), len(tb), first_diff, kl, maxabs, top1))
        print(f"  {i:2d} len {len(ta):3d}/{len(tb):3d}  {'EXACT' if exact else f'diverge@{first_diff}'}"
              f"  KL {kl if kl is None else f'{kl:.2e}'}  max|dlogit| {maxabs if maxabs is None else f'{maxabs:.3f}'}"
              f"  top1 {top1 if top1 is None else f'{top1:.3f}'}")
    k = len(rows)
    kls = [r[4] for r in rows if r[4] is not None]
    print(f"\n{n_exact}/{k} prompts token-exact; mean KL over compared steps {sum(kls) / len(kls):.2e}" if kls else f"\n{n_exact}/{k} token-exact")
    return n_exact == k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "ours", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--ckpt", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--keep-logits", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=30)
    args = ap.parse_args()
    if args.mode == "compare":
        ok = compare(*args.files[:2])
        sys.exit(0 if ok else 1)
    if not args.out:
        sys.exit("--out required")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    (run_ref if args.mode == "ref" else run_ours)(args)


if __name__ == "__main__":
    main()
