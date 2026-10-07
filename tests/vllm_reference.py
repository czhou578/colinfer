#!/usr/bin/env python3
"""vLLM side of the quantized-path parity check (PLAN.md 4.7b). It runs in the vLLM 0.25.1 venv, NOT in the
engine venv, and it imports nothing from engine/.

  VLLM_USE_FASTOKENS=0 ~/Projects/model-benchmarks/.venv/bin/python tests/vllm_reference.py \
      --ckpt nvidia/Qwen3.8-27B-NVFP4 --out tests/parity_out/vllm_nvfp4.json

It writes JSON with these items per prompt: the input ids, the greedy tokens (128), and the top-20 logprobs of the
first 32 steps. It also writes the WikiText perplexity from the prompt logprobs, over the same windows as
tests/perplexity.py.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.prompts import PROMPTS  # plain list, no engine imports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--keep-logprobs", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=30)
    ap.add_argument("--ppl-ctx", type=int, default=2048)
    ap.add_argument("--ppl-max-tokens", type=int, default=65536)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    import vllm

    path = snapshot_download(args.ckpt, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, tokenizer=path, max_model_len=args.ppl_ctx + 256, gpu_memory_utilization=0.5,
              kv_cache_dtype="auto", enable_prefix_caching=False, max_num_seqs=4, limit_mm_per_prompt={"image": 0, "video": 0})
    print(f"[vllm] {vllm.__version__} loaded {args.ckpt}")
    gc = json.load(open(os.path.join(path, "generation_config.json")))
    eos = gc.get("eos_token_id", [])
    eos = [eos] if isinstance(eos, int) else list(eos)

    prompts = []
    for p in PROMPTS[: args.prompts]:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        prompts.append(list(map(int, ids)))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, logprobs=20, stop_token_ids=eos)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp)
    results = []
    for ids, o in zip(prompts, outs):
        c = o.outputs[0]
        steps = []
        for lp in (c.logprobs or [])[: args.keep_logprobs]:
            steps.append({str(t): v.logprob for t, v in lp.items()})
        results.append(dict(input_ids=ids, tokens=list(map(int, c.token_ids)), top_logprobs=steps))
    print(f"[vllm] generated {len(results)} prompts in {time.time() - t0:.0f}s")

    # perplexity over the same windows as tests/perplexity.py
    import pyarrow.parquet as pq
    f = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/test-*.parquet")))[0]
    text = "\n\n".join(pq.read_table(f).column("text").to_pylist())
    wids = tok(text).input_ids
    n_win = min(args.ppl_max_tokens, len(wids)) // args.ppl_ctx
    windows = [wids[w * args.ppl_ctx:(w + 1) * args.ppl_ctx] for w in range(n_win)]
    t0 = time.time()
    pouts = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows], SamplingParams(max_tokens=1, prompt_logprobs=0))
    nll, count = 0.0, 0
    for w, o in zip(windows, pouts):
        for pos in range(1, len(w)):
            nll -= o.prompt_logprobs[pos][w[pos]].logprob
            count += 1
    ppl = math.exp(nll / count)
    print(f"[vllm] RESULT ppl={ppl:.4f} tokens={count} ({time.time() - t0:.0f}s)")
    json.dump(dict(side="vllm", vllm=vllm.__version__, ckpt=args.ckpt, eos=eos, results=results,
                   ppl=dict(ctx=args.ppl_ctx, tokens=count, ppl=ppl, nll=nll / count)), open(args.out, "w"))
    print(f"[vllm] wrote {args.out}")


if __name__ == "__main__":
    main()
