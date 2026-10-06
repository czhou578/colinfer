#!/usr/bin/env python3
"""Compare our model (from `tests/parity_hf.py ours --ckpt nvidia/Qwen3.8-27B-NVFP4`) against vLLM
serving the same checkpoint (`tests/vllm_reference.py`).

KL is computed over vLLM's top-20 support: sum_t p_v(t) * (log p_v(t) - log p_ours(t)), where
log p_ours comes from our full-vocabulary log-softmax. The probability mass vLLM puts outside its
top 20 is reported alongside; when it is small the truncated KL is a close lower bound.

  uv run python tests/parity_vllm.py tests/parity_out/vllm_nvfp4.json tests/parity_out/ours_nvfp4.pt
"""
import json
import sys

import torch


def main(vpath, opath):
    v = json.load(open(vpath))
    o = torch.load(opath)
    print(f"vLLM {v['vllm']} ({v['ckpt']})  vs  ours ({o['ckpt']})")
    n_exact, kls, tails, prefix_lens = 0, [], [], []
    for i, (rv, ro) in enumerate(zip(v["results"], o["results"])):
        assert rv["input_ids"] == ro["input_ids"].tolist(), f"prompt {i}: input ids differ"
        tv, to = rv["tokens"], ro["tokens"]
        n = min(len(tv), len(to))
        fd = next((j for j in range(n) if tv[j] != to[j]), None)
        exact = fd is None and len(tv) == len(to)
        n_exact += exact
        prefix_lens.append(n if fd is None else fd)
        m = min(len(rv["top_logprobs"]), 0 if ro["logits"] is None else ro["logits"].shape[0])
        m = min(m, fd if fd is not None else m)
        kl_p = []
        for s in range(m):
            lo = torch.log_softmax(ro["logits"][s].float(), -1)
            top = rv["top_logprobs"][s]
            ids = torch.tensor([int(t) for t in top])
            lv = torch.tensor(list(top.values()))
            pv = lv.exp()
            kl_p.append(float((pv * (lv - lo[ids])).sum()))
            tails.append(1.0 - float(pv.sum()))
        kl = sum(kl_p) / len(kl_p) if kl_p else None
        if kl is not None:
            kls.append(kl)
        print(f"  {i:2d} len {len(tv):3d}/{len(to):3d}  {'EXACT' if exact else f'diverge@{fd}'}"
              f"  KL(top20) {'-' if kl is None else f'{kl:.2e}'} over {m} steps")
    k = len(v["results"])
    print(f"\n{n_exact}/{k} prompts token-exact; mean matching prefix {sum(prefix_lens) / k:.1f} tokens")
    if kls:
        print(f"mean KL(vLLM || ours) over top-20 support: {sum(kls) / len(kls):.3e}; "
              f"median vLLM tail mass outside top 20: {sorted(tails)[len(tails) // 2]:.2e}")
    print(f"vLLM WikiText ppl: {v['ppl']['ppl']:.4f} (ctx {v['ppl']['ctx']}, {v['ppl']['tokens']} tokens)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
