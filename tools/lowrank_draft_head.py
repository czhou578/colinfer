#!/usr/bin/env python3
"""Low-rank draft head (docs/history/phase6_progress.md section 19): the PCA basis of the MTP drafter's normed outputs.

The draft lm head (64k static + 4k prompt rows, NVFP4, ~200 MB) is streamed once per draft step. Its argmax is
nearly always among the top few candidates of the rank-r approximation  g U (W U)^T  with U the top-r principal
directions of the drafter outputs g, so the engine scores the approximation (U^T: r x 5120, W U: V x r, NVFP4) and
rescores the top candidates exactly against the real rows (engine/spec/mtp.py).

Collects g from real greedy k=7 cycles (tools/drafter_data.py prompts, a seed the evals do not use), fits U on two
thirds, reports how often the full head's argmax is in the approximation's top K on the rest, and saves U.

   uv run python tools/lowrank_draft_head.py [--n 40] [--rank 1024] [--weights auto]
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_OUT = os.path.expanduser("~/.cache/colinfer/drafter/draft_head_pca.safetensors")


def collect(n: int, seed: int, weights, k: int = 7, max_new: int = 256) -> torch.Tensor:
    from transformers import AutoTokenizer

    from drafter_data import build_prompts
    from engine.model.fast import load_fast_model, to_fast
    from engine.spec import mtp as M
    from engine.weights.loader import resolve
    gbuf = torch.zeros(k, 5120, device="cuda", dtype=torch.bfloat16)
    calls = [0]
    orig = M.Mtp.draft

    def draft(self, g):  # captured into the cycle graph: draft step j copies its input into gbuf[j]
        gbuf[calls[0] % k].copy_(g.reshape(-1, g.shape[-1])[0])
        calls[0] += 1
        return orig(self, g)
    M.Mtp.draft = draft
    path = resolve("nvidia/Qwen3.8-27B-NVFP4")
    tok = AutoTokenizer.from_pretrained(path)
    model = to_fast(load_fast_model(path))
    M.DRAFT_STOP = 0.0  # every draft step runs: all drafter outputs are real
    gen = M.MtpGenerator(model, path, max_seq_len=4096, k=k, weights=weights, lowrank=None)  # collect with the full head
    G = []
    for p, kind, think in build_prompts(n, random.Random(seed)):
        x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=think, tokenize=True)
        x = list(x["input_ids"] if hasattr(x, "keys") else x)[-3000:]
        st, mst, cyc = gen.state, gen.mst, gen.cycle
        st.reset(); st.pos = 0; mst.pos_t.zero_(); gen.mtp.set_prompt_vocab(x)
        logits, H = M.prefill(gen.model, torch.tensor([x], device="cuda"), st, return_hidden=True)
        y = int(logits.argmax(-1)); out = [y]
        cyc.tok.copy_(torch.tensor([[y] + gen.mtp.first_drafts(torch.tensor(x[1:] + [y], device="cuda"), H, mst, k)], device="cuda"))
        while len(out) < max_new and y not in (248046, 248044):
            cyc.graph.replay()
            n_ = int(cyc.n)
            G.append(gbuf.clone())
            out += cyc.out_tok[0, :n_].tolist()
            y = out[-1]
        print(f"  {kind}: {len(out)} tokens, {len(G) * k} samples", flush=True)
    M.Mtp.draft = orig
    lm = gen.mtp.lm_draft
    from engine.weights.loader import dequant_nvfp4
    W = dequant_nvfp4(lm.w[: gen.mtp.n_static], lm.sf[: gen.mtp.n_static], torch.tensor(lm.gscale), out_dtype=torch.float32)
    return torch.cat(G).float(), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="prompts (tools/drafter_data.py mix)")
    ap.add_argument("--seed", type=int, default=5, help="prompt seed (the evals use 1)")
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--weights", default="auto", help="MTP head weights: auto = ~/.cache/colinfer/drafter/mtp_ft.safetensors if present")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    w = a.weights
    if w == "auto":
        w = os.path.expanduser("~/.cache/colinfer/drafter/mtp_ft.safetensors")
        w = w if os.path.exists(w) else None
    g, W = collect(a.n, a.seed, w)
    n = g.shape[0]
    tr, te = g[: n * 2 // 3], g[n * 2 // 3:]
    ev, U = torch.linalg.eigh(tr.T @ tr / tr.shape[0])
    U, ev = U.flip(-1), ev.flip(-1)
    print(f"{n} samples; drafter-output energy in the top {a.rank} directions: {float(ev[:a.rank].sum() / ev.sum()):.4f}")
    top1 = (te @ W.T).argmax(-1)
    for r in sorted({a.rank // 2, a.rank, a.rank * 2}):
        Ur = U[:, :r]
        approx = (te @ Ur) @ (W @ Ur).T
        rec = [float((approx.topk(K, -1).indices == top1[:, None]).any(-1).float().mean()) for K in (16, 64, 256)]
        print(f"rank {r}: full head's argmax in the approximation's top 16 / 64 / 256: " + " / ".join(f"{x:.4f}" for x in rec))
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file({"U": U[:, : a.rank].contiguous().bfloat16().cpu()}, a.out, metadata={"rank": str(a.rank), "samples": str(n),
                                                                                       "weights": str(w)})
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
