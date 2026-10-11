#!/usr/bin/env python3
"""Decode step and speculative cycle times on the real model.

The bench times the plain decode graph (T=1) and the MTP cycle graph at draft lengths k, for one request at a given
context length, as the scheduler runs them. It uses the default decode weights: the INT6 / INT5 copies when present, or the
FP8 projections with --checkpoint-weights. The cycles run each draft step. With --early-exit, they use the early exit
of the drafter, which the dummy tokens of the bench would trigger on almost every step.

   uv run python bench/decode_bench.py [--ctx 8192] [--ks 1 3 5 7]
"""
import argparse

import torch

from bench.timing import timed
from engine.runtime.build import load_model
from engine.runtime.decode import DecodeGraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7, 15])
    ap.add_argument("--checkpoint-weights", action="store_true", help="decode the FP8 projections instead of the INT6 / INT5 copies")
    ap.add_argument("--early-exit", action="store_true", help="keep the drafter's early exit (engine/spec/mtp.py DRAFT_STOP)")
    a = ap.parse_args()
    path, model = load_model(decode_weights="checkpoint" if a.checkpoint_weights else "int")
    from engine.model.prefill import prepare_prefill
    from engine.spec import mtp as mtp_mod
    from engine.spec.mtp import Mtp, MtpCycle, MtpState
    if not a.early_exit:
        mtp_mod.DRAFT_STOP = 0.0
    prepare_prefill(model)
    print(f"context {a.ctx}")
    st = model.new_state(1, a.ctx + 64)
    g = DecodeGraph(model, st)
    st.pos_t.fill_(a.ctx)
    tok = torch.zeros(1, dtype=torch.long, device="cuda")

    def step():
        g.step(tok)
        st.pos_t.fill_(a.ctx)
    dt = timed(step)
    print(f"plain decode: {dt * 1e3:6.1f} ms/step  -> {1 / dt:5.1f} tok/s")
    mtp = Mtp(model, path)
    mst = MtpState(model.cfg, a.ctx + 64, "cuda", active=st.active)
    for k in [k for k in a.ks if k + 1 <= 16]:
        cyc = MtpCycle(model, mtp, st, mst, k)

        def run():
            cyc.graph.replay()
            st.pos_t.fill_(a.ctx)
        dt = timed(run)
        print(f"MTP cycle k={k}: {dt * 1e3:6.1f} ms/cycle ({k + 1} verify rows)", flush=True)


if __name__ == "__main__":
    main()
