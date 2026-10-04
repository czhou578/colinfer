"""Single-slot generation on the Phase 2 decode path: eager chunked prefill (<= 16 tokens per
forward, the GEMVs take 4 rows at a time), then one CUDA-graph replay per decode token."""
from __future__ import annotations

import torch

from engine.model.fast import DecodeGraph, FastQwen35

PREFILL_CHUNK = 16


class FastGenerator:
    def __init__(self, model: FastQwen35, max_seq_len: int = 4096, use_graph: bool = True):
        self.model = model
        self.state = model.new_state(1, max_seq_len)
        self.graph = DecodeGraph(model, self.state) if use_graph else None

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_ids=(), keep_logits: int = 0):
        st, m = self.state, self.model
        st.reset()
        st.pos = 0
        logits = None
        for i in range(0, input_ids.shape[1], PREFILL_CHUNK):
            logits = m(input_ids[:, i:i + PREFILL_CHUNK], st, last_only=True)[:, -1]
        tok = logits.argmax(-1)
        out, kept = [], []
        eos = set(int(e) for e in eos_ids)
        for step in range(max_new_tokens):
            if step < keep_logits:
                kept.append(logits[0].clone())
            t = int(tok)
            out.append(t)
            if t in eos or step == max_new_tokens - 1:
                break
            if self.graph is not None:
                tok = self.graph.step(tok)
                logits = self.graph.logits
            else:
                logits = m(tok.view(1, 1), st, last_only=True)[:, -1]
                tok = logits.argmax(-1)
        return out, (torch.stack(kept) if kept else None)
