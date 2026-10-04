"""Speculative decoding for one slot (PLAN.md Phase 4).

SpecGraph(k): one CUDA graph per draft length k: verify the k+1 rows [last token, d1..dk], greedy
acceptance on the GPU (the target's argmax must equal the next draft token, sequentially), commit the
accepted prefix (GDN conv / recurrent state, positions; KV beyond the accepted length is simply
overwritten later). Outputs: n = accepted inputs (1..k+1) and the target's predictions; the new tokens
are d1..d_{n-1} plus the bonus prediction pred[n-1].
"""
from __future__ import annotations

import torch

from engine.model.fast import DecodeGraph, FastQwen35, FastState
from engine.model.prefill import prefill, prepare_prefill


class SpecGraph:
    def __init__(self, model: FastQwen35, state: FastState, k: int):
        self.model, self.state, self.k = model, state, k
        dev = state.pos_t.device
        self.tok = torch.zeros(1, k + 1, dtype=torch.long, device=dev)
        state.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.pred, self.n = self._body()
        state.reset()

    def _body(self):
        logits = self.model.verify(self.tok, self.state)  # [1, k+1, V]
        pred = logits.argmax(-1)                            # [1, k+1]
        match = (pred[:, : self.k] == self.tok[:, 1:]).int()
        n = (1 + match.cumprod(-1).sum(-1)).int()           # [1] accepted inputs
        self.model.commit(self.state, n)
        return pred, n

    def step(self, tokens: list[int]):
        self.tok.copy_(torch.tensor([tokens], device=self.tok.device))
        self.graph.replay()
        n = int(self.n)
        pred = self.pred[0, :n].tolist()
        return n, pred


class SpecGenerator:
    """Greedy generation with n-gram speculation for one slot; output identical to plain greedy decode."""

    def __init__(self, model: FastQwen35, max_seq_len: int = 32768, k: int = 3):
        from engine.spec.ngram import NgramDrafter
        prepare_prefill(model)
        self.model, self.k = model, k
        self.state = model.new_state(1, max_seq_len)
        self.plain = DecodeGraph(model, self.state)
        self.spec = {j: SpecGraph(model, self.state, j) for j in range(1, k + 1)}
        self.drafter = NgramDrafter()
        self.stats = dict(steps=0, spec_steps=0, tokens=0, drafted=0, accepted=0)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, eos_ids=(), use_spec: bool = True):
        """use_spec=False: identical prefill and decode graph, no drafting (the exact baseline)."""
        st = self.state
        st.reset()
        st.pos = 0
        logits = prefill(self.model, torch.tensor([input_ids], device="cuda"), st)
        y = int(logits.argmax(-1))
        out = [y]
        self.drafter.reset(list(input_ids) + [y])
        eos = set(eos_ids)
        while len(out) < max_new_tokens and y not in eos:
            room = max_new_tokens - len(out)
            draft = self.drafter.propose(min(self.k, room - 1)) if (room > 1 and use_spec) else []
            self.stats["steps"] += 1
            if draft:
                n, pred = self.spec[len(draft)].step([y] + draft)
                new = draft[: n - 1] + [pred[n - 1]]
                self.stats["spec_steps"] += 1
                self.stats["drafted"] += len(draft)
                self.stats["accepted"] += n - 1
                st.pos += len(new)
            else:
                new = [int(self.plain.step(torch.tensor([y], device="cuda")))]  # advances st.pos itself
            for t in new:  # stop at EOS / budget inside an accepted run
                out.append(t)
                if t in eos or len(out) >= max_new_tokens:
                    break
            y = out[-1]
            self.drafter.extend(new)
        self.stats["tokens"] += len(out)
        return out
