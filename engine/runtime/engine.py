"""Multi-slot engine (PLAN.md 4.4 items 5-6, Phase 3 week 10): up to `n_slots` concurrent requests.

* One batched FastState holds every slot (KV cache, GDN conv / recurrent state, positions).
* Decode: one CUDA graph per batch width n (1..n_slots) over the view of slots [0, n); slots that are
  idle or still prefilling are masked off (state.active), so the step leaves them untouched.
* Prefill: chunk by chunk on a single-slot view (engine/model/prefill.py). Each engine step runs at
  most one prefill chunk, then one decode step for every decoding slot, so a long prompt never stalls
  the other users for more than one chunk.
* Prefix checkpoints: snapshots of a slot's GDN state (conv + recurrent, ~154 MB) at prefill chunk
  boundaries, at the end of each prompt and at the end of each generation. The attention KV can be
  resumed at any length, but GDN state only where a snapshot exists. A new request goes to the free
  slot holding the longest valid checkpoint that is a prefix of its prompt, restores it, and prefills
  only the remainder: multi-turn chat pays only for the new turn. A checkpoint is valid while its
  slot's token history still starts with the checkpoint's tokens.
"""
from __future__ import annotations

import collections
import dataclasses
import time

import torch

from engine.model.fast import DecodeGraph, FastQwen35
from engine.model.prefill import CHUNK, prefill, prepare_prefill
from engine.runtime.sampler import SamplerParams, sample


@dataclasses.dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = 0
    eos_ids: tuple = ()
    # filled in by the engine
    rid: int = -1
    slot: int = -1
    output: list = dataclasses.field(default_factory=list)
    done: bool = False
    reused: int = 0          # prompt tokens served from a checkpoint
    t_submit: float = 0.0
    t_first: float = 0.0     # time of the first generated token
    t_done: float = 0.0


@dataclasses.dataclass
class Checkpoint:
    slot: int
    tokens: tuple            # the token history this GDN state corresponds to
    conv: list               # per GDN layer [C, K-1] bf16
    rec: list                # per GDN layer [Hv, dk, dv] fp32


class Slot:
    def __init__(self, idx):
        self.idx = idx
        self.tokens: list[int] = []   # tokens currently represented in this slot's KV / state
        self.req: Request | None = None
        self.phase = "idle"           # idle | prefill | decode
        self.todo: list[int] = []     # prompt tokens still to prefill
        self.last_logits = None
        self.next_token = None


class Engine:
    def __init__(self, model: FastQwen35, n_slots: int = 3, max_seq_len: int = 32768, n_checkpoints: int = 32, selftest: bool = True):
        if selftest:
            from engine.selftest import run_selftest
            run_selftest(verbose=True)  # refuses to start if any matmul path is numerically wrong
        self.model = model
        prepare_prefill(model)
        self.state = model.new_state(n_slots, max_seq_len)
        self.max_seq_len = max_seq_len
        self.slots = [Slot(i) for i in range(n_slots)]
        self.params = SamplerParams(n_slots, model.cfg.vocab_size, self.state.pos_t.device)
        self.graphs = {n: DecodeGraph(model, self.state.view(0, n), self.params.view(0, n)) for n in range(1, n_slots + 1)}
        self.params.offset.zero_()
        self.state.reset()
        self.state.active.zero_()
        self.gdn_layers = sorted(self.state.rec)
        self.ckpts: collections.deque[Checkpoint] = collections.deque(maxlen=n_checkpoints)
        self.queue: collections.deque[Request] = collections.deque()
        self.finished: list[Request] = []
        self._rid = 0

    # ---------------------------------------------------------------- checkpoints
    def _snapshot(self, s: Slot):
        if not s.tokens:
            return
        key = tuple(s.tokens)
        for c in self.ckpts:
            if c.slot == s.idx and c.tokens == key:
                return
        b = s.idx
        self.ckpts.append(Checkpoint(b, key, [self.state.conv[i][b].clone() for i in self.gdn_layers],
                                     [self.state.rec[i][b].clone() for i in self.gdn_layers]))

    def _restore(self, s: Slot, c: Checkpoint):
        b = s.idx
        for j, i in enumerate(self.gdn_layers):
            self.state.conv[i][b].copy_(c.conv[j])
            self.state.rec[i][b].copy_(c.rec[j])
        s.tokens = list(c.tokens)
        self.state.pos_t[b] = len(c.tokens)

    def _valid(self, c: Checkpoint) -> bool:
        t = self.slots[c.slot].tokens
        return len(t) >= len(c.tokens) and tuple(t[: len(c.tokens)]) == c.tokens

    # ---------------------------------------------------------------- admission
    def submit(self, req: Request) -> Request:
        req.rid, self._rid = self._rid, self._rid + 1
        req.t_submit = time.perf_counter()
        if len(req.prompt) + req.max_new_tokens > self.max_seq_len:
            raise ValueError("request exceeds the slot length")
        self.queue.append(req)
        return req

    def _admit(self):
        while self.queue:
            free = [s for s in self.slots if s.phase == "idle"]
            if not free:
                return
            req = self.queue.popleft()
            best, best_len = None, 0
            for c in self.ckpts:  # longest valid checkpoint (in a free slot) that is a proper prefix of the prompt
                L = len(c.tokens)
                if L > best_len and L < len(req.prompt) and self.slots[c.slot].phase == "idle" and self._valid(c) \
                        and tuple(req.prompt[:L]) == c.tokens:
                    best, best_len = c, L
            s = self.slots[best.slot] if best else min(free, key=lambda x: (len(x.tokens) > 0, x.idx))
            if best:
                self._restore(s, best)
            else:
                b = s.idx
                for i in self.gdn_layers:
                    self.state.conv[i][b].zero_()
                    self.state.rec[i][b].zero_()
                s.tokens = []
                self.state.pos_t[b] = 0
            # drop checkpoints of this slot that the new history will overwrite
            keep = len(s.tokens)
            self.ckpts = collections.deque([c for c in self.ckpts if c.slot != s.idx or len(c.tokens) <= keep], maxlen=self.ckpts.maxlen)
            req.slot, req.reused = s.idx, len(s.tokens)
            s.req, s.phase, s.todo = req, "prefill", list(req.prompt[len(s.tokens):])
            self.params.set(s.idx, req.temperature, req.top_k, req.top_p, req.min_p, req.seed)

    # ---------------------------------------------------------------- one engine iteration
    @torch.inference_mode()
    def step(self):
        self._admit()
        # 1) at most one prefill chunk
        pre = next((s for s in self.slots if s.phase == "prefill"), None)
        if pre is not None:
            b = pre.idx
            chunk = pre.todo[:CHUNK]
            view = self.state.view(b, b + 1)
            view.pos = len(pre.tokens)
            logits = prefill(self.model, torch.tensor([chunk], device="cuda"), view)
            pre.tokens += chunk
            pre.todo = pre.todo[len(chunk):]
            self._snapshot(pre)
            if not pre.todo:
                tok = int(sample(logits, self.params.view(b, b + 1)))
                pre.next_token, pre.phase = tok, "decode"
                self._emit(pre, tok)
        # 2) one decode step for every decoding slot
        dec = [s for s in self.slots if s.phase == "decode"]
        if dec:
            n = max(s.idx for s in dec) + 1
            g = self.graphs[n]
            act = torch.zeros(len(self.slots), dtype=torch.int32)
            toks = torch.zeros(n, dtype=torch.long)
            for s in dec:
                act[s.idx] = 1
                toks[s.idx] = s.next_token
            self.state.active.copy_(act)
            g.state.pos = 0  # per-slot limits are enforced here; the graph's own counter is not meaningful
            out = g.step(toks.cuda()).tolist()
            self.state.active.zero_()
            for s in dec:
                s.tokens.append(s.next_token)
                s.next_token = out[s.idx]
                self._emit(s, out[s.idx])

    def _emit(self, s: Slot, tok: int):
        req = s.req
        if not req.output:
            req.t_first = time.perf_counter()
        req.output.append(tok)
        if tok in req.eos_ids or len(req.output) >= req.max_new_tokens or len(s.tokens) + 1 >= self.max_seq_len:
            req.done, req.t_done = True, time.perf_counter()
            self._snapshot(s)  # end of generation: prompt + generated tokens (all but the last, which was never fed)
            s.phase, s.req = "idle", None
            self.finished.append(req)

    def busy(self) -> bool:
        return bool(self.queue) or any(s.phase != "idle" for s in self.slots)

    def run(self, requests: list[Request]) -> list[Request]:
        for r in requests:
            self.submit(r)
        while self.busy():
            self.step()
        return requests
