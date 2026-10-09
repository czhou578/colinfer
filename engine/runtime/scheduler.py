"""Request scheduler: up to `n_slots` concurrent requests, driven by one engine thread (engine/server/api.py).

* State. One batched FastState holds all slots (KV cache, GDN conv / recurrent state, positions). With speculation, a
  batched MtpState also holds the own KV cache of the drafter.
* Decode with MTP (the default). One CUDA graph per (contiguous slot range [lo, hi), greedy | sampled, draft length k)
  runs a full speculative cycle for those slots: verify, acceptance, commit and drafts (engine/spec/mtp.py). Without
  MTP, each range has one plain decode graph.
  - A step uses the smallest range that covers the decoding slots. It masks off the idle and prefilling slots inside
    the range (state.active), so a step never touches them.
  - Each cycle picks k (3 or 7, when the verify rows fit one weight pass) for the most expected tokens per second. It
    uses the running acceptance rate of each slot and the cycle times measured at startup. Code and structured output
    draft 7, and prose drafts 3.
  - With suffix_min > 0, the history of a slot can end in a repeat of at least suffix_min tokens. Then the slot drafts
    what followed the earlier occurrence instead (engine/spec/suffix.py), up to SUFFIX_K = 15 drafts when it decodes
    alone.
  - The greedy output does not depend on the other slots: each kernel computes the rows of a slot the same way at any
    batch width.
* Prefill. Each engine step runs at most one chunk on a single-slot view (engine/model/prefill.py), then one decode
  step for the decoding slots. Thus a long prompt stops the other slots for one chunk at a time. With MTP, the
  scheduler writes the KV rows of the drafter for the chunk right after the chunk.
* Prefix checkpoints. A ring, preallocated at startup, holds GDN state snapshots (conv + recurrent, 154 MB).
  - The scheduler takes a snapshot at the prompt end, at the generation end and every `ckpt_interval` prompt tokens.
  - It also takes snapshots at two chat message boundaries (`boundary_token`, <|im_start|>). These are the end of the
    first message (a shared system prompt), and the start of the last message before the generation prompt (the
    conversation so far). Prefill chunks end there.
  - Attention KV can resume at any length, but GDN state only where a snapshot exists. A new request restores the
    longest checkpoint whose tokens are a proper prefix of its prompt, and prefills only the rest.
  - If the slot of that checkpoint is busy, the scheduler copies its KV prefix into a free slot (32 KB per token with
    FP8 KV). Thus concurrent requests that share a long system prompt each pay for it once.
  - A checkpoint is valid while the token history of its slot still starts with the tokens of the checkpoint.
* Finish. A request ends at a stop token (eos_ids), at max_new_tokens, at the slot length, or by its hook (stop
  strings, client gone). The speculative cycle cuts the accepted length after a stop token on the GPU. Thus the
  history of a slot (`Slot.tokens`, exactly the tokens fed to the model) normally ends at the last token of the reply.
  Then the end-of-generation checkpoint is a prefix of the prompt of the next turn.
"""
from __future__ import annotations

import collections
import dataclasses
import time
from typing import Any

import torch

from engine.model.fast import MAX_ROWS, DecodeGraph, FastQwen35
from engine.model.prefill import CHUNK, prefill, prepare_prefill
from engine.runtime.metrics import Metrics
from engine.runtime.sampler import SamplerParams, sample
from engine.spec.suffix import SuffixIndex

MAX_STOP_IDS = 8  # stop token ids per slot visible to the GPU-side cut (more are still honored on the host)
K_OPTIONS = (3, 7)  # draft lengths a cycle chooses between (capped by k)
ACC_DECAY = 0.85    # per-cycle decay of a slot's draft-acceptance statistics
SUFFIX_K = MAX_ROWS - 1  # suffix-match draft length of a lone decoding slot: its verify rows still fit one weight pass
ATTN_ROWS = 48      # query rows (q heads x verify rows) per pass of the attention kernel over a slot's KV (csrc/attn_decode.cu)


@dataclasses.dataclass
class Request:
    prompt: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = 0
    eos_ids: tuple = ()             # stop token ids (finish_reason "stop"; the token is part of the output)
    min_tokens: int = 0             # stop tokens do not end the request before this many output tokens
    logprobs: int | None = None     # None: off; n >= 0: logprob of each output token plus the top n alternatives
    hook: Any = None                # optional: hook.feed(token, logprob) -> True to stop now; hook.finish(request)
    cache_salt: str | None = None   # checkpoints are shared only between requests with the same salt (vLLM semantics)
    # filled in by the scheduler
    rid: int = -1
    slot: int = -1
    output: list = dataclasses.field(default_factory=list)
    output_logprobs: list = dataclasses.field(default_factory=list)  # (logprob, [(id, logprob), ...]) per output token
    done: bool = False
    finish_reason: str | None = None  # stop | length | abort
    reused: int = 0                 # prompt tokens served from a checkpoint
    t_submit: float = 0.0
    t_admit: float = 0.0
    t_first: float = 0.0            # first output token
    t_done: float = 0.0


@dataclasses.dataclass
class Checkpoint:
    slot: int
    tokens: tuple   # the token history the snapshot corresponds to
    buf: int        # ring index
    salt: str | None = None


class Slot:
    def __init__(self, idx):
        self.idx = idx
        self.tokens: list[int] = []  # tokens fed to the model in this slot (its KV / state history)
        self.req: Request | None = None
        self.phase = "idle"          # idle | prefill | decode
        self.todo: list[int] = []    # prompt tokens still to prefill
        self.splits: list[int] = []  # prompt positions to end a prefill chunk at and snapshot
        self.y = None                # last output token, not yet fed
        self.h_last = None           # MTP: target post-norm hidden state at the last fed position
        self.last_used = 0.0
        self.acc = {}                      # draft source -> [decayed accepted, tried] drafts (per-token acceptance estimate)
        self.src = "mtp"                   # source of the drafts in the slot's pending input: mtp | suffix
        self.fresh = 0                     # drafts in the slot's pending input made for it (the rest are stale)
        self.sfx = None                    # suffix-match index of the history (suffix_min > 0)
        self.salt = None             # cache_salt of the request that produced this slot's history


class Scheduler:
    def __init__(self, model: FastQwen35, n_slots: int = 3, max_seq_len: int = 32768, n_checkpoints: int = 32, mtp=None, k: int = 3,
                 prefill_chunk: int = CHUNK, ckpt_interval: int = 8192, selftest: bool = True, metrics: Metrics | None = None,
                 keep_finished: bool = True, boundary_token: int | None = None, suffix_min: int = 0):
        if selftest:
            from engine.selftest import run_selftest
            run_selftest(verbose=True)  # refuses to start if any matmul path is numerically wrong
        prepare_prefill(model)
        self.model, self.mtp, self.k = model, mtp, k
        self.suffix_min = suffix_min if mtp is not None else 0  # > 0: suffix-match drafts of at least this match length
        self.kmax = max(k, SUFFIX_K) if self.suffix_min else k   # longest draft a cycle verifies
        self.n_slots, self.max_seq_len, self.chunk, self.ckpt_interval = n_slots, max_seq_len, prefill_chunk, ckpt_interval
        self.metrics = metrics or Metrics()
        self.keep_finished = keep_finished
        self.boundary = boundary_token
        self.state = model.new_state(n_slots, max_seq_len)
        dev = self.dev = self.state.pos_t.device
        cfg = model.cfg
        self.params = SamplerParams(n_slots, cfg.vocab_size, dev)
        self.margin = (self.kmax + 1) if mtp is not None else 1  # positions one decode step writes beyond the history
        if mtp is not None:
            from engine.spec.mtp import MtpCycle, MtpState
            self.mst = MtpState(cfg, max_seq_len, dev, batch=n_slots, active=self.state.active)
            self.tok = torch.zeros(n_slots, self.kmax + 1, dtype=torch.long, device=dev)
            self.stop_buf = torch.full((n_slots, MAX_STOP_IDS), -1, dtype=torch.long, device=dev)
            self.cycles, self.cycle_s = {}, {}
            for lo, hi in self._ranges():
                sv, mv = self.state.view(lo, hi), self.mst.view(lo, hi)
                for kw in self.k_options(hi - lo) + ([SUFFIX_K] if self.kmax > k and hi - lo == 1 else []):
                    for sampled in (False, True):
                        self.cycles[(lo, hi, sampled, kw)] = MtpCycle(model, mtp, sv, mv, kw, params=self.params.view(lo, hi) if sampled else None,
                                                                      tok=self.tok[lo:hi, :kw + 1], stop_ids=self.stop_buf[lo:hi],
                                                                      drafts=min(kw, k))
                    if lo == 0:  # cycle time per (width, k), for choosing k (the state is garbage here and reset below)
                        g = self.cycles[(lo, hi, False, kw)]
                        g.graph.replay()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(3):
                            g.graph.replay()
                        torch.cuda.synchronize()
                        self.cycle_s[(hi - lo, kw)] = (time.perf_counter() - t0) / 3
            self.mst.pos_t.zero_()
            # cycle_s is measured at ~0 context; per token of a slot's context a cycle also reads the target's KV once
            # (multi-row attention: whatever k) and the drafter's KV once per draft step (seconds per token, ~240 GB/s)
            kv = lambda d: sum(t.shape[1] * t.shape[-1] * t.element_size() * 2 for t in d.values())  # noqa: E731 (K + V)
            self.ctx_s = kv(self.state.k) / 2.4e11
            self.ctx_draft_s = kv(self.mst.k) / 2.4e11
            self.q_per_kv = cfg.num_attention_heads // cfg.num_key_value_heads
        else:
            self.graphs = {(lo, hi): DecodeGraph(model, self.state.view(lo, hi), self.params.view(lo, hi)) for lo, hi in self._ranges()}
        self.state.reset()
        self.state.active.zero_()
        self.slots = [Slot(i) for i in range(n_slots)]
        # checkpoint ring, allocated once
        self.gdn_layers = sorted(self.state.rec)
        self.ring_conv = {i: torch.zeros((n_checkpoints,) + self.state.conv[i].shape[1:], dtype=self.state.conv[i].dtype, device=dev)
                          for i in self.gdn_layers}
        self.ring_rec = {i: torch.zeros((n_checkpoints,) + self.state.rec[i].shape[1:], dtype=self.state.rec[i].dtype, device=dev)
                         for i in self.gdn_layers}
        self.ring_h = torch.zeros(n_checkpoints, cfg.hidden_size, dtype=torch.bfloat16, device=dev) if mtp is not None else None
        self.ckpts: list[Checkpoint] = []  # oldest first
        self.free_bufs = list(range(n_checkpoints))
        self.queue: collections.deque[Request] = collections.deque()
        self.finished: list[Request] = []
        self._rid = 0
        self._evt = torch.cuda.Event()
        torch.cuda.synchronize()

    def k_options(self, width: int) -> list[int]:
        """Draft lengths with a graph at this batch width: those of K_OPTIONS (capped by k) whose verify rows
        (width * (k+1)) fit one weight pass of the skinny GEMM (MAX_ROWS = 16), since a second pass streams every weight
        again. A slot's pending drafts are a chain, so a shorter k uses a prefix."""
        opts = sorted({min(kk, self.k) for kk in K_OPTIONS + (self.k,)})
        fit = [kk for kk in opts if width * (kk + 1) <= MAX_ROWS]
        return fit or [max(1, MAX_ROWS // width - 1)]

    def _pick_k(self, width: int, dec) -> int:
        """The draft length with the most expected tokens per second: sum over slots of (1 - a^(k+1)) / (1 - a), a = the
        slot's per-token acceptance estimate, over the cycle time: measured at startup, plus the KV reads that grow with
        the slots' context (the target's once per ATTN_ROWS query rows, the drafter's once per MTP draft step; per slot at
        128k: ~17 ms + ~1.1 ms per draft step)."""
        opts = self.k_options(width)
        if self.kmax > self.k and width == 1 and dec[0].src == "suffix":
            opts = opts + [self.kmax]
        if len(opts) == 1:
            return opts[0]

        def expected(s, kk):
            acc, tried = s.acc[s.src]
            a = min(acc / max(tried, 1e-6), 0.999)
            if s.src == "suffix":  # a suffix match may have fewer tokens to offer
                kk = min(kk, s.fresh)
            return (1 - a ** (kk + 1)) / (1 - a)
        ctx = sum(len(s.tokens) for s in dec)
        passes = lambda kk: -(-self.q_per_kv * (kk + 1) // ATTN_ROWS)  # noqa: E731 (target KV reads per cycle)
        return max(opts, key=lambda kk: sum(expected(s, kk) for s in dec) /
                   (self.cycle_s[(width, kk)] + ctx * (passes(kk) * self.ctx_s + min(kk, self.k) * self.ctx_draft_s)))

    def _ranges(self):
        """Every contiguous slot range [lo, hi): a step runs the graph of the smallest range covering the decoding
        slots, so one active request costs a width-1 step whichever slot it is in."""
        return [(lo, hi) for lo in range(self.n_slots) for hi in range(lo + 1, self.n_slots + 1)]

    # ---------------------------------------------------------------- checkpoints
    def _valid(self, c: Checkpoint) -> bool:
        t = self.slots[c.slot].tokens
        return len(t) >= len(c.tokens) and tuple(t[: len(c.tokens)]) == c.tokens

    def _drop(self, c: Checkpoint):
        self.ckpts.remove(c)
        self.free_bufs.append(c.buf)

    def _snapshot(self, s: Slot):
        if not s.tokens or (not self.free_bufs and not self.ckpts):
            return
        key = tuple(s.tokens)
        if any(c.slot == s.idx and c.tokens == key and c.salt == s.salt for c in self.ckpts):
            return
        if not self.free_bufs:  # evict a checkpoint that no longer matches its slot, else the oldest
            self._drop(next((c for c in self.ckpts if not self._valid(c)), self.ckpts[0]))
        buf, b = self.free_bufs.pop(), s.idx
        for i in self.gdn_layers:
            self.ring_conv[i][buf].copy_(self.state.conv[i][b])
            self.ring_rec[i][buf].copy_(self.state.rec[i][b])
        if self.ring_h is not None:
            self.ring_h[buf].copy_(s.h_last.view(-1))
        self.ckpts.append(Checkpoint(b, key, buf, s.salt))

    def _restore(self, s: Slot, c: Checkpoint):
        b, L = s.idx, len(c.tokens)
        if c.slot != b:  # the checkpoint's slot is busy: copy its KV prefix here
            for d in (self.state.k, self.state.v):
                for t in d.values():
                    t[b, :, :L].copy_(t[c.slot, :, :L])
            if self.mtp is not None:
                for d in (self.mst.k, self.mst.v):
                    d[0][b, :, :L].copy_(d[0][c.slot, :, :L])
        for i in self.gdn_layers:
            self.state.conv[i][b].copy_(self.ring_conv[i][c.buf])
            self.state.rec[i][b].copy_(self.ring_rec[i][c.buf])
        s.tokens = list(c.tokens)
        s.h_last = self.ring_h[c.buf].clone() if self.ring_h is not None else None
        self.state.pos_t[b] = L

    # ---------------------------------------------------------------- admission
    def submit(self, req: Request) -> Request:
        if not req.prompt:
            raise ValueError("empty prompt")
        if len(req.prompt) + self.margin > self.max_seq_len:
            raise ValueError(f"prompt of {len(req.prompt)} tokens exceeds the slot length {self.max_seq_len}")
        V = self.model.cfg.vocab_size  # an id past the embedding is a device-side assert, which ends the engine
        if min(req.prompt) < 0 or max(req.prompt) >= V or not all(0 <= t < V for t in req.eos_ids):
            raise ValueError(f"token ids must be in [0, {V})")
        req.rid, self._rid = self._rid, self._rid + 1
        req.t_submit = time.perf_counter()
        req.eos_ids = tuple(req.eos_ids)
        self.queue.append(req)
        return req

    def abort(self, rid: int) -> bool:
        for r in self.queue:
            if r.rid == rid:
                self.queue.remove(r)
                self._close(r, "abort")
                return True
        for s in self.slots:
            if s.req is not None and s.req.rid == rid:
                self._finish(s, "abort")
                return True
        return False

    def _admit(self):
        while self.queue:
            free = [s for s in self.slots if s.phase == "idle"]
            if not free:
                return
            req = self.queue.popleft()
            P = req.prompt
            best, L = None, 0
            for c in self.ckpts:  # longest valid checkpoint that is a proper prefix of the prompt
                n = len(c.tokens)
                if L < n < len(P) and c.salt == req.cache_salt and tuple(P[:n]) == c.tokens and self._valid(c):
                    best, L = c, n
            if best is not None and self.slots[best.slot].phase == "idle":
                s = self.slots[best.slot]
            else:  # an empty slot first, then the least recently used
                s = min(free, key=lambda x: (len(x.tokens) > 0, x.last_used))
            b = s.idx
            if best is not None:
                self._restore(s, best)
            else:
                for i in self.gdn_layers:
                    self.state.conv[i][b].zero_()
                    self.state.rec[i][b].zero_()
                s.tokens, s.h_last = [], None
                self.state.pos_t[b] = 0
            for c in [c for c in self.ckpts if c.slot == b and not self._valid(c)]:
                self._drop(c)  # the slot's new history overwrites them
            req.slot, req.reused, req.t_admit = b, len(s.tokens), time.perf_counter()
            s.salt = req.cache_salt
            s.req, s.phase, s.todo, s.y = req, "prefill", list(P[len(s.tokens):]), None
            s.acc = {"mtp": [1.5, 2.5], "suffix": [4.0, 5.0]}  # priors 0.6 and 0.8
            s.sfx = None
            s.splits = []
            if self.boundary is not None and (self.free_bufs or self.ckpts):
                # message starts, without the last one (it opens the reply being generated: the prompt-end snapshot)
                bs = [i for i, t in enumerate(P) if t == self.boundary and i > 0][:-1]
                s.splits = sorted({p for p in bs[:1] + bs[-1:] if p >= len(s.tokens) + 256})
            self.params.set(b, req.temperature, req.top_k, req.top_p, req.min_p, req.seed)
            if self.mtp is not None:
                ids = list(req.eos_ids)[:MAX_STOP_IDS] if req.min_tokens <= 1 else []  # the GPU cut cannot count to min_tokens
                self.stop_buf[b] = torch.tensor(ids + [-1] * (MAX_STOP_IDS - len(ids)))
                toks = list(P)  # draft vocabulary: this prompt's tokens first, then the other active requests'
                for o in self.slots:
                    if o.req is not None and o is not s:
                        toks += o.req.prompt
                self.mtp.set_prompt_vocab(toks)
            m = self.metrics
            m.prompt_tokens.inc(len(P))
            m.cached_tokens.inc(req.reused)
            m.queue_seconds.observe(req.t_admit - req.t_submit)

    # ---------------------------------------------------------------- one engine iteration
    @torch.inference_mode()
    def step(self):
        self._admit()
        pre = next((s for s in self.slots if s.phase == "prefill"), None)
        if pre is not None:
            self._prefill_chunk(pre)
        dec = [s for s in self.slots if s.phase == "decode"]
        if dec:
            self._decode(dec)
        m = self.metrics
        m.queue_depth.set(len(self.queue))
        for ph in ("prefill", "decode"):
            m.slots_busy.set(sum(s.phase == ph for s in self.slots), phase=ph)

    def _prefill_chunk(self, s: Slot):
        t0 = time.perf_counter()
        b, start = s.idx, len(s.tokens)
        n = min(self.chunk, len(s.todo))
        if len(s.todo) <= self.chunk * 5 // 4:  # finish the prompt in one chunk rather than leave a short tail (a full weight pass)
            n = len(s.todo)
        for p in s.splits:
            if start < p < start + n:
                n = p - start
                break
        chunk = s.todo[:n]
        view = self.state.view(b, b + 1)
        view.pos = start
        ids = torch.tensor([chunk], device=self.dev)
        if self.mtp is not None:
            logits, H = prefill(self.model, ids, view, return_hidden=True)
        else:
            logits = prefill(self.model, ids, view)
        s.tokens += chunk
        s.todo = s.todo[len(chunk):]
        if self.mtp is not None:  # drafter KV rows (x_{i+1}, h_i) for positions start-1 .. end-2
            mv = self.mst.slot(b)
            if start > 0:
                mv.pos_t.fill_(start - 1)
                toks, hid = chunk, torch.cat([s.h_last.view(1, -1), H[:-1]])
            else:
                mv.pos_t.fill_(0)
                toks, hid = chunk[1:], H[:-1]
            if toks:
                self.mtp.prefill(torch.tensor(toks, device=self.dev), hid, mv)
            s.h_last = H[-1]
        done = not s.todo
        if done or len(s.tokens) in s.splits or len(s.tokens) // self.ckpt_interval > start // self.ckpt_interval:
            self._snapshot(s)
        if done:
            req = s.req
            tok = int(sample(logits, self.params.view(b, b + 1), view.pos_t))
            lp = self._logprobs(logits, [tok], req.logprobs) if req.logprobs is not None else None
            s.phase, s.y = "decode", tok
            self._emit(s, [tok], lp)
            if s.phase == "decode" and self.mtp is not None:
                mv = self.mst.slot(b)
                mv.pos_t.fill_(len(s.tokens) - 1)
                d = self.mtp.first_drafts(torch.tensor([tok], device=self.dev), s.h_last.view(1, -1), mv, self.k)
                self.tok[b, :self.k + 1] = torch.tensor([tok] + d)
                s.src, s.fresh = "mtp", self.k
                if self.suffix_min:
                    s.sfx = SuffixIndex(s.tokens + [tok])
                    self._suffix_drafts(s)
        torch.cuda.synchronize()
        self.metrics.step_seconds.observe(time.perf_counter() - t0, kind="prefill")

    def _decode(self, dec: list[Slot]):
        t0 = time.perf_counter()
        lo, hi = min(s.idx for s in dec), max(s.idx for s in dec) + 1
        act = [0] * self.n_slots
        for s in dec:
            act[s.idx] = 1
        self.state.active.copy_(torch.tensor(act, dtype=torch.int32))
        m = self.metrics
        if self.mtp is not None:
            kk = self._pick_k(hi - lo, dec)
            g = self.cycles[(lo, hi, any(s.req.temperature > 0 for s in dec), kk)]
            g.graph.replay()
            self._evt.record()
            self._evt.synchronize()  # releases the GIL while the GPU works
            ns, outs = g.n.tolist(), g.out_tok.tolist()
            for s in dec:
                b = s.idx - lo
                nb = ns[b]
                o = outs[b][:nb]
                s.tokens.append(s.y)
                s.tokens += o[:-1]
                s.y, s.h_last = o[-1], g.H[b, nb - 1]
                kf = min(kk, s.fresh)  # stale drafts say nothing about acceptance
                st = s.acc[s.src]
                st[0] = ACC_DECAY * st[0] + min(nb - 1, kf)
                st[1] = ACC_DECAY * st[1] + min(nb - 1, kf) + (1 if nb - 1 < kf else 0)
                s.src, s.fresh = "mtp", min(kk, self.k)
                m.drafted.inc(kk)
                m.accepted.inc(nb - 1)
                m.tokens_per_cycle.observe(nb)
                lp = self._logprobs(g.logits[b, :nb], o, s.req.logprobs) if s.req.logprobs is not None else None
                self._emit(s, o, lp)
                if s.sfx is not None and s.phase == "decode":
                    s.sfx.extend(o)
                    self._suffix_drafts(s)
        else:
            g = self.graphs[(lo, hi)]
            toks = torch.zeros(hi - lo, dtype=torch.long)
            for s in dec:
                toks[s.idx - lo] = s.y
            nxt = g.step(toks.to(self.dev))
            self._evt.record()
            self._evt.synchronize()
            out = nxt.tolist()
            for s in dec:
                b = s.idx - lo
                s.tokens.append(s.y)
                s.y = out[b]
                lp = self._logprobs(g.logits[b:b + 1], [out[b]], s.req.logprobs) if s.req.logprobs is not None else None
                self._emit(s, [out[b]], lp)
        self.state.active.zero_()
        m.step_seconds.observe(time.perf_counter() - t0, kind="decode", width=hi - lo)

    def _suffix_drafts(self, s: Slot):
        """Replace the slot's pending MTP drafts with the continuation of the longest earlier occurrence of its history's
        suffix, when that match is at least suffix_min tokens long (engine/spec/suffix.py). The drafter's state does not
        depend on the drafts, only on accepted tokens, so either source can feed any cycle."""
        n, cont = s.sfx.draft(self.kmax)
        if n >= self.suffix_min and cont:
            self.tok[s.idx, 1:1 + len(cont)] = torch.tensor(cont)
            s.src, s.fresh = "suffix", len(cont)

    def _logprobs(self, rows: torch.Tensor, toks: list[int], n_top: int):
        """rows [r, V] raw logits of the positions that produced toks -> [(logprob, [(id, logprob) x n_top])]."""
        lsm = torch.log_softmax(rows.float(), -1)
        chosen = lsm.gather(1, torch.tensor(toks, device=rows.device)[:, None])[:, 0].tolist()
        if n_top > 0:
            tv, ti = lsm.topk(n_top, -1)
            tv, ti = tv.tolist(), ti.tolist()
            return [(chosen[i], list(zip(ti[i], tv[i]))) for i in range(len(toks))]
        return [(c, []) for c in chosen]

    def _emit(self, s: Slot, toks: list[int], lps):
        req = s.req
        for i, t in enumerate(toks):
            if not req.output:
                req.t_first = time.perf_counter()
                self.metrics.ttft_seconds.observe(req.t_first - req.t_submit)
            req.output.append(t)
            lp = lps[i] if lps is not None else None
            if lp is not None:
                req.output_logprobs.append(lp)
            hook_stop = req.hook.feed(t, lp) if req.hook is not None else False
            if hook_stop or (t in req.eos_ids and len(req.output) >= req.min_tokens):
                return self._finish(s, "stop")
            if len(req.output) >= req.max_new_tokens or len(s.tokens) + self.margin > self.max_seq_len:
                return self._finish(s, "length")

    def _finish(self, s: Slot, reason: str):
        req = s.req
        if s.phase == "decode" or not s.todo:
            self._snapshot(s)  # end of generation: everything fed so far (the last output token was never fed)
        s.phase, s.req, s.todo = "idle", None, []
        self._close(req, reason)
        s.last_used = req.t_done

    def _close(self, req: Request, reason: str):
        """The end of a request, in a slot or still queued: its fields, the metrics, the hook."""
        req.done, req.finish_reason, req.t_done = True, reason, time.perf_counter()
        self.metrics.requests.inc(reason=reason)
        self.metrics.generated_tokens.inc(len(req.output))
        if req.hook is not None:
            req.hook.finish(req)
        if self.keep_finished:
            self.finished.append(req)

    def busy(self) -> bool:
        return bool(self.queue) or any(s.phase != "idle" for s in self.slots)

    def reset(self):
        """Forgets every slot history and checkpoint, and zeroes the metrics (the server's warm-up ends with it). The
        scheduler must be idle."""
        if self.busy():
            raise RuntimeError("reset() needs an idle scheduler")
        for c in list(self.ckpts):
            self._drop(c)
        self.slots = [Slot(i) for i in range(self.n_slots)]
        self.state.pos_t.zero_()
        self.finished.clear()
        self.metrics.reset()

    def run(self, requests: list[Request]) -> list[Request]:
        for r in requests:
            self.submit(r)
        while self.busy():
            self.step()
        return requests
