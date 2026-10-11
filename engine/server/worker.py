"""The engine thread of the server, and the bridge from it to the HTTP handlers (engine/server/api.py).

One engine thread owns the GPU. It builds the engine (engine/runtime/build.py), warms every code path up, and then runs
Scheduler.step() in a loop. The HTTP handlers give it requests through a queue (Worker.submit / abort). Each request
carries a Stream as its hook: the scheduler calls feed() per token and finish() at the end on the engine thread, and
the handler reads the events from an asyncio queue.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import queue
import sys
import threading
import time
import traceback

import torch

from engine.runtime.build import MODEL, build_engine
from engine.runtime.metrics import Metrics
from engine.runtime.scheduler import Request as EngineRequest
from engine.spec.suffix import MIN_MATCH

WARMUP_IDS = (1000, 150000)  # ordinary text tokens for the warm-up prompts: past the control tokens, before the rare tail


@dataclasses.dataclass
class ServerConfig:
    """What the engine thread builds and the HTTP layer enforces. The defaults are the server's flags (engine/server/api.py
    main reads its argparse defaults from here), so a test constructs the same object with a few fields changed."""
    model: str = MODEL
    slots: int = 3
    max_seq_len: int = 262144
    checkpoints: int = 32
    spec: str = "mtp"
    k: int = 7
    drafter_weights: str = "auto"
    suffix_drafts: int = MIN_MATCH
    decode_weights: str = "int"
    boundary_token: int | str | None = "auto"  # the <|im_start|> id, "auto" (from the checkpoint) or None (no message splits)
    selftest: bool = True
    warmup: bool = True
    max_queue: int = 8
    max_output_tokens: int = 0
    max_request_seconds: float = 1800.0
    api_key: str | None = None
    mem_cap_bytes: int = 0


class Worker(threading.Thread):
    def __init__(self, cfg: ServerConfig, log=print):
        super().__init__(name="engine", daemon=True)
        self.cfg, self.log = cfg, log
        self.inbox: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.metrics = Metrics()
        self.sched = None
        self.startup: dict = {}

    def submit(self, req: EngineRequest):
        self.inbox.put(("submit", req))

    def abort(self, req: EngineRequest):
        self.inbox.put(("abort", req))

    def queued(self) -> int:
        """Requests waiting to run: in the inbox, or in the scheduler's queue (read from the HTTP thread; a count that
        is a step stale is fine for the admission limit)."""
        return self.inbox.qsize() + len(self.sched.queue)

    def _build(self):
        c, log, t0 = self.cfg, self.log, time.perf_counter()
        sched, self.startup = build_engine(c.model, slots=c.slots, max_seq_len=c.max_seq_len, checkpoints=c.checkpoints, spec=c.spec, k=c.k,
                                           drafter_weights=c.drafter_weights, suffix_drafts=c.suffix_drafts, decode_weights=c.decode_weights,
                                           boundary=c.boundary_token, selftest=c.selftest, metrics=self.metrics, log=log)
        t3 = time.perf_counter()
        if c.warmup:
            self._warmup(sched, sched.model.cfg.vocab_size)
        t4 = time.perf_counter()
        self.sched = sched
        self.startup.update(warmup_s=round(t4 - t3, 1), total_s=round(t4 - t0, 1))
        mem = torch.cuda.memory_allocated() / 1e9
        log(f"[engine] ready: {self.startup}; {mem:.1f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.1f} GB reserved; "
            f"{c.slots} slots x {c.max_seq_len} tokens, spec={c.spec}" + (f" k={c.k}" if c.spec == "mtp" else "")
            + (f", suffix drafts >= {c.suffix_drafts}" if c.spec == "mtp" and c.suffix_drafts else ""))

    def _warmup(self, sched, vocab):
        """Runs every code path once (prefill chunks, the decode / spec graphs of each slot, greedy and sampled,
        logprobs) so first-use JIT compilation and autotuning happen before serving; then forgets everything."""
        g = torch.Generator().manual_seed(0)
        lo, hi = WARMUP_IDS[0], min(vocab, WARMUP_IDS[1])
        rnd = lambda n: torch.randint(lo, hi, (n,), generator=g).tolist()  # noqa: E731
        reqs = [EngineRequest(rnd(sched.chunk + 300), max_new_tokens=6, logprobs=5),
                EngineRequest(rnd(40), max_new_tokens=6, temperature=0.7, top_p=0.95, top_k=20, seed=1),
                EngineRequest(rnd(30), max_new_tokens=6)]
        sched.run(reqs[:1])
        sched.run(reqs[1:])
        sched.run([EngineRequest(rnd(20 + i), max_new_tokens=8, temperature=0.5 * (i % 2)) for i in range(sched.n_slots)])
        sched.reset()
        torch.cuda.synchronize()

    def run(self):
        try:
            with torch.inference_mode():
                self._build()
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        with torch.inference_mode():
            self._loop()

    def _loop(self):
        sched, m, cap = self.sched, self.metrics, self.cfg.mem_cap_bytes
        while True:
            items = []
            if not sched.busy():
                try:
                    items.append(self.inbox.get(timeout=1.0))
                except queue.Empty:
                    continue
            while True:
                try:
                    items.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                for kind, req in items:
                    if kind == "submit":
                        try:
                            sched.submit(req)
                        except ValueError as e:
                            req.hook.error(str(e), 400)
                    elif req.rid >= 0 and not req.done:
                        sched.abort(req.rid)
                if sched.busy():
                    sched.step()
                m.memory.set(torch.cuda.memory_allocated(), kind="allocated")
                m.memory.set(torch.cuda.memory_reserved(), kind="reserved")
                m.memory.set(cap, kind="cap")
            except BaseException:  # noqa: BLE001  a CUDA error poisons the context: fail everything and exit
                traceback.print_exc()
                for s in sched.slots:
                    if s.req is not None and s.req.hook is not None:
                        s.req.hook.error("engine failure", 500)
                for r in sched.queue:
                    if r.hook is not None:
                        r.hook.error("engine failure", 500)
                sys.stdout.flush()
                os._exit(1)  # systemd restarts the service


class Stream:
    """One request's bridge from the engine thread (feed / finish / error) to its HTTP handler (an asyncio queue): the
    scheduler's Hook (engine/runtime/scheduler.py).

    feed and finish run on the engine thread, which exits on any exception (a CUDA error poisons the context). So a
    failure to format the output (a parser bug, a tool schema the parser does not expect) must stay here: it fails this
    request with a 500, and feed returns True so that the scheduler ends the request."""

    def __init__(self, loop, tok, parser):
        self.loop, self.tok, self.parser = loop, tok, parser
        self.q: asyncio.Queue = asyncio.Queue()
        self.ids: list[int] = []
        self.lps: list = []
        self.failed = False

    def _put(self, item):
        self.loop.call_soon_threadsafe(self.q.put_nowait, item)

    def _tokstr(self, t):
        s = self.tok.decode([t])
        return s, list(s.encode("utf-8", errors="replace"))

    def feed(self, t, lp):  # engine thread
        if self.failed:
            return True
        try:
            self.ids.append(t)
            if lp is not None:
                s, b = self._tokstr(t)
                self.lps.append({"token": s, "logprob": lp[0], "bytes": b,
                                 "top_logprobs": [dict(zip(("token", "bytes"), self._tokstr(i)), logprob=v) for i, v in lp[1]]})
            ev = self.parser.feed(t)
            if ev:
                self._put(("delta", ev, self.ids, self.lps))
                self.ids, self.lps = [], []
            return self.parser.stopped
        except Exception:  # noqa: BLE001  see the class docstring
            self._fail()
            return True

    def finish(self, req):  # engine thread
        if self.failed:
            return
        try:
            ev = self.parser.finish()
        except Exception:  # noqa: BLE001  see the class docstring
            return self._fail()
        if ev or self.ids:
            self._put(("delta", ev, self.ids, self.lps))
            self.ids, self.lps = [], []
        self._put(("done", req))

    def _fail(self):
        traceback.print_exc()
        self.failed = True
        self.error("internal error while formatting the output", 500)

    def error(self, msg, code=500):
        self._put(("error", msg, code))
