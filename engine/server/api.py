"""OpenAI-compatible HTTP server (PLAN.md 4.6, Phase 5).

    uv run python -m engine.server [--port 8000] [--slots 3] [--max-seq-len 262144] [--spec mtp|none]

Endpoints: /v1/chat/completions and /v1/completions (SSE streaming, usage + timings, logprobs, stop strings,
seeds, tools, thinking), /v1/models, /health, /metrics (Prometheus), /v1/status (slots and checkpoints).

One engine thread owns the GPU: it loads the model, captures the CUDA graphs and then runs Scheduler.step()
in a loop. HTTP handlers (asyncio, uvicorn) render the chat template and tokenize off the event loop, hand the
request to the engine thread through a queue, and receive output events back through an asyncio queue: the
engine thread detokenizes and parses each token as it is emitted (engine/server/chat.py), so stop strings end
a request in the same step that produced them. More than `--slots` concurrent requests wait in a FIFO queue.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from engine.runtime.metrics import Metrics
from engine.runtime.scheduler import Request as EngineRequest
from engine.server.chat import ChatFormat, OutputParser, TextParser

MAX_TOP_LOGPROBS = 20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ engine thread
class Worker(threading.Thread):
    def __init__(self, args):
        super().__init__(name="engine", daemon=True)
        self.args = args
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

    def _build(self):
        a, t0 = self.args, time.perf_counter()
        from engine.kernels import ops
        from engine.model.fast import attach_requant, load_fast_model, requant_path, set_linear_kernel, to_fast
        # speculation verifies many rows per weight pass: tensor-core skinny GEMM; plain decode (1-3 rows): GEMV
        set_linear_kernel("skinny" if a.spec == "mtp" else "gemv")
        from engine.runtime.scheduler import Scheduler
        from engine.weights.loader import resolve
        ops()  # build / load the CUDA extension
        t1 = time.perf_counter()
        path = resolve(a.model)
        model = to_fast(load_fast_model(path), kv_fp8=a.kv == "fp8", kv_fp4=a.kv == "fp4")
        dw = a.decode_weights
        if dw == "auto":  # the quality-gated NVFP4 attention linears when they have been made
            dw = "awq-attn" if os.path.exists(requant_path(path, "awq-attn")) else "checkpoint"
        if dw != "checkpoint":
            rq = requant_path(path, dw)
            if os.path.exists(rq):
                log(f"[engine] decode streams NVFP4 re-quantized projections ({dw}): {attach_requant(model, rq)} linears ({rq})")
            else:
                log(f"[engine] no re-quantized weights at {rq} (tools/awq_nvfp4.py / requant_nvfp4.py): decoding the checkpoint's FP8 projections")
        mtp = None
        if a.spec == "mtp":
            from engine.spec.mtp import Mtp
            dw = a.drafter_weights
            if dw == "auto":
                dw = os.path.expanduser("~/.cache/colinfer/drafter/mtp_ft.safetensors")
                dw = dw if os.path.exists(dw) else None
            elif dw == "none":
                dw = None
            mtp = Mtp(model, path, fp8=True, draft_vocab=a.draft_vocab or None, weights=dw)
            log(f"[engine] MTP drafter: {dw or 'checkpoint weights'}")
        t2 = time.perf_counter()
        sched = Scheduler(model, n_slots=a.slots, max_seq_len=a.max_seq_len, n_checkpoints=a.checkpoints, mtp=mtp, k=a.k,
                          selftest=not a.no_selftest, metrics=self.metrics, keep_finished=False, boundary_token=a.boundary_token)
        t3 = time.perf_counter()
        if not a.no_warmup:
            self._warmup(sched, model.cfg.vocab_size)
        t4 = time.perf_counter()
        self.sched = sched
        self.startup = dict(kernels_s=round(t1 - t0, 1), weights_s=round(t2 - t1, 1), graphs_selftest_s=round(t3 - t2, 1),
                            warmup_s=round(t4 - t3, 1), total_s=round(t4 - t0, 1))
        mem = torch.cuda.memory_allocated() / 1e9
        log(f"[engine] ready: {self.startup}; {mem:.1f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.1f} GB reserved; "
            f"{a.slots} slots x {a.max_seq_len} tokens, spec={a.spec}" + (f" k={a.k}" if a.spec == "mtp" else ""))

    def _warmup(self, sched, vocab):
        """Runs every code path once (prefill chunks, the decode / spec graphs at each width, greedy and sampled,
        logprobs) so first-use JIT compilation and autotuning happen before serving; then forgets everything."""
        g = torch.Generator().manual_seed(0)
        rnd = lambda n: torch.randint(1000, min(vocab, 150000), (n,), generator=g).tolist()  # noqa: E731
        reqs = [EngineRequest(rnd(sched.chunk + 300), max_new_tokens=6, logprobs=5),
                EngineRequest(rnd(40), max_new_tokens=6, temperature=0.7, top_p=0.95, top_k=20, seed=1),
                EngineRequest(rnd(30), max_new_tokens=6)]
        sched.run(reqs[:1])
        sched.run(reqs[1:])
        sched.run([EngineRequest(rnd(20 + i), max_new_tokens=8, temperature=0.5 * (i % 2)) for i in range(sched.n_slots)])
        for c in list(sched.ckpts):
            sched._drop(c)
        for s in sched.slots:
            s.tokens, s.h_last, s.last_used = [], None, 0.0
        sched.state.pos_t.zero_()
        sched.metrics.__init__()
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
        sched, m = self.sched, self.metrics
        cap = getattr(self.args, "mem_cap_bytes", 0)
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


# ------------------------------------------------------------------------------------------------ request bridge
class Stream:
    """One request's bridge from the engine thread (feed / finish / error) to its HTTP handler (an asyncio queue)."""

    def __init__(self, loop, tok, parser, logprobs: int | None):
        self.loop, self.tok, self.parser, self.logprobs = loop, tok, parser, logprobs
        self.q: asyncio.Queue = asyncio.Queue()
        self.ids: list[int] = []
        self.lps: list = []

    def _put(self, item):
        self.loop.call_soon_threadsafe(self.q.put_nowait, item)

    def _tokstr(self, t):
        s = self.tok.decode([t])
        return s, list(s.encode("utf-8", errors="replace"))

    def feed(self, t, lp):  # engine thread
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

    def finish(self, req):  # engine thread
        ev = self.parser.finish()
        if ev or self.ids:
            self._put(("delta", ev, self.ids, self.lps))
            self.ids, self.lps = [], []
        self._put(("done", req))

    def error(self, msg, code=500):
        self._put(("error", msg, code))


def _usage(req: EngineRequest) -> dict:
    pt, ct = len(req.prompt), len(req.output)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "prompt_tokens_details": {"cached_tokens": req.reused}}


def _timings(req: EngineRequest) -> dict:
    pre = len(req.prompt) - req.reused
    prefill_s = max(req.t_first - req.t_admit, 1e-9)
    decode_s = max(req.t_done - req.t_first, 1e-9)
    return {"queue_s": round(req.t_admit - req.t_submit, 4), "ttft_s": round(req.t_first - req.t_submit, 4), "prefill_tokens": pre,
            "prefill_s": round(prefill_s, 4), "prefill_tok_s": round(pre / prefill_s, 1), "decode_s": round(decode_s, 4),
            "decode_tok_s": round((len(req.output) - 1) / decode_s, 2) if len(req.output) > 1 else None}


def _error(msg, code=400, kind="invalid_request_error"):
    return JSONResponse({"error": {"message": msg, "type": kind, "code": code}}, status_code=code)


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class BadRequest(Exception):
    pass


# ------------------------------------------------------------------------------------------------ app
def build_app(worker: Worker, tokenizer, served_name: str, gen_defaults: dict, hf_config: dict, default_thinking):
    app = FastAPI(title="colin-inference-engine")
    fmt = ChatFormat(tokenizer)
    a = worker.args
    max_len, margin = a.max_seq_len, worker.sched.margin
    created = int(time.time())

    def body_of(raw: dict) -> dict:
        b = dict(raw)
        extra = b.pop("extra_body", None)
        if isinstance(extra, dict):
            b.update({k: v for k, v in extra.items() if k not in b})
        if b.get("n", 1) not in (1, None):
            raise BadRequest("n > 1 is not supported")
        return b

    def make_request(b: dict, prompt: list[int], hook) -> EngineRequest:
        if len(prompt) + margin > max_len:
            raise BadRequest(f"prompt has {len(prompt)} tokens; this server's maximum context is {max_len} tokens")
        mt = b.get("max_completion_tokens") or b.get("max_tokens")
        room = max_len - margin - len(prompt) + 1
        mt = room if mt is None else min(int(mt), room)
        if mt < 1:
            raise BadRequest("max_tokens must be at least 1")
        temp = b.get("temperature")
        temp = gen_defaults.get("temperature", 1.0) if temp is None else float(temp)
        top_k = b.get("top_k")
        top_k = gen_defaults.get("top_k", 0) if top_k is None else int(top_k)
        top_p = b.get("top_p")
        top_p = gen_defaults.get("top_p", 1.0) if top_p is None else float(top_p)
        seed = b.get("seed")
        seed = random.getrandbits(62) if seed is None else int(seed) & ((1 << 62) - 1)
        stop_ids = tuple(int(t) for t in (b.get("stop_token_ids") or []))
        eos = stop_ids if b.get("ignore_eos") else tuple(sorted(set(fmt.eos_ids) | set(stop_ids)))
        if not (0.0 <= temp <= 100.0) or not (0.0 < top_p <= 1.0):
            raise BadRequest("temperature must be in [0, 100] and top_p in (0, 1]")
        salt = b.get("cache_salt")
        if b.get("cache_prompt") is False:  # llama.cpp's switch: no prefix reuse for this request
            salt = uuid.uuid4().hex
        return EngineRequest(prompt, max_new_tokens=mt, temperature=temp, top_k=max(top_k, 0), top_p=top_p,
                             min_p=float(b.get("min_p") or 0.0), seed=seed, eos_ids=eos, min_tokens=int(b.get("min_tokens") or 0),
                             logprobs=None, hook=hook, cache_salt=None if salt is None else str(salt))

    def stops_of(b: dict) -> list[str]:
        st = b.get("stop")
        return [st] if isinstance(st, str) else list(st or [])

    async def wait_done(request, st: Stream, req: EngineRequest):
        """Collect events until done (non-streaming); aborts the request if the client goes away."""
        events = []
        while True:
            try:
                item = await asyncio.wait_for(st.q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    worker.abort(req)
                    return None
                continue
            events.append(item)
            if item[0] in ("done", "error"):
                return events

    def log_done(req: EngineRequest, kind: str):
        t = _timings(req)
        log(f"[{kind} {req.rid}] slot {req.slot} prompt {len(req.prompt)} (cached {req.reused}) -> {len(req.output)} tok, "
            f"{req.finish_reason}; queue {t['queue_s']:.2f}s ttft {t['ttft_s']:.2f}s decode {t['decode_tok_s'] or 0:.1f} tok/s")

    # -------------------------------------------------------------------------------------- chat completions
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            b = body_of(await request.json())
            msgs = b.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise BadRequest("messages must be a non-empty list")
            tool_choice = b.get("tool_choice")
            tools = b.get("tools") if tool_choice != "none" else None
            kw = dict(b.get("chat_template_kwargs") or {})
            if "enable_thinking" not in kw and b.get("enable_thinking") is not None:
                kw["enable_thinking"] = bool(b["enable_thinking"])
            effort = b.get("reasoning_effort")
            if effort and "reasoning_effort" not in kw:
                if effort == "none":
                    kw.setdefault("enable_thinking", False)
                else:
                    kw["reasoning_effort"] = {"minimal": "low", "low": "low", "medium": "medium"}.get(effort, "xhigh")
            if "enable_thinking" not in kw and default_thinking is not None:
                kw["enable_thinking"] = default_thinking
            try:
                prompt = await asyncio.to_thread(fmt.render, msgs, tools, **kw)
            except Exception as e:  # template errors (bad roles, no user message, ...)
                raise BadRequest(f"chat template: {e}")
            n_top = None
            if b.get("logprobs"):
                n_top = min(int(b.get("top_logprobs") or 0), MAX_TOP_LOGPROBS)
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser, n_top)
            req = make_request(b, prompt, st)
            req.logprobs = n_top
        except BadRequest as e:
            return _error(str(e))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return _error(f"bad request: {e}")
        rid = "chatcmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
        stream = bool(b.get("stream"))
        include_usage = bool((b.get("stream_options") or {}).get("include_usage"))
        want_ids = bool(b.get("return_token_ids"))
        worker.submit(req)

        def chunk(delta, finish=None, extra=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            if extra:
                c["choices"][0].update(extra)
            return c

        if stream:
            async def gen():
                done = False
                try:
                    n_calls, first, role_sent, carry_ids, carry_lps = 0, True, False, [], []
                    while True:
                        item = await st.q.get()
                        if not role_sent and item[0] != "error":  # the role chunk goes out with the first output, as vLLM does
                            role_sent = True
                            yield _sse(chunk({"role": "assistant", "content": ""}))
                        if item[0] == "error":
                            done = True
                            yield _sse({"error": {"message": item[1], "code": item[2]}})
                            break
                        if item[0] == "done":
                            done = True
                            fr = item[1].finish_reason
                            fr = "tool_calls" if parser.n_tool_calls and fr == "stop" else fr
                            extra = {"token_ids": carry_ids} if want_ids and carry_ids else None
                            yield _sse(chunk({}, fr if fr != "abort" else "stop", extra))
                            if include_usage:
                                yield _sse({"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                                            "choices": [], "usage": _usage(item[1]), "timings": _timings(item[1])})
                            yield "data: [DONE]\n\n"
                            log_done(item[1], "chat")
                            break
                        _, ev, ids, lps = item
                        carry_ids += ids
                        carry_lps += lps
                        delta = {}
                        for kind, x in ev:
                            if kind == "tool_call":
                                delta.setdefault("tool_calls", []).append(
                                    {"index": n_calls, "id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": x["arguments"]}})
                                n_calls += 1
                            elif kind == "reasoning":
                                delta["reasoning_content"] = delta.get("reasoning_content", "") + x
                                delta["reasoning"] = delta["reasoning_content"]
                            else:
                                delta["content"] = delta.get("content", "") + x
                        if not delta:
                            continue
                        extra = {}
                        if want_ids:
                            extra["token_ids"] = carry_ids
                        if n_top is not None:
                            extra["logprobs"] = {"content": carry_lps}
                        c = chunk(delta, None, extra)
                        if first:
                            first = False
                            c["request_metrics"] = {"time_to_first_token_s": req.t_first - req.t_submit, "queue_time_s": req.t_admit - req.t_submit,
                                                    "prompt_time_s": req.t_first - req.t_admit}
                        carry_ids, carry_lps = [], []
                        yield _sse(c)
                finally:
                    if not done:
                        worker.abort(req)  # client went away
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        events = await wait_done(request, st, req)
        if events is None:
            return _error("client disconnected", 499)
        if events[-1][0] == "error":
            return _error(events[-1][1], events[-1][2], "server_error" if events[-1][2] >= 500 else "invalid_request_error")
        reasoning, content, calls, lps = "", "", [], []
        for item in events[:-1]:
            for kind, x in item[1]:
                if kind == "tool_call":
                    calls.append({"id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": x["arguments"]}})
                elif kind == "reasoning":
                    reasoning += x
                else:
                    content += x
            lps += item[3]
        r = events[-1][1]
        fr = "tool_calls" if calls and r.finish_reason == "stop" else r.finish_reason
        msg = {"role": "assistant", "content": content if (content or not calls) else None}
        if reasoning:
            msg["reasoning_content"] = msg["reasoning"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        choice = {"index": 0, "message": msg, "logprobs": {"content": lps} if n_top is not None else None, "finish_reason": fr}
        if want_ids:
            choice["token_ids"] = r.output
        log_done(r, "chat")
        return JSONResponse({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [choice],
                             "usage": _usage(r), "timings": _timings(r)})

    # -------------------------------------------------------------------------------------- completions
    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            b = body_of(await request.json())
            p = b.get("prompt")
            if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
                p = p[0]
            if isinstance(p, str):
                prompt = await asyncio.to_thread(tokenizer.encode, p, add_special_tokens=False)
            elif isinstance(p, list) and p and all(isinstance(t, int) for t in p):
                prompt = list(p)
            else:
                raise BadRequest("prompt must be a string or a list of token ids (one prompt per request)")
            if b.get("echo"):
                raise BadRequest("echo is not supported")
            n_top = None if b.get("logprobs") is None else min(int(b["logprobs"]), MAX_TOP_LOGPROBS)
            parser = TextParser(tokenizer, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser, n_top)
            req = make_request(b, prompt, st)
            req.logprobs = n_top
        except BadRequest as e:
            return _error(str(e))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return _error(f"bad request: {e}")
        rid = "cmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
        include_usage = bool((b.get("stream_options") or {}).get("include_usage"))
        worker.submit(req)

        def lp_block(lps):
            if n_top is None:
                return None
            return {"tokens": [e["token"] for e in lps], "token_logprobs": [e["logprob"] for e in lps],
                    "top_logprobs": [{t["token"]: t["logprob"] for t in e["top_logprobs"]} for e in lps], "text_offset": []}

        def chunk(text, finish=None, lps=None):
            return {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "text": text, "logprobs": lp_block(lps or []) if lps else None, "finish_reason": finish}]}

        if b.get("stream"):
            async def gen():
                done = False
                try:
                    carry = []
                    while True:
                        item = await st.q.get()
                        if item[0] == "error":
                            done = True
                            yield _sse({"error": {"message": item[1], "code": item[2]}})
                            break
                        if item[0] == "done":
                            done = True
                            fr = item[1].finish_reason
                            yield _sse(chunk("", fr if fr != "abort" else "stop", carry))
                            if include_usage:
                                yield _sse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                                            "choices": [], "usage": _usage(item[1]), "timings": _timings(item[1])})
                            yield "data: [DONE]\n\n"
                            log_done(item[1], "cmpl")
                            break
                        _, ev, ids, lps = item
                        carry += lps
                        text = "".join(x for _, x in ev)
                        if text:
                            yield _sse(chunk(text, None, carry))
                            carry = []
                finally:
                    if not done:
                        worker.abort(req)
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        events = await wait_done(request, st, req)
        if events is None:
            return _error("client disconnected", 499)
        if events[-1][0] == "error":
            return _error(events[-1][1], events[-1][2])
        text = "".join(x for item in events[:-1] for _, x in item[1])
        lps = [e for item in events[:-1] for e in item[3]]
        r = events[-1][1]
        log_done(r, "cmpl")
        return JSONResponse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "text": text, "logprobs": lp_block(lps), "finish_reason": r.finish_reason}],
                             "usage": _usage(r), "timings": _timings(r)})

    # -------------------------------------------------------------------------------------- the rest
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "created": created, "owned_by": "colin-inference-engine",
                                            "root": a.model, "max_model_len": max_len, "config": hf_config}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(worker.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/status")
    async def status():
        sc = worker.sched
        return {"slots": [{"slot": s.idx, "phase": s.phase, "tokens": len(s.tokens), "request": s.req.rid if s.req else None,
                           "checkpoints": sorted(len(c.tokens) for c in sc.ckpts if c.slot == s.idx)} for s in sc.slots],
                "queue": len(sc.queue), "checkpoints": len(sc.ckpts), "startup": worker.startup,
                "memory_gb": {"allocated": round(torch.cuda.memory_allocated() / 1e9, 2), "reserved": round(torch.cuda.memory_reserved() / 1e9, 2)}}

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="colin-inference-engine OpenAI-compatible server")
    ap.add_argument("--model", default="nvidia/Qwen3.8-27B-NVFP4", help="HF repo id (in the local cache) or checkpoint directory")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=262144, help="tokens per slot (prompt + output)")
    ap.add_argument("--kv", choices=("fp8", "fp4"), default="fp8",
                    help="KV cache format: fp8 (32 KB / token) or fp4 (18 KB / token: e2m1 + block scales; perplexity +0.2-0.3%%, "
                         "faster decode at long context)")
    ap.add_argument("--spec", choices=("mtp", "none"), default="mtp")
    ap.add_argument("--drafter-weights", default="auto",
                    help="MTP head weights: auto = ~/.cache/colinfer/drafter/mtp_ft.safetensors (tools/train_drafter.py) when it "
                         "exists, none = the checkpoint's, or a path. Drafts only affect speed, never outputs")
    ap.add_argument("--k", type=int, default=7, help="longest MTP draft; each cycle picks 3 or k from the measured acceptance")
    ap.add_argument("--draft-vocab", type=int, default=65536, help="MTP drafts among this many frequent tokens (+ prompt tokens); 0 = full")
    ap.add_argument("--checkpoints", type=int, default=32, help="prefix checkpoint ring size (154 MB each)")
    ap.add_argument("--decode-weights", choices=("auto", "awq-attn", "requant", "checkpoint"), default="auto",
                    help="checkpoint: decode the attention / GDN projections from their FP8 originals; awq-attn: the attention "
                         "projections from AWQ NVFP4 (tools/awq_nvfp4.py --groups self_attn; ~3%% faster, perplexity within 0.5%%); "
                         "requant: attention and GDN from NVFP4 (tools/gptq_nvfp4.py --damp 0.3, else tools/requant_nvfp4.py): ~18%% "
                         "faster, but Python-code perplexity +1.5%% (docs/phase6_progress.md); auto: awq-attn if its file exists")
    ap.add_argument("--no-prefix-caching", action="store_true", help="never reuse a prompt prefix (benchmarking raw prefill; = --checkpoints 0)")
    ap.add_argument("--mem-cap-gb", type=float, default=80.0, help="hard cap on this process's GPU memory (torch allocator)")
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto", help="default enable_thinking (auto: the template's default, on)")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    if a.no_prefix_caching:
        a.checkpoints = 0

    t0 = time.perf_counter()
    total = torch.cuda.get_device_properties(0).total_memory
    a.mem_cap_bytes = int(min(a.mem_cap_gb * 1e9, total))
    torch.cuda.set_per_process_memory_fraction(a.mem_cap_bytes / total)
    log(f"[server] GPU memory cap {a.mem_cap_bytes / 1e9:.0f} GB of {total / 1e9:.0f} GB")

    from transformers import AutoTokenizer

    from engine.weights.loader import resolve
    path = resolve(a.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    a.boundary_token = tokenizer.convert_tokens_to_ids("<|im_start|>")
    cfg = json.load(open(os.path.join(path, "config.json")))
    hf_config = cfg.get("text_config", cfg)
    gen = json.load(open(os.path.join(path, "generation_config.json"))) if os.path.exists(os.path.join(path, "generation_config.json")) else {}
    gen_defaults = {k: gen[k] for k in ("temperature", "top_k", "top_p") if k in gen}

    worker = Worker(a)
    worker.start()
    worker.ready.wait()
    if worker.error is not None:
        raise SystemExit(f"engine failed to start: {worker.error!r}")
    thinking = {"auto": None, "on": True, "off": False}[a.thinking]
    app = build_app(worker, tokenizer, a.served_model_name or a.model, gen_defaults, hf_config, thinking)
    log(f"[server] startup {time.perf_counter() - t0:.1f} s; listening on http://{a.host}:{a.port}/v1")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
