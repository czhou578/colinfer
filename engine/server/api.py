"""OpenAI- and Anthropic-compatible HTTP server.

    uv run python -m engine.server [--port 8000] [--slots 3] [--max-seq-len 262144] [--spec mtp|none]

Endpoints: /v1/chat/completions and /v1/completions (SSE streaming, usage + timings, logprobs, stop strings, seeds,
tools, thinking), /v1/messages and /v1/messages/count_tokens (the Anthropic Messages API, engine/server/anthropic.py),
/v1/models, /health, /metrics (Prometheus), /v1/status (slots and checkpoints). With --api-key, the /v1/ endpoints
require the key (Authorization: Bearer, or x-api-key).

One engine thread owns the GPU. It loads the model, captures the CUDA graphs, and then runs Scheduler.step() in a loop.
The HTTP handlers (asyncio, uvicorn) render the chat template and tokenize outside the event loop. They give the
request to the engine thread through a queue, and get the output events back through an asyncio queue.

The engine thread detokenizes and parses each token as it emits it (engine/server/chat.py). Thus a stop string ends a
request in the same step that produced it. When more than `--slots` requests are active, the others wait in a FIFO
queue.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
import uuid

import jinja2
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from engine.runtime.metrics import Metrics
from engine.runtime.scheduler import Request as EngineRequest
from engine.server import anthropic as anth
from engine.server.chat import ChatFormat, OutputParser, TextParser, check_messages
from engine.spec.suffix import MIN_MATCH

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

    def queued(self) -> int:
        """Requests waiting to run: in the inbox, or in the scheduler's queue."""
        return len(self.inbox.queue) + len(self.sched.queue)

    def _build(self):
        a, t0 = self.args, time.perf_counter()
        from engine.runtime.build import build_engine
        sched, self.startup = build_engine(a.model, slots=a.slots, max_seq_len=a.max_seq_len, checkpoints=a.checkpoints, spec=a.spec, k=a.k,
                                           drafter_weights=a.drafter_weights, suffix_drafts=a.suffix_drafts, decode_weights=a.decode_weights,
                                           boundary=a.boundary_token, selftest=not a.no_selftest, metrics=self.metrics,
                                           log=log)
        t3 = time.perf_counter()
        if not a.no_warmup:
            self._warmup(sched, sched.model.cfg.vocab_size)
        t4 = time.perf_counter()
        self.sched = sched
        self.startup.update(warmup_s=round(t4 - t3, 1), total_s=round(t4 - t0, 1))
        mem = torch.cuda.memory_allocated() / 1e9
        log(f"[engine] ready: {self.startup}; {mem:.1f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.1f} GB reserved; "
            f"{a.slots} slots x {a.max_seq_len} tokens, spec={a.spec}" + (f" k={a.k}" if a.spec == "mtp" else "")
            + (f", suffix drafts >= {a.suffix_drafts}" if a.spec == "mtp" and a.suffix_drafts else ""))

    def _warmup(self, sched, vocab):
        """Runs every code path once (prefill chunks, the decode / spec graphs of each slot, greedy and sampled,
        logprobs) so first-use JIT compilation and autotuning happen before serving; then forgets everything."""
        g = torch.Generator().manual_seed(0)
        rnd = lambda n: torch.randint(1000, min(vocab, 150000), (n,), generator=g).tolist()  # noqa: E731
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
    """One request's bridge from the engine thread (feed / finish / error) to its HTTP handler (an asyncio queue).

    feed and finish run on the engine thread, which exits on any exception (a CUDA error poisons the context). So a
    failure to format the output (a parser bug, a tool schema the parser does not expect) must stay here: it fails this
    request with a 500, and feed returns True so that the scheduler ends the request."""

    def __init__(self, loop, tok, parser, logprobs: int | None):
        self.loop, self.tok, self.parser, self.logprobs = loop, tok, parser, logprobs
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


def _error_body(msg: str, code: int) -> dict:
    """An error in the format of the OpenAI API (a response body, or a stream event); the code sets the type."""
    return {"error": {"message": msg, "type": "server_error" if code >= 500 else "invalid_request_error", "code": code}}


def _error(msg, code=400):
    return JSONResponse(_error_body(msg, code), status_code=code)


_ANTH_ERROR = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error", 413: "request_too_large",
               499: "invalid_request_error", 529: "overloaded_error"}


def _anth_error_body(msg: str, code: int) -> dict:
    """An error in the format of the Anthropic API (a response body, or a stream event)."""
    return {"type": "error", "error": {"type": _ANTH_ERROR.get(code, "api_error"), "message": msg}}


def _anth_error(msg, code=400):
    return JSONResponse(_anth_error_body(msg, code), status_code=code)


def _finish_reason(r: EngineRequest) -> str:
    # an abort (the client went away) reads as a stop, a timeout as a cut-off
    return {"abort": "stop", "timeout": "length"}.get(r.finish_reason, r.finish_reason)


class ApiKey:
    """ASGI middleware: the /v1/ endpoints require the key, as `Authorization: Bearer <key>` or `x-api-key: <key>`."""

    def __init__(self, app, key: str):
        self.app, self.key = app, key.encode()

    def _ok(self, headers) -> bool:
        h = {k.decode("latin-1").lower(): v for k, v in headers}
        auth = h.get("authorization", b"")
        bearer = auth[7:] if auth[:7].lower() == b"bearer " else b""
        return hmac.compare_digest(h.get("x-api-key", b""), self.key) or hmac.compare_digest(bearer, self.key)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/v1/") and not self._ok(scope["headers"]):
            body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid or missing API key", "code": 401}}
            return await JSONResponse(body, status_code=401)(scope, receive, send)
        await self.app(scope, receive, send)


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class BadRequest(Exception):
    """The client's error (a 400). The endpoints turn only this into an error reply; any other exception is a bug of the
    server, which the app's handler logs and answers with a 500."""


class Busy(Exception):
    """Too many requests wait already (a 503; a 529 on the Anthropic API, which Claude Code retries)."""


def _num(v, kind, name: str):
    """A request field as an int / float, or a BadRequest."""
    try:
        return kind(v)
    except (TypeError, ValueError, OverflowError):  # OverflowError: int(Infinity), which Python's JSON parser accepts
        raise BadRequest(f"{name} must be a number, not {v!r}") from None


# reasoning_effort names of other APIs and templates -> the levels of the Qwen3.8 template (None: thinking off)
_EFFORTS = {"none": None, "minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


def _template_kwargs(b: dict, default_thinking: bool | None) -> dict:
    """The chat template kwargs of a chat request: chat_template_kwargs, plus the top-level enable_thinking and
    reasoning_effort. Clients written for other templates set thinking with `thinking` (DeepSeek) and effort with names
    that this template rejects ("none", "high", "max"). Thus `thinking` stands for enable_thinking, and the names in
    _EFFORTS map to the levels of this template. Other values in chat_template_kwargs go to the template unchanged."""
    kw = b.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    kw = dict(kw)
    if "enable_thinking" not in kw and isinstance(kw.get("thinking"), bool):
        kw["enable_thinking"] = kw["thinking"]
    if "enable_thinking" not in kw and b.get("enable_thinking") is not None:
        kw["enable_thinking"] = bool(b["enable_thinking"])
    if not isinstance(b.get("reasoning_effort"), (str, type(None))):
        raise BadRequest("reasoning_effort must be a string")
    top_level = "reasoning_effort" not in kw
    effort = b.get("reasoning_effort") if top_level else kw.pop("reasoning_effort")
    if isinstance(effort, str) and effort in _EFFORTS:
        if _EFFORTS[effort] is None:
            kw.setdefault("enable_thinking", False)
        else:
            kw["reasoning_effort"] = _EFFORTS[effort]
    elif effort:  # another top-level name means the template default; the template checks its own kwargs
        kw["reasoning_effort"] = "xhigh" if top_level else effort
    if "enable_thinking" not in kw and default_thinking is not None:
        kw["enable_thinking"] = default_thinking
    return kw


class Failed(Exception):
    """A request that the engine failed, or whose client went away: the message and the HTTP status."""

    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.msg, self.code = msg, code


# ------------------------------------------------------------------------------------------------ app
def build_app(worker: Worker, tokenizer, served_name: str, gen_defaults: dict, hf_config: dict, default_thinking):
    app = FastAPI(title="colin-inference-engine")
    fmt = ChatFormat(tokenizer)
    a = worker.args
    if getattr(a, "api_key", None):
        app.add_middleware(ApiKey, key=a.api_key)
    max_len, margin = a.max_seq_len, worker.sched.margin
    vocab = hf_config["vocab_size"]  # rows of the embedding: a token id beyond it is a device-side assert, fatal to the engine
    created = int(time.time())

    # apply_chat_template's own arguments: a chat_template_kwargs key with one of these names would collide with them
    reserved_kwargs = set(inspect.signature(tokenizer.apply_chat_template).parameters) - {"self", "kwargs"}

    async def json_body(request: Request) -> dict:
        try:
            b = await request.json()
        except ValueError:
            raise BadRequest("the body is not valid JSON") from None
        if not isinstance(b, dict):
            raise BadRequest("the body must be a JSON object")
        return b

    def body_of(raw: dict) -> dict:
        b = dict(raw)
        extra = b.pop("extra_body", None)
        if isinstance(extra, dict):
            b.update({k: v for k, v in extra.items() if k not in b})
        if b.get("n", 1) not in (1, None):
            raise BadRequest("n > 1 is not supported")
        # fields the engine cannot honor: reject them rather than ignore them, unless they leave the output unchanged
        for k, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            if b.get(k) is not None and _num(b[k], float, k) != neutral:
                raise BadRequest(f"{k} is not supported (only {neutral:g}, the neutral value)")
        rf = b.get("response_format")
        if rf is not None and not (isinstance(rf, dict) and rf.get("type") == "text"):
            raise BadRequest("response_format is not supported except {\"type\": \"text\"}: the server has no constrained decoding")
        return b

    def token_ids(ids, name: str) -> tuple[int, ...]:
        if not all(isinstance(t, int) and 0 <= t < vocab for t in ids):
            raise BadRequest(f"{name} must be token ids in [0, {vocab})")
        return tuple(ids)

    def make_request(b: dict, prompt: list[int], hook, logprobs: int | None = None) -> EngineRequest:
        if worker.args.max_queue and worker.queued() >= worker.args.max_queue:
            raise Busy(f"the server is busy: {worker.args.max_queue} requests are waiting")
        if not prompt:
            raise BadRequest("the prompt is empty")
        token_ids(prompt, "prompt")
        if len(prompt) + margin > max_len:  # the wording of the Anthropic API, which Claude Code recognizes
            raise BadRequest(f"prompt is too long: {len(prompt)} tokens > {max_len - margin} maximum")
        mt = b.get("max_completion_tokens")
        if mt is None:  # the older name; an `or` would read an explicit max_completion_tokens 0 as unset
            mt = b.get("max_tokens")
        room = max_len - margin - len(prompt) + 1
        mt = room if mt is None else min(_num(mt, int, "max_tokens"), room)
        if mt < 1:
            raise BadRequest("max_tokens must be at least 1")
        if worker.args.max_output_tokens:
            mt = min(mt, worker.args.max_output_tokens)
        temp = b.get("temperature")
        temp = gen_defaults.get("temperature", 1.0) if temp is None else _num(temp, float, "temperature")
        top_k = b.get("top_k")
        top_k = gen_defaults.get("top_k", 0) if top_k is None else _num(top_k, int, "top_k")
        top_p = b.get("top_p")
        top_p = gen_defaults.get("top_p", 1.0) if top_p is None else _num(top_p, float, "top_p")
        seed = b.get("seed")
        seed = random.getrandbits(62) if seed is None else _num(seed, int, "seed") & ((1 << 62) - 1)
        stop_ids = b.get("stop_token_ids") or []
        if not isinstance(stop_ids, list):
            raise BadRequest("stop_token_ids must be a list of token ids")
        stop_ids = token_ids(stop_ids, "stop_token_ids")
        eos = stop_ids if b.get("ignore_eos") else tuple(sorted(set(fmt.eos_ids) | set(stop_ids)))
        min_p = _num(b.get("min_p") or 0.0, float, "min_p")
        if not (0.0 <= temp <= 100.0) or not (0.0 < top_p <= 1.0) or not (0.0 <= min_p <= 1.0):
            raise BadRequest("temperature must be in [0, 100], top_p in (0, 1] and min_p in [0, 1]")
        salt = b.get("cache_salt")
        if b.get("cache_prompt") is False:  # llama.cpp's switch: no prefix reuse for this request
            salt = uuid.uuid4().hex
        return EngineRequest(prompt, max_new_tokens=mt, temperature=temp, top_k=max(top_k, 0), top_p=top_p,
                             min_p=min_p, seed=seed, eos_ids=eos,
                             min_tokens=_num(b.get("min_tokens") or 0, int, "min_tokens"),
                             logprobs=logprobs, hook=hook, cache_salt=None if salt is None else str(salt),
                             max_seconds=worker.args.max_request_seconds)

    def stops_of(b: dict) -> list[str]:
        st = b.get("stop")
        if st is None or isinstance(st, str):
            return [st] if st else []
        if isinstance(st, list) and all(isinstance(x, str) for x in st):
            return st
        raise BadRequest("stop must be a string or a list of strings")

    def include_usage_of(b: dict) -> bool:
        so = b.get("stream_options")
        return isinstance(so, dict) and bool(so.get("include_usage"))

    async def render(msgs, tools, kw: dict) -> list[int]:
        """The prompt of a chat. Malformed messages and the template's own checks (no user message, an unknown role) are
        the client's errors; any other exception propagates as a bug of the server."""
        try:
            check_messages(msgs, tools)
        except ValueError as e:
            raise BadRequest(str(e)) from None
        if reserved_kwargs & set(kw):
            raise BadRequest(f"chat_template_kwargs cannot set {sorted(reserved_kwargs & set(kw))}")
        try:
            return await asyncio.to_thread(fmt.render, msgs, tools, **kw)
        except jinja2.TemplateError as e:
            raise BadRequest(f"chat template: {e}") from None

    def log_done(req: EngineRequest, kind: str):
        t = _timings(req)
        log(f"[{kind} {req.rid}] slot {req.slot} prompt {len(req.prompt)} (cached {req.reused}) -> {len(req.output)} tok, "
            f"{req.finish_reason}; queue {t['queue_s']:.2f}s ttft {t['ttft_s']:.2f}s decode {t['decode_tok_s'] or 0:.1f} tok/s")

    # -------------------------------------------------------------------------------------- the two ways to answer
    # Every endpoint answers through these two: result() for a reply in one body, sse_response() for a stream. Each
    # endpoint gives only the formatting of its API.
    async def result(request, st: Stream, req: EngineRequest, kind: str) -> tuple[list, EngineRequest]:
        """Waits for a request without a stream: (its output events ("delta", parser events, ids, logprobs), the
        finished request). Raises Failed on an engine error, and aborts the request (499) when the client goes away."""
        events = []
        while True:
            try:
                item = await asyncio.wait_for(st.q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    worker.abort(req)
                    raise Failed("client disconnected", 499)
                continue
            if item[0] == "error":
                raise Failed(item[1], item[2])
            if item[0] == "done":
                log_done(item[1], kind)
                return events, item[1]
            events.append(item)

    def sse_response(st: Stream, req: EngineRequest, kind: str, on_delta, on_done, on_error, head=(), ping=None):
        """Streams a request: the events of `head`, on_delta(event) for each output event, then on_done(request) or
        on_error(message, code). Each returns a list of SSE strings. ping: (seconds, event), sent while no output
        arrives (the queue, a long prefill). Aborts the request when the client goes away before the end."""
        async def gen():
            finished = False
            try:
                for x in head:
                    yield x
                while True:
                    try:
                        item = await asyncio.wait_for(st.q.get(), timeout=ping[0] if ping else None)
                    except asyncio.TimeoutError:
                        yield ping[1]
                        continue
                    if item[0] == "delta":
                        for x in on_delta(item):
                            yield x
                        continue
                    finished = True
                    if item[0] == "error":
                        for x in on_error(item[1], item[2]):
                            yield x
                    else:
                        for x in on_done(item[1]):
                            yield x
                        log_done(item[1], kind)
                    return
            finally:
                if not finished:
                    worker.abort(req)  # the client went away
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def sse_error(msg: str, code: int) -> list[str]:
        return [_sse(_error_body(msg, code))]

    def stream_end(rid: str, obj: str, model: str, r: EngineRequest, include_usage: bool) -> list[str]:
        """The last events of an OpenAI stream: the usage chunk (stream_options.include_usage), then [DONE]."""
        out = []
        if include_usage:
            out.append(_sse({"id": rid, "object": obj, "created": int(time.time()), "model": model, "choices": [], "usage": _usage(r),
                             "timings": _timings(r)}))
        return out + ["data: [DONE]\n\n"]

    # -------------------------------------------------------------------------------------- chat completions
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            b = body_of(await json_body(request))
            msgs = b.get("messages")
            tool_choice = b.get("tool_choice")
            tools = b.get("tools") if tool_choice != "none" else None
            kw = _template_kwargs(b, default_thinking)
            prompt = await render(msgs, tools, kw)
            n_top = None
            if b.get("logprobs"):
                n_top = min(_num(b.get("top_logprobs") or 0, int, "top_logprobs"), MAX_TOP_LOGPROBS)
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser, n_top)
            req = make_request(b, prompt, st, n_top)
            include_usage, want_ids = include_usage_of(b), bool(b.get("return_token_ids"))
        except BadRequest as e:
            return _error(str(e))
        except Busy as e:
            return _error(str(e), 503)
        rid = "chatcmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
        worker.submit(req)

        def chunk(delta, finish=None, extra=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            if extra:
                c["choices"][0].update(extra)
            return c

        def tool_call(x):
            return {"id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": x["arguments"]}}

        def finish_reason(r):
            return "tool_calls" if parser.n_tool_calls and r.finish_reason == "stop" else _finish_reason(r)

        if b.get("stream"):
            role_sent, first, n_calls, carry_ids, carry_lps = False, True, 0, [], []

            def role():  # the role chunk goes out with the first output, as vLLM does
                nonlocal role_sent
                if role_sent:
                    return []
                role_sent = True
                return [_sse(chunk({"role": "assistant", "content": ""}))]

            def on_delta(item):
                nonlocal first, n_calls, carry_ids, carry_lps
                _, ev, ids, lps = item
                carry_ids += ids
                carry_lps += lps
                out, delta = role(), {}
                for kind, x in ev:
                    if kind == "tool_call":
                        delta.setdefault("tool_calls", []).append({"index": n_calls, **tool_call(x)})
                        n_calls += 1
                    elif kind == "reasoning":
                        delta["reasoning_content"] = delta.get("reasoning_content", "") + x
                        delta["reasoning"] = delta["reasoning_content"]
                    else:
                        delta["content"] = delta.get("content", "") + x
                if not delta:
                    return out
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
                return out + [_sse(c)]

            def on_done(r):  # the last chunk carries the ids / logprobs of trailing tokens without text (the stop token)
                extra = {}
                if want_ids and carry_ids:
                    extra["token_ids"] = carry_ids
                if n_top is not None and carry_lps:
                    extra["logprobs"] = {"content": carry_lps}
                return role() + [_sse(chunk({}, finish_reason(r), extra))] + stream_end(rid, "chat.completion.chunk", model, r, include_usage)
            return sse_response(st, req, "chat", on_delta, on_done, sse_error)

        try:
            events, r = await result(request, st, req, "chat")
        except Failed as e:
            return _error(e.msg, e.code)
        reasoning, content, calls, lps = "", "", [], []
        for _, ev, _, item_lps in events:
            for kind, x in ev:
                if kind == "tool_call":
                    calls.append(tool_call(x))
                elif kind == "reasoning":
                    reasoning += x
                else:
                    content += x
            lps += item_lps
        msg = {"role": "assistant", "content": content if (content or not calls) else None}
        if reasoning:
            msg["reasoning_content"] = msg["reasoning"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        choice = {"index": 0, "message": msg, "logprobs": {"content": lps} if n_top is not None else None, "finish_reason": finish_reason(r)}
        if want_ids:
            choice["token_ids"] = r.output
        return JSONResponse({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [choice],
                             "usage": _usage(r), "timings": _timings(r)})

    # -------------------------------------------------------------------------------------- completions
    @app.post("/v1/completions")
    async def completions(request: Request):
        try:
            b = body_of(await json_body(request))
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
            n_top = None if b.get("logprobs") is None else min(_num(b["logprobs"], int, "logprobs"), MAX_TOP_LOGPROBS)
            parser = TextParser(tokenizer, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser, n_top)
            req = make_request(b, prompt, st, n_top)
            include_usage = include_usage_of(b)
        except BadRequest as e:
            return _error(str(e))
        except Busy as e:
            return _error(str(e), 503)
        rid = "cmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
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
            carry = []

            def on_delta(item):
                nonlocal carry
                _, ev, _, lps = item
                carry += lps
                text = "".join(x for _, x in ev)
                if not text:
                    return []
                c, carry = chunk(text, None, carry), []
                return [_sse(c)]

            def on_done(r):
                return [_sse(chunk("", _finish_reason(r), carry))] + stream_end(rid, "text_completion", model, r, include_usage)
            return sse_response(st, req, "cmpl", on_delta, on_done, sse_error)

        try:
            events, r = await result(request, st, req, "cmpl")
        except Failed as e:
            return _error(e.msg, e.code)
        text = "".join(x for _, ev, _, _ in events for _, x in ev)
        lps = [e for _, _, _, item_lps in events for e in item_lps]
        return JSONResponse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "text": text, "logprobs": lp_block(lps), "finish_reason": _finish_reason(r)}],
                             "usage": _usage(r), "timings": _timings(r)})

    # -------------------------------------------------------------------------------------- Anthropic messages
    async def render_messages(b: dict):
        """Anthropic request body -> (prompt token ids, OpenAI-style tools)."""
        try:
            msgs, tools = anth.to_messages(b), anth.to_tools(b)
            kw = anth.template_kwargs(b, default_thinking)
        except ValueError as e:  # the conversion's checks of the request
            raise BadRequest(str(e)) from None
        return await render(msgs, tools, kw), tools

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            prompt, _ = await render_messages(await json_body(request))
        except BadRequest as e:
            return _anth_error(str(e))
        return {"input_tokens": len(prompt)}

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            b = await json_body(request)
            prompt, tools = await render_messages(b)
            stops = b.get("stop_sequences") or []
            if not isinstance(stops, list):
                raise BadRequest("stop_sequences must be a list of strings")
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, [str(s) for s in stops])
            st = Stream(asyncio.get_running_loop(), tokenizer, parser, None)
            req = make_request(b, prompt, st)
        except BadRequest as e:
            return _anth_error(str(e))
        except Busy as e:
            return _anth_error(str(e), 529)
        mid = "msg_" + uuid.uuid4().hex[:24]
        model = b.get("model") or served_name
        blocks = anth.Blocks(anth.drops_tool_calls(b))
        worker.submit(req)

        def stop_of(r: EngineRequest):
            return anth.stop_reason(r.finish_reason, blocks.n_tool_calls, parser.stop_match)

        def use(r: EngineRequest):
            return anth.usage(len(r.prompt), r.reused, len(r.output))

        if b.get("stream"):
            def on_done(r):
                sr, seq = stop_of(r)
                return blocks.close() + [anth.sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": sr, "stop_sequence": seq},
                                                                    "usage": use(r)}),
                                         anth.sse("message_stop", {"type": "message_stop"})]
            start = anth.message(mid, model, [], (None, None), anth.usage(len(prompt), 0, 0))
            return sse_response(st, req, "msg", lambda item: blocks.add(item[1]), on_done,
                                lambda msg, code: [anth.sse("error", _anth_error_body(msg, code))],
                                head=[anth.sse("message_start", {"type": "message_start", "message": start})],
                                ping=(anth.PING_S, anth.sse("ping", {"type": "ping"})))

        try:
            events, r = await result(request, st, req, "msg")
        except Failed as e:
            return _anth_error(e.msg, e.code)
        for _, ev, _, _ in events:
            blocks.add(ev)
        blocks.close()
        return JSONResponse(anth.message(mid, model, blocks.content, stop_of(r), use(r)))

    # -------------------------------------------------------------------------------------- the rest
    @app.exception_handler(Exception)
    async def server_error(request: Request, e: Exception):
        """A bug of the server, not a bad request: a 500 in the endpoint's format. Starlette raises the exception again
        afterwards, and uvicorn logs its traceback."""
        msg = f"internal error: {type(e).__name__}: {e}"
        return _anth_error(msg, 500) if request.url.path.startswith("/v1/messages") else _error(msg, 500)

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
    ap.add_argument("--slots", type=int, default=3, help="conversations whose KV stays cached; one request runs at a time")
    ap.add_argument("--max-queue", type=int, default=8,
                    help="requests that can wait for the running one; past it, a request gets a 503 (/v1/messages: 529). 0: no limit")
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="cap on the output tokens of a request, whatever its max_tokens asks (0: the room left in the slot)")
    ap.add_argument("--max-request-seconds", type=float, default=1800.0,
                    help="a request ends this long after it arrived, running or queued, with finish_reason length (0: never)")
    ap.add_argument("--max-seq-len", type=int, default=262144, help="tokens per slot (prompt + output)")
    ap.add_argument("--spec", choices=("mtp", "none"), default="mtp")
    ap.add_argument("--drafter-weights", default="auto",
                    help="MTP head weights: auto = ~/.cache/colinfer/drafter/mtp_ft.safetensors (tools/train_drafter.py) when it "
                         "exists, none = the checkpoint's, or a path. Drafts only affect speed, never outputs")
    ap.add_argument("--k", type=int, default=7, help="longest MTP draft; each cycle picks 3 or k from the measured acceptance")
    ap.add_argument("--suffix-drafts", type=int, default=MIN_MATCH, metavar="N",
                    help="with MTP: draft the continuation of an earlier occurrence of the last N+ tokens (prompt or reply so far), up to "
                         "15 tokens when one request decodes; 0 = off. Speeds up replies that repeat their input (code edits)")
    ap.add_argument("--checkpoints", type=int, default=32, help="prefix checkpoint ring size (154 MB each)")
    ap.add_argument("--decode-weights", choices=("int", "checkpoint"), default="int",
                    help="int: decode the attention / GDN projections from INT6 / INT5 copies (tools/int6_requant.py; ~9.5%% faster, "
                         "perplexity within 0.25%%) when their files exist; checkpoint: from the FP8 weights")
    ap.add_argument("--no-prefix-caching", action="store_true", help="never reuse a prompt prefix (benchmarking raw prefill; = --checkpoints 0)")
    ap.add_argument("--mem-cap-gb", type=float, default=80.0, help="hard cap on this process's GPU memory (torch allocator)")
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                    help="default enable_thinking (auto: the template's default, on; for /v1/messages without a thinking field: off)")
    ap.add_argument("--api-key", default=os.environ.get("COLINFER_API_KEY") or None,
                    help="require this key on the /v1/ endpoints (Authorization: Bearer, or x-api-key); default: $COLINFER_API_KEY, else no key")
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
