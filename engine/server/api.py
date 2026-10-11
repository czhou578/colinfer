"""OpenAI- and Anthropic-compatible HTTP server.

    uv run python -m engine.server [--port 8000] [--slots 3] [--max-seq-len 262144] [--spec mtp|none]

Endpoints: /v1/chat/completions and /v1/completions (SSE streaming, usage + timings, logprobs, stop strings, seeds,
tools, thinking; engine/server/openai.py formats them), /v1/messages and /v1/messages/count_tokens (the Anthropic
Messages API, engine/server/anthropic.py), /v1/models, /health, /metrics (Prometheus), /v1/status (slots and
checkpoints). With --api-key, the /v1/ endpoints require the key (Authorization: Bearer, or x-api-key).

The engine thread (engine/server/worker.py) owns the GPU. The HTTP handlers (asyncio, uvicorn) render the chat template
and tokenize outside the event loop. They give the request to the engine thread through a queue, and get the output
events back through an asyncio queue.

The engine thread detokenizes and parses each token as it emits it (engine/server/chat.py). Thus a stop string ends a
request in the same step that produced it. One request runs at a time; the others wait in a FIFO queue.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import json
import os
import random
import time
import uuid

import jinja2
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from engine.runtime.scheduler import Request as EngineRequest
from engine.server import anthropic as anth
from engine.server import openai as oai
from engine.server.chat import ChatFormat, OutputParser, TextParser, check_messages, template_kwargs
from engine.server.worker import Stream, Worker, log
from engine.spec.suffix import MIN_MATCH

PING_S = 10.0  # seconds between the keep-alive events of a stream that has no output yet (the queue, a long prefill)


def _is_anthropic(path: str) -> bool:
    return path.startswith("/v1/messages")


def _error(msg, code=400):
    return JSONResponse(oai.error_body(msg, code), status_code=code)


def _anth_error(msg, code=400):
    return JSONResponse(anth.error_body(msg, code), status_code=code)


class ApiKey:
    """ASGI middleware: the /v1/ endpoints require the key, as `Authorization: Bearer <key>` or `x-api-key: <key>`. The
    401 has the format of the API of the path."""

    def __init__(self, app, key: str):
        self.app, self.key = app, key.encode()

    def _ok(self, headers) -> bool:
        h = {k.decode("latin-1").lower(): v for k, v in headers}
        auth = h.get("authorization", b"")
        bearer = auth[7:] if auth[:7].lower() == b"bearer " else b""
        return hmac.compare_digest(h.get("x-api-key", b""), self.key) or hmac.compare_digest(bearer, self.key)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/v1/") and not self._ok(scope["headers"]):
            msg = "invalid or missing API key"
            body = anth.error_body(msg, 401) if _is_anthropic(scope["path"]) else oai.error_body(msg, 401)
            return await JSONResponse(body, status_code=401)(scope, receive, send)
        await self.app(scope, receive, send)


class BadRequest(Exception):
    """The client's error (a 400). The endpoints turn only this into an error reply; any other exception is a bug of the
    server, which the app's handler logs and answers with a 500."""


class Busy(Exception):
    """Too many requests wait already (a 503; a 529 on the Anthropic API, which Claude Code retries)."""


class Failed(Exception):
    """A request that the engine failed, or whose client went away: the message and the HTTP status."""

    def __init__(self, msg: str, code: int):
        super().__init__(msg)
        self.msg, self.code = msg, code


def _num(v, kind, name: str):
    """A request field as an int / float, or a BadRequest."""
    try:
        return kind(v)
    except (TypeError, ValueError, OverflowError):  # OverflowError: int(Infinity), which Python's JSON parser accepts
        raise BadRequest(f"{name} must be a number, not {v!r}") from None


# ------------------------------------------------------------------------------------------------ app
def build_app(worker: Worker, tokenizer, served_name: str, gen_defaults: dict, hf_config: dict, default_thinking,
              default_effort: str | None = None):
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

    def stops_of(b: dict, key: str = "stop") -> list[str]:
        """The stop strings of a request: `stop` (OpenAI: a string or a list) or `stop_sequences` (Anthropic: a list)."""
        st = b.get(key)
        if st is None or isinstance(st, str):
            return [st] if st else []
        if isinstance(st, list) and all(isinstance(x, str) for x in st):
            return st
        raise BadRequest(f"{key} must be a string or a list of strings")

    def include_usage_of(b: dict) -> bool:
        so = b.get("stream_options")
        return isinstance(so, dict) and bool(so.get("include_usage"))

    def kwargs_of(b: dict) -> dict:
        """The chat template kwargs of a request (engine/server/chat.py template_kwargs); a bad value is the client's error."""
        try:
            return template_kwargs(b, default_thinking, default_effort)
        except ValueError as e:
            raise BadRequest(str(e)) from None

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
        t = oai.timings(req)
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

    def sse_response(st: Stream, req: EngineRequest, kind: str, on_delta, on_done, on_error, head=(), ping=""):
        """Streams a request: the events of `head`, on_delta(event) for each output event, then on_done(request) or
        on_error(message, code). Each returns a list of SSE strings. ping: the text sent every PING_S seconds while no
        output arrives (the queue, a long prefill), so that an idle timeout of the client or a proxy does not end the
        stream. Aborts the request when the client goes away before the end."""
        async def gen():
            finished = False
            try:
                for x in head:
                    yield x
                while True:
                    try:
                        item = await asyncio.wait_for(st.q.get(), timeout=PING_S)
                    except asyncio.TimeoutError:
                        yield ping
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
        return [oai.sse(oai.error_body(msg, code))]

    openai_ping = ": keep-alive\n\n"  # an SSE comment: every client ignores it

    # -------------------------------------------------------------------------------------- chat completions
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            b = body_of(await json_body(request))
            msgs = b.get("messages")
            tool_choice = b.get("tool_choice")
            tools = b.get("tools") if tool_choice != "none" else None
            prompt = await render(msgs, tools, kwargs_of(b))
            n_top = None
            if b.get("logprobs"):
                n_top = min(_num(b.get("top_logprobs") or 0, int, "top_logprobs"), oai.MAX_TOP_LOGPROBS)
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser)
            req = make_request(b, prompt, st, n_top)
            include_usage, want_ids = include_usage_of(b), bool(b.get("return_token_ids"))
        except BadRequest as e:
            return _error(str(e))
        except Busy as e:
            return _error(str(e), 503)
        rid = "chatcmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
        worker.submit(req)

        def finish_reason(r):
            return "tool_calls" if parser.n_tool_calls and r.finish_reason == "stop" else oai.finish_reason(r)

        if b.get("stream"):
            role_sent, first, n_calls, carry_ids, carry_lps = False, True, 0, [], []

            def role():  # the role chunk goes out with the first output, as vLLM does
                nonlocal role_sent
                if role_sent:
                    return []
                role_sent = True
                return [oai.sse(oai.chat_chunk(rid, model, {"role": "assistant", "content": ""}))]

            def on_delta(item):
                nonlocal first, n_calls, carry_ids, carry_lps
                _, ev, ids, lps = item
                carry_ids += ids
                carry_lps += lps
                out, delta = role(), {}
                for kind, x in ev:
                    if kind == "tool_call":
                        delta.setdefault("tool_calls", []).append({"index": n_calls, **oai.tool_call(x)})
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
                c = oai.chat_chunk(rid, model, delta, None, extra)
                if first:
                    first = False
                    c["request_metrics"] = {"time_to_first_token_s": req.t_first - req.t_submit, "queue_time_s": req.t_admit - req.t_submit,
                                            "prompt_time_s": req.t_first - req.t_admit}
                carry_ids, carry_lps = [], []
                return out + [oai.sse(c)]

            def on_done(r):  # the last chunk carries the ids / logprobs of trailing tokens without text (the stop token)
                extra = {}
                if want_ids and carry_ids:
                    extra["token_ids"] = carry_ids
                if n_top is not None and carry_lps:
                    extra["logprobs"] = {"content": carry_lps}
                return (role() + [oai.sse(oai.chat_chunk(rid, model, {}, finish_reason(r), extra))]
                        + oai.stream_end(rid, "chat.completion.chunk", model, r, include_usage))
            return sse_response(st, req, "chat", on_delta, on_done, sse_error, ping=openai_ping)

        try:
            events, r = await result(request, st, req, "chat")
        except Failed as e:
            return _error(e.msg, e.code)
        msg, lps = oai.chat_message(events)
        choice = {"index": 0, "message": msg, "logprobs": {"content": lps} if n_top is not None else None, "finish_reason": finish_reason(r)}
        if want_ids:
            choice["token_ids"] = r.output
        return JSONResponse({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [choice],
                             "usage": oai.usage(r), "timings": oai.timings(r)})

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
            n_top = None if b.get("logprobs") is None else min(_num(b["logprobs"], int, "logprobs"), oai.MAX_TOP_LOGPROBS)
            parser = TextParser(tokenizer, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser)
            req = make_request(b, prompt, st, n_top)
            include_usage = include_usage_of(b)
        except BadRequest as e:
            return _error(str(e))
        except Busy as e:
            return _error(str(e), 503)
        rid = "cmpl-" + uuid.uuid4().hex
        model = b.get("model") or served_name
        worker.submit(req)

        if b.get("stream"):
            carry = []

            def on_delta(item):
                nonlocal carry
                _, ev, _, lps = item
                carry += lps
                text = "".join(x for _, x in ev)
                if not text:
                    return []
                c, carry = oai.text_chunk(rid, model, text, None, carry, n_top), []
                return [oai.sse(c)]

            def on_done(r):
                last = oai.text_chunk(rid, model, "", oai.finish_reason(r), carry, n_top)
                return [oai.sse(last)] + oai.stream_end(rid, "text_completion", model, r, include_usage)
            return sse_response(st, req, "cmpl", on_delta, on_done, sse_error, ping=openai_ping)

        try:
            events, r = await result(request, st, req, "cmpl")
        except Failed as e:
            return _error(e.msg, e.code)
        text, lps = oai.completion_text(events)
        return JSONResponse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "text": text, "logprobs": oai.lp_block(lps, n_top), "finish_reason": oai.finish_reason(r)}],
                             "usage": oai.usage(r), "timings": oai.timings(r)})

    # -------------------------------------------------------------------------------------- Anthropic messages
    async def render_messages(b: dict):
        """Anthropic request body -> (prompt token ids, OpenAI-style tools)."""
        try:
            msgs, tools = anth.to_messages(b), anth.to_tools(b)
            kw = anth.template_kwargs(b, default_thinking, default_effort)
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
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops_of(b, "stop_sequences"))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser)
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
                                lambda msg, code: [anth.sse("error", anth.error_body(msg, code))],
                                head=[anth.sse("message_start", {"type": "message_start", "message": start})],
                                ping=anth.sse("ping", {"type": "ping"}))

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
        return _anth_error(msg, 500) if _is_anthropic(request.url.path) else _error(msg, 500)

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
    ap.add_argument("--reasoning-effort", choices=("auto", "low", "medium", "xhigh"), default="auto",
                    help="the reasoning effort of a thinking request that does not set one (auto: the template's default, xhigh)")
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
    effort = None if a.reasoning_effort == "auto" else a.reasoning_effort
    app = build_app(worker, tokenizer, a.served_model_name or a.model, gen_defaults, hf_config, thinking, effort)
    log(f"[server] startup {time.perf_counter() - t0:.1f} s; listening on http://{a.host}:{a.port}/v1")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
