"""OpenAI- and Anthropic-compatible HTTP server.

    uv run python -m engine.server [--port 8000] [--slots 3] [--max-seq-len 524288] [--spec mtp|none]

The values above are the defaults for the 27B. The defaults of --slots, --max-seq-len and --spec come from the family
of the model (engine/model/family.py).

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
import functools
import hmac
import inspect
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import jinja2
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from engine.model.family import check_served, family_of
from engine.runtime.build import NotEnoughMemory, check_memory
from engine.runtime.grammar import Grammars
from engine.runtime.scheduler import Request as EngineRequest
from engine.server import anthropic as anth
from engine.server import committee as cm
from engine.server import images as img
from engine.server import openai as oai
from engine.server.chat import ChatFormat, OutputParser, TextParser, check_messages, template_kwargs
from engine.server.worker import ServerConfig, Stream, Worker
from engine.weights.loader import resolve

PING_S = 10.0  # seconds between the keep-alive events of a stream that has no output yet (the queue, a long prefill)
NO_MEMORY_EXIT = 3  # the exit status of a start without the memory; deploy/colinfer.service does not restart on it


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


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
    """The client's error: a 400, or the HTTP status in `code`. The endpoints turn only this into an error reply; any
    other exception is a bug of the server, which the app's handler logs and answers with a 500."""

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.code = code


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
              default_effort: str | None = None, committee: cm.Committee | None = None):
    """committee: the roles of committee mode (engine/server/committee.py); None: from the config (--committee-size,
    --committee-roles), which can turn the mode off."""
    app = FastAPI(title="colin-inference-engine")
    fmt = ChatFormat(tokenizer)
    cfg = worker.cfg
    if committee is None:
        committee = cm.Committee.from_config(cfg.committee_size, cfg.committee_roles)
    if cfg.api_key:
        app.add_middleware(ApiKey, key=cfg.api_key)
    max_len, margin = cfg.max_seq_len, worker.sched.margin
    # the prompt text is checked before its tokenize, not only its tokens after it: the tokenize holds host memory in
    # proportion to the text (1 GB for 4M digits, one token each). The limit counts bytes of UTF-8: a byte-level BPE token
    # covers at least one byte, and a rare 4-byte character is up to 4 tokens. Text and code use 2 to 4 bytes per token
    max_text_bytes = 8 * max_len
    heavy = ThreadPoolExecutor(1, thread_name_prefix="heavy")  # one tokenize or image decode at a time, for the same reason
    # the queue check of make_request comes after the body, the images and the render, so this bound holds the memory of the
    # requests before it (engine/runtime/build.py GROWTH_RESERVE_BYTES counts them)
    admit = asyncio.Semaphore(cfg.max_queue + 1)
    vocab = hf_config["vocab_size"]  # rows of the embedding: a token id beyond it is a device-side assert, fatal to the engine
    vision = getattr(worker.sched, "vision", None) is not None  # image input (engine/server/images.py)
    improc = img.ImageProcessor(resolve(cfg.model), cfg.image_max_pixels) if vision else None
    created = int(time.time())

    # apply_chat_template's own arguments: a chat_template_kwargs key with one of these names would collide with them
    reserved_kwargs = set(inspect.signature(tokenizer.apply_chat_template).parameters) - {"self", "kwargs"}

    async def json_body(request: Request) -> dict:
        """The body as a JSON object. The render and the tokenize hold host memory in proportion to the body, so a body
        past cfg.max_body_bytes is a 413, and the server stops reading it at the limit."""
        limit = cfg.max_body_bytes

        def too_large():  # a new exception at each raise: a stored one and this frame's traceback hold `raw` in a cycle
            return BadRequest(f"the request body is larger than {limit} bytes", 413)
        n = request.headers.get("content-length", "")
        if n.isdigit() and int(n) > limit:
            raise too_large()
        raw, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise too_large()
            raw.append(chunk)
        try:
            b = json.loads(b"".join(raw))
        except ValueError:
            raise BadRequest("the body is not valid JSON") from None
        if not isinstance(b, dict):
            raise BadRequest("the body must be a JSON object")
        return b

    def admitted(endpoint):
        """Bounds the requests between the body read and the submit to cfg.max_queue + 1. The next one gets the Busy reply
        of its API before the server reads its body."""
        @functools.wraps(endpoint)
        async def wrapper(request: Request):
            if admit.locked():
                msg = f"the server is busy: {cfg.max_queue + 1} requests are between the body read and the queue"
                return _anth_error(msg, 529) if _is_anthropic(request.url.path) else _error(msg, 503)
            async with admit:
                return await endpoint(request)
        return wrapper

    async def encode_text(text: str, encode) -> list[int]:
        """The token ids of a prompt text: the byte check, then `encode` on the heavy thread."""
        n = len(text) if text.isascii() else len(text.encode())
        if n > max_text_bytes:
            raise BadRequest(f"prompt is too long: {n} bytes of text > {max_text_bytes} maximum")
        return await asyncio.get_running_loop().run_in_executor(heavy, encode, text)

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
        return b

    def token_ids(ids, name: str) -> tuple[int, ...]:
        if not all(isinstance(t, int) and 0 <= t < vocab for t in ids):
            raise BadRequest(f"{name} must be token ids in [0, {vocab})")
        return tuple(ids)

    grammars = []  # the llguidance tokenizer, built on the first constrained request (~1.5 s)

    def openai_constraint(b: dict, tools) -> tuple | None:
        """(kind, arguments) of the constraint that an OpenAI request asks for: forced tool calls (tool_choice required or
        a named function; parallel_tool_calls false: one call) or JSON (response_format json_object / json_schema)."""
        tc = b.get("tool_choice")
        if tc == "required" or isinstance(tc, dict):
            name = None
            if isinstance(tc, dict):
                fn = tc.get("function") if tc.get("type") == "function" else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if not isinstance(name, str):
                    raise BadRequest("tool_choice must be none, auto, required or {\"type\": \"function\", \"function\": {\"name\": ...}}")
            if not tools:
                raise BadRequest("tool_choice required or a named function needs tools")
            return "tools", dict(tools=tools, names=None if name is None else [name], single=b.get("parallel_tool_calls") is False)
        if tc not in (None, "auto", "none"):
            raise BadRequest("tool_choice must be none, auto, required or a named function")
        rf = b.get("response_format")
        if rf is None or (isinstance(rf, dict) and rf.get("type") == "text"):
            return None
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            return "json", dict(schema=None)
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            js = rf.get("json_schema")
            schema = js.get("schema") if isinstance(js, dict) else None
            if not isinstance(schema, dict):
                raise BadRequest("response_format json_schema needs json_schema.schema, a JSON schema object")
            return "json", dict(schema=schema)
        raise BadRequest("response_format type must be text, json_object or json_schema")

    def anthropic_constraint(b: dict, tools) -> tuple | None:
        """The same for an Anthropic request: tool_choice any / tool (disable_parallel_tool_use: one call), or
        output_format (also output_config.format) json_schema."""
        tc = b.get("tool_choice")
        if isinstance(tc, dict) and tc.get("type") in ("any", "tool"):
            name = tc.get("name") if tc.get("type") == "tool" else None
            if tc.get("type") == "tool" and not isinstance(name, str):
                raise BadRequest("a tool_choice of type tool needs the tool's name")
            if not tools:
                raise BadRequest(f"tool_choice {tc.get('type')} needs tools")
            return "tools", dict(tools=tools, names=None if name is None else [name], single=bool(tc.get("disable_parallel_tool_use")))
        oc = b.get("output_config")
        of = b.get("output_format") or (oc.get("format") if isinstance(oc, dict) else None)
        if of is None:
            return None
        if not (isinstance(of, dict) and of.get("type") == "json_schema" and isinstance(of.get("schema"), dict)):
            raise BadRequest("output_format must be {\"type\": \"json_schema\", \"schema\": {...}}")
        return "json", dict(schema=of["schema"])

    async def constraint(spec: tuple | None, prompt: list[int], chat: bool = True):
        """The constraint of a request (engine/runtime/grammar.py), with free reasoning first when its prompt opens it.
        A schema or a tool name that the grammar cannot take is the client's error."""
        if spec is None:
            return None
        kind, kw = spec
        try:
            if not grammars:
                grammars.append(await asyncio.to_thread(Grammars, fmt, vocab))
            return await asyncio.to_thread(getattr(grammars[0], kind), think=chat and fmt.opens_in_reasoning(prompt), **kw)
        except ValueError as e:
            raise BadRequest(str(e)) from None

    def make_request(b: dict, prompt: list[int], hook, logprobs: int | None = None, images=(), constraint=None) -> EngineRequest:
        if cfg.max_queue and worker.queued() >= cfg.max_queue:
            raise Busy(f"the server is busy: {cfg.max_queue} requests are waiting")
        if not prompt:
            raise BadRequest("the prompt is empty")
        token_ids([t for t in prompt if t < vocab] if images else prompt, "prompt")  # image tokens: ids past the vocabulary
        if len(prompt) + margin > max_len:  # the wording of the Anthropic API, which Claude Code recognizes
            raise BadRequest(f"prompt is too long: {len(prompt)} tokens > {max_len - margin} maximum")
        mt = b.get("max_completion_tokens")
        if mt is None:  # the older name; an `or` would read an explicit max_completion_tokens 0 as unset
            mt = b.get("max_tokens")
        room = max_len - margin - len(prompt) + 1
        mt = room if mt is None else min(_num(mt, int, "max_tokens"), room)
        if mt < 1:
            raise BadRequest("max_tokens must be at least 1")
        if cfg.max_output_tokens:
            mt = min(mt, cfg.max_output_tokens)
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
        salt, keep = b.get("cache_salt"), True
        if b.get("cache_prompt") is False:  # llama.cpp's switch: no prefix reuse for this request
            salt, keep = uuid.uuid4().hex, False
        return EngineRequest(prompt, max_new_tokens=mt, temperature=temp, top_k=max(top_k, 0), top_p=top_p,
                             min_p=min_p, seed=seed, eos_ids=eos,
                             min_tokens=_num(b.get("min_tokens") or 0, int, "min_tokens"),
                             logprobs=logprobs, hook=hook, cache_salt=None if salt is None else str(salt),
                             max_seconds=cfg.max_request_seconds, disk=keep, images=list(images), constraint=constraint)

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

    async def render(msgs, tools, kw: dict) -> tuple[list[int], list]:
        """(the prompt of a chat, its images). Malformed messages, unreadable images and the template's own checks (no
        user message, an unknown role) are the client's errors; any other exception propagates as a bug of the server."""
        try:
            check_messages(msgs, tools)
            msgs, blobs = img.extract(msgs, vision)
        except ValueError as e:
            raise BadRequest(str(e)) from None
        if reserved_kwargs & set(kw):
            raise BadRequest(f"chat_template_kwargs cannot set {sorted(reserved_kwargs & set(kw))}")
        try:
            images = [await asyncio.get_running_loop().run_in_executor(heavy, improc.load, x) for x in blobs]
        except ValueError as e:
            raise BadRequest(str(e)) from None
        try:
            text = await asyncio.to_thread(fmt.text, msgs, tools, **kw)
        except jinja2.TemplateError as e:
            raise BadRequest(f"chat template: {e}") from None
        prompt = await encode_text(text, fmt.encode)
        if images:
            try:
                prompt = img.expand(prompt, images, worker.sched.image_pad, vocab)
            except ValueError as e:
                raise BadRequest(str(e)) from None
        return prompt, images

    def log_done(req: EngineRequest, kind: str):
        """One line per finished request: what it was, then one segment per phase with the tokens of that phase and its
        rate (engine/server/openai.py timings). A rate the request never reached shows as "-". The last segment, on a
        speculative server, is the draft acceptance: accepted / verified draft tokens, and the output tokens per cycle."""
        t = oai.timings(req)
        num = lambda v, nd, unit: f"{v:.{nd}f}{unit}" if v is not None else "-"  # noqa: E731
        line = (f"[{kind} {req.rid}] slot {req.slot} prompt {len(req.prompt)} (cached {req.reused}) -> {len(req.output)} tok, "
                f"{req.finish_reason}; queue {t['queue_s']:.2f}s ttft {num(t['ttft_s'], 2, 's')}"
                f" | prefill {t['prefill_tokens']} tok {num(t['prefill_tok_s'], 0, ' tok/s')}"
                f" | decode {t['decode_tokens']} tok {num(t['decode_tok_s'], 1, ' tok/s')}")
        if req.drafted:
            line += (f" | accept {100 * t['draft_accept_rate']:.0f}% ({req.accepted}/{req.drafted} drafts, "
                     f"{t['decode_tokens'] / req.cycles:.1f} tok/cycle)")
        log(line)

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
            except asyncio.CancelledError:  # the stream that waits for this request was closed (a committee's members)
                worker.abort(req)
                raise
            if item[0] == "error":
                raise Failed(item[1], item[2])
            if item[0] == "done":
                log_done(item[1], kind)
                return events, item[1]
            events.append(item)

    def sse_response(start, kind: str, on_delta, on_done, on_error, head=(), ping="", busy_code=503):
        """Streams a request: the events of `head`, on_delta(event) for each output event, then on_done(request) or
        on_error(message, code). Each returns a list of SSE strings. start: an awaitable that gives (the Stream, the
        submitted request); a committee's members run inside it (run_committee). While it runs, and then while no output
        arrives (the queue, a long prefill), `ping` goes out every PING_S seconds, so that an idle timeout of the client
        or a proxy does not end the stream. Aborts the request when the client goes away before the end."""
        async def gen():
            req, finished = None, False
            task = asyncio.ensure_future(start)
            try:
                for x in head:
                    yield x
                while True:
                    try:
                        st, req = await asyncio.wait_for(asyncio.shield(task), timeout=PING_S)
                        break
                    except asyncio.TimeoutError:
                        yield ping
                    except Failed as e:
                        finished = True
                        for x in on_error(e.msg, e.code):
                            yield x
                        return
                    except (BadRequest, Busy) as e:
                        finished = True
                        for x in on_error(str(e), busy_code if isinstance(e, Busy) else e.code):
                            yield x
                        return
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
                if not task.done():
                    task.cancel()  # the client went away during a committee's members: result() aborts the running one
                elif req is not None and not finished:
                    worker.abort(req)  # the client went away
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def ready(st: Stream, req: EngineRequest):
        """The `start` of sse_response for a request that is submitted already."""
        return st, req

    async def run_committee(request, b: dict, msgs, tools, kw: dict, stops: list, roles, kind: str) -> list[str]:
        """The members of a committee (engine/server/committee.py), one after another: their answers as text. Each is the
        conversation with a role appended, rendered as the request was, run without a stream and without the request's
        constraint or logprobs. With a seed, member i gets seed + i. A failure of a member fails the request."""
        answers = []
        seed = b.get("seed")
        for i, role in enumerate(roles):
            bi = dict(b)
            if seed is not None:
                bi["seed"] = _num(seed, int, "seed") + i
            prompt, images = await render(cm.member_messages(msgs, role), tools, kw)
            parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops)
            st = Stream(asyncio.get_running_loop(), tokenizer, parser)
            req = make_request(bi, prompt, st, None, images)
            worker.submit(req)
            events, _ = await result(request, st, req, f"{kind} member {i + 1}/{len(roles)}")
            answers.append(cm.answer_text(oai.chat_message(events)[0]))
        return answers

    def sse_error(msg: str, code: int) -> list[str]:
        return [oai.sse(oai.error_body(msg, code))]

    openai_ping = ": keep-alive\n\n"  # an SSE comment: every client ignores it

    # -------------------------------------------------------------------------------------- chat completions
    @app.post("/v1/chat/completions")
    @admitted
    async def chat(request: Request):
        try:
            b = body_of(await json_body(request))
            msgs = b.get("messages")
            tool_choice = b.get("tool_choice")
            tools = b.get("tools") if tool_choice != "none" else None
            spec = openai_constraint(b, tools)
            kw, stops = kwargs_of(b), stops_of(b)
            prompt, images = await render(msgs, tools, kw)
            n_top = None
            if b.get("logprobs"):
                n_top = min(_num(b.get("top_logprobs") or 0, int, "top_logprobs"), oai.MAX_TOP_LOGPROBS)
            model = b.get("model") or served_name
            roles = committee.roles if committee is not None and cm.is_committee(model) else ()
            if roles:  # the body's values are checked now (a 400 here, not an error in the stream); start() makes the requests
                make_request(b, prompt, None, n_top, images)
                parser = st = req = None
            else:
                parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops)
                st = Stream(asyncio.get_running_loop(), tokenizer, parser)
                req = make_request(b, prompt, st, n_top, images, await constraint(spec, prompt))
            include_usage, want_ids = include_usage_of(b), bool(b.get("return_token_ids"))
        except BadRequest as e:
            return _error(str(e), e.code)
        except Busy as e:
            return _error(str(e), 503)
        rid = "chatcmpl-" + uuid.uuid4().hex
        answers: list[str] = []

        async def start():
            """(the Stream, the submitted request) of the reply: with a committee, after its members ran."""
            nonlocal parser, st, req
            if roles:
                answers[:] = await run_committee(request, b, msgs, tools, kw, stops, roles, "chat")
                p, im = await render(cm.reconcile_messages(msgs, answers, committee.reconcile), tools, kw)
                parser = OutputParser(fmt, fmt.opens_in_reasoning(p), tools, stops)
                st = Stream(asyncio.get_running_loop(), tokenizer, parser)
                req = make_request(b, p, st, n_top, im, await constraint(spec, p))
            worker.submit(req)
            return st, req

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
            return sse_response(start(), "chat", on_delta, on_done, sse_error, ping=openai_ping)

        try:
            st, req = await start()
            events, r = await result(request, st, req, "chat")
        except Failed as e:
            return _error(e.msg, e.code)
        except BadRequest as e:
            return _error(str(e), e.code)
        except Busy as e:
            return _error(str(e), 503)
        msg, lps = oai.chat_message(events)
        choice = {"index": 0, "message": msg, "logprobs": {"content": lps} if n_top is not None else None, "finish_reason": finish_reason(r)}
        if want_ids:
            choice["token_ids"] = r.output
        body = {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [choice],
                "usage": oai.usage(r), "timings": oai.timings(r)}
        if roles:
            body["committee"] = {"roles": list(roles), "answers": answers}
        return JSONResponse(body)

    # -------------------------------------------------------------------------------------- completions
    @app.post("/v1/completions")
    @admitted
    async def completions(request: Request):
        try:
            b = body_of(await json_body(request))
            p = b.get("prompt")
            if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
                p = p[0]
            if isinstance(p, str):
                prompt = await encode_text(p, lambda t: tokenizer.encode(t, add_special_tokens=False))
            elif isinstance(p, list) and p and all(isinstance(t, int) for t in p):
                prompt = list(p)
            else:
                raise BadRequest("prompt must be a string or a list of token ids (one prompt per request)")
            if b.get("echo"):
                raise BadRequest("echo is not supported")
            n_top = None if b.get("logprobs") is None else min(_num(b["logprobs"], int, "logprobs"), oai.MAX_TOP_LOGPROBS)
            cons = await constraint(openai_constraint(b, None), prompt, chat=False)
            parser = TextParser(tokenizer, stops_of(b))
            st = Stream(asyncio.get_running_loop(), tokenizer, parser)
            req = make_request(b, prompt, st, n_top, constraint=cons)
            include_usage = include_usage_of(b)
        except BadRequest as e:
            return _error(str(e), e.code)
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
            return sse_response(ready(st, req), "cmpl", on_delta, on_done, sse_error, ping=openai_ping)

        try:
            events, r = await result(request, st, req, "cmpl")
        except Failed as e:
            return _error(e.msg, e.code)
        text, lps = oai.completion_text(events)
        return JSONResponse({"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "text": text, "logprobs": oai.lp_block(lps, n_top), "finish_reason": oai.finish_reason(r)}],
                             "usage": oai.usage(r), "timings": oai.timings(r)})

    # -------------------------------------------------------------------------------------- Anthropic messages
    def convert_messages(b: dict):
        """Anthropic request body -> (chat messages, OpenAI-style tools, chat template kwargs)."""
        try:
            return anth.to_messages(b, vision), anth.to_tools(b), anth.template_kwargs(b, default_thinking, default_effort)
        except ValueError as e:  # the conversion's checks of the request
            raise BadRequest(str(e)) from None

    @app.post("/v1/messages/count_tokens")
    @admitted
    async def count_tokens(request: Request):
        try:
            msgs, tools, kw = convert_messages(await json_body(request))
            prompt, _ = await render(msgs, tools, kw)
        except BadRequest as e:
            return _anth_error(str(e), e.code)
        return {"input_tokens": len(prompt)}

    @app.post("/v1/messages")
    @admitted
    async def messages(request: Request):
        try:
            b = await json_body(request)
            msgs, tools, kw = convert_messages(b)
            spec, stops = anthropic_constraint(b, tools), stops_of(b, "stop_sequences")
            prompt, images = await render(msgs, tools, kw)
            model = b.get("model") or served_name
            roles = committee.roles if committee is not None and cm.is_committee(model) else ()
            if roles:  # the body's values are checked now; start() makes the members and the reply
                make_request(b, prompt, None, None, images)
                parser = st = req = None
            else:
                parser = OutputParser(fmt, fmt.opens_in_reasoning(prompt), tools, stops)
                st = Stream(asyncio.get_running_loop(), tokenizer, parser)
                req = make_request(b, prompt, st, None, images, await constraint(spec, prompt))
        except BadRequest as e:
            return _anth_error(str(e), e.code)
        except Busy as e:
            return _anth_error(str(e), 529)
        mid = "msg_" + uuid.uuid4().hex[:24]
        blocks = anth.Blocks(anth.drops_tool_calls(b))

        async def start():
            """(the Stream, the submitted request) of the reply: with a committee, after its members ran."""
            nonlocal parser, st, req
            if roles:
                answers = await run_committee(request, b, msgs, tools, kw, stops, roles, "msg")
                p, im = await render(cm.reconcile_messages(msgs, answers, committee.reconcile), tools, kw)
                parser = OutputParser(fmt, fmt.opens_in_reasoning(p), tools, stops)
                st = Stream(asyncio.get_running_loop(), tokenizer, parser)
                req = make_request(b, p, st, None, im, await constraint(spec, p))
            worker.submit(req)
            return st, req

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
            first = anth.message(mid, model, [], (None, None), anth.usage(len(prompt), 0, 0))
            return sse_response(start(), "msg", lambda item: blocks.add(item[1]), on_done,
                                lambda msg, code: [anth.sse("error", anth.error_body(msg, code))],
                                head=[anth.sse("message_start", {"type": "message_start", "message": first})],
                                ping=anth.sse("ping", {"type": "ping"}), busy_code=529)

        try:
            st, req = await start()
            events, r = await result(request, st, req, "msg")
        except Failed as e:
            return _anth_error(e.msg, e.code)
        except BadRequest as e:
            return _anth_error(str(e), e.code)
        except Busy as e:
            return _anth_error(str(e), 529)
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
        entry = {"object": "model", "created": created, "owned_by": "colin-inference-engine", "root": cfg.model, "max_model_len": max_len}
        data = [{"id": served_name, **entry, "config": hf_config}]
        if committee is not None:  # the alias of committee mode (engine/server/committee.py)
            data.append({"id": served_name + cm.SUFFIX, **entry, "committee": {"roles": list(committee.roles)}})
        return {"object": "list", "data": data}

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
                "session_cache": sc.store.status() if sc.store is not None else None,
                "memory_gb": {"allocated": round(torch.cuda.memory_allocated() / 1e9, 2), "reserved": round(torch.cuda.memory_reserved() / 1e9, 2)}}

    return app


def main(argv=None):
    D = ServerConfig()  # the defaults of the flags
    ap = argparse.ArgumentParser(description="colin-inference-engine: an OpenAI- and Anthropic-compatible server")
    ap.add_argument("--model", default=D.model, help="HF repo id (in the local cache) or checkpoint directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, default=D.slots,
                    help="conversations whose KV stays cached; one request runs at a time (default: the model family's, 3 for the 27B)")
    ap.add_argument("--max-seq-len", type=int, default=D.max_seq_len,
                    help="tokens per slot (prompt + output; default: the model family's, 524288 for the 27B)")
    ap.add_argument("--spec", choices=("mtp", "none"), default=D.spec, help="speculative decoding (default: the model family's, mtp for the 27B)")
    ap.add_argument("--drafter-weights", default=D.drafter_weights,
                    help="MTP head weights: auto = the fine-tuned head of the prepared directory (tools/prepare.py) when it has one, "
                         "none = the checkpoint's, or a path. Drafts only affect speed, never outputs")
    ap.add_argument("--decode-weights", choices=("int", "checkpoint"), default=D.decode_weights,
                    help="int: decode the attention / GDN projections from the INT6 / INT5 copies of the prepared directory "
                         "(tools/prepare.py; ~9.5%% faster, perplexity within 0.25%%) when it has them; checkpoint: from the FP8 weights")
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="never reuse a prompt prefix (benchmarking raw prefill): no checkpoint ring, no conversation cache on disk")
    ap.add_argument("--session-cache-gb", type=float, default=D.session_cache_gb,
                    help="disk budget of the conversation cache (KV and GDN state of past conversations, restored after an eviction "
                         "or a restart); 0: off")
    ap.add_argument("--no-vision", action="store_true", help="no image input: images become a text note, and the vision tower is not loaded")
    ap.add_argument("--ngram-cache-gb", type=float, default=D.ngram_cache_gb,
                    help="qwen4_exp: the host row cache of the n-gram table that the PLE layer reads from the disk (GB of 160-byte rows; "
                         "0: off). It counts in the memory check. The 27B has no n-gram table")
    ap.add_argument("--mem-cap-gb", type=float, default=95.0,
                    help="hard cap on this process's GPU memory (torch allocator; GB of 10^9 bytes). The 27B reserves 88.6 GB at "
                         "startup and 89.5 GB with the vision tower, Qwen3.8-Flash-Next 91.2 GB; the rest of the unified memory stays for "
                         "the host (docs/server.md)")
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                    help="default enable_thinking (auto: the template's default, on; for /v1/messages without a thinking field: off)")
    ap.add_argument("--reasoning-effort", choices=("auto", "low", "medium", "xhigh"), default="auto",
                    help="the reasoning effort of a thinking request that does not set one (auto: the template's default, xhigh)")
    ap.add_argument("--committee-size", type=int, default=D.committee_size, metavar="N",
                    help="members of a committee reply (model name <name>-committee: each member answers the request in a role, then a "
                         "reconcile pass writes the reply; docs/server.md). Default: 3, or the roles of --committee-roles. 0: the alias is off")
    ap.add_argument("--committee-roles", default=D.committee_roles, metavar="FILE",
                    help='a JSON file {"roles": [...], "reconcile": "..."}: the members\' roles and the reconcile instruction')
    ap.add_argument("--api-key", default=os.environ.get("COLINFER_API_KEY") or None,
                    help="require this key on the /v1/ endpoints (Authorization: Bearer, or x-api-key); default: $COLINFER_API_KEY, else no key")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)

    try:  # before the model loads: a bad roles file is cheap to fix now
        committee = cm.Committee.from_config(a.committee_size, a.committee_roles)
    except (OSError, ValueError) as e:
        raise SystemExit(f"--committee-roles: {e}") from None
    t0 = time.perf_counter()
    path = resolve(a.model)
    try:  # before the CUDA context and the slow imports: both read only config.json, so a refused start costs little
        check_served(path)  # a family the engine cannot serve yet (engine/model/family.py)
        fd = family_of(path).defaults  # the flags without a value take the family's defaults
        slots = fd.slots if a.slots is None else a.slots
        max_seq_len = fd.max_seq_len if a.max_seq_len is None else a.max_seq_len
        spec = a.spec or fd.spec
        check_memory(path, slots, max_seq_len, spec, a.ngram_cache_gb)
    except NotImplementedError as e:
        log(f"[server] {e}")
        raise SystemExit(2) from None
    except NotEnoughMemory as e:
        log(f"[server] {e}")
        raise SystemExit(NO_MEMORY_EXIT) from None
    total = torch.cuda.get_device_properties(0).total_memory
    mem_cap_bytes = int(min(a.mem_cap_gb * 1e9, total))
    torch.cuda.set_per_process_memory_fraction(mem_cap_bytes / total)
    log(f"[server] GPU memory cap {mem_cap_bytes / 1e9:.0f} GB of {total / 1e9:.0f} GB")

    from transformers import AutoTokenizer  # a slow import: after the memory cap is set, so a crash here is cheap
    tokenizer = AutoTokenizer.from_pretrained(path)
    # the values without a flag (the queue depth, the request time cap, the body size, the draft lengths, the image size, the
    # cache directory) are the defaults of ServerConfig; the checkpoint ring is the family's
    cfg = ServerConfig(model=a.model, slots=slots, max_seq_len=max_seq_len, checkpoints=0 if a.no_prefix_caching else fd.checkpoints,
                       spec=spec, drafter_weights=a.drafter_weights, decode_weights=a.decode_weights,
                       boundary_token=ChatFormat(tokenizer).im_start, selftest=not a.no_selftest, warmup=not a.no_warmup,
                       api_key=a.api_key, mem_cap_bytes=mem_cap_bytes, committee_size=a.committee_size, committee_roles=a.committee_roles,
                       session_cache_gb=0 if a.no_prefix_caching else a.session_cache_gb, vision=fd.vision and not a.no_vision,
                       ngram_cache_gb=a.ngram_cache_gb)
    hf = json.load(open(os.path.join(path, "config.json")))
    hf_config = hf.get("text_config", hf)
    gen = json.load(open(os.path.join(path, "generation_config.json"))) if os.path.exists(os.path.join(path, "generation_config.json")) else {}
    gen_defaults = {k: gen[k] for k in ("temperature", "top_k", "top_p") if k in gen}

    worker = Worker(cfg, log)
    worker.start()
    worker.ready.wait()
    if isinstance(worker.error, NotEnoughMemory):  # build_engine checks again, after the CUDA context: the same exit
        log(f"[server] {worker.error}")
        raise SystemExit(NO_MEMORY_EXIT)
    if worker.error is not None:
        raise SystemExit(f"engine failed to start: {worker.error!r}")
    thinking = {"auto": None, "on": True, "off": False}[a.thinking]
    effort = None if a.reasoning_effort == "auto" else a.reasoning_effort
    app = build_app(worker, tokenizer, a.model, gen_defaults, hf_config, thinking, effort, committee)
    log(f"[server] startup {time.perf_counter() - t0:.1f} s; listening on http://{a.host}:{a.port}/v1")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
