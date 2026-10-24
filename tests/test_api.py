"""The HTTP layer of the server (engine/server/api.py) against a fake engine thread: request validation and the
responses of /v1/chat/completions, /v1/completions and /v1/messages. No GPU and no checkpoint: a character-level
tokenizer and a worker that answers each request with a scripted reply."""
import asyncio
import gc
import json
import re
import time
import types

import jinja2
import pytest
from fastapi.testclient import TestClient

from engine.runtime.metrics import Metrics
from engine.server import anthropic as anth
from engine.server import openai as oai
from engine.server.api import BadRequest, build_app
from engine.server.chat import template_kwargs
from engine.server.worker import ServerConfig

SPECIAL = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4, "<|im_end|>": 5, "<|endoftext|>": 6, "<|im_start|>": 7}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")
CHAR0 = 100  # character c is token CHAR0 + ord(c)
MAX_LEN, MARGIN, VOCAB = 4096, 8, 1 << 16


class CharTokenizer:
    """One token per character, plus the special tokens of the chat format. The template: <|im_start|>role, the content,
    <|im_end|>; the generation prompt opens a <think> block unless enable_thinking is False."""

    def convert_tokens_to_ids(self, t):
        return SPECIAL[t]

    def encode(self, text, add_special_tokens=False):
        return [SPECIAL[p] if p in SPECIAL else CHAR0 + ord(c) for p in _SPLIT.split(text) if p for c in ([p] if p in SPECIAL else p)]

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in SPECIAL.items()}
        return "".join(chr(t - CHAR0) if t >= CHAR0 else ("" if skip_special_tokens else inv[t]) for t in ids)

    def apply_chat_template(self, conversation, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        for m in conversation:
            if m["role"] not in ("system", "user", "assistant", "tool"):  # as the checkpoint's template does
                raise jinja2.TemplateError("Unexpected message role.")
        text = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in conversation)
        return text + "<|im_start|>assistant\n" + ("<think>\n" if kw.get("enable_thinking", True) else "")


class FakeWorker:
    """The engine thread's interface to the HTTP handlers (submit / abort). submit() plays `reply` (text with special
    tokens) through the request's hook at once, as the scheduler would: stop tokens, max_new_tokens and stop strings end
    it. error=(message, code) fails the request instead."""

    def __init__(self, reply="", error=None):
        self.cfg = ServerConfig(model="fake", max_seq_len=MAX_LEN, max_request_seconds=0.0)
        self.waiting = 0  # what queued() reports
        self.sched = types.SimpleNamespace(margin=MARGIN)
        self.metrics = Metrics()
        self.reply, self.error = CharTokenizer().encode(reply), error
        self.submitted, self.aborted = [], []
        self.spec = None  # (cycles, drafted, accepted) to give each request, as the speculative scheduler would

    def _spec(self, req):
        if self.spec is not None:
            req.cycles, req.drafted, req.accepted = self.spec

    def submit(self, req):
        self.submitted.append(req)
        if self.error is not None:
            return req.hook.error(*self.error)
        req.t_submit = req.t_admit = req.t_first = time.perf_counter()
        reason = "length"
        for t in self.reply:
            req.output.append(t)
            if req.hook.feed(t, (-0.25, [(t, -0.25)]) if req.logprobs is not None else None):
                reason = "stop"
                break
            if t in req.eos_ids:
                reason = "stop"
                break
            if len(req.output) >= req.max_new_tokens:
                break
        self._spec(req)
        req.done, req.finish_reason, req.t_done = True, reason, time.perf_counter()
        req.hook.finish(req)

    def abort(self, req):
        self.aborted.append(req)

    def queued(self):
        return self.waiting


def client(reply="", error=None, thinking=None, tokenizer=None, effort=None):
    """The app on a FakeWorker. A bug of the server gets its 500 reply (TestClient would raise it instead)."""
    w = FakeWorker(reply, error)
    app = build_app(w, tokenizer or CharTokenizer(), "fake", {}, {"vocab_size": VOCAB}, thinking, effort)
    return TestClient(app, raise_server_exceptions=False), w


CHAT = "/v1/chat/completions"
USER = [{"role": "user", "content": "hi"}]


def test_max_tokens_zero_is_rejected_under_either_name():
    c, w = client("ok<|im_end|>")
    for extra in ({"max_tokens": 0}, {"max_completion_tokens": 0}, {"max_completion_tokens": 0, "max_tokens": 7}, {"max_tokens": -3}):
        assert c.post("/v1/completions", json={"prompt": "hi", **extra}).status_code == 400, extra
        assert c.post(CHAT, json={"messages": USER, **extra}).status_code == 400, extra
    assert not w.submitted


def test_max_tokens_limits():
    c, w = client("ok<|im_end|>")
    c.post("/v1/completions", json={"prompt": "hi", "max_completion_tokens": 5, "max_tokens": 7})
    c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 7})
    c.post("/v1/completions", json={"prompt": "hi"})  # no limit: the room left in the slot
    c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 10**9})
    room = MAX_LEN - MARGIN - 2 + 1
    assert [r.max_new_tokens for r in w.submitted] == [5, 7, room, room]


def sse_data(r):
    """The JSON events of an OpenAI stream. A stream ends with [DONE], or with an error event."""
    import json
    lines = [x.removeprefix("data: ") for x in r.text.split("\n\n") if x]
    events = [json.loads(x) for x in lines if x != "[DONE]"]
    assert (lines[-1] == "[DONE]") != ("error" in events[-1]) and lines.count("[DONE]") <= 1
    return events


def anth_events(r):
    import json
    out = []
    for x in r.text.split("\n\n"):
        if x:
            head, data = x.split("\n", 1)
            out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


THOUGHT = "Let me add.</think>\n\nThe answer is 4.<|im_end|>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
CALL = "</think>I'll check.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>"


@pytest.mark.parametrize("reply,tools", [(THOUGHT, None), (CALL, TOOLS)])
def test_chat_stream_matches_reply(reply, tools):
    c, _ = client(reply)
    body = {"messages": USER, "tools": tools, "logprobs": True, "top_logprobs": 1}
    one = c.post(CHAT, json=body).json()["choices"][0]
    chunks = sse_data(c.post(CHAT, json={**body, "stream": True, "stream_options": {"include_usage": True}}))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    deltas = [ch["choices"][0]["delta"] for ch in chunks if ch["choices"]]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == one["message"].get("reasoning_content", "")
    assert "".join(d.get("content", "") for d in deltas) == (one["message"]["content"] or "")
    calls = [t for d in deltas for t in d.get("tool_calls", [])]
    assert [t.pop("index") for t in calls] == list(range(len(calls)))
    no_id = lambda cs: [{k: v for k, v in t.items() if k != "id"} for t in cs]  # noqa: E731  (ids are random per request)
    assert no_id(calls) == no_id(one["message"].get("tool_calls", []))
    assert chunks[-2]["choices"][0]["finish_reason"] == one["finish_reason"]
    assert chunks[-1]["usage"]["completion_tokens"] == len(CharTokenizer().encode(reply))
    lps = [e for ch in chunks if ch["choices"] and ch["choices"][0].get("logprobs") for e in ch["choices"][0]["logprobs"]["content"]]
    assert len(lps) == len(one["logprobs"]["content"]) > 0
    if tools:
        assert one["finish_reason"] == "tool_calls" and one["message"]["content"] == "I'll check."
        assert one["message"]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    else:
        assert (one["message"]["reasoning_content"], one["message"]["content"], one["finish_reason"]) == ("Let me add.", "The answer is 4.", "stop")


def test_completions_stream_matches_reply():
    c, _ = client("one two three four")
    one = c.post("/v1/completions", json={"prompt": "count:", "max_tokens": 9, "logprobs": 0}).json()["choices"][0]
    chunks = sse_data(c.post("/v1/completions", json={"prompt": "count:", "max_tokens": 9, "logprobs": 0, "stream": True}))
    assert one["text"] == "".join(ch["choices"][0]["text"] for ch in chunks) == "one two t"
    assert one["finish_reason"] == chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert one["logprobs"]["tokens"] == [t for ch in chunks if ch["choices"][0]["logprobs"] for t in ch["choices"][0]["logprobs"]["tokens"]]


def test_messages_stream_matches_reply():
    c, _ = client(CALL)
    body = {"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "weather?"}], "thinking": {"type": "enabled"},
            "tools": [{"name": "get_weather", "input_schema": TOOLS[0]["function"]["parameters"]}]}
    one = c.post("/v1/messages", json=body).json()
    ev = anth_events(c.post("/v1/messages", json={**body, "stream": True}))
    assert [e for e, _ in ev][0] == "message_start" and [e for e, _ in ev][-2:] == ["message_delta", "message_stop"]
    text = "".join(d["delta"].get("text", "") for e, d in ev if e == "content_block_delta")
    assert text == "".join(b.get("text", "") for b in one["content"]) == "I'll check."
    assert [b["type"] for b in one["content"]] == ["text", "tool_use"] and one["content"][1]["input"] == {"city": "Paris"}
    assert one["stop_reason"] == ev[-2][1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("code,kind,anth_kind", [(500, "server_error", "api_error"), (400, "invalid_request_error", "invalid_request_error")])
def test_engine_errors_have_one_format(code, kind, anth_kind):
    c, _ = client(error=("engine says no", code))
    for path, body in ((CHAT, {"messages": USER}), ("/v1/completions", {"prompt": "hi"})):
        r = c.post(path, json=body)
        assert r.status_code == code and r.json()["error"] == {"message": "engine says no", "type": kind, "code": code}
        err = [e for e in sse_data(c.post(path, json={**body, "stream": True})) if "error" in e]
        assert err == [{"error": {"message": "engine says no", "type": kind, "code": code}}]
    body = {"max_tokens": 10, "messages": USER}
    r = c.post("/v1/messages", json=body)
    assert r.status_code == code and r.json()["error"]["type"] == anth_kind
    assert anth_events(c.post("/v1/messages", json={**body, "stream": True}))[-1] == ("error", _anth(anth_kind))


def _anth(kind):
    return {"type": "error", "error": {"type": kind, "message": "engine says no"}}


def test_empty_prompt_is_rejected_before_the_engine():
    c, w = client("x")
    for stream in (False, True):
        r = c.post("/v1/completions", json={"prompt": "", "stream": stream})
        assert r.status_code == 400 and "empty" in r.json()["error"]["message"]
    assert not w.submitted


def test_stop_must_be_strings():
    c, w = client("ok<|im_end|>")
    for stop in ([1], 5, ["a", None]):  # a non-string reached the parser on the engine thread, which exits on any error
        assert c.post(CHAT, json={"messages": USER, "stop": stop}).status_code == 400, stop
        assert c.post("/v1/completions", json={"prompt": "hi", "stop": stop}).status_code == 400, stop
    assert not w.submitted
    assert c.post(CHAT, json={"messages": USER, "stop": "x"}).status_code == 200
    assert c.post("/v1/completions", json={"prompt": "hi", "stop": ["a", "b"]}).status_code == 200


def test_output_failure_fails_only_its_request(monkeypatch):
    """Stream.feed runs on the engine thread: a parser exception must end that request with a 500 (and feed must tell
    the scheduler to stop it), not escape into the scheduler step."""
    import engine.server.chat as chat

    def broken(text, tools):
        raise RuntimeError("parser bug")
    monkeypatch.setattr(chat, "parse_tool_call", broken)
    c, w = client(CALL)
    body = {"messages": USER, "tools": TOOLS}
    r = c.post(CHAT, json=body)
    assert r.status_code == 500 and r.json()["error"]["message"] == "internal error while formatting the output"
    assert w.submitted[-1].finish_reason == "stop" and len(w.submitted[-1].output) < len(CharTokenizer().encode(CALL))
    assert sse_data(c.post(CHAT, json={**body, "stream": True}))[-1]["error"]["type"] == "server_error"
    assert c.post(CHAT, json={"messages": USER}).status_code == 200  # no tools: the parser never calls it


def test_template_errors_are_the_clients_and_other_failures_the_servers():
    c, _ = client("ok<|im_end|>")
    r = c.post(CHAT, json={"messages": [{"role": "narrator", "content": "x"}]})
    assert r.status_code == 400 and r.json()["error"]["message"] == "chat template: Unexpected message role."

    class Broken(CharTokenizer):  # a bug in rendering (here: of the tokenizer) used to come back as a 400
        def apply_chat_template(self, conversation, **kw):
            raise AttributeError("'NoneType' object has no attribute 'strip'")
    c, w = client("ok", tokenizer=Broken())
    r = c.post(CHAT, json={"messages": USER})
    assert r.status_code == 500 and r.json()["error"]["type"] == "server_error" and "AttributeError" in r.json()["error"]["message"]
    r = c.post("/v1/messages", json={"max_tokens": 5, "messages": USER})
    assert r.status_code == 500 and r.json()["error"]["type"] == "api_error"
    assert not w.submitted


@pytest.mark.parametrize("body,kw", [
    ({}, {}),
    ({"reasoning_effort": "none"}, {"enable_thinking": False}),
    ({"reasoning_effort": "minimal"}, {"reasoning_effort": "low"}),
    ({"reasoning_effort": "high"}, {"reasoning_effort": "xhigh"}),
    ({"reasoning_effort": "other"}, {"reasoning_effort": "xhigh"}),  # a top-level name it does not know: the default
    ({"enable_thinking": False, "reasoning_effort": "low"}, {"enable_thinking": False, "reasoning_effort": "low"}),
    # Hermes Agent's auxiliary calls, configured for DeepSeek
    ({"chat_template_kwargs": {"thinking": False, "reasoning_effort": "none"}}, {"thinking": False, "enable_thinking": False}),
    ({"chat_template_kwargs": {"thinking": True, "enable_thinking": False}}, {"thinking": True, "enable_thinking": False}),
    ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "none"}}, {"enable_thinking": True}),
    ({"chat_template_kwargs": {"reasoning_effort": "max"}, "reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}),
    ({"chat_template_kwargs": {"reasoning_effort": "other"}}, {"reasoning_effort": "other"}),  # the template checks it
])
def test_template_kwargs_map_other_apis_names(body, kw):
    assert template_kwargs(body, None) == kw
    assert template_kwargs(body, True) == {"enable_thinking": True, **kw}


@pytest.mark.parametrize("body,effort", [
    ({}, "medium"),                                          # no effort named: the server's --reasoning-effort
    ({"reasoning_effort": "low"}, "low"),                    # a request's own level stays
    ({"reasoning_effort": "high"}, "xhigh"),
    ({"chat_template_kwargs": {"reasoning_effort": "low"}}, "low"),
    ({"reasoning_effort": "none"}, None),                    # thinking off: no effort
    ({"enable_thinking": False}, None),
])
def test_server_reasoning_effort_fills_in(body, effort):
    assert template_kwargs(body, None, "medium").get("reasoning_effort") == effort


def test_messages_reasoning_effort_default():
    from engine.server import anthropic as anth
    on = {"thinking": {"type": "enabled"}}
    assert anth.template_kwargs(on, None, "medium") == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert anth.template_kwargs({**on, "output_config": {"effort": "low"}}, None, "medium")["reasoning_effort"] == "low"
    assert anth.template_kwargs({**on, "output_config": {"effort": "high"}}, None, "medium") == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert anth.template_kwargs({}, None, "medium") == {"enable_thinking": False}
    assert anth.template_kwargs(on, None) == {"enable_thinking": True}  # auto: the template's default


def test_server_reasoning_effort_reaches_the_template():
    seen = []

    class Rec(CharTokenizer):
        def apply_chat_template(self, conversation, **kw):
            seen.append(kw.get("reasoning_effort"))
            return super().apply_chat_template(conversation, **kw)
    c, _ = client("ok<|im_end|>", tokenizer=Rec(), effort="medium")
    assert c.post(CHAT, json={"messages": USER}).status_code == 200
    assert c.post(CHAT, json={"messages": USER, "reasoning_effort": "low"}).status_code == 200
    assert c.post("/v1/messages", json={"max_tokens": 5, "messages": USER, "thinking": {"type": "enabled"}}).status_code == 200
    assert seen == ["medium", "low", "medium"]


def test_deepseek_style_thinking_off_renders_without_thinking():
    class Effort(CharTokenizer):  # checks reasoning_effort as the Qwen3.8 template does
        def apply_chat_template(self, conversation, **kw):
            effort = kw.get("reasoning_effort", "xhigh")
            if kw.get("enable_thinking", True) and effort not in ("xhigh", "medium", "low"):
                raise jinja2.TemplateError(f"Unexpected reasoning effort {effort}.")
            return super().apply_chat_template(conversation, **kw)
    c, w = client("ok<|im_end|>", tokenizer=Effort())
    r = c.post(CHAT, json={"messages": USER, "chat_template_kwargs": {"thinking": False, "reasoning_effort": "none"}})
    assert r.status_code == 200 and SPECIAL["<think>"] not in w.submitted[-1].prompt
    assert c.post(CHAT, json={"messages": USER, "reasoning_effort": "medium"}).status_code == 200
    assert SPECIAL["<think>"] in w.submitted[-1].prompt
    r = c.post(CHAT, json={"messages": USER, "chat_template_kwargs": {"reasoning_effort": "other"}})
    assert r.status_code == 400 and "Unexpected reasoning effort" in r.json()["error"]["message"]


@pytest.mark.parametrize("path,body", [
    (CHAT, [1, 2]),
    (CHAT, {"messages": ["hi"]}),
    (CHAT, {"messages": [{"role": 3, "content": "hi"}]}),
    (CHAT, {"messages": [{"role": "user", "content": 5}]}),
    (CHAT, {"messages": [{"role": "user", "content": [1]}]}),
    (CHAT, {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"function": "f"}]}, *USER]}),
    (CHAT, {"messages": [{"role": "assistant", "content": "", "reasoning_content": 7}, *USER]}),
    (CHAT, {"messages": USER, "tools": [1]}),
    (CHAT, {"messages": USER, "chat_template_kwargs": ["x"]}),
    (CHAT, {"messages": USER, "chat_template_kwargs": {"tokenize": True}}),
    (CHAT, {"messages": USER, "reasoning_effort": ["high"]}),
    (CHAT, {"messages": USER, "logprobs": True, "top_logprobs": "x"}),
    (CHAT, {"messages": USER, "temperature": "hot"}),
    ("/v1/completions", {"prompt": "hi", "stop_token_ids": 5}),
    ("/v1/completions", {"prompt": "hi", "seed": {"a": 1}}),
    ("/v1/messages", {"max_tokens": 5, "messages": [1]}),
    ("/v1/messages", {"max_tokens": 5, "messages": USER, "stop_sequences": ["a", 1]}),
    ("/v1/messages", {"max_tokens": 5, "messages": USER, "tools": [{"input_schema": {}}]}),
    ("/v1/messages", {"max_tokens": 5, "messages": USER, "tools": [{"name": "t", "input_schema": [1]}]}),
    ("/v1/messages", {"max_tokens": 5, "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "t", "input": [1]}]},
                                                     *USER]}),
    ("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": 3}]}),
])
def test_malformed_requests_are_400(path, body):
    c, w = client("ok<|im_end|>")
    r = c.post(path, json=body)
    assert r.status_code == 400, r.text
    assert not w.submitted


def test_invalid_json_is_400():
    c, _ = client()
    for path in (CHAT, "/v1/completions", "/v1/messages"):
        assert c.post(path, content=b"{not json", headers={"content-type": "application/json"}).status_code == 400


def test_a_body_past_the_size_limit_is_413_before_the_render():
    """The render and the tokenize hold host memory in proportion to the body, so a body past max_body_bytes is a 413 in
    the format of each API, with or without a Content-Length. The body never reaches the template or the engine."""
    c, w = client("ok<|im_end|>")
    w.cfg.max_body_bytes = 1 << 16
    big = {"messages": [{"role": "user", "content": "x" * (1 << 16)}]}
    r = c.post(CHAT, json=big)
    assert r.status_code == 413 and r.json()["error"]["code"] == 413
    r = c.post("/v1/messages", json={"max_tokens": 5, **big})
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"
    assert c.post("/v1/messages/count_tokens", json=big).status_code == 413
    assert not w.submitted
    w.cfg.max_body_bytes = 1 << 20
    assert c.post(CHAT, json=big).status_code == 400  # read in full: the prompt is longer than the slot
    assert c.post(CHAT, json={"messages": USER}).status_code == 200 and len(w.submitted) == 1


def asgi_post(app, path: str, chunks: list[bytes]) -> tuple[int, int]:
    """POSTs `chunks` to the ASGI app as one body without a Content-Length, one http.request message per chunk. Returns
    (the response status, the number of receive() calls). TestClient joins an iterator body into one chunk, so it cannot
    show where the server stops reading."""
    calls, status = [0], []

    async def receive():
        i = calls[0]
        calls[0] += 1
        if i < len(chunks):
            return {"type": "http.request", "body": chunks[i], "more_body": i + 1 < len(chunks)}
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}
    asyncio.run(app(scope, receive, send))
    return status[0], calls[0]


def test_the_server_stops_reading_a_body_without_a_length_at_the_size_limit():
    """A body without a Content-Length gets its 413 when its size passes max_body_bytes: the server does not receive
    the rest of the body."""
    c, w = client("ok<|im_end|>")
    w.cfg.max_body_bytes, size = 1 << 16, 1 << 10
    raw = json.dumps({"messages": [{"role": "user", "content": "x" * (1 << 18)}]}).encode()
    status, calls = asgi_post(c.app, CHAT, [raw[i:i + size] for i in range(0, len(raw), size)])
    assert status == 413 and calls == w.cfg.max_body_bytes // size + 1 and not w.submitted


def test_a_refused_body_is_freed_without_the_cyclic_gc():
    """The 413 of a body past the limit leaves no reference cycle: the chunks that the server read go away when the
    request ends, not at the next run of the cyclic GC."""
    c, w = client("ok<|im_end|>")
    w.cfg.max_body_bytes = 1 << 16
    raw = json.dumps({"messages": [{"role": "user", "content": "x" * (1 << 17)}]}).encode()
    gc.collect()
    gc.disable()
    try:
        status, _ = asgi_post(c.app, CHAT, [raw[i:i + 1024] for i in range(0, len(raw), 1024)])
        alive = [o for o in gc.get_objects() if type(o) is BadRequest]
    finally:
        gc.enable()
    assert status == 413 and not alive


def test_a_client_error_in_a_stream_keeps_its_status(monkeypatch):
    """A BadRequest that ends a stream before its first output gives the error event its own status, not a 400."""
    c, w = client("ok<|im_end|>")

    def refuse(req):
        raise BadRequest("too large", 413)
    monkeypatch.setattr(w, "submit", refuse)
    r = c.post(CHAT, json={"messages": USER, "stream": True})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
    assert events == [oai.error_body("too large", 413)]


class CountingTokenizer(CharTokenizer):
    """CharTokenizer that counts its encode calls."""

    def __init__(self):
        self.encodes = 0

    def encode(self, text, add_special_tokens=False):
        self.encodes += 1
        return super().encode(text, add_special_tokens)


def test_a_prompt_text_past_the_limit_is_refused_before_the_tokenize():
    """The tokenize holds host memory in proportion to the text, so a chat text or a completion prompt longer than 8
    bytes of UTF-8 per token of the slot is a 400 before it, in the words of the token check. The limit counts bytes: a
    rare 4-byte character is up to 4 tokens of a byte-level BPE. A text under the limit is tokenized, and then its tokens
    are checked."""
    tok = CountingTokenizer()
    c, w = client("ok<|im_end|>", tokenizer=tok)
    for path, body in ((CHAT, {"messages": [{"role": "user", "content": "x" * (8 * MAX_LEN + 1)}]}),
                       ("/v1/completions", {"prompt": "x" * (8 * MAX_LEN + 1)}),
                       ("/v1/messages", {"max_tokens": 5, "messages": [{"role": "user", "content": "x" * (8 * MAX_LEN + 1)}]}),
                       (CHAT, {"messages": [{"role": "user", "content": "\U00020000" * (2 * MAX_LEN + 1)}]})):  # 4 bytes each
        r = c.post(path, json=body)
        assert r.status_code == 400 and "bytes" in r.text, r.text
        assert tok.encodes == 0 and not w.submitted
    r = c.post(CHAT, json={"messages": [{"role": "user", "content": "x" * (8 * MAX_LEN - 100)}]})
    assert r.status_code == 400 and "tokens" in r.text and tok.encodes >= 1  # one encode per new message segment
    assert c.post(CHAT, json={"messages": USER}).status_code == 200 and len(w.submitted) == 1


def test_requests_past_the_admission_bound_are_refused_before_their_body_is_read():
    """At most max_queue + 1 requests are between the body read and the queue at once: their bodies, images and renders
    hold host memory before the queue check. The next one gets the busy reply of its API at once."""
    import asyncio
    import threading

    import httpx

    class SlowTokenizer(CharTokenizer):
        gate = threading.Event()

        def apply_chat_template(self, conversation, **kw):
            self.gate.wait(5)
            return super().apply_chat_template(conversation, **kw)
    w = FakeWorker("ok<|im_end|>", None)
    w.cfg.max_queue = 1  # the bound is 2
    app = build_app(w, SlowTokenizer(), "fake", {}, {"vocab_size": VOCAB}, None, None)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            slow = [asyncio.create_task(c.post(CHAT, json={"messages": USER})) for _ in range(2)]
            await asyncio.sleep(0.2)  # both wait in the render, on the gate
            refused = [await c.post(CHAT, json={"messages": USER}), await c.post("/v1/messages", json={"max_tokens": 5, "messages": USER})]
            SlowTokenizer.gate.set()
            return refused, await asyncio.gather(*slow)
    (r1, r2), done = asyncio.run(run())
    assert r1.status_code == 503 and r1.json()["error"]["type"] == "server_error"
    assert r2.status_code == 529 and r2.json()["error"]["type"] == "overloaded_error"
    assert [r.status_code for r in done] == [200, 200] and len(w.submitted) == 2


@pytest.mark.parametrize("body", [
    '{"prompt": [5, %d]}' % VOCAB,  # past the embedding: a device-side assert, which no later CUDA op survives
    '{"prompt": [-1, 5]}',
    '{"prompt": "hi", "stop_token_ids": [%d]}' % (1 << 40),  # overflowed the stop-id tensor on the engine thread
    '{"prompt": "hi", "stop_token_ids": [-2]}',
    '{"prompt": "hi", "min_p": 1.5}',
    '{"prompt": "hi", "max_tokens": Infinity}',  # Python's JSON parser accepts it; int() raises OverflowError
])
def test_values_that_reach_the_gpu_are_checked(body):
    c, w = client("ok")
    assert c.post("/v1/completions", content=body.encode(), headers={"content-type": "application/json"}).status_code == 400
    assert not w.submitted


def test_a_full_queue_is_refused_in_the_format_of_each_api():
    c, w = client("ok<|im_end|>")
    w.waiting = 8
    r = c.post(CHAT, json={"messages": USER})
    assert r.status_code == 503 and r.json()["error"]["type"] == "server_error"
    assert c.post("/v1/completions", json={"prompt": "hi", "stream": True}).status_code == 503
    r = c.post("/v1/messages", json={"max_tokens": 5, "messages": USER})
    assert r.status_code == 529 and r.json()["error"]["type"] == "overloaded_error"
    assert not w.submitted
    w.waiting = 7
    assert c.post(CHAT, json={"messages": USER}).status_code == 200 and len(w.submitted) == 1


def test_output_tokens_and_time_of_a_request_are_capped():
    c, w = client("ok<|im_end|>")
    w.cfg.max_output_tokens, w.cfg.max_request_seconds = 5, 7.5
    c.post(CHAT, json={"messages": USER, "max_tokens": 100})
    c.post(CHAT, json={"messages": USER, "max_tokens": 3})
    c.post("/v1/messages", json={"max_tokens": 50, "messages": USER})
    assert [r.max_new_tokens for r in w.submitted] == [5, 3, 5] and all(r.max_seconds == 7.5 for r in w.submitted)


def test_timings_count_each_phase():
    """Each phase has its tokens and its rate: the prefill over the prompt tokens no checkpoint served, the decode over
    the output tokens after the first, the acceptance over the drafts verified. A phase the request never reached (an
    end in the queue, or during the prefill) has no rate."""
    from engine.runtime.scheduler import Request
    r = Request(list(range(1000)), reused=600, t_submit=10.0, t_admit=10.5, t_first=10.7, t_done=14.7, output=list(range(201)),
                cycles=50, drafted=350, accepted=150)
    t = oai.timings(r)
    assert (t["queue_s"], t["ttft_s"], t["prefill_tokens"], t["prefill_s"], t["prefill_tok_s"]) == (0.5, 0.7, 400, 0.2, 2000.0)
    assert (t["decode_tokens"], t["decode_s"], t["decode_tok_s"]) == (200, 4.0, 50.0)
    assert (t["decode_cycles"], t["draft_tokens"], t["draft_accepted"], t["draft_accept_rate"]) == (50, 350, 150, 0.4286)
    queued = oai.timings(Request([1, 2, 3], t_submit=10.0, t_done=12.0))  # a timeout in the queue
    assert (queued["queue_s"], queued["ttft_s"], queued["prefill_s"], queued["prefill_tok_s"], queued["decode_tok_s"]) == (2.0, None, 0.0, None, None)
    cut = oai.timings(Request([1, 2, 3], t_submit=10.0, t_admit=10.0, t_done=12.0))  # an abort during the prefill
    assert (cut["queue_s"], cut["ttft_s"], cut["prefill_s"], cut["prefill_tok_s"], cut["draft_accept_rate"]) == (0.0, None, 2.0, None, None)
    one = oai.timings(Request([1, 2, 3], t_submit=1.0, t_admit=1.0, t_first=2.0, t_done=2.0, output=[7]))  # a stop token first
    assert (one["prefill_tok_s"], one["decode_tokens"], one["decode_tok_s"]) == (3.0, 0, None)


def test_the_log_line_shows_each_phase(monkeypatch):
    """One line per request: the phases with their tokens and rates, and the acceptance when the server drafted."""
    from engine.server import api
    lines = []
    monkeypatch.setattr(api, "log", lambda *a: lines.append(" ".join(map(str, a))))
    c, w = client("hello world<|im_end|>")
    w.spec = (5, 15, 9)
    assert c.post(CHAT, json={"messages": USER}).status_code == 200
    r = w.submitted[0]
    seg = lines[0].split(" | ")
    assert re.fullmatch(rf"\[chat -?\d+\] slot -1 prompt {len(r.prompt)} \(cached 0\) -> {len(r.output)} tok, stop; queue 0\.00s ttft 0\.00s", seg[0])
    assert seg[1] == f"prefill {len(r.prompt)} tok -"  # the fake's first token comes at its admission: no prefill time
    assert re.fullmatch(rf"decode {len(r.output) - 1} tok \d+\.\d tok/s", seg[2])
    assert seg[3] == f"accept 60% (9/15 drafts, {(len(r.output) - 1) / 5:.1f} tok/cycle)"
    w.spec, lines[:] = None, []  # a plain server: no accept segment
    c.post(CHAT, json={"messages": USER})
    assert len(lines) == 1 and len(lines[0].split(" | ")) == 3


def test_a_timeout_reads_as_a_cut_off():
    assert oai.finish_reason(types.SimpleNamespace(finish_reason="timeout")) == "length"
    assert anth.stop_reason("timeout", 0, None) == ("max_tokens", None)


class FakeGrammars:
    """engine/runtime/grammar.py Grammars, recording what each request asks for."""

    def __init__(self, fmt, vocab):
        pass

    def json(self, schema, think):
        return ("json", schema, think)

    def tools(self, tools, names=None, single=False, think=False):
        if names and names[0] not in [t["function"]["name"] for t in tools]:
            raise ValueError(f"tool_choice names {names} no tool of the request's tools")
        return ("tools", names, single, think)


TOOL = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def test_constraints_of_both_apis(monkeypatch):
    import engine.server.api as api
    monkeypatch.setattr(api, "Grammars", FakeGrammars)
    c, w = client("{}")
    schema = {"type": "object"}
    for body, want in (({"response_format": {"type": "json_object"}}, ("json", None, True)),
                       ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}, "enable_thinking": False},
                        ("json", schema, False)),
                       ({"tools": TOOL, "tool_choice": "required"}, ("tools", None, False, True)),
                       ({"tools": TOOL, "tool_choice": {"type": "function", "function": {"name": "f"}}, "parallel_tool_calls": False},
                        ("tools", ["f"], True, True)),
                       ({"response_format": {"type": "text"}}, None), ({"tools": TOOL, "tool_choice": "auto"}, None)):
        assert c.post(CHAT, json={"messages": USER, **body}).status_code == 200, body
        assert w.submitted[-1].constraint == want, body
    assert c.post("/v1/completions", json={"prompt": "hi", "response_format": {"type": "json_object"}}).status_code == 200
    assert w.submitted[-1].constraint == ("json", None, False)  # a raw prompt has no reasoning to skip
    for bad in ({"response_format": {"type": "json_schema"}}, {"response_format": {"type": "yaml"}}, {"tool_choice": "required"},
                {"tools": TOOL, "tool_choice": {"type": "function", "function": {"name": "g"}}}, {"tools": TOOL, "tool_choice": "sometimes"}):
        r = c.post(CHAT, json={"messages": USER, **bad})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error", bad
    anth_tool = [{"name": "f", "input_schema": {"type": "object", "properties": {}}}]
    m = {"max_tokens": 5, "messages": USER}
    for body, want in (({"tools": anth_tool, "tool_choice": {"type": "any"}}, ("tools", None, False, False)),
                       ({"tools": anth_tool, "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": True}},
                        ("tools", ["f"], True, False)),
                       ({"output_format": {"type": "json_schema", "schema": schema}}, ("json", schema, False)),
                       ({"tools": anth_tool, "tool_choice": {"type": "auto"}}, None)):
        assert c.post("/v1/messages", json={**m, **body}).status_code == 200, body
        assert w.submitted[-1].constraint == want, body
    for bad in ({"tool_choice": {"type": "any"}}, {"tools": anth_tool, "tool_choice": {"type": "tool"}}, {"output_format": {"type": "json"}}):
        assert c.post("/v1/messages", json={**m, **bad}).status_code == 400, bad


def test_main_wires_the_flags(monkeypatch):
    """main() builds the ServerConfig from the flags that remain. The values that have no flag (the queue depth, the
    request time cap, the draft lengths, the image size) are the defaults of ServerConfig. The slots, the slot length,
    the speculation and the checkpoint ring default to those of the model's family. The engine thread, the app and the
    HTTP server are stand-ins; the tokenizer is the checkpoint's."""
    import threading

    import engine.runtime.build as build
    import engine.server.api as api
    from engine.weights.loader import MODEL, resolve
    try:
        resolve(MODEL)
    except Exception:
        pytest.skip("checkpoint not in the local HF cache")
    if not api.torch.cuda.is_available():
        pytest.skip("main() reads the GPU's memory size")
    got = {}

    class FakeThread:
        def __init__(self, cfg, log):
            got["cfg"], self.ready, self.error = cfg, threading.Event(), None
            self.ready.set()

        def start(self):
            pass

    def fake_app(worker, tokenizer, name, gen_defaults, hf_config, thinking, effort, committee):
        got.update(name=name, thinking=thinking, effort=effort, committee=committee, gen=gen_defaults)
        return "app"
    import uvicorn
    monkeypatch.setattr(api, "Worker", FakeThread)
    monkeypatch.setattr(api, "build_app", fake_app)
    monkeypatch.setattr(api.torch.cuda, "set_per_process_memory_fraction", lambda f: None)
    monkeypatch.setattr(build, "meminfo_bytes", lambda key: 1 << 50)  # not the memory that is free now
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: got.update(served=True))
    api.main(["--no-prefix-caching", "--spec", "none", "--slots", "1", "--reasoning-effort", "medium", "--committee-size", "0",
              "--session-cache-gb", "3", "--no-vision", "--mem-cap-gb", "70", "--ngram-cache-gb", "0.5"])
    cfg = got["cfg"]
    assert cfg.ngram_cache_gb == 0.5
    assert (cfg.checkpoints, cfg.session_cache_gb, cfg.spec, cfg.slots, cfg.vision) == (0, 0, "none", 1, False)  # --no-prefix-caching wins
    assert cfg.mem_cap_bytes == 70 * 10**9 and cfg.committee_size == 0
    D = ServerConfig()
    assert (cfg.max_queue, cfg.max_output_tokens, cfg.max_request_seconds) == (D.max_queue, D.max_output_tokens, D.max_request_seconds)
    assert (cfg.k, cfg.suffix_drafts, cfg.image_max_pixels, cfg.session_dir, cfg.vision_preload) == (D.k, D.suffix_drafts, D.image_max_pixels,
                                                                                                        D.session_dir, D.vision_preload)
    assert got["name"] == cfg.model == D.model and got["effort"] == "medium" and got["thinking"] is None and got["committee"] is None
    assert got["served"] and "temperature" in got["gen"]
    api.main(["--session-cache-gb", "0", "--thinking", "off"])
    c = got["cfg"]  # the flags without a value: the defaults of the 27B's family (engine/model/family.py)
    assert (c.slots, c.max_seq_len, c.checkpoints, c.spec, c.vision) == (3, 524288, 32, "mtp", True)
    assert c.ngram_cache_gb == ServerConfig().ngram_cache_gb == 1.0
    assert got["thinking"] is False and got["committee"] is not None
    # the default cap: the engine's torch peak (89.5e9 with the vision tower) and a margin, not more of the host's memory
    assert got["cfg"].mem_cap_bytes == int(min(95e9, api.torch.cuda.get_device_properties(0).total_memory))


def test_the_readme_asks_for_the_memory_that_the_engine_takes_at_startup():
    """The free memory that README.md asks for covers the startup estimate of the memory check at the defaults (3 slots x
    524288 tokens, MTP) and its reserve for the growth after startup. The README figure is in GB of 10^9 bytes, as
    `--mem-cap-gb`."""
    import os

    import engine.runtime.build as build
    from engine.weights.loader import resolve
    D = ServerConfig()
    try:
        path = resolve(D.model)
    except Exception:
        pytest.skip("checkpoint not in the local HF cache")
    readme = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read()
    gb = int(re.search(r"About (\d+) GB of free unified memory", readme).group(1))
    assert build.startup_bytes(path, D.slots, D.max_seq_len, D.spec) + build.GROWTH_RESERVE_BYTES <= gb * 10**9


def test_main_refuses_to_start_without_the_memory_before_the_cuda_context(monkeypatch, capsys):
    """With less MemAvailable than the startup estimate, main() logs the figures and exits with NO_MEMORY_EXIT before it
    creates the CUDA context: an over-commit of the unified memory powers the machine off. systemd does not restart the
    server on that status, so a refused start does not repeat every 5 s."""
    import os

    import engine.runtime.build as build
    import engine.server.api as api
    from engine.weights.loader import MODEL, resolve
    try:
        resolve(MODEL)
    except Exception:
        pytest.skip("checkpoint not in the local HF cache")
    cuda = []

    def no_cuda(*a, **k):
        cuda.append(a)
        raise AssertionError("main() created the CUDA context")

    def no_worker(cfg, log):
        raise AssertionError("main() started the engine thread")
    monkeypatch.setattr(build, "meminfo_bytes", lambda key: 70 * 10**9)  # less than the defaults take
    monkeypatch.setattr(build, "process_rss_bytes", lambda: 0)  # the check adds what the process holds; the test wants a round figure
    monkeypatch.setattr(api.torch.cuda, "get_device_properties", no_cuda)
    monkeypatch.setattr(api, "Worker", no_worker)
    with pytest.raises(SystemExit) as e:
        api.main([])
    assert e.value.code == api.NO_MEMORY_EXIT and not cuda
    assert "70 GB is available" in capsys.readouterr().out
    unit = open(os.path.join(os.path.dirname(__file__), "..", "deploy", "colinfer.service")).read()
    assert f"RestartPreventExitStatus={api.NO_MEMORY_EXIT}\n" in unit
