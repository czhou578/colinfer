"""The HTTP layer of the server (engine/server/api.py) against a fake engine thread: request validation and the
responses of /v1/chat/completions, /v1/completions and /v1/messages. No GPU and no checkpoint: a character-level
tokenizer and a worker that answers each request with a scripted reply."""
import re
import time
import types

import jinja2
import pytest

from fastapi.testclient import TestClient

from engine.runtime.metrics import Metrics
from engine.server import api
from engine.server.api import build_app

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
        self.args = types.SimpleNamespace(max_seq_len=MAX_LEN, model="fake")
        self.sched = types.SimpleNamespace(margin=MARGIN)
        self.metrics = Metrics()
        self.reply, self.error = CharTokenizer().encode(reply), error
        self.submitted, self.aborted = [], []

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
        req.done, req.finish_reason, req.t_done = True, reason, time.perf_counter()
        req.hook.finish(req)

    def abort(self, req):
        self.aborted.append(req)


def client(reply="", error=None, thinking=None, tokenizer=None):
    """The app on a FakeWorker. A bug of the server gets its 500 reply (TestClient would raise it instead)."""
    w = FakeWorker(reply, error)
    app = build_app(w, tokenizer or CharTokenizer(), "fake", {}, {"vocab_size": VOCAB}, thinking)
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
    assert api._template_kwargs(body, None) == kw
    assert api._template_kwargs(body, True) == {"enable_thinking": True, **kw}


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
