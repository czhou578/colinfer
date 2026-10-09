"""The HTTP layer of the server (engine/server/api.py) against a fake engine thread: request validation and the
responses of /v1/chat/completions, /v1/completions and /v1/messages. No GPU and no checkpoint: a character-level
tokenizer and a worker that answers each request with a scripted reply."""
import os
import re
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402

from engine.runtime.metrics import Metrics  # noqa: E402
from engine.server.api import build_app  # noqa: E402

SPECIAL = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4, "<|im_end|>": 5, "<|endoftext|>": 6, "<|im_start|>": 7}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")
CHAR0 = 100  # character c is token CHAR0 + ord(c)
MAX_LEN, MARGIN = 4096, 8


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

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=True, tokenize=False, enable_thinking=True, **kw):
        text = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs)
        return text + "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "")


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


def client(reply="", error=None, thinking=None):
    w = FakeWorker(reply, error)
    return TestClient(build_app(w, CharTokenizer(), "fake", {}, {}, thinking)), w


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
