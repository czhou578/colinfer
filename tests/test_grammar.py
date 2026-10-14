"""Constrained decoding (engine/runtime/grammar.py) on the checkpoint's tokenizer (CPU): JSON schemas, forced tool calls
in Qwen's XML format with typed values, the free reasoning before them, and the masks of the verify rows of a
speculative cycle."""
import json

import numpy as np
import pytest

from engine.runtime.grammar import Grammars, mask_words
from engine.server.chat import ChatFormat, parse_tool_call

V = 248320
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"],
          "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["C", "F"]},
    "opts": {"type": "object", "properties": {"hourly": {"type": "boolean"}}}}}}},
         {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


@pytest.fixture(scope="module")
def g():
    try:
        fmt = ChatFormat.from_checkpoint()
    except Exception:  # noqa: BLE001
        pytest.skip("needs the checkpoint's tokenizer")
    return Grammars(fmt, V)


def ids(g, *parts):
    """Token ids of text parts and control token ids."""
    out = []
    for p in parts:
        out += [p] if isinstance(p, int) else g.fmt.tok.encode(p, add_special_tokens=False)
    return out


def run(c, toks):
    """(tokens the grammar accepts in a row, whether it is then complete: only the end of the reply allowed)."""
    c.m.reset()
    n = c.m.validate_tokens(toks)
    if n == len(toks):
        c.advance(toks)
    return n, c.m.is_accepting()


def allowed(mask_row, t):
    return bool((mask_row[t // 32] >> (t % 32)) & 1)


def test_json_schema(g):
    c = g.json(SCHEMA, think=False)
    good = ids(g, '{"name": "Bo", "age": 4}')
    assert run(c, good) == (len(good), True)
    for bad in ('{"name": "Bo", "age": "4"}', '{"name": "Bo", "age": 4, "x": 1}', '```json\n{"name"', "Sure! {"):
        t = ids(g, bad)
        assert run(c, t)[0] < len(t), bad
    t = ids(g, '{"name": "Bo"')
    assert run(c, t) == (len(t), False)  # a prefix: valid, not complete
    m = np.zeros((1, mask_words(V)), np.int32)
    c.masks([], m)
    assert all(allowed(m[0], e) for e in g.fmt.eos_ids) is False  # the reply cannot end before the object closes
    assert run(g.json(None, think=False), ids(g, '{"anything": [1, 2]}'))[1]


def test_reasoning_comes_first(g):
    c = g.json(SCHEMA, think=True)
    t = ids(g, "The user wants a person.\nAge 4 fits.", g.fmt.think_close, "\n\n", '{"name": "Bo", "age": 4}')
    assert run(c, t) == (len(t), True)
    t = ids(g, '{"name": "Bo", "age": 4}')
    assert run(c, t)[1] is False  # inside the reasoning, JSON is just text: the reply is not complete
    t = ids(g, "thinking", g.fmt.eos_ids[0])
    assert run(c, t)[0] == len(t) - 1  # the reply cannot end inside the reasoning


def test_forced_tool_calls(g):
    to, tc = g.fmt.tool_open, g.fmt.tool_close
    call = ("\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
            "<parameter=unit>\nF\n</parameter>\n<parameter=opts>\n{\"hourly\": true}\n</parameter>\n</function>\n")
    c = g.tools(TOOLS)
    t = ids(g, "\n\n", to, call, tc)
    assert run(c, t) == (len(t), True)
    body = g.fmt.tok.decode(ids(g, call))
    assert json.loads(parse_tool_call(body, TOOLS)["arguments"]) == {"city": "New York", "days": 3, "unit": "F", "opts": {"hourly": True}}
    t = ids(g, to, call, tc, "\n", to, "\n<function=search>\n<parameter=q>\nrain\n</parameter>\n</function>\n", tc)
    assert run(c, t) == (len(t), True)  # two calls
    assert run(g.tools(TOOLS, single=True), t)[0] < len(t)  # one call only
    for bad in ("\n<function=get_time>\n", "\n<function=get_weather>\n<parameter=days>\nthree\n",
                "\n<function=get_weather>\n<parameter=unit>\nK\n", "\n<function=get_weather>\n<parameter=town>\n"):
        t = ids(g, to, bad)
        assert run(c, t)[0] < len(t), bad
    t = ids(g, "Let me check the weather.")
    assert run(c, t)[0] < len(t)  # no text instead of the call
    n = g.tools(TOOLS, names=["search"])
    t = ids(g, to, call, tc)
    assert run(n, t)[0] < len(t)  # a named tool: only that one
    with pytest.raises(ValueError):
        g.tools(TOOLS, names=["nope"])


def test_masks_of_the_verify_rows(g):
    c = g.json(SCHEMA, think=False)
    good = ids(g, '{"name": "Bo", "age": 42}')
    W = mask_words(V)
    m = np.full((6, W), -1, np.int32)
    c.masks(good[:5], m)
    assert all(allowed(m[i], good[i]) for i in range(6))  # row i is the mask after the drafts before it
    assert not allowed(m[0], ids(g, "x")[0])
    bad = good[:2] + ids(g, "]") + good[3:5]
    m2 = np.full((6, W), 0, np.int32)
    c.masks(bad, m2)
    assert not allowed(m2[2], bad[2]) and (m2[3:] == 0).all()  # the refused draft: its row excludes it, later rows untouched
    assert run(c, good)[1]  # masks() left the matcher where it was


class LowHalf:
    """A stub constraint for the stub model of tests/test_scheduler.py: tokens below V / 2 only."""

    def __init__(self):
        self.seen = []

    def masks(self, drafts, out):
        out[:len(drafts) + 1] = 0
        out[:len(drafts) + 1, 0] = -1  # word 0: tokens 0 .. 31 of the stub's 64

    def advance(self, toks):
        self.seen += toks
        return all(t < 32 for t in toks)


def test_the_scheduler_decodes_within_the_mask(monkeypatch):
    """The stub's next token gets logit 10, every other 0: outside the mask, the argmax is the lowest allowed token."""
    import random

    import torch

    import engine.runtime.scheduler as S
    from tests.test_scheduler import BND, L, P, StubModel, prompt, stub_prefill
    from tests.test_scheduler import V as SV
    from tests.test_scheduler import reference as free
    if not torch.cuda.is_available():
        pytest.skip("needs the GPU")
    monkeypatch.setattr(S, "prefill", stub_prefill)

    def reference(p, n):
        h, s, out = 0, 0, []
        for t in p:
            h, s = (h * 31 + t) % P, s + t
        while len(out) < n:
            nxt = (h + s) % SV
            out.append(nxt if nxt < 32 else 0)
            h, s = (h * 31 + out[-1]) % P, s + out[-1]
        return out
    sched, rng = S.Scheduler(StubModel(), n_slots=2, max_seq_len=L, n_checkpoints=4, prefill_chunk=64, boundary_token=BND), random.Random(0)
    p = prompt(rng, 100)
    c = LowHalf()
    r = sched.run([S.Request(p, max_new_tokens=40, constraint=c)])[0]
    assert r.output == reference(p, 40) and c.seen == r.output and any(t >= 32 for t in free(p, 40))
    r2 = sched.run([S.Request(p, max_new_tokens=40)])[0]  # the next request in the slot decodes without the mask again
    assert r2.output == free(p, 40)
