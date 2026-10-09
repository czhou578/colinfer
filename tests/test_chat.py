"""Chat formatting / output parsing for the server (engine/server/chat.py), on the real Qwen3.8 tokenizer (CPU)."""
import json


from engine.server.chat import Detokenizer, OutputParser, TextParser, parse_tool_call


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
                                               "tags": {"type": "array", "items": {"type": "string"}}, "lat": {"type": "number"}},
                                               "required": ["city"]}}}]


def run(parser, ids):
    ev = []
    for t in ids:
        ev += parser.feed(t)
    ev += parser.finish()
    out = {"reasoning": "", "content": "", "tool_call": []}
    for k, x in ev:
        if k == "tool_call":
            out[k].append(x)
        else:
            out[k] += x
    return out


def enc(fmt, s):
    return fmt.tok.encode(s, add_special_tokens=False)


def test_detokenizer_matches_full_decode(fmt):
    text = "Héllo wörld — 日本語のテキスト 🚀🚀 and code: `x = {'a': 1}`\n\tdone"
    ids = enc(fmt, text)
    d = Detokenizer(fmt.tok)
    out = "".join(d.add(t) for t in ids) + d.flush()
    assert out == text


def test_render_thinking_and_tools(fmt):
    p_on = fmt.render([{"role": "user", "content": "hi"}])
    p_off = fmt.render([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert fmt.opens_in_reasoning(p_on) and not fmt.opens_in_reasoning(p_off)
    msgs = [{"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                   "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 2})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
    text = fmt.tok.decode(fmt.render(msgs, TOOLS))
    assert "<function=get_weather>" in text and "<parameter=days>\n2\n</parameter>" in text and "<tool_response>\nsunny" in text
    assert '"name": "get_weather"' in text


def test_reasoning_then_content(fmt):
    ids = enc(fmt, "Let me think.\nOK.\n") + [fmt.think_close] + enc(fmt, "\n\nThe answer is 4.") + [fmt.eos_ids[-1]]
    out = run(OutputParser(fmt, True, None), ids)
    assert out["reasoning"] == "Let me think.\nOK." and out["content"] == "The answer is 4." and not out["tool_call"]


def test_tool_call_parsing_and_types(fmt):
    body = ("I'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
            "<parameter=metric>\ntrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=lat>\n40.7\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nmulti\nline\n</parameter>\n</function>\n</tool_call>")
    ids = [fmt.think_close] + enc(fmt, body)
    out = run(OutputParser(fmt, True, TOOLS), ids)
    assert out["content"] == "I'll check."
    assert [c["name"] for c in out["tool_call"]] == ["get_weather", "get_weather"]
    a0 = json.loads(out["tool_call"][0]["arguments"])
    assert a0 == {"city": "New York", "days": 3, "metric": True, "tags": ["a", "b"], "lat": 40.7}
    assert json.loads(out["tool_call"][1]["arguments"]) == {"city": "multi\nline"}


def test_tool_call_without_tools_is_text(fmt):
    body = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
    out = run(OutputParser(fmt, False, None), enc(fmt, body))
    assert out["content"] == body and not out["tool_call"]


def test_malformed_tool_call_falls_back_to_text(fmt):
    out = run(OutputParser(fmt, False, TOOLS), enc(fmt, "<tool_call>\nnot a call\n</tool_call>"))
    assert not out["tool_call"] and "not a call" in out["content"]


def test_stop_strings_across_tokens(fmt):
    p = OutputParser(fmt, False, None, stops=["STOP HERE", "\n\n\n"])
    ids = enc(fmt, "alpha beta STOP HERE gamma")
    ev, stopped_at = [], None
    for i, t in enumerate(ids):
        ev += p.feed(t)
        if p.stopped:
            stopped_at = i
            break
    ev += p.finish()
    assert stopped_at is not None and "".join(x for _, x in ev) == "alpha beta"
    # a partial match that does not complete is released
    p = OutputParser(fmt, False, None, stops=["STOP HERE"])
    assert run(p, enc(fmt, "a STOP THERE b"))["content"] == "a STOP THERE b"


def test_text_parser_keeps_whitespace_and_stops(fmt):
    p = TextParser(fmt.tok, stops=["###"])
    ev = []
    for t in enc(fmt, "  leading\n\ntext ### tail"):
        ev += p.feed(t)
        if p.stopped:
            break
    ev += p.finish()
    assert "".join(x for _, x in ev) == "  leading\n\ntext "


def test_parse_tool_call_untyped_and_json():
    c = parse_tool_call("<function=f>\n<parameter=x>\n{\"k\": 1}\n</parameter>\n<parameter=y>\nhello\n</parameter>\n</function>", None)
    assert c["name"] == "f" and json.loads(c["arguments"]) == {"x": '{"k": 1}', "y": "hello"}
    c = parse_tool_call('{"name": "g", "arguments": {"a": 1}}', None)
    assert c["name"] == "g" and json.loads(c["arguments"]) == {"a": 1}
    assert parse_tool_call("garbage", None) is None


def test_tool_schemas_the_parser_does_not_expect():
    """Boolean JSON schemas (true: any value) and other odd shapes in the client's tools: untyped values, no exception
    (the parser runs on the engine thread)."""
    body = "<function=f>\n<parameter=x>\n[1, 2]\n</parameter>\n<parameter=y>\nhi\n</parameter>\n</function>"
    for params in ({"properties": {"x": True, "y": False}}, {"properties": [1]}, [1], None):
        c = parse_tool_call(body, [{"type": "function", "function": {"name": "f", "parameters": params}}, 7, {"function": "f"}])
        assert c["name"] == "f" and json.loads(c["arguments"])["y"] == "hi"
    c = parse_tool_call(body, [{"type": "function", "function": {"name": "f", "parameters": {"properties": {"x": True}}}}])
    assert json.loads(c["arguments"])["x"] == [1, 2]


def test_truncated_tool_call_keeps_complete_params(fmt):
    body = "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=days>\n4"
    out = run(OutputParser(fmt, False, TOOLS), enc(fmt, body))
    assert len(out["tool_call"]) == 1 and json.loads(out["tool_call"][0]["arguments"]) == {"city": "Rome"}
