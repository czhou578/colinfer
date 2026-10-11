"""Anthropic Messages API conversion (engine/server/anthropic.py), on the real Qwen3.8 tokenizer (CPU).

The requests have the shape of the requests of Claude Code 2.1.290: system blocks with cache_control, system messages
inside `messages`, thinking blocks sent back with their signature, tool_use / tool_result pairs.
"""
import json

import pytest

from engine.server import anthropic as anth
from engine.server.chat import OutputParser

BASH = {"name": "Bash", "description": "Run a shell command", "input_schema": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}
CACHE = {"type": "ephemeral"}


def claude_code_request(env_as_list=True):
    env = "# Environment\nPrimary working directory: /tmp/demo"
    return {
        "model": "qwen3.8-27b", "max_tokens": 32000, "stream": True, "tools": [BASH],
        "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}, "metadata": {"user_id": "x"},
        "system": [{"type": "text", "text": "You are an agent.", "cache_control": CACHE},
                   {"type": "text", "text": "Help with software tasks.", "cache_control": CACHE}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nBe brief.\n</system-reminder>"},
                                         {"type": "text", "text": "List the files here."}]},
            {"role": "system", "content": [{"type": "text", "text": env, "cache_control": CACHE}] if env_as_list else env},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "I should list files.", "signature": "colinfer"},
                                              {"type": "text", "text": "Listing."},
                                              {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls", "timeout": 5}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "a.py\nb.py", "is_error": False}]},
            {"role": "system", "content": [{"type": "text", "text": "<total_tokens>1000 tokens left</total_tokens>", "cache_control": CACHE}]},
        ]}


def render(fmt, body, default_thinking=None):
    return fmt.render(anth.to_messages(body), anth.to_tools(body), **anth.template_kwargs(body, default_thinking))


def test_claude_code_request_renders(fmt):
    body = claude_code_request()
    msgs = anth.to_messages(body)
    assert [m["role"] for m in msgs] == ["system", "user", "user", "assistant", "tool", "user"]
    assert msgs[0]["content"] == "You are an agent.\n\nHelp with software tasks."
    assert msgs[2]["content"].startswith("<system-reminder>\n# Environment")  # a system message inside `messages`
    assert msgs[3]["reasoning_content"] == "I should list files."
    assert msgs[3]["tool_calls"][0]["function"]["arguments"] == {"command": "ls", "timeout": 5}
    prompt = render(fmt, body)
    text = fmt.tok.decode(prompt)
    assert text.index("# Tools") < text.index("You are an agent.") < text.index("List the files here.") < text.index("# Environment")
    assert "<think>\nI should list files.\n</think>\n\nListing.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>" in text
    assert "<tool_response>\na.py\nb.py\n</tool_response>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n") and fmt.opens_in_reasoning(prompt)


def test_content_forms_render_the_same(fmt):
    # Claude Code sends the same system message as a block list first and as a string later: one prompt for both
    assert render(fmt, claude_code_request(True)) == render(fmt, claude_code_request(False))


def test_thinking_switch(fmt):
    body = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    assert anth.template_kwargs(body, None) == {"enable_thinking": False}  # no `thinking`: off, as on the Anthropic API
    assert anth.template_kwargs(body, True) == {"enable_thinking": True}   # unless the server runs with --thinking on
    assert anth.template_kwargs({**body, "thinking": {"type": "disabled"}}, True) == {"enable_thinking": False}
    on = {**body, "thinking": {"type": "enabled", "budget_tokens": 2048}, "output_config": {"effort": "low"}}
    assert anth.template_kwargs(on, None) == {"enable_thinking": True, "reasoning_effort": "low"}
    assert not fmt.opens_in_reasoning(render(fmt, body)) and fmt.opens_in_reasoning(render(fmt, on))


def test_tools_and_errors():
    web = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    assert anth.to_tools({"tools": [BASH, web]}) == [{"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command", "parameters": BASH["input_schema"]}}]
    with pytest.raises(ValueError, match="server tools"):
        anth.to_tools({"tools": [web]})
    with pytest.raises(ValueError, match="prefill"):
        anth.to_messages({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure,"}]})
    with pytest.raises(ValueError):
        anth.to_messages({"messages": []})
    assert anth.drops_tool_calls({"tool_choice": {"type": "none"}}) and not anth.drops_tool_calls({"tool_choice": {"type": "auto"}})


def test_tool_result_errors_are_marked():
    ok = {"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}
    bad = {"type": "tool_result", "tool_use_id": "t2", "content": "exit code 1\nno such file", "is_error": True}
    empty = {"type": "tool_result", "tool_use_id": "t3", "content": [], "is_error": True}
    msgs = anth.to_messages({"messages": [{"role": "user", "content": [ok, bad, empty]}]})
    assert [m["content"] for m in msgs] == ["a.py", f"{anth.TOOL_ERROR}\nexit code 1\nno such file", anth.TOOL_ERROR]
    assert all(m["role"] == "tool" for m in msgs)


def test_images_become_a_note():
    msgs = anth.to_messages({"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}, {"type": "text", "text": "What is this?"}]}]})
    assert msgs == [{"role": "user", "content": "[image omitted: this server reads text only]\n\nWhat is this?"}]


def events_of(chunks):
    out = []
    for c in chunks:
        head, data = c.split("\n", 1)
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_blocks_from_parser_output(fmt):
    tools = anth.to_tools({"tools": [BASH]})
    ids = (fmt.tok.encode("Check the files.", add_special_tokens=False) + [fmt.think_close]
           + fmt.tok.encode("\n\nI'll run ls.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
                            "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False))
    parser, blocks, chunks = OutputParser(fmt, True, tools), anth.Blocks(), []
    for t in ids:
        chunks += blocks.add(parser.feed(t))
    chunks += blocks.add(parser.finish()) + blocks.close()
    assert [b["type"] for b in blocks.content] == ["thinking", "text", "tool_use"]
    assert blocks.content[0] == {"type": "thinking", "thinking": "Check the files.", "signature": anth.SIGNATURE}
    assert blocks.content[1] == {"type": "text", "text": "I'll run ls."}
    assert blocks.content[2]["name"] == "Bash" and blocks.content[2]["input"] == {"command": "ls -la", "timeout": 30}
    assert blocks.content[2]["id"].startswith("toolu_")
    ev = events_of(chunks)
    kinds = [(e, d.get("index"), (d.get("delta") or {}).get("type")) for e, d in ev]
    assert kinds[0] == ("content_block_start", 0, None)
    assert ("content_block_delta", 0, "signature_delta") in kinds  # the thinking block gets its signature before it stops
    assert kinds.index(("content_block_delta", 0, "signature_delta")) < kinds.index(("content_block_stop", 0, None))
    assert kinds.index(("content_block_stop", 0, None)) < kinds.index(("content_block_start", 1, None))
    assert kinds[-3:] == [("content_block_start", 2, None), ("content_block_delta", 2, "input_json_delta"), ("content_block_stop", 2, None)]
    assert json.loads(ev[-2][1]["delta"]["partial_json"]) == {"command": "ls -la", "timeout": 30}
    assert anth.stop_reason("stop", blocks.n_tool_calls, parser.stop_match) == ("tool_use", None)


def test_tool_choice_none_drops_calls(fmt):
    tools = anth.to_tools({"tools": [BASH]})
    ids = fmt.tok.encode("<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False)
    parser, blocks = OutputParser(fmt, False, tools), anth.Blocks(drop_tool_calls=True)
    for t in ids:
        blocks.add(parser.feed(t))
    blocks.add(parser.finish())
    assert blocks.content == [] and blocks.n_tool_calls == 0


def test_api_key_middleware():
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from engine.server.api import ApiKey
    app = FastAPI()
    app.add_middleware(ApiKey, key="sk-test")
    app.get("/v1/models")(lambda: {"ok": True})
    app.get("/health")(lambda: {"status": "ok"})
    c = TestClient(app)
    assert c.get("/health").status_code == 200  # outside /v1/: open
    assert c.get("/v1/models").status_code == 401 and c.get("/v1/messages/count_tokens").status_code == 401
    assert c.get("/v1/models", headers={"authorization": "Bearer sk-wrong"}).status_code == 401
    assert c.get("/v1/models").json() == {"error": {"message": "invalid or missing API key", "type": "invalid_request_error", "code": 401}}
    assert c.get("/v1/messages/count_tokens").json() == {"type": "error", "error": {"type": "authentication_error",
                                                                                "message": "invalid or missing API key"}}
    assert c.get("/v1/models", headers={"authorization": "Bearer sk-test"}).status_code == 200
    assert c.get("/v1/models", headers={"x-api-key": "sk-test"}).status_code == 200


def test_stop_reasons(fmt):
    p = OutputParser(fmt, False, None, stops=["END"])
    for t in fmt.tok.encode("one two END three", add_special_tokens=False):
        p.feed(t)
        if p.stopped:
            break
    assert anth.stop_reason("stop", 0, p.stop_match) == ("stop_sequence", "END")
    assert anth.stop_reason("length", 1, None) == ("max_tokens", None)
    assert anth.stop_reason("stop", 0, None) == ("end_turn", None)
    assert anth.usage(100, 60, 7) == {"input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 60, "output_tokens": 7}
