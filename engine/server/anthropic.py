"""Anthropic Messages API for the server (POST /v1/messages, POST /v1/messages/count_tokens).

Claude Code and other Anthropic clients connect to the server directly through this module (docs/claude_code.md). A
request becomes the same message list that /v1/chat/completions renders with the chat template of the checkpoint:

* The top-level `system` becomes the first system message. The template accepts a system message only at the start.
  Claude Code also sends system messages inside `messages` (the environment, reminders). Each one becomes a user turn
  in <system-reminder> tags.
* `tool_use` blocks become assistant tool_calls, and `tool_result` blocks become tool messages; a result with
  `is_error` gets the TOOL_ERROR line in front, so the model sees that the call failed. `thinking` blocks become
  reasoning_content. The template renders the reasoning of earlier turns again, so the next prompt repeats the tokens
  that the model generated, and the server can reuse its prefix checkpoints.
* `thinking.type` sets enable_thinking: off for `disabled`, on for `enabled` and `adaptive`. A request without
  `thinking` runs without reasoning, as on the Anthropic API, unless the server runs with --thinking on.
  `output_config.effort` is the reasoning_effort of the chat endpoint: low / medium as they are, higher ones the
  template's default (engine/server/chat.py template_kwargs resolves both APIs).
* Image blocks (base64), also inside tool results, become image parts that the model sees (engine/server/images.py),
  when the server runs the vision tower; otherwise, and for document blocks, a short text note.
* Server tools (tools with a `type`, for example web_search_20250305) are left out. A request with only server tools
  gets an error.
* `tool_choice` none keeps the tools in the prompt (the prefix stays the same for the prefix cache) and drops the tool
  calls of the reply. `any` and a named tool force the reply to be tool calls (engine/runtime/grammar.py), one with
  `disable_parallel_tool_use`. `output_format` json_schema (also `output_config.format`) forces a JSON reply.
* The server ignores the fields that concern only the Anthropic service: cache_control, metadata,
  context_management, service_tier, and the thinking budget and display.

A reply holds the content blocks thinking, text and tool_use. A stream sends the Anthropic events: message_start,
content_block_start / _delta / _stop, message_delta and message_stop. The server also sends a ping while it has no
output, for example during a long prefill (engine/server/api.py PING_S). Thus the stream watchdog of the client does
not stop it.
"""
from __future__ import annotations

import json

from engine.server.chat import template_kwargs as _template_kwargs

SIGNATURE = "colinfer"  # thinking blocks carry a signature; clients send it back unchanged, and the server ignores it
TOOL_ERROR = "[tool error]"  # the first line of a tool result whose block has is_error
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error", 413: "request_too_large",
               499: "invalid_request_error", 529: "overloaded_error"}


def sse(event: str, obj: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"


def error_body(msg: str, code: int) -> dict:
    """An error in the format of the Anthropic API (a response body, or a stream event)."""
    return {"type": "error", "error": {"type": ERROR_TYPES.get(code, "api_error"), "message": msg}}


def _blocks(content) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]
    raise ValueError("content must be a string or a list of content blocks")


def text_of(content) -> str:
    """The text of a content field: text blocks joined by blank lines, a note for each image or document."""
    parts = []
    for b in _blocks(content):
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text") or "")
        elif t in ("image", "document"):
            parts.append(f"[{t} omitted: this server reads text only]")
    return "\n\n".join(p for p in parts if p)


def _image_url(b: dict) -> str:
    src = b.get("source")
    if not isinstance(src, dict) or src.get("type") != "base64" or not isinstance(src.get("data"), str):
        raise ValueError("an image block needs a base64 source: the server fetches nothing")
    return f"data:{src.get('media_type') or 'image/png'};base64,{src['data']}"


def content_of(content, images: bool):
    """A content field as the content of a chat message: its text (text_of), or, when it holds images and images is
    True, a list of text and image_url parts in block order."""
    parts = []
    for b in _blocks(content):
        if b.get("type") == "image" and images:
            parts.append({"type": "image_url", "image_url": {"url": _image_url(b)}})
            continue
        s = text_of([b])
        if s and parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += "\n\n" + s
        elif s:
            parts.append({"type": "text", "text": s})
    if not any(p["type"] == "image_url" for p in parts):
        return "\n\n".join(p["text"] for p in parts)
    return parts


def _reminder(text: str) -> str:
    text = text.strip()
    return text if text.startswith("<system-reminder>") else f"<system-reminder>\n{text}\n</system-reminder>"


def to_messages(body: dict, images: bool = False) -> list[dict]:
    """Anthropic `system` + `messages` -> the OpenAI-style message list of engine/server/chat.py ChatFormat.render.
    images: the image blocks of user turns and tool results become image parts (else text notes)."""
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw or not all(isinstance(m, dict) for m in raw):
        raise ValueError("messages must be a non-empty list of objects")
    if raw[-1].get("role") == "assistant":
        raise ValueError("a final assistant message (prefill) is not supported")
    out = []
    system = text_of(body.get("system"))
    if system:
        out.append({"role": "system", "content": system})
    for m in raw:
        role = m.get("role")
        if role == "system":
            text = text_of(m.get("content"))
            if text:
                out.append({"role": "user", "content": _reminder(text)})
        elif role == "user":
            pending = []
            for b in _blocks(m.get("content")):
                if b.get("type") == "tool_result":
                    c = content_of(pending, images)
                    if c:
                        out.append({"role": "user", "content": c})
                    pending = []
                    result = content_of(b.get("content"), images)
                    if b.get("is_error") and isinstance(result, list):
                        result = [{"type": "text", "text": TOOL_ERROR + "\n"}] + result
                    elif b.get("is_error"):
                        result = f"{TOOL_ERROR}\n{result}" if result else TOOL_ERROR
                    out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": result})
                else:
                    pending.append(b)
            c = content_of(pending, images)
            if c:
                out.append({"role": "user", "content": c})
        elif role == "assistant":
            reasoning, text, calls = [], [], []
            for b in _blocks(m.get("content")):
                t = b.get("type")
                if t == "thinking":
                    reasoning.append(b.get("thinking") or "")
                elif t == "text":
                    text.append(b.get("text") or "")
                elif t == "tool_use":
                    if not isinstance(b.get("input", {}), dict):
                        raise ValueError("a tool_use input must be an object")
                    calls.append({"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
            msg = {"role": "assistant", "content": "\n\n".join(t for t in text if t)}
            if any(reasoning):
                msg["reasoning_content"] = "\n\n".join(r for r in reasoning if r)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            raise ValueError(f"unsupported message role {role!r}")
    return out


def to_tools(body: dict) -> list[dict] | None:
    """Anthropic tools -> OpenAI function tools. Server tools are left out."""
    fns, server = [], []
    tools = body.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise ValueError("tools must be a list of objects")
    for t in tools:
        if t.get("type") not in (None, "custom"):
            server.append(str(t.get("type")))
            continue
        if not isinstance(t.get("name"), str) or not isinstance(t.get("input_schema", {}), dict):
            raise ValueError("a tool needs a string name, and its input_schema must be an object")
        fn = {"name": t["name"]}
        if t.get("description"):
            fn["description"] = t["description"]
        fn["parameters"] = t.get("input_schema") or {"type": "object", "properties": {}}
        fns.append({"type": "function", "function": fn})
    if server and not fns:
        raise ValueError(f"this server does not run server tools ({', '.join(server)})")
    return fns or None


def template_kwargs(body: dict, default_thinking: bool | None, default_effort: str | None = None) -> dict:
    """The chat template kwargs of an Anthropic request: `thinking.type` as enable_thinking and output_config.effort as
    reasoning_effort, resolved by the one resolver of the server (engine/server/chat.py template_kwargs). Without
    `thinking`, a request runs without reasoning, as on the Anthropic API, unless default_thinking (--thinking) says
    otherwise."""
    th = body.get("thinking")
    kind = th.get("type") if isinstance(th, dict) else None
    b = {}
    if kind == "disabled":
        b["enable_thinking"] = False
    elif kind in ("enabled", "adaptive"):
        b["enable_thinking"] = True
    oc = body.get("output_config")
    effort = oc.get("effort") if isinstance(oc, dict) else None
    if isinstance(effort, str):
        b["reasoning_effort"] = effort
    return _template_kwargs(b, bool(default_thinking), default_effort)


def drops_tool_calls(body: dict) -> bool:
    tc = body.get("tool_choice")
    return isinstance(tc, dict) and tc.get("type") == "none"


def usage(prompt_tokens: int, cached: int, output_tokens: int) -> dict:
    return {"input_tokens": prompt_tokens - cached, "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
            "output_tokens": output_tokens}


def stop_reason(finish: str, n_tool_calls: int, stop_match: str | None) -> tuple[str, str | None]:
    """(stop_reason, stop_sequence) from the scheduler's finish reason and the parser state."""
    if stop_match is not None:
        return "stop_sequence", stop_match
    if finish in ("length", "timeout"):
        return "max_tokens", None
    return ("tool_use" if n_tool_calls else "end_turn"), None


def message(msg_id: str, model: str, content: list, stop: tuple[str | None, str | None], use: dict) -> dict:
    return {"id": msg_id, "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop[0], "stop_sequence": stop[1], "usage": use}


class Blocks:
    """Parser events (engine/server/chat.py OutputParser) -> Anthropic content blocks.

    add() returns the stream events of the change. `content` holds the blocks for a reply without a stream."""

    def __init__(self, drop_tool_calls: bool = False):
        self.content: list[dict] = []
        self.open: str | None = None  # "thinking" or "text" while that block is open
        self.n_tool_calls = 0
        self.drop = drop_tool_calls

    def add(self, events) -> list[str]:
        out = []
        for kind, x in events:
            if kind == "tool_call":
                if self.drop:
                    continue
                out += self.close()
                i = len(self.content)
                block = {"type": "tool_use", "id": "toolu_" + x["id"].removeprefix("call_"), "name": x["name"],
                         "input": json.loads(x["arguments"])}
                self.content.append(block)
                self.n_tool_calls += 1
                out.append(sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}}))
                out.append(sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                       "delta": {"type": "input_json_delta", "partial_json": x["arguments"]}}))
                out.append(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
                continue
            key = "thinking" if kind == "reasoning" else "text"
            if self.open != key:
                out += self.close()
                self.open = key
                self.content.append({"type": "thinking", "thinking": "", "signature": SIGNATURE} if key == "thinking"
                                    else {"type": "text", "text": ""})
                start = {"type": "thinking", "thinking": "", "signature": ""} if key == "thinking" else {"type": "text", "text": ""}
                out.append(sse("content_block_start", {"type": "content_block_start", "index": len(self.content) - 1, "content_block": start}))
            self.content[-1][key] += x
            out.append(sse("content_block_delta", {"type": "content_block_delta", "index": len(self.content) - 1,
                                                   "delta": {"type": f"{key}_delta", key: x}}))
        return out

    def close(self) -> list[str]:
        """Closes the open thinking or text block (a thinking block gets its signature first)."""
        if self.open is None:
            return []
        i, out = len(self.content) - 1, []
        if self.open == "thinking":
            out.append(sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "signature_delta", "signature": SIGNATURE}}))
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
        self.open = None
        return out
