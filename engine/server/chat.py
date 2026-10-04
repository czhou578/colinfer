"""Chat formatting and output parsing for the OpenAI-compatible server (PLAN.md 4.6).

* Prompt: the checkpoint's own Jinja chat template via `tokenizer.apply_chat_template` (tools, enable_thinking,
  reasoning_effort and any other chat_template_kwargs pass straight through). OpenAI-style assistant
  tool_calls (arguments as a JSON string) are converted to the mapping the template iterates over.
* Output: tokens are split into reasoning / content / tool calls at the token level (`<think>`, `</think>`,
  `<tool_call>`, `</tool_call>` are single tokens in this vocabulary), each stream detokenized incrementally.
  Tool calls use Qwen's XML form (<function=name><parameter=p>value</parameter></function>), converted to
  OpenAI tool_calls with values typed by the request's JSON schema (as vLLM's qwen3_xml parser does).
* Stop strings are matched on the decoded text of each stream; text that might be the start of a stop string
  is held back until it is known not to be.
"""
from __future__ import annotations

import ast
import json
import re
import uuid

_FUNC_RE = re.compile(r"<function=([^>\n]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


class ChatFormat:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        tid = tokenizer.convert_tokens_to_ids
        self.think_open, self.think_close = tid("<think>"), tid("</think>")
        self.tool_open, self.tool_close = tid("<tool_call>"), tid("</tool_call>")
        self.eos_ids = tuple(sorted({tid("<|im_end|>"), tid("<|endoftext|>")}))

    def render(self, messages: list[dict], tools: list | None = None, **template_kwargs) -> list[int]:
        msgs = [_normalize_message(m) for m in messages]
        kw = {k: v for k, v in template_kwargs.items() if v is not None}
        text = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
        return self.tok.encode(text, add_special_tokens=False)

    def opens_in_reasoning(self, prompt: list[int]) -> bool:
        """True when the prompt ends inside an unclosed <think> (the template's thinking generation prompt)."""
        for t in reversed(prompt[-64:]):
            if t == self.think_close:
                return False
            if t == self.think_open:
                return True
        return False


def _normalize_message(m: dict) -> dict:
    m = dict(m)
    if m.get("content") is None:
        m["content"] = ""
    if m.get("tool_calls"):
        calls = []
        for c in m["tool_calls"]:
            c = dict(c)
            fn = dict(c.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    fn["arguments"] = {"arguments": args}
            c["function"] = fn
            calls.append(c)
        m["tool_calls"] = calls
    # clients send the previous reasoning back under either name
    if m.get("role") == "assistant" and m.get("reasoning_content") is None and isinstance(m.get("reasoning"), str):
        m["reasoning_content"] = m["reasoning"]
    return m


class Detokenizer:
    """Incremental detokenization (same scheme as vLLM): re-decode a short window and emit the new suffix,
    holding back text while it ends in an incomplete UTF-8 sequence."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def add(self, t: int) -> str:
        self.ids.append(t)
        prev = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=True)
        new = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=True)
        if len(new) > len(prev) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(prev):]
        return ""

    def flush(self) -> str:
        prev = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=True)
        new = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=True)
        self.prefix = self.read = len(self.ids)
        return new[len(prev):] if len(new) > len(prev) else ""


def _convert(value: str, schema: dict | None):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if value.strip().lower() == "null" and t != "string":
        return None
    try:
        if t in ("string", "str", "text"):
            return value
        if t in ("integer", "int"):
            return int(value.strip())
        if t in ("number", "float"):
            f = float(value.strip())
            return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f
        if t in ("boolean", "bool"):
            return value.strip().lower() == "true"
    except ValueError:
        return value
    if t is None and schema is None:
        return value
    for parse in (json.loads, ast.literal_eval):  # object / array / untyped
        try:
            return parse(value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return value


def parse_tool_call(text: str, tools: list | None) -> dict | None:
    """Body of one <tool_call>...</tool_call> -> {"name", "arguments" (JSON string)} or None if malformed."""
    m = _FUNC_RE.search(text)
    if not m:
        try:  # some fine-tunes emit Hermes-style JSON inside <tool_call>
            obj = json.loads(text)
            return {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False)}
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    name = m.group(1).strip()
    props = {}
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
    args = {}
    for pm in _PARAM_RE.finditer(text, m.end()):
        key, val = pm.group(1).strip(), pm.group(2)
        if val.startswith("\n"):
            val = val[1:]
        if val.endswith("\n"):
            val = val[:-1]
        args[key] = _convert(val, props.get(key) if props else None)
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


class _Stream:
    """One text stream (reasoning or content): strips leading whitespace and holds trailing whitespace (so a
    stream that ends, or is followed by a tool call, carries no dangling newlines), and holds back text that
    might begin a stop string."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.started = False
        self.pending = ""   # held text, not yet emitted
        self.stopped = False

    def push(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        if not self.started:
            text = text.lstrip()
            if not text:
                return ""
            self.started = True
        buf = self.pending + text
        for s in self.stops:
            i = buf.find(s)
            if i >= 0:
                self.stopped, self.pending = True, ""
                return buf[:i].rstrip()
        hold = 0  # the longest suffix that may start a stop string, plus the whitespace before it
        for s in self.stops:
            for n in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:n]):
                    hold = max(hold, n)
                    break
        cut = len(buf[: len(buf) - hold].rstrip())
        self.pending = buf[cut:]
        return buf[:cut]

    def flush(self) -> str:
        out, self.pending = ("" if self.stopped else self.pending.rstrip()), ""
        return out


class OutputParser:
    """Token-level splitter for chat output. feed(token) -> list of events:
    ("reasoning", text) | ("content", text) | ("tool_call", {"id", "name", "arguments"})."""

    def __init__(self, fmt: ChatFormat, in_reasoning: bool, tools: list | None, stops: list[str] = (), parse_tools: bool = True):
        self.fmt, self.tools, self.parse_tools = fmt, tools, parse_tools and bool(tools)
        self.mode = "reasoning" if in_reasoning else "content"
        self.detok = Detokenizer(fmt.tok)
        self.streams = {"reasoning": _Stream(list(stops)), "content": _Stream(list(stops))}
        self.tool_ids: list[int] = []
        self.n_tool_calls = 0
        self.any_content = False

    @property
    def stopped(self) -> bool:
        return any(s.stopped for s in self.streams.values())

    def _text(self, mode, text):
        out = self.streams[mode].push(text)
        if out and mode == "content":
            self.any_content = True
        return [(mode, out)] if out else []

    def _switch(self, mode):
        ev = self._text(self.mode, self.detok.flush()) if self.mode != "tool" else []
        if self.mode in self.streams:
            tail = self.streams[self.mode].flush()
            if tail:
                ev.append((self.mode, tail))
        self.mode = mode
        self.detok = Detokenizer(self.fmt.tok)
        if mode in self.streams:  # a new stream segment after a tool call: strip its leading whitespace again
            self.streams[mode].started = False
        return ev

    def feed(self, t: int) -> list:
        f = self.fmt
        if self.mode == "reasoning" and t == f.think_close:
            return self._switch("content")
        if self.mode == "content" and t == f.think_open and not self.any_content and not self.n_tool_calls:
            return self._switch("reasoning")  # the model opened its own thinking block
        if self.mode == "content" and t == f.tool_open and self.parse_tools:
            ev = self._switch("tool")
            self.tool_ids = []
            return ev
        if self.mode == "tool":
            if t == f.tool_close:
                return self._close_tool()
            self.tool_ids.append(t)
            return []
        return self._text(self.mode, self.detok.add(t))

    def _close_tool(self):
        body = self.fmt.tok.decode(self.tool_ids, skip_special_tokens=True)
        call = parse_tool_call(body, self.tools)
        self.mode, self.detok = "content", Detokenizer(self.fmt.tok)
        self.streams["content"].started = False
        if call is None:  # malformed: hand it back as text
            return self._text("content", "<tool_call>" + body + "</tool_call>")
        self.n_tool_calls += 1
        return [("tool_call", {"id": "call_" + uuid.uuid4().hex[:24], **call})]

    def finish(self) -> list:
        if self.mode == "tool":  # unterminated tool call (length limit, or a stop token before </tool_call>)
            # keep the call with its completed parameters, as vLLM's parser does; finish_reason stays "length"
            body = self.fmt.tok.decode(self.tool_ids, skip_special_tokens=True)
            call = parse_tool_call(body, self.tools) if _FUNC_RE.search(body) else None
            self.mode, self.detok = "content", Detokenizer(self.fmt.tok)
            if call is not None:
                self.n_tool_calls += 1
                return [("tool_call", {"id": "call_" + uuid.uuid4().hex[:24], **call})]
            return self._text("content", "<tool_call>" + body) + self._tail()
        return self._text(self.mode, self.detok.flush()) + self._tail()

    def _tail(self):
        ev = []
        for mode, st in self.streams.items():
            tail = st.flush()
            if tail:
                ev.append((mode, tail))
        return ev


class TextParser:
    """Raw-text output (/v1/completions): one content stream, no reasoning / tool parsing, stop strings."""

    def __init__(self, tok, stops: list[str] = ()):
        self.detok = Detokenizer(tok)
        self.stream = _Stream(list(stops))
        self.stream.started = True  # keep leading whitespace in raw completions
        self.stops = [s for s in stops if s]

    @property
    def stopped(self) -> bool:
        return self.stream.stopped

    def feed(self, t: int) -> list:
        out = self._push(self.detok.add(t))
        return [("content", out)] if out else []

    def _push(self, text):  # stop-string hold only; raw completions keep their whitespace
        st = self.stream
        if st.stopped or not text:
            return ""
        buf = st.pending + text
        for s in self.stops:
            i = buf.find(s)
            if i >= 0:
                st.stopped, st.pending = True, ""
                return buf[:i]
        hold = 0
        for s in self.stops:
            for n in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:n]):
                    hold = max(hold, n)
                    break
        st.pending = buf[len(buf) - hold:] if hold else ""
        return buf[: len(buf) - hold]

    def finish(self) -> list:
        out = self._push(self.detok.flush())
        st = self.stream
        if not st.stopped:
            out += st.pending
        st.pending = ""
        return [("content", out)] if out else []
