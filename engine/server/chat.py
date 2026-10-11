"""Chat formatting and output parsing for the server (PLAN.md 4.6).

* Prompt: the own Jinja chat template of the checkpoint, via `tokenizer.apply_chat_template`. tools, enable_thinking,
  reasoning_effort and any other chat_template_kwargs pass straight through. OpenAI-style assistant tool_calls
  (arguments as a JSON string) become the mapping that the template iterates over. template_kwargs() resolves the
  thinking switch and the effort level of a request, for both APIs of the server.
* Output: the parser splits the tokens into reasoning / content / tool calls at the token level. (`<think>`,
  `</think>`, `<tool_call>` and `</tool_call>` are single tokens in this vocabulary.) It detokenizes each stream
  incrementally. Tool calls use the XML form of Qwen (<function=name><parameter=p>value</parameter></function>).
  The parser converts them to OpenAI tool_calls, and the JSON schema of the request sets the types of the values (as
  the qwen3_xml parser of vLLM does).
* Stop strings: the parser matches them on the decoded text of each stream. It holds back text that can be the start
  of a stop string until it knows that it is not.
"""
from __future__ import annotations

import ast
import json
import re
import uuid

from engine.weights.loader import MODEL, resolve

_FUNC_RE = re.compile(r"<function=([^>\n]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
THINK_SCAN = 64  # tokens from the end of a prompt in which its generation prompt (<|im_start|>assistant, <think>...) lies

# reasoning_effort names of other APIs and templates -> the levels of the Qwen3.8 template (None: thinking off)
EFFORTS = {"none": None, "minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


def template_kwargs(b: dict, default_thinking: bool | None, default_effort: str | None = None) -> dict:
    """The chat template kwargs of a request: chat_template_kwargs, plus the top-level enable_thinking and
    reasoning_effort. Clients written for other templates set thinking with `thinking` (DeepSeek) and effort with names
    that this template rejects ("none", "high", "max"). Thus `thinking` stands for enable_thinking, and the names in
    EFFORTS map to the levels of this template. Other values in chat_template_kwargs go to the template unchanged.
    default_thinking: enable_thinking when the request does not say (None: the template's default). default_effort
    (--reasoning-effort): the level of a thinking request that names none (None: the template's). A bad value is a
    ValueError (the client's error)."""
    kw = b.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise ValueError("chat_template_kwargs must be an object")
    kw = dict(kw)
    if "enable_thinking" not in kw and isinstance(kw.get("thinking"), bool):
        kw["enable_thinking"] = kw["thinking"]
    if "enable_thinking" not in kw and b.get("enable_thinking") is not None:
        kw["enable_thinking"] = bool(b["enable_thinking"])
    if not isinstance(b.get("reasoning_effort"), (str, type(None))):
        raise ValueError("reasoning_effort must be a string")
    top_level = "reasoning_effort" not in kw
    effort = b.get("reasoning_effort") if top_level else kw.pop("reasoning_effort")
    if isinstance(effort, str) and effort in EFFORTS:
        if EFFORTS[effort] is None:
            kw.setdefault("enable_thinking", False)
        else:
            kw["reasoning_effort"] = EFFORTS[effort]
    elif effort:  # another top-level name means the template default; the template checks its own kwargs
        kw["reasoning_effort"] = "xhigh" if top_level else effort
    if "enable_thinking" not in kw and default_thinking is not None:
        kw["enable_thinking"] = default_thinking
    if default_effort is not None and kw.get("enable_thinking") is not False:
        kw.setdefault("reasoning_effort", default_effort)
    return kw


class ChatFormat:
    """The chat template of the checkpoint and the ids of its control tokens. The server renders every prompt through
    render(); the tools and the checks use the same object (from_checkpoint), so their prompts are the server's."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        tid = tokenizer.convert_tokens_to_ids
        self.think_open, self.think_close = tid("<think>"), tid("</think>")
        self.tool_open, self.tool_close = tid("<tool_call>"), tid("</tool_call>")
        self.im_start = tid("<|im_start|>")  # the message boundary (engine/runtime/scheduler.py takes checkpoints there)
        self.eos_ids = tuple(sorted({tid("<|im_end|>"), tid("<|endoftext|>")}))

    @classmethod
    def from_checkpoint(cls, path_or_repo: str = MODEL) -> ChatFormat:
        """The chat format of a checkpoint (a repo id in the local HF cache, or a directory)."""
        from transformers import AutoTokenizer  # a slow import, for the callers that have no tokenizer yet
        return cls(AutoTokenizer.from_pretrained(resolve(path_or_repo)))

    def render(self, messages: list[dict], tools: list | None = None, **template_kwargs) -> list[int]:
        msgs = [_normalize_message(m) for m in messages]
        kw = {k: v for k, v in template_kwargs.items() if v is not None}
        text = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False, **kw)
        return self.tok.encode(text, add_special_tokens=False)

    def opens_in_reasoning(self, prompt: list[int]) -> bool:
        """True when the prompt ends inside an unclosed <think> (the template's thinking generation prompt)."""
        for t in reversed(prompt[-THINK_SCAN:]):
            if t == self.think_close:
                return False
            if t == self.think_open:
                return True
        return False


def check_messages(messages, tools) -> None:
    """Raises ValueError unless messages / tools have the shapes that the template and the parser read: the request's
    errors are then the client's (400), and any other failure of the rendering is the server's (500)."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise ValueError(f"messages[{i}] must be an object with a string role")
        c = m.get("content")
        if not (c is None or isinstance(c, str) or (isinstance(c, list) and all(isinstance(p, dict) for p in c))):
            raise ValueError(f"messages[{i}].content must be a string, a list of content parts or null")
        for k in ("reasoning_content", "reasoning"):
            if not isinstance(m.get(k), (str, type(None))):
                raise ValueError(f"messages[{i}].{k} must be a string")
        calls = m.get("tool_calls")
        if calls is not None and not (isinstance(calls, list) and all(
                isinstance(c, dict) and isinstance(c.get("function"), dict) and isinstance(c["function"].get("arguments", {}), (str, dict))
                for c in calls)):
            raise ValueError(f"messages[{i}].tool_calls must be a list of {{function: {{name, arguments}}}} objects")
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise ValueError("tools must be a list of objects")


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
            if isinstance(args, str):  # the template iterates over a mapping: keep any other JSON as one argument
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    parsed = None
                fn["arguments"] = parsed if isinstance(parsed, dict) else {"arguments": args}
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


def _convert(value: str, schema: dict | bool | None):
    if schema is not None and not isinstance(schema, dict):  # a boolean JSON schema (true: any value) has no type
        schema = {}
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
    for t in tools or []:  # the client's schemas: any JSON may stand where an object is expected
        fn = t.get("function", t) if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            props = props if isinstance(props, dict) else {}
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
    """One text stream: holds back text that might begin a stop string, and records the stop string that ends it.
    A chat stream (reasoning or content) also strips leading whitespace and holds trailing whitespace, so that a stream
    that ends, or is followed by a tool call, carries no dangling newlines. keep_whitespace: a raw text stream
    (/v1/completions) keeps every character."""

    def __init__(self, stops: list[str], keep_whitespace: bool = False):
        self.stops = [s for s in stops if s]
        self.keep_ws = keep_whitespace
        self.started = False
        self.pending = ""   # held text, not yet emitted
        self.stopped = False
        self.matched: str | None = None  # the stop string that stopped the stream

    def push(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        if not self.started:
            if not self.keep_ws:
                text = text.lstrip()
            if not text:
                return ""
            self.started = True
        buf = self.pending + text
        for s in self.stops:
            i = buf.find(s)
            if i >= 0:
                self.stopped, self.pending, self.matched = True, "", s
                return buf[:i] if self.keep_ws else buf[:i].rstrip()
        hold = 0  # the longest suffix that may start a stop string (plus the whitespace before it, in a chat stream)
        for s in self.stops:
            for n in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:n]):
                    hold = max(hold, n)
                    break
        cut = len(buf) - hold if self.keep_ws else len(buf[: len(buf) - hold].rstrip())
        self.pending = buf[cut:]
        return buf[:cut]

    def flush(self) -> str:
        """The held text (nothing after a stop), and the end of the stream."""
        out = "" if self.stopped else (self.pending if self.keep_ws else self.pending.rstrip())
        self.pending = ""
        return out


class OutputParser:
    """Token-level splitter for chat output. feed(token) -> list of events:
    ("reasoning", text) | ("content", text) | ("tool_call", {"id", "name", "arguments"})."""

    def __init__(self, fmt: ChatFormat, in_reasoning: bool, tools: list | None, stops: list[str] = ()):
        self.fmt, self.tools = fmt, tools
        self.mode = "reasoning" if in_reasoning else "content"
        self.detok = Detokenizer(fmt.tok)
        self.streams = {"reasoning": _Stream(list(stops)), "content": _Stream(list(stops))}
        self.tool_ids: list[int] = []
        self.n_tool_calls = 0
        self.any_content = False

    @property
    def stopped(self) -> bool:
        return any(s.stopped for s in self.streams.values())

    @property
    def stop_match(self) -> str | None:
        return next((s.matched for s in self.streams.values() if s.stopped), None)

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
        if self.mode == "content" and t == f.tool_open and self.tools:
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
    """Raw-text output (/v1/completions): one content stream that keeps its whitespace, no reasoning / tool parsing,
    stop strings."""

    def __init__(self, tok, stops: list[str] = ()):
        self.detok = Detokenizer(tok)
        self.stream = _Stream(list(stops), keep_whitespace=True)

    @property
    def stopped(self) -> bool:
        return self.stream.stopped

    @property
    def stop_match(self) -> str | None:
        return self.stream.matched

    def feed(self, t: int) -> list:
        out = self.stream.push(self.detok.add(t))
        return [("content", out)] if out else []

    def finish(self) -> list:
        out = self.stream.push(self.detok.flush()) + self.stream.flush()
        return [("content", out)] if out else []
