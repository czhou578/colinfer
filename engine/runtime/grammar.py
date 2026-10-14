"""Constrained decoding: a grammar (llguidance) decides which tokens a request may emit next.

* JSON: the reply is one JSON value that matches a schema (OpenAI response_format json_object / json_schema, the
  Anthropic output_format).
* Forced tool calls (OpenAI tool_choice required / a named function, Anthropic any / tool): the reply is one or more
  tool calls in the XML format of Qwen, with the function names and parameter names of the request's tools. The JSON
  schema of each parameter types its value: integers, numbers, booleans and enums as such, objects and arrays as
  JSON, strings as free text. The server's parser (engine/server/chat.py) reads the same format.
* Reasoning: when the prompt opens a <think> block, the grammar first lets the model reason freely up to </think>.

Speculation stays exact. The scheduler gives each verify row of a cycle the mask of the matcher advanced by the drafts
before that row (Constraint.masks). A draft that the grammar refuses cannot equal the masked token of the target at
its row, so acceptance stops there, and a cycle emits exactly what plain constrained decoding would emit.
"""
from __future__ import annotations

import json

import llguidance
import llguidance.hf
import llguidance.numpy
import numpy as np

END_PARAM = "\\n</parameter>\\n"  # (as a Lark string literal)
WS = r"/[ \t\n]*/"


def mask_words(vocab: int) -> int:
    """int32 words of a token bitmask: token t is bit t % 32 of word t // 32."""
    return (vocab + 31) // 32


class Constraint:
    """One request's grammar state (an llguidance matcher), advanced by the tokens the request emits."""

    def __init__(self, grammars: Grammars, lark: str):
        self.g = grammars
        self.m = llguidance.LLMatcher(grammars.tok, llguidance.LLMatcher.grammar_from_lark(lark), log_level=0)
        if self.m.is_error():
            raise ValueError(f"cannot build the grammar: {self.m.get_error()}")

    def masks(self, drafts: list[int], out: np.ndarray):
        """out[i] (int32 [len(drafts) + 1, words]) = the mask after drafts[:i], for each i up to the first draft that the
        grammar refuses (the later rows are left as they are: the verify never accepts past that draft)."""
        if drafts:
            llguidance.numpy.fill_next_token_bitmask_par_with_draft_tokens(self.g.executor, [(self.m, 0, list(drafts))], out)
        else:
            llguidance.numpy.fill_next_token_bitmask(self.m, out, 0)

    def advance(self, tokens: list[int]) -> bool:
        """Consumes emitted tokens. False when the grammar refuses one (never, for tokens drawn under its masks)."""
        return self.m.consume_tokens(list(tokens))


def _lit(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


class Grammars:
    """The llguidance tokenizer of the checkpoint (built once, ~1.5 s) and the grammar of each kind of constraint.
    fmt: the server's ChatFormat (its control token ids); vocab: the model's vocabulary (the mask width)."""

    def __init__(self, fmt, vocab: int):
        self.fmt, self.vocab = fmt, vocab
        self.tok = llguidance.hf.from_tokenizer(fmt.tok, n_vocab=vocab, eos_token=list(fmt.eos_ids))
        self.executor = llguidance.LLExecutor(num_threads=1)

    def _head(self, think: bool, body: str) -> str:
        """The start rule: the free reasoning when the prompt opened it, then body."""
        if not think:
            return f"start: {WS} {body} {WS}\n"
        return (f"start: think {WS} {body} {WS}\n"
                f"think: THINK <[{self.fmt.think_close}]>\n"
                "THINK: /(?s:.*)/\n")

    def json(self, schema: dict | None, think: bool) -> Constraint:
        """One JSON value that matches schema (None: any JSON object)."""
        schema = {"type": "object"} if schema is None else schema
        if not isinstance(schema, dict):
            raise ValueError("a JSON schema must be an object")
        return Constraint(self, self._head(think, "%json " + json.dumps(schema)))

    def tools(self, tools: list[dict], names: list[str] | None = None, single: bool = False, think: bool = False) -> Constraint:
        """Tool calls (one with single, else one or more) to the functions `names` of tools (None: any of them)."""
        fns = {}
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                fns[fn["name"]] = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        names = list(fns) if names is None else names
        unknown = [n for n in names if n not in fns]
        if not names or unknown:
            raise ValueError(f"tool_choice names {unknown or 'no'} tool of the request's tools")
        rules, calls = [], []
        to, tc = self.fmt.tool_open, self.fmt.tool_close
        for i, name in enumerate(names):
            props = fns[name].get("properties")
            props = props if isinstance(props, dict) else {}
            params = []
            for j, (p, sch) in enumerate(props.items()):
                rules.append(f"p_{i}_{j}: {_lit('<parameter=' + p + '>' + chr(10))} {self._value(sch)}")
                params.append(f"p_{i}_{j}")
            rules.append(f"call_{i}: <[{to}]> {_lit(chr(10) + '<function=' + name + '>' + chr(10))} "
                         + (f"({' | '.join(params)})* " if params else "") + f"{_lit('</function>' + chr(10))} <[{tc}]>")
            calls.append(f"call_{i}")
        rules.append(f"call: {' | '.join(calls)}")
        rules.append("str_value[lazy]: STR_VALUE")
        rules.append(r"STR_VALUE: /(?s:.*)\n<\/parameter>\n/ & ~/(?s:.*<\/parameter>.*<\/parameter>.*)/")
        body = "call" if single else f"call ({WS} call)*"
        return Constraint(self, self._head(think, body) + "\n".join(rules) + "\n")

    @staticmethod
    def _value(schema) -> str:
        """The value of a parameter, then its end, as the parser reads it (engine/server/chat.py _convert)."""
        end = f'"{END_PARAM}"'
        s = schema if isinstance(schema, dict) else {}
        if isinstance(s.get("enum"), list) and s["enum"] and all(isinstance(v, (str, int, float)) for v in s["enum"]):
            vals = [json.dumps(v) if isinstance(v, bool) else str(v) for v in s["enum"]]
            return f"({' | '.join(_lit(v) for v in dict.fromkeys(vals))}) {end}"
        t = s.get("type")
        if t == "integer":
            return f"/-?(0|[1-9][0-9]*)/ {end}"
        if t == "number":
            return f"/-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/ {end}"
        if t == "boolean":
            return f'("true" | "false") {end}'
        if t in ("object", "array"):
            return f"%json {json.dumps(s)} {end}"
        return "str_value"  # strings, unions, untyped: free text up to the end of the parameter
