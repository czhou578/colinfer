"""The OpenAI formats of the server (engine/server/api.py): the chat completion and text completion bodies, their
stream chunks, usage and timings, and the error body. engine/server/anthropic.py is the same for the Anthropic format.

The output events come from the parser (engine/server/chat.py): ("reasoning", text), ("content", text) or
("tool_call", {"id", "name", "arguments"}), grouped by the engine thread into ("delta", events, token ids, logprobs).
"""
from __future__ import annotations

import json
import time

from engine.runtime.scheduler import Request as EngineRequest

MAX_TOP_LOGPROBS = 20


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def error_body(msg: str, code: int) -> dict:
    """An error in the format of the OpenAI API (a response body, or a stream event); the code sets the type."""
    return {"error": {"message": msg, "type": "server_error" if code >= 500 else "invalid_request_error", "code": code}}


def usage(req: EngineRequest) -> dict:
    pt, ct = len(req.prompt), len(req.output)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "prompt_tokens_details": {"cached_tokens": req.reused}}


def timings(req: EngineRequest) -> dict:
    pre = len(req.prompt) - req.reused
    prefill_s = max(req.t_first - req.t_admit, 1e-9)
    decode_s = max(req.t_done - req.t_first, 1e-9)
    return {"queue_s": round(req.t_admit - req.t_submit, 4), "ttft_s": round(req.t_first - req.t_submit, 4), "prefill_tokens": pre,
            "prefill_s": round(prefill_s, 4), "prefill_tok_s": round(pre / prefill_s, 1), "decode_s": round(decode_s, 4),
            "decode_tok_s": round((len(req.output) - 1) / decode_s, 2) if len(req.output) > 1 else None}


def finish_reason(r: EngineRequest) -> str:
    # an abort (the client went away) reads as a stop, a timeout as a cut-off
    return {"abort": "stop", "timeout": "length"}.get(r.finish_reason, r.finish_reason)


def tool_call(x: dict) -> dict:
    return {"id": x["id"], "type": "function", "function": {"name": x["name"], "arguments": x["arguments"]}}


def stream_end(rid: str, obj: str, model: str, r: EngineRequest, include_usage: bool) -> list[str]:
    """The last events of a stream: the usage chunk (stream_options.include_usage), then [DONE]."""
    out = []
    if include_usage:
        out.append(sse({"id": rid, "object": obj, "created": int(time.time()), "model": model, "choices": [], "usage": usage(r),
                        "timings": timings(r)}))
    return out + ["data: [DONE]\n\n"]


# ------------------------------------------------------------------------------------------------ chat completions
def chat_chunk(rid: str, model: str, delta: dict, finish: str | None = None, extra: dict | None = None) -> dict:
    c = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
         "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
    if extra:
        c["choices"][0].update(extra)
    return c


def chat_message(events) -> tuple[dict, list]:
    """The assistant message of a reply without a stream, from its output events: (message, logprob entries)."""
    reasoning, content, calls, lps = "", "", [], []
    for _, ev, _, item_lps in events:
        for kind, x in ev:
            if kind == "tool_call":
                calls.append(tool_call(x))
            elif kind == "reasoning":
                reasoning += x
            else:
                content += x
        lps += item_lps
    msg = {"role": "assistant", "content": content if (content or not calls) else None}
    if reasoning:
        msg["reasoning_content"] = msg["reasoning"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg, lps


# ------------------------------------------------------------------------------------------------ completions
def lp_block(lps: list, n_top: int | None) -> dict | None:
    if n_top is None:
        return None
    return {"tokens": [e["token"] for e in lps], "token_logprobs": [e["logprob"] for e in lps],
            "top_logprobs": [{t["token"]: t["logprob"] for t in e["top_logprobs"]} for e in lps], "text_offset": []}


def text_chunk(rid: str, model: str, text: str, finish: str | None = None, lps: list | None = None, n_top: int | None = None) -> dict:
    return {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "text": text, "logprobs": lp_block(lps or [], n_top) if lps else None, "finish_reason": finish}]}


def completion_text(events) -> tuple[str, list]:
    """The text of a reply without a stream, from its output events: (text, logprob entries)."""
    text = "".join(x for _, ev, _, _ in events for _, x in ev)
    lps = [e for _, _, _, item_lps in events for e in item_lps]
    return text, lps
