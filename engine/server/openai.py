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
    """The phases of a finished request (llama.cpp's `timings`), each with its token count and its rate:

    * queue: submit to admission (the restore of a checkpoint is part of it);
    * prefill: admission to the first token, over the prompt tokens that no checkpoint served;
    * decode: the first token to the end, over the output tokens after the first (the first comes from the prefill);
    * speculation: the decode cycles, the draft tokens they verified (k per cycle) and accepted, and the acceptance rate.

    A rate is None when its phase has nothing to count, or when the request ended (an abort, a timeout) before the phase:
    a request that ended in the queue has no TTFT and no prefill rate; one that ended during its prefill has no prefill
    rate (the prefill time is the time it ran)."""
    pre, dec = len(req.prompt) - req.reused, max(len(req.output) - 1, 0)
    admit = req.t_admit or req.t_done  # 0.0: it ended in the queue
    first = req.t_first or req.t_done  # 0.0: it ended before its first token
    prefill_s, decode_s = first - admit, req.t_done - first
    return {"queue_s": round(admit - req.t_submit, 4), "ttft_s": round(first - req.t_submit, 4) if req.t_first else None,
            "prefill_tokens": pre, "prefill_s": round(prefill_s, 4),
            "prefill_tok_s": round(pre / prefill_s, 1) if req.t_first and pre > 0 and prefill_s > 0 else None,
            "decode_tokens": dec, "decode_s": round(decode_s, 4),
            "decode_tok_s": round(dec / decode_s, 2) if dec > 0 and decode_s > 0 else None,
            "decode_cycles": req.cycles, "draft_tokens": req.drafted, "draft_accepted": req.accepted,
            "draft_accept_rate": round(req.accepted / req.drafted, 4) if req.drafted > 0 else None}


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
