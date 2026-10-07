#!/usr/bin/env python3
"""Black-box checks of a running server (engine/server/api.py) over HTTP.

   uv run python -m engine.server --port 8001 &      # then
   uv run python tests/server_check.py --url http://127.0.0.1:8001
"""
import argparse
import concurrent.futures as cf
import json
import time

import requests

WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                             "parameters": {"type": "object", "properties": {
                                                 "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                                 "days": {"type": "integer"}}, "required": ["city"]}}}]


def post(url, body, stream=False):
    r = requests.post(url, json=body, stream=stream, timeout=600)
    return r


def sse(r):
    for line in r.iter_lines():
        line = line.decode()
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    a = ap.parse_args()
    U = a.url + "/v1"
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    # 1. tool call, non-streaming
    t0 = time.time()
    d = post(U + "/chat/completions", {"messages": [{"role": "user", "content": "What's the weather in Paris in celsius for the next 3 days?"}],
                                       "tools": WEATHER, "temperature": 0, "max_tokens": 800}).json()
    ch = d["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("tool call (non-stream)", ch["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
          and args.get("city") == "Paris" and isinstance(args.get("days", 3), int), f"{args} in {time.time() - t0:.1f}s")

    # 2. tool round trip, streaming, thinking off
    msgs = [{"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                   "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18, "sky": "clear"}'}]
    text, fr = "", None
    for c in sse(post(U + "/chat/completions", {"messages": msgs, "tools": WEATHER, "temperature": 0, "max_tokens": 200, "stream": True,
                                                "chat_template_kwargs": {"enable_thinking": False}}, stream=True)):
        if c["choices"]:
            text += c["choices"][0]["delta"].get("content") or ""
            fr = c["choices"][0]["finish_reason"] or fr
    check("tool result -> answer (stream)", "18" in text and fr == "stop", repr(text[:100]))

    # 3. streamed tool call deltas
    calls = []
    for c in sse(post(U + "/chat/completions", {"messages": [{"role": "user", "content": "Weather in Tokyo and in Oslo? Call the tool for both."}],
                                                "tools": WEATHER, "temperature": 0, "max_tokens": 1500, "stream": True}, stream=True)):
        if c["choices"]:
            calls += c["choices"][0]["delta"].get("tool_calls") or []
            fr = c["choices"][0]["finish_reason"] or fr
    cities = sorted(json.loads(x["function"]["arguments"]).get("city", "") for x in calls)
    check("parallel tool calls (stream)", fr == "tool_calls" and cities == ["Oslo", "Tokyo"] and [x["index"] for x in calls] == [0, 1], cities)

    # 4. completions endpoint, seeded sampling reproducible, logprobs
    body = {"prompt": "The capital of France is", "max_tokens": 12, "temperature": 0.8, "seed": 42, "logprobs": 3}
    r1, r2 = post(U + "/completions", body).json(), post(U + "/completions", body).json()
    lp = r1["choices"][0]["logprobs"]
    check("completions seeded + logprobs", r1["choices"][0]["text"] == r2["choices"][0]["text"] and len(lp["tokens"]) == r1["usage"]["completion_tokens"]
          and all(len(t) <= 3 for t in lp["top_logprobs"]), repr(r1["choices"][0]["text"]))

    # 5. four concurrent requests: three slots, the fourth queues
    def one(i):
        t = time.time()
        d = post(U + "/chat/completions", {"messages": [{"role": "user", "content": f"Write a haiku about the number {i}."}], "temperature": 0,
                                           "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}).json()
        return d["timings"]["queue_s"], time.time() - t, d["usage"]["completion_tokens"]
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, range(4)))
    queued = sorted(q for q, _, _ in res)
    check("4 concurrent -> one queued", queued[-1] > 0.2 and queued[0] < 0.2, f"queue times {[round(q, 2) for q in queued]}")

    # 6. client disconnect frees the slot
    r = post(U + "/chat/completions", {"messages": [{"role": "user", "content": "Write a very long essay about oceans."}], "max_tokens": 3000,
                                       "stream": True, "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
    n = 0
    for _ in sse(r):
        n += 1
        if n > 5:
            break
    r.close()
    time.sleep(1.0)
    st = requests.get(U + "/status").json()
    check("disconnect aborts the request", all(s["phase"] == "idle" for s in st["slots"]), [s["phase"] for s in st["slots"]])

    # 7. errors
    e1 = post(U + "/chat/completions", {"messages": []})
    e2 = post(U + "/chat/completions", {"messages": [{"role": "user", "content": "x " * 300000}], "max_tokens": 4})
    e3 = post(U + "/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2})
    check("bad requests -> 400", [e1.status_code, e2.status_code, e3.status_code] == [400, 400, 400], e2.json()["error"]["message"][:80])
    # fields the engine cannot honor are rejected, unless they leave the output unchanged
    hi = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0}
    rejected = [post(U + "/chat/completions", {**hi, **f}).status_code for f in (
        {"presence_penalty": 0.5}, {"frequency_penalty": -1}, {"repetition_penalty": 1.1},
        {"response_format": {"type": "json_object"}}, {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})]
    rejected.append(post(U + "/completions", {"prompt": "hi", "max_tokens": 2, "presence_penalty": 1}).status_code)
    accepted = [post(U + "/chat/completions", {**hi, **f}).status_code for f in (
        {"presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1}, {"response_format": {"type": "text"}})]
    check("unsupported penalties / response_format -> 400, neutral values -> 200", rejected == [400] * 6 and accepted == [200, 200],
          f"{rejected} {accepted}")

    # 8. multi-turn prefix reuse
    long_sys = {"role": "system", "content": "Reference: " + " ".join(f"fact {i} is {i * 7 % 13}." for i in range(2000))}
    m1 = [long_sys, {"role": "user", "content": "What is fact 12?"}]
    d1 = post(U + "/chat/completions", {"messages": m1, "temperature": 0, "max_tokens": 50, "chat_template_kwargs": {"enable_thinking": False}}).json()
    m2 = m1 + [{"role": "assistant", "content": d1["choices"][0]["message"]["content"]}, {"role": "user", "content": "And fact 13?"}]
    d2 = post(U + "/chat/completions", {"messages": m2, "temperature": 0, "max_tokens": 50, "chat_template_kwargs": {"enable_thinking": False}}).json()
    c2 = d2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check("multi-turn prefix reuse", c2 > 0.95 * d1["usage"]["prompt_tokens"], f"turn 2: {d2['usage']['prompt_tokens']} prompt tokens, {c2} cached, "
          f"TTFT {d2['timings']['ttft_s']:.3f}s (turn 1 {d1['timings']['ttft_s']:.2f}s)")

    m = requests.get(a.url + "/metrics").text
    check("/metrics", "colinfer_step_seconds_bucket" in m and "colinfer_ttft_seconds_count" in m)
    print("SERVER CHECK", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
