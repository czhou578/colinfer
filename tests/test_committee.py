"""Committee mode (engine/server/committee.py) through the HTTP layer, on the fake engine thread of tests/test_api.py.
A worker that answers each request with the next scripted reply tells the members and the reconcile pass apart, in the
prompts it received and in the reply the client gets. No GPU and no checkpoint."""
import json

import pytest
from fastapi.testclient import TestClient

from engine.server import committee as cm
from engine.server.api import build_app
from engine.server.worker import ServerConfig
from tests.test_api import CHAT, USER, VOCAB, CharTokenizer, FakeWorker, anth_events, sse_data
from tools import committee_ab as ab


class ScriptedWorker(FakeWorker):
    """Each submitted request gets the next reply of `replies` (the last one again past the end). fail_at: the index
    of a request that fails with a 500 instead."""

    def __init__(self, replies, fail_at=None):
        super().__init__()
        self.replies, self.fail_at = [CharTokenizer().encode(r) for r in replies], fail_at

    def submit(self, req):
        i = len(self.submitted)
        self.reply = self.replies[min(i, len(self.replies) - 1)]
        self.error = ("member failed", 500) if i == self.fail_at else None
        super().submit(req)


def client(replies=("A1<|im_end|>", "A2<|im_end|>", "A3<|im_end|>", "final<|im_end|>"), fail_at=None, committee=None, size=None):
    w = ScriptedWorker(list(replies), fail_at)
    w.cfg = ServerConfig(model="fake", max_seq_len=4096, max_request_seconds=0.0, committee_size=size)
    app = build_app(w, CharTokenizer(), "fake", {}, {"vocab_size": VOCAB}, None, None, committee=committee)
    return TestClient(app, raise_server_exceptions=False), w


def prompts(w):
    return [CharTokenizer().decode(r.prompt) for r in w.submitted]


# ------------------------------------------------------------------------------------------------------ the chat API
def test_chat_committee_members_then_reconcile():
    c, w = client()
    r = c.post(CHAT, json={"model": "fake-committee", "messages": USER, "enable_thinking": False}).json()
    assert r["model"] == "fake-committee" and r["choices"][0]["message"]["content"] == "final"
    assert r["committee"] == {"roles": list(cm.DEFAULT_ROLES), "answers": ["A1", "A2", "A3"]}
    ps = prompts(w)
    assert len(ps) == 4
    for p, role in zip(ps, cm.DEFAULT_ROLES):  # each member: the request, its role appended to the user message, nothing else
        assert f"<|im_start|>user\nhi\n\n{cm.ROLE_TAG} {role}<|im_end|>" in p and "--- Answer" not in p
    rec = ps[3]
    assert "<|im_start|>user\nhi\n\n" + cm.RECONCILE.replace("{n}", "3") in rec
    assert "--- Answer 1 of 3 ---\nA1\n\n--- Answer 2 of 3 ---\nA2\n\n--- Answer 3 of 3 ---\nA3\n\n--- End of the answers ---<|im_end|>" in rec
    assert cm.ROLE_TAG not in rec


def test_chat_committee_stream_gives_the_reconciled_reply_only():
    c, w = client()
    chunks = sse_data(c.post(CHAT, json={"model": "fake-committee", "messages": USER, "stream": True, "enable_thinking": False,
                                         "stream_options": {"include_usage": True}}))
    deltas = [ch["choices"][0]["delta"] for ch in chunks if ch["choices"]]
    assert "".join(d.get("content", "") for d in deltas) == "final" and len(w.submitted) == 4
    assert chunks[-1]["usage"]["completion_tokens"] == len(CharTokenizer().encode("final<|im_end|>"))
    assert all("committee" not in ch for ch in chunks)


def test_seeds_differ_per_member_and_the_reply_keeps_the_request_seed():
    c, w = client()
    assert c.post(CHAT, json={"model": "fake-committee", "messages": USER, "seed": 7}).status_code == 200
    assert [r.seed for r in w.submitted] == [7, 8, 9, 7]
    c, w = client()
    assert c.post(CHAT, json={"model": "fake-committee", "messages": USER}).status_code == 200
    assert len({r.seed for r in w.submitted}) == 4  # no seed: each request draws its own


def test_other_model_names_run_one_request():
    for name in ("fake", "committee-of-one", "x-committee-y", None):
        c, w = client()
        body = {"messages": USER, "enable_thinking": False}
        if name is not None:
            body["model"] = name
        r = c.post(CHAT, json=body).json()
        assert len(w.submitted) == 1 and r["choices"][0]["message"]["content"] == "A1" and "committee" not in r, name


def test_committee_off_by_size_zero():
    c, w = client(size=0)
    r = c.post(CHAT, json={"model": "fake-committee", "messages": USER, "enable_thinking": False}).json()
    assert len(w.submitted) == 1 and "committee" not in r
    assert [m["id"] for m in c.get("/v1/models").json()["data"]] == ["fake"]


def test_models_lists_the_alias():
    c, _ = client()
    data = c.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["fake", "fake-committee"] and data[1]["committee"]["roles"] == list(cm.DEFAULT_ROLES)


def test_role_after_a_tool_result_is_a_new_user_turn():
    c, w = client()
    msgs = USER + [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                                       "function": {"name": "f", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "call_1", "content": "42"}]
    assert c.post(CHAT, json={"model": "fake-committee", "messages": msgs, "enable_thinking": False}).status_code == 200
    p = prompts(w)[0]
    assert f"<|im_start|>tool\n42<|im_end|>\n<|im_start|>user\n{cm.ROLE_TAG} {cm.DEFAULT_ROLES[0]}<|im_end|>" in p


def test_a_member_failure_fails_the_request_in_each_format():
    c, w = client(fail_at=1)
    r = c.post(CHAT, json={"model": "fake-committee", "messages": USER})
    assert r.status_code == 500 and r.json()["error"]["message"] == "member failed" and len(w.submitted) == 2
    c, w = client(fail_at=1)
    ev = sse_data(c.post(CHAT, json={"model": "fake-committee", "messages": USER, "stream": True}))
    assert ev[-1]["error"]["message"] == "member failed" and len(w.submitted) == 2
    c, w = client(fail_at=1)
    r = c.post("/v1/messages", json={"model": "fake-committee", "max_tokens": 10, "messages": USER})
    assert r.status_code == 500 and r.json()["error"]["type"] == "api_error"
    c, w = client(fail_at=1)
    ev = anth_events(c.post("/v1/messages", json={"model": "fake-committee", "max_tokens": 10, "messages": USER, "stream": True}))
    assert ev[0][0] == "message_start" and ev[-1] == ("error", {"type": "error", "error": {"type": "api_error", "message": "member failed"}})


def test_bad_values_are_a_400_before_any_member_runs():
    c, w = client()
    r = c.post(CHAT, json={"model": "fake-committee", "messages": USER, "max_tokens": 0})
    assert r.status_code == 400 and not w.submitted
    r = c.post(CHAT, json={"model": "fake-committee", "messages": USER, "max_tokens": 0, "stream": True})
    assert r.status_code == 400 and not w.submitted


def test_custom_committee_and_roles_file(tmp_path):
    two = cm.Committee(("Be brief.", "Be thorough."), "Merge the {n} drafts.")
    c, w = client(replies=("A1<|im_end|>", "A2<|im_end|>", "final<|im_end|>"), committee=two)
    r = c.post(CHAT, json={"model": "committee", "messages": USER, "enable_thinking": False}).json()
    assert r["committee"]["answers"] == ["A1", "A2"] and r["choices"][0]["message"]["content"] == "final"
    assert "Merge the 2 drafts.\n\n--- Answer 1 of 2 ---" in prompts(w)[2]
    f = tmp_path / "roles.json"
    f.write_text(json.dumps({"roles": ["Be brief.", "Be thorough."], "reconcile": "Merge the {n} drafts."}))
    assert cm.Committee.from_config(None, str(f)) == two
    assert cm.Committee.from_config(4, str(f)).roles == ("Be brief.", "Be thorough.", cm.EXTRA_ROLE, cm.EXTRA_ROLE)
    assert cm.Committee.from_config(1, str(f)).roles == ("Be brief.",)
    assert cm.Committee.from_config(0, str(f)) is None and cm.Committee.from_config(None, None) == cm.Committee()
    assert cm.Committee.from_config(5, None).roles == cm.DEFAULT_ROLES + (cm.EXTRA_ROLE, cm.EXTRA_ROLE)
    for bad in ({"roles": []}, {"roles": "x"}, {"roles": [1]}, {"roles": ["a"], "reconcile": 3}, []):
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            cm.Committee.from_config(None, str(f))


# ------------------------------------------------------------------------------------------------- the Anthropic API
def test_messages_committee_without_and_with_stream():
    body = {"model": "fake-committee", "max_tokens": 50, "messages": [{"role": "user", "content": "design?"}]}
    c, w = client()
    one = c.post("/v1/messages", json=body).json()
    assert one["model"] == "fake-committee" and one["content"] == [{"type": "text", "text": "final"}] and one["stop_reason"] == "end_turn"
    ps = prompts(w)
    assert len(ps) == 4 and f"design?\n\n{cm.ROLE_TAG} {cm.DEFAULT_ROLES[2]}" in ps[2] and "--- Answer 3 of 3 ---\nA3" in ps[3]
    c, w = client()
    ev = anth_events(c.post("/v1/messages", json={**body, "stream": True}))
    assert ev[0][0] == "message_start" and [e for e, _ in ev][-2:] == ["message_delta", "message_stop"]
    assert "".join(d["delta"].get("text", "") for e, d in ev if e == "content_block_delta") == "final" and len(w.submitted) == 4


def test_count_tokens_counts_the_plain_prompt():
    c, w = client()
    body = {"model": "fake-committee", "messages": [{"role": "user", "content": "hi"}]}
    n = c.post("/v1/messages/count_tokens", json=body).json()["input_tokens"]
    assert n == c.post("/v1/messages/count_tokens", json={**body, "model": "fake"}).json()["input_tokens"] and not w.submitted


# ------------------------------------------------------------------------------------------------- the pure functions
def test_is_committee():
    assert cm.is_committee("qwen3.8-27b-committee") and cm.is_committee("committee")
    assert not cm.is_committee("-committee") and not cm.is_committee("qwen") and not cm.is_committee(None) and not cm.is_committee(3)


def test_messages_are_appended_without_touching_the_prefix():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    out = cm.member_messages(msgs, "R")
    assert out[0] is not msgs[0] and out[0] == msgs[0] and out[1]["content"] == f"q\n\n{cm.ROLE_TAG} R" and msgs[1]["content"] == "q"
    parts = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    out = cm.member_messages([{"role": "user", "content": parts}], "R")
    assert out[0]["content"] == parts + [{"type": "text", "text": f"{cm.ROLE_TAG} R"}] and len(parts) == 2
    assert cm.member_messages([{"role": "user", "content": None}], "R")[0]["content"] == f"{cm.ROLE_TAG} R"
    assert cm.member_messages([{"role": "tool", "content": "42"}], "R")[-1] == {"role": "user", "content": f"{cm.ROLE_TAG} R"}
    rec = cm.reconcile_messages(msgs, ["a", " b "], "Merge {n}.")[1]["content"]
    assert rec == "q\n\nMerge 2.\n\n--- Answer 1 of 2 ---\na\n\n--- Answer 2 of 2 ---\nb\n\n--- End of the answers ---"


def test_answer_text():
    assert cm.answer_text({"content": " text "}) == "text"
    assert cm.answer_text({"content": None, "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}) == '[tool call] f({"a": 1})'
    assert cm.answer_text({"content": "x", "tool_calls": [{"function": {"name": "f"}}]}) == "x\n\n[tool call] f()"
    assert cm.answer_text({"content": ""}) == "(no answer)"


# --------------------------------------------------------------------------------------------------- the A/B tool
def test_ab_prompts_and_pairs():
    assert ab.parse_prompts("# a comment\n\nfirst question\nsecond line\n\n\nsecond question\n") == ["first question\nsecond line", "second question"]
    results = [{"prompt": "q1", "single": {"text": "s1", "seconds": 1.0}, "committee": {"text": "c1", "seconds": 4.0}},
               {"prompt": "q2", "single": {"text": "s2", "seconds": 2.0}, "committee": {"text": "c2", "seconds": 5.0}}]
    pairs = ab.make_pairs(results, seed=1)
    for p, r in zip(pairs, results):
        assert {p["A"], p["B"]} == {r["single"]["text"], r["committee"]["text"]} and p[p["committee"]] == r["committee"]["text"]
    md = ab.render_pairs(pairs)
    assert "## 1. q1" in md and "### A" in md and "### B" in md and "committee" not in md.lower().replace("committee mode", "")
    assert ab.summary(results) == "single 1.5 s, committee 4.5 s per answer (3.0x)"
