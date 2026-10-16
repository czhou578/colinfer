"""The local gateway for Claude Code (engine/server/gateway.py) between two stand-ins: a "server" and an "api.anthropic.com",
each a small HTTP server in a thread that records what it receives and answers with a JSON echo or a stream of events.
The tests check where each request goes, which headers and which system blocks reach each destination, that a stream
and an error body pass through as they are, and the answers the gateway gives itself."""
import http.server
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from engine.server import gateway as gw

SSE = [b'event: message_start\ndata: {"type":"message_start"}\n\n', b'event: ping\ndata: {"type":"ping"}\n\n',
       b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n',
       b'event: message_stop\ndata: {"type":"message_stop"}\n\n']


class Upstream:
    """A destination: it records (method, path, headers, body) per request and answers by path. /v1/messages with
    "stream": true sends SSE in four writes; otherwise a JSON echo of the body. /fail answers 400 with a fixed body."""

    def __init__(self, name):
        self.name, self.requests = name, []
        here = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _any(self):
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n) if n else b""
                here.requests.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "body": body})
                try:
                    j = json.loads(body) if body else None
                except ValueError:
                    j = None
                if self.path.startswith("/fail"):
                    out = b'{"type":"error","error":{"type":"invalid_request_error","message":"upstream says no"}}'
                    self.send_response(400)
                    self.send_header("content-type", "application/json")
                    self.send_header("x-should-retry", "false")
                    self.send_header("content-length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if isinstance(j, dict) and j.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("anthropic-ratelimit-unified-status", "allowed")
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    for ev in SSE:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(b"0\r\n\r\n")
                    return
                out = json.dumps({"from": here.name, "path": self.path, "body": j}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("x-should-retry", "false")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(out)
            do_GET = do_POST = do_HEAD = _any

            def log_message(self, *a):
                pass
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def last(self):
        return self.requests[-1]


@pytest.fixture(scope="module")
def ends():
    local, upstream = Upstream("local"), Upstream("anthropic")
    yield local, upstream
    local.srv.shutdown()
    upstream.srv.shutdown()


@pytest.fixture
def client(ends):
    local, upstream = ends
    local.requests.clear()
    upstream.requests.clear()
    app = gw.build_app(local.url, upstream.url)
    with TestClient(app) as c:
        yield c, local, upstream


HEADERS = {"authorization": "Bearer sk-ant-oat-secret", "x-api-key": "key-secret", "anthropic-version": "2023-06-01",
           "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14", "x-claude-code-session-id": "s1", "accept-encoding": "gzip"}
SYSTEM = [{"type": "text", "text": gw.ATTRIBUTION + ": cc_version=2.1.296; cch=abc"}, {"type": "text", "text": "You are Claude Code."}]


def body(model, **kw):
    return {"model": model, "max_tokens": 10, "system": SYSTEM, "messages": [{"role": "user", "content": "hi"}], **kw}


def test_a_local_model_goes_to_the_server_without_the_credential_or_the_attribution(client):
    c, local, upstream = client
    r = c.post("/v1/messages?beta=true", json=body("qwen3.8-27b"), headers=HEADERS)
    assert r.status_code == 200 and r.json()["from"] == "local" and not upstream.requests
    req = local.last()
    assert req["path"] == "/v1/messages?beta=true" and req["method"] == "POST"
    assert "authorization" not in req["headers"] and "x-api-key" not in req["headers"]
    assert req["headers"]["anthropic-version"] == "2023-06-01" and req["headers"]["anthropic-beta"] == HEADERS["anthropic-beta"]
    assert req["headers"]["x-claude-code-session-id"] == "s1" and req["headers"]["accept-encoding"] == "identity"
    sent = json.loads(req["body"])
    assert sent["system"] == SYSTEM[1:] and sent["model"] == "qwen3.8-27b" and sent["messages"] == body("x")["messages"]
    assert r.headers["x-should-retry"] == "false"  # response headers pass through


def test_a_claude_model_goes_to_anthropic_with_everything(client):
    c, local, upstream = client
    raw = json.dumps(body("claude-opus-5-5")).encode()
    r = c.post("/v1/messages?beta=true", content=raw, headers={**HEADERS, "content-type": "application/json"})
    assert r.status_code == 200 and r.json()["from"] == "anthropic" and not local.requests
    req = upstream.last()
    assert req["body"] == raw  # byte for byte: the attribution block and every field stay
    assert req["headers"]["authorization"] == HEADERS["authorization"] and req["headers"]["x-api-key"] == HEADERS["x-api-key"]
    assert req["headers"]["anthropic-beta"] == HEADERS["anthropic-beta"] and req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["headers"]["host"].startswith("127.0.0.1:") and req["headers"]["content-length"] == str(len(raw))


@pytest.mark.parametrize("model, where", [("nvidia/Qwen3.8-27B-NVFP4", "local"), ("qwen3.8-27b-committee", "local"), ("committee", "local"),
                                          ("claude-haiku-5-5", "anthropic"), ("claude-fable-5-1", "anthropic"), (None, "anthropic")])
def test_routing_by_model_name(client, model, where):
    c, local, upstream = client
    b = body(model) if model else {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/messages", json=b, headers=HEADERS).json()["from"] == where
    assert c.post("/v1/messages/count_tokens", json=b, headers=HEADERS).json()["from"] == where


def test_other_paths_and_unreadable_bodies_go_to_anthropic(client):
    c, local, upstream = client
    assert c.get("/v1/models?limit=1000", headers=HEADERS).json()["from"] == "anthropic"
    assert upstream.last()["path"] == "/v1/models?limit=1000"
    assert c.post("/v1/messages", content=b"not json {", headers={**HEADERS, "content-type": "application/json"}).json()["from"] == "anthropic"
    assert c.post("/v1/complete", json={"model": "qwen3.8-27b"}, headers=HEADERS).json()["from"] == "anthropic"  # only the messages paths route
    assert not local.requests


def test_a_stream_passes_through_in_order(client):
    c, local, upstream = client
    for model, dest in (("qwen3.8-27b", local), ("claude-opus-5-5", upstream)):
        chunks = []
        with c.stream("POST", "/v1/messages?beta=true", json=body(model, stream=True), headers=HEADERS) as r:
            assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
            assert r.headers["anthropic-ratelimit-unified-status"] == "allowed" and "content-length" not in r.headers
            for ch in r.iter_raw():
                chunks.append(ch)
        assert b"".join(chunks) == b"".join(SSE) and json.loads(dest.last()["body"])["stream"] is True


def test_an_error_body_passes_through_unchanged(client):
    c, local, upstream = client
    r = c.post("/fail", json={"model": "claude-opus-5-5"}, headers=HEADERS)
    assert r.status_code == 400 and r.json()["error"]["message"] == "upstream says no" and r.headers["x-should-retry"] == "false"


def test_a_server_that_is_down_gives_a_503_and_the_claude_rows_keep_working(ends):
    _, upstream = ends
    app = gw.build_app("http://127.0.0.1:9", upstream.url)  # port 9: nothing listens
    with TestClient(app) as c:
        r = c.post("/v1/messages", json=body("qwen3.8-27b"), headers=HEADERS)
        assert r.status_code == 503 and r.headers["retry-after"] == "10"
        e = r.json()
        assert e["type"] == "error" and e["error"]["type"] == "api_error" and "colinfer server" in e["error"]["message"]
        assert c.post("/v1/messages", json=body("claude-opus-5-5"), headers=HEADERS).json()["from"] == "anthropic"


def test_the_gateway_answers_the_probe_and_health_itself(client):
    c, local, upstream = client
    assert c.head("/api/hello").status_code == 200
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["local"] == local.url and h["local_models"] == ["qwen", "committee"]
    assert not local.requests and not upstream.requests


def test_keep_attribution_and_custom_local_models(ends):
    local, upstream = ends
    local.requests.clear()
    app = gw.build_app(local.url, upstream.url, local_models=("mymodel",), strip=False)
    with TestClient(app) as c:
        assert c.post("/v1/messages", json=body("qwen3.8-27b"), headers=HEADERS).json()["from"] == "anthropic"
        assert c.post("/v1/messages", json=body("MyModel-v2"), headers=HEADERS).json()["from"] == "local"
    assert json.loads(local.last()["body"])["system"] == SYSTEM  # the attribution block stays with --keep-attribution


def test_pure_functions():
    assert gw.is_local("qwen3.8-27b") and gw.is_local("QWEN") and gw.is_local("x-committee") and not gw.is_local("claude-opus-5-5")
    assert not gw.is_local(None) and not gw.is_local(3) and not gw.is_local("qwen", local_models=("", "committee"))
    b = {"system": SYSTEM, "model": "m"}
    assert gw.strip_attribution(b)["system"] == SYSTEM[1:] and b["system"] == SYSTEM  # a new body, the old one untouched
    assert gw.strip_attribution({"system": "a string"}) == {"system": "a string"}
    assert gw.strip_attribution({"system": SYSTEM[1:]}) == {"system": SYSTEM[1:]} and gw.strip_attribution({}) == {}


def test_main_rejects_an_empty_model_list(monkeypatch):
    assert gw.main(["--local-models", " , "]) == 2


def test_the_client_the_app_uses_is_shared(ends):
    local, upstream = ends
    own = httpx.AsyncClient()
    app = gw.build_app(local.url, upstream.url, client=own)
    assert app.state.http is own
