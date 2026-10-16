"""A local gateway for Claude Code: one base URL, two destinations.

Claude Code sends every model request to ANTHROPIC_BASE_URL. This gateway listens there and routes each request by its
model name: a local model (a name that contains one of --local-models, by default "qwen" and "committee") goes to the
colinfer server, and every other model goes to api.anthropic.com. Thus the model picker of Claude Code holds Claude
rows and Qwen rows at once (docs/claude_code.md, "Claude and Qwen in one picker").

What it does with a request:
  * It reads the model name from the JSON body of /v1/messages and /v1/messages/count_tokens. Other paths, and bodies
    it cannot read, go to api.anthropic.com.
  * To api.anthropic.com it forwards the request as it came: method, path, query, body and headers, including the
    credential (Authorization / x-api-key: the claude.ai login token of Claude Code) and anthropic-beta, which carries
    the OAuth capability. It drops only the hop-by-hop headers.
  * To the server it forwards the request without the credential headers, so the login token never reaches the local
    process, and without the attribution block that Claude Code puts first in the system prompt (api.anthropic.com strips
    that block itself; the server would see a different system prompt per session and lose its prefix cache).
  * It streams the response as the bytes arrive, status and headers included, so a stream and its ping events reach
    Claude Code at once. It never buffers, and it never rewrites a response or an error body.
  * It answers HEAD /api/hello (the connection probe of Claude Code) and GET /health itself.
  * When the server does not answer, a local request gets a 503 in the Anthropic error format, with retry-after. The
    Claude rows keep working.
It logs one line per request: method, path, model, destination, status and time. It logs no header and no body.

   uv run python -m engine.server.gateway                      # 127.0.0.1:8010 -> server 127.0.0.1:8000, else api.anthropic.com
   uv run python -m engine.server.gateway --port 8010 --local http://127.0.0.1:8000 --local-models qwen,committee

Anthropic does not support non-Claude models through a gateway. With only ANTHROPIC_BASE_URL set, Claude Code keeps
the claude.ai login as its credential, and the Claude rows bill to the subscription as before.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = "https://api.anthropic.com"
LOCAL = "http://127.0.0.1:8000"
LOCAL_MODELS = ("qwen", "committee")
ATTRIBUTION = "x-anthropic-billing-header"  # the first system block of Claude Code starts with this
# headers that describe one hop, not the request: they are not forwarded. accept-encoding is replaced by identity, so the
# upstream sends the bytes uncompressed and the gateway passes them through as they are.
HOP = frozenset(("host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
                 "proxy-connection", "proxy-authorization", "accept-encoding"))
CREDENTIALS = frozenset(("authorization", "x-api-key"))
RESPONSE_HOP = frozenset(("content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"))
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

log = logging.getLogger("gateway")


def is_local(model, local_models=LOCAL_MODELS) -> bool:
    """Whether a model name is the server's: it contains one of the local model markers (case does not matter)."""
    if not isinstance(model, str):
        return False
    m = model.lower()
    return any(x and x.lower() in m for x in local_models)


def strip_attribution(body: dict) -> dict:
    """The body without the attribution block, when the system prompt is a list whose first block starts with it."""
    system = body.get("system")
    if isinstance(system, list) and system and isinstance(system[0], dict) and str(system[0].get("text", "")).startswith(ATTRIBUTION):
        return {**body, "system": system[1:]}
    return body


def error_body(message: str) -> dict:
    return {"type": "error", "error": {"type": "api_error", "message": message}}


def build_app(local: str = LOCAL, upstream: str = UPSTREAM, local_models=LOCAL_MODELS, strip: bool = True,
              connect_timeout: float = 10.0, client: httpx.AsyncClient | None = None) -> FastAPI:
    """The gateway app. client: the HTTP client for both destinations (a test passes its own)."""
    app = FastAPI(title="colinfer gateway")
    local, upstream = local.rstrip("/"), upstream.rstrip("/")
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(connect=connect_timeout, read=None, write=120.0, pool=None), follow_redirects=False)
    app.state.local, app.state.upstream, app.state.http = local, upstream, http

    @app.get("/health")
    async def health():
        return {"status": "ok", "local": local, "upstream": upstream, "local_models": list(local_models)}

    @app.head("/api/hello")
    async def hello():  # the connection probe of Claude Code: answered here, not forwarded
        return Response(status_code=200)

    @app.api_route("/{path:path}", methods=METHODS)
    async def relay(request: Request, path: str):
        body = await request.body()
        model, parsed = None, None
        if body and path.startswith("v1/messages"):
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                model = parsed.get("model")
        to_local = is_local(model, local_models)
        dest, base = ("local", local) if to_local else ("upstream", upstream)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP and not (to_local and k.lower() in CREDENTIALS)}
        headers["accept-encoding"] = "identity"
        if to_local and strip and isinstance(parsed, dict):
            body = json.dumps(strip_attribution(parsed), ensure_ascii=False).encode()
        url = f"{base}/{path}" + (f"?{request.url.query}" if request.url.query else "")
        t0 = time.monotonic()
        try:
            resp = await http.send(http.build_request(request.method, url, headers=headers, content=body), stream=True)
        except httpx.HTTPError as e:
            log.info("%s /%s model=%s -> %s unreachable (%s)", request.method, path, model, dest, type(e).__name__)
            where = "the colinfer server" if to_local else "api.anthropic.com"
            return JSONResponse(error_body(f"{where} at {base} does not answer: {type(e).__name__}"), status_code=503, headers={"retry-after": "10"})
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESPONSE_HOP}

        async def stream():
            n = 0
            try:
                async for chunk in resp.aiter_raw():
                    n += len(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                log.info("%s /%s model=%s -> %s %d, %d bytes, %.1fs", request.method, path, model, dest, resp.status_code, n, time.monotonic() - t0)
        if request.method == "HEAD":
            await resp.aclose()
            return Response(status_code=resp.status_code, headers=out_headers)
        return StreamingResponse(stream(), status_code=resp.status_code, headers=out_headers)

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="a local gateway for Claude Code: local models to the colinfer server, the rest to api.anthropic.com")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--local", default=LOCAL, help="the colinfer server")
    ap.add_argument("--local-models", default=",".join(LOCAL_MODELS), metavar="A,B",
                    help="a model whose name contains one of these goes to the server (case does not matter)")
    ap.add_argument("--upstream", default=UPSTREAM, help="where every other model goes")
    ap.add_argument("--keep-attribution", action="store_true", help="forward the attribution block of the system prompt to the server too")
    a = ap.parse_args(argv)
    local_models = tuple(x.strip() for x in a.local_models.split(",") if x.strip())
    if not local_models:
        print("--local-models needs at least one name", flush=True)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its own line per request would repeat ours, with the URL
    app = build_app(a.local, a.upstream, local_models, strip=not a.keep_attribution)
    log.info("gateway on http://%s:%d: %s -> %s, the rest -> %s", a.host, a.port, " / ".join(local_models), a.local, a.upstream)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
