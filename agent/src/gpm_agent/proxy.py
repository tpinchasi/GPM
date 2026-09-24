"""One endpoint in front of several engine processes on one machine (D96).

Some engines serve exactly one model per process. A machine with memory to spare can run
several, but the pool dials **one** URL per host — so something on the machine has to choose
between them by the model a request names. That is all this does.

**It is not the agent.** The agent's protocol is a closed list of verbs and it is never on the
request path; this is a separate process, on a separate port, that carries inference traffic and
nothing else. They ship in the same archive because that archive is already on every host the
pool creates, not because they are the same thing.

**It routes, it does not translate.** A request arrives in the engine's own API and leaves in
it, byte for byte, with only the upstream chosen from the `model` field. The same rule the pool
itself obeys, for the same reason: translation is where tool-calling and structured-output
fidelity get lost.

Started by the machine's own start-up command, which also starts the engines and writes the map
of model to upstream. The pool names a model; it never names a port, a path or a command.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("gpm.proxy")

#: How long an upstream has to answer before it is called unhealthy. Generous: an engine that
#: is loading weights is not a broken one.
HEALTH_TIMEOUT_S = 5.0

#: Long enough for a whole generation at a large context; the upstream's own limits bound it.
INFERENCE_TIMEOUT_S = 3600.0


class Upstreams:
    """Which engine on this machine serves which model.

    Read from a file the machine's own start-up wrote. Re-read when it changes, so an engine
    added or moved does not need this process restarted — and because a start-up that writes the
    file after this starts is the ordinary case, not an error.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._stamp: Optional[float] = None
        self._by_model: dict[str, str] = {}

    def by_model(self) -> dict[str, str]:
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return self._by_model
        if stamp != self._stamp:
            self._by_model = self._read()
            self._stamp = stamp
        return self._by_model

    def _read(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("could not read the upstream map at %s: %s", self.path, exc)
            return self._by_model
        if not isinstance(raw, dict):
            log.warning("the upstream map at %s is not an object", self.path)
            return self._by_model
        found = {}
        for model, url in raw.items():
            if model and isinstance(model, str) and isinstance(url, str) and url.startswith("http"):
                found[model] = url.rstrip("/")
            else:
                log.warning("ignoring upstream entry %r: not a model and a URL", model)
        return found

    def url_for(self, model: Optional[str]) -> Optional[str]:
        return self.by_model().get(model) if model else None


def _model_of(body: bytes) -> Optional[str]:
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    model = parsed.get("model")
    return model if isinstance(model, str) else None


def _wants_stream(body: bytes) -> bool:
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    return bool(isinstance(parsed, dict) and parsed.get("stream", False))


def create_app(upstream_map: str | Path, *, client: Optional[httpx.AsyncClient] = None) -> Starlette:
    upstreams = Upstreams(upstream_map)
    http = client or httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_S, follow_redirects=False)

    async def health(request: Request) -> Response:
        """200 only when **every** engine behind this answers.

        A machine serving two models of three is not a machine the pool can treat as ready: a
        request for the missing one would be refused after being routed here. Saying so plainly
        lets the pool's own readiness rule do its job without knowing this process exists.
        """
        by_model = upstreams.by_model()
        if not by_model:
            return JSONResponse({"error": "no upstreams are configured yet"}, status_code=503)
        unhealthy = []
        for model, url in sorted(by_model.items()):
            try:
                answer = await http.get(f"{url}/health", timeout=HEALTH_TIMEOUT_S)
                if answer.status_code != 200:
                    unhealthy.append(f"{model}: /health returned {answer.status_code}")
            except httpx.HTTPError as exc:
                unhealthy.append(f"{model}: {exc or type(exc).__name__}")
        if unhealthy:
            return JSONResponse({"error": "; ".join(unhealthy)}, status_code=503)
        return Response(status_code=200)

    async def models(request: Request) -> Response:
        """Every model this machine serves, in the engine's own shape — so the pool sees one
        host holding a set, which is exactly what it is."""
        served = sorted(upstreams.by_model())
        return JSONResponse({
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "gpm"} for model in served],
        })

    async def metrics(request: Request) -> Response:
        """Every engine's metrics, one after another, each labelled with its model.

        Concatenated rather than summed: the pool reads running, waiting and cache use, and a
        sum of cache fractions across engines would mean nothing. Whoever reads them aggregates
        knowing what they are.
        """
        chunks = []
        for model, url in sorted(upstreams.by_model().items()):
            try:
                answer = await http.get(f"{url}/metrics", timeout=HEALTH_TIMEOUT_S)
                if answer.status_code == 200:
                    chunks.append(f"# gpm-proxy upstream {model}\n{answer.text}")
            except httpx.HTTPError:
                continue  # one engine that will not say must not hide the others
        return Response("\n".join(chunks), media_type="text/plain; version=0.0.4")

    async def forward(request: Request) -> Response:
        body = await request.body()
        model = _model_of(body)
        url = upstreams.url_for(model)
        if url is None:
            known = ", ".join(sorted(upstreams.by_model())) or "none"
            return JSONResponse(
                {"error": {
                    "message": f"this host serves no model named {model!r} (it serves: {known})",
                    "type": "model_not_found",
                }},
                status_code=404,
            )

        # Hop-by-hop headers are the connection's, not the message's, and forwarding them
        # breaks the connection this process owns.
        headers = {
            name: value for name, value in request.headers.items()
            if name.lower() not in ("host", "content-length", "connection", "transfer-encoding")
        }
        target = f"{url}{request.url.path}"

        if not _wants_stream(body):
            try:
                answer = await http.post(target, content=body, headers=headers)
            except httpx.HTTPError as exc:
                return JSONResponse(
                    {"error": {"message": f"the engine for {model!r} did not answer: {exc}",
                               "type": "upstream_unavailable"}},
                    status_code=502,
                )
            return Response(
                content=answer.content,
                status_code=answer.status_code,
                media_type=answer.headers.get("content-type"),
            )

        # Streaming: opened here and closed when the client goes, so an abandoned request stops
        # the work upstream rather than generating into nothing on a machine being paid for.
        async def relay():
            try:
                async with http.stream("POST", target, content=body, headers=headers) as upstream:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            except httpx.HTTPError as exc:
                log.warning("stream to %s for %s ended: %s", target, model, exc)

        return StreamingResponse(relay(), media_type="text/event-stream")

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # Only what this process opened: a client handed in belongs to the caller.
            if client is None:
                await http.aclose()

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/{path:path}", forward, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def serve(upstream_map: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run it. Loopback by default: the pool reaches this through a forward into the machine,
    so binding anything wider only exposes it (D77)."""
    import uvicorn

    uvicorn.run(create_app(upstream_map), host=host, port=port, log_level="warning")


def add_arguments(parser: Any) -> None:
    parser.add_argument("--upstreams", required=True, help="JSON file mapping model name to engine URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
