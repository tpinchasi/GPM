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

**A model may have several copies** — one per card on a machine with more than one (D107). Each
request goes to the copy with the fewest requests in flight from this process, and a copy that
refuses the connection is passed over for the next. The machine counts as serving a model only
when every copy answers: its worker count was set for all of its cards, and a host running on
half of them is not what was bought.

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
    """Which engines on this machine serve which model — one URL per copy.

    Read from a file the machine's own start-up wrote. Re-read when it changes, so an engine
    added or moved does not need this process restarted — and because a start-up that writes the
    file after this starts is the ordinary case, not an error. A model maps to one URL or to a
    list of them, one per copy.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._stamp: Optional[float] = None
        self._by_model: dict[str, list[str]] = {}

    def by_model(self) -> dict[str, list[str]]:
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return self._by_model
        if stamp != self._stamp:
            self._by_model = self._read()
            self._stamp = stamp
        return self._by_model

    def _read(self) -> dict[str, list[str]]:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("could not read the upstream map at %s: %s", self.path, exc)
            return self._by_model
        if not isinstance(raw, dict):
            log.warning("the upstream map at %s is not an object", self.path)
            return self._by_model
        found = {}
        for model, urls in raw.items():
            urls = [urls] if isinstance(urls, str) else urls
            if (model and isinstance(model, str) and isinstance(urls, list) and urls
                    and all(isinstance(u, str) and u.startswith("http") for u in urls)):
                found[model] = [u.rstrip("/") for u in urls]
            else:
                log.warning("ignoring upstream entry %r: not a model and its URLs", model)
        return found

    def urls_for(self, model: Optional[str]) -> list[str]:
        return self.by_model().get(model, []) if model else []


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
    #: Requests this process has open to each copy — how the least busy one is chosen.
    in_flight: dict[str, int] = {}

    async def unhealthy_copies(model: str, urls: list[str]) -> list[str]:
        found = []
        for url in urls:
            try:
                answer = await http.get(f"{url}/health", timeout=HEALTH_TIMEOUT_S)
                if answer.status_code != 200:
                    found.append(f"{model} at {url}: /health returned {answer.status_code}")
            except httpx.HTTPError as exc:
                found.append(f"{model} at {url}: {exc or type(exc).__name__}")
        return found

    def least_busy_first(urls: list[str]) -> list[str]:
        # Stable, so copies tied at zero are taken in the launcher's order.
        return sorted(urls, key=lambda url: in_flight.get(url, 0))

    async def health(request: Request) -> Response:
        """200 only when **every** engine behind this answers — every copy of every model.

        A machine serving two models of three is not a machine the pool can treat as ready: a
        request for the missing one would be refused after being routed here. Nor is one serving
        a model on one card of two: its worker count was set for both. Saying so plainly lets the
        pool's own readiness rule do its job without knowing this process exists.
        """
        by_model = upstreams.by_model()
        if not by_model:
            return JSONResponse({"error": "no upstreams are configured yet"}, status_code=503)
        unhealthy = []
        for model, urls in sorted(by_model.items()):
            unhealthy += await unhealthy_copies(model, urls)
        if unhealthy:
            return JSONResponse({"error": "; ".join(unhealthy)}, status_code=503)
        return Response(status_code=200)

    async def models(request: Request) -> Response:
        """The models this machine is serving **now**, in the engine's own shape.

        Only engines that answer are listed. The pool reads this as "what is resident" and calls
        the host ready when its set is here — so listing an engine still loading its weights
        would have the pool route requests to it minutes early. The map says what *will* be
        served; this says what *is*. A model with copies is listed once every copy answers.
        """
        served = []
        for model, urls in sorted(upstreams.by_model().items()):
            if not await unhealthy_copies(model, urls):
                served.append(model)
        return JSONResponse({
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "gpm"} for model in served],
        })

    async def metrics(request: Request) -> Response:
        """Every engine's metrics, one after another, each labelled with its model and URL.

        Concatenated rather than summed: the pool reads running, waiting and cache use, and a
        sum of cache fractions across engines would mean nothing. Whoever reads them aggregates
        knowing what they are.
        """
        chunks = []
        for model, urls in sorted(upstreams.by_model().items()):
            for url in urls:
                try:
                    answer = await http.get(f"{url}/metrics", timeout=HEALTH_TIMEOUT_S)
                    if answer.status_code == 200:
                        chunks.append(f"# gpm-proxy upstream {model} {url}\n{answer.text}")
                except httpx.HTTPError:
                    continue  # one engine that will not say must not hide the others
        return Response("\n".join(chunks), media_type="text/plain; version=0.0.4")

    async def forward(request: Request) -> Response:
        body = await request.body()
        model = _model_of(body)
        urls = upstreams.urls_for(model)
        if not urls:
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
        # Least busy first; a copy that refuses the connection never saw the request, so the
        # next is tried. Anything after the connection is the copy's answer and is passed on.
        candidates = least_busy_first(urls)

        if not _wants_stream(body):
            refused: Optional[httpx.HTTPError] = None
            for url in candidates:
                in_flight[url] = in_flight.get(url, 0) + 1
                try:
                    answer = await http.post(f"{url}{request.url.path}", content=body, headers=headers)
                except httpx.ConnectError as exc:
                    refused = exc
                    continue
                except httpx.HTTPError as exc:
                    refused = exc
                    break
                finally:
                    in_flight[url] -= 1
                return Response(
                    content=answer.content,
                    status_code=answer.status_code,
                    media_type=answer.headers.get("content-type"),
                )
            return JSONResponse(
                {"error": {"message": f"the engine for {model!r} did not answer: {refused}",
                           "type": "upstream_unavailable"}},
                status_code=502,
            )

        # Streaming: the engine's answer is opened before this one is, so its status and type
        # are the ones the client gets — a refusal is not a 200 — and it is closed when the
        # client goes, so an abandoned request stops the work upstream rather than generating
        # into nothing on a machine being paid for.
        upstream: Optional[httpx.Response] = None
        chosen = None
        refused = None
        for url in candidates:
            in_flight[url] = in_flight.get(url, 0) + 1
            try:
                upstream = await http.send(
                    http.build_request("POST", f"{url}{request.url.path}", content=body, headers=headers), stream=True)
                chosen = url
                break
            except httpx.ConnectError as exc:
                log.warning("the copy of %s at %s refused the connection: %s", model, url, exc)
                refused = exc
            except httpx.HTTPError as exc:
                refused = exc
                in_flight[url] -= 1
                break
            in_flight[url] -= 1
        if upstream is None:
            return JSONResponse(
                {"error": {"message": f"the engine for {model!r} did not answer: {refused}",
                           "type": "upstream_unavailable"}},
                status_code=502,
            )

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                # Raised, not swallowed: a clean end would pass a cut answer as a whole one, and
                # the pool would deliver it rather than run it again (D62).
                log.warning("stream from %s for %s broke: %s", chosen, model, exc)
                raise
            finally:
                await upstream.aclose()
                in_flight[chosen] -= 1

        return StreamingResponse(relay(), status_code=upstream.status_code,
                                 media_type=upstream.headers.get("content-type", "text/event-stream"))

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
    parser.add_argument("--upstreams", required=True, help="JSON file mapping model name to its engine URL, or a list of them (one per copy)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
