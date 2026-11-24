"""One endpoint in front of several engine processes on one machine (D96).

Some engines serve exactly one model per process. A machine with memory to spare can run
several, but the pool dials **one** URL per host — so something on the machine has to choose
between them by the model a request names. That is all this does.

**It is not the agent.** The agent's protocol is a closed list of verbs and it is never on the
request path; this is a separate process, on a separate port, that carries inference traffic and
nothing else. They ship in the same archive because that archive is already on every host the
pool creates, not because they are the same thing.

**It routes, it does not translate.** A request arrives in the engine's own API and leaves in
it, byte for byte, with only the upstream chosen. The same rule the pool itself obeys, for the
same reason: translation is where tool-calling and structured-output fidelity get lost. The
pool's router names the build it chose in the `X-GPM-Model` header, so an upload — a form
carrying a recording — is routed without being parsed (D144); the pool's headers end here and
never reach an engine. A request without that header, from an older router, is routed by its
JSON body's `model` field.

**Only inference is forwarded.** A closed list of paths, each allowed by its exact name and never
by prefix (D142): beside them, engines serve routes that load, unload or download models and
adapters, and no request reaches those through here. Anything else is refused before an engine
hears of it. An error answer is fixed text — never the model, header or path a caller sent, nor
an exception's text, since an answer can travel back through the pool to an application; those
go to this process's log.

**A model may have several copies** — one per card on a machine with more than one (D107). Each
request goes to the copy with the fewest requests in flight from this process, and a copy that
refuses the connection is passed over for the next. The machine counts as serving a model only
when every copy answers: its worker count was set for all of its cards, and a host running on
half of them is not what was bought.

Started by the machine's own start-up command, which also starts the engines and writes the map
of model to upstream. The pool names a model; it never names a port, a path or a command.

**In front of one Ollama server per card** (`--engine ollama`, D190): each upstream is a whole
Ollama serving every model on the machine's disk, pinned to its own card. A request goes to the
least busy server, whatever model it names. Ollama's own inference paths are forwarded beside
the OpenAI- and Anthropic-shaped ones, from a closed list of their own; its management paths —
pull, delete, create, copy, push, blobs — never are: the agent pulls and holds through each
server directly. What the pool and the agent *read* through this port is answered for the
machine as a whole (D192): `/api/version` and `/api/tags` only when every server answers, and
`/api/ps` listing a model only when every server holds it, each entry the one with the least of
it on its card, so a copy put on the processor is seen. `/api/show` asks the first server: the
model's files are the same for all of them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

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

#: The paths an engine is asked inference on — the only ones forwarded, each by its exact name
#: (D142). Never a prefix: beside these, engines serve routes that load and unload models and
#: adapters (vLLM's `/v1/load_lora_adapter`, whisper.cpp's `/load`, which takes a file path), and
#: a request must not reach them. Adding one is a decision, as adding a kind is.
FORWARDED_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/rerank",
})

#: Ollama's own paths an engine is asked inference on, and the OpenAI- and Anthropic-shaped ones
#: it serves too — the only ones forwarded in front of Ollama servers (D190), each by its exact
#: name. Its others — `/api/pull`, `/api/delete`, `/api/create`, `/api/copy`, `/api/push`,
#: `/api/blobs/…` — change what is on the machine and are never forwarded. Adding one is a
#: decision.
OLLAMA_FORWARDED_PATHS = frozenset({
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/api/embeddings",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/messages",
})

#: The engines this can stand in front of: one process per model (and per copy) — vLLM's shape —
#: or whole Ollama servers, one per card.
ENGINES = ("vllm", "ollama")

#: `POST /api/show` asks for a model's details by name: a small body, bounded far below the rest.
SHOW_BODY_BYTES = 65_536

#: Where the pool's router names the build it chose (D144). Headers starting `X-GPM-` are the
#: pool's own — the router drops an application's before forwarding — and none goes on to an
#: engine.
MODEL_HEADER = "x-gpm-model"
POOL_HEADER_PREFIX = "x-gpm-"

#: The largest body forwarded. Above the most the pool's own request limit may be set to (64 MB;
#: 24 MB by default, D144), so it does not refuse what the pool let through — with room for what
#: the pool adds when it rebuilds an upload's form to name the build (a boundary, the model's
#: tag), which can make a body the pool accepted at its limit a few hundred bytes longer. It is
#: here so that nothing reaching this port can make it hold an unbounded body.
MAX_BODY_BYTES = 65_000_000

#: Above this, a JSON body is read on a worker thread, so one large picture does not stall every
#: other stream this process carries — the rule the pool's router keeps for itself (D144).
READ_OFF_LOOP_BYTES = 1_000_000


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


#: A server on this machine: `http://127.0.0.1:<port>`, nothing more.
_LOOPBACK = re.compile(r"http://127\.0\.0\.1:[0-9]{1,5}")


class OllamaServers:
    """The Ollama servers on this machine, one per card, read from the file whoever started them
    wrote (`ollama_cards`): `{"servers": [{"card": "0", "url": "http://127.0.0.1:11435"}, …]}`.
    Re-read as it changes, as the model map is. Only loopback addresses with a port are taken:
    every server runs on this machine."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._stamp: Optional[float] = None
        self._servers: list[tuple[str, str]] = []

    def servers(self) -> list[tuple[str, str]]:
        """(card, url) for each server, in the order they were started."""
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return self._servers
        if stamp != self._stamp:
            self._servers = self._read()
            self._stamp = stamp
        return self._servers

    def _read(self) -> list[tuple[str, str]]:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("could not read the servers at %s: %s", self.path, exc)
            return self._servers
        listed = raw.get("servers") if isinstance(raw, dict) else None
        if not isinstance(listed, list):
            log.warning("the servers file at %s lists no servers", self.path)
            return self._servers
        found = []
        for entry in listed:
            url = entry.get("url") if isinstance(entry, dict) else None
            card = entry.get("card") if isinstance(entry, dict) else None
            if isinstance(url, str) and _LOOPBACK.fullmatch(url) and isinstance(card, str):
                found.append((card, url))
            else:
                log.warning("ignoring a server entry that is not a card and a loopback URL: %.200r", entry)
        return found


def _json_facts(body: bytes) -> tuple[Optional[str], bool]:
    """The model a JSON body names and whether it asks to stream, in one parse. A body that is
    not a JSON object names no model and does not stream."""
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    model = parsed.get("model")
    return (model if isinstance(model, str) else None), bool(parsed.get("stream", False))


def _is_upload(content_type: Optional[str]) -> bool:
    """Whether a body is a form — how a recording is sent (D142). A form is never parsed here: it
    can be tens of megabytes, and the pool's router has already read and rebuilt it (D144)."""
    return (content_type or "").split(";", 1)[0].strip().lower() == "multipart/form-data"


def _refusal(status_code: int, kind: str, message: str) -> JSONResponse:
    """An error in the shape engines answer with. `message` is always fixed text: what a caller
    sent and what went wrong upstream are logged, never answered."""
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status_code)


async def _bounded_body(request: Request, limit: int) -> Optional[bytes]:
    """The whole body, or None when it is over `limit`: refused on its declared length before
    any of it is read, and counted as it is read, since a length need not be declared."""
    declared = (request.headers.get("content-length") or "").strip()
    if declared.isascii() and declared.isdigit() and int(declared) > limit:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    upstream_map: str | Path,
    *,
    client: Optional[httpx.AsyncClient] = None,
    max_body_bytes: int = MAX_BODY_BYTES,
    engine: str = "vllm",
) -> Starlette:
    """The router. `upstream_map` is the model map (`vllm`), or for `engine="ollama"` the file
    listing one Ollama server per card (D190)."""
    if engine not in ENGINES:
        raise ValueError(f"the router stands in front of {' or '.join(ENGINES)}, not {engine!r}")
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

    def did_not_answer(model: Optional[str], refused: Optional[httpx.HTTPError]) -> Response:
        log.warning("no copy of %.200r answered: %s", model, refused)
        return _refusal(502, "upstream_unavailable", "the engine for this model did not answer")

    async def forward(request: Request) -> Response:
        # The path exactly as the server read it — not one re-parsed from a URL, which a `%3F`
        # in it would cut short — and only a path on the list is sent on.
        path = request.scope["path"]
        if path not in FORWARDED_PATHS:
            log.warning("refused a request on a path that is not forwarded: %.200r", path)
            return _refusal(404, "path_not_served", "this machine forwards inference requests only")
        body = await _bounded_body(request, max_body_bytes)
        if body is None:
            log.warning("refused a request on %s over %d bytes", path, max_body_bytes)
            return _refusal(413, "request_too_large", "the request body is over this machine's limit")

        # The pool's header, when there is one, is the model: the router chose the build and
        # wrote the same tag into the body, and reading it from there would mean parsing a form.
        named = request.headers.get(MODEL_HEADER)
        if _is_upload(request.headers.get("content-type")):
            # Never parsed. Whether its answer streams is a field of the form, so every upload's
            # answer is relayed as it comes — which is also right for one that does not stream.
            model, stream = named, True
        else:
            if len(body) > READ_OFF_LOOP_BYTES:
                in_body, stream = await asyncio.to_thread(_json_facts, body)
            else:
                in_body, stream = _json_facts(body)
            model = named if named is not None else in_body
        urls = upstreams.urls_for(model)
        if not urls:
            log.warning("refused a request for a model this machine does not serve: %.200r (it serves: %s)",
                        model, ", ".join(sorted(upstreams.by_model())) or "none")
            return _refusal(404, "model_not_found",
                            "the request names no model this machine serves; GET /v1/models lists those it does")

        # Hop-by-hop headers are the connection's, not the message's, and forwarding them
        # breaks the connection this process owns. The pool's own headers end here.
        headers = {
            name: value for name, value in request.headers.items()
            if name.lower() not in ("host", "content-length", "connection", "transfer-encoding")
            and not name.lower().startswith(POOL_HEADER_PREFIX)
        }
        # Least busy first; a copy that refuses the connection never saw the request, so the
        # next is tried. Anything after the connection is the copy's answer and is passed on.
        candidates = least_busy_first(urls)

        if not stream:
            refused: Optional[httpx.HTTPError] = None
            for url in candidates:
                in_flight[url] = in_flight.get(url, 0) + 1
                try:
                    answer = await http.post(f"{url}{path}", content=body, headers=headers)
                except httpx.ConnectError as exc:
                    log.warning("the copy of %.200r at %s refused the connection: %s", model, url, exc)
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
            return did_not_answer(model, refused)

        return await relayed(candidates, path, body, headers, model)

    async def relayed(candidates: list[str], path: str, body: bytes, headers: dict[str, str],
                      model: Optional[str]) -> Response:
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
                    http.build_request("POST", f"{url}{path}", content=body, headers=headers), stream=True)
                chosen = url
                break
            except httpx.ConnectError as exc:
                log.warning("the copy of %.200r at %s refused the connection: %s", model, url, exc)
                refused = exc
            except httpx.HTTPError as exc:
                refused = exc
                in_flight[url] -= 1
                break
            in_flight[url] -= 1
        if upstream is None:
            return did_not_answer(model, refused)

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                # Raised, not swallowed: a clean end would pass a cut answer as a whole one, and
                # the pool would deliver it rather than run it again (D62).
                log.warning("stream from %s for %.200r broke: %s", chosen, model, exc)
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

    if engine == "ollama":
        return Starlette(routes=_ollama_routes(OllamaServers(upstream_map), http, in_flight, relayed, max_body_bytes),
                         lifespan=lifespan)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/{path:path}", forward, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def _ollama_routes(
    servers: OllamaServers, http: httpx.AsyncClient, in_flight: dict[str, int],
    relayed: Callable[..., Awaitable[Response]], max_body_bytes: int,
) -> list[Route]:
    """In front of one Ollama server per card (D190): every server serves every model on the
    machine's disk, so a request goes to the least busy, and what is read is the machine's."""

    async def each(path: str) -> tuple[list[tuple[str, httpx.Response]], Optional[str]]:
        """Every server's answer to `GET path`, or what the first that did not answer was."""
        listed = servers.servers()
        if not listed:
            return [], "no Ollama servers are listed here yet"
        answers = []
        for card, url in listed:
            try:
                answer = await http.get(f"{url}{path}", timeout=HEALTH_TIMEOUT_S)
            except httpx.HTTPError as exc:
                log.warning("the Ollama server for card %s at %s did not answer %s: %s", card, url, path, exc)
                return [], f"the Ollama server for card {card} did not answer"
            if answer.status_code != 200:
                return [], f"the Ollama server for card {card} answered {answer.status_code}"
            answers.append((card, answer))
        return answers, None

    def unready(why: str) -> JSONResponse:
        # Ollama's own shape for an error. The card is this machine's own number, never a caller's.
        return JSONResponse({"error": why}, status_code=503)

    async def version(request: Request) -> Response:
        """The servers' version — only when every one of them answers, so a machine with a dead
        copy is not one the pool calls healthy (D192)."""
        answers, why = await each("/api/version")
        if why:
            return unready(why)
        return Response(answers[0][1].content, media_type="application/json")

    async def health(request: Request) -> Response:
        _, why = await each("/api/version")
        return unready(why) if why else Response(status_code=200)

    async def running(request: Request) -> Response:
        answers, why = await each("/api/version")
        return unready(why) if why else Response("Ollama is running", media_type="text/plain")

    def entries(answer: httpx.Response) -> list[dict[str, Any]]:
        try:
            listed = answer.json().get("models")
        except (ValueError, AttributeError):
            return []
        return [entry for entry in listed if isinstance(entry, dict) and isinstance(entry.get("name"), str)] \
            if isinstance(listed, list) else []

    async def common(path: str) -> Response:
        """The models every server lists, each as the server with the least of it on its card
        lists it: a model one server holds and another does not is not the machine's."""
        answers, why = await each(path)
        if why:
            return unready(why)
        found: Optional[dict[str, dict[str, Any]]] = None
        for _, answer in answers:
            mine = {entry["name"]: entry for entry in entries(answer)}
            if found is None:
                found = mine
                continue
            kept = {}
            for name, entry in found.items():
                if name in mine:
                    other = mine[name]
                    least = other if _on_card(other) < _on_card(entry) else entry
                    kept[name] = least
            found = kept
        return JSONResponse({"models": list((found or {}).values())})

    async def tags(request: Request) -> Response:
        return await common("/api/tags")

    async def loaded(request: Request) -> Response:
        return await common("/api/ps")

    async def show(request: Request) -> Response:
        """A model's details, from the first server: every server reads the same files."""
        body = await _bounded_body(request, SHOW_BODY_BYTES)
        if body is None:
            return _refusal(413, "request_too_large", "the request body is over this machine's limit")
        listed = servers.servers()
        if not listed:
            return unready("no Ollama servers are listed here yet")
        card, url = listed[0]
        try:
            answer = await http.post(f"{url}/api/show", content=body,
                                     headers={"content-type": "application/json"}, timeout=HEALTH_TIMEOUT_S * 6)
        except httpx.HTTPError as exc:
            log.warning("the Ollama server for card %s did not answer /api/show: %s", card, exc)
            return unready(f"the Ollama server for card {card} did not answer")
        return Response(answer.content, status_code=answer.status_code,
                        media_type=answer.headers.get("content-type"))

    async def forward(request: Request) -> Response:
        path = request.scope["path"]
        if path not in OLLAMA_FORWARDED_PATHS:
            log.warning("refused a request on a path that is not forwarded: %.200r", path)
            return _refusal(404, "path_not_served", "this machine forwards inference requests only")
        body = await _bounded_body(request, max_body_bytes)
        if body is None:
            log.warning("refused a request on %s over %d bytes", path, max_body_bytes)
            return _refusal(413, "request_too_large", "the request body is over this machine's limit")
        urls = [url for _, url in servers.servers()]
        if not urls:
            return _refusal(502, "upstream_unavailable", "the engine for this model did not answer")
        headers = {
            name: value for name, value in request.headers.items()
            if name.lower() not in ("host", "content-length", "connection", "transfer-encoding")
            and not name.lower().startswith(POOL_HEADER_PREFIX)
        }
        # Every server serves every model; the least busy first, stable among ties. Always
        # relayed as it comes: Ollama's own paths stream unless a request says not to, and an
        # answer in one piece is relayed whole all the same.
        candidates = sorted(urls, key=lambda url: in_flight.get(url, 0))
        return await relayed(candidates, path, body, headers, request.headers.get(MODEL_HEADER))

    return [
        Route("/", running, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/api/version", version, methods=["GET"]),
        Route("/api/tags", tags, methods=["GET"]),
        Route("/api/ps", loaded, methods=["GET"]),
        Route("/api/show", show, methods=["POST"]),
        Route("/{path:path}", forward, methods=["POST"]),
    ]


def _on_card(entry: dict[str, Any]) -> float:
    """How much of a loaded model is in its card's memory — infinite where the server does not
    say, so an entry that says is always preferred to one that does not."""
    value = entry.get("size_vram")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("inf")


def serve(upstream_map: str, host: str = "127.0.0.1", port: int = 8000, engine: str = "vllm") -> None:
    """Run it. Loopback by default: the pool reaches this through a forward into the machine,
    so binding anything wider only exposes it (D77)."""
    import uvicorn

    uvicorn.run(create_app(upstream_map, engine=engine), host=host, port=port, log_level="warning")


def add_arguments(parser: Any) -> None:
    parser.add_argument("--upstreams", required=True, help="JSON file mapping model name to its engine URL, or a list of them (one per copy)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=ENGINES, default="vllm",
                        help="what the upstreams are: one process per model (vllm), or one Ollama server per card (ollama)")
