"""A stand-in for vLLM: serves nothing until it is started with models, like the real one.

What matters about the real engine, and is reproduced here:

- It has **no pull**. Weights arrive from a hub before it starts.
- It serves **what it was started with**, and only that. Started with nothing, it answers nothing.
- Starting it again is how a downloaded model gets served, and takes a moment to come up.
- It speaks the OpenAI-shaped API, and reports its queue at `/metrics`.

`start()` is called by the test's stand-in for starting processes, which the agent's real
launcher drives — so the launcher's choice of model and name is what this ends up serving.
"""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import time
from typing import Any, Iterable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


def read_form(content_type: str, body: bytes) -> dict[str, Any]:
    """A form upload read by the standard library's MIME parser — an implementation that is not
    the pool's, so a test proves the pool's rebuilt form is read right by someone else's code."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    fields: dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        fields[name] = {"filename": part.get_filename(), "data": data} if part.get_filename() else data.decode()
    return fields


class FakeVllm:
    def __init__(self, *, load_delay_s: float = 0.0, audio_extra: bool = True):
        #: Whether the image carried vLLM's audio extra (D146). Without it the real engine starts
        #: and answers every recording 400, saying why only in its log.
        self.audio_extra = audio_extra
        #: The headers of every request, for what the pool sent beside the body.
        self.headers: list[dict[str, str]] = []
        #: How long after `start()` the engine takes to answer — a real one loads its weights.
        self.load_delay_s = load_delay_s
        self.served: set[str] = set()
        self._up_at: Optional[float] = None
        #: Every start, with what it was started on — the test reads this back.
        self.starts: list[set[str]] = []
        self.received: list[tuple[str, dict[str, Any]]] = []
        self.app = Starlette(routes=[
            Route("/health", self._health, methods=["GET"]),
            Route("/version", self._version, methods=["GET"]),
            Route("/v1/models", self._models, methods=["GET"]),
            Route("/metrics", self._metrics, methods=["GET"]),
            Route("/v1/chat/completions", self._chat, methods=["POST"]),
            Route("/v1/embeddings", self._chat, methods=["POST"]),
            Route("/v1/audio/transcriptions", self._transcribe, methods=["POST"]),
            Route("/v1/audio/translations", self._transcribe, methods=["POST"]),
            Route("/v1/rerank", self._rerank, methods=["POST"]),
        ])

    def start(self, models: Iterable[str]) -> None:
        """What a real `vllm serve` does from the pool's point of view: come up, some time
        later, serving exactly the models it was started on."""
        self.served = set(models)
        self.starts.append(set(self.served))
        self._up_at = time.monotonic() + self.load_delay_s

    @property
    def running(self) -> bool:
        return self._up_at is not None and time.monotonic() >= self._up_at and bool(self.served)

    def _down(self) -> Response:
        # A real one that is not running refuses the connection; the server here is always
        # listening, so the nearest honest answer is "not available".
        return JSONResponse({"error": "not started"}, status_code=503)

    async def _health(self, request: Request) -> Response:
        return Response(status_code=200) if self.running else self._down()

    async def _version(self, request: Request) -> Response:
        return JSONResponse({"version": "0.29.0-fake"}) if self.running else self._down()

    async def _models(self, request: Request) -> Response:
        if not self.running:
            return self._down()
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model"} for m in sorted(self.served)]})

    async def _metrics(self, request: Request) -> Response:
        if not self.running:
            return self._down()
        return PlainTextResponse(
            "vllm:num_requests_running 0.0\nvllm:num_requests_waiting 0.0\nvllm:kv_cache_usage_perc 0.0\n"
        )

    async def _chat(self, request: Request) -> Response:
        body = await request.json()
        self.received.append((request.url.path, body))
        if not self.running:
            return self._down()
        model = body.get("model")
        if model not in self.served:
            return JSONResponse({"error": {"message": f"The model `{model}` does not exist."}}, status_code=404)
        await asyncio.sleep(0)
        if request.url.path == "/v1/embeddings":
            return JSONResponse({"object": "list", "model": model,
                                 "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}]})
        return JSONResponse({
            "id": "chatcmpl-fake", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "served by vllm"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        })


    async def _transcribe(self, request: Request) -> Response:
        self.headers.append(dict(request.headers))
        fields = read_form(request.headers.get("content-type", ""), await request.body())
        self.received.append((request.url.path, fields))
        if not self.running:
            return self._down()
        model = fields.get("model")
        if model not in self.served:
            return JSONResponse({"error": {"message": f"The model `{model}` does not exist."}}, status_code=404)
        if not self.audio_extra:
            return JSONResponse({"error": {"message": "Invalid or unsupported audio file."}}, status_code=400)
        upload = fields.get("file") or {}
        text = f"heard {len(upload.get('data', b''))} bytes"
        if fields.get("response_format") == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    async def _rerank(self, request: Request) -> Response:
        self.headers.append(dict(request.headers))
        body = await request.json()
        self.received.append((request.url.path, body))
        if not self.running:
            return self._down()
        model = body.get("model")
        if model not in self.served:
            return JSONResponse({"error": {"message": f"The model `{model}` does not exist."}}, status_code=404)
        documents = body.get("documents") or []
        results = [
            {"index": i, "relevance_score": 1.0 / (1 + i), "document": {"text": d}} for i, d in enumerate(documents)
        ]
        return JSONResponse({"model": model, "results": results[: body.get("top_n") or len(results)]})


class FakeHub:
    """A model hub with a few repositories, served over real HTTP so the agent's fetch runs as
    it would against the real one — pointed here by `HF_ENDPOINT`.

    It also answers the hub's search the way the real one does (D100): `search` maps a search
    term to the entries the hub returned for it — recorded from the real hub, in the tests that
    use it — and `info` answers a repository asked for by name."""

    def __init__(self, repos: dict[str, dict[str, bytes]],
                 search: Optional[dict[str, list[dict[str, Any]]]] = None,
                 info: Optional[dict[str, dict[str, Any]]] = None):
        self.repos = repos
        self.search = search or {}
        self.info = info or {}
        #: Set to make every answer fail, as a hub that is down would.
        self.down = False
        self.requests: list[str] = []
        self.headers_seen: list[dict[str, str]] = []
        self.app = Starlette(routes=[
            Route("/api/models", self._search, methods=["GET"]),
            Route("/api/models/{owner}/{name}/tree/main", self._tree, methods=["GET"]),
            Route("/api/models/{owner}/{name}", self._info, methods=["GET"]),
            Route("/{owner}/{name}/resolve/main/{path:path}", self._file, methods=["GET"]),
        ])

    async def _search(self, request: Request) -> Response:
        term = request.query_params.get("search", "")
        self.requests.append(f"search {term}")
        self.headers_seen.append(dict(request.headers))
        if self.down:
            return JSONResponse({"error": "down"}, status_code=503)
        return JSONResponse(self.search.get(term, []))

    async def _info(self, request: Request) -> Response:
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        self.requests.append(f"info {repo}")
        if repo not in self.info:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(self.info[repo])

    async def _tree(self, request: Request) -> Response:
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        self.requests.append(f"list {repo}")
        if self.down:
            return JSONResponse({"error": "down"}, status_code=503)
        files = self.repos.get(repo)
        if files is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse([{"type": "file", "path": p, "size": len(b)} for p, b in files.items()])

    async def _file(self, request: Request) -> Response:
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        path = request.path_params["path"]
        self.requests.append(f"get {repo}/{path}")
        body = self.repos.get(repo, {}).get(path)
        if body is None:
            return Response(status_code=404)
        return Response(content=body, media_type="application/octet-stream")
