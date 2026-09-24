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
import time
from typing import Any, Iterable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


class FakeVllm:
    def __init__(self, *, load_delay_s: float = 0.0):
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


class FakeHub:
    """A model hub with a few repositories, served over real HTTP so the agent's fetch runs as
    it would against the real one — pointed here by `HF_ENDPOINT`."""

    def __init__(self, repos: dict[str, dict[str, bytes]]):
        self.repos = repos
        self.requests: list[str] = []
        self.app = Starlette(routes=[
            Route("/api/models/{owner}/{name}/tree/main", self._tree, methods=["GET"]),
            Route("/{owner}/{name}/resolve/main/{path:path}", self._file, methods=["GET"]),
        ])

    async def _tree(self, request: Request) -> Response:
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        self.requests.append(f"list {repo}")
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
