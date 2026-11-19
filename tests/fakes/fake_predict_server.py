"""What serves a trained model, as the pool sees it (D178–D180): the pool's own ONNX server or an
operator's container image. `GET /health` answers once it is up; `POST /v1/predict` answers with
outputs; it cannot be trusted to say which builds it holds — the host's agent says."""

from __future__ import annotations

import json
from typing import Any, Iterable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


class FakePredictServer:
    def __init__(self) -> None:
        self.running = False
        self.received: list[tuple[str, dict[str, Any]]] = []
        self.app = Starlette(routes=[
            Route("/health", self._health, methods=["GET"]),
            Route("/v1/predict", self._predict, methods=["POST"]),
        ])

    def start(self, models: Iterable[str]) -> None:
        self.running = bool(set(models))

    async def _health(self, request: Request) -> Response:
        if not self.running:
            return JSONResponse({"error": "not started"}, status_code=503)
        return JSONResponse({"ok": True})

    async def _predict(self, request: Request) -> Response:
        body = json.loads(await request.body())
        self.received.append((request.url.path, body))
        inputs = body.get("inputs") or {}
        return JSONResponse({"model": body.get("model"), "outputs": {"seen": sorted(inputs)}})
