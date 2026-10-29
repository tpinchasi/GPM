"""A whisper.cpp server, as the pool sees one (D148): `GET /` answers once its model is loaded,
it transcribes on OpenAI's path, and it cannot say which model it holds — its agent says."""

from __future__ import annotations

from typing import Any, Iterable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .fake_vllm import read_form


class FakeWhisperCpp:
    def __init__(self) -> None:
        self.running = False
        self.received: list[tuple[str, dict[str, Any]]] = []
        self.app = Starlette(routes=[
            Route("/", self._index, methods=["GET"]),
            Route("/v1/audio/transcriptions", self._transcribe, methods=["POST"]),
        ])

    def start(self, models: Iterable[str]) -> None:
        self.running = bool(set(models))

    async def _index(self, request: Request) -> Response:
        if not self.running:
            return JSONResponse({"error": "not started"}, status_code=503)
        return HTMLResponse("<html><body>whisper.cpp server</body></html>")

    async def _transcribe(self, request: Request) -> Response:
        fields = read_form(request.headers.get("content-type", ""), await request.body())
        self.received.append((request.url.path, fields))
        audio = fields.get("file") or {}
        return JSONResponse({"text": f"heard {len(audio.get('data', b''))} bytes"})
