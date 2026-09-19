"""A fake inference engine speaking Ollama's HTTP API.

Every test in the default suite runs against this — no cloud account, no GPU. It is
deterministic: the same request always produces the same bytes, which is what makes
"identical through the router and direct to the engine" a meaningful assertion.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

_CREATED_AT = "2026-01-01T00:00:00.000000Z"


class FakeOllama:
    def __init__(
        self,
        *,
        resident: set[str] | frozenset[str] = frozenset(),
        chunk_delay_s: float = 0.0,
        chunks: int = 3,
        healthy: bool = True,
    ):
        self.resident: set[str] = set(resident)
        self.chunk_delay_s = chunk_delay_s
        self.chunks = chunks
        self.healthy = healthy
        #: Scripted: a host that cannot fetch the model set must not join the pool.
        self.refuse_pull = False
        #: Every request as it arrived: (path, raw body bytes, headers).
        self.received: list[tuple[str, bytes, dict[str, str]]] = []
        self.started = 0
        self.completed = 0
        self.cancelled = 0
        self.app = Starlette(
            routes=[
                Route("/api/version", self._version, methods=["GET"]),
                Route("/api/ps", self._ps, methods=["GET"]),
                Route("/api/tags", self._tags, methods=["GET"]),
                Route("/api/chat", self._chat, methods=["POST"]),
                Route("/api/generate", self._generate, methods=["POST"]),
                Route("/api/embed", self._embed, methods=["POST"]),
                Route("/api/pull", self._pull, methods=["POST"]),
            ]
        )

    # --- engine state the probe reads ---

    async def _version(self, request: Request) -> Response:
        if not self.healthy:
            return JSONResponse({"error": "engine down"}, status_code=500)
        return JSONResponse({"version": "0.0.0-fake"})

    def _model_list(self) -> dict[str, Any]:
        return {"models": [{"name": tag, "model": tag, "size": 1} for tag in sorted(self.resident)]}

    async def _ps(self, request: Request) -> Response:
        if not self.healthy:
            return JSONResponse({"error": "engine down"}, status_code=500)
        return JSONResponse(self._model_list())

    async def _tags(self, request: Request) -> Response:
        return JSONResponse(self._model_list())

    # --- inference ---

    async def _record(self, request: Request) -> tuple[bytes, dict[str, Any]]:
        body = await request.body()
        self.received.append((request.url.path, body, dict(request.headers)))
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {}
        return body, parsed

    def _content_for(self, parsed: dict[str, Any]) -> str:
        if isinstance(parsed.get("format"), dict):
            return '{"answer":"42"}'
        messages = parsed.get("messages") or []
        last = messages[-1].get("content", "") if messages else parsed.get("prompt", "")
        return f"echo:{last}"

    def _message(self, parsed: dict[str, Any]) -> dict[str, Any]:
        if parsed.get("tools"):
            tool = parsed["tools"][0]["function"]["name"]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": tool, "arguments": {"city": "Paris"}}}],
            }
        return {"role": "assistant", "content": self._content_for(parsed)}

    async def _chat(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        if "messages" not in parsed:
            return JSONResponse({"error": "messages must be provided"}, status_code=400)
        model = parsed.get("model", "")
        message = self._message(parsed)
        if not parsed.get("stream", True):
            self.started += 1
            if self.chunk_delay_s:
                await asyncio.sleep(self.chunk_delay_s)
            self.completed += 1
            return JSONResponse(self._final(model, message))
        return self._stream(model, message)

    async def _generate(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        model = parsed.get("model", "")
        if "prompt" not in parsed and "keep_alive" in parsed:
            self.resident.add(model)  # a bare load, as the engine treats it
            return JSONResponse({"model": model, "done": True, "done_reason": "load"})
        content = self._content_for(parsed)
        if not parsed.get("stream", True):
            self.started += 1
            self.completed += 1
            return JSONResponse(
                {"model": model, "created_at": _CREATED_AT, "response": content, "done": True}
            )
        return self._stream(model, {"role": "assistant", "content": content}, generate=True)

    async def _embed(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        texts = parsed.get("input") or []
        if isinstance(texts, str):
            texts = [texts]
        self.started += 1
        self.completed += 1
        return JSONResponse(
            {
                "model": parsed.get("model", ""),
                "embeddings": [[float(len(text)), 0.5, 0.25] for text in texts],
            }
        )

    async def _pull(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        tag = parsed.get("model", "")
        if self.refuse_pull:
            return JSONResponse({"error": f"no such model {tag}"}, status_code=404)

        async def frames():
            yield (json.dumps({"status": "pulling", "total": 1_000_000_000}) + "\n").encode()
            self.resident.add(tag)
            yield (json.dumps({"status": "success", "total": 1_000_000_000}) + "\n").encode()

        return StreamingResponse(frames(), media_type="application/x-ndjson")

    def _final(self, model: str, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": model,
            "created_at": _CREATED_AT,
            "message": message,
            "done": True,
            "done_reason": "stop",
            "eval_count": 3,
        }

    def _stream(self, model: str, message: dict[str, Any], generate: bool = False) -> StreamingResponse:
        content = message.get("content", "")
        pieces = _split(content, self.chunks)

        async def body():
            self.started += 1
            try:
                for piece in pieces:
                    if self.chunk_delay_s:
                        await asyncio.sleep(self.chunk_delay_s)
                    if generate:
                        frame = {"model": model, "created_at": _CREATED_AT, "response": piece, "done": False}
                    else:
                        frame = {
                            "model": model,
                            "created_at": _CREATED_AT,
                            "message": {"role": "assistant", "content": piece},
                            "done": False,
                        }
                    yield (json.dumps(frame) + "\n").encode()
                if generate:
                    tail: dict[str, Any] = {
                        "model": model,
                        "created_at": _CREATED_AT,
                        "response": "",
                        "done": True,
                        "done_reason": "stop",
                    }
                else:
                    tail = self._final(model, {"role": "assistant", "content": "", **_tool_calls(message)})
                yield (json.dumps(tail) + "\n").encode()
                self.completed += 1
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            except GeneratorExit:
                self.cancelled += 1
                raise

        return StreamingResponse(body(), media_type="application/x-ndjson")


def _tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    return {"tool_calls": message["tool_calls"]} if "tool_calls" in message else {}


def _split(text: str, parts: int) -> list[str]:
    if parts <= 1 or not text:
        return [text]
    size = max(1, len(text) // parts)
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    return pieces or [""]


def unused_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
