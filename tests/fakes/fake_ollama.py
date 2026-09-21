"""A fake inference engine speaking Ollama's HTTP API.

Every test in the default suite runs against this — no cloud account, no GPU. It is
deterministic: the same request always produces the same bytes, which is what makes
"identical through the router and direct to the engine" a meaningful assertion.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

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
        available: set[str] | frozenset[str] | None = None,
        chunk_delay_s: float = 0.0,
        chunks: int = 3,
        healthy: bool = True,
    ):
        #: Loaded in memory (`/api/ps`). A tag can only be loaded if it is on disk.
        self.resident: set[str] = set(resident)
        #: On disk (`/api/tags`), loaded or not. A request for a tag on disk loads it, as the
        #: engine does; a request for one that is not is a 404 — never a download.
        self.available: set[str] = set(resident) | set(available or ())
        self.chunk_delay_s = chunk_delay_s
        self.chunks = chunks
        self.healthy = healthy
        #: Scripted: a host that cannot fetch the model set must not join the pool.
        self.refuse_pull = False
        #: Scripted: what a pull reports as its size, and a pause between its progress frames.
        self.pull_bytes = 1_000_000_000
        self.pull_delay_s = 0.0
        #: Scripted: how many pulls are cut partway, with the connection dropped mid-body the
        #: way a real one was — and how many end quietly without the engine saying "success".
        self.pull_cut_times = 0
        #: Scripted: how many generations are cut mid-stream, as a host taken away does it.
        self.cut_stream_times = 0
        self.pull_end_early_times = 0
        self.pulls = 0
        #: Scripted: how long one generation takes, and whether the engine runs them one at a
        #: time whatever it was asked for — an engine left at parallelism 1.
        self.generate_delay_s = 0.0
        self.serialise = False
        self._one_at_a_time = asyncio.Lock()
        #: Tags held with `keep_alive: -1` — what "pinned" means to this engine.
        self.pinned: set[str] = set()
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
                Route("/api/show", self._show, methods=["POST"]),
                Route("/api/pull", self._pull, methods=["POST"]),
                Route("/api/delete", self._delete, methods=["DELETE"]),
            ]
        )

    # --- engine state the probe reads ---

    async def _version(self, request: Request) -> Response:
        if not self.healthy:
            return JSONResponse({"error": "engine down"}, status_code=500)
        return JSONResponse({"version": "0.0.0-fake"})

    def _model_list(self, tags: set[str]) -> dict[str, Any]:
        return {"models": [{"name": tag, "model": tag, "size": 1} for tag in sorted(tags)]}

    async def _ps(self, request: Request) -> Response:
        if not self.healthy:
            return JSONResponse({"error": "engine down"}, status_code=500)
        return JSONResponse(self._model_list(self.resident))

    async def _tags(self, request: Request) -> Response:
        # A loaded tag is on disk by definition, however the test put it in memory.
        return JSONResponse(self._model_list(self.available | self.resident))

    def _use(self, model: str) -> Optional[Response]:
        """What the engine does with the model field of an inference call: a tag on disk is
        loaded if it was not; a tag not on disk is refused. It is never fetched."""
        if model not in self.available | self.resident:
            return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
        self.resident.add(model)
        return None

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
        refused = self._use(model)
        if refused is not None:
            return refused
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
        if _is_embedding(model):
            # What the real engine answers (seen live).
            return JSONResponse({"error": f'"{model}" does not support generate'}, status_code=400)
        refused = self._use(model)
        if refused is not None:
            return refused
        if "keep_alive" in parsed:
            (self.pinned.add if parsed["keep_alive"] == -1 else self.pinned.discard)(model)
        if "prompt" not in parsed and "keep_alive" in parsed:
            return JSONResponse({"model": model, "done": True, "done_reason": "load"})
        content = self._content_for(parsed)
        if not parsed.get("stream", True):
            self.started += 1
            if self.serialise:
                async with self._one_at_a_time:
                    await asyncio.sleep(self.generate_delay_s)
            else:
                await asyncio.sleep(self.generate_delay_s)
            self.completed += 1
            return JSONResponse(
                {
                    "model": model,
                    "created_at": _CREATED_AT,
                    "response": content,
                    "done": True,
                    "eval_count": 3,
                    "eval_duration": 30_000_000,
                }
            )
        return self._stream(model, {"role": "assistant", "content": content}, generate=True)

    async def _embed(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        texts = parsed.get("input") or []
        if isinstance(texts, str):
            texts = [texts]
        model = parsed.get("model", "")
        refused = self._use(model)
        if refused is not None:
            return refused
        if "keep_alive" in parsed:
            (self.pinned.add if parsed["keep_alive"] == -1 else self.pinned.discard)(model)
        self.started += 1
        self.completed += 1
        return JSONResponse(
            {
                "model": model,
                "embeddings": [[float(len(text)), 0.5, 0.25] for text in texts],
            }
        )

    async def _show(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        model = parsed.get("model", "")
        if model not in self.available:
            return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
        return JSONResponse({"capabilities": ["embedding"] if _is_embedding(model) else ["completion"]})

    async def _pull(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        tag = parsed.get("model", "")
        self.pulls += 1
        if self.refuse_pull:
            return JSONResponse({"error": f"no such model {tag}"}, status_code=404)

        cut = self.pull_cut_times > 0
        early = not cut and self.pull_end_early_times > 0
        if cut:
            self.pull_cut_times -= 1
        elif early:
            self.pull_end_early_times -= 1

        async def frames():
            # A pull puts the model on disk. It does not load it: that is a separate act.
            size = self.pull_bytes
            if cut or early:
                frame = {"status": "pulling", "digest": "sha256:layer", "total": size, "completed": size // 3}
                yield (json.dumps(frame) + "\n").encode()
                if cut:
                    # Raised inside the body after headers went out: the server drops the
                    # connection, and the client sees an incomplete chunked read — as seen live.
                    raise ConnectionResetError("fake: download cut")
                return
            for done in (0, size // 2, size):
                frame = {"status": "pulling", "digest": "sha256:layer", "total": size, "completed": done}
                yield (json.dumps(frame) + "\n").encode()
                if self.pull_delay_s:
                    await asyncio.sleep(self.pull_delay_s)
            self.available.add(tag)
            yield (json.dumps({"status": "success"}) + "\n").encode()

        return StreamingResponse(frames(), media_type="application/x-ndjson")

    async def _delete(self, request: Request) -> Response:
        _, parsed = await self._record(request)
        tag = parsed.get("model", "")
        if tag not in self.available | self.resident:
            return JSONResponse({"error": f"model '{tag}' not found"}, status_code=404)
        for held in (self.available, self.resident, self.pinned):
            held.discard(tag)
        return JSONResponse({})

    def _final(self, model: str, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": model,
            "created_at": _CREATED_AT,
            "message": message,
            "done": True,
            "done_reason": "stop",
            # What every engine reports of its own work, and what the pool reads to judge a
            # host by throughput rather than by latency (D67).
            "eval_count": 3,
            "eval_duration": 30_000_000,
        }

    def _stream(self, model: str, message: dict[str, Any], generate: bool = False) -> StreamingResponse:
        content = message.get("content", "")
        pieces = _split(content, self.chunks)

        async def body():
            self.started += 1
            cut = self.cut_stream_times > 0
            if cut:
                self.cut_stream_times -= 1
            try:
                for index, piece in enumerate(pieces):
                    if cut and index == 1:
                        # What an interruptible host does when it is taken away: some frames,
                        # then the connection dies with no ending frame.
                        raise ConnectionResetError("fake: host taken away mid-generation")
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


def _is_embedding(model: str) -> bool:
    """The fake's stand-in for what the real engine reads from the model file."""
    return "embed" in model
