"""The Ollama engine adapter — the first implementation of the engine interface."""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Optional

import httpx

from .base import Health, PullResult

# Matches the first `"model": "<value>"` pair, including escaped characters in the value.
_MODEL_FIELD = re.compile(rb'("model"\s*:\s*)"(?:[^"\\]|\\.)*"')

_STREAMING_PATHS = frozenset({"/api/chat", "/api/generate"})


def _decode(body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OllamaEngine:
    interface_version: ClassVar[str] = "1"
    name: ClassVar[str] = "ollama"

    def inference_paths(self) -> set[str]:
        return {"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"}

    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        model = _decode(body).get("model")
        return model if isinstance(model, str) else None

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        """Splice the new tag into the raw bytes, leaving every other byte untouched.

        Re-serialising the parsed body would reorder keys and renormalise numbers and
        whitespace, which is exactly the passthrough drift the contract forbids.
        """
        replacement = rb"\g<1>" + json.dumps(tag).encode()
        rewritten, count = _MODEL_FIELD.subn(replacement, body, count=1)
        return rewritten if count else body

    def wants_schema(self, path: str, body: bytes) -> bool:
        # A JSON schema object enforces structure; the bare string "json" is loose JSON mode,
        # which guarantees nothing about shape.
        return isinstance(_decode(body).get("format"), dict)

    def is_streaming(self, path: str, body: bytes) -> bool:
        if path not in _STREAMING_PATHS:
            return False
        return bool(_decode(body).get("stream", True))

    async def health(self, client: httpx.AsyncClient) -> Health:
        try:
            response = await client.get("/api/version")
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc))
        if response.status_code != 200:
            return Health(ok=False, detail=f"/api/version returned {response.status_code}")
        return Health(ok=True)

    async def pull(self, client: httpx.AsyncClient, tag: str, on_progress=None) -> PullResult:
        """Fetch one tag. The engine keeps the layers it already has, so a pull tried again
        after a cut picks up where it stopped rather than starting over.

        `on_progress(completed, total)` is called as layers arrive, so an operator watching a
        host be prepared can see a 19 GB download move rather than a host sitting at
        "preparing" for minutes with nothing to look at.
        """
        moved = 0
        finished = False
        try:
            async with client.stream(
                "POST", "/api/pull", json={"model": tag, "stream": True}, timeout=None
            ) as response:
                if response.status_code != 200:
                    return PullResult(
                        tag=tag, ok=False, detail=f"pull returned {response.status_code}",
                        # The engine or what is in front of it is struggling; a tag it does not
                        # have, or a request it refuses, will not change by asking again.
                        retryable=response.status_code >= 500 or response.status_code == 429,
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    if frame.get("error"):
                        error = str(frame["error"])
                        return PullResult(tag=tag, ok=False, detail=error, retryable=not _permanent(error))
                    moved = max(moved, int(frame.get("total") or 0))
                    finished = finished or frame.get("status") == "success"
                    if on_progress is not None and frame.get("total"):
                        on_progress(int(frame.get("completed") or 0), int(frame["total"]))
        except httpx.TransportError as exc:
            # Seen live: "peer closed connection without sending complete message body" — the
            # download was cut, and what arrived is kept for the next attempt.
            return PullResult(tag=tag, ok=False, detail=str(exc) or type(exc).__name__, retryable=True)
        except (httpx.HTTPError, ValueError) as exc:
            return PullResult(tag=tag, ok=False, detail=str(exc) or type(exc).__name__, retryable=True)
        if not finished:
            # The stream ended without the engine saying it finished. A cut that happens to land
            # between two lines looks exactly like this, and must not pass as a download.
            return PullResult(tag=tag, ok=False, detail="the pull ended before the engine reported success", retryable=True)
        return PullResult(tag=tag, ok=True, bytes_total=moved)

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        """`keep_alive: -1` is what "loaded, all the time" means to this engine."""
        for tag in tags:
            # An embedding model refuses /api/generate with a 400 (seen live), so it is loaded
            # through the endpoint it does serve. The engine says which kind a tag is.
            shown = await client.post("/api/show", json={"model": tag})
            shown.raise_for_status()
            capabilities = shown.json().get("capabilities") or []
            path = "/api/embed" if "embedding" in capabilities else "/api/generate"
            response = await client.post(path, json={"model": tag, "keep_alive": -1}, timeout=None)
            response.raise_for_status()
        resident = await self.models_resident(client)
        missing = set(tags) - resident
        if missing:
            raise RuntimeError(
                f"{sorted(missing)} would not stay resident together with the rest of the set"
            )

    def launch_settings(self, workers: int, context: int, n_models: int) -> dict[str, str]:
        return {
            "OLLAMA_NUM_PARALLEL": str(workers),
            "OLLAMA_MAX_LOADED_MODELS": str(n_models),
            "OLLAMA_CONTEXT_LENGTH": str(context),
            # Nothing is loaded on demand and nothing is ever swapped out.
            "OLLAMA_KEEP_ALIVE": "-1",
        }

    async def models_resident(self, client: httpx.AsyncClient) -> frozenset[str]:
        response = await client.get("/api/ps")
        response.raise_for_status()
        return _tags_in(response.json().get("models", []))

    async def models_available(self, client: httpx.AsyncClient) -> frozenset[str]:
        response = await client.get("/api/tags")
        response.raise_for_status()
        return _tags_in(response.json().get("models", []))


#: Error text that means the pull cannot succeed however often it is tried.
_PERMANENT_PULL_ERRORS = ("not found", "does not exist", "manifest unknown", "invalid model", "no space left")


def _permanent(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in _PERMANENT_PULL_ERRORS)


def _tags_in(entries: list[dict]) -> frozenset[str]:
    tags: set[str] = set()
    for entry in entries:
        for key in ("name", "model"):
            value = entry.get(key)
            if isinstance(value, str):
                tags.add(value)
    return frozenset(tags)
