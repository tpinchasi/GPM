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

    async def pull(self, client: httpx.AsyncClient, tag: str) -> PullResult:
        moved = 0
        try:
            async with client.stream(
                "POST", "/api/pull", json={"model": tag, "stream": True}, timeout=None
            ) as response:
                if response.status_code != 200:
                    return PullResult(tag=tag, ok=False, detail=f"pull returned {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    if frame.get("error"):
                        return PullResult(tag=tag, ok=False, detail=frame["error"])
                    moved = max(moved, int(frame.get("total") or 0))
        except (httpx.HTTPError, ValueError) as exc:
            return PullResult(tag=tag, ok=False, detail=str(exc))
        return PullResult(tag=tag, ok=True, bytes_total=moved)

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        """`keep_alive: -1` is what "loaded, all the time" means to this engine."""
        for tag in tags:
            response = await client.post(
                "/api/generate", json={"model": tag, "keep_alive": -1}, timeout=None
            )
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
        entries = response.json().get("models", [])
        resident: set[str] = set()
        for entry in entries:
            for key in ("name", "model"):
                value = entry.get(key)
                if isinstance(value, str):
                    resident.add(value)
        return frozenset(resident)
