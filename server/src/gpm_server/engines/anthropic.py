"""`anthropic-api`: a hosted model API speaking Anthropic's Messages format (S12).

Anthropic's own API, and any provider that serves `/v1/messages` the same way (OpenRouter,
DeepSeek, Ollama's cloud). It is reached as the `openai-api` engine is — a base URL ending at the
provider's `/v1`, a key sent as `Authorization: Bearer`, the request passed through with only the
model and key swapped, refusals read for busy, out of money or the request's own — with three
differences:

- **Its streams always carry their usage**, so nothing is added to ask for it.
- **The prompt cache is asked for where the provider caches only when asked** (`prepare`): a
  request carrying no cache mark gains the one top-level `cache_control` that turns on
  Anthropic's automatic 5-minute cache, unless the host says `cache_prompts: false`. A named
  rewrite, as D166's usage request is; every other byte is as the app sent it.
- **Every request names the API version** it is written for (`anthropic-version`); one that names
  none — translated from OpenAI's format — is sent with the stable version.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import httpx

from . import anthropic_api
from .base import CHAT, RequestFacts
from .hosted import HostedOpenAIEngine

_VERSION_HEADER = {"anthropic-version": anthropic_api.API_VERSION}


class AnthropicHostedEngine(HostedOpenAIEngine):
    name: ClassVar[str] = "anthropic-api"
    kinds: ClassVar[frozenset[str]] = frozenset({CHAT})

    # --- request path ---

    def inference_paths(self) -> set[str]:
        return set(anthropic_api.INFERENCE_PATHS)

    def kind_of(self, path: str) -> Optional[str]:
        return anthropic_api.kind_of(path)

    def request_facts(self, path: str, content_type: str, body: bytes) -> RequestFacts:
        return anthropic_api.facts(path, content_type, body)

    def rewrite_model(self, path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
        return anthropic_api.rewrite_model(path, content_type, body, tag)

    def wants_schema(self, path: str, body: bytes) -> bool:
        return anthropic_api.wants_schema(path, body)

    def is_streaming(self, path: str, body: bytes) -> bool:
        return anthropic_api.is_streaming(path, body)

    def usage(self, path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
        return anthropic_api.usage(path, tail)

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        return anthropic_api.keepalive_frame(path)

    # --- what a hosted request needs ---

    @staticmethod
    def ask_for_usage(content_type: str, body: bytes) -> tuple[bytes, bool]:
        return body, True  # every stream of this format carries its usage

    @staticmethod
    def is_usage_frame(frame: bytes) -> bool:
        return False

    @staticmethod
    def prepare(path: str, body: bytes, host: Any) -> bytes:
        """Ask for the prompt cache, where the host allows it and the request did not say."""
        if path != anthropic_api.MESSAGES_PATH or not getattr(host, "cache_prompts", True):
            return body
        return anthropic_api.with_cache_mark(body)[0]

    @staticmethod
    def request_headers(headers: dict[str, str]) -> dict[str, str]:
        if not any(key.lower() == "anthropic-version" for key in headers):
            headers["anthropic-version"] = anthropic_api.API_VERSION
        return headers

    @staticmethod
    def counts(tail: bytes) -> Optional[tuple[int, int, int, int]]:  # type: ignore[override]
        return anthropic_api.counts(tail)

    # --- probe path: its model list wants the version header too ---

    async def health(self, client: httpx.AsyncClient):  # type: ignore[override]
        from .base import Health

        try:
            response = await client.get("models", headers=_VERSION_HEADER)
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc) or type(exc).__name__)
        if response.status_code in (401, 403):
            return Health(ok=False, detail=f"the provider refused the key ({response.status_code})")
        if response.status_code != 200:
            return Health(ok=False, detail=f"/models returned {response.status_code}")
        return Health(ok=True)

    async def listing(self, client: httpx.AsyncClient) -> list:
        """Anthropic's `/models` entries: names and capabilities, never a price."""
        response = await client.get("models", headers=_VERSION_HEADER, params={"limit": 1000})
        response.raise_for_status()
        entries = response.json().get("data", [])
        return entries if isinstance(entries, list) else []

    async def _offered(self, client: httpx.AsyncClient) -> frozenset[str]:
        entries = await self.listing(client)
        return frozenset(
            entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        )
