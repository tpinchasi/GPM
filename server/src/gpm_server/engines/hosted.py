"""`openai-api`: a hosted model API reached with a key (S11, D164).

OpenAI, Together, Ollama's cloud, Groq, Fireworks, DeepInfra, OpenRouter — every provider that
serves the OpenAI-shaped `/v1` paths is this one engine: a provider is a base URL, a key and a
price list, never an adapter of its own. Claude is reached through `anthropic-api` instead
(D170): Anthropic's OpenAI-compatible layer does not cache, and is retired.

What is different about such a host, and why the adapter looks the way it does:

- **Requests are passed through as for any engine** — the shared `openai_api` module reads and
  rewrites them — but the path is the provider's: a host's base URL is the provider's whole
  OpenAI base (`…/v1`, or Groq's `…/openai/v1`), and what follows `/v1` in the request is
  appended to it (`upstream_path`).
- **A streamed request is asked for its usage** (D166): `stream_options: {include_usage: true}`
  is added, since an OpenAI-shaped stream carries its token counts only when asked, and the
  pool cannot hold a budget without them. The extra chunk is removed again where the app did not
  ask for it (`strip_usage_frame`).
- **Cost comes from the answer's own counts** (`counts`): input, cached input and output tokens,
  times the catalog's prices.
- **A refusal is read for what it means** (`refusal`): *busy* (try another host after its
  `retry-after`), *out of money* (the host is taken out: retrying would be refused for ever), or
  the request's own (returned as the provider said it). Each provider says "out of money"
  differently.
- **Nothing is pulled, loaded or launched.** Health and the models it has come from `/models`,
  which spends nothing.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Optional

import httpx

from . import openai_api
from .base import CHAT, EMBED, RERANK, TRANSCRIBE, Health, PullResult, RequestFacts

#: What each provider says when the money has run out (verified facts, D164): an answer that must
#: never be read as a rate limit, or it would be tried again for ever.
OUT_OF_FUNDS_CODES = frozenset({
    "project_spend_limit_exceeded",   # OpenAI, 429
    "insufficient_quota",             # OpenAI, 429
    "enforced_spend_limit_reached",   # Anthropic's tier cap, 429 without retry-after
    "blocked_api_access",             # Groq, 400
    "billing_hard_limit_reached",     # OpenAI
    "credit_balance_too_low",
    "billing_error",                  # Anthropic's Messages API
})
#: Words in a provider's refusal that mean the same, where it gives no code.
OUT_OF_FUNDS_WORDS = ("spend limit", "credit balance", "insufficient credit", "insufficient balance",
                      "out of credit", "billing", "quota exceeded", "payment required")
#: Statuses that mean "not now": busy, overloaded, or briefly unavailable.
BUSY_STATUSES = frozenset({429, 502, 503, 529})


class HostedOpenAIEngine:
    interface_version: ClassVar[str] = "2"
    name: ClassVar[str] = "openai-api"
    #: Whatever a provider serves of these, the catalog says per model (D142).
    kinds: ClassVar[frozenset[str]] = frozenset({CHAT, EMBED, TRANSCRIBE, RERANK})
    serves_one_model: ClassVar[bool] = False
    #: Hosted: never pulled, loaded, rented or launched (D164).
    hosted: ClassVar[bool] = True
    #: Nothing to start on a machine: the host's base URL is the provider's.
    default_port: ClassVar[int] = 0
    models_dir: ClassVar[str] = ""
    loads_by_restart: ClassVar[bool] = False
    splits_across_cards: ClassVar[bool] = False
    options: ClassVar[dict] = {}
    builds_on_hub: ClassVar[bool] = False
    image_words: ClassVar[tuple[str, ...]] = ()

    def default_start_command(self, *, port: int, models_dir: str, agent_archive: str, proxy: bool,
                              options: tuple[str, ...] = ()) -> Optional[str]:
        return None

    # --- request path ---

    def inference_paths(self) -> set[str]:
        return set(openai_api.INFERENCE_PATHS | openai_api.TRANSCRIBE_PATHS | openai_api.RERANK_PATHS)

    def kind_of(self, path: str) -> Optional[str]:
        return openai_api.kind_of(path) if path in self.inference_paths() else None

    def request_facts(self, path: str, content_type: str, body: bytes) -> RequestFacts:
        return openai_api.facts(path, content_type, body)

    def rewrite_model(self, path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
        return openai_api.rewrite_model(path, content_type, body, tag)

    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        return openai_api.requested_model(body)

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        return openai_api.with_model(body, tag)

    def wants_schema(self, path: str, body: bytes) -> bool:
        return openai_api.wants_schema(path, body)

    def is_streaming(self, path: str, body: bytes) -> bool:
        return openai_api.is_streaming(path, body)

    def usage(self, path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
        return openai_api.usage(path, tail)

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        return openai_api.keepalive_frame(path)

    @staticmethod
    def upstream_path(path: str) -> str:
        """The request's path under the provider's base URL, which already ends at its `/v1`."""
        return path[len("/v1"):] if path.startswith("/v1/") else path

    # --- what a hosted request needs besides the model (D166) ---

    @staticmethod
    def ask_for_usage(content_type: str, body: bytes) -> tuple[bytes, bool]:
        """A streamed JSON request with `stream_options.include_usage` set, and whether the app
        had asked for it itself — in which case its usage chunk is its own, and kept. Anything
        that is not a streamed JSON object is returned as it came."""
        if "json" not in (content_type or "application/json").lower():
            return body, True
        try:
            request = json.loads(body)
        except ValueError:
            return body, True
        if not isinstance(request, dict) or request.get("stream") is not True:
            return body, True
        options = request.get("stream_options")
        if isinstance(options, dict) and options.get("include_usage") is True:
            return body, True
        request["stream_options"] = {**(options if isinstance(options, dict) else {}), "include_usage": True}
        return json.dumps(request, separators=(",", ":")).encode(), False

    @staticmethod
    def is_usage_frame(frame: bytes) -> bool:
        """The chunk a stream gains for its usage: a data frame with an empty `choices` and a
        `usage` — removed from what the app receives where it did not ask for it."""
        for line in frame.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                return False
            try:
                chunk = json.loads(payload)
            except ValueError:
                return False
            return isinstance(chunk, dict) and chunk.get("choices") == [] and isinstance(chunk.get("usage"), dict)
        return False

    @staticmethod
    def counts(tail: bytes) -> Optional[tuple[int, int, int, int]]:
        """(input, read from cache, output, written to cache) tokens, as the answer reports them
        — None where it does not. OpenAI-shaped `usage`: `prompt_tokens` includes the cached ones,
        said apart in `prompt_tokens_details.cached_tokens` (some providers put them at the top
        level), and — on the newest OpenAI models — the ones written, `cache_write_tokens`."""
        for frame in openai_api._frames_from_tail(tail):
            usage = frame.get("usage")
            if not isinstance(usage, dict):
                continue
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens", 0)
            if not isinstance(prompt, (int, float)):
                continue
            details = usage.get("prompt_tokens_details")
            details = details if isinstance(details, dict) else {}
            cached = details.get("cached_tokens") or usage.get("cached_tokens") or 0
            written = details.get("cache_write_tokens") or usage.get("cache_write_tokens") or 0
            completion = completion if isinstance(completion, (int, float)) else 0
            cached = cached if isinstance(cached, (int, float)) else 0
            written = written if isinstance(written, (int, float)) else 0
            cached = min(cached, prompt)
            return int(prompt), int(cached), int(completion), int(min(written, prompt - cached))
        return None

    @staticmethod
    def refusal(status: int, body: bytes) -> str:
        """What a provider's error answer means for the pool: `out_of_funds`, `busy`, or
        `request` (the request's own; returned to the app as the provider said it)."""
        text = body[:4096].decode("utf-8", "replace")
        code = ""
        try:
            parsed = json.loads(text)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(error, dict):
                code = str(error.get("code") or error.get("type") or "")
                text = str(error.get("message") or "") + " " + code
        except ValueError:
            pass
        lowered = text.lower()
        if status == 402 or code in OUT_OF_FUNDS_CODES or any(word in lowered for word in OUT_OF_FUNDS_WORDS):
            return "out_of_funds"
        if status in BUSY_STATUSES:
            return "busy"
        return "request"

    # --- probe path: nothing here spends ---

    async def health(self, client: httpx.AsyncClient) -> Health:
        try:
            response = await client.get("models")
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc) or type(exc).__name__)
        if response.status_code in (401, 403):
            return Health(ok=False, detail=f"the provider refused the key ({response.status_code})")
        if response.status_code != 200:
            return Health(ok=False, detail=f"/models returned {response.status_code}")
        return Health(ok=True)

    async def _offered(self, client: httpx.AsyncClient) -> frozenset[str]:
        response = await client.get("models")
        response.raise_for_status()
        entries = response.json().get("data", [])
        return frozenset(
            entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        )

    async def models_resident(self, client: httpx.AsyncClient) -> frozenset[str]:
        """What the provider offers is what it serves: nothing to load."""
        return await self._offered(client)

    async def models_available(self, client: httpx.AsyncClient) -> frozenset[str]:
        return await self._offered(client)

    async def occupancy(self, client: httpx.AsyncClient) -> Any:
        return None

    async def serving_from_cpu(self, client: httpx.AsyncClient) -> Optional[frozenset[str]]:
        return None

    async def pull(self, client: httpx.AsyncClient, tag: str, on_progress=None) -> PullResult:
        return PullResult(tag=tag, ok=False, retryable=False,
                          detail="a hosted API has no models to fetch: it serves what its provider offers")

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        offered = await self._offered(client)
        missing = set(tags) - offered
        if missing:
            raise RuntimeError(f"the provider does not offer {sorted(missing)}")

    def launch_settings(self, workers: int, context: int, n_models: int, listen: Optional[str] = None) -> dict[str, str]:
        return {}
