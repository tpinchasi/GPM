"""Anthropic's Messages format (`/v1/messages`), read once for every engine that serves it (S12).

Engines that serve the same path read it identically, by construction (plugin-interfaces §2):
Anthropic's own API, vLLM (from 0.11.1) and Ollama (from 0.14.0) all serve this path, and each
reads a request here. What the pool needs from such a request is what it needs from any: the
model, the kind (chat), what it carries besides text, whether it streams — and from the answer,
the tokens generated and, for a hosted API, what they cost.

Usage is reported differently from OpenAI's shape: `input_tokens` counts only the input after the
last cache mark, so the whole input is it plus `cache_read_input_tokens` and
`cache_creation_input_tokens`. In a stream, the input counts come in `message_start`, at its head,
and the output in `message_delta`, at its tail — so counting reads both.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from . import openai_api
from .base import CHAT, MediaNotInline, RequestFacts

MESSAGES_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
INFERENCE_PATHS = frozenset({MESSAGES_PATH, COUNT_TOKENS_PATH})
#: Paths whose answer costs nothing: counting a prompt's tokens is free at every provider.
FREE_PATHS = frozenset({COUNT_TOKENS_PATH})

#: The fields the pool reads, held to D141's rule (named once, in lower case).
POOL_FIELDS = frozenset({"model", "stream", "messages", "system", "max_tokens"})
#: The `anthropic-version` sent where a request names none — the API's stable version.
API_VERSION = "2023-06-01"


def kind_of(path: str) -> Optional[str]:
    return CHAT if path in INFERENCE_PATHS else None


def _block_inputs(block: Any) -> set[str]:
    """What a content block carries besides text. A picture must come inside the request: one
    named by a URL would be fetched by whoever serves it, so it is refused (D143)."""
    if not isinstance(block, dict):
        return set()
    kind = block.get("type")
    if kind == "image":
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "url":
            raise MediaNotInline("a picture named by a URL")
        return {"image"}
    if kind == "tool_result":
        found: set[str] = set()
        content = block.get("content")
        for inner in content if isinstance(content, list) else ():
            found |= _block_inputs(inner)
        return found
    return set()


def inputs_of(fields: dict[str, Any]) -> frozenset[str]:
    found: set[str] = set()
    messages = fields.get("messages")
    for message in messages if isinstance(messages, list) else ():
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else ():
            found |= _block_inputs(block)
    return frozenset(found)


def facts(path: str, content_type: str, body: bytes) -> RequestFacts:
    found = openai_api.members(body, keep_written=True)
    if found is None:
        return RequestFacts(model=None, kind=CHAT)
    model = openai_api.model_of(found, POOL_FIELDS)
    fields = {key: value for key, value, _, _ in found}
    return RequestFacts(
        model=model,
        kind=CHAT,
        inputs=inputs_of(fields) if path == MESSAGES_PATH else frozenset(),
        stream=path == MESSAGES_PATH and fields.get("stream") is True,
        wants_schema=_wants_schema(fields),
    )


def _wants_schema(fields: dict[str, Any]) -> bool:
    shaped = fields.get("output_format")
    return isinstance(shaped, dict) and shaped.get("type") == "json_schema"


def rewrite_model(path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
    return content_type, openai_api.with_model(body, tag)


def is_streaming(path: str, body: bytes) -> bool:
    return path == MESSAGES_PATH and openai_api.decode(body).get("stream") is True


def wants_schema(path: str, body: bytes) -> bool:
    return _wants_schema(openai_api.decode(body))


def _frames(data: bytes) -> list[dict[str, Any]]:
    """Every JSON object in an answer: a whole body, or a stream's `data:` payloads. Lines that
    do not parse — a frame cut by the head or the tail kept — are skipped."""
    return openai_api._frames_from_tail(data)


def _usage_fields(frame: dict[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(frame.get("usage"), dict):
        return frame["usage"]
    message = frame.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        return message["usage"]  # message_start carries the message, with its input counts
    return None


def _number(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def counts(data: bytes) -> Optional[tuple[int, int, int, int]]:
    """(whole input, read from cache, output, written to cache) tokens, as the answer reports
    them — None where it reports none. A stream's counts are cumulative, so the most seen of each
    is the count."""
    seen: dict[str, int] = {}
    any_usage = False
    for frame in _frames(data):
        usage = _usage_fields(frame)
        if usage is None:
            continue
        any_usage = True
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
            seen[key] = max(seen.get(key, 0), _number(usage.get(key)))
    if not any_usage:
        return None
    read, written = seen.get("cache_read_input_tokens", 0), seen.get("cache_creation_input_tokens", 0)
    return seen.get("input_tokens", 0) + read + written, read, seen.get("output_tokens", 0), written


def usage(path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
    """Tokens generated; how long it took is measured by the pool, as for OpenAI's format."""
    found = counts(tail)
    return (found[2] if found is not None else None), None


def keepalive_frame(path: str) -> Optional[bytes]:
    """The keep-alive this format's streams carry anyway: a `ping` event every client ignores."""
    return b'event: ping\ndata: {"type": "ping"}\n\n' if path == MESSAGES_PATH else None


# --- the cache mark, where a provider caches only when asked (S12) ---


def _marked(value: Any) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(_marked(v) for v in value.values())
    if isinstance(value, list):
        return any(_marked(v) for v in value)
    return False


def with_cache_mark(body: bytes) -> tuple[bytes, bool]:
    """The body asking for Anthropic's automatic prompt cache — one top-level `cache_control`,
    the 5-minute cache renewed free on each use — where the request carries no cache mark of its
    own; and whether it was added. Inserted after the opening brace, every other byte as it
    came: the request is otherwise passed through untouched."""
    found = openai_api.members(body)
    if found is None:
        return body, False
    if any(_marked({key: value}) for key, value, _, _ in found if key in ("cache_control", "system", "messages", "tools")):
        return body, False
    text = body.decode("utf-8")
    at = text.index("{") + 1
    mark = '"cache_control":{"type":"ephemeral"}' + ("," if found else "")
    return (text[:at] + mark + text[at:]).encode("utf-8"), True


def error_body(error_type: str, message: str) -> dict[str, Any]:
    """An error as this format says one."""
    return {"type": "error", "error": {"type": error_type, "message": message}}


#: The pool's statuses, as this format names its error types.
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
               499: "api_error", 500: "api_error", 503: "overloaded_error", 504: "api_error"}


def pool_error(status: int, content: dict[str, Any]) -> dict[str, Any]:
    """One of the pool's own refusals, in this format's shape so Anthropic's SDK reads it, with
    the pool's own fields kept beside it (app-contract §2): `reason`, `detail`, `retry_after_s`."""
    detail = content.get("detail") or content.get("reason") or content.get("error") or "refused"
    shaped = error_body(ERROR_TYPES.get(status, "api_error"), str(detail))
    shaped.update({key: value for key, value in content.items() if key in ("reason", "detail", "retry_after_s")})
    shaped["pool_error"] = content.get("error")
    return shaped


def json_dumps(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")
