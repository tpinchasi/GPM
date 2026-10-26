"""The OpenAI-shaped HTTP surface, shared by every engine that speaks it (D89).

Two engines in this pool serve these paths: vLLM speaks them natively, and Ollama serves them
beside its own. The request-path rules — which paths take a worker, where the model name is,
when a response streams, how to read the token counts back — are properties of *the protocol*,
not of either engine, so they live here and both adapters use them.

Everything in this module is on the router's request path: cheap, synchronous, no I/O.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import AmbiguousField

#: Paths that take a worker. `/v1/models` is excluded on purpose: listing models is metadata,
#: costs an engine nothing, and must not consume capacity the pool is counting.
INFERENCE_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
})

#: Of those, the ones that can stream. Embeddings answer in one piece.
STREAMING_PATHS = frozenset({"/v1/chat/completions", "/v1/completions", "/v1/responses"})


#: The fields the pool itself reads from a JSON body. Each may appear once at the top level,
#: spelled exactly. An engine whose decoder matches keys ignoring case — Go's, so Ollama's —
#: reads `"MODEL"` where the pool reads `"model"`, and with both present the pool would check
#: one value and the engine serve another (D141).
POOL_FIELDS = frozenset({"model", "stream", "response_format", "format"})

_WHITESPACE = re.compile(r"[ \t\n\r]*")
_DECODER = json.JSONDecoder()


def members(body: bytes) -> Optional[list[tuple[str, Any, int, int]]]:
    """The top-level object's members in order, duplicates included: each key, its value, and
    where the value starts and ends in the decoded text. None when the body is not one JSON
    object in UTF-8 — including nesting too deep to parse, which is a refusal, not a crash."""
    try:
        text = body.decode("utf-8")
        at = _WHITESPACE.match(text).end()
        if text[at:at + 1] != "{":
            return None
        at = _WHITESPACE.match(text, at + 1).end()
        found: list[tuple[str, Any, int, int]] = []
        if text[at:at + 1] == "}":
            at += 1
        else:
            while True:
                if text[at:at + 1] != '"':
                    return None
                key, at = _DECODER.raw_decode(text, at)
                at = _WHITESPACE.match(text, at).end()
                if text[at:at + 1] != ":":
                    return None
                start = _WHITESPACE.match(text, at + 1).end()
                value, at = _DECODER.raw_decode(text, start)
                found.append((key, value, start, at))
                at = _WHITESPACE.match(text, at).end()
                if text[at:at + 1] == ",":
                    at = _WHITESPACE.match(text, at + 1).end()
                    continue
                if text[at:at + 1] != "}":
                    return None
                at += 1
                break
        if _WHITESPACE.match(text, at).end() != len(text):
            return None
        return found
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None


def decode(body: bytes) -> dict[str, Any]:
    found = members(body)
    return {key: value for key, value, _, _ in found} if found is not None else {}


def requested_model(body: bytes) -> Optional[str]:
    """The top-level `model`, after refusing a body in which a field the pool reads is named
    twice or in another letter case (`AmbiguousField`)."""
    found = members(body)
    if found is None:
        return None
    seen: set[str] = set()
    for key, _, _, _ in found:
        folded = key.casefold()
        if folded in POOL_FIELDS:
            if key != folded or folded in seen:
                raise AmbiguousField(folded)
            seen.add(folded)
    model = next((value for key, value, _, _ in found if key == "model"), None)
    return model if isinstance(model, str) else None


def with_model(body: bytes, tag: str) -> bytes:
    """The body with the top-level `model` value replaced — the one `requested_model` read —
    and every other byte untouched. Re-serialising would reorder keys and renormalise numbers
    and whitespace, the passthrough drift the contract forbids; a pattern over the raw bytes
    would rewrite whichever `"model"` came first, nested or not."""
    found = members(body)
    for key, _, start, end in found or ():
        if key == "model":
            text = body.decode("utf-8")
            return (text[:start] + json.dumps(tag) + text[end:]).encode("utf-8")
    return body


def is_streaming(path: str, body: bytes) -> bool:
    """**Defaults to False**, which is the opposite of Ollama's native API.

    In this protocol a response is whole unless the request asked for a stream; in Ollama's own
    API it streams unless the request asked it not to. An adapter serving both surfaces must
    not apply one default to the other — the pool would hold a stream it thinks is a body, or
    frame a body it thinks is a stream.
    """
    return path in STREAMING_PATHS and bool(decode(body).get("stream", False))


def wants_schema(path: str, body: bytes) -> bool:
    """True only for a *schema*, not for loose JSON mode.

    `{"type": "json_object"}` asks for valid JSON and guarantees nothing about its shape;
    `{"type": "json_schema", ...}` is the one the pool may route on.
    """
    response_format = decode(body).get("response_format")
    if not isinstance(response_format, dict):
        return False
    return response_format.get("type") == "json_schema"


def usage(path: str, tail: bytes) -> tuple[Optional[int], Optional[float]]:
    """Tokens generated, and milliseconds spent generating — **None** for the milliseconds.

    This protocol reports token counts but never how long generation took, so the pool measures
    duration itself. Returning a wall-clock figure here instead would quietly fold queue wait
    into generation time, and deciding a host's worker count turns on telling those apart
    (D67).

    Non-streamed responses carry `usage` in the body. Streamed ones carry it only when the
    request asked with `stream_options.include_usage`, in a final data frame; absent that,
    there is no count to read and the pool says so rather than guessing.
    """
    for frame in _frames_from_tail(tail):
        counts = frame.get("usage")
        if not isinstance(counts, dict):
            continue
        generated = counts.get("completion_tokens")
        if generated is None:
            # An embedding response reports only what it read; nothing was generated.
            continue
        return (int(generated) if isinstance(generated, (int, float)) else None), None
    return None, None


def _frames_from_tail(tail: bytes) -> list[dict[str, Any]]:
    """Every JSON object at the end of a response, newest first.

    Handles both shapes at once: a whole body is one object, and a stream is server-sent
    events whose payloads are prefixed with `data: `. The tail may begin mid-frame, so a line
    that will not parse is skipped rather than failing the read.
    """
    found = []
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        if line == b"[DONE]" or not line.startswith(b"{"):
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            found.append(parsed)
    if not found and tail.strip().startswith(b"{"):
        try:
            whole = json.loads(tail)
        except ValueError:
            return found
        if isinstance(whole, dict):
            found.append(whole)
    return found


def keepalive_frame(path: str) -> Optional[bytes]:
    """A server-sent-events comment: the one frame this protocol defines as ignorable (D62).

    While the pool holds a response until it is whole, an intermediary with an idle timeout may
    cut the connection. A client parsing SSE discards a line beginning with `:` by
    specification, so this is safe to send in a way Ollama's newline-delimited JSON has no
    equivalent for. Only on paths that actually stream — on a whole-body path any bytes at all
    would corrupt the response.
    """
    return b": keep-alive\n\n" if path in STREAMING_PATHS else None
