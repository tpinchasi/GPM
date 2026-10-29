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
from typing import Any, Iterable, Optional

from . import multipart
from .base import CHAT, EMBED, RERANK, TRANSCRIBE, AmbiguousField, MediaNotInline, RequestFacts

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

#: Paths only some engines serve (D142). Never part of `INFERENCE_PATHS`, which every engine
#: speaking this protocol inherits: an engine claims these itself, or it does not serve them.
TRANSCRIBE_PATHS = frozenset({"/v1/audio/transcriptions", "/v1/audio/translations"})
RERANK_PATHS = frozenset({"/v1/rerank"})

_KIND_BY_PATH = {
    **{path: CHAT for path in STREAMING_PATHS},
    "/v1/embeddings": EMBED,
    **{path: TRANSCRIBE for path in TRANSCRIBE_PATHS},
    **{path: RERANK for path in RERANK_PATHS},
}


def kind_of(path: str) -> Optional[str]:
    return _KIND_BY_PATH.get(path)


#: The fields the pool itself reads from a JSON body. Each may appear once at the top level,
#: spelled exactly. An engine whose decoder matches keys ignoring case — Go's, so Ollama's —
#: reads `"MODEL"` where the pool reads `"model"`, and with both present the pool would check
#: one value and the engine serve another (D141).
POOL_FIELDS = frozenset({"model", "stream", "response_format", "format"})
#: The fields the pool reads to learn what a request carries (D143), held to the same rule: an
#: engine reading `"Messages"` would see a picture the pool did not, and the pool would have sent
#: it to a build that does not take one.
INPUT_FIELDS = frozenset({"messages", "input", "query", "documents"})
#: Inside each message, the fields that carry pictures: its content parts, and Ollama's `images`.
MESSAGE_FIELDS = frozenset({"content", "images"})

_WHITESPACE = re.compile(r"[ \t\n\r]*")
_DECODER = json.JSONDecoder()


class _Written(dict):
    """A JSON object as decoded, remembering its keys as they were written — repeats and letter
    case included — so a field the pool reads inside it can be held to D141's rule."""

    __slots__ = ("written",)


def _keep_written(pairs: list[tuple[str, Any]]) -> _Written:
    found = _Written(pairs)
    found.written = tuple(key for key, _ in pairs)
    return found


_KEEPING_DECODER = json.JSONDecoder(object_pairs_hook=_keep_written)


def members(body: bytes, keep_written: bool = False) -> Optional[list[tuple[str, Any, int, int]]]:
    """The top-level object's members in order, duplicates included: each key, its value, and
    where the value starts and ends in the decoded text. None when the body is not one JSON
    object in UTF-8 — including nesting too deep to parse, which is a refusal, not a crash.
    With `keep_written`, every nested object remembers its keys as written (`written_once`)."""
    decoder = _KEEPING_DECODER if keep_written else _DECODER
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
                value, at = decoder.raw_decode(text, start)
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
    return None if found is None else model_of(found)


def model_of(found: list[tuple[str, Any, int, int]], fields: frozenset[str] = POOL_FIELDS) -> Optional[str]:
    """The top-level `model`, after holding each of `fields` to D141's rule."""
    _once((key for key, _, _, _ in found), fields)
    model = next((value for key, value, _, _ in found if key == "model"), None)
    return model if isinstance(model, str) else None


def _once(keys: Iterable[str], fields: frozenset[str]) -> None:
    """Each of `fields` at most once among these keys, spelled exactly; `AmbiguousField` if not."""
    seen: set[str] = set()
    for key in keys:
        folded = key.casefold()
        if folded in fields:
            if key != folded or folded in seen:
                raise AmbiguousField(folded)
            seen.add(folded)


def written_once(found: Any, fields: frozenset[str]) -> None:
    """D141's rule inside one nested object the pool reads — a message, a ranked document — and
    never at every depth: a JSON schema's properties may differ only in case, and the pool reads
    none of them. Needs the object decoded by `members(..., keep_written=True)`."""
    _once(getattr(found, "written", ()), fields)


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


# --- reading a request once (D144): kind, inputs and the fields the pool routes on ---

#: The parts a message may carry, and what each is (D143). A closed table: a part the pool does
#: not know might be a link an engine fetches, so it is refused rather than passed on.
_TEXT_PARTS = frozenset({"text", "input_text", "output_text", "refusal"})
_IMAGE_PARTS = frozenset({"image_url", "input_image"})
_AUDIO_PARTS = frozenset({"input_audio"})


def inputs_of_content(content: Any) -> set[str]:
    """What one message's content carries besides text. A string is text; anything that is not a
    list of parts is the engine's to judge, as it always was."""
    if not isinstance(content, list):
        return set()
    found: set[str] = set()
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in _TEXT_PARTS:
            continue
        if kind in _IMAGE_PARTS:
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not (isinstance(url, str) and url.startswith("data:")):
                raise MediaNotInline("a picture given by link rather than inside the request")
            found.add("image")
        elif kind in _AUDIO_PARTS:
            audio = part.get("input_audio")
            if not (isinstance(audio, dict) and isinstance(audio.get("data"), str) and "url" not in audio):
                raise MediaNotInline("a sound given by link rather than inside the request")
            found.add("audio")
        else:
            raise MediaNotInline("a part of a type the pool does not accept")
    return found


def _is_message(item: dict[str, Any]) -> bool:
    """A message among the other items a responses request's `input` may hold. Its role is found
    in any letter case, as an engine reading keys that way would find it."""
    return item.get("type") == "message" or any(
        key.casefold() == "role" for key in getattr(item, "written", item)
    )


def _message_inputs(message: dict[str, Any]) -> set[str]:
    written_once(message, MESSAGE_FIELDS)
    return inputs_of_content(message.get("content"))


def _messages_inputs(messages: Any) -> set[str]:
    """A chat request's `messages`: every one a message, whatever its role is spelled."""
    found: set[str] = set()
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, dict):
            found |= _message_inputs(message)
    return found


def inputs_of_request(fields: dict[str, Any]) -> frozenset[str]:
    """Pictures and sound anywhere in a chat-shaped request: `messages` (chat) or `input`
    (responses), each message's content parts. A responses request's other items — a function's
    output, a model's own reasoning — carry nothing an engine would fetch, and are left alone."""
    found = _messages_inputs(fields.get("messages"))
    items = fields.get("input")
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and _is_message(item):
            found |= _message_inputs(item)
    return frozenset(found)


def _item_inputs(item: Any) -> set[str]:
    """What one thing to embed or rank carries (D143): text or tokens; a message; an object of
    content parts (`{"content": [...]}`); a list of parts; or one part on its own. vLLM takes
    pictures in all of these and fetches a link itself (D146), so every one goes through the
    same closed table, inline only."""
    if isinstance(item, list):
        return inputs_of_content(item)
    if not isinstance(item, dict):
        return set()
    if _is_message(item):
        return _message_inputs(item)
    if any(key.casefold() == "content" for key in getattr(item, "written", item)):
        written_once(item, frozenset({"content"}))
        return inputs_of_content(item.get("content"))
    return inputs_of_content([item])


def inputs_of_embedding(fields: dict[str, Any]) -> frozenset[str]:
    """Pictures and sound in an embedding request: chat-shaped `messages`, or `input` — one item
    or a list of them."""
    found = _messages_inputs(fields.get("messages"))
    items = fields.get("input")
    for item in items if isinstance(items, list) else [items]:
        found |= _item_inputs(item)
    return frozenset(found)


def inputs_of_ranking(fields: dict[str, Any]) -> frozenset[str]:
    """Pictures and sound in a ranking request: its `query`, and each of its `documents` — or
    the one object of content parts that stands for them all."""
    found = _item_inputs(fields.get("query"))
    documents = fields.get("documents")
    for document in documents if isinstance(documents, list) else [documents]:
        found |= _item_inputs(document)
    return frozenset(found)


#: Where each kind of JSON request carries its pictures and sound.
_INPUTS_OF = {CHAT: inputs_of_request, EMBED: inputs_of_embedding, RERANK: inputs_of_ranking}


def json_facts(path: str, body: bytes) -> RequestFacts:
    """The facts of a JSON request on this protocol's paths, from one parse."""
    kind = kind_of(path) or CHAT
    found = members(body, keep_written=True)
    if found is None:
        return RequestFacts(model=None, kind=kind)
    model = model_of(found, POOL_FIELDS | INPUT_FIELDS)
    fields = {key: value for key, value, _, _ in found}
    response_format = fields.get("response_format")
    reader = _INPUTS_OF.get(kind)
    return RequestFacts(
        model=model,
        kind=kind,
        inputs=reader(fields) if reader is not None else frozenset(),
        stream=path in STREAMING_PATHS and bool(fields.get("stream", False)),
        wants_schema=isinstance(response_format, dict) and response_format.get("type") == "json_schema",
    )


def upload_facts(content_type: str, body: bytes) -> RequestFacts:
    """The facts of a transcription upload — parsed strictly, refused when it would have to be
    guessed about (`multipart.BadUpload`, D144). Its file is sound by definition."""
    parts = multipart.parse(content_type, body)
    stream = (multipart.field(parts, "stream") or "").strip().lower() == "true"
    return RequestFacts(
        model=multipart.field(parts, "model"), kind=TRANSCRIBE, inputs=frozenset({"audio"}), stream=stream
    )


def facts(path: str, content_type: str, body: bytes) -> RequestFacts:
    if path in TRANSCRIBE_PATHS:
        return upload_facts(content_type, body)
    return json_facts(path, body)


def rewrite_model(path: str, content_type: str, body: bytes, tag: str) -> tuple[str, bytes]:
    """The model set to `tag`: in place for JSON (`with_model`), or the upload rebuilt in the
    pool's own canonical form, every file's bytes untouched (D144)."""
    if path in TRANSCRIBE_PATHS:
        return multipart.rebuild(multipart.parse(content_type, body), replace={"model": tag})
    return content_type, with_model(body, tag)


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
