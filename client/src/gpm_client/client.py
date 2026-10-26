"""PoolClient / AsyncPoolClient — the "a script calls the pool directly" convenience path.

The primary integration path is `pool_transport()` injected into whatever client library an
app already uses (see `transport.py`). These classes are for scripts and batch drivers that
just want `reply.content` (docs/spec/app-contract.md §4).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import httpx

from .config import env_api_key, env_url
from .errors import PoolUnavailable
from .retry import RetryPolicy
from .transport import async_pool_transport, pool_transport


def _float_or_none(value: Optional[str]) -> Optional[float]:
    return float(value) if value not in (None, "") else None


@dataclasses.dataclass
class Reply:
    content: str
    served_model: Optional[str]
    host: Optional[str] = None
    runtime_class: Optional[str] = None
    wait_s: Optional[float] = None
    raw: dict = dataclasses.field(default_factory=dict, repr=False)


@dataclasses.dataclass
class Transcript:
    text: str
    segments: Optional[list[dict]] = None  # when the build returns them, as verbose_json does
    language: Optional[str] = None
    served_model: Optional[str] = None
    host: Optional[str] = None
    runtime_class: Optional[str] = None
    wait_s: Optional[float] = None
    raw: Union[dict, str] = dataclasses.field(default_factory=dict, repr=False)


@dataclasses.dataclass
class Ranked:
    index: int  # into the documents the call was given
    score: float
    document: Optional[str] = None


#: The two request shapes a pool can speak, and where each keeps its paths and its answer.
#:
#: `openai` is what both shipped engines serve, so it is what these convenience methods use by
#: default: the same call reaches a pool of Ollama hosts and a pool of vLLM hosts unchanged
#: (D89). `ollama` is that engine's own API, kept for callers written against it — the pool
#: still passes those paths through untouched.
DIALECTS = {
    "openai": {"chat": "/v1/chat/completions", "embed": "/v1/embeddings"},
    "ollama": {"chat": "/api/chat", "embed": "/api/embed"},
}

#: Transcription and ranking have one path each, OpenAI's, whichever shape the client was made
#: for: Ollama serves neither, and the engines that do answer only there (D142).
_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
_RERANK_PATH = "/v1/rerank"

#: Transcription formats answered as plain text rather than JSON.
_TEXT_FORMATS = frozenset({"text", "srt", "vtt"})

#: A path, bytes, or a file object opened in binary mode.
Source = Union[str, os.PathLike, bytes, BinaryIO]


def _content_of(data: dict) -> str:
    """The generated text, in whichever shape answered.

    Read rather than translated: the SDK understands both replies, and the pool never rewrites
    one into the other — that is the passthrough rule the contract rests on.
    """
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return data.get("response", "") if isinstance(data.get("response"), str) else ""


def _embeddings_of(data: dict) -> list[list[float]]:
    if isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    rows = data.get("data")
    if isinstance(rows, list):
        return [row["embedding"] for row in rows if isinstance(row, dict) and "embedding" in row]
    return []


def _schema_field(api: str) -> str:
    """What a structured-output request is called in each dialect.

    Only the *field name* differs; the schema itself is passed through as the caller wrote it.
    Naming it here keeps the two dialects in one place rather than scattering an `if` through
    every call.
    """
    return "response_format" if api == "openai" else "format"


def _facts(response: httpx.Response, model: Optional[str] = None) -> dict[str, Any]:
    """What the pool's headers say about a call: the build that ran, where, and how long the
    request waited for a worker — so queue time is never mistaken for generation time."""
    return {
        "served_model": response.headers.get("X-GPM-Served-Model", model),
        "host": response.headers.get("X-GPM-Host"),
        "runtime_class": response.headers.get("X-GPM-Runtime-Class"),
        "wait_s": _float_or_none(response.headers.get("X-GPM-Wait-S")),
    }


def _reply_from_response(response: httpx.Response) -> Reply:
    data = response.json()
    return Reply(content=_content_of(data), raw=data, **_facts(response, data.get("model")))


def _read_source(source: Source) -> tuple[Optional[str], bytes]:
    """The bytes of a path, bytes or a binary file object, and its file name when it has one.

    Read whole into memory: the pool takes media only inside the request, and the transport
    resends the same bytes when it retries.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return None, bytes(source)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        return path.name, path.read_bytes()
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("a file object must be opened in binary mode ('rb')")
    name = getattr(source, "name", None)
    return (os.path.basename(name) if isinstance(name, str) and name else None), bytes(data)


_PICTURE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_picture(data: bytes) -> Optional[str]:
    """The picture type a file's first bytes announce, for the formats engines commonly read."""
    for signature, mime in _PICTURE_SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_part(source: Source, mime: Optional[str] = None) -> dict:
    """A chat content part carrying a picture inline, as a `data:` URL.

    The pool takes a picture only inside the request — a link is refused `400 media_not_inline`
    rather than have a GPU host fetch it — and this is the way not to get that wrong. The type
    is `mime` when given, else guessed from the file name, else read from the picture's first
    bytes; a picture whose type none of those tells is refused here, not sent.
    """
    name, data = _read_source(source)
    if mime is None and name:
        guessed = mimetypes.guess_type(name)[0]
        mime = guessed if guessed and guessed.startswith("image/") else None
    mime = mime or _sniff_picture(data)
    if mime is None:
        what = repr(name) if name else "the given bytes"
        raise ValueError(
            f"cannot tell what kind of picture {what} is: name its type, "
            "e.g. image_part(..., mime='image/png')"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _transcription_upload(model: str, file: Source, **options: Any) -> dict[str, Any]:
    """The form a transcription is posted as: `model` and each option given as fields, and the
    recording as a file — held in memory, so a retry sends it again intact."""
    name, data = _read_source(file)
    fields = {"model": model}
    fields.update({key: value for key, value in options.items() if value is not None})
    return {"data": fields, "files": {"file": (name or "audio", data)}}


def _transcript_from_response(
    response: httpx.Response, response_format: Optional[str]
) -> Transcript:
    if response_format in _TEXT_FORMATS:
        return Transcript(text=response.text, raw=response.text, **_facts(response))
    data = response.json()
    text, segments, language = data.get("text"), data.get("segments"), data.get("language")
    return Transcript(
        text=text if isinstance(text, str) else "",
        segments=segments if isinstance(segments, list) else None,
        language=language if isinstance(language, str) else None,
        raw=data,
        **_facts(response),
    )


def _rerank_body(
    model: str, query: str, documents: list[str], top_n: Optional[int]
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
    if top_n is not None:
        body["top_n"] = top_n
    return body


def _ranked_of(data: dict) -> list[Ranked]:
    """The ranking, best first, whatever order the engine listed it in."""
    ranked = []
    for row in data.get("results") or []:
        document = row.get("document")
        if isinstance(document, dict):
            document = document.get("text")
        ranked.append(
            Ranked(
                index=int(row["index"]),
                score=float(row["relevance_score"]),
                document=document if isinstance(document, str) else None,
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


# Time-to-first-byte default from docs/spec/app-contract.md §5. Non-streaming calls use this
# to bound the whole response, since "first byte" is the whole body for them.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=300.0, pool=300.0)


def _budget_from_status(data: dict, current: httpx.Timeout) -> Optional[httpx.Timeout]:
    """The pool's published time-to-first-byte, when it asks for more than we allow.

    A pool that holds responses from hosts that can be taken away (D62) turns "time to first
    byte" into the whole generation, so a budget sized for a streamed first token is too short
    there. The pool publishes what it expects; this adopts it, and never shortens anything.
    """
    published = (data.get("limits") or {}).get("client_time_to_first_byte_s")
    if not isinstance(published, (int, float)):
        return None
    if current.read is not None and current.read >= published:
        return None
    return httpx.Timeout(
        connect=current.connect, read=float(published), write=current.write, pool=current.pool
    )


class PoolClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
        api: str = "openai",
    ):
        if api not in DIALECTS:
            raise ValueError(f"api must be one of {sorted(DIALECTS)}, not {api!r}")
        self.base_url = base_url or env_url()
        self.api_key = api_key or env_api_key()
        self._paths = DIALECTS[api]
        self._api = api
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            transport=transport or pool_transport(retry_policy),
            timeout=timeout,
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        format: Any = None,
        session_id: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> Reply:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format is not None:
            body[_schema_field(self._api)] = format
        if tools is not None:
            body["tools"] = tools
        body.update(kwargs)
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = self._client.post(self._paths["chat"], json=body, headers=headers)
        return _reply_from_response(response)

    def embed(self, model: str, texts: list[str], *, session_id: Optional[str] = None) -> list[list[float]]:
        body = {"model": model, "input": texts}
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = self._client.post(self._paths["embed"], json=body, headers=headers)
        return _embeddings_of(response.json())

    def transcribe(
        self,
        model: str,
        file: Source,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Transcript:
        """A recording turned into text. `file` is a path, bytes or a binary file object."""
        upload = _transcription_upload(
            model, file, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
        )
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = self._client.post(_TRANSCRIBE_PATH, headers=headers, **upload)
        return _transcript_from_response(response, response_format)

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> list[Ranked]:
        """`documents` ranked against `query`, best first."""
        headers = {"X-GPM-Session": session_id} if session_id else {}
        body = _rerank_body(model, query, documents, top_n)
        response = self._client.post(_RERANK_PATH, json=body, headers=headers)
        return _ranked_of(response.json())

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            data = self._client.get("/pool/status").json()
            wider = _budget_from_status(data, self._client.timeout)
            if wider is not None:
                self._client.timeout = wider
            if any(h.get("state") == "ready" for h in data.get("hosts", [])):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise PoolUnavailable(reason="no_ready_host", detail="no host became ready before the timeout")
            time.sleep(1.0)

    def directory(self, query: Optional[str] = None) -> dict[str, Any]:
        """What this pool could serve, beyond its model set: the directory its operator keeps
        (app contract §2). Each entry's `in_pool` says whether a request may name it today."""
        response = self._client.get("/pool/directory", params={"q": query} if query else None)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PoolClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncPoolClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
        api: str = "openai",
    ):
        if api not in DIALECTS:
            raise ValueError(f"api must be one of {sorted(DIALECTS)}, not {api!r}")
        self.base_url = base_url or env_url()
        self.api_key = api_key or env_api_key()
        self._paths = DIALECTS[api]
        self._api = api
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            transport=transport or async_pool_transport(retry_policy),
            timeout=timeout,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        format: Any = None,
        session_id: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> Reply:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format is not None:
            body[_schema_field(self._api)] = format
        if tools is not None:
            body["tools"] = tools
        body.update(kwargs)
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = await self._client.post(self._paths["chat"], json=body, headers=headers)
        return _reply_from_response(response)

    async def embed(self, model: str, texts: list[str], *, session_id: Optional[str] = None) -> list[list[float]]:
        body = {"model": model, "input": texts}
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = await self._client.post(self._paths["embed"], json=body, headers=headers)
        return _embeddings_of(response.json())

    async def transcribe(
        self,
        model: str,
        file: Source,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Transcript:
        """See `PoolClient.transcribe`."""
        upload = _transcription_upload(
            model, file, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
        )
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = await self._client.post(_TRANSCRIBE_PATH, headers=headers, **upload)
        return _transcript_from_response(response, response_format)

    async def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> list[Ranked]:
        """See `PoolClient.rerank`."""
        headers = {"X-GPM-Session": session_id} if session_id else {}
        body = _rerank_body(model, query, documents, top_n)
        response = await self._client.post(_RERANK_PATH, json=body, headers=headers)
        return _ranked_of(response.json())

    async def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            resp = await self._client.get("/pool/status")
            data = resp.json()
            wider = _budget_from_status(data, self._client.timeout)
            if wider is not None:
                self._client.timeout = wider
            if any(h.get("state") == "ready" for h in data.get("hosts", [])):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise PoolUnavailable(reason="no_ready_host", detail="no host became ready before the timeout")
            await asyncio.sleep(1.0)

    async def directory(self, query: Optional[str] = None) -> dict[str, Any]:
        """See `PoolClient.directory`."""
        response = await self._client.get("/pool/directory", params={"q": query} if query else None)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncPoolClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
