"""PoolClient / AsyncPoolClient — the "a script calls the pool directly" convenience path.

The primary integration path is `pool_transport()` injected into whatever client library an
app already uses (see `transport.py`). These classes are for scripts and batch drivers that
just want `reply.content` (docs/spec/app-contract.md §4).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any, Optional

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


def _reply_from_response(response: httpx.Response) -> Reply:
    data = response.json()
    message = data.get("message", {})
    return Reply(
        content=message.get("content", data.get("response", "")),
        served_model=response.headers.get("X-GPM-Served-Model", data.get("model")),
        host=response.headers.get("X-GPM-Host"),
        runtime_class=response.headers.get("X-GPM-Runtime-Class"),
        wait_s=_float_or_none(response.headers.get("X-GPM-Wait-S")),
        raw=data,
    )


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
    ):
        self.base_url = base_url or env_url()
        self.api_key = api_key or env_api_key()
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
            body["format"] = format
        if tools is not None:
            body["tools"] = tools
        body.update(kwargs)
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = self._client.post("/api/chat", json=body, headers=headers)
        return _reply_from_response(response)

    def embed(self, model: str, texts: list[str], *, session_id: Optional[str] = None) -> list[list[float]]:
        body = {"model": model, "input": texts}
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = self._client.post("/api/embed", json=body, headers=headers)
        return response.json().get("embeddings", [])

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
    ):
        self.base_url = base_url or env_url()
        self.api_key = api_key or env_api_key()
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
            body["format"] = format
        if tools is not None:
            body["tools"] = tools
        body.update(kwargs)
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = await self._client.post("/api/chat", json=body, headers=headers)
        return _reply_from_response(response)

    async def embed(self, model: str, texts: list[str], *, session_id: Optional[str] = None) -> list[list[float]]:
        body = {"model": model, "input": texts}
        headers = {"X-GPM-Session": session_id} if session_id else {}
        response = await self._client.post("/api/embed", json=body, headers=headers)
        return response.json().get("embeddings", [])

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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncPoolClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
