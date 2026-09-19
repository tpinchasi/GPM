"""The pool transport: an httpx transport that waits and retries for capacity by default.

This is the SDK's core (docs/spec/app-contract.md §4) — a drop-in `httpx.BaseTransport` /
`httpx.AsyncBaseTransport`, so the retry behaviour sits underneath whatever client library an
app already uses, with no call-site changes.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx

from .errors import PoolAuthError, PoolRequestError, PoolStreamInterrupted, PoolUnavailable
from .retry import RetryPolicy


def _parse_503(response: httpx.Response) -> tuple[Optional[str], Optional[float], Optional[str]]:
    try:
        body = response.json()
    except ValueError:
        body = {}
    reason = body.get("reason")
    retry_after = body.get("retry_after_s")
    if retry_after is None:
        header = response.headers.get("Retry-After")
        retry_after = float(header) if header else None
    return reason, retry_after, body.get("detail")


def _parse_error(response: httpx.Response) -> tuple[Optional[str], Optional[str]]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    return body.get("reason"), body.get("detail") or body.get("error")


def _next_wait(policy: RetryPolicy, start: float, attempt: int, retry_after: Optional[float] = None) -> Optional[float]:
    """How long to wait before trying again, or None when the budget is spent.

    A `Retry-After` the pool asks for is honoured, but never past `max_wait`: the caller said
    how long the answer stays useful to it, and that bound wins.
    """
    wait_s = retry_after if retry_after is not None else policy.backoff_s(attempt)
    if policy.max_wait_s is None:
        return wait_s
    remaining = policy.max_wait_s - (time.monotonic() - start)
    if remaining <= 0:
        return None
    return min(wait_s, remaining)


class _StreamInterruptWrapper(httpx.SyncByteStream):
    """Wraps the response body iterator: a failure after bytes were delivered becomes
    `PoolStreamInterrupted` rather than being silently retried — only the caller, who has
    already consumed partial output, can decide whether to redo the call."""

    def __init__(self, inner):
        self._inner = inner
        self._started = False

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._started = True
                yield chunk
        except PoolStreamInterrupted:
            raise
        except Exception as exc:
            if self._started:
                raise PoolStreamInterrupted(str(exc)) from exc
            raise

    def close(self):
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


class _AsyncStreamInterruptWrapper(httpx.AsyncByteStream):
    def __init__(self, inner):
        self._inner = inner
        self._started = False

    async def __aiter__(self):
        try:
            async for chunk in self._inner:
                self._started = True
                yield chunk
        except PoolStreamInterrupted:
            raise
        except Exception as exc:
            if self._started:
                raise PoolStreamInterrupted(str(exc)) from exc
            raise

    async def aclose(self):
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


class PoolTransport(httpx.BaseTransport):
    def __init__(self, policy: Optional[RetryPolicy] = None, transport: Optional[httpx.BaseTransport] = None):
        self._policy = policy or RetryPolicy()
        self._inner = transport or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        policy = self._policy
        request.read()  # cache the body as bytes so it can be resent on retry
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                response = self._inner.handle_request(request)
            except httpx.TransportError as exc:
                wait_s = _next_wait(policy, start, attempt)
                if wait_s is None:
                    raise PoolUnavailable(reason="transport_error", detail=str(exc)) from exc
                self._wait(policy, wait_s, "transport_error", start)
                attempt += 1
                continue

            if response.status_code == 401:
                response.read()
                raise PoolAuthError("missing or invalid app key")

            if response.status_code == 503:
                response.read()
                reason, retry_after, detail = _parse_503(response)
                if reason == "no_lease" and not policy.wait_without_lease:
                    raise PoolUnavailable(reason=reason, detail=detail)
                wait_s = _next_wait(policy, start, attempt, retry_after)
                if wait_s is None:
                    raise PoolUnavailable(reason=reason or "queue_timeout", detail=detail)
                self._wait(policy, wait_s, reason or "unavailable", start)
                attempt += 1
                continue

            if response.status_code >= 400:
                response.read()
                reason, detail = _parse_error(response)
                raise PoolRequestError(response.status_code, reason=reason, detail=detail)

            response.stream = _StreamInterruptWrapper(response.stream)
            return response

    def _wait(self, policy: RetryPolicy, wait_s: float, reason: str, start: float) -> None:
        if policy.on_wait:
            policy.on_wait(reason, time.monotonic() - start, wait_s)
        time.sleep(wait_s)

    def close(self) -> None:
        self._inner.close()


class AsyncPoolTransport(httpx.AsyncBaseTransport):
    def __init__(self, policy: Optional[RetryPolicy] = None, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._policy = policy or RetryPolicy()
        self._inner = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        policy = self._policy
        await request.aread()
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                response = await self._inner.handle_async_request(request)
            except httpx.TransportError as exc:
                wait_s = _next_wait(policy, start, attempt)
                if wait_s is None:
                    raise PoolUnavailable(reason="transport_error", detail=str(exc)) from exc
                await self._wait(policy, wait_s, "transport_error", start)
                attempt += 1
                continue

            if response.status_code == 401:
                await response.aread()
                raise PoolAuthError("missing or invalid app key")

            if response.status_code == 503:
                await response.aread()
                reason, retry_after, detail = _parse_503(response)
                if reason == "no_lease" and not policy.wait_without_lease:
                    raise PoolUnavailable(reason=reason, detail=detail)
                wait_s = _next_wait(policy, start, attempt, retry_after)
                if wait_s is None:
                    raise PoolUnavailable(reason=reason or "queue_timeout", detail=detail)
                await self._wait(policy, wait_s, reason or "unavailable", start)
                attempt += 1
                continue

            if response.status_code >= 400:
                await response.aread()
                reason, detail = _parse_error(response)
                raise PoolRequestError(response.status_code, reason=reason, detail=detail)

            response.stream = _AsyncStreamInterruptWrapper(response.stream)
            return response

    async def _wait(self, policy: RetryPolicy, wait_s: float, reason: str, start: float) -> None:
        if policy.on_wait:
            policy.on_wait(reason, time.monotonic() - start, wait_s)
        await asyncio.sleep(wait_s)

    async def aclose(self) -> None:
        await self._inner.aclose()


def pool_transport(policy: Optional[RetryPolicy] = None) -> PoolTransport:
    return PoolTransport(policy=policy)


def async_pool_transport(policy: Optional[RetryPolicy] = None) -> AsyncPoolTransport:
    return AsyncPoolTransport(policy=policy)
