"""Typed errors the pool client raises. See docs/spec/app-contract.md §4."""

from __future__ import annotations


class PoolError(Exception):
    """Base class for every error this package raises."""


class PoolAuthError(PoolError):
    """The app key was missing or wrong (HTTP 401). Never retried."""


class PoolRequestError(PoolError):
    """A 4xx other than 401 — for example 404 model_not_in_pool. Never retried."""

    def __init__(self, status_code: int, reason: str | None = None, detail: str | None = None):
        self.status_code = status_code
        self.reason = reason
        self.detail = detail
        super().__init__(f"{status_code} {reason or ''}: {detail or ''}".strip())


class PoolUnavailable(PoolError):
    """No capacity within the time this call was willing to wait.

    Raised when a 503 `no_lease` arrives and `wait_without_lease` is not set, or when
    `max_wait` is exceeded while waiting on any other 503 / transport error.
    """

    def __init__(self, reason: str | None = None, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason or "pool unavailable")


class PoolStreamInterrupted(PoolError):
    """The upstream failed after some output was already delivered.

    The app has consumed partial output; only it can decide whether to redo the call, so
    this is never retried automatically.
    """
