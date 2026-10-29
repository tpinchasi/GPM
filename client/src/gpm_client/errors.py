"""Typed errors the pool client raises. See docs/spec/app-contract.md §4."""

from __future__ import annotations


class PoolError(Exception):
    """Base class for every error this package raises."""


class PoolAuthError(PoolError):
    """The app key was missing or wrong (HTTP 401). Never retried."""


class PoolRequestError(PoolError):
    """The pool answered with an error no wait will change, or the call outlasted its time.
    Never retried.

    Raised for every 4xx other than 401 — `404 model_not_in_pool`, `400 kind_not_served`,
    `413 request_too_large` — and for a 504: `generation_timeout`, when the engine did not
    answer within the pool's read time, and `deadline_exceeded`.

    Also raised, with status 504 and reason `read_timeout`, when the SDK's own read timeout runs
    out after the whole request was sent. That is the client's clock, not an answer from the
    pool: the pool may still be working on the request, so sending it again could run it twice
    (D144). Allow longer, or send shorter work.
    """

    def __init__(self, status_code: int, reason: str | None = None, detail: str | None = None):
        self.status_code = status_code
        self.reason = reason
        self.detail = detail
        super().__init__(f"{status_code} {reason or ''}: {detail or ''}".strip())


class PoolUnavailable(PoolError):
    """No capacity within the time this call was willing to wait.

    Raised when a 503 `no_lease` arrives and `wait_without_lease` is not set, or when
    `max_wait` is exceeded while waiting on any other 503 / transport error. A read timeout is
    not one of those: see `PoolRequestError`.
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
