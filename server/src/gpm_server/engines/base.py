"""The engine plug-in interface.

docs/spec/plugin-interfaces.md §2. This is the phase-1 subset: the request-path operations,
which must be cheap and never block, plus the two the readiness probe needs. The
supervisor-side operations (`pull`, `load_and_pin`, `smoke_test`, `launch_settings`,
`looks_corrupt`) arrive with the supervisor in phase 2.

The router passes requests through in the engine's own API and never translates between APIs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

import httpx


@dataclasses.dataclass(frozen=True)
class Health:
    ok: bool
    detail: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PullResult:
    tag: str
    ok: bool
    #: What the pull moved, for the cost the console shows against its estimate.
    bytes_total: int = 0
    detail: Optional[str] = None
    #: Whether a failed pull is worth trying again: a connection cut mid-download is; a tag
    #: the registry does not have is not. Unset means no — an adapter that does not say is
    #: not retried, which is what it did before retries existed.
    retryable: bool = False


@runtime_checkable
class Engine(Protocol):
    interface_version: ClassVar[str]
    name: ClassVar[str]

    # --- request path: cheap, synchronous, no I/O ---

    def inference_paths(self) -> set[str]:
        """Paths that take a worker."""

    def requested_model(self, path: str, body: bytes) -> Optional[str]:
        """The model the request names, or None if the body does not name one."""

    def with_model(self, path: str, body: bytes, tag: str) -> bytes:
        """Return the body with the model field changed and nothing else.

        Byte-for-byte otherwise: the router's passthrough fidelity rests on this.
        """

    def wants_schema(self, path: str, body: bytes) -> bool:
        """True when the request asks for schema-enforced structured output."""

    def is_streaming(self, path: str, body: bytes) -> bool:
        """True when the response will be streamed."""

    def keepalive_frame(self, path: str) -> Optional[bytes]:
        """A frame that is *harmless* in this engine's stream format, or None (D62).

        While the pool holds a response until it is whole, it sends the client nothing, and an
        intermediary with an idle timeout may cut the connection. An engine whose format has a
        no-op frame — a comment line in server-sent events, say — declares it here and the pool
        sends it meanwhile. An engine whose format has none declares nothing, and the pool
        sends nothing: a frame a client might parse as content is worse than a dropped
        connection. Newline-delimited JSON has no such frame, so the first engine returns None.
        """
        return None

    # --- probe path ---

    async def health(self, client: httpx.AsyncClient) -> Health:
        """Is the engine answering?"""

    async def models_resident(self, client: httpx.AsyncClient) -> frozenset[str]:
        """The tags loaded in memory *now* — not merely present on disk."""

    async def models_available(self, client: httpx.AsyncClient) -> frozenset[str]:
        """The tags present on disk, loaded or not. What an `on_demand` host is judged by:
        a tag here can be served without a download; one absent cannot, and never will be
        because a request asked."""

    # --- preparing a host the pool created ---

    async def pull(self, client: httpx.AsyncClient, tag: str) -> PullResult:
        """Fetch one tag. Only ever called for a tag named in configuration, at prepare time —
        never triggered by a request (threat model T10)."""

    async def load_and_pin(self, client: httpx.AsyncClient, tags: list[str]) -> None:
        """Load all the tags and keep them loaded; raise if they cannot all be resident
        together."""

    def launch_settings(self, workers: int, context: int, n_models: int) -> dict[str, str]:
        """Environment that makes the engine run `workers` requests in parallel at `context`,
        holding `n_models` models. Used only on hosts the pool creates."""


class EngineNotFound(Exception):
    """The configuration named an engine that is not installed."""


#: The group third-party engines register under — the same one the shipped adapter uses.
ENTRY_POINT_GROUP = "gpm.engines"


def available_engines() -> dict[str, Any]:
    from importlib.metadata import entry_points

    return {point.name: point for point in entry_points(group=ENTRY_POINT_GROUP)}


def get_engine(name: str) -> Engine:
    """Resolve an engine adapter by name. Nothing loads that configuration does not name."""
    found = available_engines()
    if name not in found:
        raise EngineNotFound(f"unknown engine {name!r}; installed: {sorted(found) or 'none'}")
    return found[name].load()()
