"""Runtime objects: hosts and workers.

docs/spec/hosts-routing-capacity.md §1.1 and §2. A worker is a first-class object — it belongs
to one host, serves one request at a time, and has its own state and counters.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

import httpx

from .catalog import ResolvedVariant


class HostState(str, Enum):
    """The subset of the host state machine a router-only phase can reach.

    `preparing`, `recovering`, `quarantined`, `draining`, `parked` and the rented states arrive
    with the supervisor (phase 2), which is what drives those transitions.
    """

    READY = "ready"
    #: Reachable, but the pool's whole model set is not resident yet.
    PREPARING = "preparing"
    UNREACHABLE = "unreachable"
    DISABLED = "disabled"


class WorkerState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    DISABLED = "disabled"


@dataclasses.dataclass
class Worker:
    worker_id: str
    state: WorkerState = WorkerState.IDLE
    request_id: Optional[str] = None
    served: int = 0

    @property
    def is_idle(self) -> bool:
        return self.state is WorkerState.IDLE


@dataclasses.dataclass
class Host:
    host_id: str
    kind: str
    priority: int
    capabilities: frozenset[str]
    client: httpx.AsyncClient
    workers: list[Worker]
    #: Per logical model, the variants this host could serve, in preference order.
    variants: dict[str, tuple[ResolvedVariant, ...]]
    #: Every tag the pool knows by name, for requests that name a build explicitly.
    literal_variants: dict[str, ResolvedVariant]
    transport_type: str = "http"
    state: HostState = HostState.UNREACHABLE
    #: Tags the engine reports loaded right now. Empty until the first successful probe.
    resident: frozenset[str] = frozenset()
    last_error: Optional[str] = None
    last_probe_at: Optional[float] = None
    #: Counters the router publishes for the supervisor (docs/spec/supervisor.md §1).
    last_request_at: Optional[float] = None
    requests_served: int = 0
    failures: int = 0

    @property
    def busy(self) -> int:
        return sum(1 for w in self.workers if w.state is WorkerState.BUSY)

    @property
    def total_workers(self) -> int:
        return len(self.workers)

    @property
    def load(self) -> float:
        """busy / total — the within-tier tie-break (spec §5)."""
        return self.busy / self.total_workers if self.workers else 1.0

    def idle_worker(self) -> Optional[Worker]:
        return next((w for w in self.workers if w.is_idle), None)
